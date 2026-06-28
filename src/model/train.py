from src.model import data_preparation as dp
from src.model import objective as obj
import lightgbm as lgb
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle

# ---- paths ----
DATA_CACHE_PATH = 'src/model/prepared_data.pkl'
MODEL_PATH      = 'src/model/versions/model_final.txt'
EVAL_PATH       = 'src/model/versions/training_process.json'
PLOT_DIR        = 'src/model/plots'

# ---- plotting style ----
MODEL_C, MARKET_C, REF_C, ACCENT_C = '#1f77b4', '#ff7f0e', '#cccccc', '#d62728'


def load_or_prepare_data(reload_data=False):
    if not os.path.exists(DATA_CACHE_PATH) or reload_data:
        train_split, val_split, test_split = dp.prepare_input()
        with open(DATA_CACHE_PATH, 'wb') as f:
            pickle.dump({"train_split": train_split, "val_split": val_split,
                         "test_split": test_split}, f)
    else:
        with open(DATA_CACHE_PATH, 'rb') as f:
            d = pickle.load(f)
            train_split, val_split, test_split = d["train_split"], d["val_split"], d["test_split"]
    return train_split, val_split, test_split


def build_dataset(split, reference=None):
    cats = ['venue', 'going', 'course', 'jockey', 'trainer']
    ds = lgb.Dataset(split["x"], label=split["y"],
                     categorical_feature=cats, reference=reference)
    ds.race_starts  = split["race_starts"]
    ds.race_lengths = split["race_lengths"]
    return ds



def train_or_load_model(lgb_train, lgb_val, train_split, reload_data=False):
    eval_results = {}
    if not os.path.exists(MODEL_PATH) or reload_data:
        params = {
            'objective': obj.make_objective(train_split["race_starts"], train_split["race_lengths"]),
            'learning_rate': 0.05, 'num_leaves': 7, 'min_data_in_leaf': 200,
            'feature_fraction': 0.6,
            'bagging_fraction': 1.0,   # must stay 1.0 with custom race-grouped objective
            'verbose': -1, 'metric': 'None', 'lambda_l2': 20,
        }
        booster = lgb.train(
            params=params, train_set=lgb_train, num_boost_round=1000,
            valid_sets=[lgb_train, lgb_val], valid_names=['train', 'val'],
            feval=[obj.cat_cross_entr, obj.top_one_acc],
            callbacks=[lgb.early_stopping(100, first_metric_only=True),
                       lgb.log_evaluation(period=10),
                       lgb.record_evaluation(eval_results)],
        )
        gains = booster.feature_importance(importance_type='gain')
        for name, gain in sorted(zip(booster.feature_name(), gains), key=lambda x: x[1], reverse=True):
            print(f"{name:40s} {gain:12.1f}  {gain / gains.sum():6.2%}")
        booster.save_model(MODEL_PATH)
        with open(EVAL_PATH, 'w') as f:
            json.dump(eval_results, f)
    else:
        booster = lgb.Booster(model_file=MODEL_PATH)
        with open(EVAL_PATH, 'r') as f:
            eval_results = json.load(f)
    return booster, eval_results


def plot_training_curves(eval_results, market_ce, market_top1):
    ce = eval_results['train']['categorical cross entropy']
    val_ce = eval_results['val']['categorical cross entropy']
    iters = np.arange(1, len(ce) + 1)

    fig, ax = plt.subplots()
    ax.plot(iters, ce, label="Training loss", color=MODEL_C)
    ax.plot(iters, val_ce, label="Validation loss", color=ACCENT_C)
    min_y = min(val_ce); min_x = iters[val_ce.index(min_y)]
    ax.axhline(market_ce, label="Market baseline", linestyle='--', color=REF_C)
    ax.scatter(min_x, min_y, color=ACCENT_C, zorder=5, label=f"Minimum ({min_y:.3f})")
    ax.set_title("Cross-Entropy Loss over Boosting Iterations")
    ax.set_xlabel("Boosting iteration"); ax.set_ylabel("Categorical cross-entropy")
    ax.legend()
    fig.savefig(f"{PLOT_DIR}/loss_curve.pdf", bbox_inches='tight')

    acc = eval_results['train']['Top-1 accuracy']
    val_acc = eval_results['val']['Top-1 accuracy']
    iters = np.arange(1, len(acc) + 1)
    fig, ax = plt.subplots()
    ax.plot(iters, acc, label="Training accuracy", color=MODEL_C)
    ax.plot(iters, val_acc, label="Validation accuracy", color=ACCENT_C)
    ax.axhline(market_top1, label="Market baseline", linestyle='--', color=REF_C)
    ax.set_title("Top-1 Accuracy over Boosting Iterations")
    ax.set_xlabel("Boosting iteration"); ax.set_ylabel("Top-1 accuracy")
    ax.legend()
    fig.savefig(f"{PLOT_DIR}/accuracy_curve.pdf", bbox_inches='tight')



