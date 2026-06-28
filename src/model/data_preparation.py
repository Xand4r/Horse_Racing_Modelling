import numpy as np
import pandas as pd
import math
from src.data import feature_engineering as fe


def prepare_input() -> tuple[dict, dict, dict]:
    features = fe.create_features()

    prob_norm_const = features["prob_norm_const"].to_numpy()
    features.drop(columns=['prob_norm_const'], inplace=True)
    odds = features["odds"].to_numpy()
    features.drop(columns=['odds'], inplace=True)

    cat_col = ['venue','going','course','jockey','trainer']
    not_num_col = ['label', 'id', 'horse'] + cat_col
    num_col = [c for c in features.columns if c not in not_num_col]

    labels = features['label']
    race_id = features['id']
    categorical = features[cat_col].apply(lambda s: s.astype('category').cat.codes)
    numerical = features[num_col]

    grouped_lists = [list(v) for v in race_id.groupby(race_id, sort=False).groups.values()]

    # Chronological split at the level of whole races (never splitting a race):
    # first 80% of races -> train, then 75% of the remainder -> val, rest -> test,
    # giving an 80/15/5 split. Each race boundary is converted from a race index
    # into the row index of that race's first horse, since the feature matrix is
    # one row per horse.
    train_val_split = 0.8
    val_test_split = 0.75

    split_train_race_index = math.ceil(train_val_split * len(grouped_lists))
    split_train_row_index = grouped_lists[split_train_race_index][0]
    split_val_race_index = math.ceil(val_test_split * (len(grouped_lists)-split_train_race_index))+split_train_race_index
    split_val_row_index = grouped_lists[split_val_race_index][0]

    # race_lengths[i] = number of horses in race i; race_starts[i] = row offset of
    # race i in the stacked feature matrix (used by the per-race softmax/objective).
    race_lengths = np.array(list((len(group) for group in grouped_lists)))
    race_starts = np.concatenate(([0], np.cumsum(race_lengths)[:-1]))

    num_train = numerical.iloc[:split_train_row_index]
    cat_train = categorical.iloc[:split_train_row_index]
    race_lengths_train = race_lengths[:split_train_race_index]
    race_starts_train = race_starts[:split_train_race_index]
    prob_norm_const_train = prob_norm_const[:split_train_row_index]
    label_train = labels[:split_train_row_index]
    odds_train = odds[:split_train_row_index]

    num_val = numerical.iloc[split_train_row_index:split_val_row_index]
    cat_val = categorical.iloc[split_train_row_index:split_val_row_index]
    race_lengths_val = race_lengths[split_train_race_index:split_val_race_index]
    # rebase race_starts to 0 so they index into this split's own (sliced) feature matrix
    race_starts_val = race_starts[split_train_race_index:split_val_race_index] - race_starts[split_train_race_index]
    prob_norm_const_val = prob_norm_const[split_train_row_index:split_val_row_index]
    label_val = labels[split_train_row_index:split_val_row_index]
    odds_val = odds[split_train_row_index:split_val_row_index]

    num_test = numerical.iloc[split_val_row_index:]
    cat_test = categorical.iloc[split_val_row_index:]
    race_lengths_test = race_lengths[split_val_race_index:]
    race_starts_test = race_starts[split_val_race_index:] - race_starts[split_val_race_index]
    prob_norm_const_test = prob_norm_const[split_val_row_index:]
    label_test = labels[split_val_row_index:]
    odds_test = odds[split_val_row_index:]

    x_train = pd.concat([num_train, cat_train], axis=1)
    x_val = pd.concat([num_val, cat_val], axis=1)
    x_test = pd.concat([num_test, cat_test], axis=1)

    train_split = {
        "x": x_train,
        "y": label_train,
        "race_lengths": race_lengths_train,
        "race_starts": race_starts_train,
        "prob_norm_const": prob_norm_const_train,
        "odds": odds_train
    }
    val_split = {
        "x": x_val,
        "y": label_val,
        "race_lengths": race_lengths_val,
        "race_starts": race_starts_val,
        "prob_norm_const": prob_norm_const_val,
        "odds": odds_val
    }
    test_split = {
        "x": x_test,
        "y": label_test,
        "race_lengths": race_lengths_test,
        "race_starts": race_starts_test,
        "prob_norm_const": prob_norm_const_test,
        "odds": odds_test
    }

    return train_split, val_split, test_split
