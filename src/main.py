from src.model import train
from src.model import evaluation as evalu
from src.model import objective as obj
from src.model import betting as bet
import numpy as np
import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams.update({
        'figure.figsize': (7, 4.5), 'figure.dpi': 110, 'font.size': 11,
        'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'lines.linewidth': 1.8, 'lines.markersize': 5,
        'legend.frameon': False, 'figure.constrained_layout.use': True,
    })

def main():
    reload_data = False
    train_split, val_split, test_split = train.load_or_prepare_data(reload_data)
    lgb_train = train.build_dataset(train_split)
    lgb_val = train.build_dataset(val_split, reference=lgb_train)
    lgb_test = train.build_dataset(test_split, reference=lgb_train)
    booster, eval_results = train.train_or_load_model(lgb_train, lgb_val, train_split, reload_data)

    # --- validation predictions / val-set market baselines ---
    market_probs_val = val_split["x"]['implied win probability'].to_numpy()
    odds_val     = val_split["odds"]
    labels_val   = np.asarray(lgb_val.label)
    market_ce_val    = obj.implied_odds_performance(market_probs_val, lgb_val)
    market_top1_val, _ = obj.implied_top_one_acc(market_probs_val, lgb_val)
    model_probs_val  = obj.per_race_softmax(booster.predict(val_split["x"]),
                                        lgb_val.race_starts, lgb_val.race_lengths)
    model_ce_val = obj.implied_odds_performance(model_probs_val, lgb_val)
    model_top1_val, _ = obj.implied_top_one_acc(model_probs_val, lgb_val)
    print("market val cat entr:", market_ce_val)
    print("market val top 1 acc:", market_top1_val)
    print("model val cat entr:", model_ce_val)
    print("model val top 1 acc:", model_top1_val)
    # --- plots ---
    set_plot_style()
    train.plot_training_curves(eval_results, market_ce_val, market_top1_val)
    evalu.plot_divergence(model_probs_val, market_probs_val)

    _, ep_array = obj.expected_profit_rule_stat(model_probs_val, odds_val)
    print("betting rate:", ep_array.mean())
    print("won betting:", labels_val[ep_array.astype(bool)].mean())

    my_b, my_bp, my_ece = obj.prob_calibration_bin(model_probs_val, labels_val)
    mk_b, mk_bp, mk_ece = obj.prob_calibration_bin(market_probs_val, labels_val)
    evalu.plot_calibration(my_bp, my_b, mk_bp, mk_b,
                     "Calibration (Fixed-Width Bins)", "calibration_fixed.pdf", 1.05)

    my_x, my_y, my_qece = obj.quantile_calibration_bin(model_probs_val, labels_val, 10)
    mk_x, mk_y, mk_qece = obj.quantile_calibration_bin(market_probs_val, labels_val, 10)
    evalu.plot_calibration(my_x, my_y, mk_x, mk_y,
                     "Calibration (Quantile Bins)", "calibration_quantile.pdf", 0.35)
    print("ece model/market:", my_ece, mk_ece, "| quantile:", my_qece, mk_qece)

    # --- calibration on flagged bets ---
    val_fired = ep_array.astype(bool)
    ep_probs, ep_market, ep_labels = model_probs_val[val_fired], market_probs_val[val_fired], labels_val[val_fired]
    e_b, e_bp, e_ece = obj.prob_calibration_bin(ep_probs, ep_labels)
    em_b, em_bp, em_ece = obj.prob_calibration_bin(ep_market, ep_labels)
    evalu.plot_calibration(e_bp, e_b, em_bp, em_b,
                     "Calibration on Flagged Bets (Fixed-Width Bins)",
                     "calibration_fired_fixed.pdf", 1.05)
    ex, ey, _ = obj.quantile_calibration_bin(ep_probs, ep_labels, 5)
    mx, my2, _ = obj.quantile_calibration_bin(ep_market, ep_labels, 5)
    evalu.plot_calibration(ex, ey, mx, my2,
                     "Calibration on Flagged Bets (Quantile Bins)",
                     "calibration_fired_quantile.pdf", 0.35)
    print("fired ece model/market:", e_ece, em_ece)

    # --- betting strategy on val with calibrated parameters based on val results
    # (alpha, min_prob selected on validation; see documentation) ---
    for i in range(21):
        alpha = i*0.05
        val_stats_fullkelly, val_equity_fullkelly = bet.run_ep_strategy(model_probs_val, market_probs_val, odds_val,
                                                                       labels_val, alpha, lgb_val.race_starts,
                                                                       lgb_val.race_lengths, 0.10, 1,
                                                                     1000)
        val_stats_halfkelly, val_equity_halfkelly = bet.run_ep_strategy(model_probs_val, market_probs_val, odds_val,
                                                                       labels_val, alpha, lgb_val.race_starts,
                                                                       lgb_val.race_lengths,0.10, 0.5,
                                                                     1000)
        print("ALPHA: ", alpha)
        print("bet_count full: ", val_stats_fullkelly["bet_count"], "win count full: ",
              val_stats_fullkelly["win_count"], "winrate full: ", val_stats_fullkelly["win_rate"])
        print("bet_count half: ", val_stats_halfkelly["bet_count"], "win count half: ",
              val_stats_halfkelly["win_count"], "winrate half: ", val_stats_halfkelly["win_rate"])
        print("total comp val set with full kelly: ", val_equity_fullkelly[-1])
        print("total comp val set with half kelly: ", val_equity_halfkelly[-1])

    # --- results on test set ---
    print("TEST SET")
    market_probs_test = test_split["x"]['implied win probability'].to_numpy()
    odds_test = test_split["odds"]
    labels_test = np.asarray(lgb_test.label)
    market_ce_test = obj.implied_odds_performance(market_probs_test, lgb_test)
    market_top1_test, _ = obj.implied_top_one_acc(market_probs_test, lgb_test)
    model_probs_test = obj.per_race_softmax(booster.predict(test_split["x"]),lgb_test.race_starts,
                                            lgb_test.race_lengths)
    model_ce_test = obj.implied_odds_performance(model_probs_test, lgb_test)
    model_top1_test, _ = obj.implied_top_one_acc(model_probs_test, lgb_test)
    print("market test cat entr:", market_ce_test)
    print("market test top 1 acc:", market_top1_test)
    print("model test cat entr:", model_ce_test)
    print("model test top 1 acc:", model_top1_test)

    test_stats_fullkelly, test_equity_fullkelly = bet.run_ep_strategy(model_probs_test, market_probs_test, odds_test,
                                                                 labels_test, 0.55, lgb_test.race_starts,
                                                                   lgb_test.race_lengths, 0.10, 1,
                                                                   1000)
    test_stats_halfkelly, test_equity_halfkelly = bet.run_ep_strategy(model_probs_test, market_probs_test, odds_test,
                                                                 labels_test, 0.55, lgb_test.race_starts,
                                                                   lgb_test.race_lengths,0.10, 0.5,
                                                                   1000)
    print("bet_count full: ", test_stats_fullkelly["bet_count"], "win count full: ", test_stats_fullkelly["win_count"], "winrate full: ", test_stats_fullkelly["win_rate"])
    print("bet_count half: ", test_stats_halfkelly["bet_count"], "win count half: ", test_stats_halfkelly["win_count"], "winrate half: ", test_stats_halfkelly["win_rate"])
    print("total comp test set with full kelly: ", test_equity_fullkelly[-1])
    print("total comp test set with half kelly: ", test_equity_halfkelly[-1])

    model_fav_fullkelly = bet.run_fav_strategy(model_probs_test, odds_test, labels_test, lgb_test.race_starts, lgb_test.race_lengths,
                               1, 1000)
    model_fav_halfkelly = bet.run_fav_strategy(model_probs_test, odds_test, labels_test, lgb_test.race_starts,
                                               lgb_test.race_lengths,0.5, 1000)
    market_fav_fullkelly = bet.run_fav_strategy(market_probs_test, odds_test, labels_test, lgb_test.race_starts,
                                               lgb_test.race_lengths,
                                               1, 1000)
    market_fav_halfkelly = bet.run_fav_strategy(market_probs_test, odds_test, labels_test, lgb_test.race_starts,
                                               lgb_test.race_lengths, 0.5, 1000)

    full = {
         "Expected profit full kelly": test_equity_fullkelly,
         "Model favourite full kelly": model_fav_fullkelly,
         "Market favourite full kelly": market_fav_fullkelly
    }

    half = {
        "Expected profit half kelly": test_equity_halfkelly,
        "Model favourite half kelly": model_fav_halfkelly,
        "Market favourite half kelly": market_fav_halfkelly
    }
    bet.plot_equity_overtime(full, len(test_equity_fullkelly), "Full Kelly", "equity_full_kelly.pdf")
    bet.plot_equity_overtime(half, len(test_equity_halfkelly), "Half Kelly", "equity_half_kelly.pdf")
    plt.show()

if __name__ == "__main__":
    main()