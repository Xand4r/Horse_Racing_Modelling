import numpy as np
from collections.abc import Callable


def per_race_softmax(raw_scores: np.ndarray, race_start_id: np.ndarray, race_lengths: np.ndarray) -> np.ndarray:
    #for numerical stability in each race find maximum value and subtract it
    race_biggest_score = np.maximum.reduceat(raw_scores, race_start_id)
    race_biggest_score = race_biggest_score.repeat(race_lengths)
    safe_scores = raw_scores - race_biggest_score

    exp_shifted = np.exp(safe_scores)
    race_exp_sum = np.add.reduceat(exp_shifted, race_start_id)
    race_exp_sum = race_exp_sum.repeat(race_lengths)
    res = exp_shifted / race_exp_sum

    return res

def make_objective(race_starts: np.ndarray, race_lengths: np.ndarray) -> Callable:

    def objective(scores: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        # Gradient/Hessian of the per-race softmax cross-entropy w.r.t. the raw
        # scores. y_sums is the number of winners per race (1 in the normal case,
        # 2 for a dead heat), broadcast back onto each horse so the formulas stay
        # general: grad = -y + p*Σy, hess = p*(1-p)*Σy.
        y = train_data.get_label()
        probs = per_race_softmax(scores, race_starts, race_lengths)
        y_sums = np.add.reduceat(y, race_starts)
        y_sums = y_sums.repeat(race_lengths)
        grad = -y + probs * y_sums
        hess = y_sums * probs * (1-probs)
        return grad, hess

    return objective



def cat_cross_entr(scores: np.ndarray, eval_data) -> tuple[str, float, bool]:
    race_starts = eval_data.race_starts
    race_lengths = eval_data.race_lengths
    y = eval_data.get_label()
    probs = per_race_softmax(scores, race_starts, race_lengths)
    probs = np.clip(probs, 1e-12, 1.0)
    loss = - np.sum(y*np.log(probs)) / len(race_lengths)
    is_higher_better = False
    return "categorical cross entropy", loss, is_higher_better

def implied_odds_performance(implied_odds: np.ndarray, eval_data) -> float:
    race_starts = eval_data.race_starts
    race_lengths = eval_data.race_lengths
    y = eval_data.get_label()
    probs = np.clip(implied_odds, 1e-12, 1.0)
    loss = - np.sum(y * np.log(probs)) / len(race_lengths)
    return loss

def top_one_acc(scores: np.ndarray, eval_data) -> tuple[str, float, bool]:
    race_starts = eval_data.race_starts
    race_lengths = eval_data.race_lengths
    y = eval_data.get_label()
    probs = per_race_softmax(scores, race_starts, race_lengths)

    biggest_race_probs = np.maximum.reduceat(probs, race_starts)
    biggest_race_probs = biggest_race_probs.repeat(race_lengths)

    pred_vs_outcome = y*probs
    correct_prediction = (pred_vs_outcome == biggest_race_probs).astype(int)
    accuracy = np.sum(correct_prediction)/len(race_lengths)
    is_higher_better = True
    return "Top-1 accuracy", accuracy, is_higher_better

def implied_top_one_acc(implied_odds: np.ndarray, eval_data) -> (float, np.ndarray):
    race_starts = eval_data.race_starts
    race_lengths = eval_data.race_lengths
    y = eval_data.get_label()

    biggest_race_probs = np.maximum.reduceat(implied_odds, race_starts)
    biggest_race_probs = biggest_race_probs.repeat(race_lengths)

    pred_vs_outcome = y * implied_odds
    correct_prediction = (pred_vs_outcome == biggest_race_probs).astype(int)
    accuracy = np.sum(correct_prediction) / len(race_lengths)
    return accuracy, correct_prediction

def expected_profit_rule_stat(probs: np.ndarray, odds: np.ndarray) -> (float, np.ndarray):

    expected_profit = odds * probs - 1

    expected_profit[expected_profit <= 0] = 0
    expected_profit[expected_profit > 0] = 1

    possible_bet_count = sum(expected_profit)
    accuracy = possible_bet_count/len(probs)
    return accuracy, expected_profit

def prob_calibration_bin(probs: np.ndarray, labels: np.ndarray) -> (list,list, float):
    # only needed when array contains zeroes (normally softmax removes zeroes):
    fired = probs > 0
    probs = probs[fired]
    labels = labels[fired]

    bins = []
    bin_points = []
    ece = 0
    total = 0
    for i in range(0, 20):
        mask = (probs > i * 0.05) & (probs <= (i + 1) * 0.05)

        n = mask.sum()

        if n == 0:
            # empty bins are kept as NaN so the plotted curve has a gap rather than a spurious point
            bins.append(np.nan)
            bin_points.append(np.nan)
            continue
        bins.append(labels[mask].sum() / n)
        bin_points.append(probs[mask].mean())
        # ECE is the count-weighted average gap between predicted prob and observed win rate
        ece += n * abs(probs[mask].mean() - labels[mask].sum() / n)
        total += n
    ece = ece / total
    return bins, bin_points, ece

def quantile_calibration_bin(probs: np.ndarray, labels: np.ndarray, bin_count: int) -> (list, list, float):
    n_bins = bin_count
    #only needed when array contains zeroes (normally softmax removes zeroes):
    fired = probs > 0
    probs = probs[fired]
    labels = labels[fired]
    # quantile edges on the MODEL predictions
    edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
    ece = 0
    total = 0
    my_x, my_y = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # (lo, hi] bins; the top edge equals the max prediction, so <= keeps the
        # highest-probability horse in the last bin rather than dropping it
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        n = mask.sum()
        my_x.append(probs[mask].mean())  # x = mean prediction in bin
        my_y.append( labels[mask].sum() / n )  # y = win rate
        ece += n * abs(probs[mask].mean() - labels[mask].sum() / n)
        total += n
    ece = ece / total
    return my_x, my_y, ece

def normalize_probs(probs: np.ndarray, race_start_id: np.ndarray, race_lengths: np.ndarray) -> np.ndarray:
    race_sum = np.add.reduceat(probs, race_start_id)
    race_sum = race_sum.repeat(race_lengths)
    res = probs / race_sum
    return res