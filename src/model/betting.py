import numpy as np
import matplotlib.pyplot as plt
import math
from src.model import evaluation as evalu

# ---- paths ----
PLOT_DIR        = 'src/model/plots'

# ---- plotting style ----
MODEL_C, MARKET_C, REF_C, ACCENT_C = '#1f77b4', '#ff7f0e', '#cccccc', '#d62728'

def kelly_bankroll_fraction(prob: float, odd: float, frac: float) -> float:
    # Clamp at 0: a negative Kelly fraction means there is no edge, so the correct
    # action is not to bet rather than to stake a negative (inverted) amount.
    return max(0.0, frac * (odd*prob-1)/(odd-1))

def compute_payouts(init_bankroll: float, probs: np.array, odds: np.array, labels: np.ndarray, race_indices: np.ndarray,
                    kelly_frac: float,) -> np.array:
    # Stakes within the same race are all sized off the bankroll held *before* that
    # race, and the bankroll is only updated once the race is settled. This avoids
    # compounding several bets from the same race against each other.
    res = []
    current_bankroll = init_bankroll
    total_bets = len(probs)
    race_payout = 0
    race_payout_list = []
    for bet in range(total_bets):
        wager_frac = kelly_bankroll_fraction(probs[bet], odds[bet], kelly_frac)
        bet_size = math.floor(current_bankroll * wager_frac)
        payout = 0
        if labels[bet] == 1:
            payout += bet_size * odds[bet] - bet_size
            race_payout += payout
        else:
            payout -= bet_size
            race_payout += payout

        race_payout_list.append(payout)

        if bet == total_bets-1 or race_indices[bet] != race_indices[bet+1]:
            res.extend(race_payout_list)
            race_payout_list.clear()

            current_bankroll += race_payout
            race_payout = 0
    return np.array(res)

def run_ep_strategy(model_probs, market_probs, odds, labels, alpha, race_starts, race_length, min_prob, kelly_frac,
                 init_bankroll):
    race_indices = race_starts.repeat(race_length)
    stats = evalu.evaluate_betting_config(model_probs, market_probs, odds, labels, alpha, min_prob)
    fired = stats["flagged horses"]
    deltas = compute_payouts(init_bankroll, model_probs[fired], odds[fired], labels[fired], race_indices[fired],
                             kelly_frac)
    equity = np.zeros(len(market_probs))
    equity[fired] = deltas
    equity = np.cumsum(equity) + init_bankroll
    return stats, equity

def run_fav_strategy(probs, odds, labels, race_starts, race_length, kelly_frac, init_bankroll):
    race_indices = race_starts.repeat(race_length)
    biggest_race_probs = np.maximum.reduceat(probs, race_starts)
    biggest_race_probs = biggest_race_probs.repeat(race_length)
    equal = np.isclose(biggest_race_probs, probs)

    deltas = compute_payouts(init_bankroll, probs[equal], odds[equal], labels[equal], race_indices[equal],
                             kelly_frac)
    equity = np.zeros(len(probs))
    equity[equal] = deltas
    equity = np.cumsum(equity) + init_bankroll
    return equity


def plot_equity_overtime(equities_overtime: dict, length: int, title, fname):
    fig, ax = plt.subplots()
    ax.axhline(1000, color=REF_C, linewidth=1, linestyle='--', zorder=1,
               label="Starting bankroll")
    for name, equity in equities_overtime.items():
        ax.plot(np.arange(length), equity, '-', linewidth=1, label=name)
    ax.set_title(title)
    ax.set_xlabel("Bet index"); ax.set_ylabel("Bankroll")
    ax.legend()
    fig.savefig(f"{PLOT_DIR}/{fname}", bbox_inches='tight')
