from src.model import objective as obj
import matplotlib.pyplot as plt
import numpy as np

# ---- paths ----
PLOT_DIR        = 'src/model/plots'

# ---- plotting style ----
MODEL_C, MARKET_C, REF_C, ACCENT_C = '#1f77b4', '#ff7f0e', '#cccccc', '#d62728'

def plot_divergence(model_probs, market_probs):
    fig, ax = plt.subplots()
    ax.hist(abs(model_probs - market_probs), bins=50, range=(0, 0.3),
            color=MODEL_C, edgecolor='white', linewidth=0.5)
    ax.set_title("Model–Market Probability Divergence")
    ax.set_xlabel(r'$|p_{\mathrm{model}} - p_{\mathrm{market}}|$')
    ax.set_ylabel("Count")
    fig.savefig(f"{PLOT_DIR}/divergence_hist.pdf", bbox_inches='tight')

def plot_calibration(model_x, model_y, market_x, market_y, title, fname, lim):
    fig, ax = plt.subplots()
    ax.plot([0, lim], [0, lim], '--', color=REF_C, linewidth=1, zorder=1,
            label="Perfect calibration")
    ax.plot(market_x, market_y, 's-', color=MARKET_C, label="Market")
    ax.plot(model_x, model_y, 'o-', color=MODEL_C, label="Model")
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed win rate")
    ax.set_aspect('equal'); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend()
    fig.savefig(f"{PLOT_DIR}/{fname}", bbox_inches='tight')

def evaluate_betting_config(model_probs, market_probs, odds, labels, alpha, min_prob):
    adjusted = (1 - alpha) * model_probs + alpha * market_probs
    mask = adjusted >= min_prob
    adj_probs  = adjusted[mask]
    adj_odds   = odds[mask]
    adj_labels = labels[mask]

    _, ep_array = obj.expected_profit_rule_stat(adj_probs, adj_odds)
    fired = ep_array.astype(bool)
    full_fired = np.zeros(len(model_probs), dtype=bool)
    full_fired[mask] = fired

    bet_count = int(fired.sum())
    win_count = int(adj_labels[fired].sum())
    win_rate  = win_count / bet_count if bet_count else 0.0
    naive_return = float((adj_labels[fired] * adj_odds[fired]).sum() - bet_count)
    return {"alpha": alpha, "min_prob": min_prob, "bet_count": bet_count,
            "win_count": win_count, "win_rate": win_rate, "naive_return": naive_return, "flagged horses": full_fired}

