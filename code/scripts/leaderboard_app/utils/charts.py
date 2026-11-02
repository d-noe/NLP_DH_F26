import numpy as np
import pandas as pd
import altair as alt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# =================================================================
# -------------------------- Trade-off ----------------------------
# =================================================================

def compute_beta_scores_multi(leaderboard, perf_col="f1", fairness_col="fairness_epsilon", n=20):
    """
    Compute beta-weighted scores for all submissions.
    Returns a DataFrame suitable for Altair (columns: submission, beta, score)
    """
    beta_values = np.linspace(0, 1, n)
    records = []

    for _, row in leaderboard.iterrows():
        f1 = row[perf_col]
        fairness = row[fairness_col]
        scores = beta_values * f1 + (1 - beta_values) * fairness

        for b, s in zip(beta_values, scores):
            records.append({
                "submission": row["submission"],
                "beta": b,
                "score": s
            })

    return pd.DataFrame(records)


def plot_tradeoff_multi(df_beta):
    """
    Plot beta-weighted scores for all submissions with interactive legend selection.
    """
    # Define a multi-selection bound to the legend
    selection = alt.selection_multi(fields=['submission'], bind='legend')

    chart = alt.Chart(df_beta).mark_line(point=True).encode(
        x=alt.X("beta", title="More Fairness ← Beta (Performance vs Fairness) → More Performance"),
        y=alt.Y("score", title="Weighted Score (higher is better)"),
        color=alt.Color("submission:N", title="Submission"),
        tooltip=["submission", "beta", "score"],
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.1))  # highlight selected
    ).add_selection(
        selection
    ).properties(
        width=700,
        height=400,
        title="Performance vs Fairness Trade-off for All Submissions"
    ).interactive()  # Enable zooming and panning

    return chart

# =================================================================
# ------------------------ FAIRNESS CHART -------------------------
# =================================================================

from utils.metrics import FAIRNESS_METRICS

def plot_fairness_ratios(row, epsilon=0.8, title=None):
    """
    Generates a fairness indicators plot for a given submission (row from leaderboard).

    Parameters
    ----------
    row : pd.Series
        A row from the leaderboard containing the relevant fairness metrics:
        tpr_u, tpr_p, ppv_u, ppv_p, fpr_u, fpr_p, acc_u, acc_p, pr_u, pr_p
    epsilon : float, optional
        Fairness threshold for shading the "fair zone", by default 0.8
    title : str, optional
        Title of the plot. If None, will use row['submission']
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated Matplotlib figure.
    """
    def safe_div(a, b):
        return a / b if b != 0 else np.nan

    if title is None:
        title = f"Fairness Indicators: {row['submission']}"

    # Compute ratios
    # ratios = {
    #     "Equal opportunity ratio (TPR)": row["Equal opportunity ratio (TPR)"],
    #     "Predictive parity ratio (Precision)": row["Predictive parity ratio (Precision)"],
    #     "Predictive equality ratio (FPR)": row["Predictive equality ratio (FPR)"],
    #     "Accuracy equality ratio (ACC)": row["Accuracy equality ratio (ACC)"],
    #     "Statistical parity ratio (PPR)": row["Statistical parity ratio (PPR)"],
    # }
    ratios = {
        m:row[m]
        for m in FAIRNESS_METRICS
    }

    fairness_df = pd.DataFrame(list(ratios.items()), columns=["metric", "ratio"])

    # Compute maximal epsilon such that all ratios are within [ε, 1/ε]
    max_epsilon = min(np.min(fairness_df["ratio"]), 1 / np.max(fairness_df["ratio"]))
    ratios.update({'fairness_epsilon': max_epsilon})

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(fairness_df))
    ratios_values = fairness_df["ratio"].values

    # Fairness zones
    ax.axvspan(epsilon, 1/epsilon, color="palegreen", alpha=0.3,
               label=f"Fair zone [{epsilon:.2f}, {1/epsilon:.2f}]")
    ax.axvspan(0, epsilon, color="mistyrose", alpha=0.5)
    ax.axvspan(1/epsilon, max(1.5, max(ratios_values) + 0.1), color="mistyrose", alpha=0.5)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)

    # Bars
    ax.barh(y_pos, ratios_values - 1, color="teal", alpha=0.8, left=1)
    ax.scatter(
        [
            1 if np.isnan(r) else r 
            for r in ratios_values
        ], 
        y_pos, 
        marker= 'o',
        s=150, 
        c=[
            "grey" if np.isnan(r) else "green" if ((r>epsilon) and (r<(1/epsilon))) else "red" 
            for r in ratios_values
        ],
        alpha=[
            0.4 if np.isnan(r) else 1
            for r in ratios_values
        ]
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fairness_df["metric"])
    ax.invert_yaxis()  # top metric first
    ax.set_xlabel("Unprivileged / Privileged Ratio", fontsize=11)
    ax.set_xlim(0, max(1.5, np.nanmax(ratios_values) + 0.1))
    ax.set_title(title, color="darkblue", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()

    return fig


# =================================================================
# ---------------------- Confusion matrices -----------------------
# =================================================================

def plot_single_confusion_matrix(
    y_true, y_pred, ax, title="", cmap="RdBu"
):
    sns.heatmap(
        confusion_matrix(
            y_true,
            y_pred,
        ),
        cmap=cmap,
        xticklabels=['canon', 'non-canon'],
        yticklabels=['canon', 'non-canon'],
        annot=True,
        fmt=".0f",
        ax = ax,
        cbar=False
    )
    ax.set_ylabel("True Label", fontweight="semibold")
    ax.set_xlabel("Predicted Label", fontweight="semibold")
    ax.set_title(title)
    return ax

def plot_confusion_matrices(
    y_true, y_pred, attributes,
):
    # masks
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    attributes = np.array(attributes)

    unpriv_mask = attributes == 0 # female
    priv_mask = ~unpriv_mask

    y_true_male = y_true[priv_mask]
    y_pred_male = y_pred[priv_mask]
    y_true_female = y_true[unpriv_mask]
    y_pred_female = y_pred[unpriv_mask]

    # make plots
    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(15, 5)
    )
    axes[0] = plot_single_confusion_matrix(y_true, y_pred, ax=axes[0], title="Overall CM")
    axes[1] = plot_single_confusion_matrix(y_true_male, y_pred_male, ax=axes[1], title="Male-author CM", cmap="Blues")
    axes[2] = plot_single_confusion_matrix(y_true_female, y_pred_female, ax=axes[2], title="Female-author CM", cmap="Oranges")

    return fig