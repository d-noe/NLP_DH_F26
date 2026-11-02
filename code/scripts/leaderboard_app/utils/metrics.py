import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# def compute_fairness_metrics(df: pd.DataFrame):
#     # Placeholder: replace with real fairness metrics like demographic parity, equalized odds, etc.
#     # Assume df includes columns like ['true_label', 'prediction', 'sensitive_attr']
#     if "sensitive_attr" in df.columns:
#         groups = df.groupby("sensitive_attr")
#         acc_by_group = groups.apply(lambda g: (g["true_label"] == g["prediction"]).mean())
#         disparity = acc_by_group.max() - acc_by_group.min()
#         return {"fairness_disparity": disparity}
#     else:
#         return {"fairness_disparity": None}


# =================================================================
# ------------------------ PERFORMANCE ----------------------------
# =================================================================

def classification_metrics(y_true, y_pred, pos_label=0):
    """
    Computes basic classification metrics: precision, recall, F1-score, and accuracy.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    pos_label : int or str
        Label considered as the positive class.

    Returns
    -------
    metrics_dict : dict
        Dictionary containing precision, recall, f1-score, and accuracy.
    """
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    metrics_dict = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

    return metrics_dict


# =================================================================
# --------------------------- FAIRNESS ----------------------------
# =================================================================

FAIRNESS_METRICS = [
    "Equal opportunity ratio (TPR)",
    "Predictive parity ratio (PPV)",
    "Predictive equality ratio (FPR)",
    "Accuracy equality ratio (ACC)",
    "Statistical parity ratio (PPR)",
]

def fairness_module(
    y_true,
    y_pred,
    attributes,
    pos_label=0,
    protected_attribute=0,
    epsilon=.8,
    title="Fairness Check for Binary Classifier",
    plot=False,
):
    """
    Computes fairness metrics (as dalex) and plots a dalex-like fairness check.
    For more information, please refere to the dalex package: https://dalex.drwhy.ai.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    pos_label : int or str
        Label of the positive class.
    attributes : array-like
        Protected attribute values per observation.
    protected_attribute : value
        Attribute value considered unprivileged.
    epsilon : float, default=.8
        Threshold defining the "fair zone" (ε ≤ ratio ≤ 1/ε).
    title : str, optional
        Title of the plot.

    Returns
    -------
    fairness_df : pd.DataFrame
        DataFrame with fairness ratios.
    max_epsilon : float
        Largest epsilon so that all ratios lie in [ε, 1/ε].
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    attributes = np.array(attributes)

    unpriv_mask = attributes == protected_attribute
    priv_mask = ~unpriv_mask

    def confusion_vals(y_t, y_p):
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[1 - pos_label, pos_label]).ravel()
        return tn, fp, fn, tp

    tn_u, fp_u, fn_u, tp_u = confusion_vals(y_true[unpriv_mask], y_pred[unpriv_mask])
    tn_p, fp_p, fn_p, tp_p = confusion_vals(y_true[priv_mask], y_pred[priv_mask])

    def safe_div(a, b): return a / b if b != 0 else np.nan

    # Unprivileged metrics
    tpr_u = safe_div(tp_u, tp_u + fn_u)
    ppv_u = safe_div(tp_u, tp_u + fp_u)
    fpr_u = safe_div(fp_u, fp_u + tn_u)
    acc_u = safe_div(tp_u + tn_u, tp_u + tn_u + fp_u + fn_u)
    pr_u  = safe_div(tp_u + fp_u, tp_u + tn_u + fp_u + fn_u)

    # Privileged metrics
    tpr_p = safe_div(tp_p, tp_p + fn_p)
    ppv_p = safe_div(tp_p, tp_p + fp_p)
    fpr_p = safe_div(fp_p, fp_p + tn_p)
    acc_p = safe_div(tp_p + tn_p, tp_p + tn_p + fp_p + fn_p)
    pr_p  = safe_div(tp_p + fp_p, tp_p + tn_p + fp_p + fn_p)

    ratios = {}
    if "Equal opportunity ratio (TPR)" in FAIRNESS_METRICS:
        ratios.update({"Equal opportunity ratio (TPR)": safe_div(tpr_u, tpr_p)})
    if "Predictive parity ratio (PPV)" in FAIRNESS_METRICS:
        ratios.update({"Predictive parity ratio (PPV)": safe_div(ppv_u, ppv_p)})
    if "Predictive equality ratio (FPR)" in FAIRNESS_METRICS:
        ratios.update({"Predictive equality ratio (FPR)": safe_div(fpr_u, fpr_p)})
    if "Accuracy equality ratio (ACC)" in FAIRNESS_METRICS:
        ratios.update({"Accuracy equality ratio (ACC)": safe_div(acc_u, acc_p)})
    if "Statistical parity ratio (PPR)" in FAIRNESS_METRICS:
        ratios.update({"Statistical parity ratio (PPR)": safe_div(pr_u, pr_p)})

    # ratios = {
    #     "Equal opportunity ratio (TPR)": safe_div(tpr_u, tpr_p),
    #     "Predictive parity ratio (Precision)": safe_div(ppv_u, ppv_p),
    #     "Predictive equality ratio (FPR)": safe_div(fpr_u, fpr_p),
    #     "Accuracy equality ratio (ACC)": safe_div(acc_u, acc_p),
    #     "Statistical parity ratio (PPR)": safe_div(pr_u, pr_p),
    # }

    fairness_df = pd.DataFrame(list(ratios.items()), columns=["metric", "ratio"])

    # Compute maximal epsilon such that all ratios are within [ε, 1/ε]
    max_epsilon = min(np.min(fairness_df["ratio"]), 1 / np.max(fairness_df["ratio"]))
    ratios.update({'fairness_epsilon':max_epsilon})

    # ---- Plot ----
    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = np.arange(len(fairness_df))
        ratios_values = fairness_df["ratio"].values

        # Fairness zones
        ax.axvspan(epsilon, 1/epsilon, color="palegreen", alpha=0.3, label=f"Fair zone [{epsilon:.2f}, {1/epsilon:.2f}]")
        ax.axvspan(0, epsilon, color="mistyrose", alpha=0.5)
        ax.axvspan(1/epsilon, max(1.5, max(ratios_values) + 0.1), color="mistyrose", alpha=0.5)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)

        # Bars
        ax.barh(y_pos, ratios_values-1, color="teal", alpha=0.8, left=1) # bars start from 1
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fairness_df["metric"])
        ax.invert_yaxis()  # like dalex: top metric first
        ax.set_xlabel("Unprivileged / Privileged Ratio", fontsize=11)
        ax.set_xlim(0, max(1.5, np.nanmax(ratios_values) + 0.1))
        ax.set_title(title, color="darkblue", fontsize=13)
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    #return fairness_df, max_epsilon
    return ratios
