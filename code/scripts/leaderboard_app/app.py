import os
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import uuid
from utils.metrics import (
    fairness_module, 
    classification_metrics
)
from utils.charts import (
    compute_beta_scores_multi,
    plot_tradeoff_multi,
    plot_fairness_ratios,
    plot_confusion_matrices 
)
from utils.encode_decode import encode_b64, decode_b64_into_df
from utils.submission_check import check_before_submission


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Tutorial Leaderboard", layout="wide")

RESULTS_PATH = "results/leaderboard.csv"
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

# =========================
# AUTO-REFRESH
# =========================
# Refresh leaderboard every 15 seconds (adjust interval_ms if desired)
try:
    from streamlit_autorefresh import st_autorefresh  # optional helper if installed
except ImportError:
    pass

st_autorefresh = st.autorefresh if hasattr(st, "autorefresh") else None
if st_autorefresh:
    st_autorefresh(interval=15 * 1000, key="leaderboard_refresh")

##########################
# Dirty hard code
VISUALISE_COLUMNS = ["submission", "accuracy", "precision", "recall", "f1", "fairness_epsilon"]

# =========================
# LOAD SECRET LABELS
# =========================
@st.cache_data
# def load_true_labels():
#     df = pd.read_csv(SECRET_PATH)
#     return df
def load_true_labels():
    b64 = st.secrets["data"]["canon_secret_test_b64"]
    return decode_b64_into_df(b64)


true_df = load_true_labels()

# =========================
# MAIN APP
# =========================
st.title("📓 Canonicity Prediction Shared Task")
st.write("Upload your prediction CSV to see your scores!")

if "submitted_files" not in st.session_state:
    st.session_state.submitted_files = set()  # store uploaded filenames

uploaded_file = st.file_uploader("Drop your predictions CSV here", type=["csv"])

if (uploaded_file is not None) and (uploaded_file.name not in st.session_state.submitted_files):
    try:
        preds = pd.read_csv(uploaded_file)
        submission_compliant = check_before_submission(preds)
        if not submission_compliant:
            st.error("Your file does not have the correct format.")
        else:
            st.success("✅ File successfully uploaded!")

            # Align with true labels
            merged = true_df.copy()
            merged["prediction"] = preds["prediction"].values

            y_true = merged["label"]
            y_pred = merged["prediction"]

            with st.expander("🔲 View Confusion Matrices"):
                fig_cm = plot_confusion_matrices(
                    y_true=merged["label"],
                    y_pred=merged["prediction"],
                    attributes=merged["author_gender"]
                )
                st.pyplot(fig_cm)


            submission_name = uploaded_file.name.replace(".csv", "")
            results = { "submission": submission_name }
            
            # Performance metrics
            performance_metrics = classification_metrics(
                y_true=y_true, 
                y_pred=y_pred,
                # pos_label = 0 # = 'canon'
            )
            results.update(performance_metrics)

            # Fairness metrics
            fairness = fairness_module(
                y_true=y_true,
                y_pred=y_pred,
                attributes=merged["author_gender"],
                # pos_label = 0, # = 'canon'
                # protected_attribute = 0, # = 'female'
            )
            results.update(fairness)

            df_result = pd.DataFrame([results])

            # Load existing leaderboard
            try:
                leaderboard = pd.read_csv(RESULTS_PATH)
                leaderboard = pd.concat([leaderboard, df_result], ignore_index=True)
            except FileNotFoundError:
                leaderboard = df_result

            leaderboard.to_csv(RESULTS_PATH, index=False)

            st.success(f"Results added for **{submission_name}**!")

            # Mark this file as processed
            st.session_state.submitted_files.add(uploaded_file.name)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Awaiting file upload...")


# =========================
# ALWAYS SHOW LEADERBOARD
# =========================
st.subheader("🧭 Indicators table")

try:
    leaderboard = pd.read_csv(RESULTS_PATH)
    leaderboard = leaderboard

    # =========================
    # COLOR GRADIENTS
    # =========================
    perf_cols = ["accuracy", "precision", "recall", "f1"]
    fairness_cols = [c for c in leaderboard.columns if "fair" in c or "bias" in c]

    # Apply different colormaps: performance (Greens), fairness (Purples)
    styled_df = leaderboard.style

    if perf_cols:
        styled_df = styled_df.background_gradient(
            subset=perf_cols, cmap="YlGnBu",# cmap="Greens",
            vmin=0, vmax=1
        )
    if fairness_cols:
        styled_df = styled_df.background_gradient(
            subset=fairness_cols, cmap="RdPu",#cmap="Purples",
            vmin=0, vmax=1
        )

    # For displaying
    st.dataframe(
        leaderboard[VISUALISE_COLUMNS].style
        .background_gradient(subset=perf_cols, cmap="YlGnBu", vmin=0, vmax=1)
        .background_gradient(subset=fairness_cols, cmap="RdPu", vmin=0, vmax=1)
        .set_properties(subset=["submission"], **{"font-weight": "bold"}),
        use_container_width=True
    )

except FileNotFoundError:
    st.warning("No submissions yet. The leaderboard will appear here once participants upload their results.")

st.subheader("⚖️ Performance vs Fairness Trade-off")
st.text("""The following plot displays a trade-off score between performance and fairness based on a β weighting-parameter. 
The lower values of β give more importance to fairness, while greater values favor performance over bias.
It is computed as:""")
st.latex(r"\mathrm{Score}=\beta\cdot\mathrm{f_1}+(1-\beta)\cdot\epsilon^\star")
st.text("""where ϵ* is comuted as the largest possbile value so that all fairness indicators are within [ϵ, 1/ϵ] (meaning ϵ* closer to 1 is better).
""")

try:
    leaderboard = pd.read_csv(RESULTS_PATH)

    if not leaderboard.empty:
        # Compute beta-weighted scores for all submissions
        df_beta = compute_beta_scores_multi(leaderboard, perf_col="f1", fairness_col="fairness_epsilon", n=50)
        
        # Plot interactive chart
        st.altair_chart(plot_tradeoff_multi(df_beta), use_container_width=True)

except FileNotFoundError:
    st.warning("No submissions yet. The trade-off chart will appear once participants upload results.")


st.subheader("📈 Fairness Indicators per Submission")

try:
    leaderboard = pd.read_csv(RESULTS_PATH)

    if not leaderboard.empty:
        submission_to_plot = st.selectbox(
            "Select submission", leaderboard["submission"].tolist()
        )

        row = leaderboard[leaderboard["submission"] == submission_to_plot].iloc[0]

        fig = plot_fairness_ratios(row, epsilon=0.8)
        st.pyplot(fig)

except FileNotFoundError:
    st.warning("No submissions yet. Fairness visualizations will appear here once participants upload results.")

