import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="FraudShield AI",
    page_icon=Image.open("logo/logo.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

logo = Image.open("logo/logo.png")

st.markdown("""
<style>

[data-testid="stSidebar"]{
    background-color:#2563EB;
}

[data-testid="stSidebar"] *{
    color:white;
}

[data-testid="stSidebar"] a{
    color:#FFD60A !important;
    font-weight:600;
    text-decoration:none;
}

[data-testid="stSidebar"] a:hover{
    color:#FFF176 !important;
    text-decoration:underline;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost.pkl")

@st.cache_data
def load_data():
    X_test = pd.read_csv("data/app_data/X_test_sample.csv")
    Y_test = pd.read_csv("data/app_data/Y_test_sample.csv").squeeze()
    return X_test, Y_test

model     = load_model()
explainer = shap.TreeExplainer(model)
X_test, Y_test = load_data()

fraud_count   = int(Y_test.sum())
genuine_count = int((Y_test == 0).sum())
total         = len(Y_test)

with st.sidebar:
    st.image(logo, width=140)
    st.title("FraudShield AI")
    st.caption("Credit Card Fraud Detection System")
    st.divider()

    st.subheader("📊 Dataset Overview")
    st.metric("Total Transactions", f"{total:,}")
    st.metric("Fraud Cases",   f"{fraud_count:,}",   f"{fraud_count/total*100:.3f}% of total")
    st.metric("Genuine Cases", f"{genuine_count:,}", f"{genuine_count/total*100:.3f}% of total")
    st.divider()

    st.subheader("🤖 Model Performance (Test Set)")
    st.metric("Model Used",    "XGBoost")
    st.metric("PR-AUC Score",  "0.8487")
    st.metric("ROC-AUC Score", "0.9812")
    st.metric("Fraud Recall",  "82%")
    st.divider()

    st.subheader("ℹ️ How to Use")
    st.info("""
    1. Go to **Transaction Analyzer** tab
    2. Select a transaction index
    3. Click **Analyze Transaction**
    4. View fraud probability
    5. Read the SHAP explanation
    """)
    st.divider()

    st.markdown("**Connect with me:**")
    st.markdown("[GitHub](https://github.com/parthshelar-dev)")
    st.markdown("[Project Repo](https://github.com/parthshelar-dev/credit-card-fraud-detection)")
    st.markdown("[LinkedIn](https://linkedin.com/in/parth-shelar)")
    st.caption("Built by Parth Shelar")

col1, col2 = st.columns([1.2, 8])

with col1:
    st.image(logo, width=120)

with col2:
    st.title("FraudShield AI")
    st.caption("AI-Powered Credit Card Fraud Detection System")

st.write(
    "Analyze any transaction from the test dataset and understand the model's prediction using SHAP explainability."
)

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{total:,}")
col2.metric("Fraud Rate",   "0.173%", f"{fraud_count} fraud cases")
col3.metric("PR-AUC",       "0.8487", "XGBoost")
col4.metric("Fraud Recall", "82%",    "101 of 123 caught")
st.divider()

tab1, tab2 = st.tabs(["🔍 Transaction Analyzer", "📊 Model Performance"])

with tab1:

    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.subheader("🔎 Select Transaction")

        index = st.number_input(
            "Transaction Index",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
        )

        sample       = X_test.iloc[[index]]

        st.subheader("📋 Transaction Features")

        important_features = ["Amount", "Time", "V14", "V4", "V12", "V3", "V10", "V11"]
        st.dataframe(
            sample[important_features].T.rename(columns={sample.index[0]: "Value"}).round(4),
            use_container_width=True,
            height=300,
        )
        st.caption("Displaying top 8 important features. V1–V28 are PCA-transformed.")
        st.divider()

        analyze = st.button("🔍 Analyze Transaction", use_container_width=True)

    with right:
        st.subheader("📈 Analysis Result")

        if analyze:
            prediction  = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0][1]
            actual_label = Y_test.iloc[index]


            if prediction == 1:
                st.error("⚠️ **Fraud Detected**\n\nThis transaction has been classified as potentially fraudulent.")
            else:
                st.success("✅ **Genuine Transaction**\n\nNo fraud indicators were detected.")

            st.metric(
                "Fraud Probability",
                f"{probability*100:.2f}%",
                "High Risk" if probability > 0.5 else "Low Risk",
                delta_color="inverse",
            )
            st.divider()

            st.subheader("📌 Ground Truth")

            if actual_label == 1:
                st.write("**Actual Label:** 🔴 Fraud")
            else:
                st.write("**Actual Label:** 🟢 Genuine")

            if prediction == actual_label:
                st.success("✅ Prediction matched the actual label.")
            else:
                st.warning("⚠️ Prediction did not match the actual label.")

            st.divider()

            st.subheader("🧠 Why did the model make this prediction?")
            st.caption("""
            The SHAP waterfall plot explains the prediction.
            🔴 Red bars increase the fraud probability.
            🔵 Blue bars decrease the fraud probability.
            Longer bars indicate a stronger influence on the model's decision.
            """)

            with st.spinner("Generating SHAP explanation..."):
                explanation = explainer(sample)
                fig = plt.figure(figsize=(9, 6))
                shap.plots.waterfall(explanation[0], show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        else:
            st.info("Select a transaction and click **Analyze Transaction**.")

with tab2:

    st.subheader("📊 Model Comparison")
    st.caption("All metrics evaluated on the original imbalanced test set (71,202 transactions).")

    st.markdown("#### Classification Metrics")
    metrics_df = pd.DataFrame({
        "Model":     ["Logistic Regression", "Random Forest", "XGBoost ✅"],
        "Precision": [0.06, 0.85, 0.73],
        "Recall":    [0.89, 0.80, 0.82],
        "F1-Score":  [0.11, 0.83, 0.77],
        "PR-AUC":    [0.7111, 0.8515, 0.8487],
        "ROC-AUC":   [0.9720, 0.9740, 0.9812],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("#### Confusion Matrix Results")
    st.caption("Out of 123 actual fraud cases in the test set:")

    confusion_df = pd.DataFrame({
        "Model":              ["Logistic Regression", "Random Forest", "XGBoost ✅"],
        "Fraud Caught (TP)":  [109, 99, 101],
        "Fraud Missed (FN)":  [14,  24, 22],
        "False Alarms (FP)":  [1712, 17, 37],
        "Correct Genuine (TN)": [69367, 71062, 71042],
    })
    st.dataframe(confusion_df, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("#### Model Selection by Business Priority")
    priority_df = pd.DataFrame({
        "Priority":   [
            "Minimize false alarms",
            "Catch maximum fraud",
            "Best overall balance",
            "Best discrimination"
        ],
        "Best Model": [
            "Random Forest (only 17 FP)",
            "Logistic Regression (109 caught)",
            "XGBoost (101 caught, 37 FP)",
            "XGBoost (ROC-AUC 0.9812)"
        ],
    })
    st.dataframe(priority_df, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("#### SHAP Feature Importance")
    st.caption("Which features drive fraud predictions the most?")

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("**Global Feature Importance (Bar)**")
        st.image("images/shap_importance.png", use_container_width=True)
        st.caption("Top features ranked by average impact across all transactions.")

    with img_col2:
        st.markdown("**Feature Direction (Beeswarm)**")
        st.image("images/shap_beeswarm.png", use_container_width=True)
        st.caption("Red = high feature value, Blue = low. Right = pushes toward fraud.")