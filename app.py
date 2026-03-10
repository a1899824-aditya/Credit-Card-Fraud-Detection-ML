import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

st.set_page_config(page_title="Fraud Detection App", layout="wide")

# -----------------------------
# Load Models
# -----------------------------
models = {
    "RandomForest": joblib.load("final_randomforest.pkl"),
    "XGBoost": joblib.load("final_xgboost.pkl"),
    "LogisticRegression": joblib.load("final_logisticregression.pkl")
}

# Load selected features
selected_features = joblib.load("final.features.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🕵️ Fraud Detection App")
st.write("📂 Upload a dataset and select a model to detect fraud.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:

    # -----------------------------
    # Load CSV
    # -----------------------------
    input_data = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(input_data.head())

    # -----------------------------
    # Create Hour column if missing
    # -----------------------------
    if "Hour" not in input_data.columns and "Time" in input_data.columns:
        input_data["Hour"] = (input_data["Time"] // 3600).astype(int)
        st.info("🕒 'Hour' column created from 'Time'.")

    # -----------------------------
    # Column Validation
    # -----------------------------
    st.subheader("📋 Expected Input Features:")
    st.code(selected_features)

    missing_cols = [c for c in selected_features if c not in input_data.columns]
    extra_cols = [c for c in input_data.columns if c not in selected_features and c != "Class"]

    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        st.stop()
    else:
        st.success("✅ All required columns found.")

    if extra_cols:
        st.info(f"ℹ️ Extra columns will be ignored: {extra_cols}")

    # -----------------------------
    # Reorder columns
    # -----------------------------
    input_data = input_data[selected_features]

    # -----------------------------
    # Model Selection
    # -----------------------------
    model_option = st.selectbox(
        "🧠 Choose model for prediction:",
        ["RandomForest", "XGBoost", "LogisticRegression"]
    )

    model = models[model_option]

    # -----------------------------
    # Predict Button
    # -----------------------------
    if st.button("🔍 Predict Fraud"):

        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]

        # Build output DataFrame
        result_df = input_data.copy()
        result_df["Hour"] = input_data["Hour"] if "Hour" in input_data.columns else None
        result_df["Fraud_Prediction"] = predictions
        result_df["Fraud_Probability"] = probabilities.round(4)
        result_df["Model_Used"] = model_option

        st.subheader("🔎 Predictions (Preview)")
        st.dataframe(result_df.head(15))

        # Save predictions
        result_df.to_csv("predictions.csv", index=False)
        st.success("📁 predictions.csv saved")

        # -----------------------------
        # Feature Importances
        # -----------------------------
        if model_option in ["RandomForest", "XGBoost"]:
            feature_df = pd.DataFrame({
                "Feature": selected_features,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            feature_df.to_csv("feature_importance.csv", index=False)
            st.success("📈 feature_importance.csv saved")

            # -----------------------------
            # SHAP Values (optional)
            # -----------------------------
            explainer = shap.Explainer(model, input_data)
            shap_values = explainer(input_data)

            shap_df = pd.DataFrame(shap_values.values, columns=selected_features)
            shap_df.to_csv("shap_values.csv", index=False)
            st.success("🧠 shap_values.csv saved")

        else:
            # Logistic Regression coefficients
            coefs = model.coef_[0]
            feature_df = pd.DataFrame({
                "Feature": selected_features,
                "Importance": np.abs(coefs)
            }).sort_values(by="Importance", ascending=False)

            feature_df.to_csv("feature_importance.csv", index=False)
            st.success("📈 feature_importance.csv saved (coefficients)")
