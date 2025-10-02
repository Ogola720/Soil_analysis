import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap

st.set_page_config(page_title="🌱 Crop Recommendation AI", layout="wide")

# Sidebar for Data Upload
st.sidebar.header("📂 Upload Your Soil Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Dataset")
    st.dataframe(df.head())

    # Target Selection
    target = st.sidebar.selectbox("Select Target Column (e.g., crop)", df.columns)

    # Feature/Target Split
    X = df.drop(columns=[target])
    y = df[target]

    # ✅ Drop classes with fewer than 2 samples
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts > 1].index
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]

    if len(valid_classes) < 2:
        st.error("❌ Not enough valid classes for classification. Please upload a dataset with at least 2 crops having ≥2 samples each.")
    else:
        # Train/Test Split with stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Model Training (Classifier)
        rf_model = RandomForestClassifier(random_state=42, n_estimators=200)
        rf_model.fit(X_train, y_train)

        # Predictions
        y_pred = rf_model.predict(X_test)

        # Metrics
        st.subheader("📈 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        with col2:
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        # Global Feature Importances
        st.subheader("🌍 Global Feature Importances")
        importances = rf_model.feature_importances_
        feature_names = X.columns
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax)
        ax.set_title("Random Forest Feature Importances", fontsize=14)
        st.pyplot(fig)
        st.dataframe(fi_df)

        # Local Explanations (SHAP)
        st.subheader("🧪 Local Explanations with SHAP")
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X)

        # Upload a single soil sample for explanation
        st.write("Upload a CSV with a **single soil sample row** for local explanation:")
        sample_file = st.file_uploader("Upload Sample CSV", type=["csv"], key="sample")

        if sample_file is not None:
            sample_df = pd.read_csv(sample_file)
            st.write("Uploaded Sample:")
            st.dataframe(sample_df)

            if set(sample_df.columns) == set(X.columns):
                shap_sample = explainer.shap_values(sample_df)

                # Prediction for uploaded row
                pred_crop = rf_model.predict(sample_df)[0]
                st.success(f"✅ Recommended Crop for this soil sample: **{pred_crop}**")

                # SHAP Explanation
                st.write("🔬 SHAP Explanation for this Sample:")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                shap.initjs()

                # Force Plot (per-class explanation for predicted class)
                class_idx = list(rf_model.classes_).index(pred_crop)
                shap.force_plot(
                    explainer.expected_value[class_idx],
                    shap_sample[class_idx][0, :],
                    sample_df.iloc[0, :],
                    matplotlib=True, show=False
                )
                st.pyplot(bbox_inches="tight")

                # Summary Plot
                fig, ax = plt.subplots()
                shap.summary_plot(shap_sample[class_idx], sample_df, plot_type="bar", show=False)
                st.pyplot(fig)
            else:
                st.error("❌ The uploaded sample columns do not match the training features.")
else:
    st.info("👈 Please upload a dataset to get started.")
