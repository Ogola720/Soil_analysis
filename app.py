import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import joblib
import os

st.set_page_config(page_title="🌱 Crop Recommendation AI", layout="wide")

# Paths to save/load models
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode:", ["📂 Upload Dataset", "🌱 Direct Crop Recommendation"])

# --------------------------
# MODE 1: UPLOAD DATASET
# --------------------------
if app_mode == "📂 Upload Dataset":
    st.sidebar.header("Upload Your Soil Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Dataset")
        st.dataframe(df.head())

        # Target Selection
        target = st.sidebar.selectbox("Select Target Column (e.g., label/crop)", df.columns)

        # Drop non-numeric features except target
        numeric_features = df.drop(columns=[target]).select_dtypes(include=[np.number])
        X = numeric_features
        y = df[target]

        # Encode target crops
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # ✅ Drop classes with <2 samples
        class_counts = pd.Series(y).value_counts()
        valid_classes = class_counts[class_counts > 1].index
        mask = y.isin(valid_classes)
        X, y, y_encoded = X[mask], y[mask], y_encoded[mask]

        if len(valid_classes) < 2:
            st.error("❌ Not enough valid classes. Need ≥2 crops with at least 2 samples.")
        else:
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Train model
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)

            rf_model = RandomForestClassifier(random_state=42, n_estimators=200)
            rf_model.fit(X_train, y_train)

            # Save model + encoder
            joblib.dump(rf_model, MODEL_PATH)
            joblib.dump(le, ENCODER_PATH)

            # Predictions
            y_pred = rf_model.predict(X_test)
            y_pred_labels = le.inverse_transform(y_pred)
            y_test_labels = le.inverse_transform(y_test)

            # Metrics
            st.subheader("📈 Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            with col2:
                st.text("Classification Report:")
                st.text(classification_report(y_test_labels, y_pred_labels))

            # Feature Importances
            st.subheader("🌍 Global Feature Importances")
            fi_df = pd.DataFrame({"Feature": X.columns, "Importance": rf_model.feature_importances_})
            fi_df = fi_df.sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis", ax=ax)
            ax.set_title("Random Forest Feature Importances", fontsize=14)
            st.pyplot(fig)
            st.dataframe(fi_df)

            # SHAP Explanations
            st.subheader("🧪 Local Explanations with SHAP")
            explainer = shap.TreeExplainer(rf_model)

            st.write("Upload a CSV with a **single soil sample row** for explanation:")
            sample_file = st.file_uploader("Upload Sample CSV", type=["csv"], key="sample")

            if sample_file is not None:
                sample_df = pd.read_csv(sample_file)
                st.write("Uploaded Sample:")
                st.dataframe(sample_df)

                # Ensure sample has same features
                if set(sample_df.columns) == set(X.columns):
                    shap_values = explainer.shap_values(sample_df)

                    # Prediction
                    pred_encoded = rf_model.predict(sample_df)[0]
                    pred_crop = le.inverse_transform([pred_encoded])[0]
                    st.success(f"✅ Recommended Crop: **{pred_crop}**")

                    # Top-3 Probabilities
                    probs = rf_model.predict_proba(sample_df)[0]
                    top3_idx = np.argsort(probs)[::-1][:3]
                    top3_crops = le.inverse_transform(top3_idx)
                    top3_probs = probs[top3_idx] * 100
                    st.write("📊 Top-3 Crop Recommendations:")
                    for c, p in zip(top3_crops, top3_probs):
                        st.write(f"- {c}: {p:.2f}%")

                    # SHAP Plots
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.initjs()
                    class_idx = pred_encoded
                    shap.force_plot(
                        explainer.expected_value[class_idx],
                        shap_values[class_idx][0, :],
                        sample_df.iloc[0, :],
                        matplotlib=True, show=False
                    )
                    st.pyplot(bbox_inches="tight")

                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values[class_idx], sample_df, plot_type="bar", show=False)
                    st.pyplot(fig)
                else:
                    st.error("❌ Sample columns do not match training features.")
    else:
        st.info("👈 Please upload a dataset to get started.")

# --------------------------
# MODE 2: DIRECT RECOMMENDATION
# --------------------------
elif app_mode == "🌱 Direct Crop Recommendation":
    st.header("🌾 Direct Crop Recommendation Interface")

    # Load trained model + encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        rf_model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
    else:
        st.error("⚠️ No trained model found. Please upload and train a dataset first.")
        st.stop()

    # User Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.slider("Nitrogen (N)", 0, 200, 50)
        P = st.slider("Phosphorus (P)", 0, 200, 50)
        K = st.slider("Potassium (K)", 0, 200, 50)
    with col2:
        temperature = st.slider("Temperature (°C)", 0, 50, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
    with col3:
        ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.slider("Rainfall (mm)", 0, 300, 100)

    # Build DataFrame
    sample_input = pd.DataFrame([{
        "N": N, "P": P, "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    # Predict Crop
    if st.button("🌱 Recommend Crop"):
        probs = rf_model.predict_proba(sample_input)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_crops = le.inverse_transform(top3_idx)
        top3_probs = probs[top3_idx] * 100

        st.success(f"✅ Best Crop: **{top3_crops[0]}** ({top3_probs[0]:.2f}%)")
        st.write("📊 Top-3 Crop Recommendations:")
        for c, p in zip(top3_crops, top3_probs):
            st.write(f"- {c}: {p:.2f}%")
