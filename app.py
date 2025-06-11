
import streamlit as st
from utils import load_model, load_data, evaluate_model, predict_single

st.title("🏡 Ames Housing Price Prediction Dashboard")

model, feature_names = load_model()

uploaded_file = st.file_uploader("Upload your test data CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("Raw Uploaded Data")
    st.dataframe(df.head())

    st.subheader("📊 Model Evaluation")
    try:
        scores = evaluate_model(model, feature_names, df)
        for k, v in scores.items():
            st.write(f"{k}: {v:.4f}")
    except Exception as e:
        st.error(f"Evaluation error: {e}")

st.subheader("🔍 Predict House Price")

user_inputs = {}
for feature in feature_names:
    user_inputs[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Predict"):
    try:
        prediction = predict_single(model, user_inputs, feature_names)
        st.success(f"Predicted SalePrice: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
