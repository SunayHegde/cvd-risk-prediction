"""Live demo: enter a patient's details and get a predicted CVD risk from the trained XGBoost model.
Run:  streamlit run demo_app.py      (needs xgb_cardio_model.joblib, created by the last notebook cell)"""
import joblib, pandas as pd, streamlit as st

bundle = joblib.load("xgb_cardio_model.joblib")
model, features = bundle["model"], bundle["features"]

st.title("Cardiovascular Disease Risk Predictor")
st.caption("Course demo (UE24CS352A). Trained on the Kaggle Cardiovascular Disease dataset; not medical advice.")

c1, c2 = st.columns(2)
age = c1.number_input("Age (years)", 25, 80, 55)
gender = c2.selectbox("Gender code (1 or 2, as in the dataset)", [1, 2])
height = c1.number_input("Height (cm)", 120, 220, 168)
weight = c2.number_input("Weight (kg)", 30.0, 200.0, 75.0)
ap_hi = c1.number_input("Systolic blood pressure (ap_hi)", 70, 250, 130)
ap_lo = c2.number_input("Diastolic blood pressure (ap_lo)", 40, 150, 85)
chol = c1.selectbox("Cholesterol (1 normal, 2 above normal, 3 well above)", [1, 2, 3])
gluc = c2.selectbox("Glucose (1 normal, 2 above normal, 3 well above)", [1, 2, 3])
smoke = c1.selectbox("Smoker", [0, 1], format_func=lambda v: "Yes" if v else "No")
alco = c2.selectbox("Alcohol intake", [0, 1], format_func=lambda v: "Yes" if v else "No")
active = c1.selectbox("Physically active", [1, 0], format_func=lambda v: "Yes" if v else "No")

if st.button("Predict risk"):
    row = {"age": age, "gender": gender, "height": height, "weight": weight, "ap_hi": ap_hi, "ap_lo": ap_lo,
           "cholesterol": chol, "gluc": gluc, "smoke": smoke, "alco": alco, "active": active,
           "bmi": weight / (height / 100) ** 2}
    p = float(model.predict_proba(pd.DataFrame([row])[features].values.astype("float32"))[0, 1])
    st.metric("Predicted probability of CVD", f"{p:.1%}")
    st.progress(min(p, 1.0))
    st.write("BMI used:", round(row["bmi"], 1))
