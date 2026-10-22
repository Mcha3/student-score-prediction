import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "student_score_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🎓 Student Score Prediction")
st.write(
    "Enter the student's information to estimate a score "
    "using our demonstration machine-learning model."
)

st.info(
    "Educational demonstration only. The model uses synthetic data "
    "and must not be used to grade, rank, admit, or penalize real students."
)

study_hours = st.number_input(
    "Study hours",
    min_value=0.0,
    max_value=24.0,
    value=6.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=1.0
)

previous_score = st.number_input(
    "Previous score",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

if st.button("Predict score"):
    student = pd.DataFrame([{
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PreviousScore": previous_score
    }])

    predicted_score = float(model.predict(student)[0])
    predicted_score = max(0.0, min(100.0, predicted_score))

    st.success(f"Estimated score: {predicted_score:.2f} / 100")

    st.caption(
        "This estimate is not a verified assessment of a real student's ability."
    )

st.subheader("Responsible AI considerations")
st.write("- The training data is synthetic.")
st.write("- Predictions may be inaccurate.")
st.write("- Human judgement must not be replaced by this model.")
st.write("- Student information should be handled securely.")
