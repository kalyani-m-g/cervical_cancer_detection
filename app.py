import streamlit as st
import pandas as pd
import joblib

# MUST be first Streamlit command
st.set_page_config(
    page_title="Cervical Cancer Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load model and scaler
model = joblib.load("cervical_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom CSS
st.markdown("""
<style>

.stButton > button {
    width:100%;
    height:55px;
    border-radius:12px;
    background-color:#1976D2;
    color:white;
    font-size:18px;
    font-weight:bold;
}

div[data-testid="stMetric"] {
    background:#F5F9FF;
    padding:15px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/stethoscope.png",
        width=80
    )

    st.title("About This Project")

    st.markdown("""
    ### 🩺 Cervical Cancer Risk Prediction

    This application uses a Machine Learning model
    trained on clinical and patient history data
    to estimate the likelihood of cervical cancer risk.

    #### 🔬 Model Details
    - Algorithm: XGBoost Classifier
    - Features Used: 33
    - Data Preprocessing: StandardScaler
    - Output: Risk Probability Score

    #### 📋 Parameters Analyzed
    - Personal Information
    - Smoking History
    - Contraceptive Usage
    - IUD Usage
    - STD History
    - Diagnostic Test Results

    #### 🎯 Purpose
    To demonstrate how machine learning can assist
    healthcare professionals in early risk assessment.

    ---
    👩‍💻 Developed by **Kalyani M G**
    
    
    """)

# Header
st.markdown("""
<h1 style='text-align:center;color:#1976D2;'>
🩺 Cervical Cancer Risk Prediction
</h1>

<p style='text-align:center;font-size:18px;color:gray;'>
Machine Learning Based Healthcare Risk Assessment
</p>
""", unsafe_allow_html=True)

st.markdown("---")

features = [
    'Age',
    'Number of sexual partners',
    'First sexual intercourse',
    'Num of pregnancies',
    'Smokes',
    'Smokes (years)',
    'Smokes (packs/year)',
    'Hormonal Contraceptives',
    'Hormonal Contraceptives (years)',
    'IUD',
    'IUD (years)',
    'STDs',
    'STDs (number)',
    'STDs:condylomatosis',
    'STDs:cervical condylomatosis',
    'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis',
    'STDs:syphilis',
    'STDs:pelvic inflammatory disease',
    'STDs:genital herpes',
    'STDs:molluscum contagiosum',
    'STDs:AIDS',
    'STDs:HIV',
    'STDs:Hepatitis B',
    'STDs:HPV',
    'STDs: Number of diagnosis',
    'Dx:Cancer',
    'Dx:CIN',
    'Dx:HPV',
    'Dx',
    'Hinselmann',
    'Schiller',
    'Citology'
]

binary_features = [
    'Smokes',
    'Hormonal Contraceptives',
    'IUD',
    'STDs',
    'Dx:Cancer',
    'Dx:CIN',
    'Dx:HPV',
    'Dx',
    'Hinselmann',
    'Schiller',
    'Citology',
    'STDs:cervical condylomatosis',
    'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis',
    'STDs:syphilis',
    'STDs:pelvic inflammatory disease',
    'STDs:genital herpes',
    'STDs:molluscum contagiosum',
    'STDs:AIDS',
    'STDs:HIV',
    'STDs:Hepatitis B',
    'STDs:HPV'
]

inputs = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(features):

    with col1 if i % 2 == 0 else col2:

        if feature in binary_features:

            inputs[feature] = (
                1 if st.selectbox(
                    feature,
                    ["No", "Yes"],
                    key=feature
                ) == "Yes"
                else 0
            )

        else:

            inputs[feature] = st.number_input(
                feature,
                min_value=0,
                value=0,
                step=1,
                key=feature
            )

if st.button("Predict Risk"):

    patient = pd.DataFrame([inputs])

    patient_scaled = scaler.transform(patient)

    prediction = model.predict(patient_scaled)
    probability = model.predict_proba(patient_scaled)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    st.metric(
        "Risk Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    if probability < 0.30:
        st.success("🟢 Low Risk")

    elif probability < 0.70:
        st.warning("🟡 Moderate Risk")

    else:
        st.error("🔴 High Risk")

    st.info("""
    ⚠️ This prediction is generated using a machine learning model
    and should not be considered a medical diagnosis.

    Please consult a healthcare professional for medical advice.
    """)
