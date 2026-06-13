import os
import streamlit as st
import pandas as pd
import h2o

# 1. Page Configuration
st.set_page_config(page_title="Stroke Prediction Dashboard", layout="centered")
st.title("🧠 Stroke Risk Prediction App")
st.write("Input patient metrics below to estimate stroke risk using the trained H2O Gradient Boosting Model.")

# 2. Optimized H2O & Model Loading (Runs only once)
@st.cache_resource
def load_h2o_and_model():
    # Spin up background H2O micro-instance for execution
    h2o.init(nthreads=2, max_mem_size='2G', bind_to_localhost=True)
    
    # Find the generated mojo zip file from the model directory
    mojo_files = [f for f in os.listdir('model') if f.endswith('.zip')]
    if not mojo_files:
        raise FileNotFoundError("No trained MOJO model zip found in the 'model/' directory. Run train_model.py first.")
        
    model_path = os.path.join('model', mojo_files[0])
    return h2o.import_mojo(model_path)

try:
    model = load_h2o_and_model()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# 3. Build UI Input Form Layout
st.header("Patient Demographics & Metrics")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age (Years)", min_value=0.0, max_value=120.0, value=50.0, step=1.0)
    hypertension = st.selectbox("Hypertension History?", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease History?", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married?", ["Yes", "No"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=30.0, max_value=400.0, value=100.0, step=0.1)
    
    bmi_known = st.radio("Is BMI metric available?", ["Yes", "No (Treat as Missing)"])
    if bmi_known == "Yes":
        bmi = st.number_input("BMI Value", min_value=10.0, max_value=80.0, value=28.0, step=0.1)
    else:
        bmi = -99.0  # Fills missing values with placeholder exactly matching your training script
    
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# 4. Process Inputs and Run Predictions
if st.button("Predict Stroke Risk", type="primary"):
    # Map binary UI selections back to original training integers (0 and 1)
    hyp_val = 1 if hypertension == "Yes" else 0
    hd_val = 1 if heart_disease == "Yes" else 0
    
    # Create dictionary matching feature names and data types perfectly to the training data
    input_data = {
        "age": [float(age)],
        "avg_glucose_level": [float(avg_glucose_level)],
        "bmi": [float(bmi)],
        "gender": [str(gender)],
        "hypertension": [int(hyp_val)],      # Stored as numeric integer
        "heart_disease": [int(hd_val)],      # Stored as numeric integer
        "ever_married": [str(ever_married)],
        "work_type": [str(work_type)],
        "residence_type": [str(residence_type)],
        "smoking_status": [str(smoking_status)]
    }
    
    try:
        # Create a Pandas DataFrame first to preserve strict column ordering
        pd_frame = pd.DataFrame(input_data)
        
        # Convert the Pandas DataFrame into an H2OFrame safely
        input_frame = h2o.H2OFrame(pd_frame)
        
        # Explicitly enforce factor schemas ONLY for true text-based categorical columns
        categorical_cols = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
        for col in categorical_cols:
            input_frame[col] = input_frame[col].asfactor()

        # Score data safely via the underlying MOJO pipeline
        prediction_output = model.predict(input_frame)
        prediction_df = prediction_output.as_data_frame()
        
        # Extract the positive case probability ('p1')
        risk_probability = prediction_df['p1'].iloc[0]
        
        # 5. Display Clean UI Results Box
        st.markdown("---")
        st.subheader("Analysis Results")
        
        # Correctly calibrated classification matching your notebook's custom threshold (tt = 0.148)
        if risk_probability >= 0.148:
            st.error(f"⚠️ High Risk Level detected (Above notebook threshold of 14.8%). Calculated probability: **{risk_probability * 100:.2f}%**")
        elif risk_probability >= 0.05:
            st.warning(f"⚡ Moderate/Elevated Risk Level detected. Calculated probability: **{risk_probability * 100:.2f}%**")
        else:
            st.success(f"✅ Low Base Risk Level calculated. Probability: **{risk_probability * 100:.2f}%**")
            
    except Exception as prediction_error:
        st.error(f"Prediction Pipeline Error: {prediction_error}")