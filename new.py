import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load Models and Scalers
@st.cache_data
def load_attrition_model_and_scaler():
    model_path = r"D:\4th_project\myenv\Scripts\attrition.pkl"
    scaler_path = r"D:\4th_project\myenv\Scripts\scaler.pkl"
    with open(model_path, "rb") as file:
        attrition_model = pickle.load(file)
    with open(scaler_path, "rb") as file:
        attrition_scaler = pickle.load(file)
    return attrition_model, attrition_scaler

# Sidebar Navigation
st.sidebar.title("Navigation")
if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("Home"):
    st.session_state.page = "Home"
if st.sidebar.button("Attrition Prediction"):
    st.session_state.page = "Attrition Prediction"

# Home Page
# Home Page
if st.session_state.page == "Home":
    st.markdown("<h1 style='text-align: center;'>📊 Employee Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Empowering HR teams with AI-driven insights</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/219/219983.png", width=150)

    with col2:
        st.markdown("""
        ### 🔍 What You Can Do:
        - ✅ *Attrition Prediction*  
          Predict whether an employee is likely to leave the company using key attributes like age, salary, tenure, and more.

        - 📈 *AI-Powered Insights*  
          Understand workforce trends to reduce turnover and boost retention.

        - 💼 *HR Decision Support*  
          Make informed HR decisions with the help of machine learning models.

        ---
        ### 🚀 Why Use This Tool?
        - Built with *real employee data*
        - Powered by *machine learning*
        - Designed for *ease of use*

        👉 Use the sidebar to get started!
        """)


# Attrition Prediction Page
elif st.session_state.page == "Attrition Prediction":
    model, scaler = load_attrition_model_and_scaler()
    st.title("Employee Attrition Prediction")
    st.write("Enter employee details to predict the likelihood of attrition.")

    # User Input Fields
    age = st.number_input("Age",  max_value=65, step=1, format="%d")
    monthly_income = st.number_input("Monthly Income", step=500, format="%d")
    years_at_company = st.number_input("Years at Company",  max_value=40, step=1, format="%d")
    job_satisfaction = st.number_input("Job Satisfaction (1 to 4)",  max_value=4, step=1, format="%d")

    department = st.selectbox("Department", ["Sales", "HR", "R&D"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    overtime = st.selectbox("Overtime", ["Yes", "No"])

    # Encoding Categorical Features
    department_encoded = [1, 0, 0] if department == "HR" else [0, 1, 0] if department == "R&D" else [0, 0, 1]
    marital_status_encoded = [0, 0, 1] if marital_status == "Single" else [0, 1, 0] if marital_status == "Married" else [1, 0, 0]
    overtime_encoded = [1, 0] if overtime == "No" else [0, 1]

    # Create Feature Array
    user_input = np.array([[age, monthly_income, job_satisfaction, years_at_company] + 
                           department_encoded + marital_status_encoded + overtime_encoded], dtype=float)

    if st.button("Predict Attrition"):
        try:
            user_input[:, :4] = scaler.transform(user_input[:, :4])  # Scale only numerical features
            prediction = model.predict(user_input)[0]
            if prediction == 1:
                st.error("🚨 The employee is likely to leave.")
            else:
                st.success("✅ The employee is likely to stay.")
        except Exception as e:
            st.warning(f"⚠️ Error: {str(e)}")