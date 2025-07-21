import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and preprocessing objects
model = joblib.load("best_salary_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define categorical and numerical columns
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Salary Prediction App")
st.markdown("Enter your details below to predict whether your income is likely to be above or below **50K USD**.")

# --- Numeric Inputs ---
st.header("🔢 Numeric Information")
age = st.slider("Age", min_value=18, max_value=100, value=30)
fnlwgt = st.number_input("FNLWGT", min_value=10000, max_value=1000000, value=200000, step=1000)
educational_num = st.slider("Education Number", min_value=1, max_value=20, value=10)
capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=100)
capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, step=50)
hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100, value=40)

# --- Categorical Inputs ---
st.header("📋 Categorical Information")
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov',
                                       'State-gov', 'Without-pay', 'Never-worked'])
education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                                       'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th'])
marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                                                 'Widowed', 'Married-spouse-absent'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                                         'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
                                         'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
gender = st.radio("Gender", ['Male', 'Female'])
native_country = st.selectbox("Native Country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'England', 'China'])

# --- Prediction ---
if st.button("🚀 Predict Salary"):
    input_dict = {
        'age': age,
        'fnlwgt': fnlwgt,
        'educational-num': educational_num,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'workclass': workclass,
        'education': education,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'native-country': native_country
    }

    st.subheader("📝 Input Summary")
    st.json(input_dict)

    input_df = pd.DataFrame([input_dict])

    # Preprocess
    X_num = pd.DataFrame(scaler.transform(input_df[numerical_cols]), columns=numerical_cols)
    X_cat_array = encoder.transform(input_df[categorical_cols])
    if hasattr(X_cat_array, "toarray"):
        X_cat_array = X_cat_array.toarray()
    X_cat = pd.DataFrame(X_cat_array, columns=encoder.get_feature_names_out(categorical_cols))

    X_final = pd.concat([X_num, X_cat], axis=1)

    prediction = model.predict(X_final)[0]
    salary_label = label_encoder.inverse_transform([prediction])[0]

    st.subheader("💡 Prediction Result")
    if salary_label == "<=50K":
        st.success("🧾 The predicted salary is **less than or equal to 50K USD**.")
    else:
        st.success("💰 The predicted salary is **greater than 50K USD**.")
