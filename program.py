import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# GitHub URL for dataset (raw link)
data_url = "https://raw.githubusercontent.com/emonsharkar/AI-Health-Assistant/main/public_health_surveillance_dataset[1].csv"

# Load the dataset directly from GitHub
data = pd.read_csv(data_url)

# Handle missing values and infinite values
if data.isnull().sum().any():
    st.warning("Dataset contains missing values! Handling them now.")
    
    # Identify numeric columns and fill missing values with median
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    
    # For categorical columns, fill missing values with mode (most frequent value)
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns
    for column in categorical_columns:
        data[column] = data[column].fillna(data[column].mode()[0])  # Fill with the most frequent value
else:
    st.success("No missing values detected.")

# Replace infinite values with NaN, and then handle them
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.fillna(data.median())  # You can replace with mode or other strategies

# Check the data types
st.write(data.dtypes)

# Data Preprocessing (handling categorical variables)
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Location'] = label_encoder.fit_transform(data['Location'])
data['Ethnicity'] = label_encoder.fit_transform(data['Ethnicity'])
data['SES'] = label_encoder.fit_transform(data['SES'])
data['Medical_History'] = label_encoder.fit_transform(data['Medical_History'])
data['Vaccination_Status'] = label_encoder.fit_transform(data['Vaccination_Status'])
data['Immunity_Level'] = label_encoder.fit_transform(data['Immunity_Level'])
data['Reported_Symptoms'] = label_encoder.fit_transform(data['Reported_Symptoms'])
data['Outbreak_Status'] = label_encoder.fit_transform(data['Outbreak_Status'])
data['Infection_Risk_Level'] = label_encoder.fit_transform(data['Infection_Risk_Level'])
data['Disease_Severity'] = label_encoder.fit_transform(data['Disease_Severity'])
data['Hospitalization_Requirement'] = label_encoder.fit_transform(data['Hospitalization_Requirement'])

# Define features and target
X = data.drop(['Hospitalization_Requirement'], axis=1)
y = data['Hospitalization_Requirement']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# User Input Form for prediction
st.sidebar.header("Enter Your Health Details")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
location = st.sidebar.selectbox("Location", ['Urban', 'Rural'])
ethnicity = st.sidebar.selectbox("Ethnicity", ['Ethnicity1', 'Ethnicity2', 'Ethnicity3'])
ses = st.sidebar.selectbox("Socio-Economic Status (SES)", ['Low', 'Medium', 'High'])
chronic_conditions = st.sidebar.selectbox("Chronic Conditions (0 - No, 1 - Yes)", [0, 1])
vaccination_status = st.sidebar.selectbox("Vaccination Status", ['0 - No', '1 - Yes'])
immunity_level = st.sidebar.selectbox("Immunity Level", ['Low', 'Medium', 'High'])
reported_symptoms = st.sidebar.selectbox("Reported Symptoms", ['None', 'Mild', 'Moderate', 'Severe'])
outbreak_status = st.sidebar.selectbox("Outbreak Status", ['No Outbreak', 'Ongoing Outbreak'])
infection_risk_level = st.sidebar.selectbox("Infection Risk Level", ['Low Risk', 'Medium Risk', 'High Risk'])
disease_severity = st.sidebar.selectbox("Disease Severity", ['Mild', 'Moderate', 'Severe'])

# Convert inputs to the correct format for prediction
input_data = np.array([age, label_encoder.transform([gender])[0], label_encoder.transform([location])[0], 
                       label_encoder.transform([ethnicity])[0], label_encoder.transform([ses])[0], chronic_conditions,
                       label_encoder.transform([vaccination_status])[0], label_encoder.transform([immunity_level])[0],
                       label_encoder.transform([reported_symptoms])[0], label_encoder.transform([outbreak_status])[0],
                       label_encoder.transform([infection_risk_level])[0], label_encoder.transform([disease_severity])[0]])
input_data = input_data.reshape(1, -1)

# Predict hospitalization requirement
prediction = model.predict(input_data)

# Display result
if prediction == 0:
    st.subheader("You do not require hospitalization.")
else:
    st.subheader("You may require hospitalization.")

# Display overall health risk
st.sidebar.header("Health Risk Level")
health_risk = model.predict_proba(input_data)
st.sidebar.write(f"Probability of requiring hospitalization: {health_risk[0][1]*100:.2f}%")
