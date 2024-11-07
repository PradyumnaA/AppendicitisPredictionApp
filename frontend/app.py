import pandas as pd
import streamlit as st
import pickle

# Load the model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Print feature names expected by the model (for debugging)
print("Features expected by the model:", model.feature_names_in_)

# Define input fields (Streamlit UI)
age = st.number_input('Age', min_value=0, max_value=120, value=30)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=18.5)
sex = st.selectbox('Sex', ['male', 'female'])
height = st.number_input('Height (cm)', min_value=50.0, max_value=250.0, value=160.0)
weight = st.number_input('Weight (kg)', min_value=10.0, max_value=200.0, value=60.0)
length_of_stay = st.number_input('Length of Stay (days)', min_value=1, max_value=30, value=5)
alvarado_score = st.number_input('Alvarado Score', min_value=0, max_value=10, value=5)
paedriatic_appendicitis_score = st.number_input('Paediatric Appendicitis Score', min_value=0, max_value=10, value=5)
appendix_on_us = st.selectbox('Appendix on US', ['yes', 'no'])
appendix_diameter = st.number_input('Appendix Diameter (mm)', min_value=0.0, max_value=20.0, value=7.0)

# Create a DataFrame with the input values
data = pd.DataFrame({
    'Age': [age],
    'BMI': [bmi],
    'Sex': [sex],
    'Height': [height],
    'Weight': [weight],
    'Length_of_Stay': [length_of_stay],
    'Alvarado_Score': [alvarado_score],
    'Paedriatic_Appendicitis_Score': [paedriatic_appendicitis_score],
    'Appendix_on_US': [appendix_on_us],
    'Appendix_Diameter': [appendix_diameter]
})

# Ensure all features the model was trained on are present
expected_columns = model.feature_names_in_

# Add missing columns to the input data with default values
for col in expected_columns:
    if col not in data.columns:
        data[col] = 0  # Set default values (can adjust based on column type)

# Ensure categorical features are encoded the same way as during training
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # Example for encoding
data['Appendix_on_US'] = data['Appendix_on_US'].map({'yes': 1, 'no': 0})  # Example encoding

# Ensure all categorical variables are processed similarly to training
# Add more processing steps for other categorical features if needed

# Reorder columns to match the model's expected order
data = data[expected_columns]

# Make the prediction
prediction = model.predict(data)

# Display the result
if prediction == 1:
    st.write("Prediction: Appendicitis (Positive)")
else:
    st.write("Prediction: No Appendicitis (Negative)")
