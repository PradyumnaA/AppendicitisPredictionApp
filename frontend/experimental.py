import streamlit as st
import pandas as pd
import fickling

# Load the Random Forest model from the pickle file
with open('random_forest_model.pkl', 'rb') as file:
    rf_model = fickling.load(file)

# Streamlit app header
st.title('Appendicitis Prediction using Random Forest')

# Input features from the user (Example)
st.sidebar.header('User Input Features')

# Example of input fields for user (customize as per your dataset)
age = st.sidebar.slider('Age', 10, 80, 30)  # Slider for age (range from 10 to 80)
pain_intensity = st.sidebar.slider('Pain Intensity', 1, 10, 5)  # Slider for pain intensity
fever = st.sidebar.selectbox('Do you have fever?', ['Yes', 'No'])  # Dropdown for fever

# Additional inputs if your dataset has more features
# For example:
nausea = st.sidebar.selectbox('Do you feel nausea?', ['Yes', 'No'])
appetite_loss = st.sidebar.selectbox('Do you have loss of appetite?', ['Yes', 'No'])

# Convert the inputs into the correct format for prediction
user_input = pd.DataFrame({
    'Age': [age],
    'Pain_Intensity': [pain_intensity],
    'Fever': [1 if fever == 'Yes' else 0],  # Convert 'Yes'/'No' to 1/0
    'Nausea': [1 if nausea == 'Yes' else 0],
    'Appetite_Loss': [1 if appetite_loss == 'Yes' else 0]
})

# Encode categorical variables (if necessary, like your model used pd.get_dummies)
user_input_encoded = pd.get_dummies(user_input, drop_first=True)

# Ensure that the columns match the model's expected input columns
# You can use `user_input_encoded.columns` to manually align them, or use the training data's feature columns.
# Assuming the model was trained on the same set of features:
input_columns = user_input_encoded.columns  # Features from the user input

# Ensure the input data has the same columns as the model's training data (if there are any missing columns, add them)
# This ensures the prediction matches the model's expected features.
for col in rf_model.feature_importances_:
    if col not in input_columns:
        user_input_encoded[col] = 0  # Add missing columns and set them to 0

# Convert column names to strings to avoid issues during prediction
user_input_encoded.columns = user_input_encoded.columns.astype(str)

# Now, make the prediction with the properly aligned user input
prediction = rf_model.predict(user_input_encoded)

# Show the prediction result
if prediction == 1:
    st.write("Prediction: Appendicitis")
else:
    st.write("Prediction: No Appendicitis")
