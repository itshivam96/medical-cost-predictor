import streamlit as st
import joblib
import numpy as np

# Load the trained model
model_rf = joblib.load('random_forest_model.pkl')

st.title('Medical Insurance Cost Predictor')
st.write('Enter your information to predict expected insurance charges:')

# User input widgets
age = st.slider('Age', 18, 65)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.slider('BMI', 15.0, 45.0)
children = st.slider('Number of Children', 0, 5)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['northwest', 'northeast', 'southeast', 'southwest'])

# Encode categorical values:
sex_num = 1 if sex == 'male' else 0
smoker_num = 1 if smoker == 'yes' else 0
region_northwest = 1 if region == 'northwest' else 0
region_southeast = 1 if region == 'southeast' else 0
region_southwest = 1 if region == 'southwest' else 0

user_data = np.array([[age, sex_num, bmi, children, smoker_num, region_northwest, region_southeast, region_southwest]])

if st.button('Predict Insurance Charges'):
    prediction = model_rf.predict(user_data)[0]
    st.success(f'Estimated Insurance Charges: {prediction:.2f}')
