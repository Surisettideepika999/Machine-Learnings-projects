import streamlit as st
import pickle
import numpy as np



st.markdown(
    """
    <style>
    body {
        background-color: #f8b400;
        color: #333;
        font-family: 'Helvetica', sans-serif;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .card h3 {
        color: #333;
        font-size: 1.2rem;
    }
    .card .stSlider {
        margin-top: 1rem;
    }
    .prediction-card {
        text-align: center;
        background-color: #f8b400;
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .prediction-card h3 {
        font-size: 1.5rem;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the crop classes
crop_classes=['Apple','Banana','Blackgram','ChickPea', 'Coconut', 'Coffee','Cotton','Grapes','Jute','KidneyBeans','Lentil','Maize','Mango','MothBeans','MungBean','Muskmelon','Orange','Papaya', 'PigeonPeas','Pomegranate','Rice','Watermelon']

# Create the Streamlit app

st.title('🌾 Crop Prediction App')

# Collect user input
st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
nitrogen = st.slider('Nitrogen', min_value=0.0, max_value=150.0, value=50.0)
phosphorus = st.slider('Phosphorus', min_value=0.0, max_value=150.0, value=50.0)
potassium = st.slider('Potassium', min_value=0.0, max_value=200.0, value=50.0)
temperature = st.slider('Temperature', min_value=0.0, max_value=50.0, value=25.0)
humidity = st.slider('Humidity', min_value=0.0, max_value=100.0, value=50.0)
ph_value = st.slider('pH Value', min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.slider('Rainfall', min_value=0.0, max_value=300.0, value=100.0)
st.markdown('</div>', unsafe_allow_html=True)

# Make prediction
features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
prediction = model.predict(features)
predicted_crop = crop_classes[prediction[0]]

st.markdown('<div class="card prediction-card">', unsafe_allow_html=True)
st.write(f'<h3>The predicted crop is: <strong>{predicted_crop}</strong></h3>', unsafe_allow_html=True)

# Display prediction
st.markdown('</div>', unsafe_allow_html=True)
