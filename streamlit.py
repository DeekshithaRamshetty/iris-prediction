import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('svm_model.pkl')

# Species mapping
species_map = {0: '🌼 Iris-setosa', 1: '🌺 Iris-versicolor', 2: '🌸 Iris-virginica'}

# Set page config
st.set_page_config(page_title="Iris Flower Classifier 🌷", page_icon="🌿", layout="centered")

# Header section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌷 Iris Flower Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Adjust the sliders below to enter the flower's features and see which Iris species it belongs to.</p>", unsafe_allow_html=True)

# Input sliders in two columns
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider('Sepal Length (cm)', 4.3, 7.9, 5.0, step=0.1)
    petal_length = st.slider('Petal Length (cm)', 1.0, 6.9, 3.0, step=0.1)

with col2:
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.4, 3.0, step=0.1)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0, step=0.1)

# Predict button
if st.button("🌿 Predict Flower Type"):
    # Create input dataframe
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    # Make prediction
    prediction = model.predict(input_data)[0]
    predicted_species = species_map[prediction]

    # Display result
    st.success(f"🔍 **The predicted species is:** {predicted_species}")

    st.balloons()  # Fun animation on prediction

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
