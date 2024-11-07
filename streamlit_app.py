import os
   os.system('pip install scikit-learn==1.3.0')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Define symptoms and disease mapping
symptoms = [
    'fever',
    'cough',
    'headache',
    'sore_throat',
    'runny_nose',
    'fatigue',
    'nausea',
    'vomiting',
    'diarrhea',
    'muscle_pain',
    'shortness_of_breath',
    'chest_pain',
    'dizziness',
    'rash',
    'abdominal_pain'
]

disease_symptom_mapping = {
    'Common Cold': ['cough', 'sore_throat', 'runny_nose', 'fever', 'headache'],
    'Flu': ['fever', 'cough', 'muscle_pain', 'fatigue', 'headache'],
    'Migraine': ['headache', 'nausea', 'vomiting', 'dizziness'],
    'Food Poisoning': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain'],
    'Pneumonia': ['fever', 'cough', 'shortness_of_breath', 'chest_pain'],
    # ... Add more diseases and their symptoms
}

# Create a DataFrame for training
data = []
for disease, symptoms_list in disease_symptom_mapping.items():
    data.append([disease, symptoms_list])

df = pd.DataFrame(data, columns=['disease', 'symptoms'])

# MultiLabelBinarizer for symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['symptoms'])
y = df['disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
st.title("Disease Detection App")

# Multiselect widget for symptoms
selected_symptoms = st.multiselect("Select Symptoms", symptoms)

if selected_symptoms:
    # Predict probabilities
    input_vector = mlb.transform([selected_symptoms])
    predicted_probabilities = rf_classifier.predict_proba(input_vector)[0]

    # Get disease names and probabilities
    disease_names = rf_classifier.classes_
    disease_probs = dict(zip(disease_names, predicted_probabilities))

    # Display results
    st.subheader("Disease Probabilities:")
    for disease, prob in disease_probs.items():
        st.write(f"{disease}: {prob * 100:.2f}%")  # Display as percentage

else:
    st.info("Please select at least one symptom.")
