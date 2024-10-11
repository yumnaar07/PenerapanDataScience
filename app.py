import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Memuat model
model = load_model('model.h5')

# Judul aplikasi
st.title("Prediksi Status Siswa")

# Input dari pengguna
st.header("Input Data")

# Menggunakan st.number_input untuk input numerik
curricular_units_2nd_sem_grade = st.number_input("curricular_units_2nd_sem_grade")
curricular_units_2nd_sem_approved = st.number_input("curricular_units_2nd_sem_approved")
curricular_units_1st_sem_grade = st.number_input("curricular_units_1st_sem_grade")
curricular_units_1st_sem_approved = st.number_input("curricular_units_1st_sem_approved")
tuition_fees_up_to_date = st.number_input("tuition_fees_up_to_date")
course = st.number_input("course") 
scholarship_holder = st.number_input("scholarship_holder")
application_mode = st.number_input("application_mode ")  
mothers_occupation = st.number_input("mothers_occupation") 
age_at_enrollment = st.number_input("age_at_enrollment")

# Menyimpan input ke dalam DataFrame
input_data = pd.DataFrame({
    'Curricular_units_2nd_sem_grade': [curricular_units_2nd_sem_grade],
    'Curricular_units_2nd_sem_approved': [curricular_units_2nd_sem_approved],
    'Curricular_units_1st_sem_grade': [curricular_units_1st_sem_grade],
    'Curricular_units_1st_sem_approved': [curricular_units_1st_sem_approved],
    'Tuition_fees_up_to_date': [tuition_fees_up_to_date],
    'Course': [course],
    'Scholarship_holder': [scholarship_holder],
    'Application_mode': [application_mode],
    'Mothers_occupation': [mothers_occupation],
    'Age_at_enrollment': [age_at_enrollment]
})

# Melakukan prediksi
if st.button("Prediksi"):
    # Normalisasi input
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Melakukan prediksi
    prediction = model.predict(input_scaled)
    predicted_status = np.argmax(prediction, axis=1)

    # Menampilkan hasil prediksi
    status_map = {0: "Dropout", 1: "Graduate", 2: "Enrolled"}
    st.success(f"Status Prediksi: {status_map[predicted_status[0]]}")