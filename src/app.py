import numpy as np
import pandas as pd
from model import Model
import streamlit as st


# Create an instance of the Model class.
model = Model()

# Title of the app.
st.title("Student Performance Prediction")

# Define a function to collect user input.
def get_student_data():
    age = st.number_input("Age", min_value=15, max_value=22, value=17)
    gender = st.selectbox("Gender", ["Male", "Female"])
    department = st.selectbox("Department", ["Science", "Arts", "Business"])
    home_language = st.selectbox("Home Language", ["English", "Other"])
    english_fluency = st.selectbox("English Fluency", ["Fluent", "Not Fluent", "Very Fluent"])
    parents_income = st.selectbox("Parents' Income", ["Low Income", "Middle Income", "High Income"])
    parents_education = st.selectbox("Parents' Education Level", ["No Formal Education", "Primary Education", "Secondary Education", "Tertiary Education"])
    disability = st.selectbox("Disability", ["Yes", "No"])
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    private_lessons = st.selectbox("Private Lessons", ["Yes", "No"])
    extracurricular_activity = st.selectbox("Extracurricular Activity", ["Yes", "No"])
    resources_access = st.selectbox("Resources Access", ["No access", "Limited access", "Full access", "Adequate access"])
    disciplinary_actions = st.selectbox("Disciplinary Actions", ["Yes", "No"])
    class_participation = st.selectbox("Class Participation", ["Rarely", "Frequently"])
    attendance_bracket = st.selectbox("Attendance Bracket", ["51-75%", "76-90%", "91-100%"])
    
    return np.array([[age, gender, department, home_language, english_fluency,
                      parents_income, parents_education, disability, internet_access,
                      private_lessons, extracurricular_activity, resources_access,
                      disciplinary_actions, class_participation, attendance_bracket]])

# Collect student data.
input_data = get_student_data()

# Button to make predictions.
if st.button("Make Prediction"):
    predictions = model.prediction(input_data)
    probability = model.prediction_probability(input_data)

    if predictions is not None:
        st.subheader("Prediction Results")
        
        # Get the pass and fail probability.
        fail_probs = probability[0][0]
        pass_probs = probability[0][1] 
         
        if predictions[0] == 1:
            st.write("This student is likely to pass.")
            
            # Create a DataFrame for the bar chart.
            prob_df = pd.DataFrame({
                'Class': ['Fail', 'Pass'],
                'Probability': [1 - pass_probs, pass_probs]  
            })

            # Display bar chart for prediction probabilities.
            st.bar_chart(prob_df.set_index('Class'))
            
        elif predictions[0] == 0:
            st.write("This student is likely to fail.")
        
            
            # Create a DataFrame for the bar chart.
            prob_df = pd.DataFrame({
                'Class': ['Fail', 'Pass'],
                'Probability': [fail_probs, 1 - fail_probs] 
            })

            # Display bar chart for prediction probabilities.
            st.bar_chart(prob_df.set_index('Class'))
        
    # else:
    #     st.error("Prediction failed.") 

