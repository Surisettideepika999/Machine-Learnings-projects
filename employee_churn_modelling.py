# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(page_title="Employee Retention Prediction", layout="wide")

# Function to load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('emp_churn_data.csv')
    return data

# Function to preprocess input data
# def preprocess_input(data, feature_columns):
#     data = pd.get_dummies(data)
#     missing_cols = set(feature_columns) - set(data.columns)
#     for col in missing_cols:
#         data[col] = 0
#     return data[feature_columns]

# Load dataset
data = load_data()

# Prepare features and target
X = data.drop('left', axis=1)  # assuming 'left' is the target column indicating if the employee left
y = data['left']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the model
model = XGBClassifier(objective='binary:logistic',n_estimators=200,max_depth=10,learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI

# Title and description
st.title('Employee Retention Prediction')
st.markdown("""
Welcome to the Employee Retention Prediction tool. This app helps you predict whether an employee is likely to leave the company based on various features.
""")

# Sidebar for input features
st.sidebar.header('Enter Employee Details')
satisfaction_level = st.sidebar.slider('Satisfaction Level', min_value=0.0, max_value=1.0, step=0.01)
last_evaluation = st.sidebar.slider('Last Evaluation', min_value=0.0, max_value=1.0, step=0.01)
number_project = st.sidebar.number_input('Number of Projects', min_value=1, max_value=10, step=1)
average_monthly_hours = st.sidebar.number_input('Average Monthly Hours', min_value=0, max_value=500, step=1)
time_spend_company = st.sidebar.number_input('Time Spent at Company (Years)', min_value=0, max_value=10, step=1)
work_accident = st.sidebar.selectbox('Work Accident', [0, 1])
promotion_last_5years = st.sidebar.selectbox('Promotion in Last 5 Years', [0, 1])
department = st.sidebar.selectbox('Department', ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'])
salary = st.sidebar.selectbox('Salary Level', [0, 1, 2])

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'satisfaction_level': [satisfaction_level],
    'last_evaluation': [last_evaluation],
    'number_project': [number_project],
    'average_montly_hours': [average_monthly_hours],
    'time_spend_company': [time_spend_company],
    'Work_accident': [work_accident],
    'promotion_last_5years': [promotion_last_5years],
    'salary': [salary]
})

# Main section
st.header("Model Training and Evaluation")
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Button to predict
if st.button('Predict Employee Retention'):
    # input_data_processed = preprocess_input(input_data, X.columns)
    prediction = model.predict(input_data)
    if prediction == 1:
        st.markdown("<h2 style='color: red;'>The employee is likely to leave the company.</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>The employee is likely to stay with the company.</h2>", unsafe_allow_html=True)

# Display the entered data
st.header("Entered Employee Data")
st.write(input_data)

# Add some style
st.markdown("""
<style>
    .css-18e3th9 {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
    }
</style>
""", unsafe_allow_html=True)
