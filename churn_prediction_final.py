from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set Streamlit layout to wide
st.set_page_config(layout="wide")

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the MinMaxScaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the input features for the model
feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male", "HasCrCard_0", "HasCrCard_1",
    "IsActiveMember_0", "IsActiveMember_1"
]

# Columns requiring scaling
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

# Updated default values
default_values = [
    600, 30, 2, 8000, 2, 60000,
    True, False, False, True, False, False, True, False, True
]

# Sidebar setup
try:
    st.sidebar.image("Pic 1.png", use_container_width=True)
except FileNotFoundError:
    st.sidebar.write("🧠 *Customer Churn Prediction*")

st.sidebar.header("User Inputs")

# Collect user inputs
user_inputs = {}
for i, feature in enumerate(feature_names):
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=default_values[i], step=1 if isinstance(default_values[i], int) else 0.01
        )
    elif isinstance(default_values[i], bool):
        user_inputs[feature] = st.sidebar.checkbox(feature, value=default_values[i])
    else:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=default_values[i], step=1
        )

# Convert inputs to a DataFrame
input_data = pd.DataFrame([user_inputs])

# Apply MinMaxScaler to the required columns
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

# App Header
try:
    st.image("Pic 2.png", use_container_width=True)
except FileNotFoundError:
    st.title("💼 Customer Churn Prediction")

# Page Layout
left_col, right_col = st.columns(2)

# Left Page: Feature Importance
with left_col:
    st.header("🔍 Feature Importance")
    try:
        feature_importance_df = pd.read_excel("feature_importance.xlsx", usecols=["Feature", "Feature Importance Score"])
        fig = px.bar(
            feature_importance_df.sort_values(by="Feature Importance Score", ascending=False),
            x="Feature Importance Score",
            y="Feature",
            orientation="h",
            title="Top Contributing Features",
            labels={"Feature Importance Score": "Importance", "Feature": "Features"},
            width=400,
            height=500
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error loading feature importance chart: {e}")

# Right Page: Prediction and Dynamic Visualization
with right_col:
    st.header("📊 Prediction")
    if st.button("Predict"):
        try:
            probabilities = model.predict_proba(input_data_scaled)[0]
            prediction = model.predict(input_data_scaled)[0]
            prediction_label = "Churned" if prediction == 1 else "Retain"

            st.subheader(f"Predicted Value: {prediction_label}")
            st.write(f"Predicted Probability: {probabilities[1]:.2%} (Churn)")
            st.write(f"Predicted Probability: {probabilities[0]:.2%} (Retain)")
            st.markdown(f"### 🔔 Final Output: **{prediction_label}**")

            # Add Donut Chart
            chart_data = {
                "labels": ["Retain", "Churn"],
                "values": [probabilities[0], probabilities[1]],
                "colors": ["green", "red"]
            }

            fig_donut = go.Figure(data=[go.Pie(
                labels=chart_data["labels"],
                values=chart_data["values"],
                hole=0.5,
                marker=dict(colors=chart_data["colors"]),
                sort=False
            )])
            fig_donut.update_layout(
                title_text="Customer Churn vs Retain Probability",
                annotations=[dict(text='Churn', x=0.5, y=0.5, font_size=20, showarrow=False)],
                transition=dict(duration=1000, easing='cubic-in-out')
            )
            fig_donut.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_donut, use_container_width=True)

            # Blinking warning if high risk
            if probabilities[1] > 0.6:
                st.markdown(
                    """
                    <div style="animation: blink 1s infinite; color: red; font-weight: bold; font-size: 24px; text-align: center;">
                        🚨 High Risk of Churn Detected! 🚨
                    </div>
                    <style>
                    @keyframes blink {
                        0% { opacity: 1; }
                        50% { opacity: 0.2; }
                        100% { opacity: 1; }
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
