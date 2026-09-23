# app.py
import streamlit as st
import numpy as np
import joblib
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Zomato Restaurant Rating Predictor",
    page_icon="🍽️",
    layout="centered"
)

# 2. Load the Saved Model and MultiLabelBinarizer safely
@st.cache_resource
def load_artifacts():
    if os.path.exists('zomato_advanced_model.pkl') and os.path.exists('cuisine_mlb.pkl'):
        model = joblib.load('zomato_advanced_model.pkl')
        mlb = joblib.load('cuisine_mlb.pkl')
        return model, mlb
    return None, None

model, mlb = load_artifacts()

# App Header
st.title("🍽️ Zomato Restaurant Rating Predictor")
st.write("Predict a restaurant's **Aggregate Rating** using Machine Learning based on cost, votes, services, and cuisines.")
st.divider()

if model is None or mlb is None:
    st.error("⚠️ Model files (`zomato_advanced_model.pkl` or `cuisine_mlb.pkl`) not found in this folder! Please run your `generate_pkl.py` script first.")
else:
    # 3. Input Layout (Form UI)
    st.subheader("📝 Enter Restaurant Details:")

    col1, col2 = st.columns(2)

    with col1:
        avg_cost = st.number_input("Average Cost for Two", min_value=0, max_value=50000, value=500, step=50, help="Estimated cost for two people.")
        price_range = st.selectbox(
            "Price Range Category", 
            options=[1, 2, 3, 4], 
            format_func=lambda x: "Budget (1)" if x==1 else ("Moderate (2)" if x==2 else ("Expensive (3)" if x==3 else "Fine Dining (4)"))
        )
        table_booking = st.selectbox("Has Table Booking?", options=["No", "Yes"])

    with col2:
        votes = st.number_input("Number of Votes", min_value=0, max_value=20000, value=100, step=10, help="Total user reviews/votes received.")
        online_delivery = st.selectbox("Has Online Delivery?", options=["No", "Yes"])
        
    # Cuisine multi-select dropdown using the fitted MultiLabelBinarizer classes
    selected_cuisines = st.multiselect(
        "Select Cuisines Served", 
        options=list(mlb.classes_), 
        default=["North Indian"] if "North Indian" in mlb.classes_ else [mlb.classes_[0]],
        help="Select one or more cuisines offered by the restaurant."
    )

    # Convert binary dropdowns to numeric
    tb_val = 1 if table_booking == "Yes" else 0
    od_val = 1 if online_delivery == "Yes" else 0

    st.divider()

    # 4. Prediction Logic
    if st.button("Predict Rating 🚀", type="primary", use_container_width=True):
        if not selected_cuisines:
            st.warning("⚠️ Please select at least one cuisine.")
        else:
            # Transform user's selected cuisines into the exact model column format
            user_cuisine_vector = mlb.transform([selected_cuisines])
            
            # Pack numerical inputs
            numeric_inputs = np.array([[avg_cost, price_range, votes, tb_val, od_val]])
            
            # Combine numeric features + cuisine vector
            final_input = np.hstack((numeric_inputs, user_cuisine_vector))
            
            # Make prediction
            prediction = model.predict(final_input)[0]
            
            # Display Results container
            st.success(f"### Predicted Aggregate Rating: {prediction:.2f} / 5.0")
            
            # Dynamic feedback
            if prediction >= 4.0:
                st.balloons()
                st.info("🌟 **Excellent / Top-tier Restaurant!** High votes and optimal pricing point to great popularity.")
            elif prediction >= 3.0:
                st.info("👍 **Good / Average Restaurant.** Decent performance with room for customer experience enhancement.")
            else:
                st.warning("⚠️ **Needs Improvement.** Historical patterns for similar configurations show lower customer satisfaction.")