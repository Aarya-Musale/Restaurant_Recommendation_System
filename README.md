# 🍽️ Zomato Restaurant Rating Predictor

An end-to-end machine learning web application designed to predict a restaurant's aggregate rating in real-time, providing instant performance insights and scoring using a trained Random Forest model and Streamlit.

## 🚀 Project Overview / Description

* **What the project does:** The Zomato Restaurant Rating Predictor is an end-to-end web application built with Streamlit that evaluates restaurant features in real-time. It takes operational variables—such as average cost for two, price range, total votes, service flags (table booking and online delivery), and multi-hot encoded cuisines—processes them through an optimized machine learning pipeline, and instantly outputs a predicted aggregate rating score.
* **The problem it solves:** Restaurant owners, food delivery platforms, and new business investors often struggle to forecast how market positioning, service offerings, and pricing will impact customer ratings. This project addresses this challenge by automating the restaurant success prediction pipeline using predictive analytics, replacing guesswork with data-driven consistency via an interactive web interface.
* **Primary use case:** Designed for restaurant entrepreneurs, food-tech analysts, or educational environments, the primary use case is to serve as a decision-support tool or portal widget to simulate how adjustments to pricing, votes, and cuisine offerings affect a restaurant's overall rating outcome.

## ✨ Key Features

* **Interactive Web Interface:** A clean, responsive user interface built with Streamlit featuring sliders, dropdown selectors, and multi-select components for seamless input manipulation.
* **Real-Time Predictive Analytics:** Instant aggregate rating predictions (ranging from 0 to 5.0) powered by a trained Scikit-Learn Random Forest Regressor (`zomato_advanced_model.pkl`).
* **Advanced Cuisine Feature Engineering:** Dynamic multi-label binarization utilizing a pre-fitted `MultiLabelBinarizer` object (`cuisine_mlb.pkl`) to accurately handle and vectorize complex multi-cuisine restaurant inputs.
* **Dynamic Feedback & Scoring:** Visual result indicators and balloons that display exact score metrics alongside contextual performance tiers (e.g., Top-tier vs. Needs Improvement).

## 📸 Application Preview

<img width="1600" height="900" alt="res_ss" src="https://github.com/user-attachments/assets/c3943f41-c726-4b49-a7b5-bf8550644163" />


*(Note: Ensure your screenshot image file is placed inside an `assets/` folder in your project directory, or update the path below to match where your image is saved).*

```

```

## 🛠 Tech Stack & Dependencies

* **Programming Language:** Python
* **Web Framework:** Streamlit (for serving the interactive user interface and local server management)
* **Machine Learning Model:** Scikit-Learn (Random Forest Regressor and MultiLabelBinarizer)
* **Model Persistence & Handling:** Joblib, Pandas, and NumPy
* **Version Control:** Git & GitHub

## 📂 Project Structure

```text
├── assets/
│   └── screenshot_form.png        # Application output screenshot
├── app.py                         # Main Streamlit web application script
├── generate_pkl.py                # Script used for data processing, model training, and pkl export
├── zomato_advanced_model.pkl      # Pre-trained Random Forest machine learning model
├── cuisine_mlb.pkl                # Fitted MultiLabelBinarizer for cuisine vectorization
├── requirements.txt               # List of required Python packages and dependencies
├── .gitignore                     # Files and directories ignored by Git
└── README.md                      # Comprehensive project documentation

```

## 📥 Installation & Setup Guide

* **Step 1: Clone the repository**
Clone the project repository to your local machine using your terminal:
```bash
git clone <your-repository-url>
cd zomato-restaurant-predictor

```


* **Step 2: Set up a virtual environment**
Create and activate a Python virtual environment to manage dependencies locally:
* *On macOS and Linux:*
```bash
python3 -m venv venv
source venv/bin/activate

```


* *On Windows:*
```bash
python -m venv venv
venv\Scripts\activate

```




* **Step 3: Install dependencies**
Install required packages using pip:
```bash
pip install -r requirements.txt

```



## ▶️ How to Run / Usage

* **Step 1: Navigate to the project directory**
Ensure you are inside the main folder containing `app.py` and your `.pkl` model artifacts.
* **Step 2: Launch the Streamlit application**
Run the following command in your terminal:
```bash
streamlit run app.py

```


* **Step 3: Access the application in your browser**
Streamlit will automatically open a local web server window in your default browser (typically at `http://localhost:8501`).
* **Step 4: Test the application**
Input restaurant details (such as average cost, price tier, vote volume, table booking/online delivery options, and selected cuisines) into the web form and click **Predict Rating** to view real-time model outputs.

## 📊 Model & Data Details

* **Dataset Overview:** The underlying model is trained on global Zomato restaurant data (`zomato.csv`) containing location metrics, operational features, customer votes, and aggregate review ratings.
* **Features Collected & Used:**
* *Average Cost for two:* Estimated dining expense in local currency.
* *Price Range:* Categorical tier ranging from budget (1) to fine dining (4).
* *Votes:* Total user reviews and feedback volume received.
* *Has Table Booking & Online Delivery:* Binary service indicators.
* *Cuisines:* Multi-hot encoded categorical variables generated via `MultiLabelBinarizer` spanning over 140 unique cuisine profiles.


* **Machine Learning & Prediction Logic:**
* The backend utilizes a robust **Random Forest Regressor** (`zomato_advanced_model.pkl`) capable of handling non-linear relationships between customer engagement, pricing, and ratings.
* User selections are mapped directly against fitted binarized matrix structures to guarantee precise multi-feature inference alignment.



## ⚖️ License

This project is developed for educational and professional portfolio purposes.

## 👤 Author / Acknowledgments

Made with ❤️ as part of Machine Learning Application Development.
