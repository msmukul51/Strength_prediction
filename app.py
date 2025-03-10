import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load the saved model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set page layout
st.set_page_config(layout="wide")

# Title of the app
st.title("🔩 Steel Fatigue Strength Prediction App 💪")

# === 3 Column Layout ===
col1, col2, col3 = st.columns(3)

# Define input fields in 3 columns
def user_input_features():
    with col1:
        NT = st.number_input("Normalizing Temperature (°C)", 825, 930, 825)
        THT = st.number_input("Through Hardening Temperature (°C)", 825, 865, 825)
        THt = st.number_input("Through Hardening Time (sec)", 0, 30, 30)
        THQCr = st.number_input("Through Hardening Quenching Cooling Rate (°C/sec)", 0.0, 24.0, 0.0)
        CT = st.number_input("Cooling Temperature (°C)", 930.0, 930.0, 930.0)
        Ct = st.number_input("Cooling Time (sec)", 0.0, 540.0, 0.0)

    with col2:
        DT = st.number_input("Diffusion Temperature (°C)", 830.0, 903.333, 830.0)
        Dt = st.number_input("Diffusion Time (sec)", 15.0, 70.2, 15.0)
        QmT = st.number_input("Quenching Media Temperature (°C)", 30.0, 140.0, 30.0)
        TT = st.number_input("Tempering Temperature (°C)", 160.0, 680.0, 160.0)
        Tt = st.number_input("Tempering Time (sec)", 60.0, 120.0, 60.0)
        TCr = st.number_input("Tempering Cooling Rate (°C/sec)", 0.5, 24.0, 0.5)

    with col3:
        C = st.number_input("Weight % of Carbon", 0.17, 0.63, 0.17)
        Si = st.number_input("Weight % of Silicon", 0.16, 2.05, 0.16)
        Mn = st.number_input("Weight % of Manganese", 0.37, 1.6, 0.37)
        P = st.number_input("Weight % of Phosphorus", 0.002, 0.031, 0.002)
        S = st.number_input("Weight % of Sulphur", 0.003, 0.03, 0.003)
        Ni = st.number_input("Weight % of Nickel", 0.01, 2.78, 0.01)
        Cr = st.number_input("Weight % of Chromium", 0.01, 1.17, 0.01)
        Cu = st.number_input("Weight % of Copper", 0.01, 0.26, 0.01)
        Mo = st.number_input("Weight % of Molybdenum", 0.0, 0.24, 0.0)
        RedRatio = st.number_input("Reduction Ratio", 240.0, 5530.0, 240.0)
        dA = st.number_input("Area Proportion of Inclusions Deformed", 0.0, 0.13, 0.0)
        dB = st.number_input("Area Proportion of Inclusions in Discontinuous Array", 0.0, 0.05, 0.0)
        dC = st.number_input("Area Proportion of Isolated Inclusions", 0.0, 0.058, 0.0)

    return np.array([[NT, THT, THt, THQCr, CT, Ct, DT, Dt, QmT, TT, Tt, TCr, C, Si, Mn, P, S, Ni, Cr, Cu, Mo, RedRatio, dA, dB, dC]])

X_input = user_input_features()

# Function to classify steel type
def classify_steel(C):
    if C <= 0.3:
        return "Low Carbon Steel"
    elif 0.3 < C <= 0.6:
        return "Medium Carbon Steel"
    else:
        return "High Carbon Steel"

# Function to estimate mechanical properties based on fatigue strength
def estimate_properties(fatigue_strength, steel_type):
    if steel_type == "Low Carbon Steel":
        UTS = fatigue_strength / 0.45  
        YS = 0.5 * UTS  
    elif steel_type == "Medium Carbon Steel":
        UTS = fatigue_strength / 0.5  
        YS = 0.6 * UTS  
    else:  # High Carbon Steel
        UTS = fatigue_strength / 0.55  
        YS = 0.7 * UTS  

    HB = UTS / 3.45  
    HV = UTS / 3.0   
    HRB = (0.16 * HB) + 32  
    HRC = (0.1 * HB) - 10   
    Ductility = 100 - (UTS / 10)  
    Toughness = 1 / YS  

    return round(UTS, 2), round(YS, 2), round(HB, 2), round(HV, 2), round(HRB, 2), round(HRC, 2), round(Ductility, 2), round(Toughness, 6)

# Prediction Button
if st.button("🔮 Predict Steel Strength"):
    prediction = pipe.predict(X_input)[0]
    steel_type = classify_steel(X_input[0][12])
    UTS, YS, HB, HV, HRB, HRC, Ductility, Toughness = estimate_properties(prediction, steel_type)

    # Store results in session state
    st.session_state.update({
        "prediction": prediction, "steel_type": steel_type, "UTS": UTS, "YS": YS,
        "HB": HB, "HV": HV, "HRB": HRB, "HRC": HRC, "Ductility": Ductility, "Toughness": Toughness
    })

    # Display Results
    st.markdown(f"""
    <div style="padding:10px; border-radius:10px; background:#e3f2fd;">
        🔥 <b>Predicted Fatigue Strength:</b> {prediction:.2f} MPa<br>
        🏗️ <b>Steel Type:</b> {steel_type}<br>
        📏 <b>Ultimate Tensile Strength (UTS):</b> {UTS} MPa<br>
        🔩 <b>Yield Strength (YS):</b> {YS} MPa<br>
        🔨 <b>Brinell Hardness (HB):</b> {HB}<br>
        🛠️ <b>Vickers Hardness (HV):</b> {HV}<br>
        🏹 <b>Rockwell Hardness (HRB):</b> {HRB}<br>
        🔥 <b>Rockwell Hardness (HRC):</b> {HRC}<br>
        📉 <b>Ductility (Elongation %):</b> {Ductility}%<br>
        🏋️ <b>Toughness (K_IC):</b> {Toughness} MPa√m
    </div>
    """, unsafe_allow_html=True)

# 📥 **Download Steel Recipe (CSV)**
if "prediction" in st.session_state:
    df = pd.DataFrame(X_input, columns=[
        "Normalizing Temperature (NT)", "Through Hardening Temperature (THT)", "Through Hardening Time (THt)", 
        "Through Hardening Quenching Cooling Rate (THQCr)", "Cooling Temperature (CT)", "Cooling Time (Ct)",
        "Diffusion Temperature (DT)", "Diffusion Time (Dt)", "Quenching Media Temperature (QmT)", 
        "Tempering Temperature (TT)", "Tempering Time (Tt)", "Tempering Cooling Rate (TCr)",
        "Carbon (%)", "Silicon (%)", "Manganese (%)", "Phosphorus (%)", "Sulfur (%)", 
        "Nickel (%)", "Chromium (%)", "Copper (%)", "Molybdenum (%)", 
        "Reduction Ratio", "Inclusions Deformed (dA)", "Discontinuous Inclusions (dB)", "Isolated Inclusions (dC)"
    ])
    
    # Add predicted mechanical properties
    df["Predicted Fatigue Strength (MPa)"] = st.session_state["prediction"]
    df["Steel Type"] = st.session_state["steel_type"]
    df["Ultimate Tensile Strength (UTS) MPa"] = st.session_state["UTS"]
    df["Yield Strength (YS) MPa"] = st.session_state["YS"]
    df["Brinell Hardness (HB)"] = st.session_state["HB"]
    df["Vickers Hardness (HV)"] = st.session_state["HV"]
    df["Rockwell Hardness (HRB)"] = st.session_state["HRB"]
    df["Rockwell Hardness (HRC)"] = st.session_state["HRC"]
    df["Ductility (%)"] = st.session_state["Ductility"]
    df["Toughness (MPa√m)"] = st.session_state["Toughness"]

    # Convert DataFrame to CSV and enable download
    csv_file = df.to_csv(index=False).encode()
    st.download_button("📥 Download Steel Report (CSV)", csv_file, "Steel_Properties_Report.csv")

# === What-If Analysis (Sensitivity Analysis) ===
st.header("🔍 What-If Analysis: See How Changes Affect Strength")

# Select Variable to Adjust
sensitivity_param = st.selectbox("Select a parameter to modify:", 
                                 ["Carbon (%)", "Tempering Temperature (°C)", "Quenching Media Temperature (°C)"])

# Define Input Values (Baseline)
X_baseline = np.array([[850, 850, 10, 5, 900, 300, 850, 20, 50, 450, 1800, 10, 0.2, 0.5, 0.6, 0.01, 0.01, 0.1, 0.2, 0.1, 0.01, 5, 0.05, 0.05, 0.05]])

# Define Range for What-If Analysis
if sensitivity_param == "Carbon (%)":
    values = np.linspace(0.1, 1.0, 10)  # Vary Carbon from 0.1% to 1.0%
    param_index = 12  # Carbon column index
elif sensitivity_param == "Tempering Temperature (°C)":
    values = np.linspace(200, 700, 10)  # Vary Tempering Temperature from 200°C to 700°C
    param_index = 10  # Tempering Temperature column index
elif sensitivity_param == "Quenching Media Temperature (°C)":
    values = np.linspace(30, 200, 10)  # Vary Quenching Temperature
    param_index = 8  # Quenching Media Temperature column index

# Run Analysis Only When Button is Clicked
if st.button("Run What-If Analysis"):
    fatigue_strengths = []
    tensile_strengths = []

    for value in values:
        X_modified = X_baseline.copy()
        X_modified[0, param_index] = value  # Ensure explicit update
        fatigue_prediction = pipe.predict(X_modified)[0]
        fatigue_strengths.append(fatigue_prediction)
        tensile_strengths.append(fatigue_prediction / 0.5)  # Approximate UTS estimation

    # Plot Results
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(values, fatigue_strengths, label="Fatigue Strength (MPa)", marker="o")
    ax.plot(values, tensile_strengths, label="Tensile Strength (MPa)", marker="s", linestyle="dashed")
    ax.set_xlabel(sensitivity_param)
    ax.set_ylabel("Strength (MPa)")
    ax.set_title(f"Impact of {sensitivity_param} on Strength")
    ax.legend()
    ax.grid(True)

    # Display Plot
    st.pyplot(fig)


# 📊 **Show Feature Importance (SHAP)**
if st.checkbox("📊 Show Feature Importance (SHAP)"):
    explainer = shap.TreeExplainer(pipe.named_steps["Regressor"])
    shap_values = explainer.shap_values(X_input)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    shap_values_mean = np.abs(shap_values).mean(axis=0)  
    sorted_idx = np.argsort(shap_values_mean)  

    ax.barh(np.array(df.columns)[sorted_idx], shap_values_mean[sorted_idx], color='skyblue')
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# 🌡️ **Show Correlation Heatmap**
if st.checkbox("🌡️ Show Correlation Heatmap"):
    df = pd.read_csv("data.csv")  
    df.drop(columns=["Sl. No."], inplace=True, errors="ignore")  
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(15,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)



