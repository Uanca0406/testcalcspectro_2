import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Function for calculating the regression line
def regression_analysis(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(x.reshape(-1, 1), y)
    return slope, intercept, r_squared

# Function for calculating RPD
def calculate_rpd(actual, predicted):
    return np.abs(actual - predicted) / actual * 100

# Function to calculate accuracy
def calculate_accuracy(actual, predicted):
    return np.abs((actual - predicted) / actual) * 100

# Webpage title
st.title("Spektrofotometri Calculator")

# Input for sample data
st.header("Masukkan Data Sampel dan Absorbansi")

# Upload sample data
sample_data = st.file_uploader("Upload file CSV dengan kolom 'gram' dan 'absorbansi' (tanpa header)", type="csv")

if sample_data is not None:
    # Read CSV file
    df = pd.read_csv(sample_data, header=None, names=['gram', 'absorbansi'])

    # Display the sample data
    st.write("Data Sampel:")
    st.dataframe(df)

    # Input untuk persamaan kurva kalibrasi (y = mx + b)
    st.header("Masukkan Persamaan Kurva Kalibrasi")

    intercept_input = st.number_input("Intercept (b)", value=0.0)
    slope_input = st.number_input("Slope (m)", value=1.0)

    # Calculate concentration based on absorbance
    df['konsentrasi'] = (df['absorbansi'] - intercept_input) / slope_input
    st.write("Konsentrasi Sampel:")
    st.dataframe(df)

    # Calculate regression parameters using the data
    x = np.array(df['gram']).reshape(-1, 1)
    y = np.array(df['absorbansi'])
    
    slope, intercept, r_squared = regression_analysis(x, y)

    # Display regression results
    st.subheader("Hasil Regresi Linear")
    st.write(f"Slope: {slope:.4f}")
    st.write(f"Intercept: {intercept:.4f}")
    st.write(f"R² (Koefisien Determinasi): {r_squared:.4f}")

    # Calculate RPD for each sample
    predicted_absorbance = slope * df['gram'] + intercept
    df['RPD'] = calculate_rpd(df['absorbansi'], predicted_absorbance)
    st.write("RPD Sampel:")
    st.dataframe(df)

    # Calculate accuracy
    df['Akurasi'] = calculate_accuracy(df['absorbansi'], predicted_absorbance)
    st.write("Akurasi Sampel:")
    st.dataframe(df)

    # Plot the calibration curve and regression line
    fig, ax = plt.subplots()
    ax.scatter(df['gram'], df['absorbansi'], color='blue', label='Data Sampel')
    ax.plot(df['gram'], predicted_absorbance, color='red', label=f'Regresi: y = {slope:.4f}x + {intercept:.4f}')
    ax.set_xlabel('Gram Sampel')
    ax.set_ylabel('Absorbansi')
    ax.legend()

    st.pyplot(fig)

else:
    st.warning("Harap unggah file CSV untuk melanjutkan perhitungan.")

