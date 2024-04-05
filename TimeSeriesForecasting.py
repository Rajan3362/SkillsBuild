import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Input

# Global variables to store scaled data and scaler
scaled_data = None
scaler = MinMaxScaler(feature_range=(0, 1))

# Function to load and preprocess data
def load_data():
    global scaled_data, scaler
    filename = filedialog.askopenfilename(title="Select CSV file")
    if filename:
        data = pd.read_csv(filename)
        scaled_data = scaler.fit_transform(data["value"].values.reshape(-1, 1))
        status_label.config(text="Data loaded successfully!", fg="green")
    else:
        status_label.config(text="No file selected", fg="red")

# Function to train and predict with LSTM model
def train_and_predict():
    global scaled_data
    if scaled_data is None:
        status_label.config(text="Please load data first", fg="red")
        return
    n_steps = 10
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i - n_steps:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    predictions = model.predict(X_test)
    display_results(predictions, y_test)

# Function to display results
def display_results(predictions, y_test):
    global scaler
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    plt.figure(figsize=(12, 6))
    plt.plot(predictions, label="Predictions")
    plt.plot(y_test, label="Actual")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Time Series Forecasting")
    plt.show()

# GUI setup
root = tk.Tk()
root.title("Time Series Forecasting")

# Style for the buttons
style = ttk.Style()
style.configure("TButton", foreground="black", background="lightgray")

# Frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

# Load Data button
load_button = ttk.Button(button_frame, text="Load Data", command=load_data)
load_button.grid(row=0, column=0, padx=10)

# Train and Predict button
train_button = ttk.Button(button_frame, text="Train and Predict", command=train_and_predict)
train_button.grid(row=0, column=1, padx=10)

# Status label
status_label = tk.Label(root, text="", fg="black")
status_label.pack(pady=10)

root.mainloop()
