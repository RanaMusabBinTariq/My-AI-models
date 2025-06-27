import pandas as pd
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

# Function to generate synthetic data
def generate_synthetic_data(num_samples=1000):
    genders = ['Male', 'Female']
    activity_levels = ['Low', 'Moderate', 'High']
    creatine_phases = ['Maintenance', 'Loading']
    weather_conditions = ['Cold', 'Moderate', 'Hot']
    data = []

    for _ in range(num_samples):
        gender = random.choice(genders)
        weight = random.randint(50, 120)  # Weight between 50kg and 120kg
        height = random.uniform(1.5, 2.0)  # Random height between 1.5m and 2.0m
        activity_level = random.choice(activity_levels)
        creatine_dosage = round(random.uniform(3, 10), 1)  # Between 3g and 10g
        creatine_phase = random.choice(creatine_phases)
        weather = random.choice(weather_conditions)
        
        # Calculate BMI
        bmi = weight / (height ** 2)
        
        # Calculate the daily water intake (ml) based on these factors
        base_water_intake = 35 * weight  # A general formula: 35ml per kg of body weight

        # Adjust for activity level
        if activity_level == 'Low':
            base_water_intake += 0
        elif activity_level == 'Moderate':
            base_water_intake += 200
        else:  # High activity
            base_water_intake += 400
        
        # Adjust for creatine phase
        if creatine_phase == 'Loading':
            base_water_intake += 300  # Extra 300ml for loading phase

        # Adjust for weather
        if weather == 'Cold':
            base_water_intake += 0
        elif weather == 'Moderate':
            base_water_intake += 150
        else:  # Hot weather
            base_water_intake += 300  # Extra 300ml for hot weather
        
        data.append([gender, weight, height, activity_level, creatine_dosage, creatine_phase, weather, bmi, base_water_intake])
    
    df = pd.DataFrame(data, columns=['Gender', 'Weight', 'Height', 'ActivityLevel', 'CreatineDosage', 'CreatinePhase', 'Weather', 'BMI', 'WaterIntake'])
    return df

# Generate synthetic data
df = generate_synthetic_data(num_samples=1000)

# Preprocess data
def preprocess_data(df):
    label_encoders = {}
    categorical_columns = ['Gender', 'ActivityLevel', 'CreatinePhase', 'Weather']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df[['Gender', 'Weight', 'Height', 'ActivityLevel', 'CreatineDosage', 'CreatinePhase', 'Weather', 'BMI']]
    y = df['WaterIntake']
    
    return X, y, label_encoders

# Split data into training and test sets
X, y, label_encoders = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train[['Weight', 'Height', 'BMI']] = scaler.fit_transform(X_train[['Weight', 'Height', 'BMI']])
X_test[['Weight', 'Height', 'BMI']] = scaler.transform(X_test[['Weight', 'Height', 'BMI']])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Best Model Mean Absolute Error: {mae}")

# Save the best model and label encoders for later use
joblib.dump(best_model, 'hydration_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Prediction function with scaling
def predict_water_intake(gender, weight, height, activity_level, creatine_dosage, creatine_phase, weather):
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Weight': [weight],
        'Height': [height],
        'ActivityLevel': [activity_level],
        'CreatineDosage': [creatine_dosage],
        'CreatinePhase': [creatine_phase],
        'Weather': [weather]
    })
    
    # Calculate BMI
    input_data['BMI'] = input_data['Weight'] / (input_data['Height'] ** 2)
    
    # Encode categorical features using the label encoders
    for col in ['Gender', 'ActivityLevel', 'CreatinePhase', 'Weather']:
        input_data[col] = label_encoders[col].transform(input_data[col])
    
    # Scale the features
    input_data[['Weight', 'Height', 'BMI']] = scaler.transform(input_data[['Weight', 'Height', 'BMI']])
    
    # Predict water intake
    predicted_water_intake = best_model.predict(input_data)
    
    return predicted_water_intake[0]

# Example prediction
print("Predicted Water Intake:", predict_water_intake('Male', 80, 1.8, 'High', 5.0, 'Loading', 'Hot'))
import tkinter as tk
from tkinter import ttk, messagebox

# GUI for testing predictions
def run_gui():
    def on_predict():
        try:
            gender = gender_var.get()
            weight = float(weight_entry.get())
            height = float(height_entry.get())
            activity_level = activity_var.get()
            creatine_dosage = float(creatine_entry.get())
            creatine_phase = phase_var.get()
            weather = weather_var.get()

            result = predict_water_intake(
                gender, weight, height,
                activity_level, creatine_dosage,
                creatine_phase, weather
            )
            messagebox.showinfo("Prediction", f"Predicted Daily Water Intake: {round(result)} ml")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Main window
    window = tk.Tk()
    window.title("Water Intake Predictor")

    # Gender
    tk.Label(window, text="Gender").grid(row=0, column=0)
    gender_var = ttk.Combobox(window, values=["Male", "Female"])
    gender_var.grid(row=0, column=1)

    # Weight
    tk.Label(window, text="Weight (kg)").grid(row=1, column=0)
    weight_entry = tk.Entry(window)
    weight_entry.grid(row=1, column=1)

    # Height
    tk.Label(window, text="Height (m)").grid(row=2, column=0)
    height_entry = tk.Entry(window)
    height_entry.grid(row=2, column=1)

    # Activity Level
    tk.Label(window, text="Activity Level").grid(row=3, column=0)
    activity_var = ttk.Combobox(window, values=["Low", "Moderate", "High"])
    activity_var.grid(row=3, column=1)

    # Creatine Dosage
    tk.Label(window, text="Creatine Dosage (g)").grid(row=4, column=0)
    creatine_entry = tk.Entry(window)
    creatine_entry.grid(row=4, column=1)

    # Creatine Phase
    tk.Label(window, text="Creatine Phase").grid(row=5, column=0)
    phase_var = ttk.Combobox(window, values=["Maintenance", "Loading"])
    phase_var.grid(row=5, column=1)

    # Weather
    tk.Label(window, text="Weather").grid(row=6, column=0)
    weather_var = ttk.Combobox(window, values=["Cold", "Moderate", "Hot"])
    weather_var.grid(row=6, column=1)

    # Predict button
    predict_button = tk.Button(window, text="Predict Water Intake", command=on_predict)
    predict_button.grid(row=7, column=0, columnspan=2, pady=10)

    window.mainloop()

# Run GUI
run_gui()