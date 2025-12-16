#  user interface kinda  

import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import joblib
import os
import re
import numpy as np 


# This corrects the Concept Drift issue caused by the old data.
APPRECIATION_FACTOR = 1.65 

try:
    # Load models and necessary objects from the 'models' folder
    lr_model = joblib.load("models/linear_model.joblib")
    rf_model = joblib.load("models/random_forest_model.joblib")
    features = joblib.load("models/model_features.pkl")
    location_encoder = joblib.load("models/location_encoder.joblib") 
except FileNotFoundError:
    messagebox.showerror("Error", "Models or encoders not found. Run main.py first.")
    exit()

# location mapping and cleaning function (must match main.py)
def clean_column(name):
    name = name.strip().lower()
    name = name.replace(' ', '_')
    name = re.sub(r'[()]', '', name)
    return name

# Prepare city to location map for dropdowns 
data_folder = 'archive'
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

city_location_map = {}
for file in csv_files:
    
    city_name_raw = os.path.splitext(file)[0]
    city_name = clean_column(city_name_raw)
    
    df = pd.read_csv(os.path.join(data_folder, file))
    df.columns = [clean_column(c) for c in df.columns]
    
    if 'location' in df.columns:
        city_location_map[city_name] = sorted(df['location'].dropna().unique().tolist())
    else:
        city_location_map[city_name] = []


# gui setup
root = tk.Tk()
root.title("🏠 Housing Price Prediction (India)")
root.geometry("500x550") 
root.resizable(False, False)

tk.Label(root, text="Enter House Details", font=("Arial", 14, "bold")).pack(pady=10)


# Input fields
tk.Label(root, text="Area (sqft):").pack(pady=3)
area_entry = tk.Entry(root)
area_entry.pack()

tk.Label(root, text="Number of Bedrooms:").pack(pady=3)
bedrooms_entry = tk.Entry(root)
bedrooms_entry.pack()

tk.Label(root, text="Resale/New:").pack(pady=3)
resale_var = tk.StringVar(value='New')
resale_dropdown = ttk.Combobox(root, textvariable=resale_var, state="readonly")
resale_dropdown['values'] = ['Resale', 'New'] 
resale_dropdown.pack()

tk.Label(root, text="City:").pack(pady=3)
city_var = tk.StringVar()
city_dropdown = ttk.Combobox(root, textvariable=city_var, state="readonly")
city_dropdown['values'] = sorted(city_location_map.keys())
city_dropdown.pack()

tk.Label(root, text="Location:").pack(pady=3)
location_var = tk.StringVar()
location_dropdown = ttk.Combobox(root, textvariable=location_var, state="readonly")
location_dropdown.pack()

# location updates when city is changed
def update_locations(event):
    city = city_var.get()
    if city in city_location_map:
        location_dropdown['values'] = city_location_map[city]
        if city_location_map[city]:
            location_var.set(city_location_map[city][0])
        else:
            location_var.set('')
    else:
        location_dropdown['values'] = []
        location_var.set('')

city_dropdown.bind("<<ComboboxSelected>>", update_locations)

# Helper Function for BI Output Formatting
def format_price(price):
    """Converts raw rupee value to Lakhs or Crores for readability."""
    if price >= 10000000: # 1 Crore = 100 Lakhs
        return f"₹{price / 10000000:,.2f} Crore"
    elif price >= 100000: # 1 Lakh
        return f"₹{price / 100000:,.2f} Lakhs"
    else:
        return f"₹{price:,.0f} Rupees"


# Prediction Function 
def predict_price():
    try:
        
        # 1. Assemble Input DataFrame
        input_data = {
            'area': [float(area_entry.get())],
            'no._of_bedrooms': [int(bedrooms_entry.get())],
            'resale': [resale_var.get()],
            'location': [location_var.get()]
        }
        input_df = pd.DataFrame(input_data)

        
        # 2. Encode Categorical Data (Matching main.py preprocessing)
        # Note: 'resale' encoding must match how LabelEncoder handled 'New' and 'Resale' in main.py
        input_df['resale'] = input_df['resale'].apply(lambda x: 1 if x == 'Resale' else 0)
        
        # Encode 'location' using the SAVED encoder (CRITICAL STEP)
        location_str = input_df['location'].iloc[0]
        try:
            # Transform the location string into the correct integer the model expects
            encoded_location = location_encoder.transform([location_str])[0]
        except ValueError:
            # Handle unknown location 
            encoded_location = 0 
            messagebox.showwarning("Warning", f"Location '{location_str}' is unknown to the model. Prediction accuracy is lower.")

        input_df['location'] = encoded_location
        
        
        # 3. Align Features
        input_df = input_df[features]

        
        # 4. Predict (Output is in log-space)
        lr_pred_log = lr_model.predict(input_df)[0]
        rf_pred_log = rf_model.predict(input_df)[0]
        
        
        # 5. Inverse Log Transform (Get Historical Price)
        lr_pred_historical = np.expm1(lr_pred_log)
        rf_pred_historical = np.expm1(rf_pred_log)
        avg_pred_historical = (lr_pred_historical + rf_pred_historical) / 2
        
        # 6. APPLY BI CORRECTION 
        # Multiply the raw historical predictions by the 65% appreciation factor
        lr_pred_corrected = lr_pred_historical * APPRECIATION_FACTOR
        rf_pred_corrected = rf_pred_historical * APPRECIATION_FACTOR
        avg_pred_corrected = (lr_pred_corrected + rf_pred_corrected) / 2

        # formatting display results
        result_label.config(
            text=f" Raw Model Output \n"
                 f"Linear Regression: {format_price(lr_pred_historical)}\n"
                 f"Random Forest: {format_price(rf_pred_historical)}\n"
                 f"----------------------------------------\n"
                 f"Current Market Estimate:\n"
                 f"Random Forest (Adjusted): {format_price(rf_pred_corrected)}",
            font=("Arial", 12, "bold") 
        )
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for Area and Bedrooms.")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An unexpected error occurred. Details: {e}")

#  Button and Result Label 
tk.Button(root, text="Predict Price", command=predict_price, bg="#4CAF50", fg="white", width=20).pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.pack(pady=10)


root.mainloop()
