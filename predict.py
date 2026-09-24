import pickle
import pandas as pd

# Model load చేయటం
with open("models/price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input (Year, Month, Rainfall, Demand, Supply)
sample = pd.DataFrame([[2025, 6, 120, 60, 40]], 
                      columns=["Year", "Month", "Rainfall", "Demand", "Supply"])

# Prediction
pred_price = model.predict(sample)[0]
print(f"🌾 Predicted Crop Price: ₹{pred_price:.2f}")
