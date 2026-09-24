# train_crop_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ==============================
# Step 1: Load Dataset
# ==============================
DATA_PATH = "Crop_recommendation.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "Crop_Recommendation_Model.pkl")

try:
    data = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    raise Exception(f"❌ Dataset not found! Please place '{DATA_PATH}' in the project folder.")

# ==============================
# Step 2: Features & Target
# ==============================
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# ==============================
# Step 3: Split Data
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Step 4: Train Model
# ==============================
print("⏳ Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ==============================
# Step 5: Evaluate Model
# ==============================
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")

# ==============================
# Step 6: Save Model
# ==============================
os.makedirs(MODEL_DIR, exist_ok=True)  # create 'models/' folder if not exists
joblib.dump(model, MODEL_PATH)
print(f"📁 Model saved at: {MODEL_PATH}")
