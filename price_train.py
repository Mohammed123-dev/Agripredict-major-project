import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from preprocess import load_and_preprocess

# Load dataset
df = load_and_preprocess("crop_prices.csv")
print("✅ Cleaned Dataset Shape:", df.shape)

# Encode categorical columns
le_state = LabelEncoder()
le_crop = LabelEncoder()

df["State"] = le_state.fit_transform(df["State"])
df["Crop"] = le_crop.fit_transform(df["Crop"])

# Features (X) and target (y)
X = df.drop("Price", axis=1)   # all except Price
y = df["Price"]               # target = Price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("✅ Model trained successfully!")
print("R² Score (train):", model.score(X_train, y_train))
print("R² Score (test):", model.score(X_test, y_test))
