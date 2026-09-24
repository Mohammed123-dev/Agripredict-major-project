import mysql.connector
import pandas as pd

# ✅ MySQL connection
mydb = mysql.connector.connect(
    host="localhost",
    user="maibu",          # <-- నీ username
    password="22j21a05d5", # <-- నీ password
    database="chand",
    auth_plugin='mysql_native_password'
)

print("✅ Connected to MySQL:", mydb.is_connected())

# ✅ Query data
query = """
SELECT 
    id,
    crop_name,
    disease_name,
    LEFT(recommendation, 80) AS short_recommendation,
    created_at
FROM disease_prediction_results
"""
df = pd.read_sql(query, mydb)

# ✅ Show data in nice table
print("\n📊 Disease Prediction Results (Preview):\n")
print(df.to_string(index=False))

# ✅ Close connection
mydb.close()
