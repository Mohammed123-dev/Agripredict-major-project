import mysql.connector
import re

# --- HTML remove function ---
def clean_html(raw_text):
    if raw_text is None:
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', raw_text)

# ✅ Connect to MySQL
mydb = mysql.connector.connect(
    host="localhost",
    user="maibu",
    password="22j21a05d5",
    database="chand",
    auth_plugin='mysql_native_password'
)

print("✅ Connected to MySQL:", mydb.is_connected())
cursor = mydb.cursor()

# ✅ Select all rows from table
cursor.execute("SELECT id, recommendation FROM disease_prediction_results")
rows = cursor.fetchall()

# ✅ Clean each row
for row in rows:
    record_id = row[0]           # id
    recommendation_text = row[1] # recommendation text
    clean_text = clean_html(recommendation_text)

    cursor.execute(
        "UPDATE disease_prediction_results SET recommendation = %s WHERE id = %s",
        (clean_text, record_id)
    )
    print(f"✅ Cleaned record ID {record_id}")

# ✅ Commit once after loop
mydb.commit()

# ✅ Close connection
cursor.close()
mydb.close()

print("🎉 All records cleaned successfully!")
