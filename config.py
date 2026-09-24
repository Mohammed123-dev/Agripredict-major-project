# config.py
import os
# config.py
import mysql.connector

def get_db_connection():
    connection = mysql.connector.connect(
        host="localhost",      # లేదా 127.0.0.1
        user="root",           # నీ MySQL username
        password="root",       # నీ MySQL password
        database="chand",      # నీ database పేరు
        auth_plugin="mysql_native_password"
    )
    return connection


def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",         # 👉 మీ MySQL username
        password="yourpass", # 👉 మీ MySQL password
        database="agriculture_db",
        auth_plugin="mysql_native_password"
    )


"""
Configuration file for storing API keys and global settings.
Make sure to set your WEATHER_API_KEY before running the app.
"""

# ✅ Load API key from environment variable
weather_api_key = os.getenv("WEATHER_API_KEY", "3d8ed1a8de0dfcee8d0e132be58a53af")

# ✅ Debugging info
if weather_api_key and weather_api_key != "":
    print("🔑 WEATHER_API_KEY loaded successfully")
else:
    print("❌ WEATHER_API_KEY not found. Please set it before running the app.")
