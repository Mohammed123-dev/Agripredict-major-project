import mysql.connector

try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="maibu",
        password="22j21a05d5",
        database="chand",
        auth_plugin='mysql_native_password'
    )

    if mydb.is_connected():
        print("✅ Connected to MySQL successfully")

        cursor = mydb.cursor()
        cursor.execute("SHOW TABLES;")
        print("📋 Tables in database:")
        for x in cursor:
            print("   -", x[0])

except mysql.connector.Error as e:
    print("❌ Error while connecting to MySQL:", e)
