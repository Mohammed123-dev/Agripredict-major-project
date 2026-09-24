# Importing essential libraries and modules
import os
import mysql.connector
from flask import Flask, render_template, request, Markup, make_response
from utils.model import ResNet9, predict_image
from utils.model import predict_image
from PIL import Image
import io
import torch
from torchvision import transforms
import joblib
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ----------------------------------------------------------------------------------------------------------------------
import pyttsx3
from flask import Flask, request, send_file
import os
# =======================================================================================================================
from deep_translator import GoogleTranslator

# price predction

import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
# ===================================


from google.cloud import texttospeech

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

from deep_translator import GoogleTranslator
# ==============================================================================================================================

# Database connection
from flask import Flask, render_template, request, Markup, make_response
import pandas as pd
import re
from config import get_db_connection   # ✅ config.py నుంచి DB connection తీసుకుంటాం
from utils.fertilizer import fertilizer_dic

from flask import Flask, render_template, request, Markup
from config import get_db_connection   # ✅ మీ config.py లోని DB connection తీసుకుంటుంది
from utils.model import predict_image
from utils.disease import disease_dic

# ----------------- DB Connection -----------------
# ----------------- Database Connection (Safe for Render) -----------------
mydb = None
cursor = None

try:
    mydb = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "maibu"),
        password=os.getenv("DB_PASSWORD", "22j21a05d5"),
        database=os.getenv("DB_NAME", "chand"),
        auth_plugin="mysql_native_password"
    )
    print("✅ Connected to MySQL:", mydb.is_connected())
    cursor = mydb.cursor()
except Exception as e:
    print("⚠️ MySQL connection failed (app will still start):", e)


# ===================================================================================================================================
# Example: English → Telugu
translated = GoogleTranslator(source="en", target="te").translate("Hello")
print(translated)


# ==============================================
engine = pyttsx3.init()

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/Plant_Disease_Model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location = torch.device('cpu')))
disease_model.eval()

# -------------------
# Image Prediction Function
# -------------------
def predict_image(img_bytes):
    # Open image
    img = Image.open(io.BytesIO(img_bytes))

    # ✅ Force convert to RGB (fix for 4-channel images like PNG with alpha)
    img = img.convert("RGB")

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Prediction
    with torch.no_grad():
        outputs = disease_model(img_tensor)
        _, predicted = outputs.max(1)
        return class_names[predicted.item()]


# Loading crop recommendation model
crop_recommendation_model = joblib.load("models/Crop_Recommendation_Model.pkl")
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# =========================================================================================

# Custom functions for calculations

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity or None
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = f"{base_url}appid={api_key}&q={city_name}"
    response = requests.get(complete_url)
    x = response.json()

    if response.status_code == 200 and "main" in x:
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        print(f"[ERROR] Weather fetch failed: {x.get('message', 'Unknown error')}")
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)

# render home page

@ app.route('/')
def home():
    title = 'Agronomy - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page

@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Agronomy - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page

@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Agronomy - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# =============================================================================================================================================================================================

# render crop recommendation result page

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="maibu",
        password="22j21a05d5",
        database="chand",
        auth_plugin="mysql_native_password"
    )

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Agronomy - Crop Recommendation'

    if request.method == 'POST':
        try:
            # Get form inputs
            N = int(request.form.get('nitrogen', 0))
            P = int(request.form.get('phosphorous', 0))
            K = int(request.form.get('pottasium', 0))
            ph = float(request.form.get('ph', 7.0))
            rainfall = float(request.form.get('rainfall', 50.0))
            city = request.form.get("city", "")

            # Fetch weather
            try:
                weather = weather_fetch(city)
            except Exception as e:
                print(f"[ERROR] Weather fetch failed: {e}")
                weather = None

            if weather:
                temperature, humidity = weather
            else:
                temperature, humidity = 25.0, 50.0

            # Model prediction
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            # ✅ Save result into MySQL
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                insert_query = """
                    INSERT INTO crop_prediction_results 
                    (nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, predicted_crop) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (N, P, K, temperature, humidity, ph, rainfall, final_prediction)
                cursor.execute(insert_query, values)
                conn.commit()
                cursor.close()
                conn.close()
                print("✅ Data inserted successfully into crop_prediction_results")
            except Exception as db_err:
                print(f"❌ Database Insert Error: {db_err}")

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        except ValueError as ve:
            return render_template('try_again.html', title=title, error=f"Invalid input: {ve}")

        except Exception as e:
            return render_template('try_again.html', title=title, error=f"Something went wrong: {e}")

# =========================================================================================================================================================================================
# render fertilizer recommendation result page

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Agronomy - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    # ✅ CSV నుంచి ideal NPK తీసుకోవడం
    df = pd.read_csv('Data/fertilizer.csv')
    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]

    if max_value == "N":
        key = 'NHigh' if n < 0 else 'Nlow'
    elif max_value == "P":
        key = 'PHigh' if p < 0 else 'Plow'
    else:
        key = 'KHigh' if k < 0 else 'Klow'

    response = Markup(str(fertilizer_dic[key]))

    # ✅ Database లో save చేయడం
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO fertilizer_prediction_results 
            (crop_name, nitrogen, phosphorous, potassium, recommendation)
            VALUES (%s, %s, %s, %s, %s)
        """
        values = (crop_name, N, P, K, str(fertilizer_dic[key]))
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Fertilizer prediction saved in DB")
    except Exception as e:
        print("❌ DB Insert Error:", e)

    return render_template('fertilizer-result.html',
                           recommendation=response,
                           title=title)

# ----------------------------------------------------
# Helper function (HTML remove చేసి download కోసం)
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

@app.route('/download1', methods=['GET', 'POST'])
def download1():
    if request.method == 'POST':
        f = request.form['fileData']
        f = striphtml(f)
        response = make_response(f)
        response.headers["Content-Disposition"] = "attachment; filename=Fertilizer Prediction Report.txt"
        return response
    return render_template('index.html')

@app.route('/download2', methods=['GET', 'POST'])
def download2():
    if request.method == 'POST':
        f = request.form['fileData']
        f = striphtml(f)
        response = make_response(f)
        response.headers["Content-Disposition"] = "attachment; filename=Disease Prediction Report.txt"
        return response
    return render_template('index.html')


# ============================================================================================================================================================================================
# render disease prediction result page


# --- HTML remove function ---
def clean_html(raw_text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', raw_text)


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Agronomy - Disease Detection'
    if request.method == 'POST':
        if 'file' not in request.files:
            print("❌ No file part in request")
            return render_template('disease.html', title=title, error="No file part in the request.")

        file = request.files.get('file')
        if not file or file.filename == '':
            print("❌ No file selected")
            return render_template('disease.html', title=title, error="No file selected for uploading.")

        try:
            # ✅ Read image bytes
            img_bytes = file.read()
            print("📂 File uploaded, size:", len(img_bytes))

            # ✅ Call your custom prediction function
            prediction_label = predict_image(img_bytes)
            print("✅ Prediction Label:", prediction_label)

            # ✅ Prediction validate
            if prediction_label not in disease_dic:
                print("❌ Unknown prediction:", prediction_label)
                return render_template('disease.html', title=title, error="Prediction failed. Unknown disease label.")

            # ✅ Disease description (with HTML for frontend display)
            prediction_text = Markup(str(disease_dic[prediction_label]))
            prediction_text_str = str(prediction_text)

            # ✅ Clean text for saving in MySQL
            description_clean = clean_html(prediction_text_str)

            # ✅ Save to MySQL
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                sql = """
                    INSERT INTO disease_prediction_results (crop_name, disease_name, recommendation) 
                    VALUES (%s, %s, %s)
                """
                values = ("Unknown Crop", prediction_label, description_clean)  # 👈 cleaned text
                cursor.execute(sql, values)
                conn.commit()

                print("✅ Saved in MySQL (HTML removed)")
            except Exception as db_err:
                print("❌ Database Error:", db_err)
            finally:
                if cursor: cursor.close()
                if conn: conn.close()

            # ✅ Render result page (HTML allowed here)
            return render_template(
                'disease-result.html',
                prediction=prediction_label,
                description=prediction_text,   # 👈 render with HTML
                title=title
            )

        except Exception as e:
            print("❌ Error during prediction:", str(e))
            return render_template('disease.html', title=title, error=f"Error during prediction: {e}")

    # ✅ GET request
    return render_template('disease.html', title=title, prediction=None)


#================================================================================================================================================================================================
# Crop price prediction page

@app.route("/price", methods=["GET", "POST"])
def price():
    if request.method == "POST":
        try:
            crop = "grapes"  # (Should come from your ML model ideally)
            demand = int(request.form["demand"])
            supply = int(request.form["supply"])
            quantity = float(request.form["quantity"])
            unit = request.form.get("unit", "kg")  # default unit = kg

            # 🔄 Convert input quantity into kg
            if unit == "quintal":
                quantity_kg = quantity * 100
            elif unit == "ton":
                quantity_kg = quantity * 1000
            else:
                quantity_kg = quantity

            price = round(32.03, 2)  # (Predict from your ML model instead)
            total = round(price * quantity_kg, 2)

            return render_template(
                "price.html",
                crop=crop,
                price=price,
                demand=demand,
                supply=supply,
                quantity=quantity,
                unit=unit,
                total=total
            )
        except Exception as e:
            return render_template("price.html", error=f"Error: {str(e)}")

    return render_template("price.html")


@app.route('/price')
def price_page():
    return render_template("price.html")


# ------------------ Load trained model safely ------------------
try:
    with open("price_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# ------------------ Price Prediction API ------------------
@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # 📥 Get request data
        data = request.get_json(force=True)
        print("📥 Received Data:", data)

        crop = data.get('crop', 'Unknown')
        year = int(data.get('year', 2025))
        month = int(data.get('month', 1))
        rainfall = float(data.get('rainfall', 0))
        demand = float(data.get('demand', 0))
        supply = float(data.get('supply', 0))
        quantity = float(data.get('quantity', 1))
        unit = data.get("unit", "kg")

        # 🔄 Convert all quantities to kg
        if unit == "quintal":
            quantity_kg = quantity * 100
        elif unit == "ton":
            quantity_kg = quantity * 1000
        else:
            quantity_kg = quantity

        # 🔮 Predict price per kg
        features = np.array([[year, month, rainfall, demand, supply]])
        predicted_price_per_kg = float(model.predict(features)[0])
        predicted_price_per_kg = max(predicted_price_per_kg, 1.0)  # prevent negatives

        # 💰 Calculate total
        total_amount = predicted_price_per_kg * quantity_kg

        # ✅ Save to MySQL
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            sql = """INSERT INTO price_prediction_results 
                     (crop_name, year, month, rainfall, demand, supply, quantity, unit, predicted_price_per_kg, total_amount) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

            values = (crop, year, month, rainfall, demand, supply, quantity, unit,
                      predicted_price_per_kg, total_amount)

            cursor.execute(sql, values)
            conn.commit()
            print("✅ Price Prediction Saved in MySQL")

        except Exception as db_err:
            print("❌ Database Error:", db_err)
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

        # 📤 Response
        return jsonify({
            "predicted_price_per_kg": round(predicted_price_per_kg, 2),
            "quantity": quantity,
            "unit": unit,
            "total_amount": round(total_amount, 2),
            "crop": crop
        })

    except Exception as e:
        print("❌ Error in /predict_price:", e)
        return jsonify({"error": str(e)}), 500


# =================================================================================================================================================================================================
@app.route("/tts", methods=["POST"])
def tts():
    text = request.json.get("text", "")
    if not text:
        return {"error": "No text"}, 400
    
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    # Telugu voice systemలో ఉంటే set చేయవచ్చు
    for v in voices:
        if "telugu" in v.name.lower():
            engine.setProperty("voice", v.id)
            break
    
    filename = "output.mp3"
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return send_file(filename, mimetype="audio/mpeg")

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
