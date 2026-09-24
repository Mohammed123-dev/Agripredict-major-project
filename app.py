# ============================================================
# AGRIPREDICT - FLASK APPLICATION
# ============================================================

import os
import io
import re
import pickle
import warnings

import numpy as np
import pandas as pd
import requests
import joblib
import mysql.connector

from PIL import Image
from flask import Flask, render_template, request, make_response, jsonify, send_file, Markup

import torch
from torchvision import transforms

from deep_translator import GoogleTranslator
import pyttsx3

from utils.model import ResNet9
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic

import config

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn"
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    """
    Creates MySQL connection using environment variables.

    For local development:
        DB_HOST=localhost
        DB_USER=maibu
        DB_PASSWORD=22j21a05d5
        DB_NAME=chand

    For Render:
        Add these values in Render Environment Variables.
    """

    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "maibu"),
        password=os.getenv("DB_PASSWORD", "22j21a05d5"),
        database=os.getenv("DB_NAME", "chand"),
        auth_plugin="mysql_native_password",
        connection_timeout=10
    )


# Test database connection
try:
    test_db = get_db_connection()

    if test_db.is_connected():
        print("✅ MySQL connection successful")

    test_db.close()

except Exception as e:
    print("⚠️ MySQL connection failed. App will still start.")
    print("Database error:", e)


# ============================================================
# WEATHER
# ============================================================

def weather_fetch(city_name):
    """
    Fetch temperature and humidity from OpenWeatherMap.
    """

    try:
        api_key = config.weather_api_key

        base_url = "http://api.openweathermap.org/data/2.5/weather"

        params = {
            "appid": api_key,
            "q": city_name
        }

        response = requests.get(
            base_url,
            params=params,
            timeout=10
        )

        data = response.json()

        if response.status_code == 200 and "main" in data:

            temperature = round(
                data["main"]["temp"] - 273.15,
                2
            )

            humidity = data["main"]["humidity"]

            return temperature, humidity

        print(
            "[ERROR] Weather fetch failed:",
            data.get("message", "Unknown error")
        )

        return None

    except Exception as e:

        print("[ERROR] Weather API error:", e)

        return None


# ============================================================
# DISEASE CLASSES
# ============================================================

disease_classes = [
    'Apple___Apple_scab',
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
    'Tomato___healthy'
]


# ============================================================
# LOAD DISEASE MODEL
# ============================================================

disease_model_path = "models/Plant_Disease_Model.pth"

try:

    disease_model = ResNet9(
        3,
        len(disease_classes)
    )

    disease_model.load_state_dict(
        torch.load(
            disease_model_path,
            map_location=torch.device("cpu")
        )
    )

    disease_model.eval()

    print("✅ Disease model loaded successfully")

except Exception as e:

    print("❌ Error loading disease model:", e)

    disease_model = None


# ============================================================
# DISEASE PREDICTION FUNCTION
# ============================================================

def predict_image(img_bytes):

    if disease_model is None:
        raise Exception("Disease model is not loaded.")

    image = Image.open(
        io.BytesIO(img_bytes)
    ).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image_tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        outputs = disease_model(
            image_tensor
        )

        _, predicted = torch.max(
            outputs,
            dim=1
        )

    prediction = disease_classes[
        predicted.item()
    ]

    return prediction


# ============================================================
# LOAD CROP RECOMMENDATION MODEL
# ============================================================

crop_recommendation_model = None

try:

    crop_recommendation_model = joblib.load(
        "models/Crop_Recommendation_Model.pkl"
    )

    print(
        "✅ Crop recommendation model loaded"
    )

except Exception as e:

    print(
        "⚠️ Crop recommendation model not loaded:",
        e
    )


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

try:

    random_forest_model_path = (
        "models/RandomForest.pkl"
    )

    crop_recommendation_model = pickle.load(
        open(
            random_forest_model_path,
            "rb"
        )
    )

    print(
        "✅ RandomForest model loaded"
    )

except Exception as e:

    print(
        "⚠️ RandomForest model not loaded:",
        e
    )


# ============================================================
# LOAD PRICE MODEL
# ============================================================

try:

    with open(
        "price_model.pkl",
        "rb"
    ) as f:

        price_model = pickle.load(f)

    print(
        "✅ Price model loaded successfully"
    )

except Exception as e:

    print(
        "⚠️ Price model loading failed:",
        e
    )

    price_model = None


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    title = "Agronomy - Home"

    return render_template(
        "index.html",
        title=title
    )


# ============================================================
# CROP RECOMMENDATION PAGE
# ============================================================

@app.route("/crop-recommend")
def crop_recommend():

    title = "Agronomy - Crop Recommendation"

    return render_template(
        "crop.html",
        title=title
    )


# ============================================================
# FERTILIZER PAGE
# ============================================================

@app.route("/fertilizer")
def fertilizer_recommendation():

    title = "Agronomy - Fertilizer Suggestion"

    return render_template(
        "fertilizer.html",
        title=title
    )


# ============================================================
# CROP PREDICTION
# ============================================================

@app.route(
    "/crop-predict",
    methods=["POST"]
)
def crop_prediction():

    title = "Agronomy - Crop Recommendation"

    try:

        # ----------------------------------------------------
        # GET INPUT
        # ----------------------------------------------------

        N = int(
            request.form.get(
                "nitrogen",
                0
            )
        )

        P = int(
            request.form.get(
                "phosphorous",
                0
            )
        )

        K = int(
            request.form.get(
                "pottasium",
                0
            )
        )

        ph = float(
            request.form.get(
                "ph",
                7.0
            )
        )

        rainfall = float(
            request.form.get(
                "rainfall",
                50.0
            )
        )

        city = request.form.get(
            "city",
            ""
        )


        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        weather = weather_fetch(city)

        if weather:

            temperature, humidity = weather

        else:

            temperature = 25.0
            humidity = 50.0


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        if crop_recommendation_model is None:

            raise Exception(
                "Crop recommendation model is not loaded."
            )

        data = np.array([
            [
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]
        ])

        prediction = (
            crop_recommendation_model
            .predict(data)
        )

        final_prediction = prediction[0]


        # ----------------------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------------------

        conn = None
        cursor = None

        try:

            conn = get_db_connection()

            cursor = conn.cursor()

            insert_query = """
                INSERT INTO crop_prediction_results
                (
                    nitrogen,
                    phosphorous,
                    potassium,
                    temperature,
                    humidity,
                    ph,
                    rainfall,
                    predicted_crop
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """

            values = (
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall,
                final_prediction
            )

            cursor.execute(
                insert_query,
                values
            )

            conn.commit()

            print(
                "✅ Crop prediction saved"
            )

        except Exception as db_error:

            print(
                "⚠️ Crop DB error:",
                db_error
            )

        finally:

            if cursor:
                cursor.close()

            if conn:
                conn.close()


        return render_template(
            "crop-result.html",
            prediction=final_prediction,
            title=title
        )


    except ValueError as e:

        return render_template(
            "try_again.html",
            title=title,
            error=f"Invalid input: {e}"
        )


    except Exception as e:

        print(
            "❌ Crop prediction error:",
            e
        )

        return render_template(
            "try_again.html",
            title=title,
            error=f"Something went wrong: {e}"
        )


# ============================================================
# FERTILIZER PREDICTION
# ============================================================

@app.route(
    "/fertilizer-predict",
    methods=["POST"]
)
def fert_recommend():

    title = "Agronomy - Fertilizer Suggestion"

    try:

        crop_name = str(
            request.form["cropname"]
        )

        N = int(
            request.form["nitrogen"]
        )

        P = int(
            request.form["phosphorous"]
        )

        K = int(
            request.form["pottasium"]
        )


        # ----------------------------------------------------
        # READ FERTILIZER CSV
        # ----------------------------------------------------

        df = pd.read_csv(
            "Data/fertilizer.csv"
        )

        crop_data = df[
            df["Crop"] == crop_name
        ]

        if crop_data.empty:

            raise Exception(
                "Crop not found in fertilizer.csv"
            )

        nr = crop_data["N"].iloc[0]
        pr = crop_data["P"].iloc[0]
        kr = crop_data["K"].iloc[0]


        # ----------------------------------------------------
        # CALCULATE
        # ----------------------------------------------------

        n = nr - N
        p = pr - P
        k = kr - K

        temp = {
            abs(n): "N",
            abs(p): "P",
            abs(k): "K"
        }

        max_value = temp[
            max(temp.keys())
        ]


        if max_value == "N":

            key = (
                "NHigh"
                if n < 0
                else "Nlow"
            )

        elif max_value == "P":

            key = (
                "PHigh"
                if p < 0
                else "Plow"
            )

        else:

            key = (
                "KHigh"
                if k < 0
                else "Klow"
            )


        response = Markup(
            str(
                fertilizer_dic[key]
            )
        )


        # ----------------------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------------------

        conn = None
        cursor = None

        try:

            conn = get_db_connection()

            cursor = conn.cursor()

            sql = """
                INSERT INTO fertilizer_prediction_results
                (
                    crop_name,
                    nitrogen,
                    phosphorous,
                    potassium,
                    recommendation
                )
                VALUES (%s,%s,%s,%s,%s)
            """

            values = (
                crop_name,
                N,
                P,
                K,
                str(
                    fertilizer_dic[key]
                )
            )

            cursor.execute(
                sql,
                values
            )

            conn.commit()

            print(
                "✅ Fertilizer prediction saved"
            )

        except Exception as db_error:

            print(
                "⚠️ Fertilizer DB error:",
                db_error
            )

        finally:

            if cursor:
                cursor.close()

            if conn:
                conn.close()


        return render_template(
            "fertilizer-result.html",
            recommendation=response,
            title=title
        )


    except Exception as e:

        print(
            "❌ Fertilizer prediction error:",
            e
        )

        return render_template(
            "try_again.html",
            title=title,
            error=f"Something went wrong: {e}"
        )


# ============================================================
# REMOVE HTML
# ============================================================

def striphtml(data):

    p = re.compile(
        r"<.*?>"
    )

    return p.sub(
        "",
        data
    )


# ============================================================
# DOWNLOAD FERTILIZER REPORT
# ============================================================

@app.route(
    "/download1",
    methods=["GET", "POST"]
)
def download1():

    if request.method == "POST":

        f = request.form[
            "fileData"
        ]

        f = striphtml(f)

        response = make_response(f)

        response.headers[
            "Content-Disposition"
        ] = (
            "attachment; "
            "filename=Fertilizer_Prediction_Report.txt"
        )

        return response

    return render_template(
        "index.html"
    )


# ============================================================
# DOWNLOAD DISEASE REPORT
# ============================================================

@app.route(
    "/download2",
    methods=["GET", "POST"]
)
def download2():

    if request.method == "POST":

        f = request.form[
            "fileData"
        ]

        f = striphtml(f)

        response = make_response(f)

        response.headers[
            "Content-Disposition"
        ] = (
            "attachment; "
            "filename=Disease_Prediction_Report.txt"
        )

        return response

    return render_template(
        "index.html"
    )


# ============================================================
# CLEAN HTML
# ============================================================

def clean_html(raw_text):

    clean = re.compile(
        "<.*?>"
    )

    return re.sub(
        clean,
        "",
        raw_text
    )


# ============================================================
# DISEASE PREDICTION
# ============================================================

@app.route(
    "/disease-predict",
    methods=["GET", "POST"]
)
def disease_prediction():

    title = "Agronomy - Disease Detection"

    if request.method == "GET":

        return render_template(
            "disease.html",
            title=title,
            prediction=None
        )


    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "file" not in request.files:

        return render_template(
            "disease.html",
            title=title,
            error="No file part in the request."
        )


    file = request.files.get(
        "file"
    )


    if not file or file.filename == "":

        return render_template(
            "disease.html",
            title=title,
            error="No file selected."
        )


    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        img_bytes = file.read()

        print(
            "📂 File uploaded, size:",
            len(img_bytes)
        )


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction_label = predict_image(
            img_bytes
        )

        print(
            "✅ Prediction:",
            prediction_label
        )


        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        if prediction_label not in disease_dic:

            return render_template(
                "disease.html",
                title=title,
                error="Prediction failed."
            )


        # ----------------------------------------------------
        # DESCRIPTION
        # ----------------------------------------------------

        prediction_text = Markup(
            str(
                disease_dic[
                    prediction_label
                ]
            )
        )

        prediction_text_str = str(
            prediction_text
        )

        description_clean = clean_html(
            prediction_text_str
        )


        # ----------------------------------------------------
        # SAVE TO DATABASE
        # ----------------------------------------------------

        conn = None
        cursor = None

        try:

            conn = get_db_connection()

            cursor = conn.cursor()

            sql = """
                INSERT INTO disease_prediction_results
                (
                    crop_name,
                    disease_name,
                    recommendation
                )
                VALUES (%s,%s,%s)
            """

            values = (
                "Unknown Crop",
                prediction_label,
                description_clean
            )

            cursor.execute(
                sql,
                values
            )

            conn.commit()

            print(
                "✅ Disease prediction saved"
            )

        except Exception as db_error:

            print(
                "⚠️ Disease DB error:",
                db_error
            )

        finally:

            if cursor:
                cursor.close()

            if conn:
                conn.close()


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return render_template(
            "disease-result.html",
            prediction=prediction_label,
            description=prediction_text,
            title=title
        )


    except Exception as e:

        print(
            "❌ Disease prediction error:",
            e
        )

        return render_template(
            "disease.html",
            title=title,
            error=f"Error during prediction: {e}"
        )


# ============================================================
# PRICE PAGE + PRICE PREDICTION
# ============================================================

@app.route(
    "/price",
    methods=["GET", "POST"]
)
def price():

    if request.method == "GET":

        return render_template(
            "price.html"
        )


    try:

        crop = "grapes"

        demand = int(
            request.form["demand"]
        )

        supply = int(
            request.form["supply"]
        )

        quantity = float(
            request.form["quantity"]
        )

        unit = request.form.get(
            "unit",
            "kg"
        )


        # ----------------------------------------------------
        # CONVERT TO KG
        # ----------------------------------------------------

        if unit == "quintal":

            quantity_kg = (
                quantity * 100
            )

        elif unit == "ton":

            quantity_kg = (
                quantity * 1000
            )

        else:

            quantity_kg = quantity


        # ----------------------------------------------------
        # PRICE
        # ----------------------------------------------------

        price_value = 32.03

        total = round(
            price_value * quantity_kg,
            2
        )


        return render_template(
            "price.html",
            crop=crop,
            price=price_value,
            demand=demand,
            supply=supply,
            quantity=quantity,
            unit=unit,
            total=total
        )


    except Exception as e:

        return render_template(
            "price.html",
            error=f"Error: {str(e)}"
        )


# ============================================================
# PRICE PREDICTION API
# ============================================================

@app.route(
    "/predict_price",
    methods=["POST"]
)
def predict_price():

    try:

        if price_model is None:

            return jsonify({
                "error": "Price model not loaded"
            }), 500


        # ----------------------------------------------------
        # REQUEST DATA
        # ----------------------------------------------------

        data = request.get_json(
            force=True
        )

        print(
            "📥 Received Data:",
            data
        )


        crop = data.get(
            "crop",
            "Unknown"
        )

        year = int(
            data.get(
                "year",
                2025
            )
        )

        month = int(
            data.get(
                "month",
                1
            )
        )

        rainfall = float(
            data.get(
                "rainfall",
                0
            )
        )

        demand = float(
            data.get(
                "demand",
                0
            )
        )

        supply = float(
            data.get(
                "supply",
                0
            )
        )

        quantity = float(
            data.get(
                "quantity",
                1
            )
        )

        unit = data.get(
            "unit",
            "kg"
        )


        # ----------------------------------------------------
        # CONVERT QUANTITY
        # ----------------------------------------------------

        if unit == "quintal":

            quantity_kg = (
                quantity * 100
            )

        elif unit == "ton":

            quantity_kg = (
                quantity * 1000
            )

        else:

            quantity_kg = quantity


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        features = np.array([
            [
                year,
                month,
                rainfall,
                demand,
                supply
            ]
        ])

        predicted_price_per_kg = float(
            price_model.predict(
                features
            )[0]
        )


        # Prevent negative price

        predicted_price_per_kg = max(
            predicted_price_per_kg,
            1.0
        )


        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        total_amount = (
            predicted_price_per_kg
            * quantity_kg
        )


        # ----------------------------------------------------
        # SAVE DATABASE
        # ----------------------------------------------------

        conn = None
        cursor = None

        try:

            conn = get_db_connection()

            cursor = conn.cursor()

            sql = """
                INSERT INTO price_prediction_results
                (
                    crop_name,
                    year,
                    month,
                    rainfall,
                    demand,
                    supply,
                    quantity,
                    unit,
                    predicted_price_per_kg,
                    total_amount
                )
                VALUES
                (
                    %s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s
                )
            """

            values = (
                crop,
                year,
                month,
                rainfall,
                demand,
                supply,
                quantity,
                unit,
                predicted_price_per_kg,
                total_amount
            )

            cursor.execute(
                sql,
                values
            )

            conn.commit()

            print(
                "✅ Price prediction saved"
            )

        except Exception as db_error:

            print(
                "⚠️ Price DB error:",
                db_error
            )

        finally:

            if cursor:
                cursor.close()

            if conn:
                conn.close()


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "predicted_price_per_kg":
                round(
                    predicted_price_per_kg,
                    2
                ),

            "quantity":
                quantity,

            "unit":
                unit,

            "total_amount":
                round(
                    total_amount,
                    2
                ),

            "crop":
                crop
        })


    except Exception as e:

        print(
            "❌ Price prediction error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# TRANSLATION API
# ============================================================

@app.route(
    "/translate",
    methods=["POST"]
)
def translate():

    try:

        data = request.get_json(
            force=True
        )

        text = data.get(
            "text",
            ""
        )

        target = data.get(
            "target",
            "te"
        )


        if not text:

            return jsonify({
                "error": "No text provided"
            }), 400


        translated = GoogleTranslator(
            source="en",
            target=target
        ).translate(text)


        return jsonify({
            "original": text,
            "translated": translated,
            "language": target
        })


    except Exception as e:

        print(
            "❌ Translation error:",
            e
        )

        return jsonify({
            "error": "Translation service is temporarily unavailable."
        }), 503


# ============================================================
# TEXT TO SPEECH
# ============================================================

@app.route(
    "/tts",
    methods=["POST"]
)
def tts():

    try:

        data = request.get_json(
            force=True
        )

        text = data.get(
            "text",
            ""
        )


        if not text:

            return jsonify({
                "error": "No text"
            }), 400


        engine = pyttsx3.init()

        voices = engine.getProperty(
            "voices"
        )


        # Try to find Telugu voice

        for voice in voices:

            if "telugu" in voice.name.lower():

                engine.setProperty(
                    "voice",
                    voice.id
                )

                break


        filename = "output.mp3"

        engine.save_to_file(
            text,
            filename
        )

        engine.runAndWait()


        return send_file(
            filename,
            mimetype="audio/mpeg"
        )


    except Exception as e:

        print(
            "❌ TTS error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "message": "AgriPredict Flask server is running"
    })


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
