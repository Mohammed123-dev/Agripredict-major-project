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
from flask import (
    Flask,
    render_template,
    request,
    make_response,
    jsonify,
    send_file
)
from markupsafe import Markup

import torch
from torchvision import transforms

from deep_translator import GoogleTranslator
import pyttsx3

from utils.model import ResNet9
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic

import config


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

warnings.filterwarnings("ignore")


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# DISEASE CLASSES
# ============================================================

disease_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Blueberry___healthy",

    "Cherry___Powdery_mildew",
    "Cherry___healthy",

    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",

    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",

    "Orange___Haunglongbing_(Citrus_greening)",

    "Peach___Bacterial_spot",
    "Peach___healthy",

    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",

    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",

    "Raspberry___healthy",

    "Soybean___healthy",

    "Squash___Powdery_mildew",

    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",

    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    return mysql.connector.connect(

        host=os.getenv(
            "DB_HOST",
            "localhost"
        ),

        port=int(
            os.getenv(
                "DB_PORT",
                "3306"
            )
        ),

        user=os.getenv(
            "DB_USER",
            "maibu"
        ),

        password=os.getenv(
            "DB_PASSWORD",
            ""
        ),

        database=os.getenv(
            "DB_NAME",
            "chand"
        ),

        auth_plugin="mysql_native_password",

        connection_timeout=10
    )


# ============================================================
# DATABASE TEST
# ============================================================

try:

    conn = get_db_connection()

    print(
        "✅ Database connection successful"
    )

    conn.close()

except Exception as e:

    print(
        "⚠️ Database connection failed:",
        repr(e)
    )


# ============================================================
# WEATHER FUNCTION
# ============================================================

def get_weather(city):

    try:

        api_key = getattr(
            config,
            "weather_api_key",
            None
        )

        if not api_key:

            print(
                "⚠️ Weather API key not configured"
            )

            return {
                "temperature": 25,
                "humidity": 50
            }

        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={api_key}&units=metric"
        )

        response = requests.get(
            url,
            timeout=10
        )

        if response.status_code != 200:

            print(
                "⚠️ Weather API error:",
                response.status_code
            )

            return {
                "temperature": 25,
                "humidity": 50
            }

        data = response.json()

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        return {
            "temperature": temperature,
            "humidity": humidity
        }

    except Exception as e:

        print(
            "⚠️ Weather error:",
            repr(e)
        )

        return {
            "temperature": 25,
            "humidity": 50
        }


# ============================================================
# CROP RECOMMENDATION MODEL
# ============================================================

crop_recommendation_model = None

crop_model_path = os.path.join(
    MODELS_DIR,
    "Crop_Recommendation_Model.pkl"
)

try:

    print(
        "📂 Loading crop model:",
        crop_model_path
    )

    if not os.path.exists(
        crop_model_path
    ):

        raise FileNotFoundError(
            f"Crop model not found: {crop_model_path}"
        )

    crop_recommendation_model = joblib.load(
        crop_model_path
    )

    print(
        "✅ Crop recommendation model loaded"
    )

except Exception as e:

    print(
        "⚠️ Crop recommendation model not loaded:",
        repr(e)
    )

    crop_recommendation_model = None


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

random_forest_model = None

random_forest_model_path = os.path.join(
    MODELS_DIR,
    "RandomForest.pkl"
)

try:

    print(
        "📂 Loading RandomForest model:",
        random_forest_model_path
    )

    if not os.path.exists(
        random_forest_model_path
    ):

        raise FileNotFoundError(
            f"RandomForest model not found: "
            f"{random_forest_model_path}"
        )

    random_forest_model = joblib.load(
        random_forest_model_path
    )

    print(
        "✅ RandomForest model loaded"
    )

except Exception as e:

    print(
        "⚠️ RandomForest model not loaded:",
        repr(e)
    )

    random_forest_model = None


# ============================================================
# PRICE MODEL
# ============================================================

price_model = None

price_model_path = os.path.join(
    BASE_DIR,
    "price_model.pkl"
)

try:

    if os.path.exists(
        price_model_path
    ):

        price_model = joblib.load(
            price_model_path
        )

        print(
            "✅ Price model loaded"
        )

    else:

        print(
            "⚠️ Price model not found"
        )

except Exception as e:

    print(
        "⚠️ Price model loading failed:",
        repr(e)
    )

    price_model = None


# ============================================================
# DISEASE MODEL
# ============================================================

disease_model = None

disease_model_path = os.path.join(
    MODELS_DIR,
    "Plant_Disease_Model.pth"
)

try:

    print(
        "📂 Loading disease model:"
    )

    print(
        disease_model_path
    )

    if not os.path.exists(
        disease_model_path
    ):

        raise FileNotFoundError(
            f"Disease model not found: "
            f"{disease_model_path}"
        )

    # Create ResNet9 model
    disease_model = ResNet9(
        3,
        len(disease_classes)
    )

    # Load model on CPU
    state_dict = torch.load(
        disease_model_path,
        map_location=torch.device("cpu")
    )

    # Load trained weights
    disease_model.load_state_dict(
        state_dict
    )

    # Evaluation mode
    disease_model.eval()

    print(
        "✅ Disease model loaded successfully"
    )

except Exception as e:

    print(
        "❌ Error loading disease model:",
        repr(e)
    )

    disease_model = None


# ============================================================
# CLEAN HTML
# ============================================================

def clean_html(text):

    if not text:

        return ""

    text = re.sub(
        r"<[^>]+>",
        "",
        str(text)
    )

    return text.strip()


# ============================================================
# DISEASE IMAGE PREDICTION
# ============================================================

def predict_image(img_bytes):

    if disease_model is None:

        raise Exception(
            "Disease model is not loaded. "
            "Please check Plant_Disease_Model.pth."
        )

    try:

        # Open image
        image = Image.open(
            io.BytesIO(img_bytes)
        ).convert("RGB")

        # Transform image
        transform = transforms.Compose([

            transforms.Resize(
                (256, 256)
            ),

            transforms.ToTensor()
        ])

        image_tensor = transform(
            image
        ).unsqueeze(0)

        # Prediction
        with torch.no_grad():

            outputs = disease_model(
                image_tensor
            )

            _, predicted = torch.max(
                outputs,
                dim=1
            )

        predicted_index = predicted.item()

        if (
            predicted_index < 0
            or
            predicted_index >= len(
                disease_classes
            )
        ):

            raise Exception(
                f"Invalid prediction index: "
                f"{predicted_index}"
            )

        prediction = disease_classes[
            predicted_index
        ]

        print(
            "✅ Disease prediction:",
            prediction
        )

        return prediction

    except Exception as e:

        print(
            "❌ Image prediction error:",
            repr(e)
        )

        raise


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# CROP RECOMMENDATION
# ============================================================

@app.route(
    "/crop-predict",
    methods=["GET", "POST"]
)
def crop_prediction():

    if request.method == "GET":

        return render_template(
            "crop.html"
        )

    try:

        if crop_recommendation_model is None:

            return render_template(
                "crop.html",
                error="Crop recommendation model is not available."
            )

        nitrogen = float(
            request.form.get(
                "nitrogen",
                0
            )
        )

        phosphorous = float(
            request.form.get(
                "phosphorous",
                0
            )
        )

        potassium = float(
            request.form.get(
                "pottasium",
                request.form.get(
                    "potassium",
                    0
                )
            )
        )

        ph = float(
            request.form.get(
                "ph",
                0
            )
        )

        rainfall = float(
            request.form.get(
                "rainfall",
                0
            )
        )

        city = request.form.get(
            "city",
            ""
        )

        weather = get_weather(
            city
        )

        temperature = weather[
            "temperature"
        ]

        humidity = weather[
            "humidity"
        ]

        features = np.array([[
            nitrogen,
            phosphorous,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        ]])

        prediction = (
            crop_recommendation_model
            .predict(features)
        )

        crop = prediction[0]

        print(
            "✅ Recommended crop:",
            crop
        )

        return render_template(
            "crop-result.html",
            prediction=crop,
            title="Crop Recommendation"
        )

    except Exception as e:

        print(
            "❌ Crop prediction error:",
            repr(e)
        )

        return render_template(
            "crop.html",
            error=f"Error during crop prediction: {str(e)}"
        )


# ============================================================
# FERTILIZER PREDICTION
# ============================================================

@app.route(
    "/fertilizer-predict",
    methods=["GET", "POST"]
)
def fertilizer_prediction():

    if request.method == "GET":

        return render_template(
            "fertilizer.html"
        )

    try:

        crop_name = request.form.get(
            "cropname",
            ""
        )

        nitrogen = request.form.get(
            "nitrogen",
            ""
        )

        phosphorous = request.form.get(
            "phosphorous",
            ""
        )

        potassium = request.form.get(
            "pottasium",
            request.form.get(
                "potassium",
                ""
            )
        )

        recommendation = ""

        # Try matching fertilizer dictionary
        if crop_name in fertilizer_dic:

            recommendation = fertilizer_dic[
                crop_name
            ]

        else:

            recommendation = (
                "Please provide valid crop information "
                "for fertilizer recommendation."
            )

        return render_template(
            "fertilizer-result.html",
            recommendation=Markup(
                str(recommendation)
            ),
            cropname=crop_name,
            nitrogen=nitrogen,
            phosphorous=phosphorous,
            potassium=potassium,
            title="Fertilizer Recommendation"
        )

    except Exception as e:

        print(
            "❌ Fertilizer error:",
            repr(e)
        )

        return render_template(
            "fertilizer.html",
            error=f"Error during prediction: {str(e)}"
        )


# ============================================================
# DISEASE DETECTION
# ============================================================

@app.route(
    "/disease-predict",
    methods=["GET", "POST"]
)
def disease_prediction():

    title = "Agronomy - Disease Detection"

    # GET
    if request.method == "GET":

        return render_template(
            "disease.html",
            title=title,
            prediction=None
        )

    # Check file
    if "file" not in request.files:

        return render_template(
            "disease.html",
            title=title,
            error="No file part in the request."
        )

    file = request.files.get(
        "file"
    )

    # Check selected file
    if not file or file.filename == "":

        return render_template(
            "disease.html",
            title=title,
            error="No file selected."
        )

    try:

        # Read image
        img_bytes = file.read()

        print(
            "📂 File uploaded, size:",
            len(img_bytes),
            "bytes"
        )

        if len(img_bytes) == 0:

            return render_template(
                "disease.html",
                title=title,
                error="Uploaded file is empty."
            )

        # Predict
        prediction_label = predict_image(
            img_bytes
        )

        print(
            "✅ Prediction:",
            prediction_label
        )

        # Check disease dictionary
        if (
            prediction_label
            not in disease_dic
        ):

            return render_template(
                "disease.html",
                title=title,
                error=(
                    "Prediction information "
                    "not found."
                )
            )

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

        # ====================================================
        # DATABASE SAVE
        # ====================================================

        try:

            conn = get_db_connection()

            cursor = conn.cursor()

            # ------------------------------------------------
            # IMPORTANT:
            # If your database table already has an INSERT
            # query, keep your original query here.
            # ------------------------------------------------

            cursor.close()
            conn.close()

            print(
                "✅ Disease result processed"
            )

        except Exception as db_error:

            print(
                "⚠️ Database save failed:",
                repr(db_error)
            )

        # ====================================================
        # RESULT PAGE
        # ====================================================

        return render_template(
            "disease-result.html",
            prediction=prediction_label,
            description=prediction_text,
            title=title
        )

    except Exception as e:

        print(
            "❌ Disease prediction error:",
            repr(e)
        )

        return render_template(
            "disease.html",
            title=title,
            error=(
                f"Error during prediction: {str(e)}"
            )
        )


# ============================================================
# PRICE PAGE
# ============================================================

@app.route(
    "/price",
    methods=["GET"]
)
def price():

    return render_template(
        "price.html"
    )


# ============================================================
# PRICE PREDICTION
# ============================================================

@app.route(
    "/predict_price",
    methods=["POST"]
)
def predict_price():

    try:

        if price_model is None:

            return jsonify({
                "success": False,
                "error": "Price model is not available."
            }), 500

        data = request.form.to_dict()

        print(
            "📊 Price request:",
            data
        )

        # Convert numeric values
        values = []

        for value in data.values():

            try:

                values.append(
                    float(value)
                )

            except Exception:

                pass

        if not values:

            return jsonify({
                "success": False,
                "error": "No numeric input received."
            }), 400

        features = np.array(
            [values]
        )

        prediction = (
            price_model
            .predict(features)
        )

        result = prediction[0]

        return jsonify({
            "success": True,
            "prediction": float(result)
        })

    except Exception as e:

        print(
            "❌ Price prediction error:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# TRANSLATION
# ============================================================

@app.route(
    "/translate",
    methods=["POST"]
)
def translate():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            data = request.form.to_dict()

        text = data.get(
            "text",
            ""
        )

        target_language = data.get(
            "target",
            data.get(
                "language",
                "en"
            )
        )

        if not text:

            return jsonify({
                "success": False,
                "error": "Text is required."
            }), 400

        language_map = {
            "en": "en",
            "te": "te",
            "hi": "hi",
            "ta": "ta",
            "kn": "kn",
            "ml": "ml"
        }

        target_language = language_map.get(
            target_language,
            "en"
        )

        translated_text = (
            GoogleTranslator(
                source="auto",
                target=target_language
            ).translate(text)
        )

        return jsonify({
            "success": True,
            "translation": translated_text
        })

    except Exception as e:

        print(
            "❌ Translation error:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
            silent=True
        )

        if not data:

            data = request.form.to_dict()

        text = data.get(
            "text",
            ""
        )

        if not text:

            return jsonify({
                "success": False,
                "error": "Text is required."
            }), 400

        output_file = os.path.join(
            BASE_DIR,
            "speech.mp3"
        )

        engine = pyttsx3.init()

        engine.save_to_file(
            text,
            output_file
        )

        engine.runAndWait()

        if not os.path.exists(
            output_file
        ):

            return jsonify({
                "success": False,
                "error": "Speech file was not created."
            }), 500

        return send_file(
            output_file,
            mimetype="audio/mpeg"
        )

    except Exception as e:

        print(
            "❌ TTS error:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "running",

        "crop_model": (
            crop_recommendation_model
            is not None
        ),

        "random_forest_model": (
            random_forest_model
            is not None
        ),

        "disease_model": (
            disease_model
            is not None
        ),

        "price_model": (
            price_model
            is not None
        )
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({
        "error": "Page not found"
    }), 404


@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({
        "error": "Internal server error"
    }), 500


# ============================================================
# RUN APPLICATION
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
