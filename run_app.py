from waitress import serve
from app import app  # make sure your Flask app object is named 'app'

serve(app, host='0.0.0.0', port=5000)
