from flask import Flask, request, send_file, render_template_string
from gtts import gTTS
import os
import uuid

app = Flask(__name__)

# 🏠 Home page (సాధారణ UI)
@app.route("/")
def home():
    return render_template_string("""
    <h2>🌐 Multi-Language Text to Speech Demo</h2>
    <form action="/speak" method="post">
        <textarea name="text" rows="4" cols="50" placeholder="Type text here..."></textarea><br><br>
        <label>Choose language:</label>
        <select name="lang">
            <option value="en">English</option>
            <option value="te">Telugu</option>
            <option value="hi">Hindi</option>
            <option value="ta">Tamil</option>
            <option value="kn">Kannada</option>
            <option value="ml">Malayalam</option>
        </select><br><br>
        <button type="submit">🔊 Speak</button>
    </form>
    """)

# 🎙️ Generate speech
@app.route("/speak", methods=["POST"])
def speak():
    text = request.form.get("text")
    lang = request.form.get("lang", "en")

    if not text:
        return "❌ Please provide text"

    # unique filename
    filename = f"output_{uuid.uuid4().hex}.mp3"

    # gTTS generate audio
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)

    # send MP3 file
    return send_file(filename, mimetype="audio/mpeg", as_attachment=False)

if __name__ == "__main__":
    app.run(debug=True)
