from google.cloud import texttospeech

# ✅ Google TTS Client
client = texttospeech.TextToSpeechClient()

# ✅ Telugu లో ఒక sample text
synthesis_input = texttospeech.SynthesisInput(text="హలో! ఇది తెలుగు వాయిస్ టెస్ట్.")

# ✅ Voice Config
voice = texttospeech.VoiceSelectionParams(
    language_code="te-IN",
    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)

# ✅ Audio Config (MP3 Output)
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

# ✅ Call API
response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config
)

# ✅ Save Output
out_path = r"C:\Users\Maibu\output.mp3"
with open(out_path, "wb") as out:
    out.write(response.audio_content)

print("✅ Audio saved at:", out_path)
