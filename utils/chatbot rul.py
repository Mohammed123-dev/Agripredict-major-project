# utils/chatbot_rules.py

def get_bot_response(user_input: str) -> str:
    """
    Simple rule-based chatbot for agriculture-related queries.
    """

    # Convert input to lowercase for matching
    user_input = user_input.lower()

    responses = {
        "hello": "Hi! 👋 How can I help you with farming today?",
        "hi": "Hello! 🌱 Do you want crop suggestion, fertilizer advice, or disease detection?",
        "fertilizer": "Please provide your crop name and soil type, I will suggest suitable fertilizer.",
        "crop": "Enter your soil details (N, P, K, pH, rainfall, temperature, humidity) in the crop recommendation section.",
        "disease": "Upload a clear leaf image in the disease detection section to identify the problem.",
        "thanks": "You're welcome! Happy farming 🌾",
        "bye": "Goodbye! Take care of your crops 🌱"
    }

    # Default reply if no rule matches
    default_reply = "Sorry, I didn’t understand 🤔. Please ask about crops, fertilizers, or plant diseases."

    # Return matched response or default
    return responses.get(user_input, default_reply)
