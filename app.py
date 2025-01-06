from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('TUMABOT.h5')

# Initialize Flask app
app = Flask(__name__)

# Define chatbot response function
def chatbot_response(input_text):
    # Placeholder: Modify this function to process input_text and predict using the model
    input_vector = np.array([input_text])  # Convert input to a suitable format
    response_vector = model.predict(input_vector)
    response_text = "".join(map(str, response_vector[0]))  # Placeholder logic
    return response_text

# API route for chatbot
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'error': 'Message is required.'}), 400

    bot_response = chatbot_response(user_message)
    return jsonify({'response': bot_response})

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
