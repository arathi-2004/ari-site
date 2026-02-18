import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'rice_crop_model.h5'
DATA_DIR = 'Original Images'
# Use absolute path for upload folder to avoid relative path issues on Windows
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print(f"Upload Folder Path: {UPLOAD_FOLDER}")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD MODEL ---
print("Loading AI Brain...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    # Get class names from directory structure if available, else use a default list or handle error
    if os.path.exists(DATA_DIR):
        class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    else:
        # Fallback if directory isn't there (though it should be based on your setup)
        class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast', 'Leaf scald', 'Narrow Brown Leaf Spot', 'Rice Hispa', 'Sheath Blight']
            
    print(f"Model loaded. Classes: {class_names}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    class_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving file to: {filepath}")
        try:
            file.save(filepath)
            print("File saved successfully.")
        except Exception as e:
            print(f"Error saving file: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Failed to save file: {str(e)}'})
        
        if model:
            try:
                # Process image
                print(f"Processing image: {filepath}")
                img_array = prepare_image(filepath)
                
                # Predict
                print("Running prediction...")
                predictions = model.predict(img_array)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = float(100 * np.max(predictions))
                print(f"Prediction: {predicted_class} ({confidence}%)")
                
                # Translations
                translations = {
                    'Bacterial Leaf Blight': 'ബാക്ടീരിയൽ ലീഫ് ബ്ലൈറ്റ്',
                    'Brown Spot': 'തവിട്ടു പുള്ളി രോഗം',
                    'Healthy Rice Leaf': 'ആരോഗ്യമുള്ള ഇല',
                    'Leaf Blast': 'ബ്ലാസ്റ്റ് രോഗം',
                    'Leaf scald': 'ഇല കരിച്ചിൽ',
                    'Narrow Brown Leaf Spot': 'നാരോ ബ്രൗൺ ലീഫ് സ്പോട്ട്',
                    'Rice Hispa': 'റൈസ് ഹിസ്പ',
                    'Sheath Blight': 'പോള രോഗം'
                }
                
                translated_class = translations.get(predicted_class, predicted_class)

                return jsonify({
                    'class': translated_class,
                    'confidence': confidence,
                    'image_url': f'/static/uploads/{filename}'
                })
            except Exception as e:
                print(f"Error during prediction: {e}")
                import traceback
                traceback.print_exc()
                # Return full traceback to the user for debugging
                return jsonify({'error': f'Error processing image: {str(e)}', 'traceback': traceback.format_exc()})
        else:
            return jsonify({'error': 'Model not loaded'})

    return jsonify({'error': 'Invalid file type'})

import json
from datetime import datetime

# --- CHAT STORAGE ---
CHAT_FILE = 'chat_history.json'

def load_messages():
    if os.path.exists(CHAT_FILE):
        try:
            with open(CHAT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_message(username, text):
    messages = load_messages()
    new_msg = {
        'username': username,
        'text': text,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    messages.append(new_msg)
    # Keep last 100 messages
    messages = messages[-100:]
    with open(CHAT_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    return new_msg

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/api/messages', methods=['GET'])
def get_messages():
    return jsonify(load_messages())

@app.route('/api/messages', methods=['POST'])
def post_message():
    data = request.json
    if not data or 'username' not in data or 'text' not in data:
        return jsonify({'error': 'Invalid data'}), 400
    
    msg = save_message(data['username'], data['text'])
    return jsonify(msg)

if __name__ == '__main__':
    app.run(debug=True)
