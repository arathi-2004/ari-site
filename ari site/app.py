import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
import sqlite3
from PIL import Image
import json
from datetime import datetime

app = Flask(__name__)

# --- CONFIGURATION ---
# Use absolute paths where possible to avoid issues on Windows
MODEL_PATH = 'rice_crop_model.h5'
DATA_DIR = 'Original Images'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'rice_doctor_super_secret_key_123'  # Required for sessions
print(f"Upload Folder Path: {UPLOAD_FOLDER}")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- DATABASE SETUP ---
DB_FILE = 'users.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# --- LOAD MODEL ---
print("Loading AI Brain...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(DATA_DIR):
        class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    else:
        # Fallback class names
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

def get_weather_integrated_recommendation(disease, temp_c, humidity, is_rain):
    """Returns one unified recommendation shaped by disease + current weather."""
    hot      = temp_c is not None and temp_c >= 32
    humid    = humidity is not None and humidity >= 80
    cool_dry = temp_c is not None and temp_c < 24 and (humidity is None or humidity < 60)

    if is_rain:         scenario = 'rainy'
    elif hot and humid: scenario = 'hot_humid'
    elif humid:         scenario = 'humid'
    elif hot:           scenario = 'hot'
    elif cool_dry:      scenario = 'cool_dry'
    else:               scenario = 'normal'

    t = round(temp_c) if temp_c else '?'
    h = int(humidity) if humidity else '?'

    matrix = {
        'Bacterial Leaf Blight': {
            'rainy':     '🌧️ മഴ ഉള്ളതിനാൽ ഇന്ന് കെമിക്കൽ തളിക്കൽ ഒഴിവാക്കുക. ജലനിർഗ്ഗമനം ഉറപ്പാക്കുക, ബാധിത ചെടികൾ നീക്കം ചെയ്യുക. മഴ ശമിച്ചശേഷം കോപ്പർ ഓക്‌സിക്ലോറൈഡ് ഉപയോഗിക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ബാക്ടീരിയ ദ്രുതഗതിയിൽ പടരാൻ സാധ്യതയുണ്ട്. ഉടൻ കോപ്പർ ഓക്‌സിക്ലോറൈഡ് വൈകുന്നേരം തളിക്കുക. നൈട്രജൻ വളം ഒഴിവാക്കുക; ജലനിർഗ്ഗമനം ഉറപ്പാക്കുക.',
            'humid':     f'💧 {h}% ആർദ്രത — രോഗ വ്യാപന ഭീഷണി. വൈകുന്നേരം കോപ്പർ കുമിൾനാശിനി തളിക്കുക. നൈട്രജൻ വളം ഉപയോഗിക്കുന്നത് നിർത്തുക.',
            'hot':       f'🌡️ {t}°C — ബാക്ടീരിയ ആക്ടീവ് ആണ്. രാവിലെ 7–9 മണിക്ക് കോപ്പർ ഓക്‌സിക്ലോറൈഡ് തളിക്കുക. ഉച്ചസമയത്ത് ജലസേചനം ഒഴിവാക്കുക.',
            'cool_dry':  f'❄️ {t}°C — ഇപ്പോൾ രോഗ ഭീഷണി കുറവാണ്. ബാധിത ചെടികൾ നീക്കം ചെയ്ത് കോപ്പർ കുമിൾനാശിനി പ്രതിരോധമായി ഉപയോഗിക്കുക.',
            'normal':    'കോപ്പർ ഓക്‌സിക്ലോറൈഡ് തളിക്കുക. ബാധിത ചെടികൾ നീക്കം ചെയ്യുക. ജലനിർഗ്ഗമനം ഉറപ്പാക്കുക. നൈട്രജൻ വളം കുറയ്ക്കുക.',
        },
        'Brown Spot': {
            'rainy':     '🌧️ ഫംഗൽ സ്‌പോറുകൾ മഴവെള്ളത്തിലൂടെ പടരാം. ഇന്ന് ഫംഗിസൈഡ് തളിക്കുന്നത് ഒഴിവാക്കുക; മഴ ശമിച്ചശേഷം മാൻകോസേബ് ഉപയോഗിക്കുക. ജലനിർഗ്ഗമനം ഉറപ്പാക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ഫംഗൽ ഭീഷണി കൂടുതലാണ്. ഉടൻ മാൻകോസേബ് വൈകുന്നേരം തളിക്കുക. സിലിക്കൺ-പൊട്ടാഷ് വളങ്ങൾ ചേർക്കുക.',
            'humid':     f'💧 {h}% — ഫംഗൽ ഭീഷണി. മാൻകോസേബ് (2.5 g/L) വൈകുന്നേരം തളിക്കുക.',
            'hot':       f'🌡️ {t}°C — ഫംഗസ് ആക്ടീവ് ആണ്. രാവിലെ ട്രൈസൈക്ലസോൾ ഉപയോഗിക്കുക. പൊട്ടാഷ്-സിലിക്കൺ വളങ്ങൾ നൽകുക.',
            'cool_dry':  f'❄️ {t}°C — ഫംഗൽ ഭീഷണി കുറവാണ്. ഒരു ഫംഗിസൈഡ് ഉപയോഗിക്കുക. നല്ലയിനം വിത്തുകൾ ഉപയോഗിക്കുന്നത് ഉറപ്പാക്കുക.',
            'normal':    'മാൻകോസേബ് അല്ലെങ്കിൽ ട്രൈസൈക്ലസോൾ ഫംഗിസൈഡ് ഉപയോഗിക്കുക. സിലിക്കൺ, പൊട്ടാഷ് വളങ്ങൾ ഉപയോഗിക്കുക.',
        },
        'Healthy Rice Leaf': {
            'rainy':     '🌧️ ചെടി ആരോഗ്യകരമാണ്! എങ്കിലും മഴക്കാലത്ത് ഫംഗൽ ഭീഷണി ഉണ്ടാകാം — ജലനിർഗ്ഗമനം ഉറപ്പാക്കുക; ആഴ്ചതോറും ഇലകൾ നിരീക്ഷിക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ചെടിക്ക് നിലവിൽ കുഴപ്പമില്ല. എന്നാൽ ഈ കാലാവസ്ഥ ബ്ലാസ്റ്റ്, പോള രോഗം എന്നിവയ്ക്ക് കാരണമായേക്കാം. ആഴ്ചതോറും ഇലകൾ പരിശോധിക്കുക.',
            'humid':     f'💧 {h}% — ആരോഗ്യം നിലനിർത്തുന്നു. ഫംഗൽ ഭീഷണി ഉണ്ടാകാം; 5–7 ദിവസത്തിലൊരിക്കൽ ഇലകൾ നിരീക്ഷിക്കുക.',
            'hot':       f'🌡️ {t}°C — ചെടി ആരോഗ്യകരമാണ്. ഉച്ചസമയത്ത് ജലസേചനം ഒഴിവാക്കുക; ബ്ലാസ്റ്റ് രോഗത്തിന്റെ ലക്ഷണങ്ങൾ ശ്രദ്ധിക്കുക.',
            'cool_dry':  f'✅ {t}°C — കാലാവസ്ഥ നിലവിൽ അനുകൂലമാണ്. കൃഷി ഇതുപോലെ തുടരുക. ആഴ്ചതോറും ഇലകൾ പരിശോധിച്ച് ഉറപ്പുവരുത്തുക.',
            'normal':    '✅ ചെടി ആരോഗ്യകരമാണ്! കൃത്യമായ ജലം, വളം, കളനാശിനി എന്നിവ തുടരുക. ആഴ്ചതോറും ഒന്നു വീക്ഷിക്കുന്നത് നന്നായിരിക്കും.',
        },
        'Leaf Blast': {
            'rainy':     '🌧️ ഇന്ന് ഫംഗിസൈഡ് ഒഴിവാക്കുക; മഴ ശമിച്ചശേഷം ഉടൻ ട്രൈസൈക്ലസോൾ തളിക്കുക. ഇളം ഇലകൾ പ്രത്യേകം ശ്രദ്ധിക്കുക — അവ വേഗത്തിൽ രോഗബാധിതരാകാം.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ഇത് ഏറ്റവും ഗുരുതരമായ അവസ്ഥയാണ്. ഉടൻ ട്രൈസൈക്ലസോൾ (0.6 g/L) വൈകുന്നേരം തളിക്കുക. നൈട്രജൻ വളങ്ങൾ നൽകുന്നത് ഉടൻ നിർത്തുക.',
            'humid':     f'💧 {h}% — ബ്ലാസ്റ്റ് ഫംഗസ് ആക്ടീവ് ആണ്. ഇന്ന് വൈകുന്നേരം ട്രൈസൈക്ലസോൾ തളിക്കുക. 10–14 ദിവസങ്ങൾക്ക് ശേഷം ഇത് ആവർത്തിക്കുക.',
            'hot':       f'🌡️ {t}°C — ഫംഗസ് ഭീഷണി. ഉടൻ ചികിത്സ ആരംഭിക്കുക. ഉച്ചസമയത്ത് തളിക്കുന്നത് ഒഴിവാക്കി രാവിലെ അല്ലെങ്കിൽ വൈകുന്നേരം ഉപയോഗിക്കുക.',
            'cool_dry':  f'❄️ {t}°C — ഭീഷണി കുറഞ്ഞ അവസ്ഥയാണ്. ട്രൈസൈക്ലസോൾ ഒരു തവണ ഉപയോഗിക്കുക. അടുത്ത സീസണിൽ ബ്ലാസ്റ്റ്-പ്രതിരോധ ശേഷിയുള്ള ഇനങ്ങൾ തിരഞ്ഞെടുക്കുക.',
            'normal':    'ട്രൈസൈക്ലസോൾ ഫംഗിസൈഡ് ഉടൻ ഉപയോഗിക്കുക. നൈട്രജൻ വളം കുറയ്ക്കുക. ബ്ലാസ്റ്റ്-പ്രതിരോധ ശേഷിയുള്ള ഇനങ്ങൾ തിരഞ്ഞെടുക്കുക.',
        },
        'Leaf scald': {
            'rainy':     '🌧️ രോഗബാധിത ഇലകൾ ഇന്ന് തന്നെ നീക്കം ചെയ്ത് നശിപ്പിക്കുക. മഴ ശമിച്ചശേഷം പ്രൊപ്പിക്കൊണസോൾ (Propiconazole) ഉപയോഗിക്കുക. വായുസഞ്ചാരം ഉറപ്പാക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — രോഗവ്യാപനം അതിവേഗത്തിലാകാൻ സാധ്യതയുണ്ട്. പ്രൊപ്പിക്കൊണസോൾ അല്ലെങ്കിൽ ടെബുകോണസോൾ വൈകുന്നേരം തളിക്കുക.',
            'humid':     f'💧 {h}% — ഫംഗൽ ഭീഷണി. ബാധിത ഭാഗങ്ങൾ നീക്കം ചെയ്ത് പ്രൊപ്പിക്കൊണസോൾ വൈകുന്നേരം തളിക്കുക.',
            'hot':       f'🌡️ {t}°C — ഇലകൾ കരിയാൻ സാധ്യതയുണ്ട്. പ്രൊപ്പിക്കൊണസോൾ ഉടൻ ഉപയോഗിക്കുക; ഉച്ചസമയത്തെ നനയ്ക്കൽ ഒഴിവാക്കുക.',
            'cool_dry':  f'❄️ {t}°C — ഫംഗൽ ആക്ടിവിറ്റി കുറവാണ്. ബാധിത ഭാഗങ്ങൾ നീക്കം ചെയ്ത് ഒരു ഫംഗിസൈഡ് തളിക്കുക.',
            'normal':    'ബാധിത ഭാഗങ്ങൾ നീക്കം ചെയ്ത് പ്രൊപ്പിക്കൊണസോൾ ഉപയോഗിക്കുക. ചെടികൾ തമ്മിലുള്ള അകലം വർദ്ധിപ്പിക്കുക.',
        },
        'Narrow Brown Leaf Spot': {
            'rainy':     '🌧️ ഫംഗൽ സ്‌പോറുകൾ മഴവെള്ളത്തിലൂടെ പടരാം. ഇന്ന് ഫംഗിസൈഡ് ഒഴിവാക്കുക; മഴ ശമിച്ചശേഷം മാൻകോസേബ് ഉപയോഗിക്കുക. ജലനിർഗ്ഗമനം ഉറപ്പാക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ഫംഗൽ ഭീഷണി. ഉടൻ മാൻകോസേബ് അല്ലെങ്കിൽ ഐപ്രോഡിയോൺ (Iprodione) വൈകുന്നേരം ഉപയോഗിക്കുക. പൊട്ടാഷ്, ഫോസ്ഫേറ്റ് വളങ്ങൾ നൽകുക.',
            'humid':     f'💧 {h}% — ഫംഗൽ ഭീഷണി. മാൻകോസേബ് (2.5 g/L) വൈകുന്നേരം ഉപയോഗിക്കുക.',
            'hot':       f'🌡️ {t}°C — ഫംഗൽ വ്യാപനത്തിന് സാധ്യത. രാവിലെ ഫംഗിസൈഡ് ഉപയോഗിക്കുക. അടുത്ത സീസണിൽ ആരോഗ്യകരമായ വിത്തുകൾ തിരഞ്ഞെടുക്കുക.',
            'cool_dry':  f'❄️ {t}°C — ഫംഗൽ ഭീഷണി കുറവാണ്. ഒരു ഫംഗിസൈഡ് ഉപയോഗിക്കുക. വിത്ത് ഗുണമേന്മയുള്ളതാണെന്ന് ഉറപ്പാക്കുക.',
            'normal':    'മാൻകോസേബ് അല്ലെങ്കിൽ ഐപ്രോഡിയോൺ ഫംഗിസൈഡ് ഉപയോഗിക്കുക. പൊട്ടാഷ്, ഫോസ്ഫേറ്റ് വളങ്ങൾ ഉചിതമായ അളവിൽ നൽകുക.',
        },
        'Rice Hispa': {
            'rainy':     '🌧️ ഇന്ന് കീടനാശിനി പ്രയോഗിക്കുന്നത് ഒഴിവാക്കുക. ബാധിത ഇലകൾ കൈകൊണ്ട് പറിച്ചു നശിപ്പിക്കുക. മഴ ശമിച്ചശേഷം ക്ലോർ‌പൈ‌രിഫോസ് ഉപയോഗിക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — ഹിസ്പ വണ്ടുകൾ സജീവമായ അവസ്ഥയാണ്. ഉടൻ ക്ലോർ‌പൈ‌രിഫോസ് തളിക്കുക. ഞാറ്‌ നേരത്തേ നടുന്നത് ഭാവിയിൽ ഭീഷണി കുറയ്ക്കാൻ സഹായിക്കും.',
            'humid':     f'💧 {h}% — കീടങ്ങൾ സജീവമാകാൻ സാധ്യത. ഫിപ്രോണിൽ അല്ലെങ്കിൽ ക്ലോർ‌പൈ‌രിഫോസ് ഉപയോഗിക്കുക. ബാധിത ഇലകൾ ആദ്യം നീക്കം ചെയ്യുക.',
            'hot':       f'🌡️ {t}°C — കീടങ്ങൾ സജീവം. ഉച്ചസമയത്ത് ഒഴിവാക്കി രാവിലെ അല്ലെങ്കിൽ വൈകുന്നേരം കീടനാശിനി ഉപയോഗിക്കുക.',
            'cool_dry':  f'❄️ {t}°C — കീടാക്രമണം കുറവാണ്. ബാധിത ഇലകൾ നീക്കം ചെയ്ത് ആവശ്യമെങ്കിൽ കീടനാശിനി പ്രയോഗിക്കുക.',
            'normal':    'ക്ലോർ‌പൈ‌രിഫോസ് അല്ലെങ്കിൽ ഫിപ്രോണിൽ കീടനാശിനി ഉപയോഗിക്കുക. ബാധിത ഇലകൾ കൈകൊണ്ട് പറിച്ചു നശിപ്പിക്കുക. ഞാറു നടുന്നത് നേരത്തേയാക്കുക.',
        },
        'Sheath Blight': {
            'rainy':     '🌧️ പോള രോഗം അല്ലെങ്കിൽ ഷീത് ബ്ലൈറ്റ് മഴയിലൂടെ വേഗം പടരാം. ഇന്ന് ഫംഗിസൈഡ് ഒഴിവാക്കുക; മഴ ശമിച്ചശേഷം ഹെക്‌സ‌കൊ‌ണ‌സോൾ ഉപയോഗിക്കുക.',
            'hot_humid': f'🌡️💧 {t}°C / {h}% — രോഗബാധ ഏറ്റവും ഗുരുതരമായേക്കാം. ഉടൻ ഹെക്‌സ‌കൊ‌ണ‌സോൾ അല്ലെങ്കിൽ ട്രൈക്കോഡെർമ (Trichoderma) ഉപയോഗിക്കുക. നൈട്രജൻ വളങ്ങൾ നിർത്തുക.',
            'humid':     f'💧 {h}% — ഫംഗൽ ഭീഷണി. ഹെക്‌സ‌കൊ‌ണ‌സോൾ അല്ലെങ്കിൽ ട്രൈക്കോഡെർമ ഉപയോഗിക്കുക. വായുസഞ്ചാരം ഉറപ്പാക്കാൻ ചെടി നിബിഡത കുറയ്ക്കുക.',
            'hot':       f'🌡️ {t}°C — ഫംഗസ് സജീവമാണ്. ഉടൻ ഹെക്‌സ‌കൊ‌ണ‌സോൾ ഉപയോഗിക്കുക. ചെടികൾ തിങ്ങിനിറഞ്ഞു നിൽക്കുന്നത് ഒഴിവാക്കുക.',
            'cool_dry':  f'❄️ {t}°C — ഫംഗൽ ഭീഷണി കുറവാണ്. ഒരു ഫംഗിസൈഡ് ഉപയോഗിക്കുക. വായുസഞ്ചാരം ഉറപ്പാക്കുക.',
            'normal':    'ഹെക്‌സ‌കൊ‌ണ‌സോൾ അല്ലെങ്കിൽ ട്രൈക്കോഡെർമ ഉപയോഗിക്കുക. ചെടികൾ തിങ്ങിനിറഞ്ഞു നിൽക്കുന്നത് ഒഴിവാക്കുക. നൈട്രജൻ വളത്തിന്റെ അളവ് നിയന്ത്രിക്കുക.',
        },
    }

    disease_matrix = matrix.get(disease)
    if not disease_matrix:
        return None
    return disease_matrix.get(scenario, disease_matrix.get('normal', None))

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# --- AUTH ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        # user[2] is the hashed password
        if user and check_password_hash(user[2], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('തെറ്റായ യൂസർനെയിം അല്ലെങ്കിൽ പാസ്‌വേഡ് (Invalid Username or Password)')
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Check if user already exists
        c.execute('SELECT * FROM users WHERE username = ?', (username,))
        if c.fetchone():
            flash('യൂസർനെയിം നിലവിലുണ്ട് (Username already exists)')
            conn.close()
            return render_template('signup.html')
            
        hashed_pw = generate_password_hash(password)
        try:
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
            conn.commit()
            flash('അക്കൗണ്ട് വിജയകരമായി നിർമ്മിച്ചു! ദയവായി ലോഗിൻ ചെയ്യുക.')
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('സെർവർ തകരാർ (Server error)')
            conn.close()
            
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({'error': f'Failed to save file: {str(e)}'})
        
        if model:
            try:
                img_array = prepare_image(filepath)
                predictions = model.predict(img_array)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = float(100 * np.max(predictions))
                
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

                recommendations = {
                    'Bacterial Leaf Blight': 'ചെമ്പ് അടങ്ങിയ കുമിൾനാശിനികൾ (കോപ്പർ ഓക്‌സിക്ലോറൈഡ്) തളിക്കുക. രോഗബാധിത ചെടികൾ നീക്കം ചെയ്യുക. വയലിൽ വെള്ളം കെട്ടിനിൽക്കാതെ ശ്രദ്ധിക്കുക. നൈട്രജൻ വളം ആവശ്യത്തിൽ കൂടുതൽ ഉപയോഗിക്കരുത്.',
                    'Brown Spot': 'മാൻകോസേബ് അല്ലെങ്കിൽ ട്രൈസൈക്ലസോൾ കൊണ്ടുള്ള കുമിൾനാശിനി തളിക്കുക. സിലിക്കൺ, പൊട്ടാഷ് എന്നിവ അടങ്ങിയ വളങ്ങൾ ഉപയോഗിക്കുക. വിതക്കുന്നതിന് മുൻപ് വിത്ത് ശുദ്ധീകരിക്കുക.',
                    'Healthy Rice Leaf': 'ചെടി ആരോഗ്യത്തോടെ ഇരിക്കുന്നു! പതിവ് നനയ്ക്കൽ, വളം ഇടൽ, കളനാശിനി ഉപയോഗം തുടരുക. മുൻകരുതൽ ആയി ആഴ്ചയിൽ ഒരിക്കൽ ചെടി നിരീക്ഷിക്കുക.',
                    'Leaf Blast': 'ട്രൈസൈക്ലസോൾ (Tricyclazole) കൊണ്ടുള്ള കുമിൾനാശിനി ഉടൻ തളിക്കുക. നൈട്രജൻ വളം കുറയ്ക്കുക. ബ്ലാസ്റ്റ്-പ്രതിരോധ ശേഷിയുള്ള ഇനങ്ങൾ തിരഞ്ഞെടുക്കുക.',
                    'Leaf scald': 'ബാധിത ഭാഗങ്ങൾ നീക്കം ചെയ്ത് നശിപ്പിക്കുക. കുമിൾനാശിനി (ഉദാ: Propiconazole) തളിക്കുക. സസ്യങ്ങൾ തമ്മിലുള്ള അകലം വർദ്ധിപ്പിക്കുക.',
                    'Narrow Brown Leaf Spot': 'കുമിൾനാശിനി (മാൻകോസേബ്) തളിക്കുക. പൊട്ടാഷ്, ഫോസ്ഫറസ് വളങ്ങൾ ഉചിതമായ അളവിൽ ഉപയോഗിക്കുക. ആരോഗ്യകരമായ ഇനം വിത്തുകൾ ഉപയോഗിക്കുക.',
                    'Rice Hispa': 'ക്ലോറോപൈരിഫോസ് അല്ലെങ്കിൽ ഫിപ്രോണിൽ കൊണ്ടുള്ള കീടനാശിനി തളിക്കുക. ബാധിത ഇലകൾ കൈകൊണ്ട് പറിച്ചു നീക്കി നശിപ്പിക്കുക. ഞാറുനടുന്ന സമ൯ം നേരത്തെ ആക്കുക.',
                    'Sheath Blight': 'ഹെക്‌സകൊണസോൾ (Hexaconazole) അല്ലെങ്കിൽ ഇൻ‌ഡ്യൂസ്ഡ് കോന്ത്ര കുമിൾ (Trichoderma) ഉപയോഗിക്കുക. സസ്യനിബിഡത കുറയ്ക്കുക. നൈട്രജൻ വളം ശ്രദ്ധിച്ചു ഉപയോഗിക്കുക.'
                }

                translated_class = translations.get(predicted_class, predicted_class)

                # Get weather context sent by frontend
                weather_rec = None
                try:
                    temp_c = float(request.form.get('temp_c', 0)) or None
                    humidity = float(request.form.get('humidity', 0)) or None
                    is_rain = request.form.get('is_rain', 'false').lower() == 'true'
                    if temp_c is not None or is_rain:
                        weather_rec = get_weather_integrated_recommendation(
                            predicted_class, temp_c, humidity, is_rain
                        )
                except Exception:
                    pass

                # Use weather-fused recommendation if available, else base fallback
                recommendation = weather_rec if weather_rec else recommendations.get(
                    predicted_class, 'ഒരു കൃഷി വിദഗ്ദ്ധനെ സമീപിക്കുക.'
                )

                return jsonify({
                    'class': translated_class,
                    'confidence': confidence,
                    'image_url': f'/static/uploads/{filename}',
                    'recommendation': recommendation
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Error processing image: {str(e)}'})
        else:
            return jsonify({'error': 'Model not loaded'})

    return jsonify({'error': 'Invalid file type'})

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
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html')

@app.route('/contact')
def contact():
    # Allow public access or protect it, let's protect to be safe since it's an app for farmers
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('contact.html')

@app.route('/api/messages', methods=['GET'])
def get_messages():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(load_messages())

@app.route('/api/messages', methods=['POST'])
def post_message():
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid data'}), 400
        
    # Ignore username provided by frontend, force the session username
    sender_username = session['username']
    msg = save_message(sender_username, data['text'])
    return jsonify(msg)

if __name__ == '__main__':
    app.run(debug=True)
