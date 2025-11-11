import os
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sqlite3

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Emotion labels (use the same as your training classes)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model('face_emotionModel.h5')

# Helper: Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper: Connect to DB
def insert_data(name, email, img_path, detected_emotion):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT, email TEXT, img_path TEXT, emotion TEXT)''')
    c.execute('INSERT INTO users (name, email, img_path, emotion) VALUES (?, ?, ?, ?)',
              (name, email, img_path, detected_emotion))
    conn.commit()
    conn.close()

# Processing image to predict emotion
def predict_emotion(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    emotion = emotion_labels[np.argmax(preds)]
    return emotion

# Greeting by emotion
def emotion_message(emotion):
    messages = {
        'angry': "You look angry. What's upsetting you?",
        'disgust': "You look disgusted. Is something unpleasant?",
        'fear': "You look scared. What's worrying you?",
        'happy': "You're smiling! Glad you're happy!",
        'sad': "You are frowning. Why are you sad?",
        'surprise': "You seem surprised!",
        'neutral': "Your expression is neutral."
    }
    return messages.get(emotion, "Emotion detected: " + emotion)

# Main form route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        file = request.files['image']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            emotion = predict_emotion(img_path)
            insert_data(name, email, img_path, emotion)
            message = emotion_message(emotion)
            return render_template('index.html', message=message)
        else:
            return render_template('index.html', message="Invalid file format.")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)