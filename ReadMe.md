Face Emotion Detection Web App
A simple Flask-based web application that detects facial emotions from uploaded images using a trained CNN model.

Features
Upload a face image via a web form.

Detect emotions such as happy, sad, angry, surprise, fear, disgust, and neutral.

Save user's name, email, detected emotion, and image path in an SQLite database.

Simple, visually appealing HTML/CSS interface.

Ready for deployment on Render.

Project Structure

FACE_DETECTION/
├── app.py
├── model_training.py
├── face_emotionModel.h5
├── requirements.txt
├── .gitignore
├── database.db
├── link_web_app.txt
├── templates/
│   └── index.html
├── uploads/
└── data/


Getting Started
Prerequisites
Python 3.9 or 3.10 (not 3.13)

pip for package management


Setup Steps
Clone this repository

bash
git clone https://github.com/yourusername/your-repo-name.git

cd FACE_DETECTION

Create and activate a virtual environment

bash
python3.10 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
Install dependencies

bash
pip install -r requirements.txt
(Optional) Train the model

Place your dataset images in data/ subdirectories, one per emotion.

Edit and run model_training.py to generate face_emotionModel.h5.

Run the web app locally

bash
python app.py
Access the app at http://127.0.0.1:5000/

Deployment
Push your code to GitHub.


Deploy on Render with:

Build command: pip install -r requirements.txt

Start command: gunicorn app:app

Save your web app link in link_web_app.txt.

Contributing
Pull requests are welcome!

License
MIT"# FACE-EMOTION-DETECTOR" 
