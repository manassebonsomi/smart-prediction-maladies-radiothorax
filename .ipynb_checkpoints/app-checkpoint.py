
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing import image
#from PIL import Image
app = Flask(__name__)

# Charger le modèle Xception
MODEL_PATH = "xception_radiotherapy_model.h5"
model = load_model(MODEL_PATH)

# Définir les classes
CLASS_NAMES = ['covid19', 'pneumonie', 'normal', 'tuberculose']

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Taille adaptée à Xception
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisation

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    accuracy = round(100 * np.max(predictions), 2)

    return predicted_class, accuracy


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        predicted_class, accuracy = predict_image(file_path)

        return render_template('result.html', image_url=file_path, label=predicted_class, accuracy=accuracy)


if __name__ == '__main__':
    app.run(debug=True)
