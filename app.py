from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
import numpy as np
import gdown
import os
from PIL import Image
import io
import requests

# Flask==2.3.3
# tensorflow==2.16.1
# requests==2.28.2
# Pillow==9.4.0
# numpy==1.24.4
# gunicorn==23.0.0

app = Flask(__name__)

model_gdrive_path = "https://drive.google.com/file/d/1KtANvt9plFhGHohOokbR82duH77JbE7i/view?usp=sharing"
local_model_path = "./FoodClassifier.h5"
# Function to download the model if it doesn't exist locally
def download_model():
    if not os.path.exists(local_model_path):
        # url = f'https://drive.google.com/uc?id={model_gdrive_path}'
        gdown.download(model_gdrive_path, local_model_path, quiet=False)
        
def load_model_from_file():
    if not os.path.exists(local_model_path):
        download_model()
    model = load_model(local_model_path)
    return model
# Load the model
model = load_model_from_file()

category={
    0: ['burger','Burger'], 1: ['butter_naan','Butter Naan'], 2: ['chai','Chai'],
    3: ['chapati','Chapati'], 4: ['chole_bhature','Chole Bhature'], 5: ['dal_makhani','Dal Makhani'],
    6: ['dhokla','Dhokla'], 7: ['fried_rice','Fried Rice'], 8: ['idli','Idli'], 9: ['jalegi','Jalebi'],
    10: ['kathi_rolls','Kaathi Rolls'], 11: ['kadai_paneer','Kadai Paneer'], 12: ['kulfi','Kulfi'],
    13: ['masala_dosa','Masala Dosa'], 14: ['momos','Momos'], 15: ['paani_puri','Paani Puri'],
    16: ['pakode','Pakode'], 17: ['pav_bhaji','Pav Bhaji'], 18: ['pizza','Pizza'], 19: ['samosa','Samosa']
}
#Null
def predict_image(filename,model):
    img_ = filename
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0) 
    img_processed /= 255.   
    
    prediction = model.predict(img_processed)
    
    index = np.argmax(prediction)
    
    return category[index][1]
    # plt.title("Prediction - {}".format(category[index][1]))
    # plt.imshow(img_array)

# Define a route for health check (GET)
@app.route('/health', methods=['GET'])
def health_check():
    return "Model is loaded and ready to use", 200

# Define a route to make predictions (POST)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    image_url = data['url']

    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()  # Check if the request was successful
        img = Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    except IOError:
        return jsonify({"error": "Invalid image format"}), 400
    
    predicted_food = predict_image(img,model)
    print(predicted_food)
    return jsonify({"prediction": predicted_food}), 200

# Define a route to make predictions using a local file path (POST)
@app.route('/predict_path', methods=['POST'])
def predict_path():
    # data = request.get_json()

    if 'file_path' not in request.files:
        return jsonify({"error": "No file path provided"}), 400

    file_path = request.files['file_path']

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 400

    img = Image.open(file_path)
    predicted_food = predict_image(img,model)
    print(predicted_food)
    return jsonify({"prediction": predicted_food}), 200

if __name__ == '__main__':
    app.run(debug=True)
