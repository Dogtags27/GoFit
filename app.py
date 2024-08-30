from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
import numpy as np
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


def download_model_from_dropbox(dropbox_url, local_file_path):
    # Download the file from Dropbox
    response = requests.get(dropbox_url, stream=True)
    if response.status_code == 200:
        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Model downloaded successfully and saved to {local_file_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def load_model_from_file(local_file_path):
    # Load the model from the file
    if os.path.exists(local_file_path):
        model = load_model(local_file_path)
        print("Model loaded successfully.")
        return model
    else:
        print(f"File not found: {local_file_path}")
        return None

# Define the Dropbox URL and local file path
dropbox_url = 'https://www.dropbox.com/scl/fi/yvuqg7l5lo6gk8ccys0ow/FoodClassifier.h5?rlkey=bixec1rz520g4y0pklko0aave&st=y7ah7xgf&dl=1'
local_file_path = 'FoodClassifier.h5'

# Download and load the model
download_model_from_dropbox(dropbox_url, local_file_path)
model = load_model_from_file(local_file_path)

# Now you can use 'model' in your Flask API


################################
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
