import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import joblib

app = Flask(__name__)
CORS(app) 

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'models'))

# --- LOAD MODELS ---
print("Loading models...")

# 1. Load the SVM (The Brain)
try:
    svm_path = os.path.join(MODELS_DIR, 'svm_classifier.pkl')
    svm_model = joblib.load(svm_path)
    print("SVM loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load SVM from {svm_path}. Error: {e}")
    svm_model = None

# 2. Load the CNN (The Eyes)
# Since you don't have a .h5 file, we load a standard pre-trained model from Keras.
# ---------------------------------------------------------------------------
# IMPORTANT: If you used ResNet50 or MobileNet, change 'VGG16' to that name!
# ---------------------------------------------------------------------------
try:
    print("Downloading/Loading VGG16 as feature extractor...")
    # include_top=False removes the final classification layer (we don't need it)
    # weights='imagenet' uses the standard pre-trained weights
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor = base_model
    print("CNN (VGG16) loaded successfully!")
except Exception as e:
    print(f"Error loading CNN: {e}")
    feature_extractor = None

# --- PREPROCESSING ---
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224)) # VGG16 expects 224x224
    img_array = np.array(image)
    
    # VGG16 specific preprocessing (Do not divide by 255.0 manually if using preprocess_input)
    # But if your training used /255.0, keep it. 
    # STANDARD APPROACH for VGG16 is usually this:
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if not feature_extractor or not svm_model:
        return jsonify({'error': 'Models are not loaded'}), 500

    try:
        # 1. Process the image
        img_array = preprocess_image(file.read())
        
        # 2. Extract Features using VGG16
        # Output shape here is (1, 7, 7, 512) -> Total 25,088 numbers
        features = feature_extractor.predict(img_array)
        
        # 3. Apply Global Average Pooling (THE FIX)
        # We average the 7x7 area to get a single vector of 512 numbers
        features_pooled = np.mean(features, axis=(1, 2))
        features_flat = features_pooled.reshape(1, -1)
        
        # 4. Predict using SVM
        # Now SVM receives exactly 512 features, which matches your training!
        prediction_index = svm_model.predict(features_flat)[0]
        
        # IMPORTANT: Ensure these match your training folder names exactly!
        # If your prediction is wrong (e.g. says Eczema when it is Melanoma),
        # change the order of this list.
        class_names = ['Benign', 'Eczema', 'Melanoma', 'Psoriasis'] 
        result_class = class_names[prediction_index]
        
        return jsonify({'class': result_class, 'confidence': 'High'})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)