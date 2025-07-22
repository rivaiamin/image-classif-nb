#!/usr/bin/env python3
"""
Web Application for Hawar Daun Bakteri Classifier
Reuses existing code from main.py and feature_extraction.py
"""

import os
import sys
import cv2
import numpy as np
import joblib
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Import existing modules
from feature_extraction import FeatureExtractor, log_feature_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

app.config['MODEL_PATH'] = 'models/model_enhanced.pkl'
app.config['LOG_FILE'] = 'logs/training_enhanced_log.txt'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and feature extractor
model = None
feature_extractor = None

def load_model():
    """Load the trained model and feature extractor - reuses existing code"""
    global model, feature_extractor
    
    try:
        model_path = app.config['MODEL_PATH']
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            feature_extractor = FeatureExtractor()
            logger.info("Model loaded successfully")
            return True
        else:
            logger.error(f"Model file not found: {model_path}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_single_image(image_path):
    """
    Classify a single image using the existing feature extraction code
    This reuses the logic from main.py but for a single image
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Could not load image"
        
        # Extract features using existing FeatureExtractor
        features, feature_names = feature_extractor.extract_all_features(image)
        
        # Make prediction using the loaded model
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        
        # Get confidence score
        confidence = max(probabilities)
        
        # Map prediction to result (reusing logic from main.py)
        if prediction == 1:
            result = "Hawar Daun Bakteri"
            confidence_percent = probabilities[1] * 100
        else:
            result = "Bukan Hawar Daun Bakteri"
            confidence_percent = probabilities[0] * 100
        
        # Log the classification (reusing existing logging)
        log_feature_info(f"Web classification: {result} with {confidence_percent:.1f}% confidence")
        
        return {
            'classification': result,
            'confidence': confidence_percent,
            'probabilities': {
                'hawar_daun_bakteri': probabilities[1] * 100,
                'bukan_hawar_daun_bakteri': probabilities[0] * 100
            },
            'feature_count': len(features),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        return None, f"Error during classification: {str(e)}"

@app.route('/')
def index():
    """Render the main page using Jinja2 templates"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """Handle image upload and classification - reuses existing logic"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diupload'}), 400
        
        file = request.files['image']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format file tidak didukung. Gunakan PNG, JPG, atau JPEG'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Classify the image using existing logic
        result = classify_single_image(filepath)
        
        if result is None:
            return jsonify({'error': 'Gagal menganalisis gambar'}), 500
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Return JSON response for Alpine.js
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in classify endpoint: {e}")
        return jsonify({'error': 'Terjadi kesalahan dalam memproses gambar'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info')
def model_info():
    """Get model information for debugging"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'Gaussian Naive Bayes',
        'classes': model.classes_.tolist(),
        'class_priors': model.class_prior_.tolist(),
        'feature_count': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'Unknown'
    })

def main():
    """Main function to run the web application"""
    print("=" * 60)
    print("Hawar Daun Bakteri Classifier - Web Application")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(app.config['MODEL_PATH']):
        print("❌ Error: Model file not found!")
        print("Please run the training script first:")
        print("  python main.py")
        sys.exit(1)
    
    # Load model
    print("📦 Loading model...")
    if load_model():
        print("✅ Model loaded successfully!")
    else:
        print("❌ Failed to load model!")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main() 