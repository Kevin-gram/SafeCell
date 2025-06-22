from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from keras.models import load_model
import numpy as np
import uvicorn
import os
import logging
from PIL import Image
import io
from typing import Dict, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Global model variable
model = None

# Constants
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def load_malaria_model():
    """Load the malaria detection model"""
    global model
    try:
        MODEL_PATH = os.path.join("model", "saved_models", "malaria_model.h5")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
        model = load_model(MODEL_PATH)
        logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def calculate_target_size(expected_features: int, channels: int = 3) -> tuple:
    """Calculate target image size based on expected features"""
    pixels_needed = expected_features // channels
    side_length = int(np.sqrt(pixels_needed))
    
    # Try common sizes
    common_sizes = [(64, 64), (112, 112), (128, 128), (150, 150), (224, 224)]
    
    for size in common_sizes:
        if size[0] * size[1] * channels == expected_features:
            return size
    
    return (side_length, pixels_needed // side_length)

def preprocess_image(file_content: bytes) -> np.ndarray:
    """Preprocess image for model prediction"""
    try:
        # Open and convert image
        img = Image.open(io.BytesIO(file_content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Determine target size and processing method
        if len(model.input_shape) == 4:  # 2D input (batch, height, width, channels)
            target_size = (model.input_shape[1], model.input_shape[2])
            flatten = False
        else:  # Flattened input (batch, features)
            expected_features = model.input_shape[1]
            target_size = calculate_target_size(expected_features, 3)
            flatten = True
        
        # Resize and process
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        if flatten:
            img_array = img_array.reshape(1, -1)
        
        img_array /= 255.0  # Normalize
        
        logger.info(f"Processed image shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    if not load_malaria_model():
        raise RuntimeError("Could not load malaria detection model")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Predict malaria from uploaded cell image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate file
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        # Read and validate file content
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Process image
        img_array = preprocess_image(file_content)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Interpret results
        logger.info(f"Raw prediction output: {prediction}")
        logger.info(f"Prediction shape: {prediction.shape}")
        
        if prediction.shape[1] == 1:  # Single output (binary classification)
            confidence_score = float(prediction[0][0])
            
            # Based on your notebook: class_names = ['Parasitized', 'Uninfected']
            # Single output typically gives probability of the positive class (class 1)
            # So confidence_score is probability of "Uninfected" (class 1)
            result = "Uninfected" if confidence_score > 0.5 else "Parasitized"
            confidence = confidence_score if confidence_score > 0.5 else 1 - confidence_score
            
        else:  # Two outputs [parasitized_prob, uninfected_prob]
            prob_parasitized = float(prediction[0][0])  # Class 0: Parasitized
            prob_uninfected = float(prediction[0][1])   # Class 1: Uninfected
            
            # Log both probabilities to understand the model output
            logger.info(f"Parasitized probability (Class 0): {prob_parasitized}")
            logger.info(f"Uninfected probability (Class 1): {prob_uninfected}")
            
            # Based on your training: class_names = ['Parasitized', 'Uninfected']
            if prob_parasitized > prob_uninfected:
                result = "Parasitized"
                confidence = prob_parasitized
            else:
                result = "Uninfected"
                confidence = prob_uninfected
        
        return JSONResponse(
            status_code=200,
            content={
                "result": result,
                "confidence": round(confidence, 4),
                "confidence_percentage": f"{round(confidence * 100, 2)}%"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)