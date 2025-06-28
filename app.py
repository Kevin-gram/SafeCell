from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import numpy as np
import uvicorn
import os
import logging
from PIL import Image
import io
from typing import Dict, Any, List
import traceback
from pymongo import MongoClient
from pydantic import BaseModel
from bson import ObjectId
import json
from dotenv import load_dotenv
from datetime import datetime, timezone
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


model = None
db = None


ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


MONGODB_URL = os.getenv("MONGODB_URL")

class PredictionResults(BaseModel):
    result: str
    confidenceLevel: int
    rawResult: str
    rawConfidence: float
    processingTime: int

class DetectionData(BaseModel):
    element: str
    province: str
    district: str
    sector: str
    hospital: str
    predictionResults: PredictionResults
    userId: str

def connect_to_mongodb():
    """Connect to MongoDB"""
    global db
    try:
        
        client = MongoClient(MONGODB_URL)
        db = client["safe_cell_db"]  
      
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        return False

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
    
   
    common_sizes = [(64, 64), (112, 112), (128, 128), (150, 150), (224, 224)]
    
    for size in common_sizes:
        if size[0] * size[1] * channels == expected_features:
            return size
    
    return (side_length, pixels_needed // side_length)

def preprocess_image(file_content: bytes) -> np.ndarray:
    """Preprocess image for model prediction"""
    try:
      
        img = Image.open(io.BytesIO(file_content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
       
        if len(model.input_shape) == 4: 
            target_size = (model.input_shape[1], model.input_shape[2])
            flatten = False
        else:  
            expected_features = model.input_shape[1]
            target_size = calculate_target_size(expected_features, 3)
            flatten = True
        
      
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        if flatten:
            img_array = img_array.reshape(1, -1)
        
        img_array /= 255.0 
        
        logger.info(f"Processed image shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model and connect to database on startup"""
    if not load_malaria_model():
        raise RuntimeError("Could not load malaria detection model")
    
    if not connect_to_mongodb():
        raise RuntimeError("Could not connect to MongoDB")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Predict malaria from uploaded cell image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
    
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file format")
        
        
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
      
        img_array = preprocess_image(file_content)
        
    
        prediction = model.predict(img_array, verbose=0)
        
        
        logger.info(f"Raw prediction output: {prediction}")
        logger.info(f"Prediction shape: {prediction.shape}")
        
        if prediction.shape[1] == 1:  
            confidence_score = float(prediction[0][0])
            
           
            result = "Uninfected" if confidence_score > 0.5 else "Parasitized"
            confidence = confidence_score if confidence_score > 0.5 else 1 - confidence_score
            
        else:  
            prob_parasitized = float(prediction[0][0])  
            prob_uninfected = float(prediction[0][1])   
            
           
            logger.info(f"Parasitized probability (Class 0): {prob_parasitized}")
            logger.info(f"Uninfected probability (Class 1): {prob_uninfected}")
            
          
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

@app.post("/detection-data/")
async def save_detection_data(data: DetectionData):
    """Save combined detection data to MongoDB"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # Convert to dict and add server timestamp
        data_dict = data.dict()
        
        # Add server-side timestamps
        current_time = datetime.now(timezone.utc)
        data_dict["createdAt"] = current_time
        data_dict["updatedAt"] = current_time
        
        # Optional: Add a human-readable timestamp as well
        data_dict["timestamp"] = current_time.isoformat()
        
        collection = db["detection_results"]
        result = collection.insert_one(data_dict)
        
        logger.info(f"Data saved with ID: {result.inserted_id} at {current_time}")
        
        return JSONResponse(
            status_code=201,
            content={
                "message": "Detection data saved successfully",
                "id": str(result.inserted_id),
                "timestamp": current_time.isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error saving detection data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")

@app.get("/detection-data/")
async def get_all_detection_data():
    """Get all detection data from MongoDB"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        collection = db["detection_results"]
        
        # Sort by creation date (newest first)
        cursor = collection.find({}).sort("createdAt", -1)
        data_list = []
        
        for document in cursor:
            document["_id"] = str(document["_id"])
            # Convert datetime objects to ISO strings for JSON serialization
            if "createdAt" in document:
                document["createdAt"] = document["createdAt"].isoformat()
            if "updatedAt" in document:
                document["updatedAt"] = document["updatedAt"].isoformat()
            data_list.append(document)
        
        logger.info(f"Retrieved {len(data_list)} detection records")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Retrieved {len(data_list)} records",
                "data": data_list
            }
        )
        
    except Exception as e:
        logger.error(f"Error retrieving detection data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data: {str(e)}")

@app.get("/detection-data/by-date/")
async def get_detection_data_by_date(
    start_date: str = None,
    end_date: str = None,
    limit: int = 100
):
    """Get detection data filtered by date range"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        collection = db["detection_results"]
        
        # Build query filter
        query_filter = {}
        
        if start_date or end_date:
            date_filter = {}
            
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    date_filter["$gte"] = start_dt
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO format.")
            
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    date_filter["$lte"] = end_dt
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO format.")
            
            query_filter["createdAt"] = date_filter
        
        # Execute query with sorting and limit
        cursor = collection.find(query_filter).sort("createdAt", -1).limit(limit)
        data_list = []
        
        for document in cursor:
            document["_id"] = str(document["_id"])
            # Convert datetime objects to ISO strings for JSON serialization
            if "createdAt" in document:
                document["createdAt"] = document["createdAt"].isoformat()
            if "updatedAt" in document:
                document["updatedAt"] = document["updatedAt"].isoformat()
            data_list.append(document)
        
        logger.info(f"Retrieved {len(data_list)} detection records with date filter")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Retrieved {len(data_list)} records",
                "data": data_list,
                "filter_applied": query_filter != {}
            }
        )
        
    except Exception as e:
        logger.error(f"Error retrieving detection data by date: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data: {str(e)}")

@app.delete("/detection-data/{record_id}")
async def delete_detection_data(record_id: str):
    """Delete a specific detection record by ID"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        collection = db["detection_results"]
        
        # Convert string ID to ObjectId
        object_id = ObjectId(record_id)
        
        # Delete the document
        result = collection.delete_one({"_id": object_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Record not found")
        
        logger.info(f"Deleted record with ID: {record_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Record deleted successfully",
                "deleted_id": record_id
            }
        )
        
    except ObjectId.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid record ID format")
    except Exception as e:
        logger.error(f"Error deleting detection data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete data: {str(e)}")

@app.delete("/detection-data/")
async def delete_all_detection_data():
    """Delete all detection records"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        collection = db["detection_results"]
        
        # Delete all documents
        result = collection.delete_many({})
        
        logger.info(f"Deleted {result.deleted_count} records")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"All records deleted successfully",
                "deleted_count": result.deleted_count
            }
        )
        
    except Exception as e:
        logger.error(f"Error deleting all detection data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete all data: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_loaded": model is not None,
            "database_connected": db is not None
        }
    )

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)