"""
FastAPI application for IRIS classification prediction.
Loads model from MLflow and serves predictions via REST API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import mlflow.sklearn
import numpy as np
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IRIS Classification API",
    description="ML-powered API for classifying IRIS flowers using MLflow models",
    version="1.0.0"
)

# Global model variable
model = None
model_info = {}


class IrisFeatures(BaseModel):
    """Input features for IRIS prediction"""
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_info: Dict


def load_model_from_mlflow():
    """Load model from MLflow registry or local artifacts"""
    global model, model_info
    
    try:
        # Try to load from MLflow registry (Production stage)
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///./mlruns")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        model_name = os.getenv("MODEL_NAME", "iris-classifier")
        model_stage = os.getenv("MODEL_STAGE", "Production")
        
        logger.info(f"Attempting to load model '{model_name}' from stage '{model_stage}'")
        
        try:
            model_uri = f"models:/{model_name}/{model_stage}"
            model = mlflow.sklearn.load_model(model_uri)
            model_info = {
                "source": "mlflow_registry",
                "model_name": model_name,
                "stage": model_stage,
                "tracking_uri": mlflow_tracking_uri
            }
            logger.info(f"✓ Model loaded from MLflow registry: {model_uri}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load from MLflow registry: {e}")
            
            # Fallback: Try to load from local mlruns
            logger.info("Attempting to load from local mlruns...")
            
            # Get the latest run from the default experiment
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("iris-classification")
            
            if experiment:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
                if runs:
                    run = runs[0]
                    model_uri = f"runs:/{run.info.run_id}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    model_info = {
                        "source": "mlflow_local",
                        "run_id": run.info.run_id,
                        "experiment_id": experiment.experiment_id
                    }
                    logger.info(f"✓ Model loaded from local run: {run.info.run_id}")
                    return True
            
            # Final fallback: Train a simple model
            logger.warning("No model found in MLflow, training a default model...")
            return load_default_model()
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return load_default_model()


def load_default_model():
    """Train and load a simple default model as fallback"""
    global model, model_info
    
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_iris
        
        logger.info("Training default model...")
        iris = load_iris()
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(iris.data, iris.target)
        
        model_info = {
            "source": "default_trained",
            "note": "Fallback model trained on startup"
        }
        logger.info("✓ Default model trained and loaded")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    logger.info("Starting IRIS Classification API...")
    success = load_model_from_mlflow()
    
    if not success:
        logger.error("Failed to load any model!")
    else:
        logger.info("API ready to serve predictions")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "IRIS Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/batch-predict (POST)",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint for Kubernetes probes"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_info": model_info
    }


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_info": model_info,
        "model_type": str(type(model).__name__),
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": ["setosa", "versicolor", "virginica"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: IrisFeatures):
    """
    Predict IRIS flower species from input features
    
    Args:
        features: IRIS flower measurements
        
    Returns:
        Prediction with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Map to class names
        class_names = ["setosa", "versicolor", "virginica"]
        predicted_class = class_names[prediction]
        confidence = float(probabilities[prediction])
        
        prob_dict = {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.3f})")
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_version": model_info.get("model_name", "default"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(features_list: List[IrisFeatures]):
    """
    Predict multiple IRIS samples in batch
    
    Args:
        features_list: List of IRIS flower measurements
        
    Returns:
        List of predictions with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(features_list) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 samples")
    
    try:
        results = []
        
        for features in features_list:
            input_data = np.array([[
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ]])
            
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            class_names = ["setosa", "versicolor", "virginica"]
            predicted_class = class_names[prediction]
            confidence = float(probabilities[prediction])
            
            prob_dict = {
                class_names[i]: float(probabilities[i])
                for i in range(len(class_names))
            }
            
            results.append({
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": prob_dict
            })
        
        logger.info(f"Batch prediction completed: {len(results)} samples")
        
        return {
            "predictions": results,
            "count": len(results),
            "model_version": model_info.get("model_name", "default"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics (for Prometheus/monitoring)"""
    return {
        "model_loaded": model is not None,
        "model_source": model_info.get("source", "unknown"),
        "uptime": "healthy"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )