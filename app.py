from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict
import tensorflow as tf
from PIL import Image
import numpy as np
from recommendation_service import RecommendationService
from predict import predict, read_imagefile

app = FastAPI()

model = tf.keras.models.load_model('models/model.h5')
recommendation_service = RecommendationService('Product.csv')

class PredictionRequest(BaseModel):
    answers: Dict[int, str]
    issue: str

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    prediction = predict(image, model)
    return JSONResponse(content={"prediction": str(prediction)})

@app.post("/recommend/")
async def get_recommendation(request: PredictionRequest):
    try:
        recommendations = recommendation_service.get_recommendations(request.answers, request.issue)
        return JSONResponse(content={"recommendations": recommendations})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

