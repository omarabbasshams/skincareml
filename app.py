from fastapi import FastAPI, HTTPException , File, UploadFile
from pydantic import BaseModel
from recommendation_service import get_recommendations
from pydantic import BaseModel
from typing import Dict
import tensorflow as tf
from PIL import Image
import numpy as np
from predict import predict, read_imagefile
from fastapi.responses import JSONResponse
app = FastAPI()

class RecommendationRequest(BaseModel):
    skin_type: str
    issue: str

class RecommendationResponse(BaseModel):
    recommendations: list




class PredictionRequest(BaseModel):
    answers: Dict[int, str]
    issue: str

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return JSONResponse(content={"prediction": str(prediction)})

@app.post("/recommend/", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    try:
        recommendations = get_recommendations(request.skin_type, request.issue)
        if not recommendations:
            print(f"No recommendations found for skin_type: {request.skin_type} and issue: {request.issue}")  # Log no recommendations case
        return {"recommendations": recommendations}
    except Exception as e:
        print(f"Error: {str(e)}")  # Log any errors
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
