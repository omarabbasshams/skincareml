from fastapi import FastAPI, File, UploadFile , HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from pydantic import BaseModel
from recommendation_service import get_recommendations

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('models/model.h5')

def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

def predict(image: Image.Image):
    image = image.resize((224, 224))  # Ensure the image size matches your model input
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return JSONResponse(content={"prediction": int(prediction)})
class RecommendationRequest(BaseModel):
    skin_type: str
    issue: str

class RecommendationResponse(BaseModel):
    recommendations: list

@app.post("/recommendations/", response_model=RecommendationResponse)
async def recommend_products(request: RecommendationRequest):
    try:
        recommendations = get_recommendations(request.skin_type, request.issue)
        return RecommendationResponse(recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
