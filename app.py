from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from recommendation_service import RecommendationService
from predict import predict_image, read_imagefile

app = FastAPI()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommendation_service import RecommendationService

app = FastAPI()

class RecommendationRequest(BaseModel):
    skin_type: str
    issue: str

class RecommendationResponse(BaseModel):
    recommendations: list

recommendation_service = RecommendationService('Product.csv')

@app.post("/recommend/", response_model=RecommendationResponse)
async def recommend_products(request: RecommendationRequest):
    try:
        recommendations = recommendation_service.get_recommendations(request.skin_type, request.issue)
        return RecommendationResponse(recommendations=recommendations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image_array = read_imagefile(image_data)
        prediction = predict_image(image_array)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
