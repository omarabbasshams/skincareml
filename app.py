from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from recommendation_service import get_recommendations
from predict import predict, read_imagefile

app = FastAPI()

class RecommendationRequest(BaseModel):
    answers: dict
    issue: str

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return {"prediction": int(prediction)}

@app.post("/recommend/")
def recommend(request: RecommendationRequest):
    try:
        skin_type = request.answers.get('1')  # Assuming the skin type question has ID '1'
        issue = request.issue

        if not skin_type or not issue:
            raise HTTPException(status_code=400, detail="Missing skin type or issue information")

        recommendations = get_recommendations(skin_type, issue)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
