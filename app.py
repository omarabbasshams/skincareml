from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommendation_service import get_recommendations

app = FastAPI()

class RecommendationRequest(BaseModel):
    skin_type: str
    issue: str

class RecommendationResponse(BaseModel):
    recommendations: list



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
