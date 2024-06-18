import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List

app = FastAPI()

# Load the CSV file
file_path = 'Product.csv'
products_df = pd.read_csv(file_path)

# Define the request model
class RecommendationRequest(BaseModel):
    answers: Dict[int, str]
    prediction: str

# Function to get recommendations
def get_recommendations(skin_type: str, prediction: str) -> List[str]:
    # Filter products based on skin type
    filtered_products = products_df[products_df['SkinType'].str.contains(skin_type, case=False, na=False)]
    
    # Match the prediction to the relevant column
    recommendation_column = prediction.lower()
    if recommendation_column in filtered_products.columns:
        recommendations = filtered_products[recommendation_column].dropna().tolist()
    else:
        recommendations = []
    
    return recommendations

@app.post("/recommend/")
async def recommend(request: RecommendationRequest):
    skin_type_answer = request.answers.get(1, "").lower()  # Assume question ID 1 is for skin type
    if not skin_type_answer:
        raise HTTPException(status_code=400, detail="Skin type answer is missing")

    recommendations = get_recommendations(skin_type_answer, request.prediction)
    return {"recommendations": recommendations}
