import uvicorn
from fastapi import FastAPI, File, UploadFile
from predict import predict_image, read_imagefile
from recommendation_service import get_recommendations

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = read_imagefile(await file.read())
    prediction = predict_image(image)
    return {"prediction": int(prediction)}

@app.post("/recommend/")
async def recommend(skin_type: str, issue: str):
    recommendations = get_recommendations(skin_type, issue)
    return {"recommendations": recommendations}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
