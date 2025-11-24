from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from similarity import get_similarity
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/similarity")
def read_root(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        dataset, score = get_similarity(temp_path)
        os.remove(temp_path) 
        if dataset == "none_human":
            return {"error": "No human face skin detected in the image, cannot process."}
        if dataset == "none_disease_light":
            return {
                "skin_color": "light",
                "disease_type": "none",
                "accuracy_score": float(score)
            }
        return {
            "skin_color": dataset.split("_")[0],
            "disease_type": dataset.split("_")[1],
            "accuracy_score": float(score)
        }
    except ValueError as e:
        os.remove(temp_path)
        return {"error": str(e)}
