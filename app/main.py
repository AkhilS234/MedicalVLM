from fastapi import FastAPI, UploadFile, File
from typing import List
import torch
from PIL import Image
import io
from pydantic import BaseModel
from src.model import MedicalCLIPModel
from src.dataset import MedicalVLMDataset
from src.inference import retrieve_by_text, retrieve_by_image, all_image_paths, all_texts


app = FastAPI(title= "Medical VLM")
dataset = MedicalVLMDataset("data/processed/metadata.csv")
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/examples")
def examples():
    results = []

    for i in range(5):
        results.append({
            "id": i, 
            "image_path": dataset.data.iloc[i]["image_path"],
            "text": dataset.data.iloc[i]["text"]
            })
    
    return {"examples": results}

class TextQuery(BaseModel):
    query:str
    top_k: int


@app.post("/search/text")
def search_text(request: TextQuery):
    results = retrieve_by_text(request.query, request.top_k)
    return {
        "query": request.query,
        "results": results
    }

@app.post("/search/image")
async def search_image(file: UploadFile=File(...), top_k: int = 5):
    image_bytes = await file.read() # await: pause execution of async function until all bytes of image are processed
    results = retrieve_by_image(image_bytes, top_k)
    return {
        "filename": file.filename, 
        "top_k":top_k, 
        "results":results,
    }




