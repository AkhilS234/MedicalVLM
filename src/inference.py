from tkinter import Image
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from src.model import MedicalCLIPModel
from src.dataset import MedicalVLMDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MedicalCLIPModel()
model.load_state_dict(torch.load("outputs/checkpoints/medical_clip_epoch_5.pt", map_location=device))
model = model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

dataset = MedicalVLMDataset("data/processed/metadata.csv")
loader = DataLoader(dataset, batch_size=8, shuffle=False)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

all_image_emb = []
all_text_emb = []
all_texts = []
all_image_paths = []

with torch.no_grad():
    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        image_emb, text_emb = model(image, input_ids, attention_mask)

        all_image_emb.append(image_emb.cpu())
        all_text_emb.append(text_emb.cpu())
        all_texts.extend(batch["text"])
        all_image_paths.extend(batch["image_path"])

all_image_emb = torch.cat(all_image_emb, dim=0)
all_text_emb = torch.cat(all_text_emb, dim=0)

print("Embeddings loaded:", all_image_emb.shape)

def retrieve_by_text(query: str, top_k: int = 5):
    encoded = tokenizer(
        query, 
        padding ="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt" # Return data as PyTorch Tensors 
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        text_emb = model.encode_text(input_ids, attention_mask)
        text_emb = text_emb.cpu()

    scores = all_image_emb @ text_emb.T
    scores = scores.squeeze(1)

    topk = torch.topk(scores, k=top_k)

    results = []
    for rank, idx in enumerate(topk.indices.tolist(), start=1):
        results.append({
            "rank": rank,
            "image_path": all_image_paths[idx],
            "score": float(topk.values[rank - 1])
        })

    return results

def retrieve_by_image(image_bytes, top_k: int = 5):
    image = Image.open(image_bytes).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_emb = model.encode_image(image)
        image_emb = image_emb.cpu()

    scores = image_emb @ all_text_emb.T
    scores = scores.squeeze(0)

    topk = torch.topk(scores, k=top_k)

    results = []
    for rank, idx in enumerate(topk.indices.tolist(), start=1):
        results.append({
            "rank": rank,
            "text": all_texts[idx],
            "score": float(topk.values[rank - 1])
        })

    return results







