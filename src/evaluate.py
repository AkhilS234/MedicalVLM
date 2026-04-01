import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from src.model import MedicalCLIPModel
from src.dataset import MedicalVLMDataset
import glob
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


dataset = MedicalVLMDataset("data/processed/metadata.csv")
loader = DataLoader(dataset, batch_size=8, shuffle=False)
model = MedicalCLIPModel()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
checkpoint_dir = "outputs/checkpoints"
checkpoints = sorted(glob.glob(f"{checkpoint_dir}/*.pt"))

if not checkpoints:
    raise FileNotFoundError("No checkpoints found. Run train.py first.")

latest = checkpoints[-1]
print(f"Loading checkpoint: {latest}")
model.load_state_dict(torch.load(latest, map_location=device))

all_image_emb = []
all_text_emb = []
all_texts = []
all_image_paths = []
model.eval()

with torch.no_grad():
    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        image_emb, text_emb = model(image, input_ids, attention_mask)

        all_image_emb.append(image_emb.cpu()) # List of image tensors, one tensor per batch 
        all_text_emb.append(text_emb.cpu()) # List of text tensors
        all_texts.extend(batch["text"])
        all_image_paths.extend(batch["image_path"])

all_image_emb = torch.cat(all_image_emb, dim=0)
all_text_emb = torch.cat(all_text_emb, dim=0)

similarity = all_image_emb @ all_text_emb.T
print(similarity.shape)

query_image_idx = 0
scores = similarity[query_image_idx]
k = min(5, scores.shape[0])
topk = torch.topk(scores, k=k)

print("\n=== Image -> Top 5 Text Matches ===")
print("Query image:", all_image_paths[query_image_idx])
for rank, idx in enumerate(topk.indices.tolist(), start=1):
    print(f"{rank}. {all_texts[idx]}  (score={topk.values[rank-1].item():.4f})")

query_text_idx = 0
scores = similarity[:, query_text_idx]
k = min(5, scores.shape[0])
topk = torch.topk(scores, k=k)

print("\n=== Text -> Top 5 Image Matches ===")
print("Query text:", all_texts[query_text_idx])
for rank, idx in enumerate(topk.indices.tolist(), start=1):
    print(f"{rank}. {all_image_paths[idx]}  (score={topk.values[rank-1].item():.4f})")


num_samples = similarity.shape[0]

recall_at_1 = 0
recall_at_5 = 0

for i in range(num_samples):
    top1 = torch.topk(similarity[i], k=1).indices.tolist()
    top5 = torch.topk(similarity[i], k=5).indices.tolist()

    if i in top1:
        recall_at_1 += 1
    if i in top5:
        recall_at_5 += 1

recall_at_1 /= num_samples
recall_at_5 /= num_samples

# After your existing recall calculation add these:

# Mean Reciprocal Rank
mrr = 0
for i in range(num_samples):
    ranked = torch.argsort(similarity[i], descending=True).tolist()
    if i in ranked:
        rank = ranked.index(i) + 1
        mrr += 1 / rank
mrr /= num_samples

# Recall@10
recall_at_10 = 0
for i in range(num_samples):
    top10 = torch.topk(similarity[i], k=min(10, num_samples)).indices.tolist()
    if i in top10:
        recall_at_10 += 1
recall_at_10 /= num_samples

# Median Rank
ranks = []
for i in range(num_samples):
    ranked = torch.argsort(similarity[i], descending=True).tolist()
    ranks.append(ranked.index(i) + 1)
median_rank = sorted(ranks)[len(ranks) // 2]

print(f"Recall@1:    {recall_at_1:.4f}")
print(f"Recall@5:    {recall_at_5:.4f}")
print(f"Recall@10:   {recall_at_10:.4f}")
print(f"MRR:         {mrr:.4f}")
print(f"Median Rank: {median_rank}")

true_labels = [1 if "Abnormal" in t else 0 for t in all_texts]
pred_labels = []

normal_prompt = "Normal chest X-ray with clear lung fields. No active disease."
tb_prompt = "Abnormal chest X-ray showing tuberculosis findings."

with torch.no_grad():
    prompts = [normal_prompt, tb_prompt]
    encoded = dataset.tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    dummy_images = torch.zeros(2, 3, 224, 224).to(device)
    _, prompt_emb = model(dummy_images, input_ids, attention_mask)
    prompt_emb = F.normalize(prompt_emb, dim=-1)

normal_emb = prompt_emb[0].cpu()
tb_emb = prompt_emb[1].cpu()

for i in range(num_samples):
    img_emb = F.normalize(all_image_emb[i], dim=-1)
    normal_score = (img_emb @ normal_emb).item()
    tb_score = (img_emb @ tb_emb).item()
    pred_labels.append(0 if normal_score > tb_score else 1)

print("\n=== Classification Report ===")
print(classification_report(true_labels, pred_labels, target_names=["Normal", "TB"]))

cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "TB"],
            yticklabels=["Normal", "TB"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix — Zero-Shot TB Classification")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.show()
print("Saved confusion matrix to outputs/confusion_matrix.png")
