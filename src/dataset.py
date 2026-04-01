import os
from typing import Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MedicalVLMDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "distilbert-base-uncased",
        image_size: int = 224,
        max_length: int = 128,
    ) -> None:
        self.data = pd.read_csv(csv_path)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  #
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        required_cols = {"image_path", "text"}
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]

        image_path = row["image_path"]
        text = str(row["text"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "text": text,
            "image_path": image_path,
        }

# Loads rows from a CSV, opens the image from the path in that row, applies torchvision transformers to convert it into a tensor,
# tokenize the paired text using HugginFace tokenizer and return a dictionary for DataLoader to batch it