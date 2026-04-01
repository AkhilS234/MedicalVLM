import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from src.dataset import MedicalVLMDataset
from src.model import MedicalCLIPModel
import torch.nn.functional as F
import os 

dataset = MedicalVLMDataset("data/processed/metadata.csv")
loader = DataLoader(dataset, batch_size= 8, shuffle=True, num_workers = 2)
model = MedicalCLIPModel()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
num_epochs = 10

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr = 1e-4, # peak learning rate which the scheduler is allowed to reach 
    steps_per_epoch = len(loader), # One step happens every batch 
    epochs = num_epochs
)

os.makedirs("outputs/checkpoints", exist_ok=True)

# A single forward pass using one batch of data

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch in loader:

        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device) # Move data from the batch to the device 

        image_emb, text_emb = model(image, input_ids, attention_mask) # forward pass through the model with image input, input_ids as tokenized text, and attention_mask for necessary padding

        similarity = image_emb @ (text_emb.T)
        similarity = similarity / 0.1
        # Each row in the similarity matrix represents how similar an image is to each text 
        # F is used for applying neural network operations directly to tensors

        # Contrastive Learning builds a matrix of similarity scores between embeddings, which are logits
        # Softmax converts logits into probabilites, cross entropy loss is applied to pick correct match 

        batch_size = similarity.shape[0]
        labels = torch.arange(batch_size, device=similarity.device)
        loss_i2t = F.cross_entropy(similarity, labels)
        loss_t2i = F.cross_entropy(similarity.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        optimizer.zero_grad() # Clear old gradients 
        loss.backward() # Compute gradients through backpropogation
        optimizer.step() # Update model weights (gradient descent)
        scheduler.step()
        epoch_loss += loss.item()


    avg_loss = epoch_loss / len(loader)
    print(f"Epoch{epoch+1}/{num_epochs}, Loss: {avg_loss:4f}")

if epoch == num_epochs - 1:
    torch.save(model.state_dict(), "outputs/checkpoints/medical_clip_final.pt")

        # takes batches from the dataset and runs the training loop to learn model parameters using forward pass, loss, and back propogation


