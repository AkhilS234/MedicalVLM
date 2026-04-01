import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torchvision import models

class MedicalCLIPModel(nn.Module):
    def __init__(self, embed_dim = 256):
        
        super().__init__()
        
        self.image_encoder = models.resnet18(pretrained=True) # 18 refers to the number of layers in the CNN
        self.image_encoder.fc = nn.Identity() #fc refers to final layer, set it to nn.Identity() to take away image classification feature and instead output an embedding 
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")

        self.image_projection = nn.Linear(512, embed_dim)
        self.text_projection = nn.Linear(768, embed_dim)
        # Applying Linear Projection to resize the output vectors of image and text projections to 256. Both are same size and in the same vector space 

        self.logit_scale = nn.Parameter(torch.tensor(1/0.07).log())


    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.image_encoder(image)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        #input_ids: unique id numbers for each word, attention_mask: adds extra zeroes as padding for shorter sentences 
        #last_hidden_state -> 3D tensor, [batch size ( : -> every sentence) , sequence length, hidden size( : -> 768) ]

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        image_embeddings = F.normalize(image_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        # scale each vector to have a length of 1. This makes calculating the closeness between an image and its caption easier


        return image_embeddings,text_embeddings
    
    def get_similarity_logits(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        self.logit_scale.exp()
        logits = scale * image_embeddings @ text_embeddings.T
        return logits
    
    # Creates a similarity scoreboard that compares every image to every caption, applying a learnable scaling factor to amplify signal of correct matches for training loss


    



    



    

