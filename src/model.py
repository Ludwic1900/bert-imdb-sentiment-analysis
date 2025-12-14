import torch
from torch import nn
from transformers import AutoModel
#BERT-based text classification model
class TextClassificationModel(nn.Module):
    def __init__(self, num_labels=7):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, src):
        outputs = self.bert(**src).last_hidden_state[:, 0, :]
        return self.predictor(outputs)
