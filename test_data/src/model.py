import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class SentimentModel(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=num_labels
        )

    def forward(self, **kwargs):
        return self.backbone(**kwargs)
