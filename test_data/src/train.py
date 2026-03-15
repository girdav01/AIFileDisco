import torch
import torch.nn as nn
from transformers import AutoModel

class Trainer:
    def __init__(self, model, lr=2e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_epoch(self, dataloader):
        self.model.train()
        for batch in dataloader:
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
