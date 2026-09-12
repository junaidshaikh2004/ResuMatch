"""
A small, hand-built feedforward neural network for the resume/JD fit
classifier. Nothing pretrained here — this is a plain PyTorch nn.Module
with two hidden layers, trained from scratch in train_models.py on the
same 7-number feature vector the logistic regression baseline uses.
"""

import torch.nn as nn


class FitNet(nn.Module):
    def __init__(self, input_size, hidden1=16, hidden2=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        # Returns raw logits (no sigmoid) — paired with BCEWithLogitsLoss
        # during training for numerical stability. predict.py applies
        # sigmoid itself to turn this into a 0-1 probability.
        return self.network(x)
