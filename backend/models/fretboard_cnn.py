import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class FretBoardCNN(nn.Module):
    """
    1D-CNN Context-based approach to mapping midi to fretboard
    Temporal Convolutional Network
    
    """
    def __init__(self, context_window=5, pitch_embed=32):
        super().__init__()
        self.context_window = context_window
        self.sequence_length = context_window * 2 + 1
        # embed pitch to vector 
        self.pitch_embedding = nn.Embedding(num_embeddings=128, embedding_dim=pitch_embed)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=pitch_embed, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU())
        
        self.dropout = nn.Dropout(0.25)
        
        
        flattened_size = 256 * (context_window * 2 + 1)
        
        self.total_class = nn.Linear(flattened_size, 150)
        
        # # split output to fret and string
        # # 6 strings
        # self.string = nn.Linear(flattened_size, 6)
        # # 0 (open) to 24 frets
        # self.fret = nn.Linear(flattened_size, 25)  
        # self.dropout = nn.Dropout(0.25)
        

        
    def forward(self, input_pitch_data):
        embededed_x = self.pitch_embedding(input_pitch_data)
        transposed_x = embededed_x.transpose(1, 2)
        
        # convolution
        input_conv = self.conv1(transposed_x)
        hidden_conv = self.conv2(input_conv)
        output_conv = self.conv3(hidden_conv)
        
        # pooling
        flattened = output_conv.view(output_conv.size(0), -1)
    
        
        x = self.dropout(flattened)
        
        # predicted_string = self.string(x)
        # predicted_fret = self.fret(x)
        
        logits = self.total_class(x)
        
        return logits
    


def evaluate_model(model):
    # accuracy metric
    acc = Accuracy(task="multiclass", num_classes=10)
    
    # iterate over the dataset
    model.eval()
    with torch.no_grad():
        for midi, labels in ___:
            # predicted probabilities for test data batch
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            acc(preds, labels)
            precision(preds, labels)
            recall(preds, labels)
    # total test accuracy
    accuracy = acc.compute()
    print(f"Accuracy: {accuracy}")
    return accuracy
        


if __name__ == '__main__':
    test_model()
        
      
    
        