import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from backend.models.goat_cnn import GoatFretboardCNN

class GoatDataset(Dataset):
    def __init__(self, data_dir):
        self.spec_dir = os.path.join(data_dir, "specs")
        self.label_dir = os.path.join(data_dir, "labels")
        self.files = [f for f in os.listdir(self.spec_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        
        # Load spec
        spec = np.load(os.path.join(self.spec_dir, file_name))
        # Add channel dimension: (1, 128, Time)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        
        # Load label [String, Fret]
        label_raw = np.load(os.path.join(self.label_dir, file_name))
        string_idx = int(label_raw[0]) - 1 # Convert 1-6 to 0-5
        fret = int(label_raw[1])
        
        # Convert to single class ID (0-149)
        class_id = (string_idx * 25) + fret
        
        return spec, torch.tensor(class_id, dtype=torch.long)

def train():
    # Config
    DATA_DIR = "backend/data/processed_goat"
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Data
    dataset = GoatDataset(DATA_DIR)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training on {len(train_data)} samples. Validating on {len(val_data)} samples.")
    
    # Model
    model = GoatFretboardCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"backend/models/goat_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()