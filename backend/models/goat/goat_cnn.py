import torch
import torch.nn as nn

class GoatFretboardCNN(nn.Module):
    # 6 strings * 25 frets = 150 possible positions
    def __init__(self, num_classes=150):
        super(GoatFretboardCNN, self).__init__()
        
        # Input: (Batch, 1, 128, TimeFrames)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.flatten = nn.Flatten()
        
        # Calculate size: 128 mels / 8 = 16. 
        # Time frames / 8 = approximates to 2-4.
        # self.fc1 = nn.Linear(128 * 16 * 2, 512) 
        self.fc1 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output Logits
        return x