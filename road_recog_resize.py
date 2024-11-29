import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class road_recog_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes an RGB image (1920x1080) as input and applies 32 convolutional filters of size 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Takes 32 input channels and applies 64 convolutional filters of size 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Takes 64 input channels and applies 128 convolutional filters of size 3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Takes 128 input channels and applies 256 convolutional filters of size 3x3
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Takes 256 input channels and applies 512 convolutional filters of size 3x3
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512*12*6, 256)  # Adjusted based on the input size
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        #Applies first convolutional layer and relu
        x = F.relu(self.conv1(x))

        #Applies pooling 
        x = F.max_pool2d(x, kernel_size=2)
        
        # Applies second convolutional layer
        x = F.relu(self.conv2(x))
        
        #Applies pooling
        x = F.max_pool2d(x, kernel_size=2)
        
        #Applies third convolutional layer
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)

        #Applies fourth convolutional layer
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2)

        #Applies fifth convolutional layer
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=2)
        
        # Flatten the layer
        x = x.view(x.size(0), -1)  # Automatically calculates the right shape
        
        #Applies first fully connected layer
        x = self.fc1(x)
        #Applies relu
        x = F.relu(x)
        
        #Applies the second fully connected layer
        x = self.fc2(x)
        return x       
    
# Load the data

transform = transforms.Compose([
    transforms.Resize((384, 216)),  # Resize to 384x216 pixels
    transforms.ToTensor(),  # Convert the image to a tensor
])

roads_train = datasets.ImageFolder(root="./Roads/Data/train", transform=transform)
roads_test = datasets.ImageFolder(root="./Roads/Data/validation", transform=transform)
train_loader = torch.utils.data.DataLoader(roads_train, batch_size=1500, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(roads_test, batch_size=1500, shuffle=True, num_workers=0)

## Training
# Instantiate model  
model = road_recog_CNN() 

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # <---- change here

# Iterate through train set minibatchs 
for epoch in range(6):  # <---- change here
    for images, labels in train_loader:
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        x = images  # <---- change here 
        y = model(x)
        loss = criterion(y, labels)
        # Backward pass
        loss.backward()
        optimizer.step()

## Testing
correct = 0
total = len(roads_test)

with torch.no_grad():
    # Iterate through test set minibatchs 
    for images, labels in test_loader:
        # Forward pass
        x = images  # <---- change here 
        y = model(x)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))