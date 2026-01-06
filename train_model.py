import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import onnx

class CNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1=nn.Conv2d(1,32, kernel_size=3)
    self.conv2=nn.Conv2d(32,64, kernel_size=3)
    self.conv3=nn.Conv2d(64,128, kernel_size=3)
    self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Custom model logic: 128*1*1
    # Let's verify dimensions:
    # Input: 28x28
    # Conv1(3x3) -> 26x26 -> Pool -> 13x13 (32 ch)
    # Conv2(3x3) -> 11x11 -> Pool -> 5x5 (64 ch)
    # Conv3(3x3) -> 3x3 -> Pool -> 1x1 (128 ch)
    # Flatten -> 128 * 1 * 1 = 128
    self.fc1=nn.Linear(128*1*1,128)
    self.fc2=nn.Linear(128,10)


  def forward(self,x):
    x=F.relu(self.conv1(x))
    x=self.pool(x)
    x=F.relu(self.conv2(x))
    x=self.pool(x)
    x=F.relu(self.conv3(x))
    x=self.pool(x)
    
    # Flatten: view(batch_size, -1)
    x = x.view(x.size(0), -1)
    
    x=F.relu(self.fc1(x))
    x=self.fc2(x)
    return x

def run_training_and_export():
    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Custom Transform
    Transform= transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    print("Downloading dataset...")
    final_dataset= datasets.MNIST(root='data', train=True, download=True, transform=Transform)
    
    # Custom Split Logic (3:2)
    total_size= len(final_dataset)
    train_size= int(total_size*0.6)
    test_size= total_size-train_size
    train_dataset, test_dataset= random_split(final_dataset, [train_size, test_size])
    
    batch_size=64
    train_loader= DataLoader(train_dataset, batch_size, shuffle=True)
    
    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 3 # Can be increased to 10 for better accuracy.
    
    print(f"Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = (correct_predictions / total_samples) * 100
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.2f}%")

    print("\n--- Exporting to ONNX for Android ---")
    
    # 1. Prepare for export (Evaluation mode)
    model.eval()
    
    # 2. Create Dummy Input (1 channel, 28x28 images)
    # IMPORTANT: Move dummy input to same device as model
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    
    output_path = "mnist_cnn.onnx"
    
    # 3. Export
    torch.onnx.export(model, 
                      dummy_input, 
                      output_path, 
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    # 4. Optimize/Fix for Android (Merged Weights)
    print("Optimizing ONNX file for mobile...")
    onnx_model = onnx.load(output_path)
    onnx.save(onnx_model, output_path)
    
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Success! Model saved to '{output_path}' ({size_kb:.2f} KB)")
    print("Now copy this file to 'android/app/src/main/assets/'")

if __name__ == "__main__":
    run_training_and_export()
