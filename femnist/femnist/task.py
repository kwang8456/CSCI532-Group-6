"""femnist: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from PIL import Image
from torch.utils.data import Dataset

# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """CNN for FEMNIST (28x28 grayscale images)"""
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # First convolutional block: 3x3 kernels, 32 feature maps
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 channel for grayscale
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Second convolutional block: 3x3 kernels, 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FEMNISTDataset(Dataset):
    """Custom Dataset for loading FEMNIST from local folders."""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to client folder (e.g., 'data/client_0')
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Load all images and labels
        for label in range(10):  # Labels 0-9
            label_dir = os.path.join(root_dir, str(label))
            if not os.path.exists(label_dir):
                continue
            
            for img_file in os.listdir(label_dir):
                if img_file.endswith('.png'):
                    img_path = os.path.join(label_dir, img_file)
                    self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image as grayscale
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, data_root: str = "dataset"):
    """Load partition FEMNIST data from local filesystem."""
    
    # Construct path to client folder
    client_folder = os.path.join(data_root, f"client_{partition_id}")
    
    if not os.path.exists(client_folder):
        raise ValueError(f"Client folder not found: {client_folder}")
    
    # Load full dataset for this client
    full_dataset = FEMNISTDataset(client_folder, transform=pytorch_transforms)
    
    # Split into train (80%) and test (20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)
    
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
