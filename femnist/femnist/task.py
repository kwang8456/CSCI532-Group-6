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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Add batch normalization
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block: 3x3 kernels, 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Add batch normalization
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
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


def load_data(partition_id: int, num_partitions: int, data_root: str = "femnist/dataset"):
    """Load partition FEMNIST data from local filesystem."""
    
    # If data_root is relative, make it absolute from current working directory
    if not os.path.isabs(data_root):
        # When running with flwr, CWD is the project root
        cwd_path = os.path.join(os.getcwd(), data_root)
        
        if os.path.exists(cwd_path):
            data_root = cwd_path
        else:
            # Fallback: try relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, "..", "dataset")
            if os.path.exists(alt_path):
                data_root = os.path.abspath(alt_path)
            else:
                raise ValueError(f"Cannot find dataset. Tried:\n  {cwd_path}\n  {alt_path}")
    
    # Rest of the function stays the same...
    client_folder = os.path.join(data_root, f"client_{partition_id}")
    
    if not os.path.exists(client_folder):
        print(f"❌ ERROR: Client folder not found!")
        print(f"   Looking for: {client_folder}")
        print(f"   data_root: {data_root}")
        print(f"   Current working directory: {os.getcwd()}")
        raise ValueError(f"Client folder not found: {client_folder}")
    
    full_dataset = FEMNISTDataset(client_folder, transform=pytorch_transforms)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No images found in {client_folder}")
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
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
