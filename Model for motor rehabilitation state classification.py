import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet101
from torch.utils.data import DataLoader, Dataset

# Custom dataset to load left and right foot images from folders
class FootDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform to apply to the images.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_pairs = []
        self.labels = []

        # Assume left foot images are in 'left' folder, right foot in 'right', and labels in a text file
        left_dir = os.path.join(data_dir, "left")
        right_dir = os.path.join(data_dir, "right")
        label_file = os.path.join(data_dir, "labels.txt")

        # Read image names and labels
        with open(label_file, "r") as f:
            for line in f:
                left_image, right_image, label = line.strip().split(",")
                self.image_pairs.append((os.path.join(left_dir, left_image), os.path.join(right_dir, right_image)))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        left_image_path, right_image_path = self.image_pairs[idx]
        label = self.labels[idx]

        # Load images
        left_image = Image.open(left_image_path).convert("RGB")
        right_image = Image.open(right_image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, label


# Define ResNet-101 encoder
class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        resnet = resnet101(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification head

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return x


# Define the full model
class MobilityRehabilitationModel(nn.Module):
    def __init__(self):
        super(MobilityRehabilitationModel, self).__init__()
        self.left_foot_encoder = ResNetEncoder()
        self.right_foot_encoder = ResNetEncoder()

        # MLP part for feature decoding
        self.mlp = nn.Sequential(
            nn.Linear(2048 * 2, 256),  # ResNet-101 outputs 2048 features each
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # Output for 3 states classification
        )

    def forward(self, left_foot, right_foot):
        left_encoded = self.left_foot_encoder(left_foot)
        right_encoded = self.right_foot_encoder(right_foot)
        combined = torch.cat((left_encoded, right_encoded), dim=1)
        output = self.mlp(combined)
        return output


# Initialize the model and hyperparameters
model = MobilityRehabilitationModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define optimizer and loss function, can be replaced by other optimizers, e.g. Ranger used in our study 
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.99),
    weight_decay=1e-5,
    eps=1e-8
)
criterion = nn.CrossEntropyLoss()

# Define transformations for the dataset, for utilizing the pre-trained ResNets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
data_dir = "path_to_your_dataset"  # Replace with your dataset path
train_dataset = FootDataset(data_dir=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for left_foot_images, right_foot_images, labels in train_loader:
        left_foot_images = left_foot_images.to(device)
        right_foot_images = right_foot_images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(left_foot_images, right_foot_images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), "mobility_rehabilitation_model.pth") ## Change the path based on your needs

# Define the test loop
def test_model(model, test_loader, device):
    """
    Test the model on the test dataset.
    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for left_foot_images, right_foot_images, labels in test_loader:
            left_foot_images = left_foot_images.to(device)
            right_foot_images = right_foot_images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(left_foot_images, right_foot_images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


# Load test dataset
test_data_dir = "path_to_your_test_dataset"  # Replace with your test dataset path
test_dataset = FootDataset(data_dir=test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Test the trained model
test_loss, test_accuracy = test_model(model, test_loader, device)


