import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# 1. Read the CSV files
# -------------------------------
train_csv = "train.csv"
test_csv = "test.csv"

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

print("First 5 rows of train.csv:")
print(train_df.head())
print("\nFirst 5 rows of test.csv:")
print(test_df.head())

# -------------------------------
# 2. Define custom datasets
# -------------------------------
class TrainImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.copy()
        self.transform = transform
        # Convert labels to integers if necessary.
        if self.df['label'].dtype == object:
            self.label_mapping = {label: idx for idx, label in enumerate(self.df['label'].unique())}
            self.df['label'] = self.df['label'].map(self.label_mapping)
            print("Label mapping:", self.label_mapping)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['file_name']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.getcwd(), img_path)
        image = Image.open(img_path).convert("RGB")
        label = int(row['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

class TestImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['id']
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.getcwd(), img_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx  # return idx as dummy label

# -------------------------------
# 3. Define image transformations
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create dataset instances
full_train_dataset = TrainImageDataset(train_df, transform=transform)
full_test_dataset = TestImageDataset(test_df, transform=transform)

# -------------------------------
# 4. Reduce dataset size and split train into training and validation sets
# -------------------------------
max_train_samples = 300
max_test_samples = 200

# Create a subset of the full training dataset and test dataset.
train_subset = Subset(full_train_dataset, list(range(min(max_train_samples, len(full_train_dataset)))))
test_subset = Subset(full_test_dataset, list(range(min(max_test_samples, len(full_test_dataset)))))

# Split the train_subset: 70% training, 30% validation.
train_len = int(0.7 * len(train_subset))
val_len = len(train_subset) - train_len
train_split, val_split = random_split(train_subset, [train_len, val_len])

batch_size = 32
train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# -------------------------------
# 5. Define a small CNN model
# -------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # from 3 channels to 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32 -> 64 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64 -> 128 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 32 -> 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Determine number of classes from train.csv.
num_classes = len(train_df['label'].unique())
print("Number of classes:", num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN(num_classes=num_classes).to(device)
print(model)

# -------------------------------
# 6. Define loss function and optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 7. Training loop with tqdm and metrics tracking
# -------------------------------
num_epochs = 10  # Fewer epochs for faster training
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    epoch_loss = running_loss / len(train_split)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}")
    
    # Evaluate on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    val_acc = correct / total
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {val_acc:.4f}")

# (Optional) Save the trained model
torch.save(model.state_dict(), "small_model.pth")
