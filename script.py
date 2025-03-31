import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

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
        # Convert labels to integers (if they aren't already)
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
        return image, idx  # return idx as a dummy label

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
# 4. Reduce dataset size to speed up training
# -------------------------------
# Use only a subset (e.g., first 1000 samples for training, 200 for test)
max_train_samples = 1000
max_test_samples = 200

train_subset = Subset(full_train_dataset, list(range(min(max_train_samples, len(full_train_dataset)))))
test_subset = Subset(full_test_dataset, list(range(min(max_test_samples, len(full_test_dataset)))))

batch_size = 32
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# -------------------------------
# 5. Define a smaller CNN model
# -------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # From 3 to 16 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # From 16 to 32 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 64 -> 32
        )
        # After two poolings, image size is 128/4 = 32
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),  # 32 channels * 32 * 32 feature map
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Determine number of classes from train.csv
num_classes = len(train_df['label'].unique())
print("Number of classes:", num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN(num_classes=num_classes).to(device)
print(model)

# -------------------------------
# 6. Loss function and optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 7. Training loop with tqdm
# -------------------------------
num_epochs = 5  # Fewer epochs for faster training
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
    
    epoch_loss = running_loss / len(train_subset)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

# -------------------------------
# 8. Evaluation on test set
# -------------------------------
model.eval()
predictions = []
with torch.no_grad():
    for images, indices in tqdm(test_loader, desc="Testing", unit="batch"):
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

print("Some test predictions:")
print(predictions[:10])

# (Optional) Save the trained model
torch.save(model.state_dict(), "small_model.pth")
