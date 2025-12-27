# ============================================================
# CIFAR-10 CNN for Amazon SageMaker - Production Ready
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import zipfile

# ---------------------- Dataset Extraction ------------------
# CRITICAL: Unzip dataset BEFORE attempting to load it
DATA_ZIP = "/opt/ml/input/data/dataset/dataset.zip"
EXTRACT_DIR = "/opt/ml/input/data"

print("=" * 60)
print("DATASET EXTRACTION")
print("=" * 60)
print(f"Checking dataset zip path: {DATA_ZIP}")

if os.path.exists(DATA_ZIP):
    print("✓ Dataset zip found")
    print(f"Unzipping dataset to: {EXTRACT_DIR}")
    with zipfile.ZipFile(DATA_ZIP, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("✓ Unzip completed successfully")
    
    # Verify extraction
    extracted_contents = os.listdir(EXTRACT_DIR)
    print(f"✓ Extracted contents: {extracted_contents}")
else:
    raise FileNotFoundError(f"❌ {DATA_ZIP} not found. Please ensure dataset is uploaded to S3.")

print("=" * 60 + "\n")

# ---------------------- SageMaker Configuration -------------
# SageMaker automatically mounts S3 data to these paths
TRAIN_DIR = "/opt/ml/input/data/train"
TEST_DIR = "/opt/ml/input/data/test"

# SageMaker saves anything in /opt/ml/model to S3 as model.tar.gz
MODEL_DIR = "/opt/ml/model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")

# Training Configuration
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
print(f"Train directory: {TRAIN_DIR}")
print(f"Test directory: {TEST_DIR}")
print(f"Model directory: {MODEL_DIR}\n")

# ---------------------- Dataset Class Names -----------------
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ---------------------- Data Transforms ---------------------
# Data augmentation for training
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# No augmentation for test
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# ---------------------- Load Dataset from S3 ----------------
print("Loading datasets from S3-mounted directories...")

trainset = ImageFolder(TRAIN_DIR, transform=transform_train)
testset = ImageFolder(TEST_DIR, transform=transform_test)

trainloader = DataLoader(
    trainset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)

testloader = DataLoader(
    testset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

print(f"✓ Training samples: {len(trainset)}")
print(f"✓ Testing samples: {len(testset)}")
print(f"✓ Number of classes: {len(trainset.classes)}\n")

# ---------------------- Define CNN Model --------------------
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        # Convolutional Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        
        # Convolutional Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

# ---------------------- Initialize Model --------------------
model = ImprovedCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=5, 
    verbose=True
)

# ---------------------- Load Checkpoint ---------------------
start_epoch = 0
best_acc = 0.0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

if os.path.exists(CHECKPOINT_PATH):
    print(f"✓ Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    history = checkpoint['history']
    print(f"✓ Resuming from epoch {start_epoch}, Best Accuracy: {best_acc:.2f}%\n")
else:
    print("✓ Starting training from scratch\n")

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")

# ---------------------- Training Function -------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ---------------------- Validation Function -----------------
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ---------------------- Train Model -------------------------
print("=" * 60)
print("Starting Training...")
print("=" * 60 + "\n")

for epoch in range(start_epoch, EPOCHS):
    train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, testloader, criterion, DEVICE)
    
    # Update history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Learning rate scheduling
    scheduler.step(val_acc)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Save checkpoint after every epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': max(best_acc, val_acc),
        'history': history,
        'class_names': class_names
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(checkpoint, MODEL_PATH)
        print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
    print()

print("=" * 60)
print("Training Complete!")
print(f"✓ Best Validation Accuracy: {best_acc:.2f}%")
print("=" * 60 + "\n")

# ---------------------- Final Evaluation --------------------
print("Evaluating on Test Set...")
test_loss, test_acc = validate(model, testloader, criterion, DEVICE)
print(f"✓ Final Test Accuracy: {test_acc:.2f}%")
print(f"✓ Final Test Loss: {test_loss:.4f}\n")

# ---------------------- Plot Training History ---------------
plt.figure(figsize=(14,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1,2,2)
plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, 'training_history.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Training history plot saved as '{plot_path}'\n")
plt.close()

# ---------------------- Sample Predictions ------------------
model.eval()
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(DEVICE), labels.to(DEVICE)

with torch.no_grad():
    outputs = model(images[:16])
    _, predicted = outputs.max(1)

# Denormalize images for display
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])

plt.figure(figsize=(12,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    
    # Denormalize
    img = images[i].cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    plt.imshow(img)
    true_label = trainset.classes[labels[i].item()]
    pred_label = trainset.classes[predicted[i].item()]
    color = 'green' if predicted[i] == labels[i] else 'red'
    plt.title(f'True: {true_label}\nPred: {pred_label}', fontsize=8, color=color)
    plt.axis('off')

plt.suptitle(f'Sample Predictions (Test Accuracy: {test_acc:.2f}%)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
predictions_path = os.path.join(MODEL_DIR, 'sample_predictions.png')
plt.savefig(predictions_path, dpi=300, bbox_inches='tight')
print(f"✓ Sample predictions saved as '{predictions_path}'\n")
plt.close()

print("=" * 60)
print(f"✓ Best model saved at: {MODEL_PATH}")
print(f"✓ Checkpoint saved at: {CHECKPOINT_PATH}")
print(f"✓ All outputs saved to: {MODEL_DIR}")
print("=" * 60)
print("\n✅ SageMaker will automatically upload {MODEL_DIR} to S3 as model.tar.gz")