import torch
import torch.nn as nn
import json
import io
from torchvision import transforms
from PIL import Image

# ------------------ Class Labels ------------------
CLASS_NAMES = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# ------------------ Device ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Model Architecture ------------------
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()

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
        return self.fc(x)

# ------------------ Load Model ------------------
def model_fn(model_dir):
    checkpoint = torch.load(f"{model_dir}/checkpoint.pth", map_location=DEVICE)

    model = ImprovedCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model

# ------------------ Input Processing ------------------
def input_fn(request_body, content_type):
    image = Image.open(io.BytesIO(request_body)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    return transform(image).unsqueeze(0)

# ------------------ Prediction ------------------
def predict_fn(input_data, model):
    input_data = input_data.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_data)
        pred_idx = outputs.argmax(dim=1).item()

    return CLASS_NAMES[pred_idx]

# ------------------ Output ------------------
def output_fn(prediction, accept):
    return json.dumps({"prediction": prediction}), "application/json"
