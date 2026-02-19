import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

model_path = "multilebel.pth"
image_path = "images/images/image_10.jpg"
num = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
imageNO = os.path.basename(image_path)

model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num)
model.load_state_dict(torch.load(model_path, map_location = device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    op = model(image)
    probabilities = torch.sigmoid(op)
    predictions = (probabilities > 0.5).int()

print(f"IMAGE NO --> {imageNO}")
print("PROBABILITIES --> ", probabilities.cpu().numpy())
print("PREDICTED LABELS -->", predictions.cpu().numpy())