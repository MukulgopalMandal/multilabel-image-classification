import torch
import sys
import os
import torch.nn as nn
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
import torchvision.models as models
from DATA import MultiLableDataset
sys.path.append(os.getcwd())

dev = "cuda" if torch.cuda.is_available() else "cpu"

DATA = MultiLableDataset(img_dir = "images/images", lable = "labels.txt")

loader = DataLoader(DATA, batch_size = 32, shuffle = True)

model = models.resnet18(pretrained = True)
model.fc = nn.Linear(model.fc.in_features, 4)

model = model.to(dev)

lables_all = torch.stack([y for _, y in DATA])

positive = (lables_all == 1).sum(dim = 0)
negative = (lables_all == 0).sum(dim = 0)
pos_weight = negative / (positive + 1e-6)

criterion = nn.BCEWithLogitsLoss(reduction = "none", pos_weight = pos_weight.to(dev))
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

loss_history = []
iter = 0
epochs = 10

for epoch in range(epochs):
    model.train()
    for images, labels in loader:
        images = images.to(dev)
        labels = labels.to(dev)

        op = model(images)

        mask = labels != -1
        lossMatrix = criterion(op, labels)
        loss = (lossMatrix * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        iter += 1

    print(f"epoch {epoch + 1}/{epochs} - loss : {loss.item():.4f}")

torch.save(model.state_dict(), "multilebel.pth")
plt.figure()
plt.plot(loss_history)
plt.xlabel("iteration_number")
plt.ylabel("training_loss")
plt.title("Aimonk_multilable_problem")
plt.show()