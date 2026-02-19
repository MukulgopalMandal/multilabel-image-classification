import os
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

class MultiLableDataset(Dataset):
    def __init__(self, img_dir, lable):
        base_dir = os.path.dirname(__file__)
        self.img_dir = os.path.join(base_dir, img_dir)
        lable = os.path.join(base_dir, lable)

        self.sample = []

        with open(lable, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            part = line.strip().split()

            if len(part) != 5:
                continue

            imgName = part[0]
            rawLables = part[1:]

            lable = []
            for val in rawLables:
                if val == "NA":
                    lable.append(-1)
                else:
                    lable.append(int(val))

            imagePath = os.path.join(self.img_dir, imgName)

            if not os.path.exists(imagePath):
                continue

            self.sample.append((imgName, torch.tensor(lable, dtype = torch.float)))

        print(f"loaded {len(self.sample)} valid samples")

        self.transform = T.Compose([
            T.Resize((224, 224)), 
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, index):
        imgName, lable = self.sample[index]
        imagePath = os.path.join(self.img_dir, imgName)
        img = Image.open(imagePath).convert("RGB")
        img = self.transform(img)

        return img, lable

