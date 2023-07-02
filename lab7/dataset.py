import json
import numpy as np
from torch.utils import data
from PIL import Image
import os
import torchvision.transforms as transforms
import torch

"""
Mean: [0.48603287 0.48267534 0.4774533 ]
Std: [0.07425077 0.0715564  0.07178217]
"""

transformer = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5 ], std=[0.5,0.5,0.5]),
        # transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)


class iclevr_dataset(data.Dataset):
    def __init__(self, file, mode):
        with open("./objects.json", "r") as f:
            classes = json.load(f)
        self.classes = [key for key, _ in classes.items()]
        self.img_dir = "./iclevr"
        self.mode = mode

        with open(file, "r") as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        if self.mode == "train":
            img_name, labels = list(self.data.items())[index]
            img = Image.open(self.img_dir + "/" + img_name).convert("RGB")
            img = transformer(img)
            one_hot_labels = [0] * len(self.classes)
            for label in labels:
                one_hot_labels[self.classes.index(label)] = 1
            return img, torch.tensor(one_hot_labels, dtype=torch.float32)

        if self.mode == "test":
            labels = list(self.data)[index]
            one_hot_labels = [0] * len(self.classes)
            for label in labels:
                one_hot_labels[self.classes.index(label)] = 1
            return torch.tensor(one_hot_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
