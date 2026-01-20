import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, label, transform=None):
        self.data_dir = Path(data_dir)
        self.label = label
        self.transform = transform
        self.image_paths = list(self.data_dir.glob('*.jpg')) + list(self.data_dir.glob('*.png'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, self.label
