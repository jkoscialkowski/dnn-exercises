import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FruitDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.class_labels = os.listdir(path)
        self.class_counts = [len(os.listdir(path + '/' + d))
                             for d in os.listdir(path)]
        self.cum_class_counts = np.cumsum(self.class_counts)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return sum(self.class_counts)

    def __getitem__(self, item):
        # Correct class is the one for which `item` divided by cumulative sum
        # of class counts and floored is 0 for the first time. We want to keep
        # the class label as integer
        y = np.sum((item // self.cum_class_counts) > 0)
        label = self.class_labels[y]

        # Pick correct filename
        if y == 0:
            filename = os.listdir(self.path + '/' + label)[item]
        else:
            filename = os.listdir(
                self.path + '/' + label
            )[item - self.cum_class_counts[y - 1]]

        img = Image.open(self.path + '/' + label + '/' + filename)
        img = self.transform(img)
        return {'image': img, 'y': y, 'filename': filename}
