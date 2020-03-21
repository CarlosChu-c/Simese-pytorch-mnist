import random
import numpy as np
import torch
from torch.utils.data import Dataset


class getDataset(Dataset):

    def __init__(self, getDataset, transform=None, relables=False):
        self.getDataset = getDataset
        self.transform = transform
        self.relables = relables

    def __getitem__(self, index):
        datas = self.getDataset.data
        labels = self.getDataset.targets
        rand_i = random.choice(range(len(datas)))
        img0 = datas[rand_i]

        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                rand_j = random.choice(range(len(datas)))
                img1 = datas[rand_j]
                if labels[rand_i] == labels[rand_j]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                rand_j = random.choice(range(len(datas)))
                img1 = datas[rand_j]
                if labels[rand_i] != labels[rand_j]:
                    break
        img0 = img0[np.newaxis, :, :]
        img1 = img1[np.newaxis, :, :]

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        if self.relables:
            return img0, img1, torch.from_numpy(np.array([int(labels[rand_i] != labels[rand_j])], dtype=np.float32)), \
                   labels[rand_i], labels[rand_j]
        else:
            return img0, img1, torch.from_numpy(np.array([int(labels[rand_i] != labels[rand_j])], dtype=np.float32))

    def __len__(self):
        return len(self.getDataset.data)
