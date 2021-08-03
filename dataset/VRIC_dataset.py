import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

import os


class VRICDataset(Dataset):
    def __init__(self, transforms=None, split="train"):
        self.data_dir = None
        self.data_csv = None
        self.transforms = transforms
        if split == "train":
            self.data_csv = "data/VRIC/vric_train.csv"
            self.data_dir = "data/VRIC/train_images"
        elif split == "val":
            self.data_csv = "data/VRIC/vric_gallery.csv"
            self.data_dir = "data/VRIC/gallery_images"
        elif split == "test":
            self.data_csv = "data/VRIC/vric_probe.csv"
            self.data_dir = "data/VRIC/probe_images"
        self.data_list = pd.read_csv(self.data_csv)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_filename, idx, cam_idx = self.data_list.iloc[idx]
        image = read_image(os.path.join(self.data_dir, img_filename))
        label = idx
        if self.transforms:
            image = self.transforms(image)
        return image, label
