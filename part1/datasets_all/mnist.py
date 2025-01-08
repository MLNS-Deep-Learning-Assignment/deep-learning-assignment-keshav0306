import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
import gzip
from typing import List, Optional, Union, Tuple
import cv2

class SumMNIST(Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        if(split == "train"):
            self.imgs = np.concatenate([np.load(root + "data0.npy"), np.load(root + "data1.npy"),np.load(root + "data2.npy")])
            self.labels = np.concatenate([np.load(root + "lab0.npy"), np.load(root + "lab1.npy"),np.load(root + "lab2.npy")])
        else:
            self.imgs = np.concatenate([np.load(root + "data2.npy")])
            self.labels = np.concatenate([np.load(root + "lab2.npy")])
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        
        img = image / 255
        data = {"img" : img.astype(np.float32), "sum" : label.astype(np.int32)}
        return data

if __name__ == "__main__":
    root = "/ssd_scratch//cvit/keshav/DL-Project/"
    dataset = SumMNIST(root, split="val")
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    for batch in loader:
        images, sums = batch['img'], batch['sum']
        print(images.shape, sums.shape)
        print(sums)
        # for i, (img, sum) in enumerate(zip(images, sums)):
        #     img = img.cpu().numpy().transpose(1, 2, 0)
        #     dep = dep.cpu().numpy()[0]
        #     dep = np.tile(dep[..., None], (1, 1, 3))
        #     dep = (dep - dep.min()) / (dep.max() - dep.min())
        #     cv2.imwrite(f"viz/dep_{i}.png", np.hstack([img, dep * 255]))
        # break