import torch
import numpy as np
import cv2
import lightning as L
from tqdm import tqdm
from datasets_all.mnist import SumMNIST
import yaml
from mnist_model import LITMNISTModel

class Config:
    def __init__(self, config):
        self.config = config
        for k, v in self.config.items():
            self.__setattr__(k,  v)

config = "configs/red.yaml"

with open(config, "r") as f:
    config = Config(yaml.safe_load(f))
    
dataset_class = {"mnist" : SumMNIST}
val_dataset = dataset_class[config.dataset_type](config.dataset_config, split="val")
valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=1024, num_workers=4)
model = LITMNISTModel.load_from_checkpoint("/ssd_scratch/cvit/keshav/mlns_a1_ckpts/best_loss_resnet18_mnist-v7.ckpt", config=config)
model.eval()
count = 0
device = torch.device("cuda:0")
metrics = {"acc" : 0}

with torch.no_grad():
    for batch in tqdm(valloader):
        data = batch
        for k, v in batch.items():
            batch[k] = v.to(device)
        output = model.model.validate(batch)
        acc = output['acc']
        metrics['acc'] += acc * len(data['img'])
        count += len(data['img'])
        
acc = metrics['acc'] / count
print(acc)