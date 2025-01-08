import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from datasets_all.mnist import SumMNIST
from mnist_model import LITMNISTModel
import yaml
from torch.utils.data import WeightedRandomSampler
import numpy as np

L.seed_everything(2024)
class Config:
    def __init__(self, config):
        self.config = config
        for k, v in self.config.items():
            self.__setattr__(k,  v)

config = "configs/red.yaml"
with open(config, "r") as f:
    config = Config(yaml.safe_load(f))

dataset_class = {"mnist" : SumMNIST}

train_dataset = dataset_class[config.dataset_type](config.dataset_config, split="train")
val_dataset = dataset_class[config.dataset_type](config.dataset_config, split="val")

shuffle = False
trainloader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=shuffle)#, sampler=train_sampler)
valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, shuffle=False)

model = LITMNISTModel(config)
logger = WandbLogger(name="debug_run", project="a1_mlns", config=config)
# logger = CSVLogger("logs", name="a1_mlns")
checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=config.ckpt_dir, filename=f"best_loss_{config.backbone}_{config.dataset_type}", mode='min')

trainer = L.Trainer(devices=[0, 1, 2, 3], max_epochs=2000, callbacks=[checkpoint_callback], logger=[logger],\
    strategy='ddp_find_unused_parameters_true', detect_anomaly=True)

trainer.fit(model, trainloader, valloader)