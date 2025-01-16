import torch
import lightning as L
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from diffusers import DDPMScheduler
import argparse
from lightning.pytorch.loggers import WandbLogger
from diffusers import UNet2DModel
from alignment import align_depth_least_square
import metric
from scipy.optimize import linear_sum_assignment
# from transformers import  AutoImageProcessor, \
#     EfficientNetModel, SwinModel, \
#          ViTModel, MobileNetV2Model

class GeneralEncoder(nn.Module):
    def __init__(self, backbone = 'resnet18', pretrained = True, num_images=1, init_ch=3):
        super(GeneralEncoder, self).__init__()
        print("inside general encoder class")
        self.backbone = backbone
        # breakpoint()
        if 'resnet' in backbone:
            self.img_preprocessor = None
            self.encoder = ResNetEncoder(backbone=backbone,
                                         pretrained=pretrained,
                                         num_images = num_images,
                                         init_ch=init_ch)
            self.encoder_dims = 512
        elif backbone == 'efficientnet':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
            self.encoder = EfficientNetModel.from_pretrained("google/efficientnet-b0") 
            self.encoder_dims = 1280
        elif backbone == 'swinmodel':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.encoder = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.encoder_dims = 768
        elif backbone == 'vit':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self.encoder_dims = 768
        elif backbone == 'mobilenet':
            self.encoder_dims = 1280
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
            self.encoder = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        if 'resnet' in self.backbone:
            # print("in enc forward")
            # breakpoint()
            return self.encoder(x)
        # breakpoint()
        device = x.device
        x = self.img_preprocessor(x, return_tensors = 'pt')
        pixel_values = x['pixel_values'].to(device)
        enc_output = self.encoder(pixel_values=pixel_values)
        outputs = enc_output.last_hidden_state
        
        if self.backbone == 'vit':
            # reshaped_tensor.permute(0, 2, 1)[:,:,1:].reshape(-1, 768, 7, 7)
            reshaped_tensor = outputs.permute(0, 2, 1)[:, :, 1:].reshape(-1, 768, 14, 14)
            return reshaped_tensor
        
        if self.backbone == 'swinmodel':
            # breakpoint()
            reshaped_tensor = outputs.permute(0, 2, 1).reshape(-1, 768, 7, 7)
            return reshaped_tensor
        
        return outputs
            
        
class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, num_images=1, init_ch=3):
        super(ResNetEncoder, self).__init__()
        
        # Load the pre-trained ResNet model
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if(num_images > 1):
            self.model.conv1 = nn.Conv2d(init_ch*num_images, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.model.conv1.weight.device)
        self.layer0 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool)
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

    def forward(self, x):
        # Forward pass through each ResNet block
        x = x/255
        outputs = {}
        x0 = self.layer0(x)  # First downsample: output after conv1, bn1, relu, and maxpool
        x1 = self.layer1(x0)  # Second downsample: layer1
        x2 = self.layer2(x1)  # Third downsample: layer2
        x3 = self.layer3(x2)  # Fourth downsample: layer3
        x4 = self.layer4(x3)  # Final downsample: layer4

        outputs[0], outputs[1], outputs[2], outputs[3], outputs[4] = x0, x1, x2, x3, x4
        return outputs

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q = nn.Linear(config.query_dim, config.query_dim)
        self.k = nn.Linear(config.query_dim, config.query_dim)
        self.v = nn.Linear(config.query_dim, config.query_dim)
        assert config.attention_emb_dim % config.mha_heads == 0, "mha_heads must be divisible by attention_emb_dim"
        self.mha = nn.MultiheadAttention(config.attention_emb_dim, config.mha_heads, batch_first=True)
        self.out_linear = nn.Linear(config.attention_emb_dim, config.query_dim)
    
    def forward(self, q, k, v, return_attn_maps=False):
        out, attn_maps = self.mha(self.q(q), self.k(k), self.v(v), need_weights=return_attn_maps)
        # print(len(out), out[0].shape, out[1].shape)
        out = self.out_linear(out)
        if(return_attn_maps):
            return out, attn_maps
        return out

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.query_dim)
        self.norm2 = nn.LayerNorm(config.query_dim)
        # ca and sa block
        self.sa = Attention(config)
        self.ca = Attention(config)
        self.ff1 = nn.Linear(config.query_dim, 2*config.query_dim)
        self.ff2 = nn.Linear(2*config.query_dim, config.query_dim)
        
    def forward(self, queries, img_feats, return_attn_maps=False):
        queries = self.norm1(queries)
        queries_new = self.sa(queries, queries, queries)
        queries = queries_new + queries
        
        queries = self.norm2(queries)
        if(return_attn_maps):
            queries_new, attn_maps = self.ca(queries, img_feats, img_feats, return_attn_maps=return_attn_maps)
        else:
            queries_new = self.ca(queries, img_feats, img_feats, return_attn_maps=return_attn_maps)
        queries = queries_new + queries
        queries = self.ff2(F.relu(self.ff1(queries))) + queries
        if(return_attn_maps):
            return queries, attn_maps
        return queries
    
def fourier_embedding(x, D):
    # freqs = torch.tensor([2**i for i in range(D // 2)], dtype=torch.float32).to(x.device)[None]
    freqs = torch.tensor([i+1 for i in range(D // 2)], dtype=torch.float32).to(x.device)[None]
    emb_sin = torch.sin(freqs * x)
    emb_cos = torch.cos(freqs * x)
    embedding = torch.cat([emb_sin, emb_cos], dim=-1)
    
    return embedding

class SumMNISTModel2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.backbone = ResNetEncoder(config.backbone)
        self.patch_conv = nn.Conv2d(32, 128, kernel_size=self.config.patch_size, stride=self.config.patch_size)
        self.transformer = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_tr_layers)])
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, config.query_dim//2, kernel_size=3, padding=1)
        self.linear = nn.Linear(config.query_dim, 2)
        
        self.predefined_offsets = torch.linspace(0, 1, 6)[1:-1]
        self.fixed_offsets = torch.stack([torch.linspace(0, 1, 4)[:, None].repeat(1, 4), torch.linspace(0, 1, 4)[None].repeat(4, 1)]) # (2, 4, 4)
        print(self.predefined_offsets)
        exit(0)
        
        self.queries = nn.Parameter(torch.randn(4, config.query_dim))
        pos_embed = torch.arange(21)[:, None]
        self.pos_embed = fourier_embedding(pos_embed, config.query_dim//2)
        
        grid = torch.arange(10 ** 4).reshape(10, 10, 10, 10)
        indices = torch.where(grid >= 0)
        self.sum_tensor = torch.stack(indices).sum(0).reshape(10, 10, 10, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.max_pool2d(self.conv3(x), kernel_size=2)#.mean(-1).mean(-1).reshape(len(x), -1)
        x = x.mean(2).permute(0, 2, 1) # (flatten) # (b, 21, d)
        x = torch.cat([x, self.pos_embed[None].repeat(len(x), 1, 1).to(x)], -1)
        
        queries = self.queries[None].repeat(len(x), 1, 1)
        for tr in self.transformer:
            queries = tr(queries, x)

        logits = self.linear(queries)
        scores = F.softmax(logits, -1)
        # x = self.patch_conv(x).flatten(2).permute(0, 2, 1)
        # for tr in self.transformer:
        #     x = tr(x, x)
        # x = x.mean(1)
        # logits = logits # these would be normalized and of shape (b, 4, 10)
        
        
        # logits = self.linear(x)
        
        return scores
    
    def compute_loss(self, batch, validate=False):
        img, labels = batch['img'], batch['sum'].to(torch.long)
        pred = self(img[:, None]) # these would be normalized and of shape (b, 4, 10)
        b = len(img)
        sum_tensor = self.sum_tensor[None].repeat(b, 1, 1, 1, 1).to(self.conv1.weight.device)
        prob_prod =   pred[:, 0][:, :, None, None, None]\
                    * pred[:, 1][:, None, :, None, None]\
                    * pred[:, 2][:, None, None, :, None]\
                    * pred[:, 3][:, None, None, None, :]
        mask = sum_tensor == labels[:, None, None, None, None]
        loss = 0
        for i in range(b):
            loss += -torch.log(prob_prod[i][mask[i]].sum())

        loss = loss / len(img)
        # loss = F.cross_entropy(pred, labels)
        
        if(validate):
            acc = (self.get_acc(prob_prod, sum_tensor) == labels).sum() / len(labels)
        else:
            acc = None
        return {"loss": loss, "pred_label" : pred.argmax(-1), "acc" : acc}

    def get_acc(self, prob_prod, sum_tensor):
        # x would be normalized and of shape (b, 4, 10)
        labels = torch.ones(len(prob_prod)).to(torch.long).to(prob_prod.device) # (b)
        prob_mat = torch.zeros(len(prob_prod), 37).to(prob_prod)
        for i in range(37):
            mask = sum_tensor == (labels[:, None, None, None, None] * i)
            for j in range(len(prob_prod)):
                prob_mat[j, i] = prob_prod[j][mask[j]].sum()
        
        preds = torch.argmax(prob_mat, -1)
        return preds
    
    def validate(self, batch):
        return self.compute_loss(batch, validate=True)
    
    def infer(self, img):
        return self(img)

class SumMNISTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.backbone = ResNetEncoder(config.backbone)
        self.patch_conv = nn.Conv2d(32, 128, kernel_size=self.config.patch_size, stride=self.config.patch_size)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, config.query_dim//2, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, config.query_dim, kernel_size=3, padding=1)
        self.linear = nn.Linear(config.query_dim, 10)
        # self.linear = nn.Linear(config.query_dim, 4*10)
        
        self.transformer = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_tr_layers)])
        self.queries = nn.Parameter(torch.randn(4, config.query_dim))
        pos_embed = torch.arange(21)[:, None]
        self.pos_embed = fourier_embedding(pos_embed, config.query_dim//2)
        
        grid = torch.arange(10 ** 4).reshape(10, 10, 10, 10)
        indices = torch.where(grid >= 0)
        self.sum_tensor = torch.stack(indices).sum(0).reshape(10, 10, 10, 10)
        # for i in range(37):
        #     print(i, (self.sum_tensor == i).sum().item())
        # exit(0)
    
    def forward(self, x, labels=None):
        # to_viz = x.cpu().numpy()
        
        # for i in range(min(x.shape[0], 32)):
        #     cv2.imwrite(f"vis/vis_{i}_{labels[i].item()}.png", to_viz[i].transpose(1, 2, 0) * 255)
            
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.max_pool2d(self.conv3(x), kernel_size=2)#.mean(-1).mean(-1).reshape(len(x), -1)
        
        # x = x.mean(-1).mean(-1) # (flatten)
        
        x = x.mean(2).permute(0, 2, 1) # (flatten)
        x = torch.cat([x, self.pos_embed[None].repeat(len(x), 1, 1).to(x)], -1)
        
        queries = self.queries[None].repeat(len(x), 1, 1)
        for tr in self.transformer:
            queries = tr(queries, x)

        logits = self.linear(queries).reshape(-1, 4, 10)
        # logits = self.linear(x).reshape(-1, 4, 10)
        # scores = F.softmax(logits, -1)
        scores = F.softmax(logits, -1)
        return scores
    
    def compute_loss(self, batch, validate=False):
        img, labels = batch['img'], batch['sum'].to(torch.long)
        pred = self(img[:, None], labels) # these would be normalized and of shape (b, 4, 10)
        b = len(img)
        sum_tensor = self.sum_tensor[None].repeat(b, 1, 1, 1, 1).to(self.conv1.weight.device)
        prob_prod =   pred[:, 0][:, :, None, None, None]\
                    * pred[:, 1][:, None, :, None, None]\
                    * pred[:, 2][:, None, None, :, None]\
                    * pred[:, 3][:, None, None, None, :]
        mask = sum_tensor == labels[:, None, None, None, None]
        loss = 0
        for i in range(b):
            loss += -torch.log(prob_prod[i][mask[i]].sum())

        loss = loss / len(img)
        # loss = F.cross_entropy(pred, labels)
        
        if(validate):
            acc = (self.get_acc(prob_prod, sum_tensor) == labels).sum() / len(labels)
        else:
            acc = None
        return {"loss": loss, "pred_label" : pred.argmax(-1), "acc" : acc}

    def get_acc(self, prob_prod, sum_tensor):
        # x would be normalized and of shape (b, 4, 10)
        labels = torch.ones(len(prob_prod)).to(torch.long).to(prob_prod.device) # (b)
        prob_mat = torch.zeros(len(prob_prod), 37).to(prob_prod)
        
        for i in range(37):
            mask = sum_tensor == (labels[:, None, None, None, None] * i)
            for j in range(len(prob_prod)):
                prob_mat[j, i] = prob_prod[j][mask[j]].sum()
        
        preds = torch.argmax(prob_mat, -1)
        return preds
    
    def validate(self, batch):
        return self.compute_loss(batch, validate=True)
    
    def infer(self, img):
        return self(img)

class LITMNISTModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = torch.compile(SumMNISTModel(config))
        # self.model = torch.compile(SumMNISTModel2(config))
        self.save_hyperparameters()
        
    def training_step(self, batch, idx):
        # compute the training loss and log it to wandb and return it as well to update the model
        output = self.model.compute_loss(batch)
        train_loss = output['loss']
        
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=True)
        # self.log("train_acc", train_acc, sync_dist=True, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, idx):
        # log the validation_loss, visualization images to wandb
        data = batch
        output = self.model.validate(data)

        self.log("val_loss", output['loss'], sync_dist=True, prog_bar=True)
        self.log("val_acc", output['acc'], sync_dist=True, prog_bar=True)
        
        # if(idx == 0):
        #     pred_depth = self.model.infer(data['img'])
        #     vis_imgs = self.visualize(pred_depth, data['img'])
        #     for i, img in enumerate(vis_imgs):
        #         cv2.imwrite(f"vis/{i}.png", img)
        
    def testing_step(self, batch, idx):
        output = self.model.validate(batch)
        
        return output['acc']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.config.learning_rate)
        return optimizer