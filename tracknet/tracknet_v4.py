import glob
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tracknet.dataset import TrackNetDataset
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.utils import LOGGER, RANK
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

class TrackNetV4(DetectionModel):
    def init_criterion(self):
        # Replace with your custom loss function
        return loss_fn()

def loss_fn(y_pred, y_true):
    ...


class CustomTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return TrackNetDataset(root_dir=r"C:\Users\user1\bartek\github\BartekTao\ultralytics\tracknet\train_data")

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, ch=30, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

def log_model(trainer):
    last_weight_path = trainer.last
    torch.save(trainer.model.state_dict(), last_weight_path)

# overrides={"OPTIMIZER.LR": 0.001, "DATALOADER.BATCH_SIZE": 16}
trainer = CustomTrainer()
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
trainer.train()