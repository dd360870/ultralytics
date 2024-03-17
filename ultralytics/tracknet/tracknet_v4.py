import copy
import argparse

import torch
from .val import TrackNetValidator
from .dataset import TrackNetDataset
from ultralytics.yolo.utils import RANK
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from .utils.loss import TrackNetLoss
from ultralytics.nn.tasks import DetectionModel

class TrackNetV4(DetectionModel):
    def init_criterion(self):
        return TrackNetLoss(self)