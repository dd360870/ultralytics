import copy
import argparse

import torch
from val import TrackNetValidator
from dataset import TrackNetDataset
from ultralytics.yolo.utils import RANK
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from utils.loss import TrackNetLoss
from ultralytics.nn.tasks import DetectionModel

class TrackNetV4(DetectionModel):
    def init_criterion(self):
        return TrackNetLoss(self)
    
class TrackNetTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return TrackNetDataset(root_dir=img_path)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = TrackNetV4(cfg, ch=10, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch
    def get_validator(self):
        return TrackNetValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    
def log_model(trainer):
    last_weight_path = trainer.last
    torch.save(trainer.model.state_dict(), last_weight_path)

def main(model_path, mode, data, epochs, plots, batch):
    overrides = {}
    overrides['model'] = model_path
    overrides['mode'] = mode
    overrides['data'] = data
    overrides['epochs'] = epochs
    overrides['plots'] = plots
    overrides['batch'] = batch
    trainer = TrackNetTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a custom model with overrides.')
    
    parser.add_argument('--model_path', type=str, default='', help='Path to the model')
    parser.add_argument('--mode', type=str, default='train', help='Mode for the training (e.g., train, test)')
    parser.add_argument('--data', type=str, default='tracknet.yaml', help='Data configuration (e.g., tracknet.yaml)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--plots', type=bool, default=False, help='Whether to plot or not')
    parser.add_argument('--batch', type=int, default=20, help='Batch size')
    
    args = parser.parse_args()
    args.model_path = r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\tracknetv4.yaml'
    main(args.model_path, args.mode, args.data, args.epochs, args.plots, args.batch)