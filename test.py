import glob
import os
import random
import numpy as np
import torch
import torch.nn as nn
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.utils import LOGGER, RANK
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

# class TrackNetV4(DetectionModel):
#     def init_criterion(self):
#         # Replace with your custom loss function
#         return loss_fn()

# def loss_fn(y_pred, y_true):
#     ...


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

class TrackNetDataset(Dataset):
    def __init__(self, root_dir, num_input=10, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_input = num_input
        self.samples = []

        # Traverse all matches
        for match_dir in os.listdir(root_dir):
            match_dir_path = os.path.join(root_dir, match_dir)
            
            # Check if it is a match directory
            if not os.path.isdir(match_dir_path):
                continue
            
            video_dir = os.path.join(match_dir_path, 'video')
            ball_trajectory_dir = os.path.join(match_dir_path, 'ball_trajectory')

            # Traverse all videos in the match directory
            for video_file in os.listdir(video_dir):
                # Get the video file name (without extension)
                video_file_name = os.path.splitext(video_file)[0]

                # The ball_trajectory csv file has the same name as the video file
                ball_trajectory_file = os.path.join(ball_trajectory_dir, video_file_name + '.csv')
                
                # Read the ball_trajectory csv file
                ball_trajectory_df = pd.read_csv(ball_trajectory_file)
                ball_trajectory_df['dX'] = ball_trajectory_df['X'].diff().fillna(0)
                ball_trajectory_df['dY'] = ball_trajectory_df['Y'].diff().fillna(0)
                ball_trajectory_df = ball_trajectory_df.drop(['Visibility', 'Fast'], axis=1)

                # Get the directory for frames of this video
                frame_dir = os.path.join(match_dir_path, 'frame', video_file_name)

                # Get all frame file names and sort them by frame ID
                frame_files = sorted(os.listdir(frame_dir), key=lambda x: int(os.path.splitext(x)[0]))

                # Create sliding windows of 10 frames
                for i in range(len(frame_files) - (num_input-1)):
                    frames = [os.path.join(frame_dir, frame) for frame in frame_files[i: i + num_input]]
                    ball_trajectory = ball_trajectory_df.iloc[i: i + num_input].values
                    self.samples.append((frames, ball_trajectory))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, ball_trajectory = self.samples[idx]
        
        # Load images and convert them to tensors
        images = [self.open_image(frame) for frame in frames]
        images = [self.pad_to_square(img, 0) for img in images]
        images = torch.cat(images, 0)  # Concatenate along the channel dimension

        # Convert ball_trajectory to tensor
        ball_trajectory = torch.from_numpy(ball_trajectory)
        
        return images, ball_trajectory

    def open_image(self, path):
        # Open the image file, convert it to RGB, and then to a tensor
        return transforms.ToTensor()(Image.open(path).convert('RGB'))

    def pad_to_square(self, img, pad_value):
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=pad_value)

        return img

from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO(r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\yolov8.yaml')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='coco128.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# overrides={"OPTIMIZER.LR": 0.001, "DATALOADER.BATCH_SIZE": 16}
trainer = CustomTrainer(r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\yolov8.yaml')
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
trainer.train()