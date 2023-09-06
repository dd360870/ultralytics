
import os
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
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_pip_update_available, check_yaml
from ultralytics.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, ROOT, callbacks,
                                    is_git_dir, yaml_load)
from ultralytics.yolo.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TrackNetV4(DetectionModel):
    def init_criterion(self):
        # Replace with your custom loss function
        return TrackNetLoss(self)

class TrackNetLoss:
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

    def __call__(self, preds, batch):
        # preds = [[10*50*80*80]]
        preds = preds[0] # only pick first (stride = 16)
        batch_target = batch['target']
        loss = torch.zeros(2, device=self.device)  # box, cls, dfl
        batch_size = preds.shape[0]
        # for each batch
        for idx, pred in enumerate(preds):
            # pred = [50 * 80 * 80]
            pred_distri, pred_scores = torch.split(pred, [40, 10], dim=0)

            targets = pred_distri.clone().detach()
            cls_targets = torch.zeros(10, pred_scores.shape[1], pred_scores.shape[2])
            stride = self.stride[0]
            for idx, target in enumerate(batch_target[idx]):
                if target[1] == 1:
                    # xy
                    grid_x, grid_y, offset_x, offset_y = targetGrid(target[2], target[3], stride)
                    targets[4*idx, grid_y, grid_x] = offset_x
                    targets[4*idx + 1, grid_y, grid_x] = offset_y
                    targets[4*idx + 2, grid_y, grid_x] = target[4]
                    targets[4*idx + 3, grid_y, grid_x] = target[5]

                    ## cls
                    cls_targets[idx, grid_y, grid_x] = 1
            weight = 10
            loss[0] += weight * F.mse_loss(pred_distri, targets, reduction='mean')
            loss[1] += focal_loss(pred_scores, cls_targets, alpha=[0.94, 0.06], weight=weight)

        return loss.sum() * batch_size, loss.detach()

def targetGrid(target_x, target_y, stride):
    grid_x = int(target_x / stride)
    grid_y = int(target_y / stride)
    offset_x = (target_x / stride) - grid_x
    offset_y = (target_y / stride) - grid_y
    return grid_x, grid_y, offset_x, offset_y

def focal_loss(pred_logits, targets, alpha=0.95, gamma=4.0, epsilon=1e-6, weight=10):
    """
    :param pred_logits: 預測的logits, shape [batch_size, 1, H, W]
    :param targets: 真實標籤, shape [batch_size, 1, H, W]
    :param alpha: 用於平衡正、負樣本的權重。這裡可以是一個scalar或一個list[alpha_neg, alpha_pos]。
    :param gamma: 用於調節著重於正確或錯誤預測的程度
    :return: focal loss
    """
    pred_probs = torch.sigmoid(pred_logits)
    # pred_probs = torch.clamp(pred_probs, epsilon, 1.0 - epsilon)
    if isinstance(alpha, (list, tuple)):
        alpha_neg = alpha[0]
        alpha_pos = alpha[1]
    else:
        alpha_neg = (1 - alpha)
        alpha_pos = alpha

    pt = torch.where(targets == 1, pred_probs, 1 - pred_probs)
    alpha_t = torch.where(targets == 1, alpha_pos, alpha_neg)
    
    ce_loss = -torch.log(pt)
    fl = alpha_t * (1 - pt) ** gamma * ce_loss
    fl = torch.where(targets == 1, fl * weight, fl)
    return fl.mean()


class CustomTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return TrackNetDataset(root_dir=r"C:\Users\user1\bartek\github\BartekTao\ultralytics\tracknet\train_data")

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = TrackNetV4(cfg, ch=10, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    def preprocess_batch(self, batch):
        return batch
    #def get_validator(self):
    def update_metrics(self, preds, batch):
        return


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
                ball_trajectory_df['dX'] = (-ball_trajectory_df['X'].iloc[::-1].diff().iloc[::-1]).fillna(0)
                ball_trajectory_df['dY'] = (-ball_trajectory_df['Y'].iloc[::-1].diff().iloc[::-1]).fillna(0)

                # ball_trajectory_df['dX'] = ball_trajectory_df['X'].diff().fillna(0)
                # ball_trajectory_df['dY'] = ball_trajectory_df['Y'].diff().fillna(0)
                ball_trajectory_df = ball_trajectory_df.drop(['Fast'], axis=1)

                # Get the directory for frames of this video
                frame_dir = os.path.join(match_dir_path, 'frame', video_file_name)

                # Get all frame file names and sort them by frame ID
                frame_files = sorted(os.listdir(frame_dir), key=lambda x: int(os.path.splitext(x)[0]))

                # Create sliding windows of 10 frames
                for i in range(len(frame_files) - (num_input-1)):
                    frames = [os.path.join(frame_dir, frame) for frame in frame_files[i: i + num_input]]
                    ball_trajectory = ball_trajectory_df.iloc[i: i + num_input].values
                    self.samples.append((frames, ball_trajectory))
            # check label result
            # idx = np.random.randint(50, 100)
            # frames, ball_trajectory = self.samples[206]
            # ball_trajectory = torch.from_numpy(ball_trajectory)
            # # Load images and convert them to tensors
            # images = [self.open_image(frame) for frame in frames]

            # idx = np.random.randint(0, 9)
            # i = images[idx]
            # xy = [(ball_trajectory[idx][2].item(), ball_trajectory[idx][3].item())]
            #self.display_image_with_coordinates(i, xy)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, ball_trajectory = self.samples[idx]
        ball_trajectory = torch.from_numpy(ball_trajectory)
        ball_trajectory = self.transform_coordinates(ball_trajectory, 1280, 720)
        # Load images and convert them to tensors
        images = [self.open_image(frame) for frame in frames]
        images = [self.pad_to_square(img, 0) for img in images]

        # check resize image
        # idx = np.random.randint(0, 10)
        # i = images[idx]
        # xy = [(ball_trajectory[idx][2].item(), ball_trajectory[idx][3].item())]
        # self.display_image_with_coordinates(i, xy)

        images = torch.cat(images, 0)  # Concatenate along the channel dimension

        # Convert ball_trajectory to tensor
        
        shape = images.shape
        shape = ball_trajectory.shape
        return {"img": images, "target": ball_trajectory}

    def transform_coordinates(self, data, w, h, target_size=640):
        """
        Transform the X, Y coordinates in data based on image resizing and padding.
        
        Parameters:
        - data (torch.Tensor): A tensor of shape (N, 6) with columns (Frame, Visibility, X, Y, dx, dy).
        - w (int): Original width of the image.
        - h (int): Original height of the image.
        - target_size (int): The desired size for the longest side after resizing.
        
        Returns:
        - torch.Tensor: A transformed tensor of shape (N, 6).
        """
        
        # Clone the data to ensure we don't modify the original tensor in-place
        data_transformed = data.clone()
        
        # Determine padding
        max_dim = max(w, h)
        pad_diff = max_dim - min(w, h)
        pad1 = pad_diff // 2
        
        # Indices where x and y are not both 0
        indices_to_transform = (data[:, 2] != 0) | (data[:, 3] != 0)
        
        # Adjust for padding
        if h < w:
            data_transformed[indices_to_transform, 3] += pad1
        else:
            data_transformed[indices_to_transform, 2] += pad1  # if height is greater, adjust X

        # Adjust for scaling
        scale_factor = target_size / max_dim
        data_transformed[:, 2] *= scale_factor  # scale X
        data_transformed[:, 3] *= scale_factor  # scale Y
        data_transformed[:, 4] *= scale_factor  # scale dx
        data_transformed[:, 5] *= scale_factor  # scale dy
        
        return data_transformed
    def display_image_with_coordinates(self, img_tensor, coordinates):
        """
        Display an image with annotated coordinates.

        Parameters:
        - img_tensor (torch.Tensor): The image tensor of shape (C, H, W) or (H, W, C).
        - coordinates (list of tuples): A list of (X, Y) coordinates to be annotated.
        """
        
        # Convert the image tensor to numpy array
        img_array = img_tensor.permute(1, 2, 0).cpu().numpy()

        # Create a figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(img_array)

        # Plot each coordinate
        for (x, y) in coordinates:
            ax.scatter(x, y, s=50, c='red', marker='o')
            # Optionally, you can also draw a small rectangle around each point
            rect = patches.Rectangle((x-5, y-5), 10, 10, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        plt.show()

    def open_image(self, path):
        # Open the image file
        img = Image.open(path).convert('L')
        
        # Reduce the resolution to half
        width, height = img.size
        img = img.resize((width // 2, height // 2))
        
        # Convert the image to a tensor
        return transforms.ToTensor()(img)

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

# from ultralytics import YOLO

# # Create a new YOLO model from scratch
# model = YOLO(r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\yolov8.yaml')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)
# # results = model.train(data='tracknet.yaml', epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()

overrides = {}
overrides['model'] = r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\yolov8.yaml'
overrides['mode'] = 'train'
# overrides['data'] = 'coco128.yaml'
overrides['data'] = 'tracknet.yaml'
overrides['epochs'] = 3
overrides['plots'] = False
overrides['batch'] = 10
# overrides={"OPTIMIZER.LR": 0.001, "DATALOADER.BATCH_SIZE": 16}
# torch.autograd.set_detect_anomaly(True)
# trainer = DetectionTrainer(overrides=overrides)
trainer = CustomTrainer(overrides=overrides)
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
trainer.train()