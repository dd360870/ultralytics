
import argparse
from copy import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
from ultralytics.yolo.data import dataloaders
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics
from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.yolo.utils import LOGGER, RANK, ops
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
from pathlib import Path

#imagePath = r"C:\Users\user1\bartek\github\BartekTao\ultralytics\tracknet\train_data"
#modelPath = r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\tracknetv4.yaml'

weight_pos = 1
weight_mov = 1
weight_conf = 1000
# check_training_img_path = r'C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\check_training_img\img_'
check_training_img_path = r'/usr/src/datasets/tracknet/visualize_train_img/img_'
mode_flag = 'train'

class TrackNetV4(DetectionModel):
    def init_criterion(self):
        return TrackNetLoss(self)

class TrackNetLoss:
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        pos_weight = torch.tensor(1000).to(device)
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        self.mse = nn.MSELoss()
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.batch_count = 0

    def __call__(self, preds, batch):
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[0].to(self.device) # only pick first (stride = 16)
        batch_target = batch['target'].to(self.device)
        batch_img = batch['img'].to(self.device)

        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        batch_size = preds.shape[0]

        # check
        rand_batch = np.random.randint(0, 16) if batch_target.shape[0] == 16 else batch_target.shape[0]-1

        # for each batch
        for idx, pred in enumerate(preds):
            # pred = [50 * 20 * 20]
            stride = self.stride[0]
            pred_distri, pred_scores = torch.split(pred, [40, 10], dim=0)
            pred_pos, pred_mov = torch.split(pred_distri, [20, 20], dim=0)

            pred_pos = torch.sigmoid(pred_pos)
            pred_mov = torch.tanh(pred_mov)
            
            cls_targets = torch.zeros(pred_scores.shape, device=self.device)
            
            position_loss = torch.tensor(0.0, device=self.device)
            move_loss = torch.tensor(0.0, device=self.device)

            pred_xy_list = []
            target_xy_list = []

            pred_dxdy_list = []
            target_dxdy_list = []
            for target_idx, target in enumerate(batch_target[idx]):
                if target[1] == 1:
                    # xy
                    grid_x, grid_y, offset_x, offset_y = targetGrid(target[2], target[3], stride)

                    pred_x = pred_pos[2*target_idx, grid_y, grid_x]
                    pred_y = pred_pos[2*target_idx + 1, grid_y, grid_x]
                    pred_xy_list.append(torch.tensor([pred_x, pred_y]))
                    target_xy_list.append(torch.tensor([offset_x/stride, offset_y/stride]))

                    pred_dx = pred_mov[2*target_idx, grid_y, grid_x]
                    pred_dy = pred_mov[2*target_idx + 1, grid_y, grid_x]
                    pred_dxdy_list.append(torch.tensor([pred_dx, pred_dy]))
                    target_dxdy_list.append(torch.tensor([target[4]/640, target[5]/640]))

                    ## cls
                    cls_targets[target_idx, grid_y, grid_x] = 1
            if len(pred_xy_list) > 0:
                pred_xy_tensor = torch.stack(pred_xy_list, dim=0)
                target_xy_tensor = torch.stack(target_xy_list, dim=0)
                position_loss = self.mse(pred_xy_tensor, target_xy_tensor)

            if len(pred_dxdy_list) > 0:
                pred_dxdy_tensor = torch.stack(pred_dxdy_list, dim=0)
                target_dxdy_tensor = torch.stack(target_dxdy_list, dim=0)
                move_loss = self.mse(pred_dxdy_tensor, target_dxdy_tensor)

            # target_scores_sum = max(cls_targets.sum(), 1)
            # test = torch.zeros(pred_scores.shape, device=self.device)
            # target = torch.ones([10, 1], dtype=torch.float32)  # 64 classes, batch size = 10
            # output = torch.full([10, 1], 1)  # A prediction (logit)
            # conf_loss = self.bce(output, target).sum()
            # conf_loss = self.bce(pred_scores, cls_targets).sum() / target_scores_sum

            conf_loss = custom_loss(cls_targets, pred_scores, [1, weight_conf])
            # conf_loss = focal_loss(pred_scores, cls_targets, alpha=[0.998, 0.002], weight=weight_conf)
            if torch.isnan(position_loss).any() or torch.isinf(position_loss).any():
                LOGGER.warning("NaN or Inf values in position_loss!")
            if torch.isnan(conf_loss).any() or torch.isinf(conf_loss).any():
                LOGGER.warning("NaN or Inf values in conf_loss!")

            # check
            if rand_batch == idx and mode_flag == 'train' and self.batch_count%90 == 0:
                for rand_idx in range(1):
                    pred_conf = torch.sigmoid(pred_scores[rand_idx]).cpu()
                    img = batch_img[rand_batch][rand_idx]
                    x = (batch_target[rand_batch][rand_idx][2].item() // 32)*32
                    y = (batch_target[rand_batch][rand_idx][3].item() // 32)*32
                    max_position = torch.argmax(pred_conf)
                    max_x, max_y = np.unravel_index(max_position, pred_conf.shape)
                    filename = f'{self.batch_count//979}_{int(self.batch_count%979)}'

                    count_ge_05 = np.count_nonzero(pred_conf >= 0.5)
                    count_lt_05 = np.count_nonzero(pred_conf < 0.5)
                    loss_list = [conf_loss.item()]
                    loss_list.append(count_ge_05)
                    loss_list.append(count_lt_05)
                    loss_list.append(pred_conf[int(x/32)][int(y/32)])

                    display_image_with_coordinates(img, [(x, y)], [(max_x*32, max_y*32)], filename, loss_list)

            #loss[0] += position_loss * weight_pos
            #loss[1] += move_loss * weight_mov
            loss[2] += conf_loss
        tlose = loss.sum() * batch_size
        tlose_item = loss.detach()
        # LOGGER.info(f'tloss: {tlose}, tlose_item: {tlose_item}')
        self.batch_count+=1
        return tlose, tlose_item

def targetGrid(target_x, target_y, stride):
    grid_x = int(target_x / stride)
    grid_y = int(target_y / stride)
    offset_x = (target_x % stride)
    offset_y = (target_y % stride)
    return grid_x, grid_y, offset_x, offset_y

def custom_loss(y_true, y_pred, class_weight):
    y_pred = torch.sigmoid(y_pred)  

    custom_weights = torch.square(1 - y_pred) * y_true + torch.square(y_pred) * (1 - y_true)

    class_weights = class_weight[0] * (1 - y_true) + class_weight[1] * y_true

    loss = (-1) * class_weights * custom_weights * (y_true * torch.log(torch.clamp(y_pred, min=torch.finfo(y_pred.dtype).eps, max=1)) + 
                                                    (1 - y_true) * torch.log(torch.clamp(1 - y_pred, min=torch.finfo(y_pred.dtype).eps, max=1)))
    penalty = (y_true * (1 - y_pred) * 100)

    return torch.mean(loss+penalty)

def focal_loss(pred_logits, targets, alpha=0.95, gamma=2.0, epsilon=1e-3, weight=10):
    """
    :param pred_logits: 預測的logits, shape [batch_size, 1, H, W]
    :param targets: 真實標籤, shape [batch_size, 1, H, W]
    :param alpha: 用於平衡正、負樣本的權重。這裡可以是一個scalar或一個list[alpha_neg, alpha_pos]。
    :param gamma: 用於調節著重於正確或錯誤預測的程度
    :return: focal loss
    """
    pred_probs = torch.sigmoid(pred_logits)

    pred_probs = torch.clamp(pred_probs, epsilon, 1.0-epsilon)  # log(0) 會導致無窮大
    if isinstance(alpha, (list, tuple)):
        alpha_neg = alpha[0]
        alpha_pos = alpha[1]
    else:
        alpha_neg = (1 - alpha)
        alpha_pos = alpha

    pt = torch.where(targets == 1, pred_probs, 1 - pred_probs)
    alpha_t = torch.where(targets == 1, alpha_pos, alpha_neg)
    
    ce_loss = -torch.log(pt)
    # if torch.isinf(ce_loss).any():
    #     LOGGER.warning("ce_loss value is infinite!")
    fl = alpha_t * (1 - pt) ** gamma * ce_loss
    test = fl.mean() * weight_conf
    return test

    foreground_loss = 0
    background_loss = 0
    if (targets == 1).sum() > 0:
        mask1 = (targets == 1) & (pred_probs <= 0.6)
        mask2 = (targets == 1) & (pred_probs > 0.6)
        
        if mask1.sum() > 0:
            loss = fl[mask1].mean() * weight
            foreground_loss += loss

        if mask2.sum() > 0:
            loss = fl[mask2].mean()
            foreground_loss += loss
    mask1 = (targets == 0) & (pred_probs <= 0.6)
    mask2 = (targets == 0) & (pred_probs > 0.6)
    if mask1.sum() > 0:
        background_loss += fl[mask1].mean()
    if mask2.sum() > 0:
        background_loss += fl[mask2].mean() * weight

    combined_loss = foreground_loss + background_loss
    if combined_loss < 0:
        LOGGER.warning("combined_loss < 0")
    return combined_loss


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
        self.loss_names = 'pos_loss', 'mov_loss', 'conf_loss'
        return TrackNetValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (3 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Size')


# def log_model(trainer):
#     LOGGER.info()

class TrackNetValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.acc = 0
    
    def get_dataloader(self, dataset_path, batch_size):
        """For TrackNet, we can use the provided TrackNetDataset to get the dataloader."""
        dataset = TrackNetDataset(root_dir=dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
    
    def preprocess(self, batch):
        """In this case, the preprocessing step is mainly handled by the dataloader."""
        return batch
    
    def postprocess(self, preds):
        """Postprocess the model predictions if needed."""
        # For TrackNet, there might not be much postprocessing needed.
        return preds
    
    def init_metrics(self, model):
        mode_flag = 'valid'
        """Initialize some metrics."""
        # Placeholder for any metrics you might want to use.
        self.total_loss = 0.0
        self.num_samples = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.acc = 0
        self.hasMax = 0
        self.hasBall = 0
    
    def update_metrics(self, preds, batch):
        """Calculate and update metrics based on predictions and batch."""
        # Placeholder for loss calculation, etc.
        # preds = [[batch*50*20*20]]
        # batch['target'] = [batch*10*6]
        preds = preds[0] # only pick first (stride = 32)
        batch_target = batch['target']
        batch_size = preds.shape[0]
        if preds.shape == (50, 20, 20):
            self.update_metrics_once(0, preds, batch_target)
        else:
            # for each batch
            for idx, pred in enumerate(preds):
                self.update_metrics_once(idx, pred, batch_target[idx])
        #print((self.TP, self.FP, self.FN))
    def update_metrics_once(self, batch_idx, pred, batch_target):
        # pred = [50 * 20 * 20]
        # batch_target = [10*6]
        pred_distri, pred_scores = torch.split(pred, [40, 10], dim=0)
        pred_probs = torch.sigmoid(pred_scores)
        # pred_probs = [10*20*20]
        
        pred_pos, pred_mov = torch.split(pred_distri, [20, 20], dim=0)
        # pred_pos = torch.sigmoid(pred_pos)
        # pred_mov = torch.tanh(pred_mov)

        max_values_dim1, max_indices_dim1 = pred_probs.max(dim=2)
        final_max_values, max_indices_dim2 = max_values_dim1.max(dim=1)
        max_positions = [(index.item(), max_indices_dim1[i, index].item()) for i, index in enumerate(max_indices_dim2)]

        #targets = pred_distri.clone().detach()
        #cls_targets = torch.zeros(10, pred_scores.shape[1], pred_scores.shape[2])
        stride = 32
        for idx, target in enumerate(batch_target):
            if target[1] == 1:
                # xy
                grid_x, grid_y, offset_x, offset_y = targetGrid(target[2], target[3], stride)
                if (grid_x > 20 or grid_y > 20):
                    LOGGER.Warning("target grid transform error")
                if (pred_probs[idx][grid_x][grid_y] > 0.5):
                    self.hasBall += 1
                
                # print(f"target: {(grid_x, grid_y, offset_x, offset_y)}, ")
                # print(f"predict_conf: {pred_probs[idx][grid_x][grid_y]}, ")
                # print(f"pred_pos: {pred_pos[idx][grid_x][grid_y]}")
                # print(pred_probs[idx][max_positions[idx]])
                # print(max_positions[idx])
                if pred_probs[idx][max_positions[idx]] > 0.5:
                    self.hasMax += 1
                    x, y = max_positions[idx]
                    real_x = x*stride + pred_pos[idx][x][y] #*stride
                    real_y = y*stride + pred_pos[idx][x][y] #*stride
                    if (grid_x, grid_y) == max_positions[idx]:
                        self.TP+=1
                    else:
                        self.FN+=1
                else:
                    self.FN+=1
            elif pred_probs[idx][max_positions[idx]] > 0.5:
                self.FP+=1
            else:
                self.TN+=1
    def finalize_metrics(self):
        """Calculate final metrics for this validation run."""
        self.acc = (self.TN + self.TP) / (self.FN+self.FP+self.TN + self.TP)

    def get_stats(self):
        """Return the stats."""
        return {'FN': self.FN, 'FP': self.FP, 'TN': self.TN, 'TP': self.TP, 'acc': self.acc, 'max_conf>0.5': self.hasMax, 'correct_cell>0.5':self.hasBall}
    
    def print_results(self):
        mode_flag = 'train'

        """Print the results."""
        precision = 0
        recall = 0
        f1 = 0
        if self.TP > 0:
            precision = self.TP/(self.TP+self.FP)
            recall = self.TP/(self.TP+self.FN)
            f1 = (2*precision*recall)/(precision+recall)
        print(f"Validation Accuracy: {self.acc:.4f}, Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}, , Validation F1-Score: {f1:.4f}")

    def get_desc(self):
        """Return a description for tqdm progress bar."""
        return "Validating TrackNet"


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
            ball_trajectory_dir = os.path.join(match_dir_path, 'csv')

            # Traverse all videos in the match directory
            for video_file in os.listdir(video_dir):
                # Get the video file name (without extension)
                video_file_name = os.path.splitext(video_file)[0]

                # The ball_trajectory csv file has the same name as the video file
                ball_trajectory_file = os.path.join(ball_trajectory_dir, video_file_name + "_ball" + '.csv')
                
                # Read the ball_trajectory csv file
                ball_trajectory_df = pd.read_csv(ball_trajectory_file)
                ball_trajectory_df['dX'] = ball_trajectory_df['X'].diff().fillna(0)
                ball_trajectory_df['dY'] = ball_trajectory_df['Y'].diff().fillna(0)
                
                ball_trajectory_df = ball_trajectory_df.drop(['Fast'], axis=1)

                # Get the directory for frames of this video
                frame_dir = os.path.join(match_dir_path, 'frame', video_file_name)

                # Get all frame file names and sort them by frame ID
                frame_files = sorted(os.listdir(frame_dir), key=lambda x: int(os.path.splitext(x)[0]))

                # Create sliding windows of num_input frames
                for i in range(len(frame_files) - (num_input-1)):
                    frames = [os.path.join(frame_dir, frame) for frame in frame_files[i: i + num_input]]
                    ball_trajectory = ball_trajectory_df.iloc[i: i + num_input].values
                    # Avoid invalid data
                    if len(frames) == num_input and len(ball_trajectory) == num_input:
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

class TrackNetPredictor(BasePredictor):
    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

# from ultralytics import YOLO

# # Create a new YOLO model from scratch
# model = YOLO(r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\yolov8.yaml')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)
# # results = model.train(data='tracknet.yaml', epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()

def display_image_with_coordinates(img_tensor, coordinates, p_coordinates, fileName, input_number = None):
    """
    Display an image with annotated coordinates.

    Parameters:
    - img_tensor (torch.Tensor): The image tensor of shape (C, H, W) or (H, W, C).
    - coordinates (list of tuples): A list of (X, Y) coordinates to be annotated.
    """
    
    # Convert the image tensor to numpy array
    img_array = img_tensor.cpu().numpy()

    # Create a figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img_array)

    img_height, img_width = img_array.shape[:2]

    # Plot each coordinate
    for (x, y) in coordinates:
        ax.scatter(x, y, s=50, c='red', marker='o')
        # Optionally, you can also draw a small rectangle around each point
        rect = patches.Rectangle((x-5, y-5), 10, 10, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    for (x, y) in p_coordinates:
        ax.scatter(x, y, s=50, c='blue', marker='o')
        # Optionally, you can also draw a small rectangle around each point
        rect = patches.Rectangle((x-5, y-5), 10, 10, linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    # for i in range(p_array.shape[0]):
    #     for j in range(p_array.shape[1]):
    #         # Scaling the coordinates
    #         scaled_x = int(j * img_width / p_array.shape[1])
    #         scaled_y = int(i * img_height / p_array.shape[0])

    #         # Plotting the value
    #         ax.text(scaled_x, scaled_y, str(p_array[i, j]), color='blue', fontsize=8)
    if input_number:
        ax.text(img_width * 0.9, img_height * 0.1, str(input_number), color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    # plt.show()

    plt.savefig(check_training_img_path+fileName, bbox_inches='tight')
    plt.close()

def main(model_path, mode, data, epochs, plots, batch, source):
    overrides = {}
    overrides['model'] = model_path
    overrides['mode'] = mode
    overrides['data'] = data
    overrides['epochs'] = epochs
    overrides['plots'] = plots
    overrides['batch'] = batch
    overrides['patience'] = 300

    if mode == 'train':
        trainer = TrackNetTrainer(overrides=overrides)
        # trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
        trainer.train()
    elif mode == 'predict':
        model, _ = attempt_load_one_weight(model_path)
        worker = 0
        if torch.cuda.is_available():
            model.cuda()
            worker = 1
        dataset = TrackNetDataset(root_dir=source)
        dataloader = build_dataloader(dataset, batch, worker, shuffle=False, rank=-1)
        overrides = overrides.copy()
        overrides['save'] = False
        predictor = TrackNetPredictor(overrides=overrides)
        predictor.setup_model(model=model, verbose=False)
        pbar = enumerate(dataloader)
        elapsed_times = 0.0
        for i, batch in pbar:
            target = batch['target'][0]
            input_data = batch['img']
            idx = np.random.randint(0, 10)
            hasBall = target[idx][1].item()
            t_x = target[idx][2].item()
            t_y = target[idx][3].item()
            xy = [(t_x, t_y)]
            
            if torch.cuda.is_available():
                input_data = input_data.cuda()
            #start_time = time.time()

            # [1*50*20*20]
            p = predictor.inference(input_data)
            # [5*20*20]
            p_check = p[0, 5*idx:5*(idx+1), :]
            p_conf = torch.sigmoid(p_check[4, :, :])
            p_cell_x = torch.sigmoid(p_check[0, :, :])
            p_cell_y = torch.sigmoid(p_check[1, :, :])

            max_position = torch.argmax(p_conf)
            max_x, max_y = np.unravel_index(max_position, p_conf.shape)
            p_x = p_cell_x[max_x, max_y]*32
            p_y = p_cell_y[max_x, max_y]*32
            p_gridx = max_x*32 + p_cell_x[max_x, max_y]*32
            p_gridy = max_y*32 + p_cell_y[max_x, max_y]*32
            x, y, gx, gy = targetGrid(t_x, t_y, 32)

            if hasBall and i%10 == 0:
                pos_weight = torch.tensor([400])
                l = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
                cls_targets = torch.zeros(p_conf.shape[0], p_conf.shape[1])
                cls_targets[x][y] = 1
                cls_pred = p_check[4, :, :]
                count_ge_05 = np.count_nonzero(p_conf >= 0.5)
                count_lt_05 = np.count_nonzero(p_conf < 0.5)
                correct = True if p_conf[x][y]>=0.5 else False
                loss = l(cls_pred, cls_targets).sum()
                loss_list = [loss.item()]
                loss_list.append(count_ge_05)
                loss_list.append(count_lt_05)
                loss_list.append(correct)
                loss_list.append(p_conf[x][y])
                display_image_with_coordinates(input_data[0][idx], [(x*32, y*32)], [(max_x*32, max_y*32)], str(i), loss_list)
            #end_time = time.time()
            #elapsed_time = (end_time - start_time) * 1000
            #print(f'{elapsed_time:.2f}ms')
            #elapsed_times+=elapsed_time

        print(f"程序運行了 {elapsed_times:.2f} 毫秒, 平均一個batch {(elapsed_times)/len(dataloader):.2f} ms")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a custom model with overrides.')
    
    parser.add_argument('--model_path', type=str, default=r'C:\Users\user1\bartek\github\BartekTao\ultralytics\ultralytics\models\v8\tracknetv4.yaml', help='Path to the model')
    parser.add_argument('--mode', type=str, default='train', help='Mode for the training (e.g., train, test)')
    parser.add_argument('--data', type=str, default='tracknet.yaml', help='Data configuration (e.g., tracknet.yaml)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--plots', type=bool, default=False, help='Whether to plot or not')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--source', type=str, default=r'C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\train_data', help='source')
    
    args = parser.parse_args()
    # args.epochs = 50
    # args.batch = 1
    # args.mode = 'predict'
    # args.model_path = r'C:\Users\user1\bartek\github\BartekTao\ultralytics\runs\detect\prod_train81\weigths\last.pt'
    # args.source = r'C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\val_data'
    main(args.model_path, args.mode, args.data, args.epochs, args.plots, args.batch, args.source)