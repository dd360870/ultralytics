from matplotlib import patches, pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

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
                ball_trajectory_df['dX'] = -1*ball_trajectory_df['X'].diff(-1).fillna(0)
                ball_trajectory_df['dY'] = -1*ball_trajectory_df['Y'].diff(-1).fillna(0)
                ball_trajectory_df['hit'] = ((ball_trajectory_df['Event'] == 1) | (ball_trajectory_df['Event'] == 2)).astype(int)

                ball_trajectory_df['prev_hit'] = ball_trajectory_df['hit'].shift(fill_value=0)
                ball_trajectory_df['next_hit'] = ball_trajectory_df['hit'].shift(-1, fill_value=0)
                ball_trajectory_df['hit'] = ball_trajectory_df[['hit', 'prev_hit', 'next_hit']].max(axis=1)
                
                ball_trajectory_df = ball_trajectory_df.drop(['Fast'], axis=1)
                ball_trajectory_df = ball_trajectory_df.drop(['Event'], axis=1)
                ball_trajectory_df = ball_trajectory_df.drop(['Z'], axis=1)
                ball_trajectory_df = ball_trajectory_df.drop(['Shot'], axis=1)
                ball_trajectory_df = ball_trajectory_df.drop(['player_X'], axis=1)
                ball_trajectory_df = ball_trajectory_df.drop(['player_Y'], axis=1)
                ball_trajectory_df = ball_trajectory_df.drop(['prev_hit'], axis=1)
                ball_trajectory_df = ball_trajectory_df.drop(['next_hit'], axis=1)

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