import os
import random
from glob import glob
import shutil

dataset_name = 'tracknet-2'
include_matches = ['alex_tracknet/']

#dataset_name = 'tracknet-1'
#include_matches = ['profession_match*/', 'alex_tracknet/']

DATASET_DIR = "/hdd/marvin/dataset"

TRAIN, VAL = 8, 2 # 80% / 20%


match_dirs = []
for g in include_matches:
    match_dirs.extend(glob(g, root_dir=DATASET_DIR))

video_list = []

for match_dir in match_dirs:
    csv_path = os.path.join(DATASET_DIR, match_dir, 'csv')
    for video_name in glob('*.csv', root_dir=csv_path):
        video_list.append({
            'match_name': match_dir.removesuffix("/"),
            'video_name': video_name.removesuffix('_ball.csv'),
        })

def create_dataset(video_list, save_path):
    shutil.rmtree(save_path)
    for v in video_list:
        for d, suffix in zip(['video', 'frame', 'csv'], ['.mp4', '', '_ball.csv']):
            os.makedirs(os.path.join(save_path, v['match_name'], d), exist_ok=True)

            src = os.path.join(DATASET_DIR, v['match_name'], d, v['video_name']+suffix)
            dst = os.path.join(save_path, v['match_name'], d, v['video_name']+suffix)
            os.symlink(src, dst)

random.shuffle(video_list)

mid = round(len(video_list)*TRAIN/(TRAIN+VAL))

create_dataset(video_list[:mid], f"./datasets/{dataset_name}/train_data")
create_dataset(video_list[mid:], f"./datasets/{dataset_name}/val_data")
