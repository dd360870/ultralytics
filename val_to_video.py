import cv2
import argparse
import pandas as pd
import glob
import numpy as np
import random
from pathlib import Path

def transform_coordinates(coord, w, h, target_size=640):
    
    # Determine padding
    max_dim = max(w, h)
    pad = (max_dim - min(w, h)) // 2
    
    # Adjust for scaling
    scale_factor = max_dim / target_size
    coord[0] *= scale_factor  # scale X
    coord[1] *= scale_factor  # scale Y

    # Adjust for padding
    if h < w:
        coord[1] -= pad
    else:
        coord[0] -= pad  # if height is greater, adjust X
    
    return coord

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="./datasets/tracknet-2/val_data")
parser.add_argument("--val_dir", type=str, default="./runs/detect/val")
args = parser.parse_args()

source_dir = Path(args.source)
val_dir = Path(args.val_dir)

print(f"val_dir: {val_dir}")
    
pred_json = pd.read_json(val_dir.joinpath("predictions.json"), dtype={"match_name":str, "video_name":str})
#print(j.iloc[0])

videos = glob.glob("*/video/*.mp4", root_dir=source_dir)

metrics_global = {
    "FN": 0,
    "FP": 0,
    "TN": 0,
    "TP": 0,
}

for x in videos:
    metrics_local = {
        "FN": 0,
        "FP": 0,
        "TN": 0,
        "TP": 0,
    }

    video_path = Path(x)
    match_name = video_path.parts[0]
    video_name = video_path.stem

    gt = pd.read_csv(source_dir.joinpath(Path(f"{match_name}/csv/{video_name}_ball.csv")))

    # filtering by match_name, video_name
    preds = pred_json.loc[(pred_json['match_name'] == match_name) & (pred_json['video_name'] == video_name)]

    data = []

    # 
    #for i in range(len(preds)):
    #    config_frame = preds.iloc[i].frame_id_max - preds.iloc[i].frame_id_min + 1
    #    for idx, frame_id in enumerate(range(preds.iloc[i].frame_id_min, preds.iloc[i].frame_id_max + 1)):

    #        #TODO: only process frame 4
    #        if config_frame == 10 and idx not in [4, 5]:
    #            continue
    #        elif config_frame == 3 and idx not in [1]:
    #            continue

    #        if len(preds.iloc[i].pred[idx]) > 0:
    #            df = pd.DataFrame(preds.iloc[i].pred[idx])
    #            df = df.loc[df['confidence'] >= 0.8]
    #            df.insert(0, "frame_id", frame_id, False)
    #            data.append(df)
    current_begin = 0
    for i in range(len(preds)):
        config_frame = preds.iloc[i].frame_id_max - preds.iloc[i].frame_id_min + 1

        if current_begin + config_frame > preds.iloc[i].frame_id_min:
            continue

        current_begin = preds.iloc[i].frame_id_min

        for idx, frame_id in enumerate(range(preds.iloc[i].frame_id_min, preds.iloc[i].frame_id_max + 1)):

            if len(preds.iloc[i].pred[idx]) > 0:
                df = pd.DataFrame(preds.iloc[i].pred[idx])
                df = df.loc[df['confidence'] >= 0.8]
                df.insert(0, "frame_id", frame_id, False)
                data.append(df)

    df = pd.concat(data)
    #df = df.groupby(['frame_id']).mean()

    # TODO: process video
    cap = cv2.VideoCapture(str(source_dir.joinpath(video_path)))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # draw ground truth
        print("frame_id: ", frame_id)
        if gt.iloc[frame_id].Visibility == 1:
            frame = cv2.circle(frame, (gt.iloc[frame_id].X, gt.iloc[frame_id].Y), 3, color=(0, 255, 0), thickness=2)
        data = df.loc[df['frame_id'] == frame_id]
        data = data.sort_values("confidence", ascending=False)
        data = data.reset_index()
        print("sorted: \n", data)

        # TODO: preprocess
        #if len(data) > 0:
        #    filtered_data = []
        #    while len(data) > 0:
        #        if 'dist' in data.columns:
        #            data = data.drop(columns=['dist'])
        #        data['dist'] = np.linalg.norm(np.array(data.loc[:, ['x', 'y']].values) - np.repeat([data.loc[0, ['x', 'y']].values], len(data), axis=0), axis=1)
        #        #data.insert(0, "dist", np.linalg.norm(np.array(data.loc[:, ['x', 'y']].values) - np.repeat([data.loc[0, ['x', 'y']].values], len(data), axis=0), axis=1), allow_duplicates=True)
        #        filtered_data.append(data.loc[data['dist'] <= 10].groupby('frame_id').mean())
        #        data = data.loc[data['dist'] > 10].reset_index(drop=True)
        #        print('data: \n', data)
        #    filtered_data = pd.concat(filtered_data)
        #    data = filtered_data

        print(data)

        found = False

        for i in range(len(data)):
            coord = [data.iloc[i].x, data.iloc[i].y]
            coord = transform_coordinates(coord, 1280, 720)
            #print(coord)
            color=(0, 0, 255)

            if gt.iloc[frame_id].Visibility == 1 and np.linalg.norm(np.array(coord) - np.array([gt.iloc[frame_id].X, gt.iloc[frame_id].Y])) < 5:
                # TP
                color = (255, 0, 0)
                found = True
            else:
                metrics_local['FP'] += 1

            frame = cv2.circle(frame, [round(coord[0]), round(coord[1])], 3, color=color, thickness=2)
            frame = cv2.putText(frame, str(round(data.iloc[i].confidence, 3)), [round(coord[0])+10, round(coord[1])+10], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if found and (gt.iloc[frame_id].Visibility == 1):
            metrics_local['TP'] += 1
        if (not found) and (gt.iloc[frame_id].Visibility == 1):
            metrics_local['FN'] += 1
        elif len(data) == 0 and gt.iloc[frame_id].Visibility == 0:
            metrics_local['TN'] += 1

        print(metrics_local)

        cv2.imshow('frame', frame)
        if cv2.waitKey(0) == ord('q'):
            break
        frame_id += 1
    cap.release()
    cv2.destroyAllWindows()
