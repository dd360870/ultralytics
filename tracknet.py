#!/usr/bin/env python3

import argparse
import time
from tqdm import tqdm
import numpy as np
import torch
from ultralytics.tracknet.dataset import TrackNetDataset
from ultralytics.tracknet.predict import TrackNetPredictor
from ultralytics.tracknet.train import TrackNetTrainer
from ultralytics.tracknet.val import TrackNetValidator
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.utils import DEFAULT_CFG, TQDM_BAR_FORMAT, callbacks
from ultralytics.yolo.cfg import get_cfg
import torch
import numpy as np
from pathlib import Path
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_pip_update_available, check_yaml

def main(arg):
    overrides = {}
    overrides['model'] = arg.model_path
    overrides['mode'] = arg.mode
    overrides['data'] = arg.data
    overrides['epochs'] = arg.epochs
    overrides['plots'] = True
    overrides['save'] = True
    overrides['batch'] = arg.batch
    overrides['patience'] = 300
    overrides['val'] = arg.val
    overrides['use_dxdy_loss'] = arg.use_dxdy_loss

    if arg.mode == 'train':
        trainer = TrackNetTrainer(overrides=overrides)
        trainer.train()
    elif arg.mode == 'val':
        model, _ = attempt_load_one_weight(arg.model_path)
        if torch.cuda.is_available():
            model.cuda()
        
        overrides = overrides.copy()
        overrides['save'] = True
        overrides['rect'] = True  # rect batches as default
        overrides['save_json'] = True
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(model, (str, Path)):
            args.imgsz = model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = TrackNetValidator(args=args, _callbacks=callbacks.get_default_callbacks())
        validator(model=model)

        print(validator.metrics)
    elif arg.mode == 'predict':
        model, _ = attempt_load_one_weight(arg.model_path)
        worker = 0
        if torch.cuda.is_available():
            model.cuda()
            worker = 1
        dataset = TrackNetDataset(root_dir=arg.source)
        dataloader = build_dataloader(dataset, overrides['batch'], worker, shuffle=False, rank=-1)
        overrides = overrides.copy()
        overrides['save'] = False
        predictor = TrackNetPredictor(overrides=overrides)
        predictor.setup_model(model=model, verbose=False)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), bar_format=TQDM_BAR_FORMAT)
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
            start_time = time.time()

            # [1*50*20*20]
            p = predictor.inference(input_data)
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            elapsed_times+=elapsed_time
            pbar.set_description(f'{elapsed_times / (i+1):.2f}  {i+1}/{len(pbar)}')

            # [5*20*20]
            # p_check = p[0, 5*idx:5*(idx+1), :]
            # p_conf = torch.sigmoid(p_check[4, :, :])
            # p_cell_x = torch.sigmoid(p_check[0, :, :])
            # p_cell_y = torch.sigmoid(p_check[1, :, :])

            # max_position = torch.argmax(p_conf)
            # max_y, max_x = np.unravel_index(max_position, p_conf.shape)
            # p_x = p_cell_x[max_x, max_y]*32
            # p_y = p_cell_y[max_x, max_y]*32
            # p_gridx = max_x*32 + p_cell_x[max_x, max_y]*32
            # p_gridy = max_y*32 + p_cell_y[max_x, max_y]*32
            # x, y, gx, gy = targetGrid(t_x, t_y, 32)

            # if hasBall and i%10 == 0:
            #     pos_weight = torch.tensor([400])
            #     l = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
            #     cls_targets = torch.zeros(p_conf.shape[0], p_conf.shape[1])
            #     cls_targets[x][y] = 1
            #     cls_pred = p_check[4, :, :]
            #     count_ge_05 = np.count_nonzero(p_conf >= 0.5)
            #     count_lt_05 = np.count_nonzero(p_conf < 0.5)
            #     correct = True if p_conf[x][y]>=0.5 else False
            #     loss = l(cls_pred, cls_targets).sum()
            #     loss_list = [loss.item()]
            #     loss_list.append(count_ge_05)
            #     loss_list.append(count_lt_05)
            #     loss_list.append(correct)
            #     loss_list.append(p_conf[x][y])
            #     display_image_with_coordinates(input_data[0][idx], [(x*32, y*32)], [(max_x*32, max_y*32)], str(i), loss_list)
            #end_time = time.time()
            #elapsed_time = (end_time - start_time) * 1000
            #print(f'{elapsed_time:.2f}ms')
            #elapsed_times+=elapsed_time

        print(f"avg predict time: { elapsed_times / len(dataloader):.2f} 毫秒")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a custom model with overrides.')
    
    parser.add_argument('--model_path', type=str, default=r'./ultralytics/models/v8/tracknetv4.yaml', help='Path to the model')
    parser.add_argument('--mode', type=str, default='train', help='Mode for the training (e.g., train, test)')
    parser.add_argument('--data', type=str, default='tracknet-2.yaml', help='Data configuration (e.g., tracknet.yaml)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    #parser.add_argument('--source', type=str, default=r'./datasets/tracknet/train_data', help='source')
    parser.add_argument('--val', type=bool, default=True, help='run val')
    parser.add_argument('--use_dxdy_loss', type=bool, default=True, help='use dxdy loss or not')
    parser.add_argument('--weights', type=str, default=None, help='transfer weights')
    
    args = parser.parse_args()

    # DEBUG
    #python tracknet.py --mode=val --model_path=./runs/detect/train59/weights/last.pt 
    #args.mode = 'val'
    #args.model_path = './runs/detect/train59/weights/last.pt'

    main(args)