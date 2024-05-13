import torch
from pathlib import Path
from ultralytics.tracknet.dataset import TrackNetDataset
from ultralytics.tracknet.utils.transform import target_grid
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.metrics import DetMetrics
import numpy as np

class TrackNetValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.seen = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.acc = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
    
    def get_dataloader(self, dataset_path, batch_size):
        """For TrackNet, we can use the provided TrackNetDataset to get the dataloader."""
        dataset = TrackNetDataset(root_dir=dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
    
    def preprocess(self, batch):
        """In this case, the preprocessing step is mainly handled by the dataloader."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['target']:
            batch[k] = batch[k].to(self.device)

        return batch
    
    def postprocess(self, preds):
        """Postprocess the model predictions if needed."""
        if isinstance(preds, list):
            preds = preds[0]

        res = []
        for i in range(preds.shape[0]):
            res.append(self.teardown(preds[i]))
        return res

    def teardown(self, preds):
        #preds size: b*60*20*20

        pred_distri, pred_scores, pred_hits = torch.split(preds, [40, 10, 10], dim=0)
        pred_distri = pred_distri.reshape(4, 10, 20, 20)
        pred_pos, pred_mov = torch.split(pred_distri, [2, 2], dim=0)

        # pred_pos shape: [2,10,20,20]
        # permute() makes shape to [10,2,20,20]
        pred_pos = pred_pos.permute(1, 0, 2, 3).contiguous()
        pred_mov = pred_mov.permute(1, 0, 2, 3).contiguous()

        pred_pos = torch.sigmoid(pred_pos)
        pred_mov = torch.tanh(pred_mov)

        # [10, 20, 20]
        pred_scores

        # [2, 10, 20, 20]
        pred_pos

        # [2, 10, 20, 20]
        pred_mov

        # [10, 20, 20]
        pred_hits

        pred_conf_all = torch.sigmoid(pred_scores.detach()).cpu()
        pred_mov_all = pred_mov.detach().clone()
        pred_pos_all = pred_pos.detach().clone()

        res = [] # shape: (frame, labels)

        for i in range(10):
            pred_conf = pred_conf_all[i]

            pred_conf_np = pred_conf.numpy()
            y_positions, x_positions = np.where(pred_conf_np >= 0.5)

            detects = []

            for x, y in zip(x_positions.tolist(), y_positions.tolist()):
                detects.append({
                    "cell_x": x,
                    "cell_y": y,
                    "x": x*32+pred_pos_all[i][0][y][x].item()*32,
                    "y": y*32+pred_pos_all[i][1][y][x].item()*32,
                    "confidence": pred_conf[y][x].item()
                })
            res.append(detects)

        return res

    def init_metrics(self, model):
        """Initialize some metrics."""
        # Placeholder for any metrics you might want to use.
        self.seen = 0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.acc = 0
        self.precision = 0
        self.recall = 0
        self.f1 = 0
    
    def update_metrics(self, preds, batch):
        """Calculate and update metrics based on predictions and batch."""

        #nl, npr = batch['target'].shape[0], pred.shape[0]  # number of labels, predictions
        # iterate batch item
        for batch_idx, pred in enumerate(preds):
            self.update_metrics_once(batch_idx, pred, batch['target'][batch_idx])

            # Save
            if self.args.save_json:
                self.pred_to_json(pred, batch['match_name'][batch_idx], batch['video_name'][batch_idx], batch['frame_idx_begin'][batch_idx], batch['frame_idx_end'][batch_idx])


    def pred_to_json(self, predn, match_name, video_name, fid_min, fid_max):
        """Serialize YOLO predictions to COCO json format."""
        self.jdict.append({
            'match_name': match_name,
            'video_name': video_name,
            'frame_id_min': int(fid_min),
            'frame_id_max': int(fid_max),
            'pred': predn
            })

    def update_metrics_once(self, batch_idx, pred, batch_target):

        # 16 pixel
        threshold_distance = 4

        # iterate each frame
        for target, pr in zip(batch_target, pred):
            self.seen += 1
            _, visibility, x, y, _, _, _ = target

            if len(pr) > 0 and visibility == 0:
                self.FP += len(pr)
            elif len(pr) > 0 and visibility == 1:
                found = 0 
                for p in pr:
                    dist = np.linalg.norm(torch.tensor([x - p['x'], y - p['y']]).numpy())

                    if dist < threshold_distance:
                        found += 1
                    else:
                        self.FP += 1                   
                if found > 0:
                    self.TP += 1
                    self.FN += (found - 1)
                else:
                    self.FN += 1
            elif len(pr) == 0 and visibility == 0:
                self.TN += 1
            elif len(pr) == 0 and visibility == 1:
                self.FN += 1

        return
        # pred = [50 * 20 * 20]
        # batch_target = [10*6]
        pred_distri, pred_scores, pred_hits = torch.split(pred, [40, 10, 10], dim=0)
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
        if len(batch_target.shape) == 3:
            batch_target = batch_target[0]
        for idx, target in enumerate(batch_target):
            if target[1] == 1:
                # xy
                grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
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
        pass

    def get_stats(self):
        """Return the stats."""
        #return {'FN': self.FN, 'FP': self.FP, 'TN': self.TN, 'TP': self.TP, 'acc': self.acc, 'max_conf>0.5': self.hasMax, 'correct_cell>0.5':self.hasBall}
        self.acc = (self.TN + self.TP) / (self.FN + self.FP + self.TN + self.TP)
        if (self.TP + self.FP) > 0:
            self.precision = self.TP / (self.TP + self.FP)
        if (self.TP + self.FN) > 0:
            self.recall = self.TP / (self.TP + self.FN)
        if self.precision > 0 and self.recall > 0:
            self.f1 = 2 / ((1 / self.precision) + (1 / self.recall))

        return {
            'metrics/FN': self.FN,
            'metrics/FP': self.FP,
            'metrics/TN': self.TN,
            'metrics/TP': self.TP,
            'metrics/accuracy': self.acc,
            'metrics/precision': self.precision,
            'metrics/recall': self.recall,
            'metrics/f1': self.f1}
    
    def print_results(self):
        """Print the results."""
        pf = '%22s' + '%11.3g' * 4  # print format
        LOGGER.info(pf % (self.seen, self.acc, self.precision, self.recall, self.f1))
        pass

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ('%22s' + '%11s' * 4) % ('Frames', 'Accuracy', 'Precision', 'Recall', 'F1-score')

    def plot_val_samples(self, batch, ni):
        #TODO:
        pass

    def plot_predictions(self, batch, preds, ni):
        #TODO:
        pass
