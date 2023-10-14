import torch
from dataset import TrackNetDataset
from utils.transform import target_grid
from ultralytics.yolo.data.build import build_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils.metrics import DetMetrics


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
        """Initialize some metrics."""
        # Placeholder for any metrics you might want to use.
        self.total_loss = 0.0
        self.num_samples = 0
    
    def update_metrics(self, preds, batch):
        """Calculate and update metrics based on predictions and batch."""
        # Placeholder for loss calculation, etc.
        # preds = [[10*50*40*40]]
        preds = preds[0] # only pick first (stride = 16)
        batch_target = batch['target']
        batch_size = preds.shape[0]
        if preds.shape == (50, 40, 40):
            self.update_metrics_once(0, preds, batch_target)
        else:
            # for each batch
            for idx, pred in enumerate(preds):
                self.update_metrics_once(idx, pred, batch_target)
        #print((self.TP, self.FP, self.FN))
    def update_metrics_once(self, batch_idx, pred, batch_target):
        # pred = [50 * 40 * 40]
        pred_distri, pred_scores = torch.split(pred, [40, 10], dim=0)
        pred_probs = torch.sigmoid(pred_scores)

        max_values_dim1, max_indices_dim1 = pred_probs.max(dim=2)
        final_max_values, max_indices_dim2 = max_values_dim1.max(dim=1)
        max_positions = [(index.item(), max_indices_dim1[i, index].item()) for i, index in enumerate(max_indices_dim2)]

        targets = pred_distri.clone().detach()
        cls_targets = torch.zeros(10, pred_scores.shape[1], pred_scores.shape[2])
        stride = 16
        for idx, target in enumerate(batch_target[batch_idx]):
            if target[1] == 1:
                # xy
                grid_x, grid_y, offset_x, offset_y = target_grid(target[2], target[3], stride)
                # print(pred_probs[idx][max_positions[idx]])
                # print(max_positions[idx])
                if pred_probs[idx][max_positions[idx]] > 0.5:
                    x, y = max_positions[idx]
                    real_x = x*16 + pred_distri[idx][x][y]
                    real_y = y*16 + pred_distri[idx][x][y]
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
        return {'FN': self.FN, 'FP': self.FP, 'TN': self.TN, 'TP': self.TP}
    
    def print_results(self):
        """Print the results."""
        precision = 0
        recall = 0
        f1 = 0
        if self.TP > 0:
            precision = self.TP/(self.TP+self.FP)
            recall = self.TP/(self.TP+self.FN)
            f1 = (2*precision*recall)/(precision+recall)
        print(f"Validation Loss: {self.loss}Validation Accuracy: {self.acc:.4f}, Validation Precision: {precision:.4f}, Validation Recall: {recall:.4f}, , Validation F1-Score: {f1:.4f}")

    def get_desc(self):
        """Return a description for tqdm progress bar."""
        return "Validating TrackNet"
