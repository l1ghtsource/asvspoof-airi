from sklearn import metrics
import numpy as np


class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        '''
        y_true: Ground truth labels (1D tensor or array)
        y_pred: Predicted probabilities or logits (1D tensor or array)
        '''
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())

    @property
    def avg(self):
        '''
        Calculate metrics: ROC AUC, F1 Score, and Accuracy.
        '''
        y_true_np = np.array(self.y_true)
        y_pred_np = np.array(self.y_pred)
        y_pred_labels = (y_pred_np >= 0.5).astype(int)  # convert probabilities to binary predictions

        # ROC AUC
        roc_auc = metrics.roc_auc_score(y_true_np, y_pred_np)
        # F1 Score
        f1 = metrics.f1_score(y_true_np, y_pred_labels)
        # Accuracy
        accuracy = metrics.accuracy_score(y_true_np, y_pred_labels)

        return {
            'roc_auc': roc_auc,
            'f1': f1,
            'accuracy': accuracy
        }
