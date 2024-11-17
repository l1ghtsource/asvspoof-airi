import torch
import time
from transformers import Trainer, TrainerCallback
from losses import FocalLoss


class FocalTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        '''
        Args:
            class_weights (list or torch.Tensor): Весы для классов.
        '''
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            if isinstance(class_weights, list):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.class_weights = class_weights.cuda() if torch.cuda.is_available() else class_weights
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        loss_foc = FocalLoss(alpha=self.class_weights, gamma=2)
        loss = loss_foc(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss


class TimeLimitCallback(TrainerCallback):
    def __init__(self, time_limit_hours):
        self.time_limit_seconds = time_limit_hours * 3600  # Convert hours to seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.time_limit_seconds:
            control.should_training_stop = True
