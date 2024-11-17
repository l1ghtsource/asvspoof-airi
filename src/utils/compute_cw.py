import numpy as np
import torch


def compute_class_weights(dataset):
    all_labels = []

    for sample in dataset.audio_data:
        labels = sample['labels']
        all_labels.append(labels)

    all_labels = np.array(all_labels)
    class_counts = np.bincount(all_labels)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float)
