from scipy.signal import butter, lfilter
import numpy as np


class ReduceHighFrequencies:
    def __init__(self, cutoff_freq=10000, reduction_db=3, sample_rate=22050):
        self.cutoff_freq = cutoff_freq
        self.reduction_db = reduction_db
        self.sample_rate = sample_rate

    def __call__(self, samples: np.ndarray):
        b, a = butter(6, self.cutoff_freq / (self.sample_rate / 2), btype='low')
        low_passed = lfilter(b, a, samples)
        factor = 10 ** (-self.reduction_db / 20)
        high_frequencies = samples - low_passed
        samples = low_passed + high_frequencies * factor
        return samples
