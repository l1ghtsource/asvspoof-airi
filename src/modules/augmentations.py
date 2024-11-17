from scipy.signal import butter, lfilter
import numpy as np
import torch


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


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        '''
        Mixup coefficient generator.
        '''
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        '''
        Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        '''
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return torch.from_numpy(np.array(mixup_lambdas, dtype=np.float32))


def do_mixup(x: torch.Tensor, mixup_lambda: torch.Tensor):
    '''
    Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    '''
    out = (x[0::2].transpose(0, -1) * mixup_lambda[0::2] +
           x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0, -1)
    return out
