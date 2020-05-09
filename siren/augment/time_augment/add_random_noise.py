import numpy as np
import tensorflow as tf
import math

from .base import DataAugment

class AddRandomNoise(DataAugment):
    """
    Parameters
    ----------
    distribution: str
        The distribution of the added noise
    snr: float
        The signal-to-noise ration in dB

    References
    ----------
    Computation of SNR: https://dsp.stackexchange.com/questions/42543/snr-calculation
    """
    def __init__(self, distribution = 'normal', snr=40):
        self._distribution = distribution
        self._snr = snr

        # parameters for gamma distribution
        self._k = 0.1

    def update_snr(self, new_snr):
        self._snr = new_snr

    def _generate_noise(self, inputs):
        p_input = np.mean(np.square(inputs), axis = 0 if inputs.ndim == 1 else 1) 
        p_noise = p_input / np.power(10, (self._snr / 10))
        if inputs.ndim == 2:
            p_noise = p_noise[:, np.newaxis]
        switcher = {
            'normal'  : np.random.normal(loc = np.zeros(p_noise.shape), scale=np.sqrt(p_noise), size=inputs.shape),
            'uniform' : np.random.uniform(low=-np.sqrt(3 * p_noise), high=np.sqrt(3 * p_noise), size=inputs.shape),
            'gamma'   : np.random.gamma(shape = self._k, scale=np.sqrt(p_noise / self._k / (self._k + 1)), size=inputs.shape),
        }
        return switcher.get(self._distribution, np.zeros(inputs.shape))
    
    def forward(self, inputs):
        return inputs + self._generate_noise(inputs)
    
    def gradient(self, inputs):
        return np.ones(inputs.shape)

    def forward_and_gradient(self, inputs):
        augmented_audio = inputs + self._generate_noise(inputs)
        gradient = np.ones(inputs.shape)
        return augmented_audio, gradient       

    def backward(self, inputs, grad_loss_input):
        return grad_loss_input