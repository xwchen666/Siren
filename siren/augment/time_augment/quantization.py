import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import math

from .base import DataAugment

class Quantization(DataAugment):
    """
    Rounding the amplitude of audio sampled data into the nearest 
    integer multiple of q
    The adversarial perturbation could be disrupted since its amplitude 
    is usually small in the input space. 
    Parameters
    ----------
    q: int
        Quantization parameter

    Remarks
    -------
    In the paper 
        "Characterizing Audio Adversarial Examples Using Temporal Dependency. 
        Zhuolin Yang et al. ICLR'19",
    Parameter q is set to 128, 256, 512, 1024
    """
    def __init__(self, q = 128):
        self._q = q

    def _quantize(self, inputs):
        # first step: rescale
        input_max = np.max(np.abs(inputs), axis = 1)[:, np.newaxis]
        inputs = inputs / input_max * (2**15 - 1)
        # step 2: quantize
        inputs = np.trunc(inputs / self._q) * self._q / (2**15 - 1) * input_max
        return inputs

    def forward(self, inputs):
        return self._quantize(inputs)

    def gradient(self, inputs):
        return np.ones(inputs.shape)

    def forward_and_gradient(self, inputs):
        return self._quantize(inputs), np.ones(inputs.shape)

    def backward(self, inputs, grad_loss_input):
        return grad_loss_input