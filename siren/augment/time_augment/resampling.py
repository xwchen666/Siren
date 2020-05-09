import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import math

from .base import DataAugment
import scipy.signal

class Resample(DataAugment):
    def __init__(self, rescale = 0.5):
        """
        Parameters
        ----------
        rescale: float
            The ratio of lengths between the target audio and the input audio, 
            i.e., len(target_audio) / len(input_audio)

        DONE: validate the correctness through tensorflow graph
        TODO: prove the correctness of the gradient & backward function
        """

        self._rescale = rescale
    def _update_rescale(self, rescale):
        if rescale is None:
            rescale = self._rescale # use old rescale
        else:
            self._rescale = rescale # update
        return rescale

    def forward(self, inputs, rescale = None):
        rescale = self._update_rescale(rescale)
        l = int(inputs.shape[1] * rescale)
        o = scipy.signal.resample(inputs, l, axis = 1)
        return np.array(o) 

    def gradient(self, inputs):
        return rescale * np.ones(inputs.shape)

    def forward_and_gradient(self, inputs, rescale = None):
        rescale = self._update_rescale(rescale)
        l = int(inputs.shape[1] * rescale)
        o = scipy.signal.resample(inputs, l, axis = 1)
        g = rescale * np.ones(inputs.shape)
        return np.array(o), np.array(g)

    def backward(self, inputs, grad_loss_input):
        g = scipy.signal.resample(grad_loss_input, inputs.shape[1], axis = 1) * self._rescale
        return np.array(g) 
    def compute_output_lens(self, input_lens):
        return np.array(input_lens * self._rescale, dtype=int)

TimeStretch = Resample

class DownSampling(Resample):
    def __init__(self, old_sr = 16000, new_sr = 8000):
        rescale = new_sr / old_sr
        self._rescale = rescale
