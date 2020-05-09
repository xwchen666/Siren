import abc
from abc import abstractmethod
import numpy as np 

import tensorflow as tf

from .base import SpecAugment

class SpecAddRandomNoise(SpecAugment):
    """
    Add random noise to the magnitude of the spectrogram of audios

    Parameters
    ----------
    frame_length: float
        Frame length in microseconds
    frame_step: float
        Frame step in microseconds
    distribution: str
        One of the following strings, selecting the types of noise to add:
        'normal'    Normal-distributed noise
        'uniform'   Uniform-distributed noise
        'gamma'     Gamma-distributed noise

    Remarks
    -------
    A reivew on noise types in images: https://arxiv.org/pdf/1505.03489.pdf
    """
    def __init__(self, frame_length, frame_step, distribution = 'normal', noise_type='additive', **kwargs):
        self._distribution = distribution
        self._kwargs = kwargs
        self._noise_type = noise_type
        
        # convert the microseconds to integer length of STFT
        frame_length = int(frame_length * 16000 / 1000)
        frame_step = int(frame_step * 16000 / 1000)

        super().__init__(frame_length, frame_step)

    def augment(self, spec_input):
        """
        Parameters
        ----------
        spec_input: tf.tensor
            Spectrogram of the audios
        multiplicative: boolean
            If true, the noise is multiplicative. Otherwise, the noise is additive
        """
        spec_input_magnitude = tf.abs(spec_input)
        spec_input_angle = tf.math.angle(spec_input)

        augmented_spec_magnitude = self._augment_spec_magnitude(spec_input_magnitude)

        augmented_spec_magnitude = tf.clip_by_value(augmented_spec_magnitude, clip_value_min=0, clip_value_max=np.inf)
        output = tf.math.exp(1j * tf.cast(spec_input_angle, dtype=tf.complex64)) *\
                        tf.cast(augmented_spec_magnitude, dtype=tf.complex64)
 
        return output

    def _augment_spec_magnitude(self, spec_input_magnitude):
        snr = self._kwargs.get('snr', 40)
        # calculate the power of the noise
        if self._noise_type == 'additive':
            p_input = tf.reduce_mean(spec_input_magnitude, axis=None)
        else:
            p_input = tf.cast(1.0, dtype=tf.float32)
        p_noise = p_input / np.power(10, (snr / 10))

        if self._distribution == 'normal':
            random_generator = tf.random.normal(shape=tf.shape(spec_input_magnitude), 
                                                mean=tf.zeros_like(p_noise),
                                                stddev=tf.sqrt(p_noise))
        elif self._distribution == 'uniform':
            random_generator = tf.random.uniform(shape=tf.shape(spec_input_magnitude),
                                                 minval=-tf.sqrt(3 * p_noise),
                                                 maxval=tf.sqrt(3 * p_noise))
        elif self._distribution == 'gamma':
            k = 0.1
            random_generator = tf.random.gamma(shape=tf.shape(spec_input_magnitude),
                                               alpha=k, 
                                               beta=tf.sqrt(k * (k + 1) / p_noise))
        else:
            raise ValueError

        if self._noise_type == 'additive':
            output = spec_input_magnitude + random_generator
        else: 
            output = (1 + random_generator) * spec_input_magnitude 

        return output