import abc
from abc import abstractmethod
import numpy as np 

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

class SpecAugment(abc.ABC):
    """
    Base class for spectrogram augmentation

    We provide the following spectrogram augmentation methods
    1. Add (random) noise
    2. (Randomized) Frequency masking
    3. Time warp
    4. Multiply random noise with spectrogram
    """
    def __init__(self, frame_length, frame_step):
        # set up the basic parameters
        self._frame_length = frame_length
        self._frame_step = frame_step
        
        # create the computation graph
        self._input_tensor = input_tensor = tfv1.placeholder(tf.float32, shape = [None, None], name='spec_aug_input_tensor')
        input_spec = tf.signal.stft(input_tensor, frame_length=frame_length, frame_step=frame_step, pad_end=True)
        augment_spec = self.augment(input_spec)
        self._output_tensor = output_tensor = tf.signal.inverse_stft(augment_spec, frame_length=frame_length, frame_step=frame_step)
        (self._grad_out_input,) = tf.gradients(output_tensor, input_tensor)
        
        # backward propogation
        self._grad_loss_input_tensor = tfv1.placeholder(tf.float32)
        (self._backward_grad,) = tf.gradients(output_tensor, input_tensor, grad_ys = self._grad_loss_input_tensor)

    @abstractmethod 
    def augment(self, spec_input):
        """
        Augment the spectrograms

        Parameters
        ----------
        spec_input: tensorflow.tensor
            The input spectrogram

        Returns
        -------
        tensorflow.tensor
            Augmented spectrogram
        """
        raise NotImplementedError

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: numpy.array
            The original input audios

        Returns
        -------
        numpy.array
            The augmented audio through the spectrogram augmentation
        """
        with tfv1.Session() as sess:
            output = sess.run(self._output_tensor, feed_dict={self._input_tensor:inputs})
        return output

    def gradient(self, inputs):
        """
        Parameters
        ----------
        inputs: numpy.array
            The original input audios

        Returns
        -------
        numpy.array
            The gradient of output w.r.t. the input through the spec augmentation
        """
        with tfv1.Session() as sess:
            g = sess.run(self._grad_out_input, feed_dict={self._input_tensor:inputs})
        return g

    def forward_and_gradient(self, inputs):
        """
        Parameters
        ----------
        inputs: numpy.array
            The original input audios

        Returns
        -------
        numpy.array
            The augmented audio through the spectrogram augmentation
        numpy.array
            The gradient of output w.r.t. the input through the spec augmentation
        """
        with tfv1.Session() as sess:
            o, g = sess.run([self._output_tensor, self._grad_out_input], feed_dict={self._input_tensor:inputs})
        return o, g

    def backward(self, inputs, grad_loss_aug_input):
        """
        Parameters
        ----------
        inputs: numpy.ndarray
            The original input audios
        grad_loss_input: numpy.ndarray
            The gradient of loss w.r.t. the augmented input audios 

        Returns
        -------
        numpy.ndarray
            The gradient of the loss w.r.t. the original inputs 
        """
        with tfv1.Session() as sess:
            g = sess.run(self._backward_grad, feed_dict={self._input_tensor:inputs, self._grad_loss_input_tensor:grad_loss_aug_input})
        return g

    def forward_one(self, x):
        """
        Take a single input audio and return the augmented audio

        Parameters
        ----------
        x: `numpy.ndarray`
            Single input audio with shape (audio_length, )

        Returns
        -------
        `numpy.ndarray` with same shape as input x
        """
        return np.squeeze(self.forward(x[np.newaxis,:]), axis=0)

    def gradient_one(self, x):
        """
        Take a single input and return the gradient of
        the augmented audio w.r.t to the input audio x

        Parameters
        ----------
        x: `numpy.ndarray`
            Single input audio with shape (audio_length, )

        Returns
        -------
        `numpy.ndarray`
            The gradient of the augmented audio w.r.t. the input audio
        """
        return np.squeeze(self.gradient(x[np.newaxis,:]), axis=0)


    def forward_and_gradient_one(self, x):
        """
        Return the augmented audio, and the gradient of
        the augmented audio w.r.t to the input audio x

        Parameters
        ----------
        x: `numpy.ndarray`
            Single input audio with shape (audio_length, )

        Returns
        -------
        `numpy.ndarray`
            Augmented audio
        `numpy.ndarray`
            The gradient of the augmented audio w.r.t. the input audio
        """
        o, g = self.forward_and_gradient(x[np.newaxis, :])
        return np.squeeze(o), np.squeeze(g) 

    def backward_one(self, x, grad_loss_input):
        """
        Backpropogates the gradient of some loss w.r.t. the augmented input audio, 
        through the augmentation layer

        Parameters
        ----------
        x: `numpy.ndarray`
            Single input audio with shape (audio_length, )

        Returns
        -------
        `numpy.ndarray`
            Augmented audio
        `numpy.ndarray`
            The gradient of the respective loss w.r.t. the input audio
        """
        g = self.backward(x[np.newaxis,:], grad_loss_input[np.newaxis,:])
        return np.squeeze(g, axis = 0) 