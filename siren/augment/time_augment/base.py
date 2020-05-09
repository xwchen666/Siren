import abc
from abc import abstractmethod
import numpy as np

class DataAugment(abc.ABC):
    """
    Base class for data augmentation

    We provide the following data augmentation methods
    1. Add Random Noise 
    2. Filter
        (randomized) moving average
        (randomized) recursive filter
        room reverberation
    3. Quantization
    4. Resample
        TimeStretch
        DownSampling 
    TODO:
    1. MP3 Compression
    2. AutoEncoder
    """
    @abstractmethod
    def forward(self, inputs):
        """
        Take a batch of input audios and return the augmented audios

        Parameters
        ----------
        inputs: `numpy.ndarray`
            Batch of input audio which needs augmentation

        Returns
        -------
        numpy.ndarray with same shape of inputs
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, inputs):
        """
        Return the gradient of the augmented audio w.r.t. the input audio

        Parameters
        ----------
        inputs: `numpy.ndarray` with shape (batch_size, max_audio_length)
            Batch of input audios which need augmentation

        Returns
        -------
        `numpy.ndarray`
            The gradient of the augmented audios w.r.t. the input audios
        """
        raise NotImplementedError

    @abstractmethod
    def forward_and_gradient(self, inputs):
        """
        Return the augmented audio, and the gradient of
        the augmented audio w.r.t the input audio

        Parameters
        ----------
        inputs: `numpy.ndarray` with shape (batch_size, max_audio_length)
            Batch of input audios which need augmentation

        Returns
        -------
        `numpy.ndarray`
            Augmented audios
        `numpy.ndarray`
            The gradient of the augmented audios w.r.t. the input audios
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, inputs, grad_loss_input):
        """
        Backpropogates the gradient of some loss w.r.t. the augmented input audio, 
        through the augmentation layer

        Parameters
        ----------
        inputs: `numpy.ndarray` with shape (batch_size, max_audio_length)
            Batch of input audios which need augmentation
        grad_loss_input: `numpy.ndarray` with shape (batch_size, max_audio_length)
            The gradient of the respective loss w.r.t. the augmented input

        Returns
        -------
        `numpy.ndarray`
            Augmented audios
        `numpy.ndarray`
            The gradient of the respective loss w.r.t. the inputs
        """
        raise NotImplementedError

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

    def name(self):
        return self.__class__.__name__

    def compute_output_lens(self, input_lens):
        """
        Compute the lengths of the output audios

        input_lens: 1D numpy.ndarray
            The lengths of the input audios
        
        Returns
        -------
        1D numpy.ndarray
        """
        return input_lens