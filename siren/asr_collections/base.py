import numpy as np 
import abc
from abc import abstractmethod

class Model(abc.ABC):
    @abstractmethod
    def set_loss(self, target_phrases):
        """
        Set the loss tensor for the model

        Parameters
        ----------
        target_phrases: List[str] or np.ndarray[str]
            Target phrases such as ['ok google', 'hello alexa']
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, inputs, input_lens=None):
        """
        Take batch of inputs and returns the transcripts predicted by the underlying model
        If the loss is already set, then the loss will also be returned 
        
        Parameters
        ----------
        inputs: :class: `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model. 
        input_lens: :class: `numpy.ndarray`
            Lengths of the input audios. If not provided, the input lens will be inferred directly from inputs 

        Returns
        -------
        List[str]
            Transcriptions
        np.ndarray[float] (if loss is set)
            Losses
        """
        raise NotImplementedError

    def gradient(self, inputs, input_lens=None):
        """
        Take batch of inputs and return the gradients of ctc loss w.r.t. the inputs

        Parameters
        ----------
        inputs: :class: `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model. 
        input_lens: :class: `numpy.ndarray`
            Lengths of the input audios. If not provided, the input lens will be inferred directly from inputs 

        Returns
        -------
        np.ndarray 
            Gradient of loss w.r.t. the inputs
        """
        raise NotImplementedError

    def forward_and_gradient(self, inputs, input_lens=None):
        """
        Take batch of inputs, and return the logits and gradients

        Parameters
        ----------
        inputs: `numpy.ndarray`
            Batch of inputs with shape as expected by the underlying model. 
        input_lens: :class: `numpy.ndarray`
            Lengths of the input audios. If not provided, the input lens will be inferred directly from inputs 
 
        Returns
        -------
        List[str]
            Transcriptions
        numpy.ndarray 
            Loss
        numpy.ndarray 
            Gradient of loss w.r.t. the inputs
        """
        raise NotImplementedError
