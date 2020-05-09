"""
Provides classes that define what is adversarial

Criteria
--------
TargetClass

TODO
----
1. Provide more criteria
"""

import abc
from abc import abstractmethod
import numpy as np 
from functools import wraps
from util.similarity import WordSimilarity

def input_decorator(call_fn):
    @wraps(call_fn)
    def wrapper(self, preds, refs):
        is_single_input = isinstance(preds, str) or (isinstance(preds, np.ndarray) and preds.ndim == 0)
        is_single_ref   = isinstance(refs, str) or (isinstance(refs, np.ndarray) and refs.ndim == 0)
        if is_single_input:
            preds = np.array([preds])
        else:
            preds = np.array(preds)

        if is_single_ref:
            refs = np.array([refs])
        else:
            refs = np.array(refs)

        try:
            res = call_fn(self, preds, refs)
        except:
            raise RuntimeError('Fail to call is_adversarial function')
        if is_single_input:
            return res[0]
        return res
    return wrapper

class Criterion(abc.ABC) :
    """
    Base class for criteria that define what is adversarial 

    The :class: `Criterion` class represents a criterion used to
    determine if predictions for the speech are adversarial given a 
    reference label. It should be subclassed when implementing 
    new criteria. Subclass must implement is_adversarial.
    """

    def name(self):
        """
        Return a human readable name that uniquely identifies 
        the criterion with its hyperparameters.


        Returns
        -------
        str
            Human readable name that uniquely identifies the criterion
            with its hyperparameters.
        """
        return self.__class__.__name__

    @abstractmethod
    def is_adversarial(self, preds, refs):
        """
        Decide if predictions for the speeches are adversarial given reference
        labels (sentences). 

        Parameters
        ----------
        preds: numpy.ndarray of `str` 
            Predictions of the audios

        refs: numpy.ndarray of 'str'
            reference labels

        Returns
        -------
        numpy.array
            An array with boolean values, where an element is True indicates that 
            the speech with the given precitions is an adversarial example
        """
        raise NotImplementedError
    
class TargetSequence(Criterion):
    """
    Target sequence criteria, the audios are adversarial if their predictions are
    identical to the target sequence

    Parameters
    ----------
    target_sequence: numpy.ndarray
        A list of target sequences
    """
    def __init__(self, **kw):
        super(TargetSequence, self).__init__()

    def name(self):
        return "{}-{}".format(self.__class__.__name__, self.target_sequence())

    @input_decorator
    def is_adversarial(self, preds, refs):
        preds = np.array(preds)
        return preds == refs

class SimilarSequence(Criterion):
    """
    Similar sequence criteria, the audios are adversarial if their predictions and 
    the target sequences has similarity higher than the threshold with the given 
    similarity function

    Parameters
    ----------
    similarity_fn: a funtion handle
        A function which will be used to compute similarity between two list of strs
    threshold: float
        If the similarity value is above the threshold, the audios with the predictions are adversarial example
    """
    def __init__(self, similarity_fn=WordSimilarity(), threshold=-np.inf, **kw):
        #if isinstance(target_sequence, str):
        #    target_sequence = np.array([target_sequence])
        #self._target_sequence = np.array(target_sequence)
        self._similarity_fn = similarity_fn
        self._threshold = threshold

    @input_decorator
    def is_adversarial(self, preds, refs):
        if threshold == -np.inf:
            # if the threshold is zero, it means any inputs would be viewed as adv examples
            return np.full(shape=self._target_sequence.shape, fill_value=True, dtype=bool)
        similarity = self._similarity_fn(preds, refs)
        similarity = np.array(similarity)
        return similarity >= self.threshold
