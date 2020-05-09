from abc import ABC
from abc import abstractmethod
from functools import wraps
import warnings
import numpy as np
import scipy.io.wavfile as wav
import librosa
import os

from util.distance import MSE
from util.criteria import TargetSequence
from util.similarity import WordSimilarity
from attacks.helper import StopAttack, AttackConfigCreator, AdvStatusContainerCreator, AdvServer

class Attack(ABC):
    """
    Abstract base class for attacking the DNN-based automatic 
    speech recognition (ASR) systems.

    The :class: `Attack` class represents an atatck method that searches 
    for the adversarial examples with minimum perturbation. 
    It should be subclassed when implementing new attack methods.

    Parameters
    ----------
    models: a list of :class: `Model` instances
        The ASR model that should be fooled by the adversarial.     
    augments: a list of :class: `DataAugment` instances
        The data augmentation method. If None, the audio is fed into the ASR 
        model directly
    criterion: a :class: `Criterion` instance
        The criterion that determines which inputs are adversarial
        The default criterion is `TargetSequence`
    distance: a :class: `Distance`
        The distance measure used to quantify the perturbation scale
        The default is `MSE` (Mean Square Error)
    threshold: float 
        If not None, the attack will stop as soon as the adversarial perturbation has
        a size smaller than this threshold
    """
    def __init__(self, models, augments=None, criterion=TargetSequence(), distance=MSE(), threshold=None, sim_fn=WordSimilarity()):
        self.cfg = AttackConfigCreator(models, augments, criterion, distance, threshold, sim_fn)
 
    def _initialize(self):
        """Additional initializer that can be overwritten by
        subclasses without redefining the full `__init__` method
        including all arguments and documentation.
        """
        pass

    def name(self):
        """Returns a human readable name that uniquely identifies
        the attack with its hyperparameters.
        Returns
        -------
        str
            Human readable name that uniquely identifies the attack
            with its hyperparameters.
        Notes
        -----
        Defaults to the class name but subclasses can provide more
        descriptive names and must take hyperparameters into account.
        """
        return self.__class__.__name__

    def __call__(self, unperturbed, target_phrases, original_trans = None, store_dir = None, ori_audio_names = None, unpack=True, **kwargs):
        """
        Parameters
        ----------
        unperturbed: list of 1D np.ndarray
            List of input audios
        target_phrases: list of str
            List of target phrase of the attack
        original_trans: list of str
            List of ground truth transcripts of the audios
        store_dir: str
            If not None, store the generated adversarial examples in the dir `store_dir`
        ori_audio_names: list of str
            List of filenames of the input audios
        """

        # create a status data container
        adv_status_container = AdvStatusContainerCreator(unperturbed, target_phrases, original_trans, store_dir, ori_audio_names)

        # create a adversarial server which provides multiple functions for the attack methods
        adv_server = AdvServer(self.cfg, adv_status_container)
        
        """This part of the code runs the attack"""
        try:
            _ =  self._attack_implementation(adv_server, **kwargs)
        except StopAttack:
            print('Fail to call the attack function')
      
        if unpack: # only return the adversarial examples
            return adv_server.perturbed
        else:
            return adv_server.adv # return the adv status container
 
    @abstractmethod
    def _attack_implementation(self, adv_server, **kwargs):
        raise NotImplementedError
