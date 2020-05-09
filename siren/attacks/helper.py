from abc import ABC
from abc import abstractmethod
import warnings
import numpy as np
import scipy.io.wavfile as wav
import os
from recordtype import recordtype

class StopAttack(Exception):
    """
    Exception thrown to request early stopping of an attack
    if a given (optional) threshold is reached
    """
    pass

# A data structure to keep all the general configuration for attacks
AttackConfig = recordtype('AttackConfig', 
                          ['models',
                           'augments',
                           'criterion',
                           'distance',
                           'threshold',
                           'sim_fn',])


def AttackConfigCreator(models, augments, criterion, distance, threshold, sim_fn):
    """
    Function to create the config data container based on given parameters
    """
    def _has_gradient(models, augments):
        """
        Return true if gradient exists         
        """
        # check whether the data augment method has gradient method
        for augment in augments:
            if augment:
                try:
                    augment.forward_and_gradient
                    augment.backward
                except AttributeError('Data augmentation {} does not has gradient'.format(augment)):
                    return False
        # check whether the model has gradient method
        for model in models:
            try:
                model.forward_and_gradient
            except AttributeError('Model {} does not have gradient'.format(model)):
                return False
        return True

    if not isinstance(models, list):
        models = [models]
    if not isinstance(augments, list):
        augments = [augments]
    assert _has_gradient(models, augments)

    config = AttackConfig(models=models, augments=augments, criterion=criterion, distance=distance, threshold=threshold, sim_fn=sim_fn)
    return config

# A data structure to keep all the temporary data during generate adversarial examples
AdvStatusContainer = recordtype('AdvStatusContainer', 
                                ['unperturbed',
                                 'input_lens',
                                 'mask',
                                 'target_phrases',
                                 'ground_truth_trans',
                                 'batch_size',
                                 'store_dir',
                                 'ori_audio_names',
                                 'best_adversarial',
                                 'best_distance',
                                 'best_similarity',
                                 'iteration_num',])

# A function to generate the adv status container
def AdvStatusContainerCreator(unperturbed, target_phrases, original_trans, store_dir, ori_audio_names):
    # prepare unperturbed data: note down the input lens and the corresponding seq lens
    if isinstance(unperturbed, np.ndarray):
        if unperturbed.ndim == 1:
            unperturbed = unperturbed[np.newaxis,:]
        input_lens = np.array([unperturbed.shape[1]] * unperturbed.shape[0])
        max_len = np.max(input_lens)
        mask = np.ones(shape=(len(unperturbed), max_len))
    elif isinstance(unperturbed, list):
        input_lens = np.array([len(audio) for audio in unperturbed])
        # pad the unperturbed audios into a numpy.array
        max_len = np.max(input_lens)
        padded_unperturbed = np.zeros(shape=(len(unperturbed), max_len))
        mask = np.zeros(shape=(len(unperturbed), max_len))
        for i in range(len(unperturbed)):
            padded_unperturbed[i, 0:len(unperturbed[i])] = unperturbed[i]
            mask[i, 0:len(unperturbed[i])] = 1
        unperturbed = padded_unperturbed
    else:
        ValueError("Input audios are not np.ndarray or list")

    # prepare target phrases
    if isinstance(target_phrases, str) or (isinstance(target_phrases, np.ndarray) and target_phrases.ndim == 0):
        target_phrases = np.array([target_phrases])
    elif isinstance(target_phrases, list):
        target_phrases = np.array(target_phrases)
    elif not isinstance(target_phrases, np.ndarray):
        raise ValueError('Type of target_phrases {} is not supported!'.format(type(target_phrases)))

    batch_size = unperturbed.shape[0]

    adv_status_container = AdvStatusContainer(
                             unperturbed        = unperturbed,
                             input_lens         = input_lens,
                             mask               = mask,
                             target_phrases     = target_phrases,
                             ground_truth_trans = original_trans, # ground truth transcription
                             batch_size         =  batch_size, # batch size of the inputs
                             store_dir          = store_dir, # the dir we want to store the adv examples
                             ori_audio_names    = ori_audio_names,  # original base name of the unperturbed audios
                             best_adversarial   = [None] * batch_size, # best adv example found by far
                             best_distance      = np.full(batch_size, fill_value=np.inf), # smallest distance
                             best_similarity    = np.full(batch_size, fill_value=-np.inf), # similarity value
                             iteration_num      = 0,)
    return adv_status_container

class AdvServer(object):
    """
    A server class to provide services to attack methods
    The adv status container will be updated every time the service `post_new_data` 
    is called by the attack methods

    Parameters
    ----------
    attack_config: instance of :class: `AttackConfig` 
        A container which contains all the configuration info for the attack
    adv_status_container: instance of :class: `AdvStatusContainer` 
        A container which contains all the status of the adversarial examples
    """
    def __init__(self, attack_config, adv_status_container):
        self.cfg = attack_config
        self.adv = adv_status_container

        # set the ctc loss for all the models
        for model in self.cfg.models:
            model.set_loss(self.adv.target_phrases)
    @property
    def unperturbed(self):
        """ Return the unperturbed audios. """
        return self.adv.unperturbed

    @property
    def perturbed(self):
        """ Return the best adversarial example found so far. """
        return self.adv.best_adversarial

    def _is_adversarial(self, x, pred_trans):
        """
        A helper function
        Determine whether the inputs x are adversarial examples

        Parameters
        ----------
        x: :class: `numpy.ndarray` 
            The input that should be checked
        pred_trans: list of list of str
            Prediction transcripts for inputs under different ASRs and augmentation combination

        Returns
        -------
        np.ndarray of bool
            True if the input with the given precitions is an adversarial example
        np.ndarray of bool 
            True if the input is the best adversarial 
        np.ndarray of float
            Distance to the unperturbed input
        np.ndarray of float
            Accumulated similarity between predictions and target_phrases
        """
        distance = self.cfg.distance.forward(x - self.adv.unperturbed)

        N = len(pred_trans)
        similarity = np.zeros(self.adv.batch_size)
        for i in range(0, N):
            similarity += self.cfg.sim_fn(pred_trans[i], self.adv.target_phrases)

        is_adversarial = np.full(shape=self.adv.batch_size, fill_value=True, dtype=bool)
        for i in range(0, N):
            is_adversarial = np.logical_and(is_adversarial, self.cfg.criterion.is_adversarial(pred_trans[i], self.adv.target_phrases))

        if np.any(is_adversarial):
            is_best = self._save_best(x, distance, similarity, is_adversarial)
        else:
            is_best = np.full(shape=self.adv.batch_size, fill_value=False, dtype=bool)

        return is_adversarial, is_best, distance, similarity

    def _save_best(self, inputs, distances, similarity, is_adversarial):
        """
        Update the best adversarial example if the new adversarial example is better

        Parameters
        ----------
        inputs: 2D :class: `numpy.ndarray` 
            Input audio datas
        distances: 1D float :class: `numpy.ndarray`
            Distances between the original audio and current inputs
        similarity: 1D float :class: `numpy.ndarray`
            Similarity between the predictions and target_phrases
        is_adversarial: 1D bool `numpy.ndarray`
            Is the audio example adversarial

        Returns
        -------
        numpy.ndarray
            An array of boolean values. True if the newly found adversarial 
            example input is the best adversarial
        """
        inputs = np.copy(inputs) # avoid accidental inplace modifications
        is_better_example = np.logical_or(similarity > self.adv.best_similarity, np.logical_and(similarity == self.adv.best_similarity, distances < self.adv.best_distance))
        is_new_adversarial = np.logical_and(is_better_example, is_adversarial)
        # update adv status container
        self.adv.best_distance[is_new_adversarial]   = distances[is_new_adversarial]
        self.adv.best_similarity[is_new_adversarial] = similarity[is_new_adversarial]

        # store the adv examples
        if self.adv.store_dir and np.any(is_new_adversarial):
            for i in range(self.adv.batch_size):
                if is_new_adversarial[i]:
                    # adv example name is assigned as integer if the original audio name is not provided 
                    stored_audio_name = str(i) + '_adv.wav' if self.adv.ori_audio_names is None else self.adv.ori_audio_names[i].split('.')[0] + '_adv.' + self.adv.ori_audio_names[i].split('.')[-1]
                    stored_path = os.path.join(self.adv.store_dir, stored_audio_name)
                    wav.write(stored_path, 16000, np.array(inputs[i, 0:self.adv.input_lens[i]], dtype=np.int16))
        return is_new_adversarial

    def _squeeze_all(self, *args):
        return tuple(np.squeeze(arg) for arg in args)

    def _concatenate_trans(self, total_trans):
        batch_size = len(total_trans[0])
        out = [None] * batch_size
        N = len(total_trans)
        for i in range(batch_size):
            preds = [total_trans[j][i] for j in range(N)]
            out[i] = ' | '.join(preds)
        return out

    def reached_threshold(self):
        """ 
        Return true if a threshold is given and the currrently 
        best adversarial distance is smaller than the threshold 

        Similar to a `GET` service
        """
        return self.adv.threshold and self.adv.best_distance < self.cfg.threshold 

    def post_new_data(self, x):
        """
        Post the latest crafted audios to the adv_server. In this function, the
        server will: update status and return gradients.
        1. Forward the input audios through data augmentation methods and ASRs
        2. Determine whether the newly posted input audios are adversarial examples
        3. If the newly posted audios are adversarial examples, save the adv examples, and update 
        the adv status container
        4. Return gradients back to the attack methods, so that the attack methods could generate new
        possible adversarial examples

        This function will
            - catch the best adversarial found so far
            - update adv status container
            - return transcriptions and gradient

        Parameters
        ----------
        x: :class: `np.ndarray`
            Input audio files
        return_details: bool
            If true, return a more detailed results

        Returns
        -------
        numpy.ndarray of str
            predictions for the augmented input audios
        numpy.ndarray
            gradient of the loss w.r.t. the inputs

        Remarks
        -------
        The adv status container will be updated every time the service `post_new_data` is called 
        by the attack methods

        """
        self.adv.iteration_num = self.adv.iteration_num + 1

        is_single_input = x.ndim == 1

        if x.ndim == 1:
            x = x[np.newaxis,:]

        x = x * self.adv.mask
        x = np.clip(x, -2**15, 2**15 - 1).astype(np.int16)

        total_pred_trans = []
        total_g_loss_input = np.zeros(shape=x.shape)

        for augment in self.cfg.augments:
            # step 1: data augmentation
            if augment is None:
                input_audios = x
                input_lens   = self.adv.input_lens
            else:
                input_audios = augment.forward(x) 
                input_lens   = augment.compute_output_lens(self.adv.input_lens)
            input_audios = np.clip(input_audios, -2**15, 2**15-1).astype(np.int16)
            # step 2: ASR (MFCC + DNN) + Loss
            for model in self.cfg.models:
                trans, loss, g_loss_aug_input = model.forward_and_gradient(input_audios, input_lens)
                total_pred_trans.append(trans)
                # step 3: backpropogation through data augmentation to get the final gradient
                if augment:
                    g_loss_input = augment.backward(x, g_loss_aug_input)
                else:
                    g_loss_input = g_loss_aug_input
                total_g_loss_input += g_loss_input

        total_g_loss_input = total_g_loss_input * self.adv.mask
        
        is_adversarial, is_best, distance,_ = self._is_adversarial(x, total_pred_trans)

        g_distance_input = self.cfg.distance.gradient(x - self.adv.unperturbed) * self.adv.mask

        cat_out = self._concatenate_trans(total_pred_trans)

        if is_single_input:
            return self._squeeze_all(cat_out, total_g_loss_input, g_distance_input, is_adversarial, is_best, distance)
        else:
            return cat_out, total_g_loss_input, g_distance_input, is_adversarial, is_best, distance
