"""
Decode the logit (after-softmax) to huam-readable sentence
We provide two decoders

    GreedyDecoder
    BeamSearchWithLMDecoder

Examples
--------
Greedy decoder
>>> from decoder import GreedyDecoder
>>> decoder1 = GreedyDecoder()
>>> trans = decoder1.decode(logits) 

Beam search decoder with language model
>>> from decoder import BeamSearchWithLMDecoder
>>> alphabet = " abcdefghijklmnopqrstuvwxyz'-"
>>> lm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './external_source/lm.binary')
>>> decoder3 = BeamSearchWithLMDecoder(alphabet = alphabet, lm_path = lm_path, beam_width = 10)
"""

import abc
from abc import abstractmethod
import torch
import scipy.special
import numpy as np 
import re
from itertools import groupby  
 
class BasicDecoder(abc.ABC):
    @abstractmethod
    def decode(self, logits, seq_lens = None):
        """
        Convert the logits to the human-readable transcription 

        Parameters
        ----------
        logits: numpy.ndarray
            Logits vector with dimension [max_len, batch_size, token_len], 
            before the softmax layer
        seq_lens: numpy.ndarray or None
            The lengths of mfccs for each audio

        Returns
        -------
        A list of str with size [batch_size]
            Return the human-readable transcription of the speech

        Remarks
        -------
        Support `batch` decode

        """
        raise NotImplementedError

    def decode_one(self, logit):
        """
        Convert the single logit to human-readable transcription

        Parameters
        ----------
        logit: numpy.ndarray
            Logits vector with dimension [max_len, token_len], 
            before the softmax layer

        Returns
        -------
        str 
            Return the human-readable transcription of the speech
        """
        logits = logit[np.newaxis,:] 
        return self.decode(logits)[0]

class GreedyDecoder(BasicDecoder):
    """
    Perform CTC greedy decoding on the logits given input
    """
    def __init__(self, alphabet = None, blank_char='-'):
        if alphabet is None:
            self._alphabet = " abcdefghijklmnopqrstuvwxyz'-"
        else:
            self._alphabet = alphabet

        self._blank_char = blank_char

    def _convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]]).lower()

    def _remove_repetition(self, sequence):
        res = ''.join([g[0] for g in groupby(sequence)])
        res = res.replace(self._blank_char, '')
        return re.sub(' +', ' ', res)

    def decode(self, logits, seq_lens = None):
        """
        Given the logits (pre-softmax), compute the transcription with 
        the ctc greedy decoder implemented in `TensorFlow` 

        Parameters
        ----------
        logits: numpy.ndarray
            Logits vector with dimension [batch_size, max_len, token_len], 
            after or before the softmax layer
        seq_lens: numpy.ndarray or None
            The lengths of mfccs for each audio

        Returns
        -------
        A list of str with size [batch_size]
            Return the human-readable transcription of the speech
        """
        # seq_len: 1d array with size [batch_size]
        #   A vector contains the feature lengths for all speech in the batch
        if seq_lens is None:
            seq_lens = np.ones(logits.shape[1]) * logits.shape[0]   

        res = np.argmax(logits, axis=-1)
        trans = [self._convert_to_string(res[ii], self._alphabet, seq_lens[ii]) for ii in range(len(res))]
        trans = [self._remove_repetition(tran) for tran in trans]
        return trans
 
class BeamSearchWithLMDecoder(BasicDecoder):   
    """
    Decode the probability sequence with language model
    To use the decoder, the package `ctcdecode` is required
    To install the `ctcdecode` package, please follow the instruction in the
    repository https://github.com/parlance/ctcdecode

    Parameters
    ----------
    alphabet: str
        A list of characters which corresponding the predicted classes in each time window
        Example: list(" abcdefghijklmnopqrstuvwxyz'-"). 
    beam_width: int
        The width of beam in the beam search
    lm_path: str
        The path of the language model if we choose the paddle as the backend
        The pretrained language model can be obtained from
            * http://www.openslr.org/11/
            * https://github.com/mozilla/DeepSpeech/releases/tag/v0.5.1

    Notes
    -----
    The format of the language model could be binary or arpa. For different language models, the 
    alphabet could be different. For example, the alphabet for the language model downloaded from 
    "http://www.openslr.org/11/" should  be upper cases of " abcdefghijklmnopqrstuvwxyz'-".
    Please provide corret alphabet for the corresponding language model
    """

    def __init__(self, alphabet, lm_path, beam_width = 10):
        super().__init__()
        self._alphabet = alphabet
        self._lm_path = lm_path
        self._beam_width = beam_width

        import torch
        import ctcdecode
        self.decoder = ctcdecode.CTCBeamDecoder(self._alphabet, beam_width=self._beam_width, 
                                            blank_id=self._alphabet.index('-'),
                                            model_path=self._lm_path,
                                            alpha=0.75, beta=1.85)

    def _convert_to_string(self, tokens, vocab, seq_len):
        return ''.join([vocab[x] for x in tokens[0:seq_len]]).lower()

    def decode(self, logits, seq_lens=None, before_softmax = True):
        """
        Parameters
        ----------
        logits: numpy.ndarray
            Logits vector with dimension [batch_size, max_len, token_len], 
            after the softmax layer

        Returns
        -------
        A list of str with size [batch_size]
            Return the human-readable transcription of the speech
        """
        probs_seq = torch.FloatTensor(logits)
        seq_lens  = torch.Tensor(seq_lens)

        beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(probs_seq, seq_lens)
        trans = [self._convert_to_string(beam_result[ii][0], self._alphabet, out_seq_len[ii][0]) for ii in range(beam_result.shape[0])]
        return trans
