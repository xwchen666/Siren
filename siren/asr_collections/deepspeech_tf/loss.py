import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np

class CTCLoss:
    """
    Compute the CTC loss for the given logits and target phrases

    Parameters
    ----------
    target_pharases: list of strs
        The target pharases that the ctc loss is coomputed for
    alphabet: str
        A list of characters which corresponds to the predicted class in each time frame
        of the logits
    device: str
        The type of device that the computation is executed on

    Remarks
    -------
    The batch size is determined by the number of target phrases
    """
    def __init__(self, target_phrases, alphabet = None, device = 'cpu'):
        if alphabet is None:
            self._alphabet = " abcdefghijklmnopqrstuvwxyz'-"
        else:
            self._alphabet = alphabet

        self._batch_size = batch_size = len(target_phrases)

        self._logits = tfv1.placeholder(tf.float32, shape = (None, batch_size, len(self._alphabet)), name = 'asr_logits')
        self._logit_lens = tfv1.placeholder(tf.int32, shape=(batch_size), name = 'asr_logit_lengths')
        # convert target phrase to int index
        self._target_phrase_lens = [len(target_phrase) for target_phrase in target_phrases]
        self._target_phrases = target_phrases = [[self._alphabet.index(x) for x in target_phrase] \
                                    for target_phrase in target_phrases]

        if device == 'cpu':
            # convert target phrase to sparse tensor
            self._labels = self._ctc_label_dense_to_sparse(target_phrases)
            self._label_length = None
        else:
            # On TPU and GPU, only dense padded labels are supported
            # pad zeros
            max_len = max(self._target_phrase_lens)
            target_phrases = [target_phrase + [0]*(max_len - len(target_phrase)) for target_phrase in target_phrases]
            self._labels = tf.convert_to_tensor(target_phrases)
            self._label_length = tf.convert_to_tensor(self._target_phrase_lens)

    def compute_loss_tensor(self, logits_tensor, logit_lens):
        """
        Provide a ctc loss tensor

        Parameters
        ----------
        logits_tensor: tf.tensor
            Tensor of logits
        logits_lens: tf.tensor
            Tensor of logit lengths

        Returns
        -------
        tf.tensor
            CTC loss tensor

        Remarks
        -------
        We provide this function due to the randomizeness of some ASR models
        """
        ctc_loss = tfv1.nn.ctc_loss_v2(labels=self._labels, logits=logits_tensor, 
                                                    logit_length=logit_lens, 
                                                    label_length = self._label_length,
                                                    logits_time_major = True,
                                                    blank_index=self._alphabet.index('-'),
                                                    name = 'external_asr_ctc_loss')
        return ctc_loss

    def _ctc_label_dense_to_sparse(self, target_phrases):
        """
        Convert the ctc label to sparse matrix
        Ref: https://stackoverflow.com/questions/46654212/tf-sparsetensor-and-tf-nn-ctc-loss

        Parameters
        ----------
        target_pharases: 2D integer lists
            The target pharases in int index of the alphabet
        """
        import itertools
        indicies = []
        for ii in range(len(target_phrases)):
            indicies.extend(list(zip([ii] * len(target_phrases[ii]), range(len(target_phrases[ii])))))

        indicies = [list(indice) for indice in indicies]
        indicies = np.asarray(indicies, dtype=np.int32)

        values = np.asarray(list(itertools.chain(*target_phrases)), dtype=np.int32)

        shape = np.asarray([len(target_phrases), max(self._target_phrase_lens)], dtype=np.int64)

        return tf.SparseTensor(indicies, values, shape)
