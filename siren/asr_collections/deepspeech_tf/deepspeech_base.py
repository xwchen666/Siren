import tensorflow as tf
import numpy as np
from abc import abstractmethod

class DeepSpeechBase:
    def __init__(self, inputs_tensor, input_lens_tensor, checkpoint_path, **decoder_params):
        self._checkpoint = checkpoint_path
        self._inputs_tensor = inputs_tensor
        self._input_lens_tensor = input_lens_tensor
        self._decoder_name = decoder_params.get('decoder_name', 'greedy')

    def _set_outputs(self, decoder_params):
        """
        Set the final tensor we would to eval according to different decoder
        If the decoder is `greedy` or `beamsearch`, the output would be the decoded tensor
        If the decoder is `beamsearch_lm`, the output would be the logit tensor and logit_lens tensor
        """
        if self._decoder_name == 'greedy':
            decoded = tf.nn.ctc_greedy_decoder(self._logits, self._feature_lens, merge_repeated=True)
            self._outputs = [decoded]
        elif self._decoder_name == 'beamsearch':
            beam_width = decoder_params.get('beam_width', 10)
            decoded = tf.nn.ctc_beam_search_decoder(self._logits, self._feature_lens, beam_width=beam_width, top_paths=1)
            self._outputs = [decoded]
        elif self._decoder_name == 'beamsearch_lm':
            alphabet = decoder_params.get('alphabet', list(" abcdefghijklmnopqrstuvwxyz'-"))
            lm_path = decoder_params.get('lm_path', None)
            if lm_path is None:
                raise ValueError('Path to LM must be provided for the decoder')
            beam_width = decoder_params.get('beam_width', 10)
            from asr_collections.decoder import BeamSearchWithLMDecoder
            self._decoder = BeamSearchWithLMDecoder(alphabet=alphabet, lm_path=lm_path, beam_width=beam_width) 
            logits_after_softmax = tf.nn.softmax(self._logits, axis=-1)
            transposed_logits_after_softmax = tf.transpose(logits_after_softmax, perm=[1, 0, 2])
            self._outputs = [transposed_logits_after_softmax, self._feature_lens]
        else:
            raise ValueError('Decoder name shoud be: greedy/beamsearch/beamsearch_lm')

    def set_loss(self, target_pharases):
        """
        Set the ctc loss tensor
        """
        from asr_collections.deepspeech_tf.loss import CTCLoss
        loss = CTCLoss(target_pharases)
        self._loss = ctc_loss_tensor = loss.compute_loss_tensor(self._logits, self._feature_lens)

    def process_outputs(self, outputs):
        """
        Process the output value to human-readable transcripts
        """
        if self._decoder_name == 'greedy' or self._decoder_name == 'beamsearch':
            alphabet = " abcdefghijklmnopqrstuvwxyz'-"
            outs,_  =  outputs[0]           
            res = np.zeros(outs[0].dense_shape) + len(alphabet) - 1
            for ii in range(len(outs[0].values)):
                x, y = outs[0].indices[ii]
                res[x, y] = outs[0].values[ii]
            res = ["".join([alphabet[int(x)] for x in y]).replace("-", "") for y in res]
        else:
            logits, logit_lens = outputs # unzip
            res = self._decoder.decode(logits, logit_lens)
        return res

    @abstractmethod
    def compute_seq_lengths(self, input_lens):
        """
        Compute the length of the MFCC features

        input_lens: 1D :class: `numpy.array` with length batch size
            The length of audios 
        """
        raise NotImplementedError

    @property
    def sess(self):
        """
        return current session
        """
        return self._sess

    @property
    def inputs(self):
        """
        return the placeholder of the input audio
        """
        return self._inputs

    @property
    def outputs(self):
        """
        return the placeholder of the input audio
        """
        return self._outputs

    @property
    def loss(self):
        """
        return the ctc loss tensor
        """
        return self._loss

    @property
    def logits(self):
        """
        return the logits tensor (before softmax)
        """
        return self._logits

    @property
    def mfccs(self):
        """
        return the mfccs tensor of the input audio
        """
        return self._mfccs 

    @property
    def mfcc_lens(self):
        """
        return the lens tensor of mfccs/features
        """
        return self._feature_lens

    @property
    def features(self):
        """
        DNN input (tensor), overlapped MFCCs
        """
        return self._features
