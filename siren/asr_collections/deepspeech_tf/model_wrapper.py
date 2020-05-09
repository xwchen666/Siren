import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import warnings
from functools import wraps

from asr_collections.base import Model

def input_decorator(call_fn):
    @wraps(call_fn)
    def wrapper(self, inputs, input_lens=None, **kwargs):
        is_single_input = (isinstance(inputs, np.ndarray) and inputs.ndim == 1)
        is_single_input = is_single_input or (isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], (int, float)))

        # pad the inputs if it is list
        if isinstance(inputs, list):
            if is_single_input:
                inputs = np.array(inputs)[np.newaxis,:]
            else:
                input_lens    = np.array([len(audio) for audio in inputs])
                max_audio_len = np.max(input_lens) 
                batch_size    = len(inputs) 
                padded_inputs = np.zeros(shape=(batch_size, max_audio_len))
                for i in range(batch_size):
                    padded_inputs[i, 0:input_lens[i]] = inputs[i]
                inputs = padded_inputs
        elif isinstance(inputs, np.ndarray):
            if is_single_input:
                inputs = inputs[np.newaxis, :]
        else:
            raise ValueError('Input type should be list or np.ndarray!')

        try:
            outputs = call_fn(self, inputs, input_lens, **kwargs)
            if is_single_input:
                outputs = tuple([out[0] if out else None for out in outputs])
        except RuntimeError:
            print('Fail to call the function {} in Model {}'.format(call_fn.__name__, self.__class__.__name__))
        return outputs 
    return wrapper

class DeepSpeechTFModel(Model):
    """
    Creates a :class: `Model` instance from existing `Tensorflow` tensors

    Parameters
    ----------
    version: str
        The version of the Mozilla DeepSpeech, chosen from 1, 5, 6 
    ckpt_path: str
        The dir of the checkpoint files
    batch_size: int or None
        The batch size of the inputs
    target_phrases: List[str] or np.ndarray[str]
        Target phrase for the ctc loss tensor. If None, we would set the loss tensor later
    """
    def __init__(self, version=6, ckpt_path=None, batch_size=None, target_phrases=None, **decoder_params):
        # load ASR model 
        if version == 5:
            from asr_collections.deepspeech_tf.deepspeech_tf_5 import DeepSpeech5 as ASR
        elif version == 6:
            from asr_collections.deepspeech_tf.deepspeech_tf_6 import DeepSpeech6 as ASR
        else:
            raise ValueError('Mozilla DeepSpeech version {} not supported!'.format(version))
        
        # check parameters: checkpoint, batch dimension, audio max len
        if ckpt_path is None:
            raise ValueError('Pretrained model is required for Mozilla DeepSpeech {}!'.format(version))


        self.batch_size = batch_size

        self._graph = tf.Graph() # allow import multiple tensorflow graph at the same time
        with self._graph.as_default():          
             # define inputs and input_lens tensor
            self._inputs_tensor = inputs_tensor = tfv1.placeholder(tf.float32, shape=[batch_size, None], name='inputs')
            self._input_lens_tensor = input_lens_tensor = tfv1.placeholder(tf.float32, shape=[batch_size], name='input_lens')
            # create an ASR instance
            self._asr     = asr = ASR(inputs_tensor, input_lens_tensor, ckpt_path, **decoder_params)
            # default session
            self._sess    = sess = asr.sess
            # output we would process
            self._outputs = asr.outputs

            if target_phrases:
                self.set_loss(target_phrases)
            else:
                self._loss_tensor = None # set up loss tensor later

    def compute_seq_lengths(self, input_lens):
        """
        Given lengths of input audios, return the resulted MFCC lengths

        Parameters
        ----------
        input_lens: :class: `numpy.ndarray`
            The lengths of the input audios

        Returns
        -------
        numpy.ndarray
            The lengths of the resulted MFCCs
        """
        return self._asr.compute_seq_lengths(input_lens)

    def set_loss(self, target_phrases):
        """
        Set the loss tensor for the model

        Parameters
        ----------
        target_phrases: List[str] or np.ndarray of str
            Target phrases such as ['ok google', 'hello alexa']
        """
        with self._graph.as_default():
            # compute loss tensor
            self._asr.set_loss(target_phrases)
            # loss (usually ctc loss), tf.tensor
            self._loss_tensor = self._asr.loss
            # gradient of loss w.r.t. the inputs
            (grad_loss_input,) = tf.gradients(self._loss_tensor, self._inputs_tensor) 
            self._grad_loss_input = grad_loss_input

    @input_decorator
    def forward(self, inputs, input_lens=None): 
        if input_lens is None:
            input_lens = [inputs.shape[1]] * inputs.shape[0]

        feed_dict={self._inputs_tensor:inputs, 
                   self._input_lens_tensor:input_lens}

        if self._loss_tensor:
            eval_tensors = self._outputs.append(self._loss_tensor)
            *outputs, loss = self._sess.run(eval_tensors, feed_dict=feed_dict)
        else:
            outputs = self._sess.run(self._outputs, feed_dict=feed_dict)
            loss = None

        trans = self._asr.process_outputs(outputs)
        return trans, loss

    @input_decorator
    def gradient(self, inputs, input_lens = None):
        if self._loss_tensor is None:
            warnings.warn('CTC loss is not set, please call .set_loss(target_phrases) to set loss')
            return None

        if input_lens is None:
            input_lens = [inputs.shape[1]] * inputs.shape[0]

        feed_dict={self._inputs_tensor:inputs, 
                   self._input_lens_tensor:input_lens}

        g_loss_input = self._sess.run(self._grad_loss_input, feed_dict = feed_dict)
        return g_loss_input

    @input_decorator
    def forward_and_gradient(self, inputs, input_lens = None):
        if self._loss_tensor is None:
            warnings.warn('CTC loss is not set, please call .set_loss(target_phrases) to set loss')
            return self.forward(inputs, input_lens)

        if input_lens is None:
            input_lens = [inputs.shape[1]] * inputs.shape[0]

        feed_dict={self._inputs_tensor:inputs, 
                   self._input_lens_tensor:input_lens}

        eval_tensors = self._outputs + [self._loss_tensor, self._grad_loss_input]

        *outputs, loss, g_loss_input = self._sess.run(eval_tensors, feed_dict = feed_dict)
        
        trans = self._asr.process_outputs(outputs)
        return trans, loss, g_loss_input
