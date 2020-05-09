# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
from functools import partial
import numpy as np
import pandas
import tensorflow as tf
import sys
import progressbar
import shutil
import tensorflow.compat.v1 as tfv1

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

class FLAGS:
    feature_win_len     = 32        #feature extraction audio window length in milliseconds
    feature_win_step    = 20        #feature extraction window step length in milliseconds
    audio_sample_rate   = 16000     #sample rate value expected by model
    
    
    export_batch_size   = 1         #number of elements per batch on the exported graph
    export_tflite       = False     #export a graph ready for TF Lite engine
    # Checkpointing

    checkpoint_dir      = ''        #directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification
    checkpoint_secs     = 600       #checkpoint saving interval in seconds
    max_to_keep         = 5         #number of checkpoint files to keep - default value is 5
    load                = 'auto'    #"last" for loading most recent epoch checkpoint, "best" for loading best validated checkpoint, "init" for initializing a fresh model, "auto" for trying the other options in order last > best > init'

    # Geometry
    n_hidden            = 2048      #layer width to use when initialising layers

    use_seq_length      = True      #have sequence length in the exported graph

def samples_to_mfccs(input_audio, sample_rate):
    # number of mfcc features
    n_input = 26
    # sample rate value expected by the model is 16000
    audio_sample_rate = 16000
    # feature extration audio window length in milliseconds
    feature_win_len = 32
    # size of audio window in samples
    audio_window_samples = audio_sample_rate * (feature_win_len / 1000.0) 
    # stride for feature computation in samples
    audio_step_samples   = audio_sample_rate * (feature_win_len / 1000.0)

    samples = tf.expand_dims(input_audio, -1)

    spectrogram = contrib_audio.audio_spectrogram(samples,
                                                  window_size=audio_window_samples,
                                                  stride=audio_step_samples,
                                                  magnitude_squared=True)
    mfccs = contrib_audio.mfcc(spectrogram, sample_rate, dct_coefficient_count=n_input)
    mfccs = tf.reshape(mfccs, [-1, n_input])

    return mfccs, tf.shape(mfccs)[0]

def samples_to_mfccs_gradient(inputs, sample_rate, input_lens):
    # number of mfcc features
    n_input = 26
    # sample rate value expected by the model is 16000
    audio_sample_rate = 16000
    # feature extration audio window length in milliseconds
    feature_win_len = 32
    # size of audio window in samples
    audio_window_samples = int(audio_sample_rate * (feature_win_len / 1000.0))
    # stride for feature computation in samples
    audio_step_samples   = int(audio_sample_rate * (feature_win_len / 1000.0))
    # number of fft bins for stft
    n_fft = int(2 ** np.ceil(np.log2(audio_step_samples)))
    # input fft length for mfcc computation
    input_fft_length = int(n_fft // 2 + 1)

    stfts = tf.signal.stft(inputs, frame_length=audio_window_samples, frame_step=audio_step_samples, pad_end=False)
    spectrograms = tf.abs(stfts)

    from asr_collections.deepspeech_tf.compute_mfcc import MFCC_TF
    m = MFCC_TF(input_fft_length, sample_rate, filterbank_channel_count=40, dct_coefficient_count=n_input)
    mfccs = m.compute(spectrograms, squared=False) 

    mfcc_lens = tf.cast(tf.math.floor(input_lens / audio_window_samples), dtype=tf.int32)

    return mfccs, mfcc_lens

def create_overlapping_windows(batch_x):
    n_input   = int(26)
    n_context = int(9)
    batch_size = tf.shape(batch_x)[0]
    window_width = 2 * n_context + 1
    num_channels = n_input

    # Create a constant convolution filter using an identity matrix, so that the
    # convolution returns patches of the input tensor as is, and we can create
    # overlapping windows over the MFCCs.
    eye_filter = tf.constant(np.eye(window_width * num_channels)
                               .reshape(window_width, num_channels, window_width * num_channels), tf.float32) # pylint: disable=bad-continuation

    # Create overlapping windows
    batch_x = tf.nn.conv1d(batch_x, eye_filter, stride=1, padding='SAME')

    # Remove dummy depth dimension and reshape into [batch_size, n_windows, window_width, n_input]
    batch_x = tf.reshape(batch_x, [batch_size, -1, window_width, num_channels])

    return batch_x

# Graph Creation
# ==============
def variable_on_cpu(name, shape, initializer):
    r"""
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_cpu()``
    used to create a variable in CPU memory.
    """
    # Use the /cpu:0 device for scoped operations
    with tf.device('/cpu:0'):
        # Create or get apropos variable
        var = tfv1.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def dense(name, x, units, dropout_rate=None, relu=True):
    with tf.variable_scope(name):
        bias = variable_on_cpu('bias', [units], tf.zeros_initializer())
        weights = variable_on_cpu('weights', [x.shape[-1], units], tf.contrib.layers.xavier_initializer())

    output = tf.nn.bias_add(tf.matmul(x, weights), bias)

    if relu:
        relu_clip = 20.0
        output = tf.minimum(tf.nn.relu(output), relu_clip)

    if dropout_rate is not None:
        output = tf.nn.dropout(output, rate=dropout_rate)

    return output

def rnn_impl_lstmblockfusedcell(x, seq_length, previous_state, reuse):
    # Forward direction cell:
    fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(FLAGS.n_hidden, reuse=reuse)

    output, output_state = fw_cell(inputs=x,
                                   dtype=tf.float32,
                                   sequence_length=seq_length,
                                   initial_state=previous_state)

    return output, output_state

def rnn_impl_static_rnn(x, seq_length, previous_state, reuse):
    # Forward direction cell:
    fw_cell = tf.nn.rnn_cell.LSTMCell(FLAGS.n_hidden, reuse=reuse)

    # Split rank N tensor into list of rank N-1 tensors
    x = [x[l] for l in range(x.shape[0])]

    # We parametrize the RNN implementation as the training and inference graph
    # need to do different things here.
    output, output_state = tf.nn.static_rnn(cell=fw_cell,
                                            inputs=x,
                                            initial_state=previous_state,
                                            dtype=tf.float32,
                                            sequence_length=seq_length)
    output = tf.concat(output, 0)

    return output, output_state


def create_model(batch_x, seq_length, dropout, reuse=False, previous_state=None, overlap=True, rnn_impl=rnn_impl_lstmblockfusedcell):
    layers = {}

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_size = tf.shape(batch_x)[0]

    # Create overlapping feature windows if needed
    if overlap:
        batch_x = create_overlapping_windows(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2, 3])
    # Reshape to prepare input for first layer
    n_input = 26
    n_context = 9
    batch_x = tf.reshape(batch_x, [-1, n_input + 2*n_input*n_context]) # (n_steps*batch_size, n_input + 2*n_input*n_context)
    layers['input_reshaped'] = batch_x

    # The next three blocks will pass `batch_x` through three hidden layers with
    # clipped RELU activation and dropout.
    layers['layer_1'] = layer_1 = dense('layer_1', batch_x, FLAGS.n_hidden, dropout_rate=dropout[0])
    layers['layer_2'] = layer_2 = dense('layer_2', layer_1, FLAGS.n_hidden, dropout_rate=dropout[1])
    layers['layer_3'] = layer_3 = dense('layer_3', layer_2, FLAGS.n_hidden, dropout_rate=dropout[2])

    # `layer_3` is now reshaped into `[n_steps, batch_size, 2*n_cell_dim]`,
    # as the LSTM RNN expects its input to be of shape `[max_time, batch_size, input_size]`.
    layer_3 = tf.reshape(layer_3, [-1, batch_size, FLAGS.n_hidden])

    # Run through parametrized RNN implementation, as we use different RNNs
    # for training and inference
    output, output_state = rnn_impl(layer_3, seq_length, previous_state, reuse)

    # Reshape output from a tensor of shape [n_steps, batch_size, n_cell_dim]
    # to a tensor of shape [n_steps*batch_size, n_cell_dim]
    output = tf.reshape(output, [-1, FLAGS.n_hidden])
    layers['rnn_output'] = output
    layers['rnn_output_state'] = output_state

    # Now we feed `output` to the fifth hidden layer with clipped RELU activation
    layers['layer_5'] = layer_5 = dense('layer_5', output, FLAGS.n_hidden, dropout_rate=dropout[5])

    # Now we apply a final linear layer creating `n_classes` dimensional vectors, the logits.
    n_hidden_6 = 28 + 1
    layers['layer_6'] = layer_6 = dense('layer_6', layer_5, n_hidden_6, relu=False)

    # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
    # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
    # Note, that this differs from the input in that it is time-major.

    layer_6 = tf.reshape(layer_6, [-1, batch_size, n_hidden_6], name='raw_logits')
    layers['raw_logits'] = layer_6

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, layers

def create_feature_graph(input_audio, input_audio_lens):
    # Create feature computation graph
    mfccs, mfccs_len = samples_to_mfccs_gradient(input_audio, FLAGS.audio_sample_rate, input_audio_lens)
    features_len = mfccs_len
    features = create_overlapping_windows(mfccs)

    return mfccs, features, features_len

def create_inference_graph(input_tensor, seq_length, batch_size=-1, n_steps=-1):
    batch_size = batch_size if batch_size > 0 else None

    n_input   = int(26)
    n_context = int(9)

    if batch_size is None:
        # no state management since n_step is expected to be dynamic too (see below)
        previous_state = previous_state_c = previous_state_h = None
    else:
        previous_state_c = variable_on_cpu('previous_state_c', [batch_size, FLAGS.n_hidden], initializer=None)
        previous_state_h = variable_on_cpu('previous_state_h', [batch_size, FLAGS.n_hidden], initializer=None)
        previous_state = tf.contrib.rnn.LSTMStateTuple(previous_state_c, previous_state_h)

    # One rate per layer
    no_dropout = [None] * 6

    rnn_impl = rnn_impl_lstmblockfusedcell

    logits, layers = create_model(batch_x=input_tensor,
                                  seq_length=seq_length if FLAGS.use_seq_length else None,
                                  dropout=no_dropout,
                                  previous_state=previous_state,
                                  overlap=False,
                                  rnn_impl=rnn_impl)

    if batch_size is None:
        if n_steps > 0:
            raise NotImplementedError('dynamic batch_size expect n_steps to be dynamic too')
        outputs = {'outputs': tf.identity(logits, name='logits'), 'initialize_state': None}
        return outputs, layers

    new_state_c, new_state_h = layers['rnn_output_state']

    zero_state = tf.zeros([batch_size, FLAGS.n_hidden], tf.float32)
    initialize_c = tf.assign(previous_state_c, zero_state)
    initialize_h = tf.assign(previous_state_h, zero_state)
    initialize_state = tf.group(initialize_c, initialize_h, name='initialize_state')
    with tf.control_dependencies([tf.assign(previous_state_c, new_state_c), tf.assign(previous_state_h, new_state_h)]):
        logits = tf.identity(logits, name='logits')

    outputs = {
        'outputs': logits,
        'initialize_state': initialize_state,
    }

    return outputs, layers

from asr_collections.deepspeech_tf.deepspeech_base import DeepSpeechBase
class DeepSpeech5(DeepSpeechBase):
    # expected input range (-1, 1)
    require_batch_dimension = False
    require_audio_length = False
    def __init__(self, inputs_tensor, input_lens_tensor, checkpoint_path, **decoder_params):
        super().__init__(inputs_tensor, input_lens_tensor, checkpoint_path, **decoder_params)

        tfv1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        # load model
        log_placement = False # whether to log device placement of the operators to the console
        inter_op_parallelism_threads = 0 # number of inter-op parallelism threads - see tf.ConfigProto for more details. USE OF THIS FLAG IS UNSUPPORTED
        intra_op_parallelism_threads = 0 # number of intra-op parallelism threads - see tf.ConfigProto for more details. USE OF THIS FLAG IS UNSUPPORTED
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_placement,
                                      inter_op_parallelism_threads=inter_op_parallelism_threads,
                                      intra_op_parallelism_threads=intra_op_parallelism_threads)

        self._sess = tf.Session(config=session_config)

        rescaled_inputs_tensor = tf.cast(self._inputs_tensor, dtype=tf.float32) / 2**15

        self._mfccs, self._features, self._feature_lens = create_feature_graph(rescaled_inputs_tensor, self._input_lens_tensor)

        self._outputs, _ = create_inference_graph(self._features, self._feature_lens)

        if self._outputs['initialize_state']:
            self._sess.run(self._outputs['initialize_state'])

        self._logits = self._outputs['outputs']

        # prepare decoder
        self._set_outputs(decoder_params)

        # Create a saver using variables from the above newly created graph
        mapping = {v.op.name: v for v in tf.global_variables() if not v.op.name.startswith('previous_state_')}
        self._saver = tf.train.Saver(mapping)

        self._saver.restore(self._sess, checkpoint_path)

    def compute_seq_lengths(self, input_lens):
        audio_sample_rate = 16000
        # feature extration audio window length in milliseconds
        feature_win_len = 32
        # size of audio window in samples
        audio_window_samples = int(audio_sample_rate * (feature_win_len / 1000.0))

        return np.floor(input_lens / audio_window_samples).astype(int)
