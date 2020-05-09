import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import math
import random
import pyroomacoustics as pra

from .base import DataAugment

class FilterHelper:
    @staticmethod
    def moving_average_filter(x, window_size = 5, randomized = False):
        y = np.zeros(len(x))
        if randomized:
            kernel = np.random.dirichlet(np.ones(window_size))  # generate a random vector with length "window size" and sum 1.0
        else:
            kernel = np.ones(window_size) / window_size
        for i in range(len(y)):
            x_tmp = x[max(0, i - window_size + 1) : i + 1]
            y[i] = sum(kernel[-len(x_tmp):] * x_tmp)
        return y

    @staticmethod
    def recursive_average_filter(x, window_size = 5, randomized = False):
        y = np.zeros(len(x))
        l_x = window_size - window_size // 2
        l_y = window_size // 2
        if randomized:
            kernel = np.random.dirichlet(np.ones(window_size))
            kernel_x, kernel_y = kernel[0 : l_x], kernel[l_x:]
        else:
            kernel_x, kernel_y = np.ones(l_x) / window_size, np.ones(l_y) / window_size
        for i in range(len(y)):
            x_tmp = x[max(0, i - l_x + 1) : i + 1]
            y_tmp = y[max(0, i - l_y) : i]
            y[i] = sum(kernel_x[-(len(x_tmp)):] * x_tmp)
            if len(y_tmp) > 0:
                y[i] += sum(kernel_y[-len(y_tmp):] * y_tmp)
        return y
    
    @staticmethod
    def room_reverberation():
        # define the dimension of the room
        width  = random.randint(3, 5)
        length = random.randint(4, 6)
        height = random.randint(2, 4)
        room_dim = [width, length, height]
        # define the location of the signal source
        x_source = random.randint(0, width*10)/10. 
        y_source = random.randint(0, length*10)/10.
        z_source = random.randint(0, height*10)/10.
        source = [x_source, y_source, z_source]
        # define the location of the microphone
        x_mic = random.randint(0, width*10)/10.
        y_mic = random.randint(0, length*10)/10.
        z_mic = random.randint(0, height*10)/10.
        microphone = np.array([[x_mic], [y_mic], [z_mic]])

	# set max_order to a low value for a quick (but less accurate) RIR
        room = pra.ShoeBox(room_dim, fs=16000, max_order=100, absorption=0.2)
        signal = np.zeros(2000)
        signal[0] = 2**15 - 1
        room.add_source(source, signal=signal)

        room.add_microphone_array(pra.MicrophoneArray(microphone, room.fs))

        room.compute_rir()
        rir = room.rir[0][0]
        # rescale
        rir = rir / (np.sum(rir))
        return rir

    @staticmethod
    def phone_effect(low_freq=300, high_freq=3000, fs=16000, order=5):
        """
        Ref1: https://stackoverflow.com/questions/21871834/adding-effects-to-make-voice-sound-like-it-s-over-a-telephone 
        Ref2: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html 
        """
        from scipy import signal

        nyq = 0.5 * fs
        low = low_freq / nyq
        high = high_freq / nyq
        butter = signal.dlti(*signal.butter(order, [low, high], btype='band'))
        _, impulse = signal.dimpulse(butter, n=2000)
        return impulse[0].flatten()
    
    @staticmethod
    def echo_addition(amplitude = 0.3, shift = 20, sample_rate = 16000):
        d = int(shift / 1000 * sample_rate)
        impulse = np.zeros(d)
        impulse[0] = 1
        impulse[d - 1] = -amplitude
        return impulse

    @staticmethod
    def impulse_response(filter_fn, window_size = 5, randomized = False, len_limit = 2000):
        x = np.zeros(len_limit)
        x[0] = 1 # an impulse
        return filter_fn(*(x, window_size, randomized))

    @staticmethod
    def get_kernel(filter_name, window_size):
        switcher = {
            'moving_average'               : np.ones(window_size) /  window_size,
            'randomized_moving_average'    : np.random.dirichlet(np.ones(window_size)),
            'recursive_average'            : FilterHelper.impulse_response(FilterHelper.recursive_average_filter, window_size, randomized = False),
            'randomized_recursive_average' : FilterHelper.impulse_response(FilterHelper.recursive_average_filter, window_size, randomized = True),
            'phone_effect'                 : FilterHelper.phone_effect(),
            'echo_addition'                : FilterHelper.echo_addition(),
            'room_reverberation'           : FilterHelper.room_reverberation(),
        }
        kernel = switcher.get(filter_name, None) 
        return np.trim_zeros(kernel, trim='b') 

class Filter(DataAugment):
    def __init__(self, filter_name, window_size, use_fft = False):

        from .reverberation import Reverberation
        if use_fft:
            reverb = Reverberation(cutoff=True)
            self._reverb = reverb
        else:
            input_tensor = tfv1.placeholder(tf.float32, shape = [None, None], name='filter_input')
            kernel_tensor = tfv1.placeholder(tf.float32, shape=[None, ], name ='filter_kernel')

            batch_size = tf.convert_to_tensor(tf.shape(input_tensor[:, 0])[0])
            kernel_len = tf.convert_to_tensor(tf.shape(kernel_tensor)[0])

            pad_shape = tf.convert_to_tensor([[0, 0], [kernel_len - 1, 0]])

            padded_input = tf.pad(input_tensor, pad_shape,  'CONSTANT')
            data = tf.reshape(padded_input, [batch_size, 1, -1, 1])
            kernel = tf.reshape(kernel_tensor, [1, -1, 1, 1])
            output = tf.nn.conv2d(data, kernel, strides = [1, 1, 1, 1], padding='VALID')
            output = tf.reshape(output, [batch_size, -1]) # flatten
            # normalization
            input_max = tf.reduce_max(tf.abs(input_tensor), axis=1, keepdims=True)
            output_max = tf.reduce_max(tf.abs(output), axis=1, keepdims=True)
            output = output * (input_max / output_max)
            #===============
            (gradient,) = tf.gradients(output, input_tensor)

            self._input_tensor = input_tensor
            self._kernel_tensor = kernel_tensor
            self._output = output
            self._gradient = gradient

            self._grad_loss_input_tensor = tfv1.placeholder(tf.float32)
            (self._backward_grad, ) = tf.gradients(self._output, self._input_tensor, grad_ys=self._grad_loss_input_tensor)
        self._use_fft = use_fft

        # parameters for filters
        self._filter_name = filter_name
        self._window_size = window_size
        self._update_kernel = 'random' in filter_name
        self._filter_kernel_np = FilterHelper.get_kernel(filter_name, window_size) 

        self._sess = tfv1.Session()

    def _update(self):
            self._filter_kernel_np = FilterHelper.get_kernel(self._filter_name, self._window_size)

    @property
    def kernel(self):
        return self._filter_kernel_np

    def forward(self, inputs):
        if self._update_kernel:
            self._update()
        batch_size = inputs.shape[0]
        if self._use_fft:
            filter_kernel_np = np.tile(self._filter_kernel_np, (batch_size, 1))
            return self._reverb.forward(inputs, filter_kernel_np)
        else:
            out = self._sess.run(self._output, feed_dict={self._input_tensor:inputs, self._kernel_tensor:self._filter_kernel_np[::-1]})
            return out
            
    def gradient(self, inputs):
        batch_size = inputs.shape[0]
        if self._use_fft:
            filter_kernel_np = np.tile(self._filter_kernel_np, (batch_size, 1))
            return self._reverb.gradient(inputs, filter_kernel_np)
        else:
            g = self._sess.run(self._gradient, feed_dict={self._input_tensor:inputs, self._kernel_tensor:self._filter_kernel_np[::-1]})
            return g

    def forward_and_gradient(self, inputs):
        if self._update_kernel:
            self._update()
        batch_size = inputs.shape[0]
        if self._use_fft:
            filter_kernel_np = np.tile(self._filter_kernel_np, (batch_size, 1))
            return self._reverb.forward_and_gradient(inputs, filter_kernel_np)
        else:
            o, g = self._sess.run([self._output, self._gradient], feed_dict={self._input_tensor:inputs, self._kernel_tensor:self._filter_kernel_np[::-1]})
            return o, g

    def backward(self, inputs, grad_loss_input):
        """
        References
        ----------
        Backpropogation through conv. layer
        https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
        """
        batch_size = inputs.shape[0]
        if self._use_fft:
            filter_kernel_np = np.tile(self._filter_kernel_np, (batch_size, 1))
            return self._reverb.backward(inputs, filter_kernel_np, grad_loss_input)
        else:
            g = self._sess.run(self._backward_grad, feed_dict={
                                                    self._input_tensor:inputs, 
                                                    self._kernel_tensor:self._filter_kernel_np[::-1],
                                                    self._grad_loss_input_tensor:grad_loss_input})
            return g
