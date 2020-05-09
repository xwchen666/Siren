import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import math


from .base import DataAugment

class Reverberation(DataAugment):
    ''' 
    Compute the convolution of the input_audios and the impulse responses
    the impulse response could be a room impulse response
    Here we use fft to compute the convolution due to smaller time complexity 
    For the time complexity analysis, please refer to https://ccrma.stanford.edu/~jos/ReviewFourier/FFT_Convolution_vs_Direct.html 

    Parameters
    ----------
    cutoff: bool
        if true, cutoff the output as the same length as the input
    '''
    def __init__(self, cutoff = False):

        self._cutoff = cutoff

        self._input_tensor = input_tensor = tfv1.placeholder(tf.float32, shape=[None, None], name="reverberation_input")
        self._impulse_tensor = impulse_tensor = tfv1.placeholder(tf.float32, shape=[None, None], name="reverberation_impulse")

        s1 = tf.convert_to_tensor(tf.shape(input_tensor[0,:])) # impulse length
        s2 = tf.convert_to_tensor(tf.shape(impulse_tensor[0,:])) # impulse length

        shape = s1 + s2 - 1

        sp1 = tf.signal.rfft(input_tensor, shape)
        sp2 = tf.signal.rfft(impulse_tensor, shape)
        ret = tf.signal.irfft(sp1 * sp2, shape)

        # normalization
        #ret /= tf.reduce_max(tf.abs(ret))
        #ret *= 2 ** (16 - 1) - 1
        #ret = tf.clip_by_value(ret, -2**(16 - 1), 2**(16-1) - 1)
        # normalization
        input_max = tf.reduce_max(tf.abs(input_tensor), axis=1, keepdims=True)
        output_max = tf.reduce_max(tf.abs(ret), axis=1, keepdims=True)
        ret = ret * (input_max / output_max)
        #===============


        if cutoff:
            self._ret = ret[:, 0:s1[0]]
        else:
            self._ret = ret

        (self._gradient,) = tf.gradients(self._ret, input_tensor)

        self._grad_loss_input_tensor = tfv1.placeholder(tf.float32)
        self._backward_ret = self._ret * self._grad_loss_input_tensor
        #(self._backward_grad, ) = tf.gradients(self._ret, input_tensor, grad_ys = self._grad_loss_input_tensor)
        (self._backward_grad, ) = tf.gradients(self._backward_ret, input_tensor)


    def forward(self, inputs, impulses):
        with tfv1.Session() as sess:
            ret = sess.run(self._ret, feed_dict = {self._input_tensor:inputs,
                                                   self._impulse_tensor:impulses})
        return np.array(ret)

    def gradient(self, inputs, impulses):
        with tfv1.Session() as sess:
            g = sess.run(self._gradient, feed_dict = {self._input_tensor:inputs,
                                                      self._impulse_tensor:impulses})
        return np.array(g)

    def forward_and_gradient(self, inputs, impulses):
        with tfv1.Session() as sess:
            ret, g = sess.run([self._ret, self._gradient], 
                                        feed_dict = {self._input_tensor:inputs,
                                                     self._impulse_tensor:impulses})
        return np.array(ret), np.array(g)

    def backward(self, inputs, impulses, grad_loss_input):
        with tfv1.Session() as sess:
            g = sess.run(self._backward_grad, feed_dict = {self._input_tensor:inputs,
                                                     self._impulse_tensor:impulses,
                                                     self._grad_loss_input_tensor: grad_loss_input})
        return np.array(g)

    def forward_one(self, x, impulse):
        return np.squeeze(self.forward(x[np.newaxis,:], impulse[np.newaxis,:]))

    def gradient_one(self, x, impulse):
        return np.squeeze(self.gradient(x[np.newaxis,:], impulse[np.newaxis,:]))

    def forward_and_gradient_one(self, x, impulse):
        o, g = self.forward_and_gradient(x[np.newaxis, :], impulse[np.newaxis,:])
        return np.squeeze(o), np.squeeze(g) 

    def backward_one(self, x, impulse, grad_loss_input):
        g = self.backward(x[np.newaxis,:], impulse[np.newaxis,:], grad_loss_input[np.newaxis,:])
        return np.squeeze(g)  

    def compute_output_lens(self, input_lens, impulse_lens = None):
        if impulse_lens is None or self._cutoff:
            return input_lens
        return input_lens + impulse_lens
