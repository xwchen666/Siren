import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import math
from abc import abstractmethod

from .base import DataAugment

class DynamicConv(DataAugment):
    """
    Dynamic convolution, i.e., the kernel changes at each step
    y_n = a_0 * x_n + a_1 * x_{n-1} + a_2 * x_{n-1} + ....
    The coefficient (a_0, a_1, a_2, ...) is a dynamic kernel

    Parameters
    ----------
    kernel_width: int
        The width of the kernel
    randomized: boolean
        If true, the dynamic kernel is randomizedly generated

    Remarks
    -------
    conv_kernel: np.ndarray
    gradient_kernel: np.ndarray
        The i-th row of the gradient_kernel is [...,\partial y_{i+j} /\partial x_i,...]
        for j >= 0
    """

    def __init__(self, kernel_width = 7, randomized = False):
        self._kernel_width   = kernel_width
        self._randomized     = randomized
        self._conv_kernel = None
        self._gradient_kernel = None # for gradient computation convenience

    @abstractmethod
    def _update_kernel(self, L):
        """
        Paramters
        ---------
        L: int
            The length of the input signals
        """
        raise NotImplementedError

    def broadcasting_app(self, inputs, win_len, S=1):
        """
        parameters
        ----------
        inputs: numpy.array
            The original inputs
        win_len: int
            Window len
        S: int
            Stride len
        """
        nrows = ((inputs.shape[-1] - win_len) // S) + 1
        return inputs[..., S * np.arange(nrows)[:,None] + np.arange(win_len)]

    def forward(self, inputs):
        assert inputs.ndim == 2
        L = inputs.shape[-1]
        if self._randomized or self._conv_kernel is None or self._conv_kernel.shape[0] != L:
            self._update_kernel(L)
        # do the convolution
        padded_inputs = np.pad(inputs, ((0, 0), (self._kernel_width - 1, 0)), 'constant')
        stacked_inputs = self.broadcasting_app(padded_inputs, self._kernel_width)
        output = np.einsum('...ij, ij->...i', stacked_inputs, self._conv_kernel) 
        return output
    
    def gradient(self, inputs):
        assert inputs.ndim == 2
        g = np.sum(self._gradient_kernel, axis=1)
        return np.tile(g, (inputs.shape[0], 1))

    def forward_and_gradient(self, inputs):
        out = self.forward(inputs)
        g = self.gradient(inputs) 
        return out, g
    
    def backward(self, inputs, g_loss_input):
        L = inputs.shape[-1]
        # pad and stack the gradient
        padded_g_loss_input = np.pad(g_loss_input, ((0, 0), (0, self._kernel_width - 1)), 'constant')
        stacked_g_loss_input = self.broadcasting_app(padded_g_loss_input, self._kernel_width) 
        g = np.einsum('...ij,ij->...i', stacked_g_loss_input, self._gradient_kernel) 
        return g

class NonRecursiveDynamicFilter(DynamicConv):
    """
    Non recursive dynamic filter, i.e., y_n is a linear combination
    of previous x_n 
    y_n = a_0 * x_n + a_1 * x_{n-1} + a_2 * x_{n-1} + ....
    The coefficient (a_0, a_1, a_2, ...) is a dynamic

    Parameters
    ----------
    window_size: int
        The window size of the filter
    randomized: boolean
        If true, the dynamic coefficient is randomizedly generated
    """
    def __init__(self, window_size = 7, randomized = False):
        super().__init__(kernel_width=window_size, randomized=randomized)
        self._window_size  = window_size

    def _update_kernel(self, L):
        # random kernel is a numpy.array with size (L, kernel_width)
        # each row of random kernel sums up to 1
        self._conv_kernel = np.random.dirichlet(np.ones(self._kernel_width), size=L) 
        # padd the bottom of the random kernel with (kernel_width - 1) rows of zeros 
        padded_kernel = np.pad(self._conv_kernel[:, ::-1], ((0, self._kernel_width - 1), (0, 0)), 'constant')
        # generate corresponding idx 
        Idx_X = np.arange(L)[:, None] + np.arange(self._kernel_width)
        Idx_Y = np.arange(self._kernel_width)[None,:] + np.zeros((L, 1), dtype=int)
        self._gradient_kernel = padded_kernel[Idx_X, Idx_Y]

class RecursiveDynamicFilter(DynamicConv):
    """
    Recursive dynamic filter, i.e., the coefficient changes at each step
    y_n = a_0 * x_n + a_1 * x_{n-1} + a_2 * x_{n-1} + ....
        + b_1 * y_{n - 1} + b2 * y_{n-2} + ...
    The coefficient (a_0, a_1, a_2, ..., b_1, b_2, ...) is dynamic 
    We convert the recursive dynamic filter to a non-recursive one via computing
    "impulse response"

    Parameters
    ----------
    window_size: int
        The window size of the filter
    randomized: boolean
        If true, the dynamic coefficient is randomizedly generated
    impulse_len: int
        The length of the impulse response
    """
    def __init__(self, window_size=7, randomized=False, impulse_len=50):
        super().__init__(kernel_width=impulse_len, randomized=randomized)
        self._window_size = window_size
        self._impulse_len = impulse_len
    
    def _get_impulse(self, filter_coefficient, impulse_len):
        """
        Get the impulse response

        Parameters
        ----------
        filter_coefficient: np.ndarray
            The recursive filter coefficient 
        impulse_len: int
            The length of the impulse response
        """
        L           = filter_coefficient.shape[0]
        window_size = filter_coefficient.shape[1]
        impulse_response = np.zeros(shape=(L, impulse_len))

        left_w = window_size - window_size // 2
        right_w = window_size // 2

        for idx in range(L):
            tmp = np.zeros(window_size - 1 + impulse_len)
            tmp[window_size - 1] = 1
            out = np.zeros(len(tmp))
            for i in range(window_size - 1, len(tmp)):
                z = i - window_size + 1
                if idx + z >= L:
                    break
                out[i] += tmp[(i + 1 - left_w):(i + 1)].dot(filter_coefficient[idx + z][0:left_w]) 
                out[i] += out[(i - right_w):i].dot(filter_coefficient[idx + z][left_w:]) 
            impulse_response[idx] = out[window_size - 1:]
        return impulse_response

    def _update_kernel(self, L):
        # each row of cofficient sums up to 1
        filter_coefficient = np.random.dirichlet(np.ones(self._window_size), size=L) 
        # get the gradient kernel first
        self._gradient_kernel = impulse = self._get_impulse(filter_coefficient, self._impulse_len)
        # get the conv kernel
        padded_impulse = np.pad(impulse, ((self._impulse_len - 1, 0), (0, 0)), 'constant')
        Idx_X = np.arange(L)[:, None] + np.arange(self._impulse_len)
        Idx_Y = np.arange(self._impulse_len - 1, -1, step=-1)[None,:] + np.zeros((L, 1), dtype=int)
        self._conv_kernel = padded_impulse[Idx_X, Idx_Y]
