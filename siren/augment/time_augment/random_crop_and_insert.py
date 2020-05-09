import numpy as np
import numpy.random
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from .base import DataAugment

class RandomCrop(DataAugment):
    """
    Randomly crop data points from the signal

    Parameters
    ----------
    crop_num: int
        The number of data points we want to crop out from the original signal
    bulk: bool
        If True, the cropped data points are in a consecutive bulk, otherwise, the 
        cropped points are randomly distributed
    
    """
    def __init__(self, crop_num=1000, bulk=False):
        self._crop_num = crop_num
        self._bulk = bulk
        self._mask = None # crop i-th data point when the mask[i] equals to 0

        # define variables for backpropogation
        self._uncropped_index = None

    def _generate_random_crop_positions(self, input_len):
        assert self._crop_num <= input_len
        mask = np.ones(input_len)
        if self._bulk:
            random_start_position = np.random.randint(low=0, high=input_len - self._crop_num + 1)
            mask[random_start_position:random_start_position+self._crop_num] = 0
        else:
            index = np.arange(input_len)
            np.random.shuffle(index)
            mask[index[0:self._crop_num]] = 0
        return mask
        
    def _update_mask(self, input_len):
        # update mask
        self._mask = self._generate_random_crop_positions(input_len)
        # update mapped index
        index = np.arange(input_len)
        self._uncropped_index = index[self._mask==1]

    def forward(self, inputs, reuse_mask=False):
        input_len = inputs.shape[1]
        if not reuse_mask or mask is None or input_len != len(self._mask):
            self._update_mask(input_len)

        return inputs[:, self._mask == 1]

    def gradient(self, inputs):
        return np.tile(self._mask, (inputs.shape[0], 1))

    def forward_and_gradient(self, inputs, reuse_mask=False):
        output = self.forward(inputs, reuse_mask=reuse_mask)
        return output, self._mask

    def backward(self, inputs, grad_loss_inputs):
        g = np.zeros(inputs.shape)
        g[:, self._uncropped_index] = grad_loss_inputs
        return g

    def compute_output_lens(self, input_lens):
        return input_lens - self._crop_num


class RandomInsert(DataAugment):
    """
    Randomly insert zeros into the signal

    Parameters
    ----------
    insert_num: int
        The number of zeros we want to insert
    bulk: bool
        If True, the insert zero points are in a consecutive bulk, otherwise, the 
        insert zero points are randomly distributed
    """
    def __init__(self, insert_num=1000, bulk=False):
        self._insert_num = insert_num
        self._bulk = bulk
        # we expand the signal as [0, x1, 0, x2, 0, x3, 0, ........ 0, x_n, 0]
        # to insert zeros, we first select all data points from the expaned signal 
        # in odd position, then we choose `insert_num` zeros from even positions
        self._insert_position = None
        self._selected_position = None
        self._mapped_position = None

    def _generate_random_insert_positions(self, input_len):
        index = np.arange(input_len + 1)
        if self._bulk:
            random_start_position = np.random.randint(low=0, high=input_len - self._insert_num + 2)
            insert_position = index[random_start_position:random_start_position+self._insert_num]
        else:
            index = np.arange(input_len + 1)
            np.random.shuffle(index)
            insert_position = index[0:self._insert_num]
        return insert_position

    def _update_insert_position(self, input_len):
        # update insert position
        self._insert_position = self._generate_random_insert_positions(input_len)
        self._selected_position = np.concatenate((2 * self._insert_position, 2 * np.arange(input_len) + 1))
        self._selected_position.sort()
        # update map position
        index = np.arange(input_len + self._insert_num)
        self._mapped_position = index[self._selected_position % 2 == 1]
 
    def forward(self, inputs, reuse=False):
        input_len = inputs.shape[1]
        if not reuse or self._insert_position is None or input_len < np.max(self._insert_position):
            self._update_insert_position(input_len)

        output = np.zeros((inputs.shape[0], 2 * input_len + 1))
        output[:, np.arange(input_len) *2 + 1] = inputs
        return output[:, self._selected_position]

    def gradient(self, inputs):
        return np.ones(inputs.shape)

    def forward_and_gradient(self, inputs, reuse=False):
        output = self.forward(inputs, reuse=reuse)
        return output, np.ones(inputs.shape)

    def backward(self, inputs, grad_loss_inputs):
        return grad_loss_inputs[:, self._mapped_position]

    def compute_output_lens(self, input_lens):
        return input_lens + self._insert_num

class RandomMask(DataAugment):
    """
    Randomly set some positions to zeros

    Parameters
    ----------
    mask_num: int
        The number of points we want to mask out
    bulk: bool
        If True, the masked data points are in a consecutive bulk, otherwise, the 
        masked points are randomly distributed
 
    """
    def __init__(self, mask_num=1000, bulk=False):
        self._mask_num = mask_num
        self._bulk = bulk
        self._mask = None

    def _generate_random_crop_positions(self, input_len):
        assert self._mask_num <= input_len
        mask = np.ones(input_len)
        if self._bulk:
            random_start_position = np.random.randint(low=0, high=input_len - self._mask_num + 1)
            mask[random_start_position:random_start_position+self._mask_num] = 0
        else:
            index = np.arange(input_len)
            np.random.shuffle(index)
            mask[index[0:self._mask_num]] = 0
        return mask
        
    def _update_mask(self, input_len):
        self._mask = self._generate_random_crop_positions(input_len)

    def forward(self, inputs, reuse_mask=False):
        input_len = inputs.shape[1]
        if not reuse_mask or self._mask is None:
            self._update_mask(input_len)
        
        return inputs * self._mask

    def gradient(self, inputs):
        return self._mask

    def forward_and_gradient(self, inputs, reuse_mask=False):
        output = self.forward(inputs, reuse_mask=reuse_mask)
        return output, self._mask

    def backward(self, inputs, grad_loss_inputs):
        return grad_loss_inputs * self._mask
