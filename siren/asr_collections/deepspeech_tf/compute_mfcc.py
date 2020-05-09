import math
import numpy as np
import tensorflow as tf

kDefaultDCTCoefficientCount = 13
kDefaultFilterbankChannelCount = 40
kFilterbankFloor = 1e-12
kDefaultLowerFrequencyLimit = 20
kDefaultUpperFrequencyLimit = 4000

class MFCC_Dct:
    """
    Parameters
    ----------
    input_length: int
        filterbank coefficient count
    coefficient_count: int
        DCT cofficient count

    C++ implementation 
    ------------------
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/mfcc_dct.cc
    """
    def __init__(self, input_length, coefficient_count):
        assert coefficient_count >= 1
        assert input_length >= 1
        assert coefficient_count <= input_length

        cosines = np.zeros((coefficient_count, input_length))
        fnorm = math.sqrt(2.0 / input_length)
        pi = math.atan(1) * 4
        arg = pi / input_length
        for i in range(coefficient_count):
            cosines[i, :] = fnorm * np.cos(i * arg * (np.arange(input_length) + 0.5))

        self._cosines = cosines.T

    @property
    def cosine_coefficient(self):
        return self._cosines

    def compute(self, input):
        """
        Perform DCT transformation on the input spectrogram

        Parameters
        ----------
        input: numpy.ndarray of shape (time_frame_lens, input_length)
            The spectrogram after mel filter bank 

        Returns
        -------
        numpy.ndarray of shape (time_frame_lens, coefficient_count)
        """
        return np.matmul(input, self._cosines)

class MFCC_MelFilterbank:
    """
    Parameters
    ----------
    input_length: int
        Number of FFT bins of the input spectrogram
    input_sample_rate: int
        sample rate of the input audio
    output_channel_count: int (default 40)
        filterbank output channel count
    lower_frequency_limit: int (default 20)
        lower frequency limit of the filterbank
    upper_frequency_limit: int (default 4000)
        upper frequency limit of the filterbank

    Remarks
    -------
    This code resamples the FFT bins, and smooths then with triangle-shaped
    weights to create a mel-frequency filter bank. For filter i centered at f_i,
    there is a triangular weighting of the FFT bins that extends from
    filter f_i-1 (with a value of zero at the left edge of the triangle) to f_i
    (where the filter value is 1) to f_i+1 (where the filter values returns to
    zero).

    Note: this code fails if you ask for too many channels.  The algorithm used
    here assumes that each FFT bin contributes to at most two channels: the
    right side of a triangle for channel i, and the left side of the triangle
    for channel i+1.  If you ask for so many channels that some of the
    resulting mel triangle filters are smaller than a single FFT bin, these
    channels may end up with no contributing FFT bins.  The resulting mel
    spectrum output will have some channels that are always zero.

    C++ implementation
    ------------------
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/mfcc_mel_filterbank.cc
    """
    def __init__(self, 
                input_length, 
                input_sample_rate, 
                output_channel_count = 40, 
                lower_frequency_limit = 20, 
                upper_frequency_limit = 4000):

        self._num_channels = output_channel_count
        self._sample_rate = input_sample_rate
        self._input_length = input_length

        assert self._num_channels >= 1 
        assert self._sample_rate > 0
        assert self._input_length >= 2
        assert lower_frequency_limit >= 0
        assert upper_frequency_limit > lower_frequency_limit

        # An extra center frequency is computed at the top to get the upper
        # limit on the high side of the final triangular filter.
        mel_low = self.FreqToMel(lower_frequency_limit)
        mel_hi = self.FreqToMel(upper_frequency_limit)
        mel_span = mel_hi - mel_low
        mel_spacing = mel_span / (self._num_channels + 1)
        center_frequencies = np.zeros(self._num_channels + 1)
        center_frequencies = (np.arange(0, self._num_channels + 1) + 1) * mel_spacing + mel_low

        # always exclude DC; emulate HTK
        hz_per_sbin =  0.5 * self._sample_rate / (self._input_length - 1)
        self.start_index = start_index = int(1.5 + (lower_frequency_limit / hz_per_sbin))
        self.end_index = end_index = int(upper_frequency_limit / hz_per_sbin)

        # Maps the input spectrum bin indices to filter bank channels/indices. For
        # each FFT bin, band_mapper tells us which channel this bin contributes to
        # on the right side of the triangle.  Thus this bin also contributes to the
        # left side of the next channel's triangle response.
        channel = 0
        band_mapper = np.zeros(self._input_length, dtype=np.int32)
        band_mapper_matrix_left = np.zeros((self._num_channels, self._input_length))
        band_mapper_matrix_right = np.zeros((self._num_channels, self._input_length))
        for i in range(self._input_length):
            melf = self.FreqToMel(i * hz_per_sbin)
            if i < start_index or i > end_index:
                band_mapper[i] = -2
            else:
                while channel < self._num_channels and center_frequencies[channel] < melf:
                    channel += 1
                band_mapper[i] = int(channel) - 1
                if band_mapper[i] >= 0:
                    band_mapper_matrix_left[band_mapper[i], i] = 1
                if band_mapper[i] + 1 < self._num_channels:
                    band_mapper_matrix_right[band_mapper[i] + 1, i] = 1


        # Create the weighting functions to taper the band edges.  The contribution
        # of any one FFT bin is based on its distance along the continuum between two
        # mel-channel center frequencies.  This bin contributes weights_[i] to the
        # current channel and 1-weights_[i] to the next channel.
        weights = np.zeros(self._input_length)
        for i in range(self._input_length):
            channel = band_mapper[i]
            if i >= start_index and i <= end_index:
                if channel >= 0:
                    weights[i] = (center_frequencies[channel + 1] - self.FreqToMel(i * hz_per_sbin)) /\
                                 (center_frequencies[channel + 1] - center_frequencies[channel])
                else:
                    weights[i] = (center_frequencies[0] - self.FreqToMel(i * hz_per_sbin)) / \
                                 (center_frequencies[0] - mel_low)

        # Check the sum of FFT bin weights for every mel band to identify
        # situations where the mel bands are so narrow that they don't get
        # significant weight on enough (or any) FFT bins -- i.e., too many
        # mel bands have been requested for the given FFT size.
        bad_channels = [] 
        for c in range(self._num_channels):
            band_weights_sum = 0.0
            for i in range(self._input_length):
                if band_mapper[i] == c - 1:
                    band_weights_sum += (1.0 - weights[i])
                elif band_mapper[i] == c:
                    band_weights_sum += weights[i]
            if band_weights_sum < 0.5:
                bad_channels.append(c)

        if len(bad_channels) > 0:
            print('Missing ' + len(bad_channels) + " bands!")
            exit(-1)

        M = np.matmul(band_mapper_matrix_left, np.diag(weights)) + \
            np.matmul(band_mapper_matrix_right, np.diag(1 - weights)) 
        
        self._mel_filterbank_matrix = M.T

    @property
    def mel_weighted_matrix(self):
        return self._mel_filterbank_matrix

    def compute(self, input, squared=True):
        """
        Compute the mel spectrum from the squared-magnitude FFT input by taking the
        square root, then summing FFT magnitudes under triangular integration windows
        whose widths increase with frequency.

        Parameters
        ----------
        input: numpy.ndarray with shape (time_frame_lens, fft_bins)
            The input spectrogram

        Returns
        -------
        numpy.ndarray with shape (time_frame_lens, output_channel_count)
        """

        if len(input) <= self.end_index:
            print("Input too short to compute filterbank")
            return None

        if squared:
            spec_val_vec = np.sqrt(input)
        else:
            spec_val_vec = input

        return spec_val_vec @ self._mel_filterbank_matrix

    def FreqToMel(self, freq):
        return 1127.0 * math.log1p(freq / 700.0)

class MFCC:
    """
    Compute the MFCCs

    Parameters
    ----------
    input_length: int
        Number of FFT bins of the input spectrogram
    input_sample_rate: int
        sample rate of the input audio
    filterbank_channel_count: int (default 40)
        filterbank output channel count
    dct_coefficient_count: int (default 13)
        DCT cofficient count
    lower_frequency_limit: int (default 20)
        lower frequency limit of the filterbank
    upper_frequency_limit: int (default 4000)
        upper frequency limit of the filterbank

    C++ implementation
    ------------------
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/mfcc.cc
    """

    def __init__(self, 
                input_length, 
                input_sample_rate,
                filterbank_channel_count = kDefaultFilterbankChannelCount,
                dct_coefficient_count = kDefaultDCTCoefficientCount,
                lower_frequency_limit = kDefaultLowerFrequencyLimit,
                upper_frequency_limit = kDefaultUpperFrequencyLimit
                ):
        self.mel_filterbank = MFCC_MelFilterbank(input_length, 
                                            input_sample_rate, 
                                            filterbank_channel_count,
                                            lower_frequency_limit,
                                            upper_frequency_limit)
        self.dct = MFCC_Dct(filterbank_channel_count, dct_coefficient_count)

    def compute(self, spectrograms, squared = True):
        """
        Compute mfcc for the input spectrograms

        Parameters
        ----------
        spectrograms: numpy.ndarray with shape (time_frame_lens, fft_bin_lens)
            The input spectrograms
        squared (optional): bool
            If true, it indicates the input spectrograms are already squared

        Returns
        -------
        numpy.ndarray with shape (time_frame_lens, dct_cofficient_count)
        """

        working = self.mel_filterbank.compute(spectrograms, squared)
        working[working < kFilterbankFloor] = kFilterbankFloor
        working = np.log(working)
        return self.dct.compute(working)

class MFCC_TF:
    """
    Compute the MFCCs

    Parameters
    ----------
    input_length: int
        Number of FFT bins of the input spectrogram
    input_sample_rate: int
        sample rate of the input audio
    filterbank_channel_count: int (default 40)
        filterbank output channel count
    dct_coefficient_count: int (default 13)
        DCT cofficient count
    lower_frequency_limit: int (default 20)
        lower frequency limit of the filterbank
    upper_frequency_limit: int (default 4000)
        upper frequency limit of the filterbank

    C++ implementation
    ------------------
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/mfcc.cc
    """

    def __init__(self, 
                input_length, 
                input_sample_rate,
                filterbank_channel_count = kDefaultFilterbankChannelCount,
                dct_coefficient_count = kDefaultDCTCoefficientCount,
                lower_frequency_limit = kDefaultLowerFrequencyLimit,
                upper_frequency_limit = kDefaultUpperFrequencyLimit
                ):
        self.mel_filterbank = MFCC_MelFilterbank(input_length, 
                                            input_sample_rate, 
                                            filterbank_channel_count,
                                            lower_frequency_limit,
                                            upper_frequency_limit)
        self.dct = MFCC_Dct(filterbank_channel_count, dct_coefficient_count)

        self.linear_to_mel_weight_matrix = tf.convert_to_tensor(self.mel_filterbank.mel_weighted_matrix, dtype=tf.float32)
        self.dct_coefficient_matrix = tf.convert_to_tensor(self.dct.cosine_coefficient, dtype=tf.float32)

    def compute(self, spectrograms, squared = True):
        """
        Compute mfcc for the input spectrograms

        Parameters
        ----------
        spectrograms: tensorflow tensor with shape (fft_bin_lens, time_frame_lens)
            The input spectrograms
        squared (optional): bool
            If true, it indicates the input spectrograms are already squared

        Returns
        -------
        Tensorflow tensor with shape (dct_cofficient_count, time_frame_lens)
        """

        if squared:
            spectrograms = tf.math.sqrt(spectrograms)

        mel_spectrograms = tf.tensordot(
            spectrograms, self.linear_to_mel_weight_matrix, 1)

        mel_spectrograms = tf.clip_by_value(mel_spectrograms, 
                                            clip_value_min = kFilterbankFloor,
                                            clip_value_max = tf.float32.max)
        log_mel_spectrograms = tf.math.log(mel_spectrograms)

        mfccs = tf.tensordot(log_mel_spectrograms, self.dct_coefficient_matrix, 1)

        return mfccs
        