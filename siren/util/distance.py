import abc
from abc import abstractmethod

import numpy as np
import tensorflow as tf

class Distance(abc.ABC):
    """
    Base class for distances.

    This class should be subclassesed when implementating
    new distances. Subclasses must implement _forward and 
    _gradient.
    """

    @abstractmethod
    def forward(self, delta):
        """ Return the (normalized) distance value """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, delta):
        """ Return the gradient value w.r.t the delta """
        raise NotImplementedError

    def forward_and_gradient(self, delta):
        """ Return the distance and gradient w.r.t. to delta"""
        return self.forward(delta), self.gradient(delta)

    def name(self):
        return self.__class__.__name__

"""
L_p Norm Distances
"""
class MeanSquareDistance(Distance):
    def __init__(self, input = None):
        if input:
            n = len(input)
            # normalization factor
            self._f = n * (np.max(input) - np.min(input)) ** 2
        else:
            self._f = 1.0

        assert self._f != 0

    def forward(self, delta):
        return np.sum(np.square(delta), axis=-1) / self._f

    def gradient(self, delta):
        return 2 * delta / self._f

MSE = MeanSquareDistance

class MeanAbsoluteDistance(Distance):
    def __init__(self, input = None):
        if input:
            n = len(input)
            # normalization factor
            self._f = n * (np.max(input) - np.min(input))
        else:
            self._f = 1.0

        assert self._f != 0

    def forward(self, delta):
        return np.sum(np.abs(delta) / self._f, axis=-1).astype(np.float32)

    def gradient(self, delta):
        return np.sign(delta) / self._f

MAE = MeanAbsoluteDistance

class Linfinity(Distance):
    def __init__(self, input = None):
        if input:
            # normalization factor
            self._f = (np.max(input) - np.min(input))
        else:
            self._f = 1.0

    def forward(self, delta):
        return np.max(np.abs(delta), axis=-1).astype(np.float32)

    def gradient(self, delta):
        idx = np.argmax(np.abs(delta), axis=-1, keepdims=True)
        res = np.zeros(delta.shape)
        res[idx] = np.sign(delta[idx])
        return res

Linf = Linfinity

class L0(Distance):
    def forward(self, delta):
        return np.sum(delta != 0, axis=-1)

    def gradient(self, delta):
        return None


class MaskThresholdDistance(Distance):
    """
    Mask threshold distance

    Remarks
    -------
    audio is required when creating an instance of the class
    """
    def __init__(self, audios, sample_rate=16000, window_size=2048):
        self.audios = audios
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.frame_length = int(window_size)
        self.frame_step = int(window_size / 4)
        self.batch_size = audios.shape[0]
        
        self.sess = tf.compat.v1.Session() 

        self.PSDs, self.theta_xss, self.psd_maxs = self._compute_masking_threshold(self.audios, self.sample_rate)

        self.delta_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=audios.shape, name = 'delta_tensor')
        self.loss_tensor, self.gradient_tensor = self._loss()
    
    @property
    def psd(self):
        """ Return the normalized power spectral density (PSD) """
        return self.PSDs

    @property
    def mask_threshold(self):
        """ Return the masking threshold in log scale"""
        return 10 * np.log10(self.theta_xss)

    def _compute_masking_threshold(self, audios, sample_rate, window_size=2048):
        """
        Compute the masking threshold of the original audio signal
        The implementation is taken from https://github.com/tensorflow/cleverhans/blob/master/examples/adversarial_asr/generate_masking_threshold.py 

        References:
        1. Qin, Yao, et al. "Imperceptible, robust, and targeted adversarial examples for automatic speech recognition.", ICML'19 
        2. Mitchell, Joan L. "Introduction to digital audio coding and standards." Journal of Electronic Imaging, 2004
        3. Lin, Yiqing, and Waleed H. Abdulla. "Principles of psychoacoustics." Audio Watermark, 2015
        """
        import scipy.io.wavfile as wav
        from scipy.fftpack import fft
        from scipy.fftpack import ifft
        from scipy import signal
        import scipy
        import librosa

        def compute_PSD_matrix(audio, window_size):
            """
        	First, perform STFT.
        	Then, compute the PSD.
        	Last, normalize PSD.
            """
            scale = np.sqrt(8.0/3.) 
            # tf.contrib.signal.stft or tf.signal.stft has the same results as librosa.stft
            #stfts = scale * tf.contrib.signal.stft(tf.convert_to_tensor(audios), self.frame_length, self.frame_step)
            #stfts = scale * tf.signal.stft(tf.convert_to_tensor(audios), frame_length=self.frame_length, frame_step=self.frame_step, pad_end=False)
            #win = self.sess.run(stfts)
            win = scale * librosa.core.stft(audio, center=False)
            z = abs(win / window_size)
            psd_max = np.max(z*z)
            psd = 10 * np.log10(z * z + 0.0000000000000000001)
            PSD = 96 - np.max(psd) + psd
            return PSD, psd_max   

        def Bark(f):
            """returns the bark-scale value for input frequency f (in Hz)"""
            return 13*np.arctan(0.00076*f) + 3.5*np.arctan(pow(f/7500.0, 2))

        def Quiet(f):
             """returns threshold in quiet measured in SPL at frequency f with an offset 12(in Hz)"""
             thresh = 3.64*pow(f*0.001,-0.8) - 6.5*np.exp(-0.6*pow(0.001*f-3.3,2)) + 0.001*pow(0.001*f,4) - 12
             return thresh

        def two_slopes(bark_psd, delta_TM, bark_maskee):
            """
        	returns the masking threshold for each masker using two slopes as the spread function 
            """
            Ts = []
            for tone_mask in range(bark_psd.shape[0]):
                bark_masker = bark_psd[tone_mask, 0]
                dz = bark_maskee - bark_masker
                zero_index = np.argmax(dz > 0)
                sf = np.zeros(len(dz))
                sf[:zero_index] = 27 * dz[:zero_index]
                sf[zero_index:] = (-27 + 0.37 * max(bark_psd[tone_mask, 1] - 40, 0)) * dz[zero_index:] 
                T = bark_psd[tone_mask, 1] + delta_TM[tone_mask] + sf
                Ts.append(T)
            return Ts

        def compute_th(PSD, barks, ATH, freqs):
            """ returns the global masking threshold
            """
            # Identification of tonal maskers
            # find the index of maskers that are the local maxima
            length = len(PSD)
            masker_index = signal.argrelextrema(PSD, np.greater)[0]


            # delete the boundary of maskers for smoothing
            masker_index = np.delete(masker_index, np.where(masker_index==0))
            masker_index = np.delete(masker_index, np.where(masker_index==length - 1))
            num_local_max = len(masker_index)

            # treat all the maskers as tonal (conservative way)
            # smooth the PSD 
            p_k = pow(10, PSD[masker_index]/10.)    
            p_k_prev = pow(10, PSD[masker_index - 1]/10.)
            p_k_post = pow(10, PSD[masker_index + 1]/10.)
            P_TM = 10 * np.log10(p_k_prev + p_k + p_k_post)

            # bark_psd: the first column bark, the second column: P_TM, the third column: the index of points
            _BARK = 0
            _PSD = 1
            _INDEX = 2
            bark_psd = np.column_stack((barks[masker_index], P_TM, masker_index))
            bark_psd[:, _INDEX] = masker_index

            # delete the masker that doesn't have the highest PSD within 0.5 Bark around its frequency 
            for i in range(num_local_max):
                next = i + 1
                if next >= bark_psd.shape[0]:
                    break

                while bark_psd[next, _BARK] - bark_psd[i, _BARK]  < 0.5:
                    # masker must be higher than quiet threshold
                    if Quiet(freqs[int(bark_psd[i, _INDEX])]) > bark_psd[i, _PSD]:
                        bark_psd = np.delete(bark_psd, (i), axis=0)
                    if next == bark_psd.shape[0]:
                        break

                    if bark_psd[i, _PSD] < bark_psd[next, _PSD]:
                        bark_psd = np.delete(bark_psd, (i), axis=0)
                    else:
                        bark_psd = np.delete(bark_psd, (next), axis=0)
                    if next == bark_psd.shape[0]:
                        break        
                    
            # compute the individual masking threshold
            delta_TM = 1 * (-6.025  -0.275 * bark_psd[:, 0])
            Ts = two_slopes(bark_psd, delta_TM, barks) 
            Ts = np.array(Ts)

            # compute the global masking threshold
            theta_x = np.sum(pow(10, Ts/10.), axis=0) + pow(10, ATH/10.) 

            return theta_x

        """
        Main body of the function
        returns the masking threshold theta_xs and the max psd of the audio
        """
        PSDs = [None] * self.batch_size
        theta_xss = [None] * self.batch_size
        psd_maxs = [None] * self.batch_size
        for idx in range(self.batch_size):
            PSD, psd_max= compute_PSD_matrix(audios[idx], window_size)  
            freqs = librosa.core.fft_frequencies(sample_rate, window_size)
            barks = Bark(freqs)

            # compute the quiet threshold 
            ATH = np.zeros(len(barks)) - np.inf
            bark_ind = np.argmax(barks > 1)
            ATH[bark_ind:] = Quiet(freqs[bark_ind:])

            # compute the global masking threshold theta_xs 
            theta_xs = [None] * PSD.shape[1] 
            # compute the global masking threshold in each window
            for i in range(PSD.shape[1]):
                theta_xs[i] = compute_th(PSD[:,i], barks, ATH, freqs)
            theta_xs = np.array(theta_xs)

            PSDs[idx] = PSD
            theta_xss[idx] = theta_xs 
            psd_maxs[idx] = psd_max
        return np.array(PSDs), np.array(theta_xss), np.array(psd_maxs)

    def _loss(self):
        scale = np.sqrt(8.0 / 3)
        s = scale * tf.contrib.signal.stft(self.delta_tensor, self.frame_length, self.frame_step)
        z = tf.abs(s / self.window_size)
        psd = tf.square(z)
        PSD = tf.pow(10., 9.6) / self.psd_maxs[:, None, None] * psd  
        loss_th = tf.reduce_mean(tf.nn.relu(PSD - self.theta_xss), axis=(1, 2))
        (gradient,) = tf.gradients(loss_th, self.delta_tensor) 
        return loss_th, gradient

    def forward(self, delta):
        value = self.sess.run(self.loss_tensor, feed_dict={self.delta_tensor:delta})
        return value

    def gradient(self, delta):
        gradient = self.sess.run(self.gradient_tensor, feed_dict={self.delta_tensor:delta})
        return gradient

    def forward_and_gradient(self, delta):
        # TODO: adapt to latest version of tensorflow
        # TODO: normalization of masking threshold loss
        value, gradient = self.sess.run([self.loss_tensor, self.gradient_tensor], feed_dict={self.delta_tensor:delta})
        return value, gradient

MTD = MaskThresholdDistance 
