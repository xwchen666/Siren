import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch import Tensor
from typing import Callable

class AudioParser(torch.nn.Module):
    """
    Parse raw audio data into spectrograms

    Parameters
    ----------
    sample_rate: int, optional
        Sample rate of the input audio (Default: 16000)
    window_size: float, optional
        Length of  window (in seconds) of the stft (Default: 0.02)
    window_stride: float
        Stride length (in seconds) of the stft (Default: 0.01)
    window_fn: Callable[..., Tensor], optional 
        A function to create a window tensor that is applied/multipiled to each frame/window (Default: ``torch.hamming_window``)
    normalized: bool, optional
        Whether to normalize the final stft magnitude (Default: ``True``)

    TODO
    ----
    Run on GPU
    """
    def __init__(self, 
                 sample_rate : int = 16000, 
                 window_size : float = 0.02, 
                 window_stride : float = 0.01, 
                 window_fn : Callable[..., Tensor] = torch.hamming_window, 
                 normalize : bool = True) -> None:
        super(AudioParser, self).__init__()
        self.n_fft       = n_fft = int(sample_rate * window_size)
        self.win_length  = win_length = n_fft
        self.hop_length  = hop_length = int(sample_rate * window_stride)
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, 
                                win_length=win_length, 
                                hop_length=hop_length, 
                                window_fn=window_fn, 
                                power=2, normalized=False)
        self.normalize = normalize

    def _get_batch_spectrogram(self, batch):
        #batch = sorted(batch, key=lambda sample: sample.size(1), reverse=True)
        minibatch_size = len(batch)
        freq_size = batch[0].size(0)
        max_seqlength = max([sample.size(1) for sample in batch])
        inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
        input_sizes = torch.IntTensor(minibatch_size) 
        for x in range(minibatch_size):
            sample = batch[x]
            seq_length = sample.size(1)
            inputs[x][0].narrow(1, 0, seq_length).copy_(sample)
            input_sizes[x] = seq_length
        return inputs, input_sizes

    def _forward_list(self, waveforms):
        batch = []
        for x in waveforms:
            x = x / 32767 # normalize audio
            x = self.spectrogram(x)
            x = torch.sqrt(x)
            x = torch.log1p(x)
            if self.normalize:
                mean = x.mean()
                std = x.std()
                x = x - mean
                x = x / std
                batch.append(x)
        outputs, output_sizes = self._get_batch_spectrogram(batch)
        return outputs, output_sizes

    def _compute_seq_lengths(self, input_lens):
        seq_lens = 1 + (np.array(input_lens) - self.win_length + self.win_length // 2 * 2) // self.hop_length
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        return seq_lens 

    def _forward_np(self, inputs, input_lens = None):
        if input_lens is None:
            input_lens = [inputs.shape[0]] * inputs.shape[1]
        seq_lens = self._compute_seq_lengths(input_lens)
        x = inputs / 32767 # normalize audio
        x = self.spectrogram(x)
        x = torch.sqrt(x)
        x = torch.log1p(x)
        if self.normalize:
            avg   = torch.zeros_like(x, requires_grad=False)#(size=x.shape, dtype=np.float32, requires_grad=False)
            scale = torch.ones_like(x, requires_grad=False)
            for i in range(x.size(0)):
                mean = x[i, :, 0:seq_lens[i]].mean()
                std  = x[i,:, 0:seq_lens[i]].std() 
                avg[i, :, 0:seq_lens[i]]  =  mean
                scale[i,:, 0:seq_lens[i]] *= std
            x = x - avg
            x = x / scale

        # add a dimension
        x = x[:, None, :, :]
        return x, seq_lens 

    def forward(self, inputs, input_lens = None):
        """
        Return spectrograms for the inputs

        Parameters
        ----------
        inputs: list or np.ndarray
            List of audio arrays
        input_lens: List[int]
            Lengths of the input audios

        Returns
        -------
        Tensor:
            Spectrogram of size (..., freq, time), where freq is ``self.n_fft // 2 + 1``, where ``self.n_fft`` is number of Fourier bins, and time is the number of window hops 
        """

        if isinstance(inputs, (np.ndarray, torch.Tensor)):
            return self._forward_np(inputs, input_lens)
        elif isinstance(inputs, list):
            return self._forward_list(inputs)
        else:
            raise ValueError('Input type {} is not supported'.format(type(inputs)))
