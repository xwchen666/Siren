from __future__ import absolute_import, division, print_function
import numpy as np
import warnings
import torch
from functools import wraps

from warpctc_pytorch import CTCLoss
from asr_collections.base import Model
from asr_collections.deepspeech_pt.model import DeepSpeech
from asr_collections.deepspeech_pt.audio_parser import AudioParser

def input_decorator(call_fn):
    @wraps(call_fn)
    def wrapper(self, inputs, input_lens=None, **kwargs):
        is_single_input = isinstance(inputs, np.ndarray) and inputs.ndim == 1
        # process inputs
        if isinstance(inputs, list): # input is a list
            input_lens  = np.array([len(audio) for audio in inputs])
            inputs = [torch.tensor(audio, dtype=torch.float32, requires_grad=True) for audio in inputs]
            #for i in range(len(inputs)):
            #    inputs[i] = torch.tensor(inputs[i], dtype=torch.float32, requires_grad=True)
        elif isinstance(inputs, np.ndarray):
            if inputs.ndim == 1:
                inputs = inputs[np.newaxis, :]
            inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
            if input_lens is None:
                input_lens = np.array([inputs.shape[1]] * inputs.shape[0])
        else:
            raise ValueError('Input type should be list or np.ndarray!')

        try:
            out = call_fn(self, inputs, input_lens, **kwargs)
            if is_single_input:
                out =  tuple(o[0] if o else None for o in out)
        except RuntimeError:
            print('Fail to call the function {} in Model {}'.format(call_fn.__name__, self.__class__.__name__))

        return out
    return wrapper

class DeepSpeechPTModel(Model):
    def __init__(self, ckpt_path=None, target_phrases=None, device="cpu", **decoder_params):
        # audio parser
        self._audio_parser = AudioParser()
        self._audio_parser.to(device)
        # load model
        model = DeepSpeech.load_model(ckpt_path)
        model.eval()
        self._model = model
        # set decoder
        self._labels = [ "-", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " " ]
        self._decoder_name = decoder_params.get('decoder_name', 'greedy')
        if self._decoder_name == 'greedy':
            from asr_collections.decoder import GreedyDecoder
            self._decoder = GreedyDecoder(alphabet=self._labels, blank_char="-")
        elif self._decoder_name == 'beamsearch_lm':
            lm_path = decoder_params.get('lm_path', None)
            if lm_path is None:
                raise ValueError('Path to LM must be provided for the decoder')
            beam_width = decoder_params.get('beam_width', 10)
            from util.decoder import BeamSearchWithLMDecoder
            self._decoder = BeamSearchWithLMDecoder(alphabet=self._labels, lm_path=lm_path, beam_width=beam_width) 
        else:
            raise ValueError('Decoder name shoud be: greedy or beamsearch_lm')
        # set loss
        if target_phrases:
            self.set_loss(target_phrases)
        else:
            self._criterion = None

    def compute_seq_lengths(self, input_lens):
        input_lens = torch.tensor(input_lens)
        output_lengths = self._model.get_seq_lens(input_lens)
        return output_lengths.numpy()

    def set_loss(self, target_phrases):
        def text2ids(transcripts):
            labels_map = dict([(self._labels[i], i) for i in range(len(self._labels))])
            targets = []
            target_sizes = []
            for transcript in transcripts:
                transcript = list(filter(None, [labels_map.get(x) for x in list(transcript)]))
                targets.extend(transcript)
                target_sizes.append(len(transcript))
            return torch.IntTensor(targets), torch.IntTensor(target_sizes)
        # load criterion
        self._criterion = CTCLoss()
        self._targets, self._target_sizes = text2ids(target_phrases)

    def compute_logits_and_loss(self, inputs, input_lens):
        permutation = np.arange(len(input_lens))
        permutation = [x for _, x in sorted(zip(input_lens, permutation), reverse=True)]
        reverse_permutation = np.arange(len(input_lens))
        for i in range(len(input_lens)):
            reverse_permutation[permutation[i]] = i

        if isinstance(inputs, list):
            sorted_inputs = [inputs[i] for i in permutation]
        else:
            sorted_inputs = inputs[permutation]
        sorted_input_lens = input_lens[permutation]

        spects, spect_lens  = self._audio_parser(sorted_inputs, sorted_input_lens)
        logits, logit_sizes = self._model(spects, spect_lens)

        # permutation back
        logits = logits[reverse_permutation]
        logit_sizes = logit_sizes[reverse_permutation]

        if self._criterion: # the loss is already set 
            loss = self._criterion(logits.transpose(0, 1), self._targets, logit_sizes, self._target_sizes)
        else:
            loss = None
        return logits, logit_sizes, loss

    @input_decorator
    def forward(self, inputs, input_lens=None):
        with torch.no_grad():
            logits, logit_sizes, loss = self.compute_logits_and_loss(inputs, input_lens)
        trans = self._decoder.decode(logits.numpy(), logit_sizes.numpy().astype(int))
        return trans, loss.numpy() if loss else loss

    def _process_grads(self, loss, inputs, input_lens):
        if loss:
            loss.backward()
            if isinstance(inputs, list):
                grads = [audio.grad.numpy() for audio in inputs]
            else:
                grads = inputs.grad
            for i in range(len(grads)):
                grads[i][np.isnan(grads[i])] = 0
                grads[i][input_lens[i]:] = 0
            loss = loss.detach().numpy()
        else:
            grads = None
        return grads

    @input_decorator
    def gradient(self, inputs, input_lens=None):
        _, _, loss = self.compute_logits_and_loss(inputs, input_lens)
        return self._process_grads(loss, inputs, input_lens)

    @input_decorator
    def forward_and_gradient(self, inputs, input_lens=None):
        logits, logit_sizes, loss = self.compute_logits_and_loss(inputs, input_lens)
        trans = self._decoder.decode(logits.detach().numpy(), logit_sizes.detach().numpy().astype(int))
        grads = self._process_grads(loss, inputs, input_lens)
        return trans, loss, grads
