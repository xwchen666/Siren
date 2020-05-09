""" most of the code taken from: https://github.com/NVIDIA/NeMo/blob/master/examples/asr/jasper_eval.py"""
import argparse
import copy
import os
import pickle
from typing import Optional, List

import torch
import numpy as np
from ruamel.yaml import YAML
from functools import wraps

import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils.helpers import get_cuda_device
from nemo.collections.asr.helpers import post_process_predictions, post_process_transcripts, word_error_rate
from nemo.collections.asr.parts import collections, parsers

logging = nemo.logging

from asr_collections.nemo_asr.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from asr_collections.nemo_asr.losses import CTCLossNM
from asr_collections.base import Model

def load_vocab(vocab_file):
    """
    :param vocab_file: one character per line
    :return: labels: list of character
    """
    labels = []
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            labels.append(line)
    return labels

def input_decorator(call_fn):
    @wraps(call_fn)
    def wrapper(self, inputs, input_lens=None, **kwargs):
        is_single_input = isinstance(inputs, np.ndarray) and inputs.ndim == 1
        # process inputs
        if isinstance(inputs, list):
            input_lens  = np.array([len(audio) for audio in inputs])
        elif isinstance(inputs, np.ndarray):
            if inputs.ndim == 1:
                inputs = inputs[np.newaxis,:]
            if input_lens is None:
                input_lens = np.array([inputs.shape[1]] * inputs.shape[0])
        else:
            raise ValueError('Input type should be list or np.ndarray!')

        padded_inputs = np.zeros(shape=(len(input_lens), max(self.max_audio_len, np.max(input_lens))))
        for i in range(len(inputs)):
            padded_inputs[i][0:len(inputs[i])] = inputs[i]
        inputs = padded_inputs
        
        inputs_tensor     = torch.tensor(inputs, dtype=torch.float32, device=self.device, requires_grad=True)
        input_lens_tensor = torch.tensor(input_lens, device=self.device,)

        try:
            out = call_fn(self, inputs_tensor, input_lens_tensor, **kwargs)
            if isinstance(out, tuple):
                outputs = tuple([o[:, 0:np.max(input_lens)] if isinstance(o, np.ndarray) and o.ndim==2 else o for o in out])
            elif isinstance(out, np.ndarray) and out.ndim==2:
                outputs = out[:, 0:np.max(input_lens)]
                
            if is_single_input:
                outputs =  tuple(o[0] if o else None for o in out)
        except RuntimeError:
            print('Fail to call the function {} in Model {}'.format(call_fn.__name__, self.__class__.__name__))
        return outputs
    return wrapper
        
class JasperModel(Model):
    def __init__(self, model_config: str, 
            ckpt_dir: str,
            target_phrases:Optional[List[str]] = None,
            vocab_file: str=None,
            placement = nemo.core.DeviceType.GPU,
            local_rank: int=None,
            amp_opt_level: str="O1",
            **decoder_params)->None:

        self.placement = placement
        self.device = get_cuda_device(placement)
        self.max_audio_len = 302800 # just a magic number

        yaml = YAML(typ="safe")
        with open(model_config) as f:
            jasper_params = yaml.load(f)
        
        # load vocab files
        if vocab_file:
            vocab = load_vocab(vocab_file)
        elif jasper_params.get('labels', None) is not None:
            vocab = jasper_params['labels']
        else:
            raise ValueError('Vocab file must be provided!')
        self.vocab = vocab
        
        sample_rate = jasper_params['sample_rate']

        # Instantiate Neural Factory with supported backend
        neural_factory = nemo.core.NeuralModuleFactory(
            backend=nemo.core.Backend.PyTorch,
            local_rank=local_rank,
            optimization_level=amp_opt_level,
            placement=placement,
        )

        if local_rank is not None:
            logging.info('Doing ALL GPU')
        
        # jasper asr pipeline:
        # data_preprocessor -> encoder -> decoder -> greedy_decoder
        data_preprocessor = AudioToMelSpectrogramPreprocessor(
                sample_rate=sample_rate, placement=placement, **jasper_params["AudioToMelSpectrogramPreprocessor"]
        )
        jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=jasper_params["AudioToMelSpectrogramPreprocessor"]["features"], **jasper_params["JasperEncoder"]
        )
        jasper_decoder = nemo_asr.JasperDecoderForCTC(
            feat_in=jasper_params["JasperEncoder"]["jasper"][-1]["filters"], num_classes=len(vocab)
        )
        greedy_decoder = nemo_asr.GreedyCTCDecoder()

        # load encoder and decoder checkpoint
        jasper_encoder_ckpt = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if os.path.isfile(os.path.join(ckpt_dir, f)) and 'JasperEncoder' in f]
        jasper_decoder_ckpt = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if os.path.isfile(os.path.join(ckpt_dir, f)) and 'JasperDecoder' in f]
        if len(jasper_encoder_ckpt) == 0:
            raise ValueError('JasperEncoder checkpoint not provided!')
        if len(jasper_decoder_ckpt) == 0:
            raise ValueError('JasperDecoder checkpoint not provided!')

        jasper_encoder.load_state_dict(torch.load(jasper_encoder_ckpt[0]))
        jasper_encoder.eval()
        jasper_decoder.load_state_dict(torch.load(jasper_decoder_ckpt[0]))
        jasper_decoder.eval()
        
        self.data_preprocessor = data_preprocessor
        self.jasper_encoder    = jasper_encoder
        self.jasper_decoder    = jasper_decoder
        self.greedy_decoder    = greedy_decoder

        if target_phrases:
            self.set_loss(target_phrases)
        else:
            self.ctc_loss = None

    def set_loss(self, target_phrases):
        self.ctc_loss = CTCLossNM(num_classes=len(self.vocab))
        parser = parsers.CharParser(self.vocab)
        target_phrases_int = [parser(x) for x in target_phrases]
        target_lens = np.array([len(x) for x in target_phrases_int])
        target_phrases_np = np.zeros(shape=(len(target_phrases), np.max(target_lens)), dtype=np.int)
        for i in range(len(target_phrases)):
            target_phrases_np[i,0:target_lens[i]] = target_phrases_int[i]
        self.target_phrases_tensor = torch.tensor(target_phrases_np, requires_grad=False, device=self.device)        
        self.target_lens_tensor    = torch.tensor(target_lens, requires_grad=False, device=self.device)

    def compute_logits_and_loss(self, inputs, input_lens):
        spec, spec_lens = self.data_preprocessor.forward(input_signal=inputs, length=input_lens)
        encoder_res, encoder_lens = self.jasper_encoder(spec, spec_lens)
        decoder_res = self.jasper_decoder(encoder_res)
        if self.ctc_loss:
            loss = self.ctc_loss(log_probs=decoder_res, targets=self.target_phrases_tensor, input_length=encoder_lens, target_length=self.target_lens_tensor)
        else:
            loss = None
        return decoder_res, loss

    @input_decorator
    def forward(self, inputs, input_lens):
        with torch.no_grad():
            log_probs, loss = self.compute_logits_and_loss(inputs, input_lens)
        preds = self.greedy_decoder(log_probs)
        trans = post_process_predictions(preds[None,:, :], self.vocab)
        return trans, loss.cpu().numpy() if loss else loss

    @input_decorator
    def gradient(self, inputs, input_lens):
        _, loss = self.compute_logits_and_loss(inputs, input_lens)
        loss.backward()
        return inputs.grad.cpu().numpy()

    @input_decorator
    def forward_and_gradient(self, inputs, input_lens):
        log_probs, loss = self.compute_logits_and_loss(inputs, input_lens)
        preds = self.greedy_decoder(log_probs)
        trans = post_process_predictions(preds[None,:, :], self.vocab)
        loss.backward()
        grad = inputs.grad.cpu().numpy()
        return trans, loss.detach().cpu().numpy(), grad 
