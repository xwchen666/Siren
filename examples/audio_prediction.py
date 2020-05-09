import sys
sys.path.insert(0, '/workspace/siren_v2')

import os
import argparse
import scipy.io.wavfile as wav
import numpy as np
import librosa
import json
import pandas as pd
from ruamel.yaml import YAML
from util.opts import parse_args_and_load_module

def read_audio(audio_file):
    """
    audio_file: str
        Filename of the audio
    """
    if audio_file.split(".")[-1] == 'wav':
        _, audio_temp = wav.read(audio_file)
    else:
        raise Exception('Unknown file format')
    
    return  audio_temp

def get_all_audios(audio_list):
    """
    audio_file_list: str
        File which contains the filename of audios we want to load
    """
    audio_nps = [None] * len(audio_list)
    for i in range(len(audio_list)):
        audio_nps[i] = read_audio(audio_list[i])
    return audio_nps

def process_audios(audios, augment):
    input_lens = np.array([len(audio) for audio in audios])
    num_audios = len(audios)
    max_len    = np.max(input_lens)
    padded_audios = np.zeros(shape=(num_audios, max_len))
    for i in range(len(audios)):
        audio = audios[i]
        padded_audios[i, 0:len(audio)] = audio
    if augment:
        padded_audios = augment.forward(padded_audios)
        padded_audios = np.clip(padded_audios, -2**15, 2**15-1).astype(np.int16)
    return padded_audios, input_lens
    
def main():
    yaml = YAML(typ='safe')
    with open('/workspace/siren_v2/configs/augment_config.yaml') as f:
        augment_arguments = yaml.load(f)
        augment_method_collections = list(augment_arguments.keys())

    with open('/workspace/siren_v2/configs/model_config.yaml') as f:
        model_arguments = yaml.load(f)
        model_collections = list(model_arguments.keys())

    parser = argparse.ArgumentParser()

    parser.add_argument('-in', '--input_file', action='store', 
                        type=str, required=True,
                        help='Single audio filename (.wav) or the file which contains the list of \
                                audios we need to transcribe, their original transcript, and the \
                                target transcripts (.csv)')
    parser.add_argument('--batch_size', action='store',
                        type=int, default=10,
                        help='Batch size when do predicion')
    parser.add_argument('-asr', '--asr_name', action='store', 
                        type=str, default='DeepSpeechTF',
                        choices=model_collections,
                        help='The name of ASR model') 
    parser.add_argument('-aug', '--augment_method', action='store',
                        type=str, default=None,
                        choices=augment_method_collections,
                        help='Augmentation method')

    # extract parameters 
    args = parser.parse_known_args()[0]
    input_file   = args.input_file
    asr_name     = args.asr_name
    augment_name = args.augment_method
    
    # step 1: load the instance of the augment method
    if augment_name in augment_method_collections:
        augment = parse_args_and_load_module(configuration_file= '/workspace/siren_v2/configs/augment_config.yaml', key=args.augment_method, description='Parser for augment methods')
    else:
        augment = None

    # step 2: load the instance of the model
    fmodel = parse_args_and_load_module(configuration_file= '/workspace/siren_v2/configs/model_config.yaml', key=args.asr_name, description='ASR (automatic speech recognition) model parameters')

    # step 3: load audio data 
    suffix = input_file.split('.')[-1]
    if suffix == 'wav':
        batch_size = 1
        all_audios    = [read_audio(input_file)]
    elif suffix == 'csv':
        batch_size   = args.batch_size
        data         = pd.read_csv(input_file)
        audio_list   = list(data['audio_name'])
        all_audios   = get_all_audios(audio_list) 
    else:
        raise Valueerror('Input file format not supported')

    # step 4: start prediction
    for i in range(0, len(all_audios), batch_size):
        local_audio, input_lens = process_audios(all_audios[i:i+batch_size], augment)
        trans,_ = fmodel.forward(local_audio, input_lens)
        print('\n'.join(trans))

if __name__ == '__main__':
    main()
