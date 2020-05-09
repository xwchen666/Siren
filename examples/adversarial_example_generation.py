import sys
sys.path.insert(0, '/workspace/siren_v3/siren')

import os
import numpy as np
import pandas as pd
import argparse
import scipy.io.wavfile as wav
import numpy as np
import json
import importlib
from ruamel.yaml import YAML

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
    limit: float
        The max value of the audios
    """
    audio_nps = [None] * len(audio_list)
    for i in range(len(audio_list)):
        audio_nps[i] = read_audio(audio_list[i])
    return audio_nps

def process_audios(audios):
    input_lens = np.array([len(audio) for audio in audios])
    num_audios = len(audios)
    max_len    = np.max(input_lens)
    padded_audios = np.zeros(shape=(num_audios, max_len))
    for i in range(len(audios)):
        audio = audios[i]
        padded_audios[i, 0:len(audio)] = audio
    return padded_audios
 
def main():
    yaml = YAML(typ='safe')
    with open('/workspace/siren_v2/configs/augment_config.yaml') as f:
        augment_arguments = yaml.load(f)
        augment_method_collections = list(augment_arguments.keys())

    with open('/workspace/siren_v2/configs/model_config.yaml') as f:
        model_arguments = yaml.load(f)
        model_collections = list(model_arguments.keys())

    parser = argparse.ArgumentParser()
    ex_group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument('-in', '--input_file', action='store', 
                        type=str, required=True,
                        help='Single audio filename (.wav) or the file which contains the list of audios we need to transcribe, their original transcript, and the target transcripts (.csv)')
    ex_group.add_argument('-range', action='store', type=int,
                        nargs='+',
                        default=[0,None],
                        help='The index range of the audio files we want to generate adv examples')
    ex_group.add_argument('-tgt', '--target', action='store', type=str,
                        nargs='+',
                        default=[''],
                        help='Target phrase')
    parser.add_argument('--attack', action='store',
                        type=str, default='CarliniWagnerAttack',
                        choices=['CarliniWagnerAttack', 'YaoCarliniAttack'],
                        help='Attack methods')            
    parser.add_argument('-cfg', '--config_file', action='store', 
                        type=str, required=True,
                        help='Configuration files for models and augmentation methods') 
    parser.add_argument('--distance', action='store', 
                        type=str, default=None,
                        choices=['MSE', 'MAE', 'Linf', 'L0', 'MTD'],
                        help='Distance measure between original audio and the manipulated audio')
    parser.add_argument('--store_dir', action='store',
                        type=str, default='adv_examples',
                        help='The dir to store the adv examples')

    # extract parameters 
    args = parser.parse_known_args()[0]

    with open(args.config_file) as f:
        configs = yaml.load(f)

    # step 1: load models
    print('Load Models...')
    models = []
    for params in configs['models']:
        model_name, model_params = list(params.items())[0]
        module_name = model_arguments[model_name]['module_name']
        class_name  = model_arguments[model_name]['class_name']
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        fmodel = class_(**model_params)
        models.append(fmodel)

    # step 2: load augmentation methods
    print('Load Augmentation Methods...')
    augments = [] 
    for params in configs['augments']:
        if params is None:
            augments.append(None)
            continue
        aug_name, aug_params = list(params.items())[0]
        module_name = augment_arguments[aug_name]['module_name']
        class_name  = augment_arguments[aug_name]['class_name']
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        augment = class_(**aug_params)
        augments.append(augment)

    # step 3: load all audios 
    print('Load Audios...')
    input_file    = args.input_file
    suffix = input_file.split('.')[-1]
    if suffix == 'wav':
        all_audios = [read_audio(input_file)]
        target_sequences = [' '.join(args.target)]
        ori_audio_names = [os.path.basename(input_file)]
    elif suffix == 'csv':
        # load data (list of audio filenames, original transcript, adv transcript)
        idx_range     = args.range
        if len(idx_range) == 1:
            idx_range.append(None) 
        data = pd.read_csv(input_file)[idx_range[0]:idx_range[1]]
        audio_list        = list(data['audio_name'])
        all_audios        = get_all_audios(audio_list)
        target_sequences  = list(data['target'])
        ori_audio_names   = [os.path.basename(audio_name) for audio_name in audio_list]
    else:
        raise Valueerror('Input file format not supported')

    # step 4: set up the criterion    
    print('Load Criterion...')
    from util.criteria import TargetSequence
    criterion = TargetSequence()

    # step 5: load the instance of distance
    print('Load Distance...')
    module = importlib.import_module('util.distance')
    class_ = getattr(module, args.distance)
    if args.distance != 'MTD':
        distance = class_()
    else:
        padded_audios = process_audios(all_audios)
        distance = class_(padded_audios)


    # step 6: load the attack metric

    print('Load Metric...')
    module_name_dict = {'CarliniWagnerAttack':'attacks.carlini_wagner', 'YaoCarliniAttack':'attacks.yao_carlini'} 
    module = importlib.import_module(module_name_dict[args.attack])

    class_ = getattr(module, args.attack)
    attack = class_(models=models, augments=augments, criterion=criterion, distance=distance)

    # Start attack
    print('Start Attack')
    attack(unperturbed=all_audios, target_phrases=target_sequences, store_dir=args.store_dir,
                ori_audio_names=ori_audio_names)

if __name__ == '__main__':
    main()
