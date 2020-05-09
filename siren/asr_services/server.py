import sys
sys.path.insert(0, '/workspace/siren_v2')

import argparse
import logging
import os
import pickle
import numpy as np
from scipy.io import wavfile as wav
from tempfile import NamedTemporaryFile
from util.opts import parse_args_and_load_module
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/transcribe', methods=['GET', 'POST'])
def transcribe_file():
    if request.method == 'GET' or request.method == 'POST': 
        res = {}
        if 'file' not in request.files:
            res['status'] = 'error'
            res['message'] = 'audio file shoule be provided'
            return jsonify(res)
        file_ = request.files['file']
        filename = file_.filename
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() not in set(['.wav', '.mp3', '.ogg', '.webm']):
            res['status'] = 'error'
            res['message'] = '{} is not supported format'.format(file_extension)
            return jsonify(res)
        with NamedTemporaryFile(suffix=file_extension) as tmp_saved_audio_file:
            file_.save(tmp_saved_audio_file.name)
            sample_rate, audio = wav.read(tmp_saved_audio_file) 
            trans, _ = fmodel.forward(audio)
            res['status'] = "OK"
            res["transcription"] = trans
            return jsonify(res)

@app.route('/forward', methods=['GET', 'POST'])
def forward():
    res = {}
    if request.data is None:
        res['status'] = 'error'
        res['message'] = 'data shoule be provided'
        return pickle.dumps(res, protocol=2)
    data = pickle.loads(request.data)
    if isinstance(data, dict):
        inputs = data.get('inputs', None)
        input_lens = data.get('input_lens', None)
    elif isinstance(data, (list, np.ndarray)):
        inputs = data
        input_lens = None
    else:
        res["status"] = "error"
        res["message"] = "Data format not supported"
        return pickle.dumps(res, protocol=2)
    trans, loss = fmodel.forward(inputs, input_lens)
    res = {'satus': "OK", 'transcription':trans, 'loss':loss}
    return pickle.dumps(res, protocol=2)

@app.route('/forward_and_gradient', methods=['GET', 'POST'])
def forward_and_gradient():
    res = {}
    if request.data is None:
        res['status'] = 'error'
        res['message'] = 'data shoule be provided'
        return pickle.dumps(res, protocol=2)
    data = pickle.loads(request.data)
    if isinstance(data, dict):
        inputs = data.get('inputs', None)
        input_lens = data.get('input_lens', None)
    elif isinstance(data, (list, np.ndarray)):
        inputs = data
        input_lens = None
    else:
        res["status"] = "error"
        res["message"] = "Data format not supported"
        return pickle.dumps(res, protocol=2)
    trans, loss, grad = fmodel.forward_and_gradient(inputs, input_lens)
    res = {'satus': "OK", 'transcription':trans, 'loss':loss, 'gradient':grad}
    return pickle.dumps(res, protocol=2)

@app.route('/set_loss', methods=['POST'])
def set_loss():
    res = {}
    if request.data is None:
        res['status'] = 'error'
        res['message'] = 'data shoule be provided'
        return pickle.dumps(res, protocol=2)
    data = pickle.loads(request.data)
    if isinstance(data, (np.ndarray, list)):
        target_phrases = data
    else:
        res["status"] = "error"
        res["message"] = "Data format not supported"
        return pickle.dumps(res, protocol=2)
    fmodel.set_loss(target_phrases)
    res = {'satus': "OK"}
    return pickle.dumps(res, protocol=2)

def main():
    global fmodel

    parser = argparse.ArgumentParser(description="ASR model server")
    parser.add_argument('--host', type=str, 
                        default='0.0.0.0', 
                        help='Host to be used by the server')
    parser.add_argument('--port', type=int, 
                        default=8888, 
                        help='Port to be used by the server')
    parser.add_argument('--asr_name', '-asr', type=str, 
                        default='DeepSpeechTF',
                        choices=['DeepSpeechTF', 'DeepSpeechPT', 'Jasper'],
                        help='The name of the ASR')

    # extract parameters 
    args = parser.parse_known_args()[0]

    fmodel = parse_args_and_load_module(configuration_file= '/workspace/siren_v2/configs/model_config.yaml', key=args.asr_name, description='ASR (automatic speech recognition) model parameters')

    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
