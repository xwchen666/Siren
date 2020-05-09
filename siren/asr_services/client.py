import pickle
import requests

from asr_collections.base import Model

class WebModel(Model):
    """
    A wrapper for ASR web service

    Parameters
    ----------
    host: str
        IP addreess of the host server of the ASR service
    port: int
        Port of the ASR service
    """
    def __init__(self, host:str, port: int):
        self.url = "http://{}:{}/".format(host, port) 

    def request(self, name, data):
        r = requests.post(self.url + name, data=data)
        res = pickle.loads(r.content)
        return res

    def set_loss(self, target_phrases):
        data = pickle.dumps(target_phrases, protocol=2)
        self.request('set_loss', data)

    def forward(self, inputs, input_lens):
        data = pickle.dumps({'inputs': inputs, 'input_lens': input_lens}, protocol=2)
        res = self.request('forward', data)
        return res['transcription'], res['loss']

    def forward_and_gradient(self, inputs, input_lens):
        data = pickle.dumps({'inputs': inputs, 'input_lens': input_lens}, protocol=2)
        res = self.request('forward_and_gradient', data)
        return res['transcription'], res['loss'], res['gradient'] 
