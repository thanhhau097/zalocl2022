import requests
import base64
import pandas as pd
import glob
import os
import json
from tqdm import tqdm


SPEECH_URL = "https://jp-asr-api.namisense.ai/transcript"
SPEECH_PARAMS = {
    "audio": "",
    "lang": "vi",
    "dict": []
}
DIR_FILE = '/home/anhph/youtube/chunks/'
DIR_SAVE = '/home/anhph/youtube/labels/'

files = glob.glob(DIR_FILE + '*.wav')
for file in tqdm(files):
    enc=base64.b64encode(open(file, "rb").read()).decode('utf-8')
    SPEECH_PARAMS["audio"] = enc
    try:
        r = requests.post(url = SPEECH_URL, json = SPEECH_PARAMS)
        our_data = json.loads(r.content.decode('utf-8'))['words']
        new_data = [{'s':0 , 'e':int(our_data[-1]['end']*1000), 'l': []}]
        for i in range(len(our_data)):
            new_data[0]['l'].append({'s': int(our_data[i]['start']*1000) , 'e': int(our_data[i]['end']*1000), 'd': our_data[i]['word']})
        json.dump(new_data, open(os.path.join(DIR_SAVE, os.path.basename(file)[:-3] + 'json'), 'w'))
    except:
        print(os.path.join(DIR_FILE, file))
        print(r.content.decode('utf-8'))