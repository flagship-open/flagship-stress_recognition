from ast import literal_eval
import os
from os.path import splitext, isdir
import pandas as pd
import requests

# Open Session
sess = requests.Session()
sess.trust_env = False
n_class = 3
# URL
ip_address = 'http://165.132.56.182:8888/uploader'


filename = './demo/191129_02_script_kr_2.wav'
req = sess.post(ip_address, data={'file': filename})
if req.status_code == 200:
    print(req,'success')
    print(req.content)
else:
    print('fail')

print(req)
