from collections import OrderedDict
import json
import os
import time
from flask import Flask, request, make_response, render_template
import keras
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

# custom package
from models.flip_gradient_tf import *
from models.multi_head_attention import MultiHeadAttention
from utils.preprocessing import *


# For check time
bb = time.time()

# For Flask API
app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto(device_count = {'GPU': 1})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
set_session(session)

n_class = 2
### Load model
if n_class == 2:
    model_stress = load_model('./models/model_stress_2stages.h5', \
                              custom_objects={
                                              'MultiHeadAttention':MultiHeadAttention, \
                                              'GradientReversal':GradientReversal
                                             })
elif n_class == 3:
    model_stress = load_model('./models/model_stress_3stages.h5', \
                              custom_objects={
                                              'MultiHeadAttention':MultiHeadAttention, \
                                              'GradientReversal':GradientReversal
                                             })
else:
    raise ValueError

global graph
graph = tf.get_default_graph()

# For check time2
start_time = time.time()


#@app.route('/upload')
#def load_file():
#   return render_template('./upload.html')


@app.route('/uploader', methods=['POST', 'GET'])
def delivered_json():
    print('request.form : {}'.format(request.form))
    objects = request.form
    filename = objects['file']

    aa = time.time()

    result = generate(filename, model_stress, graph, crop_size, n_class)
    result = json.dumps(result)
    print('prediction takes {}s'.format(time.time() - aa))
    return result


if __name__ == '__main__':
    crop_size = 500
    is_mel = 0

    print('pre-loading takes {}s'.format(time.time() - bb))
    app.run(host=os.environ.get('165.132.56.182', '0.0.0.0'),
            port=8888, threaded=False, debug=True)
