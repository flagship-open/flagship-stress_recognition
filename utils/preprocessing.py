import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import librosa
import math
import scipy
from scipy import io
from pyvad import trim
from collections import OrderedDict


def preprocessing(y, sr, num_frame=1500, stride=0.01, window_size=0.025, \
                  num_mels=40, pre_emphasis=0.97, is_mel=False):

    # Resampling to 16kHz
    if sr != 16000:
        sr_re = 16000  # sampling rate of resampling
        y = librosa.resample(y, sr, sr_re)
        sr = sr_re

    # Denoising
    y[np.argwhere(y == 0)] = 1e-10
    y_denoise = scipy.signal.wiener(y, mysize=None, noise=None)

    # Pre Emphasis filter
    y_emphasis = np.append(y_denoise[0], \
                           y_denoise[1:] - pre_emphasis * y_denoise[:-1])

    # Normalization (Peak)
    y_max = max(y_emphasis)
    y_emphasis = y_emphasis / y_max

    # Voice Activity Detection (VAD)
    vad_mode = 2  # VAD mode = 0 ~ 3
    y_vad = trim(y_emphasis, sr, vad_mode=vad_mode)
    if y_vad is None:
        y_vad = y_emphasis

    # De normalization
    y_vad = y_vad * y_max

    # Obtain the mel spectrogram
    S = librosa.feature.melspectrogram(y=y_vad, sr=sr, \
                                       hop_length=int(sr * stride), \
                                       n_fft=int(sr * window_size), \
                                       n_mels=num_mels, power=2.0)
    # Mel or Log Mel
    EPS = 1e-8
    S = np.log(S + EPS)
    r, frame_length = S.shape
    print('\n** log mel **')
    print('S.shape', S.shape)

    # Obtain the normalized mel spectrogram
    s_norm = (S - np.mean(S)) / np.std(S)

    # zero padding
    input_mels = np.zeros((r, num_frame), dtype=float)
    if frame_length < num_frame:
        input_mels[:, :frame_length] = s_norm[:, :frame_length]
    else:
        input_mels[:, :num_frame] = s_norm[:, :num_frame]

    return input_mels, frame_length


def crop_mels(input_mels_origin, each_frame_num, crop_size, max_frame_num):
    input_mels_origin = input_mels_origin.T

    # Calculate the number of cropped mel-spectrogram
    if each_frame_num > max_frame_num:
        number_of_crop = math.floor(max_frame_num/int(crop_size/2)-1)
    else:
        if each_frame_num < crop_size:
            number_of_crop = 1
        else:
            number_of_crop = int(round(each_frame_num/int(crop_size/2))) - 1

    # Crop
    crop_num_frame = crop_size    # Frame size of crop
    cropped_mels = np.zeros((number_of_crop, crop_num_frame, \
                             input_mels_origin.shape[1]))
    crop_num = 0
    if each_frame_num > max_frame_num:
        each_crop_num = math.floor(max_frame_num/int(crop_size/2)-1)
        print('each_crop_num = ', each_crop_num)
        for n_crop in range(0, each_crop_num):
            cropped_mels[crop_num, :, :] = input_mels_origin \
                                           [n_crop * int(crop_size/2): \
                                            n_crop * int(crop_size/2) + crop_size, :]
            crop_num += 1
    else:
        if each_frame_num < crop_size:    # If the frame number is lower than 200, the number of crop is 1
            cropped_mels[crop_num, :, :] = input_mels_origin[:crop_size, :]
            crop_num += 1
        else:
            each_crop_num = int(round(each_frame_num / int(crop_size/2))) - 1    # Calculate the number of crop
            if round(each_frame_num / int(crop_size/2)) < each_frame_num / int(crop_size/2):
                for n_crop in range(0, each_crop_num):
                    cropped_mels[crop_num, :, :] = input_mels_origin\
                                                   [n_crop * int(crop_size/2):\
                                                    n_crop * int(crop_size/2) + crop_size, :]
                    crop_num += 1
            else:
                for n_crop in range(0, each_crop_num - 1):
                    cropped_mels[crop_num, :, :] = input_mels_origin\
                                                   [n_crop * int(crop_size/2):\
                                                    n_crop * int(crop_size/2) + crop_size, :]
                    crop_num += 1
                shift_frame = int((each_frame_num / int(crop_size/2) - \
                                   round(each_frame_num / int(crop_size/2))) * int(crop_size/2))
                cropped_mels[crop_num, :, :] = input_mels_origin\
                                               [(each_crop_num - 1) * int(crop_size/2) + shift_frame:\
                                                (each_crop_num - 1) * int(crop_size/2) + shift_frame + crop_size,:]
                crop_num += 1
    return cropped_mels, number_of_crop


def generate(filename, modelname, graph, crop_size, n_class=3):
    y, sr = librosa.load(filename)
    num_frame = 1500

    # Preprocessing(Resampling, Normalization, Denoising, Pre-emphasis, VAD)
    input_mels, frame_length = preprocessing(y, sr, num_frame=num_frame, \
                                             is_mel=0)

    # Crop mel-spectrogram
    cropped_mels, number_of_crop = crop_mels(input_mels, frame_length, \
                                             crop_size, max_frame_num = num_frame)
    cropped_mels = np.reshape(cropped_mels, \
                              (cropped_mels.shape[0], \
                               cropped_mels.shape[1], \
                               cropped_mels.shape[2], \
                               1)
                             )

    # Predict (cropped version)
    with graph.as_default():
        y_stress_pred = np.mean(modelname.predict(cropped_mels)[0], axis=0)
    # Ready for data(json data)
    result = OrderedDict()

    if n_class == 2:
    # stress (binary)
    # neutral: 10001, stressed: 10002
        result["10001"] = round(float(y_stress_pred[0]),4)
        result["10002"] = round(float(y_stress_pred[1]),4)
    else:
    # stress (3 cases)
    # neutral: 10001, weakly stressed: 10002, strongly stressed: 10003
        result["10001"] = round(float(y_stress_pred[0]),4)
        result["10002"] = round(float(y_stress_pred[1]),4)
        result["10003"] = round(float(y_stress_pred[2]),4)

    return result
