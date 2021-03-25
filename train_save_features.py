# -*- coding: utf-8 -*-
'''
Data Load & Feature extraction

####################################################################################################
Process)

    1. data load
    2. feature extraction, cropping
    3. save features

####################################################################################################
For run the code)

    데이터 경로 저장된 파일 설정: ** 1 **
    feature 저장할 경로 설정: ** 2 ** : features폴더가 생성되고 그 하위에 feature파일이 저장되게 됨.
    ** 1 **,** 2 ** 후에 Run the code.

    Run the code) python3 data_load_2_states.py                 #  --> for creating training features
                  python3 data_load_2_states.py --istest 1      #  --> for creating testing features

#####################################################################################################
'''
import os
import time
from tqdm import tqdm
import datetime
import numpy as np
import pandas as pd
import math
import librosa
import json
import scipy
from scipy import io
import argparse
from pyvad import trim

### Pre-processing
Num_Frame = 1500    # max wave length (15 sec)
Stride = 0.01       # stride (10ms)
Window_size = 0.025 # filter window size (25ms)
Num_data = 1
Num_mels = 40       # Mel filter number
pre_emphasis = 0.97  # Pre-Emphasis filter coefficient


def preprocessing(y, sr, is_mel=True):

    # Resampling to 16kHz
    if sr != 16000:
        sr_re = 16000  # sampling rate of resampling
        y = librosa.resample(y, sr, sr_re)
        sr = sr_re

    # Denoising
    y[np.argwhere(y == 0)] = 1e-10
    y_denoise = scipy.signal.wiener(y, mysize=None, noise=None)

    # Pre Emphasis filter
    y_Emphasis = np.append(y_denoise[0], y_denoise[1:] - pre_emphasis * y_denoise[:-1])

    # Normalization (Peak)
    y_max = max(y_Emphasis)
    y_Emphasis = y_Emphasis / y_max

    # Voice Activity Detection (VAD)
    vad_mode = 2  # VAD mode = 0 ~ 3
    y_vad = trim(y_Emphasis, sr, vad_mode=vad_mode, thr=0.01)
    if y_vad is None:
        y_vad = y_Emphasis

    # De normalization
    y_vad = y_vad * y_max

    y_vad = y_Emphasis
    # Obtain the mel spectrogram or log mel spectrogram
    S = librosa.feature.melspectrogram(y=y_vad, sr=sr, hop_length=int(sr * Stride), n_fft=int(sr * Window_size), n_mels=Num_mels, power=2.0)
    if is_mel == False:
        EPS = 1e-8
        S = np.log(S + EPS)
        r, Frame_length = S.shape
        print('\n** log mel **')
        print('S.shape', S.shape)
    else:
        r, Frame_length = S.shape
        print('\n** mel **')
        print('S.shape', S.shape)

    # Obtain the normalized mel spectrogram
    S_norm = (S - np.mean(S)) / np.std(S)

    # zero padding
    Input_Mels = np.zeros((r, Num_Frame), dtype=float)
    if Frame_length < Num_Frame:
        Input_Mels[:, :Frame_length] = S_norm[:, :Frame_length]
    else:
        Input_Mels[:, :Num_Frame] = S_norm[:, :Num_Frame]

    return Input_Mels, Frame_length


def Crop_Mels(Input_Mels_origin, Each_Frame_Num, Crop_Size, Max_Frame_Num):
    Input_Mels_origin = Input_Mels_origin.T

    # Calculate the number of cropped mel-spectrogram
    if Each_Frame_Num > Max_Frame_Num:
        Number_of_Crop = math.floor(Max_Frame_Num/int(Crop_Size/2)-1)
    else:
        if Each_Frame_Num < Crop_Size:
            Number_of_Crop = 1
        else:
            Number_of_Crop = int(round(Each_Frame_Num/int(Crop_Size/2))) - 1

    ## Crop
    Crop_Num_Frame = Crop_Size    # Frame size of crop
    Cropped_Mels = np.zeros((Number_of_Crop,Crop_Num_Frame,Input_Mels_origin.shape[1]))
    crop_num = 0  # Crop된 data의 number
    if Each_Frame_Num > Max_Frame_Num:
        Each_Crop_Num = math.floor(Max_Frame_Num/int(Crop_Size/2)-1)
        print('Each_Crop_Num = ', Each_Crop_Num)
        for N_crop in range(0, Each_Crop_Num):
            Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * int(Crop_Size/2):N_crop * int(Crop_Size/2) + Crop_Size, :]
            crop_num += 1
    else:
        if Each_Frame_Num < Crop_Size:    # If the frame number is lower than 200, the number of crop is 1
            Cropped_Mels[crop_num, :, :] = Input_Mels_origin[:Crop_Size, :]
            crop_num += 1
        else:
            Each_Crop_Num = int(round(Each_Frame_Num / int(Crop_Size/2))) - 1    # Calculate the number of crop
            if round(Each_Frame_Num / int(Crop_Size/2)) < Each_Frame_Num / int(Crop_Size/2):
                for N_crop in range(0, Each_Crop_Num):
                    Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * int(Crop_Size/2):N_crop * int(Crop_Size/2) + Crop_Size, :]
                    crop_num += 1
            else:
                for N_crop in range(0, Each_Crop_Num - 1):
                    Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * int(Crop_Size/2):N_crop * int(Crop_Size/2) + Crop_Size, :]
                    crop_num += 1
                shift_frame = int((Each_Frame_Num / int(Crop_Size/2) - round(Each_Frame_Num / int(Crop_Size/2))) * int(Crop_Size/2))
                Cropped_Mels[crop_num, :, :] = Input_Mels_origin[(Each_Crop_Num - 1) * int(Crop_Size/2) + shift_frame:(Each_Crop_Num - 1) * int(Crop_Size/2) + shift_frame + Crop_Size,:]
                crop_num += 1
    return Cropped_Mels, Number_of_Crop


# Main Code
def generate(y, sr, Crop_Size, is_mel):

    # Preprocessing(Resampling, Normalization, Denoising, Pre-emphasis, VAD)
    Input_Mels, Frame_length = preprocessing(y, sr, is_mel)

    # Crop mel-spectrogram
    Cropped_Mels, Number_of_Crop = Crop_Mels(Input_Mels, Frame_length, Crop_Size, Max_Frame_Num = Num_Frame)
    Cropped_Mels = np.reshape(Cropped_Mels, (Cropped_Mels.shape[0], Cropped_Mels.shape[1], Cropped_Mels.shape[2], 1))

    return Cropped_Mels, Number_of_Crop


def get_parser():
    '''
    n_states : # of states (2: stressed / non-stressed, 3: nonstress / weakly-stressed / strongly-stressed )
    data_folder: folder that contains data
    cropsize : Default is 200, It means 2s slicing
    ismel : Default is 1, It means extracting mel spectrogram, If value is 0, then log mel spectrogram
    istest : Default is 0, It means extracting train data features, If values is 1, then test data features
    '''
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_states', type=int, default=2, choices=[2, 3]) # default : 2
    parser.add_argument('--data_folder', type=str, default='./4th_stress_data') 
    parser.add_argument('--cropsize', type=int, default=500) # default : 5 sec
    parser.add_argument('--ismel', type=int, default=0, choices=[None, 0, 1]) # default: log mel
    parser.add_argument('--istest', type=int, default=0, choices=[None, 0, 1])
    return parser


if __name__ == '__main__':

    now = datetime.datetime.now()
    Today = now.strftime('%Y%m%d')

    parser = get_parser()
    args = parser.parse_args()
    Crop_Size = args.cropsize
    is_mel = args.ismel
    is_test = args.istest
    n_states = args.n_states

    if is_mel == False:
        is_mel_str = '_log_mel'
    else:
        is_mel_str = '_mel'

    if is_test == 0:
        '''
        (trainset) 데이터 경로 저장된 파일 설정: ** 1 **
        '''
        # TRAIN
        file_manage2 = pd.read_excel('{}/data_file_info_chunk_15_trainset.xlsx'.format(data_folder), index=None)

    elif is_test == 1:
        '''
        (testset) 데이터 경로 저장된 파일 설정: ** 1 **
        '''
        # TEST
        file_manage2 = pd.read_excel('{}/data_file_info_chunk_15_testset.xlsx'.format(data_folder), index=None)


    features = []
    labels = []
    speaker_labels = []
    for idx, row in tqdm(file_manage2.iterrows()):
        y, sr = librosa.load(row['filename_2'], sr = 16000)
        feature, Number_of_Crop = generate(y, sr, Crop_Size, is_mel)
        features.extend(feature)
        labels.extend(np.repeat(row['label'], Number_of_Crop))
        speaker_labels.extend(np.repeat(row['speaker_label'], Number_of_Crop))

        print('-----------raw data-------------')
        print('len(features) = ',len(features))
        print('len(labels) = ', len(labels))
        print('len(speaker_labels) = ', len(speaker_labels))

    if len(features) == len(labels):
        print('=====================')
        print('same length! success!')
        print('=====================')

    else:
        print('=====================')
        print('different length! there is some problems.')
        print('len(features) = ',len(features))
        print('len(labels) = ', len(labels))
        print('len(speaker_labels) = ', len(speaker_labels))
        print('=====================')


    if 'features' not in os.listdir():
        os.mkdir('./features')

    if is_test == 0:
        '''
        (trainset) feature 저장할 경로 설정: ** 2 **
        '''
        # TRAIN
        np.save('./features/' + 'features_{}states'.format(n_states) + '_cropsize_' + str(Crop_Size) +'_Num_Frame_'+ str(Num_Frame) + is_mel_str + '_feature.npy', features)
        np.save('./features/' + 'features_{}states'.format(n_states) + '_cropsize_' + str(Crop_Size) +'_Num_Frame_'+ str(Num_Frame) + is_mel_str +'_label.npy', labels)
        np.save('./features/' + 'features_{}states'.format(n_states) + '_cropsize_' + str(Crop_Size) +'_Num_Frame_'+ str(Num_Frame) + is_mel_str +'_speaker_label.npy', speaker_labels)

    elif is_test == 1:
        # TEST
        '''
        (testset) feature 저장할 경로 설정: ** 2 **
        '''
        np.save('./features/' + 'features_{}states'.format(n_states) + '_cropsize_' + str(Crop_Size) +'_Num_Frame_'+ str(Num_Frame) + is_mel_str + '_feature.npy', features)
        np.save('./features/' + 'features_{}states'.format(n_states) + '_cropsize_' + str(Crop_Size) +'_Num_Frame_'+ str(Num_Frame) + is_mel_str +'_label.npy', labels)
        np.save('./features/' + 'features_{}states'.format(n_states) + '_cropsize_' + str(Crop_Size) +'_Num_Frame_'+ str(Num_Frame) + is_mel_str +'_speaker_label.npy', speaker_labels)
