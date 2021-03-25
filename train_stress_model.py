from keras import backend as K
from keras.layers import Dense, Activation, Reshape, Permute, Lambda, Multiply, BatchNormalization, Bidirectional, RepeatVector
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, TimeDistributed, Embedding, AveragePooling2D, Conv1D, MaxPooling1D
from keras.models import Model, Sequential
from keras.layers.merge import add, concatenate
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session, clear_session
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras_layer_normalization import LayerNormalization
from multi_head_attention import MultiHeadAttention
from flip_gradient_tf import *
import keras
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import datetime
import os
import h5py


def build_Model(input_shape, n_states=2, n_speaker=40):
    '''
    architecture:
        1. 4 layers CNN (dilated convolution)
        2. multi head attention(head num: 2)
        3. FC
    '''
    # Input layer
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # Convolution layer (VGG)
    inner = Conv2D(32, (3, 3), padding='same', name='conv1', dilation_rate=2, kernel_initializer='he_normal')(inputs)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

    inner = Conv2D(64, (3, 3), padding='same', name='conv2', dilation_rate=2, kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', dilation_rate=2, kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', dilation_rate=2, kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    # CNN reshape
    inner = Reshape(target_shape=((125, 2560)), name='reshape')(inner)
    inner = Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)

    # Multi-head attention layer
    inner = MultiHeadAttention(head_num=2,name='Multi-Head')(inner)
    inner = Lambda(lambda xin: K.sum(xin, axis=1))(inner)

    # Gradient Reversal Layer
    Flip = GradientReversal(hp_lambda=0.31)
    dann_in = Flip(inner)
    dann_out = Dense(units=n_speaker, activation='softmax', name='gradient_reversal')(dann_in)

    # transforms RNN output to character activations:
    predictions = Dense(units=n_states, activation='softmax',name='output_layer')(inner) # (None, 3)

    model = Model(inputs=inputs, outputs=[predictions, dann_out])
    adam = optimizers.Adam(lr=0.00001)
    model.compile(optimizer=adam, loss={'output_layer':'categorical_crossentropy', 'gradient_reversal':'categorical_crossentropy'}, loss_weights={'output_layer':0.997, 'gradient_reversal':0.003}, metrics=['accuracy'])
    model.summary()
    return model


def load_data(feature_path, label_path, speaker_label_path):
    feature = np.load(feature_path, allow_pickle=True)
    label = np.load(label_path, allow_pickle=True)
    speaker_label = np.load(speaker_label_path, allow_pickle=True)
    return feature, label, speaker_label


def get_parser():
    '''
    epoch : Default is 300
    batchsize : Default is 32
    cropsize : Default is 500
    '''
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', '-e', type=int, default=300)
    parser.add_argument('--batchsize', '-bs', type=int, default=32)
    parser.add_argument('--cropsize', type=int, default=500)

    return parser


if __name__ == '__main__':

    #=============Setting=============
    # GPU Setting  : 30%
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config = tf.ConfigProto(device_count = {'GPU': 1})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.Session(config=config)
    set_session(session)
    early_stopping = EarlyStopping(monitor='val_loss', patience=150)

    # Set number of speakers in train set and number of states (2 or 3)
    n_states = 2
    n_speaker = 49

    #=============Load Data=============
    if n_states == 2:

        feature_path = './features/features_2states_cropsize_500_Num_Frame_1500_log_mel_feature.npy'
        label_path = './features/features_2states_cropsize_500_Num_Frame_1500_log_mel_label.npy'
        speaker_label_path = './features/features_2states_cropsize_500_Num_Frame_1500_log_mel_speaker_label.npy'

        test_feature_path = './features/features_2states_cropsize_500_Num_Frame_1500_log_mel_feature.npy'
        test_label_path = './features/features_2states_cropsize_500_Num_Frame_1500_log_mel_label.npy'
        test_speaker_label_path = './features/features_2states_cropsize_500_Num_Frame_1500_log_mel_speaker_label.npy'
    else:
        feature_path = './features/features_3states_cropsize_500_Num_Frame_1500_log_mel_feature.npy'
        label_path = './features/features_3states_cropsize_500_Num_Frame_1500_log_mel_label.npy'
        speaker_label_path = './features/features_3states_cropsize_500_Num_Frame_1500_log_mel_speaker_label.npy'

        test_feature_path = './features/features_3states_cropsize_500_Num_Frame_1500_log_mel_feature.npy'
        test_label_path = './features/features_3states_cropsize_500_Num_Frame_1500_log_mel_label.npy'
        test_speaker_label_path = './features/features_3states_cropsize_500_Num_Frame_1500_log_mel_speaker_label.npy'

    ## TRAIN data ##
    x, y, speaker_y = load_data(feature_path, label_path, speaker_label_path)

    print('x.shape = ', x.shape)
    print('y.shape = ', y.shape)
    print('speaker_y = ', speaker_y.shape)

    x_data = np.asarray(x)
    y_data = np.asarray(y)
    speaker_y_data = np.asarray(speaker_y)

    del x, y, speaker_y
    y_data = to_categorical(y_data, n_states)
    speaker_y_data = to_categorical(speaker_y_data, n_speaker)
    y_total_data = np.asarray([np.asarray([x, y]) for x, y in zip(y_data, speaker_y_data)])

    print(np.shape(x_data), np.shape(y_data))
    print('len(x_data) = ', len(x_data))
    print('len(y_data) = ', len(y_data))
    print('x_data.shape = ', x_data.shape)
    print('y_data.shape = ', y_data.shape)

    ## VAL data ##
    x_data, x_valid, y_data, y_valid = train_test_split(x_data, y_total_data, test_size=0.2, shuffle= True)

    ## TEST data ##
    x_test, y_test, speaker_y_test = load_data(test_feature_path, test_label_path, test_speaker_label_path)
    x_test_data = np.asarray(x_test)
    y_test_data = np.asarray(y_test)
    speaker_y_test_data = np.asarray(speaker_y_test)

    y_test_data = to_categorical(y_test_data, n_states)
    speaker_y_test_data = to_categorical(speaker_y_test_data, n_speaker)

    print(np.shape(x_test_data), np.shape(y_test_data), np.shape(speaker_y_test_data))
    print('len(x_test_data) = ', len(x_test_data))
    print('len(y_test_data) = ', len(y_test_data))
    print('x_test_data.shape = ', x_test_data.shape)
    print('y_test_data.shape = ', y_test_data.shape)

    now = datetime.datetime.now()
    Today = now.strftime('%Y%m%d')

    ## Parameter
    parser = get_parser()
    args = parser.parse_args()
    epochs = args.epoch
    batch_size = args.batchsize
    input_shape = (args.cropsize, 40, 1)

    print('input_shape = ', input_shape)

    ## Load Model
    if args.cropsize == 500:
        model = build_Model(input_shape, n_states)

    ## Training
    '''
    - acc: 매 epoch 마다의 훈련 정확도
    - loss: 매 epoch 마다의 훈련 손실 값
    - val_acc: 매 epoch 마다의 검증 정확도
    - val_loss: 매 epoch 마다의 검증 손실 값
    '''
    # Start Training
    history = model.fit(x_data, {'output_layer': np.stack(y_data[:,0]), 'gradient_reversal': np.stack(y_data[:,1])}, batch_size=batch_size, epochs=epochs, shuffle=True,
                        validation_data=(x_valid, {'output_layer': np.stack(y_valid[:,0]), 'gradient_reversal': np.stack(y_valid[:,1])}), verbose = 1, callbacks=[early_stopping])

    # Save model
    model.save('./model/' + str(Today) + '_{}states_'.format(n_states)+ 'epoch_' + str(epochs) + '_batchsize_' + str(batch_size) + '_' + os.path.splitext(os.path.split(feature_path)[-1])[0].replace('_features','').replace('_feature','') + '.h5')
    # Evaluation
    score = model.evaluate(x_test_data, {'output_layer': y_test_data, 'gradient_reversal': speaker_y_test_data}, batch_size=batch_size)
    print('loss, acc : ', score)


    ## Plotting learning curve
    try:
        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        # loss graph
        loss_ax.plot(history.history['loss'], color='#F15F5F', label='train loss')
        loss_ax.plot(history.history['output_layer_loss'], color='#F29661', label='train stress loss')
        loss_ax.plot(history.history['gradient_reversal_loss'], color='#F2CB61', label='train speaker loss')
        loss_ax.plot(history.history['val_loss'], color='#BCE55C', label='val loss')
        loss_ax.plot(history.history['val_output_layer_loss'], color='#86E57F', label='val stress loss')
        loss_ax.plot(history.history['val_gradient_reversal_loss'], color='#5CD1E5', label='val speaker loss')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')

        # accuracy graph
        acc_ax.plot(history.history['output_layer_acc'], color='#6B66FF', label='train stress acc')
        acc_ax.plot(history.history['gradient_reversal_acc'], color='#A566FF', label='train speaker acc')
        acc_ax.plot(history.history['val_output_layer_acc'], color='#F361DC', label='val stress acc')
        acc_ax.plot(history.history['val_gradient_reversal_acc'], color='#F361A6', label='val speaker acc')
        acc_ax.set_ylabel('accuracy')
        acc_ax.set_ylim([0, 1.1])
        fig.legend(loc='upper left')


        plt.savefig('./model/' + str(Today) + '_{}states_'.format(n_states) + os.path.splitext(os.path.split(feature_path)[-1])[0].replace('_features','').replace('_feature','') + '.png')

    except Exception as e:
        print('Didn`t save graph!')
        print(e)
        pass

    loss_acc_lst = []
    loss_acc_lst.append(history.history['loss'])
    loss_acc_lst.append(history.history['output_layer_loss'])
    loss_acc_lst.append(history.history['gradient_reversal_loss'])
    loss_acc_lst.append(history.history['val_loss'])
    loss_acc_lst.append(history.history['val_output_layer_loss'])
    loss_acc_lst.append(history.history['val_gradient_reversal_loss'])
    loss_acc_lst.append(history.history['output_layer_acc'])
    loss_acc_lst.append(history.history['gradient_reversal_acc'])
    loss_acc_lst.append(history.history['val_output_layer_acc'])
    loss_acc_lst.append(history.history['val_gradient_reversal_acc'])
    loss_acc_lst.append(score)
    numpy_history = np.array(loss_acc_lst)

    np.savetxt('./model/' + str(Today) + '_{}states_'.format(n_states) + 'epoch_' + str(epochs) + '_batchsize_' + str(batch_size) + '_' + os.path.splitext(os.path.split(feature_path)[-1])[0].replace('_features','').replace('_feature','') + ".txt", numpy_history, delimiter=",", fmt='%s')

    clear_session()