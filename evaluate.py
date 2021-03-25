from ast import literal_eval
import os
from os.path import splitext, isdir
import pandas as pd
import requests
from glob import glob

# Open Session
sess = requests.Session()
sess.trust_env=False
# URL
ip_address = 'http://165.132.56.182:8888/uploader'

file_speaker = pd.read_csv('./static/speaker_index.csv')
path_dir = '/home/flagship/Database/4th_stress_data_final/TEST'

## TEST mode
result = []

file_list = glob(os.path.join(path_dir, '*/*.wav'))

for i in range(len(file_list)):
    filename = file_list[i]
    dic = {'filename': '', 'original_label' : '', \
           'prediction_result' : '', 'speaker_label': '', \
           'wav_accuracy':''}
    
    # Filename
    dic['filename'] = filename
    print('filename = ', filename)
    
    # Original label
    # Neutral
    if 'script' in filename:
        dic['original_label'] = '10001'
    # Stress
    elif 'interview' in filename:
        dic['original_label'] = '10002'
    
    # Request to server for prediction
#    filename = os.path.join(root, filename)
    req = sess.post(ip_address, data={'file': filename})
    if req.status_code == 200:
        print(req,'success')
        print(req.content)
    else:
        print('fail')
    
    # Prediction label
    dic['prediction_result'] = max(literal_eval(req.content.decode("utf-8")), \
                                   key=literal_eval(req.content.decode("utf-8")).get)
    # Speaker label
    speaker_label_info = dict(zip(file_speaker.filename, file_speaker.label))
    dic['speaker_label'] = speaker_label_info[os.path.basename(filename)[7:9]+"'"]
    
    # Correction of original and prediction
    if dic['original_label'] == dic['prediction_result']:
        dic['wav_accuracy'] = 1
    else:
        dic['wav_accuracy'] = 0
    print('wav file accuracy = ', dic['wav_accuracy'])
    result.append(dic)

file_manage = pd.DataFrame(result)
file_manage = file_manage.sort_values(by=['speaker_label'])
# Save to csv file
file_manage.to_csv('./result/total_result_info.csv', index=False)

# Prediction accuracy per speaker
speaker_accuracy = pd.DataFrame(file_manage['wav_accuracy'].groupby(file_manage['speaker_label']).mean())
print('speaker accuracy mean = ', speaker_accuracy)
speaker_accuracy.to_csv('./result/total_accuracy_info.csv', index=True)

# Total prediction accuracy
total_accuracy = file_manage['wav_accuracy'].mean()*100
print('total accuracy mean = ', total_accuracy, '%')
