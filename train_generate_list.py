import os
from os.path import splitext, isdir
import glob
import pandas as pd


def generate_spk_label(data_folder, leave_one_out=True):
    # Split train/test data by speaker ID
    if leave_one_out == True:
        speaker_train_lst = ['01', '02', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42']
        speaker_test_lst = ['03', '26', '43', '44', '45', '46', '47', '48', '49', '50']

        speaker_train_dic = {speaker_train_lst[i] : i for i in range(len(speaker_train_lst))}
        speaker_test_dic = {speaker_test_lst[i] : i for i in range(len(speaker_test_lst))}

        data = {'speaker_name': speaker_train_lst, 'speaker_label': [speaker_train_dic[i] for i in speaker_train_lst]}
        file_manage_train = pd.DataFrame.from_dict(data)
        print(file_manage_train.speaker_label.value_counts()) # confirm label count
        file_manage_train.to_excel('{}/speaker_train_index.xlsx'.format(data_folder), index=None)

        data = {'speaker_name': speaker_test_lst, 'speaker_label': [speaker_test_dic[i] for i in speaker_test_lst]}
        file_manage_test = pd.DataFrame.from_dict(data)
        print(file_manage_test.speaker_label.value_counts()) # confirm label count
        file_manage_test.to_excel('{}/speaker_test_index.xlsx'.format(data_folder), index=None)

    # Split train/test data w/ same speaker ID
    else:
        # except 03 speaker, because no korean script data in 03 speaker
        speaker_lst = ['01', '02', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']
        speaker_dic = {speaker_lst[i] : i for i in range(len(speaker_lst))}

        data = {'speaker_name': speaker_lst, 'speaker_label': [speaker_dic[i] for i in speaker_lst]}
        file_manage = pd.DataFrame.from_dict(data)
        print(file_manage.speaker_label.value_counts()) # confirm label count
        file_manage.to_excel('{}/speaker_index.xlsx'.format(data_folder), index=None)


def generate_file_list(data_folder, subset='train', n_states=2)
    # TRAIN 경로 excel 파일 추출
    result = []
    if subset is 'train':
        folder_path = '{}/TRAIN/'.format(data_folder)
    else:
        folder_path = '{}/TEST/'.format(data_folder)
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if splitext(filename)[1] == ".wav":
                dic = {'stress_degree' : '', 'foldername' : '','filename_1' : '', 'filename_2': '', \
                        'label' : '', 'speaker_label' : ''}
                print(os.path.join(root, filename))
                print(root)
                if n_states == 2:
                    if '/ns' in root: # Non-stressed
                        stress_degree = 'ns'
                        label = 0
                    elif '/ss' in root: # Stressed
                        stress_degree = 'ss'
                        label = 1
                else:
                    if '/ns' in root: # Non-stressed
                        stress_degree = 'ns'
                        label = 0
                    elif '/is' in root: # Weakly-stressed
                        stress_degree = 'is'
                        label = 1
                    elif '/ss' in root: # Strongly-stressed
                        stress_degree = 'ss'
                        label = 2

                dic['stress_degree'] = stress_degree
                dic['foldername'] = root.split('/')[-1]
                dic['filename_1'] = filename
                dic['filename_2'] = str(os.path.join(root, filename)).replace("/", '/')
                dic['label'] = label
                dic['speaker_label'] = speaker_dic[str(splitext(filename)[0])[7:9]]
                result.append(dic)

    file_manage = pd.DataFrame(result)
    print(file_manage.label.value_counts()) # confirm label count
    file_manage.to_excel('{}/data_file_info_chunk_15_{}set.xlsx'.format(data_folder, subset), index=None)


if __name__ == '__main__':
    data_folder = './4th_stress_data'
    leave_one_out = False
    n_states = 2

    generate_spk_label(data_folder, leave_one_out)
    generate_file_list(data_folder, 'train', n_states)
    generate_file_list(data_folder, 'test', n_states)