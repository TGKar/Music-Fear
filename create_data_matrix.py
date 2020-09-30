import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import sklearn

def create_data_mat(csv_path='./set1/mp3/Soundtrack360_mp3/'):

    os.chdir(csv_path)
    csv_files = glob.glob('*.{}'.format('csv'))
    data_mat = []
    for file_name in csv_files:
        print(file_name)
        with open(file_name, 'r') as file:
            features_line = file.readlines()[-1]
            data_mat.append([float(feature) for feature in features_line.split(',')[1:-1]])
    data_mat = np.array(data_mat)
    feature_max = np.max(data_mat, 0)
    feature_min = np.min(data_mat, 0)
    data_mat = (data_mat - feature_min) / (feature_max - feature_min)  # Normalize
    np.savetxt('data_mat.csv', data_mat, delimiter=',')

create_data_mat()