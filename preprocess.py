import librosa
import pandas as pd
import numpy as np

import os
# import csv

# Comment out these 2 lines to show warnings
import warnings
warnings.filterwarnings('ignore')


# put audio files of each class in a separate folder and name them starting from '0','1','2'.... with as many speakers/classes you want

list_dicts = []

class_dirs = [d for d in os.listdir(os.getcwd()) if d.isdigit()]

for class_dir in class_dirs:
    files = [f for f in os.listdir(os.path.join(os.getcwd(),class_dir)) if 'mp3 in f']
    for file in files:
        file_data = {}
        file_data['file'] = os.path.join(os.getcwd(),class_dir,file)
        y, sr = librosa.load(file_data['file'])
        file_data['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        file_data['rmse'] = np.mean(librosa.feature.rms(y=y))
        file_data['spec_cent'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        file_data['spec_bw'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        file_data['rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        file_data['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        
        for i, m in enumerate(mfcc):
            file_data['m'+str(i)] = np.mean(m)
        file_data['label'] = class_dir
        list_dicts.append(file_data)
    print('class',class_dir,'completed')

data_frame = pd.DataFrame(list_dicts)
data_frame.to_csv('out.csv')