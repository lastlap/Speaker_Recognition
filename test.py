# import IPython

import pandas as pd
from sklearn.preprocessing import StandardScaler
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from xgboost import XGBClassifier
import numpy as np

import time

# import sounddevice as sd
# from scipy.io import wavfile

# Comment these 2 lines to show warnings
import warnings
warnings.filterwarnings('ignore')

def extract_features(file):

    file_data = []
    y, sr = librosa.load(file)
    file_data.append(np.mean(librosa.feature.chroma_stft(y=y, sr=sr))) # chroma_stft
    file_data.append(np.mean(librosa.feature.rms(y=y))) # rmse
    file_data.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) # spec_cent
    file_data.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))) # spec_bw
    file_data.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))) # rolloff
    file_data.append(np.mean(librosa.feature.zero_crossing_rate(y))) # zcr
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    for i, m in enumerate(mfcc):
        file_data.append(np.mean(m))
    return file_data

data = pd.read_csv('out.csv')
data = data.drop(['Unnamed: 0','file'],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
y = np.array(data['label'],dtype=int)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = XGBClassifier()
clf = clf.fit(X,y)
# print(X.shape)

# Add test file names to the list to test on them
test_files = ['1_test.mp3','0_test.mp3','2_test.mp3']

for file in test_files:
    file_data = extract_features(file)
    audio_features = np.array(file_data)[:].reshape(-1,X.shape[1])
    y_pred = clf.predict(scaler.transform(audio_features))
    print('Predicted Class for file',file,'is: ',y_pred)