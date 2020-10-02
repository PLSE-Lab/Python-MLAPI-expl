#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa as ls
import os


# In[ ]:


emotion_dic = {
                '01' : 'neutral', 
                '02' : 'calm' ,
                '03' : 'happy' ,
                '04' : 'sad' ,
                '05' : 'angry', 
                '06' : 'fearful' ,
                '07' : 'disgust', 
                '08' : 'surprised'
}


# In[ ]:


our_emotion = ['happy','sad','angry','disgust']


# In[ ]:


from glob import glob


# In[ ]:


def extract_feature(file_name, mfcc, chroma, mel):
        X,sample_rate = ls.load(file_name)
        if chroma:
            stft=np.abs(ls.stft(X))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(ls.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(ls.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(ls.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        return result


# In[ ]:


audio_files = glob('/kaggle/input/ravdess-emotional-speech-audio'+'/*/*.wav')


# In[ ]:


for i in range(5):
    audio,sfreq = ls.load(audio_files[i])
    time = np.arange(0,len(audio))/sfreq
    fig,ax = plt.subplots()
    ax.plot(time,audio)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')


# In[ ]:


x,y=[],[]
from IPython.display import clear_output
e = set()
for file in audio_files:
        clear_output(wait=True)
        file_name = file.split('/')[-1]
        emotion=emotion_dic[file_name.split("-")[2]]
        if emotion not in our_emotion:
            continue
        e.add(file.split('/')[-2])
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        print(e)
        x.append(feature)
        y.append(emotion)


# In[ ]:


x = np.array(x)
x


# In[ ]:


y


# In[ ]:


from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)
y


# In[ ]:


y = to_categorical(y)
print(y.shape)
print(y)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler  =  MinMaxScaler()
x = scaler.fit_transform(x)


# In[ ]:


x


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42,shuffle=True)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten


# In[ ]:


x = np.expand_dims(x,axis=2)


# In[ ]:


model = Sequential()
model.add(Dense(256,input_shape=(x.shape[1],1)))
model.add(Dense(512))
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Dense(256))
model.add(Flatten())
model.add(Dense(4,activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size = 64,epochs=150,validation_data=(x_test,y_test))


# In[ ]:


model.save('Speech-Emotion-Recognition.h5')


# In[ ]:




