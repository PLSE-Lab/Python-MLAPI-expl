#!/usr/bin/env python
# coding: utf-8

# I have just made changes to the existing kernel : https://www.kaggle.com/pavansanagapati/urban-sound-classification-using-cnn-model

# In[ ]:


import IPython.display as ipd
ipd.Audio('../input/train/Train/1008.wav')


# To load the audio files into the jupyter notebook ass a numpy array I have used 'librosa' library in python by using the pip command as follows
# 
#  ***pip install librosa***

# In[ ]:


import os

import pandas as pd

import librosa

import librosa.display

import glob

get_ipython().run_line_magic('pylab', 'inline')

from sklearn.preprocessing import LabelEncoder

import numpy as np

from scipy.fftpack import fft

from scipy import signal

from scipy.io import wavfile

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.optimizers import Adam

from sklearn import metrics 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))


# Now let us load a sample audio file using librosa

# In[ ]:


data,sample_rate = librosa.load('../input/train/Train/2004.wav')

plt.figure(figsize=(12,4))

librosa.display.waveplot(data,sr=sample_rate)


# As you can see the air conditioner class is shown as random class and we can see its pattern.Let us again see another class by using the same code to randomly select another class and observe its pattern

# It appears that jackhammer has more count than any other classes
# 
# Now let us see how we can leverage the concepts we learned above to solve the problem. We will follow these steps to solve the problem.
# 
# - Step 1: Load audio files & Extract features
# - Step 2: Convert the data to pass it in our deep learning model
# - Step 3: Run a deep learning model and get results
# 
# #### Step 1: Load audio files & Extract features
# 
# Let us create a function to load audio files and extract features

# #### Step 2: Convert the data to pass it in our deep learning model

# In[ ]:


import numpy as np
import pandas as pd
import librosa

def job():
    df = pd.read_csv('../input/train.csv')

    Classes = df.Class.unique().tolist()
    y = []
    X = []; yp = []; new_X = []
    for i in df.ID:
        new, rate = librosa.load('../input/train/Train/%d.wav'%i)
        mfccs = np.mean(librosa.feature.mfcc(y=new, sr=rate, n_mfcc=200).T, axis = 0)
        idx = Classes.index(df[df.ID == i]['Class'].tolist()[0])
        yp.append(idx)
        print (i)
        new_X.append(mfccs)
    return new_X, yp


# ### Step 3: Run a deep learning model and get results,
# I will use Random Forest Classifier

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time

t_beg = time.time()

NewX, y = job()

NewX_train, NewX_test, Newy_train, Newy_test = train_test_split(NewX, y, test_size=0.2, shuffle = True)
print ('successfully splitted')
t0 = time.time()
print ('time elapsed for reading: ', t0-t_beg)

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
#clf = LogisticRegressionCV(multi_class = 'ovr')
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(NewX_train, Newy_train)
t1 = time.time()
print ('time elapsed for fitting: ', t1-t0)
print ('done fitting')

print ('fit to train new: ', clf.score(NewX_train, Newy_train))
print ('fit to test: ', clf.score(NewX_test, Newy_test))


# In[ ]:




