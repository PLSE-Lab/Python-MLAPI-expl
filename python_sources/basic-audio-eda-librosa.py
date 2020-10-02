#!/usr/bin/env python
# coding: utf-8

# In[ ]:


conda install -c conda-forge librosa


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


sample_audio = [
    'aldfly/XC134874.mp3',
    'amegfi/XC109299.mp3',
    'brebla/XC104521.mp3',
    'lewwoo/XC161334.mp3',
    'macwar/XC125970.mp3',
    'norwat/XC124175.mp3',
    'pinjay/XC153392.mp3',
    'rufhum/XC133552.mp3',
    'weskin/XC124287.mp3',
    'yetvir/XC120867.mp3'    
]


# In[ ]:


BASE_PATH = '../input/birdsong-recognition'

# image and mask directories
train_data_dir = f'{BASE_PATH}/train_audio'


# In[ ]:


for audio in sample_audio:
    print("Audio sample of bird", audio.split('/')[0])
    audio_file = f"{train_data_dir}/{audio}"
    display(ipd.Audio(audio_file))
    
    signal1, rate1 = librosa.load(audio_file, duration=5)   #default sampling rate is 22 HZ
    dur=librosa.get_duration(signal1)
    print("Duration in seconds. ",librosa.get_duration(signal1))
    print(signal1.shape, rate1)

