#!/usr/bin/env python
# coding: utf-8

# # DCASE 2020 Task 2 Preprocessed Dataset EDA
# 
# This is a Kaggle notebook version from [github repository](https://github.com/daisukelab/dcase2020_task2_variants/blob/master/data_eda.ipynb), will perform simple exploratory data analysis on the [DCASE 2020 Task 2 dataset](https://zenodo.org/record/3678171#.XnRgCi3AOL4).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

print('Files in this dataset:')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
from pathlib import Path
import pandas as pd

import numpy as np
import librosa
import librosa.core
import librosa.feature

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

sns.set_palette("husl")

def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : np.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        print("file_broken or not exists!! : {}".format(wav_name))


# ## Download file information
# 
# The `file_info.csv` is created by a script [get_file_info.py](https://github.com/daisukelab/dcase2020_task2_variants/blob/master/get_file_info.py) on the repository.
# 
# Running the script to create here takes long, then simply download a copy.

# In[ ]:


get_ipython().system(' wget https://raw.githubusercontent.com/daisukelab/dcase2020_task2_variants/master/file_info.csv')

df = pd.read_csv('file_info.csv')
df.file = df.file.map(lambda f: str(f).replace('/data/task2/dev', '/kaggle/input/dc2020task2'))
types = df.type.unique()

df.head()


# ## Number of samples
# 
# Let's check number of original audio samples for each machine types.

# In[ ]:


agg = df[['file', 'type', 'split']].groupby(['type', 'split']).agg('count')
fig = plt.figure(figsize=(12.0, 6.0))
g = sns.barplot(x="type", y="file", hue="split", data=agg.reset_index())
plt.ylabel("machine type")
plt.ylabel("number of files")
plt.show()
agg.transpose()


# ## Shape of numpy files
# 
# This preprocessed dataset has 12 numpy files, one file corresponds to train or test split for one of machine types.
# 
# All the samples in one of original folder were converted into log mel spectrogram, then packed into single numpy array and saved to a file.
# 
# Then the first dimension is number of files, the second dimension is `n_mels` and the last dimension is number of frames.

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in sorted(filenames):
        x = np.load(Path(dirname)/filename)
        _, mtype, split = Path(filename).stem.split('-')
        print(f"{mtype}'s {split} split numpy data file {filename}, shape (files, n_mels, frames): {x.shape}")


# ## Check/confirm all samples completely
# 
# Do all samples have consistent property?
# 
# Followings are checked here:
# 
# - sampling rate
# - frame length, duration

# In[ ]:


df.groupby(['type', 'split']).describe()


# Now we could confirm that:
# 
# - Data format is consistent among files in one machine type.
# - Only ToyCar has longer audio 11s, all others are 10s long.
# - Sampling rate is consistently 16,000 Hz.
# 
# ## Frequency Bins Statistics
# 
# The followings will show visual basic statistics of each frequency bins in a log mel spectrogram data.

# In[ ]:


for t in types:
    for split in ['train', 'test']:
        type_df = df[df['type'] == t][df.split == split].reset_index()
        file_path = Path(f'/kaggle/input/dc2020task2prep/dc2020t2l1-{t}-{split}.npy')
        X = np.load(file_path)
        # visualize
        R = 2
        fig, ax = plt.subplots(R, 4, figsize = (15, 5*R//2))
        print(f'=== Machine type [{t}], {split} set ===')
        for i in range(R * 4):
            file_index = i % 4 + ((i // 8) * 4)
            if (i % 8) < 4:
                ax[i//4, i%4].set_title(Path(type_df.file[file_index]).name)
                ax[i//4, i%4].plot(X[file_index])
            else:
                ax[i//4, i%4].violinplot(X[file_index, ::4, :].T)
                ax[i//4, i%4].get_xaxis().set_visible(False)
        plt.show()


# ## Spectrograms
# 
# Samples from each machine types are shown here.

# In[ ]:


for t in types:
    for split in ['train', 'test']:
        type_df = df[df['type'] == t][df.split == split].reset_index()
        file_path = Path(f'/kaggle/input/dc2020task2prep/dc2020t2l1-{t}-{split}.npy')
        X = np.load(file_path)
        # visualize
        R = 2
        fig, ax = plt.subplots(R, 1, figsize = (15, 2.5*R))
        print(f'=== Machine type [{t}], {split} set ===')
        for i in range(R * 1):
            file_index = i
            file_path = Path(type_df.file[file_index])
            mels = X[i]
            ax[i].set_title(file_path.name)
            ax[i].imshow(mels)
            ax[i].axis('off')
        plt.show()


# In[ ]:




