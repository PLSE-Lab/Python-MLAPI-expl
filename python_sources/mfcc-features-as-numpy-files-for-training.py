#!/usr/bin/env python
# coding: utf-8

# If you, as me, wanted to start with competition [Cornell Birdcall Identification](https://www.kaggle.com/c/birdsong-recognition/overview) but did not find an easy way to deal with a lot of audio files, this notebook may help you.
# 
# In this competition, we are given audio files to identify birds by their calls. Training models on them requires feature extraction. One of the common methods is MFCC. There are two issues with doing this on-the-fly: 
# * It takes a lot of time to process ALL given files.
# * The extracted features do not fit into RAM.
# 
# This notebook fixes both of these issues by extracting features from ALL audio files and saving the results in separate files as standard **NumPy arrays**. The files are archived in TAR file **train_features.xz** (lzma compression) which you will need to open in the reading mode. To train your models, you can load features from those files (you can do that in batches as well). Here is a minimal example showing how to load features from the archive:
# 
#      import tarfile
#      from io import BytesIO
#      
#      with tarfile.open("../input/mfcc-features-as-numpy-files-for-training/train_features.xz", "r:xz") as tar:
#           for member in tar.getmembers():
#               np_file = tar.extractfile(member)
#               features = np.load(BytesIO(np_file.read()))
# 
# The files are named as XC\*\*\*.mp3.npy, so you can load features for any audio file from the training set using its name. The link between the file names and labels (bird identifiers) can be found in **train.csv**.
# 
# I will make another notebook for test audio files once they become available. Please upvote if you find this useful.

# ## Imports

# In[ ]:


import numpy as np
import pandas as pd
import warnings
import random

import librosa
import librosa.display
from tqdm import tqdm_notebook as tqdm

import tarfile
from pathlib import Path

warnings.filterwarnings('ignore')


# ## Extract MFCC features and save them to files

# In[ ]:


input_dir = Path('../input/birdsong-recognition/train_audio')


# In[ ]:


MFCC = {
    "sr": 22050, # sampling rate for loading audio
    "n_mfcc": 12 # number of MFCC features per frame that can fit in HDD
}


# In[ ]:


def load_audio(filename):
    try:
        return librosa.load(filename, sr=None)
    except Exception as e:
        print(f"Cannot load '{filename}': {e}")
        return None


# In[ ]:


def extract_mfcc(y, sr=22050, n_mfcc=10):
    try:
        return librosa.feature.mfcc(y=y, 
                                    sr=sr if sr > 0 else MFCC["sr"], 
                                    n_mfcc=n_mfcc)
    except Exception as e:
        print(f"Cannot extract MFCC: {e}")
        return None


# In[ ]:


def parse_audio(input_dir, output_file, max_per_label=10000):
    
    with tarfile.open(output_file, "w:xz") as tar:
    
        sub_dirs = list(input_dir.iterdir())    
        for sub_dir in tqdm(sub_dirs):

            for i, mp3 in enumerate(sub_dir.glob("*.mp3")):

                if i >= max_per_label:
                    break

                ysr = load_audio(mp3)
                if ysr is None:
                    continue

                mfcc = extract_mfcc(y=ysr[0], 
                                    sr=ysr[1], 
                                    n_mfcc=MFCC['n_mfcc'])
                if mfcc is None:
                    continue
                
                filename = Path(f"{mp3.name}.npy")
                np.save(filename, mfcc)            
                tar.add(filename)
                filename.unlink()


# In[ ]:


output_file = Path('train_features.xz')


# In[ ]:


parse_audio(input_dir, output_file)


# In[ ]:


sub_df = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')
sub_df.to_csv('submission.csv', index = None)

