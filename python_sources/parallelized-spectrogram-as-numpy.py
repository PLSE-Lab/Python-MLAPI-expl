#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import re


# In[ ]:


import librosa
import librosa.display
from joblib import Parallel, delayed, cpu_count

import torchaudio
from scipy import signal

from zipfile import ZipFile
import gc


# In[ ]:


df = pd.read_csv("../input/birdsong-recognition/train.csv")


# ### Images to Spectogram arrays every 5 seconds of audio
# If audio is less than 5 seconds first, then it is assumed to have a bird call (so no_call cannot be there) and is zeropadded to complete the 5 seconds of audio. 
# In the other case when audio is more that 5 seconds if the audio has a standard deviation inferior to $0.005$ and a mean smaller than $0.005$ it is assumed to have no call (just white noise), this is very simplistic as a model, one might even want to test with statistical test if the audio is a white noise, but with kaggle limitation on computation time, this seems the only easy way out.
# 
# Convert all the audio files into spectogram as numpy arrays, this has several advantages such as having temporal infomation and you can split the spectrogram based on time later during the training solving one of the issues regarding the different length in audio.
# 
# I am saving here as numpy compressed format and as `int8` because I am limited by the kaggle disk space of 5GB, this will slow down the loading time of the spectrogram since it will need to decompress the arrays and I am losing some information saving the values as int instead of `float`. If you have the possibility to use it on you personal PC please change the only one line of code:
# 
# `
# np.savez_compressed(image_fname, np.round(log_S,0).astype("int8"))
# `
# 
# to 
# 
# `
# np.save(image_fname, log_S)
# `

# In[ ]:


zipFile = ZipFile('birdsong.zip', 'w')

SPEC_TIME = 5
MIN_STD = 0.005
MIN_MEAN = 0.005

def load_audio(path, sr):
    "This Loads the audio file faster than the librosa library!"
    x, x_sr = torchaudio.load(path)
    x = x.numpy()

    if x.ndim > 1:
        x = np.mean(x, axis=0)
        
    if sr:
        x = signal.resample(x, int(x.size*float(sr)/x_sr))
        x_sr = sr
    return x, x_sr

def save_spectrogram(audio_fname):
    """
    This will save in the the numpy file instead of image so the 
    difference of times can be managed later (or may be used by a RNN)
    """
    try:
        image_fname = audio_fname.replace("/input/",
                                          "/working/").replace("_audio/", 
                                                               "_npy/").replace(".mp3",
                                                                                  ".npz")
        fname = re.search("/([a-zA-Z0-9\.]+$)", audio_fname).group().strip("/")
        sr_ = 32000
        #sr_ = int(df.sampling_rate[df.filename==fname].values[0].replace(" (Hz)", ""))
        
        y, sr = load_audio(audio_fname, sr=sr_)
        duration = np.floor(len(y)/sr).astype(int)
        
        if duration >= SPEC_TIME:
            splits = int(duration/SPEC_TIME)
            for split in range(splits):
                y_temp = y[split*sr_*SPEC_TIME:(split+1)*sr_*SPEC_TIME]
                if np.std(y_temp) < MIN_STD and np.mean(y_temp) < MIN_MEAN:
                    last = f"_{split}_nc.npz"
                else:
                    last = f"_{split}.npz"
                fname_temp = image_fname.replace(".npz", last)
                
                S = librosa.feature.melspectrogram(y_temp, 
                                                   sr=sr, 
                                                   n_mels=96)
                log_S = librosa.power_to_db(S, ref=np.max)
                
                np.savez_compressed(fname_temp, log_S.astype("int8"))
        else:
            y_temp = np.zeros((SPEC_TIME*sr,))
            y_temp[:y.shape[0]] = y
            
            fname_temp = image_fname.replace(".npz", "_0.npz")
                
            S = librosa.feature.melspectrogram(y_temp, 
                                               sr=sr, 
                                               n_mels=96)
            log_S = librosa.power_to_db(S, ref=np.max)

            np.savez_compressed(fname_temp, log_S.astype("int8"))
        
        
        #This part will save the images instead!
        #librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        #fig1 = plt.gcf()
        #plt.axis('off')
        #fig1.savefig(image_fname, bbox_inches='tight', 
        #             pad_inches=0, dpi=100)
    except Exception as e:
        print(str(e) + "WTH!?"+image_fname)
        pass
    
def create_data(path, parallel=False):
    count = 0
    for dirname, _, filenames in os.walk(path):
        count += 1
        labels = []
        features = []
        image_dir = dirname.replace("/input/","/working/").replace("_audio", "_npy")
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        print("doing " + dirname + " number: " + str(count) + "\t")
        if parallel:
            all_paths = []
            for i, filename in enumerate(filenames):
                all_paths.append(os.path.join(dirname, filename))
            Parallel(n_jobs=cpu_count(), verbose=0)(delayed(save_spectrogram)(audio_fname=path) for path in all_paths)        
        else:
            all_paths = []
            for i, filename in enumerate(filenames):
                print("done " + str(filename), end="\r")
                all_paths.append(os.path.join(dirname, filename))
                save_spectrogram(os.path.join(dirname, filename))
                
        for dir_, _, files_ in os.walk(image_dir):
            for file_ in files_:
                zipFile.write(os.path.join(dir_,file_))
                
        if image_dir != "/kaggle/working/birdsong-recognition/train_npy/":     
            os.system(f"rm -r {image_dir}")
            
        os.system('echo $(df -h /kaggle/working)')
        gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif not os.path.exists("/kaggle/working/birdsong-recognition/"):\n    os.mkdir("/kaggle/working/birdsong-recognition/")\nelse:\n    os.system("rm -r /kaggle/working/birdsong-recognition/")\n    os.mkdir("/kaggle/working/birdsong-recognition/")\n\ntrain_path = "/kaggle/input/birdsong-recognition/train_audio/"\ncreate_data(train_path, parallel=True)')


# In[ ]:


zipFile.close()


# One of the questions on this notebook was the performace, in terms of time I gained using the Parallel processing of the audios instead of one at a time, well on the 4 CPUs of kaggle it ain't much but it should be a lot while using a decent PC with more processors and the other advantage (probably the bigger one) is the reduction in RAM memory usage since python has to free and return the memory to the system after the completion of each process (so each audio) while processing one audio at a time, python prefers to keep some of freed memory for itself after the completion and after a while a crash can occur due to excess memory (RAM) usage.  
