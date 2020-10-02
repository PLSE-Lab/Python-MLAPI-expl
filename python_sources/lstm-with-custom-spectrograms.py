#!/usr/bin/env python
# coding: utf-8

# ## LSTM Sound Classification
# 
# #### Why should you look through this notebook?
# The performance of my model is not great. But I think my spectrogram generator is quite nice

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.models import load_model
import os
from tqdm import tqdm
Files = {    
        'Test':{
                'csv':'../input/freesound-audio-tagging-2019/sample_submission.csv',
                'wav_dir':'../input/freesound-audio-tagging-2019/test'}
        }


# In[ ]:


df = pd.read_csv(Files['Test']['csv'])
wavs = df.loc[:,'fname'].tolist()
PATH = Files['Test']['wav_dir']


# ## Custom Spectrogram
# 
# I started with Librosa's spectrogram generator but decided to create my own so that I could more easily control things like pixel:frequency ratios. 
# 
# #### Sidenote
# I found that Librosa.load(fname) was quite slow at unpacking .wav files. I also tried my own method, using python's Struct and Wave modules but saw no improvement. Scipy's read_wav consistently took 33% less time than Librosa's load function. This makes a big difference when reading so many files.
# 
# ### Technical details
# 
# A spectrogram is just a bunch of frequency distributions placed side by side in order to show visually how the frequency distribution changes as time progresses. In order to create one, you must break up the audio at regular intervals and obtain the frequency spectrum for each of those segments. This is achieved using a (Fast) Fourier Transformation or (FFT).
# 
# 
# #### Window Size
# How many samples long should the segments be?
# 
# * Powers of 2 seem to speed this up a LOT. 
# 
# * Larger window --> higher frequency resolution but lower time evolution resolution
#     
# * A window length of 1 second (44100 samples) gives you a 1Hz resolution. Typical humans have about 3Hz resolution, so 0.3 seconds (14700 samples) is probably the largest window you would want. (2048 samples) gives you (~1/20 seconds) time resolution and(20 Hz) frequency resolution. Which I think is a nice middle ground.
# 
# #### Cuttoff Frequency
# 
# * To save some space and to direct the algorithm to the most important information, I send it only the frequency information from 0-3000Hz. This should probably be closer to 5000Hz
# 
# #### Spectrogram Window
# * In the following IPython cell you can see the spectrogram function has an input variable `N`. This is the width of the output image in pixels, and the number of segments of audio over which we compute an FFT. This means length of the audio represented in the final image is fixed by `N` and `window_length`. The relationship is shown by
# 
# $Time = N*\dfrac{window\_length}{sample\_rate} = 50*\dfrac{2048}{44100}=2.32$ 
# 
# * In order to fix the time dimension of our spectrograms I somewhat arbitrarily chose each spectrogram to cover 2.3 seconds of the audio file in question
# * For files shorter than 2.3 seconds I pad the waveform with zeros
# * For those larger, I select the loudest consecutive 2.3 seconds of the clip
#  * This is probably a very large source of error
# 

# In[ ]:


from scipy.fftpack import rfft
from scipy.io.wavfile import read as read_wav

# N is the width of the output image and determines the duration of the audio which is processed
def spectrogram(y,sr,N=50): 
    
    # ---------------Segment audio for FFTs-------------
    window_length = 2048 #1024          
    num_windows = len(y)//window_length
    
    
    
    # -------------Select Portion of Audio to use-----------------
    # If we have the exact length for only one choice for our 2.3 second window
    if num_windows==N:
        y = y[:N*window_length]
    
    # If we need to add some zeros to bring it up to size
    elif num_windows<N:
        diff = N*window_length - len(y)
        before = diff//2
        after = diff-before
        y = np.pad(y,(before,after), mode='constant', constant_values=0)
    
    #------------------------------------------
    # If we need to select the best portion to use
    
    #  THIS IS THE PLACE TO START FOR IMPROVEMENT
    
    # A better approach would be to select one of the loud portions at random
    #    in an image generator for the learning model. But due to some memory
    #    issues I'm having in Kaggle Kernels, I could not get this to work. And
    #    it takes much longer
    #------------------------------------------
    else:
        
        
        volume = []
        for i in range(0,len(y)-window_length*N+1,window_length):
            volume.append(abs(y[i:i+window_length*N]).sum())
        volume = np.array(volume)
 
        m = max(np.array(volume).argmax()-5,0)
        y = y[window_length*m:window_length*(m+N)]
    

    # --------------------Compute FFT----------------------
    y = y.reshape((N,window_length))
    Y = abs(rfft(y,axis=1)).T
    

    # ---------------------Normalize-----------------------
    Y = (Y-Y.min())
    if Y.max()==0:
        pass
    else:
        Y = Y/Y.max()
    
    # ---------Apply Cutoff Frequency (20-3000Hz)----------
    # Convert to np.float32 to save memory
    return Y[1:150,:].astype(np.float32)


def make_spec(path, filename,N=50):
    fname = path+'/'+filename
    sr,y = read_wav(fname)
    return np.flip(spectrogram(y,sr,N).T,0)


# In[ ]:


X = []
for i in tqdm(wavs):
    X.append(make_spec(PATH,i))


# ### That is the entire Test set loaded and converted into a spectrogram in 35 seconds
# That is roughly $\dfrac{1}{10}th$ the time it takes for librosa to load in the wav files (without computing the spectrograms)
# 
# First of all Scipy's read_wav function is much faster than librosa.load(), so that is a great start. As for the spectrogram, I'm more concerned with the customizable pixel to frequency/time ratios, the speed is not really comparable to Librosa's implementation because I only compute FFTs on a fraction of the input audio. 

# In[ ]:


import librosa
tmp = []
for i in tqdm(wavs[:100]):
    y,sr = librosa.load(PATH+'/'+i)
    tmp.append(y)
    
del tmp


# ### Input Spectrograms
# Here is a sample of what I'm sending into the LSTM model

# In[ ]:


for n in range(3):
    plt.figure(1)
    for i in range(5):
        plt.subplot(151+i)
        plt.imshow(make_spec(PATH,wavs[i+n*5]))
        plt.yticks([])
        plt.xticks([])
        plt.title(wavs[i+n*5], fontsize=8)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
    plt.show()


# In[ ]:


LSTM_model = load_model('../input/freesound-keras-lstm-model/Keras LSTM Model')
LSTM_model.summary()


# In[ ]:


X = np.array(X)
y_pred = LSTM_model.predict(X)


# In[ ]:


fnames = np.array(wavs).reshape((len(wavs),1))
data = np.concatenate((fnames,y_pred),1)
pd.DataFrame(data, columns=list(df)).to_csv('submission.csv', index=False)


# ## Spoiler
# The accuracy I was able to get with these spectrograms is 0.299. Though, this is my first attempt at deep learning so you can almost certainly do much better. To conlude, if you want to use a similar spectrogram generator I would advise either; randomizing the location of the spectrogram window at each training epoch, or finding a more reliable method for determining the location of the window (more reliable than __the loudest one possible__ that is). But as it stands, it is a pretty convinient function. The entire dataset (Curated, Noisy, Train) takes up about 1.7 Gb of memory using the dimentional parameters (`N` and `window_length`) explained at the top of this notebook

# In[ ]:




