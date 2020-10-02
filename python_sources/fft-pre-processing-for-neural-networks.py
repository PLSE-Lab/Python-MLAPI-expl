#!/usr/bin/env python
# coding: utf-8

# 

# # 1. Introduction
# 
# In this kernel, I am illustrating a basic methodology to obtain a 3D tensor of frequencies using Fast Fourier Transforms (FFT). My lectures on the subject are a bit outdated, so I defer to [Wikipedia](https://en.wikipedia.org/wiki/Fast_Fourier_transform) for a better explanation of the matter:
# 
# > A fast Fourier transform (FFT) is an algorithm that samples a signal over a period of time (or space) and divides it into its frequency components.[1](https://en.wikipedia.org/wiki/Fast_Fourier_transform#cite_note-1) These components are single sinusoidal oscillations at distinct frequencies each with their own amplitude and phase. 
# 
# There is a lot that can be refined in the final output. First, it can be reduced to a 2D tensor by dropping the imaginary part of the frequency domain. Second, in order to have a fixed shape, I padded the end of the tensor with 0s. An alternative method that has the advantage to be cheaper computationally would be to limit the size of the audio file. Third, to simplify the problem of the tensor size, I did not use any overlap when computing the spectrum of the signal.
# 
# I hope this may be useful for you if you intended to go through frequencies. If you have any comments, I would be happy to here them!
# 
# ## 1.1 Acknowledgements
# Giving credit where credit is due, since I never used audio data before this competition, I used [@Zafar's Kernel](https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data) to quickstart this one. I had trouble selecting among the various options I could use to apply the FFT on data, and I finally opted for scipy's spectrogram thanks to [@Lathwal's kernel](https://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis).
# 
# ## 1.2 Importing Datas

# In[108]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")
train.head()


# # 2. FFT on a Single File
# ## 2.1 Getting the Audio Signal for a Single File
# In this section, my aim is to build basis understanding of applying a FFT to an audio signal. First, let us try one of the audio file.

# In[109]:


import IPython.display as ipd  # To play sound in the notebook
fname = '../input/audio_train/' + '001ca53d.wav'
ipd.Audio(fname)


# Now, let us import the audio file as a np array.

# In[110]:


from scipy.io import wavfile
rate, data = wavfile.read(fname)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
print(data)


# A final step is to normalize data. A quick look at the [data description](https://www.kaggle.com/c/freesound-audio-tagging/data) reveals that all audio files are encoded as "uncompressed PCM 16 bit". After a research on internet, it seems that we may normalize the signal with the following formula:
# $$ \frac{x}{2^{16}} * 2 $$
# I have to admit here that my knowledge of data encoding for audio file is limited, so if anyone is more knowledgeable on the matter than me, any comment (either confirmation or a correct normalization method) would be appreciated.

# In[111]:


data = np.array([(e/2**16.0)*2 for e in data]) #16 bits tracks, normalization
plt.plot(data)


# ## 2.2 Retrieving the Frequency Domain
# From this point, we can now use Scipy's signal module to retrieve frequencies. In this case, I opted for the retrieval of the full transform as a complex number rather than the energy level alone. The reason is that I suspect that it provides more information. 
# 
# However, I am not sure of this, and provided that the dataset is small, it might be inneficient to increase the feature space this way. Whatever, if you feel that having the complex values for frequencies may help you, there you go: 

# In[112]:


from scipy import signal
#data
freqs, times, specs = signal.spectrogram(data,
                                         fs=rate,
                                         window="boxcar",
                                        nperseg=13230,
                                        noverlap=0,
                                        detrend=False,
                                        mode = 'complex')

plt.plot(freqs,np.absolute(specs[:,0]))


# ## 2.3 Obtaining a 3D Tensor on a Single File
# Based on the above work, let us create a 3D tensor for a single audio file. Ultimately, I would like to have a tensor with a fixed shape that is not dependent of the audio file so it can be processed in Keras.
# 
# Here, we know that the sampling rate is fixed (44.1kHz) and that audio files vary in length from 0.3 to 30 seconds. I am using this information and the fact that the resulting frequencies of an FFT are fixed to define ex ante the shape of the input tensor for my neural network.
# 
# The first dimension of the tensor will be dictated by the number frames used to compute the spectrum of the signal. Here, I made the choice to use a fixed number of frame per segment. In this case, I opted for the number of frame of the smallest audio file (0.3 seconds, i.e. 13230 frames). This results in $13230/2+1=6615$ bins of frequencies, which is our first dimension size.
# 
# For the second dimension, it is simply the ratio of the largest number of frame over the number of frames per segment. In this case, this results to 100.
# 
# Finally, the third dimension has size 2, one for the real part of the frequency domain, one for the imaginary part. This dimension could be omitted if necessary by taking the magnitude.

# In[113]:


RATE = 44100                                                     #44.1 kHz
MAX_FRAME = int(RATE * 30)                                       #Max frame = 44.1 kHz * 30 seconds
MIN_FRAME = int(RATE * 0.3)                                      #Min frame = 44.1 kHz * 0.3 seconds
NORM_FACTOR = 1.0/2**16.0                                        # Used later to normalize audio signal
 
MAX_INPUT = int(MAX_FRAME / MIN_FRAME)                           #Size of the second dimension
FREQUENCY_BINS = int(MIN_FRAME / 2) + 1                          #Size of the first dimension

#Input of the NN
nn_input = np.zeros((FREQUENCY_BINS,
                    MAX_INPUT,
                    2))

freqs, times, specs = signal.spectrogram(data,                          #Signal               
                                         fs=RATE,                       #Sampling rate
                                         window="boxcar",               #Rectangular segments
                                         nperseg=MIN_FRAME,             #Number of frames per segments
                                         noverlap=0,                    #No overlap
                                         detrend=False,
                                         mode = 'complex')              #Retrieve complex numbers

#Fill the first component of the 3rd dimension with real part
nn_input[:,:specs.shape[1],0] = np.real(specs)
#Fill the first component of the 3rd dimension with imaginary part
nn_input[:,:specs.shape[1],1] = np.imag(specs)

#Display output for a small part of the tensor
nn_input[:3,:3,:]


# # 3. Useful Functions
# 
# I provide here some function to reproduce the above work (see description in their respective \__docstring__)
# 
# ## 3.1 Single Audio File Pre-Processing

# In[114]:


from scipy.io import wavfile

RATE = 44100
 
MAX_INPUT = int(MAX_FRAME / MIN_FRAME)
FREQUENCY_BINS = int(MIN_FRAME / 2) + 1

MAX_FRAME = int(RATE * 30)
MIN_FRAME = int(RATE * 0.3)
NORM_FACTOR = 1.0/2**16.0

def make_tensor(fname):
    """
    Brief
    -----
    Creates a 3D tensor from an audio file
    
    Params
    ------
    fname: name of the file to pre-process
    
    Returns
    -------
    A 3D tensor of the audio file as an np.array
    """
    rate, data = wavfile.read(fname)
    data = np.array([(e*NORM_FACTOR)*2 for e in data])
    output = nn_input = np.zeros((FREQUENCY_BINS,
                                  MAX_INPUT,
                                  2))
    freqs, times, specs = signal.spectrogram(data,                                         
                                         fs=RATE,
                                         window="boxcar",
                                         nperseg=MIN_FRAME,
                                         noverlap=0,
                                         detrend=False,
                                         mode = 'complex')
    output[:,:specs.shape[1],0] = np.real(specs)
    output[:,:specs.shape[1],1] = np.imag(specs)
    return output
    
make_tensor(fname)[1:5,1:5,:]
    
    


# ## 3.2 Pre-Process a List of File (or a Directory)
# 

# In[115]:


import os

def make_input_data(audio_dir, fnames=None):
    """
    Brief
    -----
    Pre-process a list of file or a full directory.
    
    Params
    ------
    audio_dir: str
        Directory where files are stored
    fnames: str or None
        List of filenames to preprocess. If None: pre-process the full directory.
    
    Returns
    -------
    A 4D tensor (last dimension refers to observations) as an np.array
    """
    if fnames is None:
        fnames = os.listdir(AUDIO_DIR)
    else:
        fnames = [fname + '.wav' for fname in fnames]
    output = np.zeros((FREQUENCY_BINS,MAX_INPUT,2,len(fnames)))
    i = 0
    for fname in fnames:
        full_path = os.path.join(audio_dir,fname)
        
        output[:,:,:,i] = make_tensor(full_path)
        i+1
    return output


#Example
AUDIO_DIR = '../input/audio_train/'
fnames = ['00044347','001ca53d']
make_input_data(AUDIO_DIR,fnames)

#This takes too long to run
#make_input_data(AUDIO_DIR)


