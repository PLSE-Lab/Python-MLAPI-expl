#!/usr/bin/env python
# coding: utf-8

# # Audio Classification using FastAI
# 
# Source: https://github.com/sevenfx/fastai_audio

# To begin, we download a bunch of [utility functions](https://github.com/sevenfx/fastai_audio/blob/master/notebooks/utils.py) for I/O and conversion of audio files to spectrogram images.

# In[ ]:


get_ipython().system('rm -rf utils.py')
get_ipython().system('wget https://raw.githubusercontent.com/sevenfx/fastai_audio/master/notebooks/utils.py')


# We can import the necessary modules and functions

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
from pathlib import Path
from IPython.display import Audio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils import read_file, transform_path


# Next, let's download the data. We'll use the free spoken digits database: https://github.com/Jakobovski/free-spoken-digit-dataset

# In[ ]:


get_ipython().system('rm -rf free-spoken-digit-dataset-master master.zip')
get_ipython().system('wget https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip')
get_ipython().system('unzip -q master.zip')
get_ipython().system('rm -rf master.zip')
get_ipython().system('ls')


# In[ ]:


AUDIO_DIR = Path('free-spoken-digit-dataset-master/recordings')
IMG_DIR = Path('imgs')
get_ipython().system('mkdir {IMG_DIR} -p')


# Let's see how many recordings we have, and some sample files.

# In[ ]:


fnames = os.listdir(str(AUDIO_DIR))
len(fnames), fnames[:5]


# As before we can play the recording using the `Audio` widget.

# In[ ]:


fn = fnames[94]
print(fn)
Audio(str(AUDIO_DIR/fn))


# In[ ]:


# ??read_file


# In[ ]:


x, sr = read_file(fn, AUDIO_DIR)
x.shape, sr, x.dtype


# In[ ]:


def log_mel_spec_tfm(fname, src_path, dst_path):
    x, sample_rate = read_file(fname, src_path)
    
    n_fft = 1024
    hop_length = 256
    n_mels = 40
    fmin = 20
    fmax = sample_rate / 2 
    
    mel_spec_power = librosa.feature.melspectrogram(x, sr=sample_rate, n_fft=n_fft, 
                                                    hop_length=hop_length, 
                                                    n_mels=n_mels, power=2.0, 
                                                    fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
    dst_fname = dst_path / (fname[:-4] + '.png')
    plt.imsave(dst_fname, mel_spec_db)


# In[ ]:


log_mel_spec_tfm(fn, AUDIO_DIR, IMG_DIR)
img = plt.imread(str(IMG_DIR/(fn[:-4] + '.png')))
plt.imshow(img, origin='lower');


# In[ ]:


transform_path(AUDIO_DIR, IMG_DIR, log_mel_spec_tfm, fnames=fnames, delete=True)


# In[ ]:


os.listdir(str(IMG_DIR))[:10]


# ## Image Classifier

# In[ ]:


import fastai
fastai.__version__


# In[ ]:


from fastai.vision import *


# In[ ]:


digit_pattern = r'(\d+)_\w+_\d+.png$'


# In[ ]:


data = (ImageList.from_folder(IMG_DIR)
        .split_by_rand_pct(.2)
        .label_from_re(digit_pattern)
        .transform(size=(128,64))
        .databunch())
data.c, data.classes


# In[ ]:


xs, ys = data.one_batch()
xs.shape, ys.shape


# In[ ]:


xs.min(), xs.max(), xs.mean(), xs.std()


# In[ ]:


data.show_batch(4, figsize=(5,9), hide_axis=False)


# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(4)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(10, 10), dpi=60)


# In[ ]:


# Clean up
get_ipython().system('rm -rf {AUDIO_DIR}')
get_ipython().system('rm -rf {IMG_DIR}')


# In[ ]:




