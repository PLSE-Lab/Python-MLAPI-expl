#!/usr/bin/env python
# coding: utf-8

# # ffmpeg-python example to extract audio from mp4 video
# 
# Documentation of ffmpeg-python --> [ffmpeg-python in Github](https://github.com/kkroening/ffmpeg-python)
# 
# I hope the code was self descriptive.
# 

# In[ ]:


get_ipython().system('tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz')
get_ipython().system('mkdir -p /tmp/pip/cache/')
get_ipython().system('cp ../input/ffmpegpython/ffmpeg_python-0.2.0-py3-none-any.whl /tmp/pip/cache/')
get_ipython().system('pip install --no-index --find-links /tmp/pip/cache/ ffmpeg_python')


# In[ ]:


import pandas as pd

path = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
df = pd.read_json(path + '/metadata.json')
df = df.T
df['filename'] = df.index


# In[ ]:


import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


def melspectrogram(audio, sr=44100, n_mels=128):
    return librosa.amplitude_to_db(librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels))

def show_melspectrogram(mel, sr=44100):
    plt.figure(figsize=(14,4))
    librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Log mel spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()


# In[ ]:


import numpy as np
import ffmpeg
from ffmpeg import Error

class ffmpegProcessor:
    def __init__(self):
        self.cmd = 'ffmpeg-git-20191209-amd64-static/ffmpeg'
        
    def extract_audio(self, filename):
        try:
            out, err = (
                ffmpeg
                .input(filename)
                .output('-', format='f32le', acodec='pcm_f32le', ac=1, ar='44100')
                .run(cmd=self.cmd, capture_stdout=True, capture_stderr=True)
            )
        except Error as err:
            print(err.stderr)
            raise
        
        return np.frombuffer(out, np.float32)


# In[ ]:


ap = ffmpegProcessor()

sample = df.sample(4)

for index, row in sample.iterrows():
    audio = ap.extract_audio(path + row.filename)
    show_melspectrogram(melspectrogram(audio))

