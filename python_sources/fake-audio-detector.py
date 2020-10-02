#!/usr/bin/env python
# coding: utf-8

# ## This notebook is a fork from below the excellent notebook with a little tweak
# 
# https://www.kaggle.com/rakibilly/extract-audio-starter
# 
# 
# ## Therefore please upvote the original one .

# ### Here we are trying to convert videos into wav file from a folder and then trying to find out if there are any audio fakes . We can replace the train_video_sample folder to any folder . Next version , will save the generated images and create dataset for CNN.

# In[ ]:


import numpy as np 
import pandas as pd 
import subprocess
import glob
import os
from pathlib import Path
import shutil
from zipfile import ZipFile
from scipy import signal
from scipy.io import wavfile
from skopt import gp_minimize
from skopt.space import Real
from functools import partial
import librosa.display
import librosa.filters
import matplotlib.pyplot as plt
import skimage


# Using the Static Build of ffmpeg from https://johnvansickle.com/ffmpeg/ because internet is not available. <br>
# The public data set can be found here:
# https://www.kaggle.com/rakibilly/ffmpeg-static-build
# 

# In[ ]:


get_ipython().system(' tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz')


# ### Specify output format and create a directory for the output Audio files
# For 400 mp3 files, the directory is approx 94 MB.<br>
# For 400 wav files, the directory is approx 673 MB.

# In[ ]:


DATA_FOLDER = '../input/deepfake-detection-challenge/'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos/'
TEST_FOLDER = 'test_videos/'
DATA_PATH = os.path.join(DATA_FOLDER,TRAIN_SAMPLE_FOLDER)
os.makedirs('/kaggle/working/output', exist_ok=True)
os.makedirs('/kaggle/working/test_output', exist_ok=True)
OUTPUT_PATH = '/kaggle/working/output'
TEST_OUTPUT_PATH = '/kaggle/working/test_output/'
print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")
print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")
SPLIT='00'


# In[ ]:


train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))
ext_dict = []
for file in train_list:
    file_ext = file.split('.')[1]
    if (file_ext not in ext_dict):
        ext_dict.append(file_ext)
print(f"Extensions: {ext_dict}")      


# In[ ]:


json_file = [file for file in train_list if  file.endswith('json')][0]
print(f"JSON file: {json_file}")


# In[ ]:


def get_meta_from_json(path):
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df

meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)
meta_train_df.head(20)


# In[ ]:


output_format = 'wav'  # can also use aac, wav, etc

output_dir = Path(f"{output_format}s")
Path(output_dir).mkdir(exist_ok=True, parents=True)
fake_name ='aaeflzzhvy'
real_name = 'flqgmnetsg'


# In[ ]:


meta = (list(meta_train_df.index))


# ### Get the list of videos to extract audio from

# In[ ]:


INPUT_PATH = '../input/realfake045/assorted/'
WAV_PATH = './wavs/'


# In[ ]:


list_of_files = []
for file in os.listdir(os.path.join(DATA_FOLDER,TRAIN_SAMPLE_FOLDER)):
    filename = os.path.join(DATA_FOLDER,TRAIN_SAMPLE_FOLDER)+file
    list_of_files.append(filename)


# ### Extract the audio from files

# In[ ]:


def create_wav(list_of_files):
    for file in list_of_files:
        command = f"../working/ffmpeg-git-20191209-amd64-static/ffmpeg -i {file} -ab 192000 -ac 2 -ar 44100 -vn {output_dir/file[-14:-4]}.{output_format}"
        subprocess.call(command, shell=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'create_wav(list_of_files)')


# In[ ]:


def create_spectogram(name,sr):
    audio_array, sample_rate = librosa.load(WAV_PATH+f'{name}', sr=sr)
    trim_audio_array, index = librosa.effects.trim(audio_array)
    S = librosa.feature.melspectrogram(y=trim_audio_array, sr=sr, n_mels=128, fmax=8000)
    S_dB = np.log(S + 1e-9)
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(S_dB, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    #S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB ,img

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


# In[ ]:


get_ipython().run_cell_magic('time', '', 'i=0\nsr=20000\nfor index,row in meta_train_df.iterrows():\n    if row.label == \'FAKE\':\n        if os.path.exists(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER,row.original)):\n              if os.path.exists(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER,index)):\n                    fake_name = index.split(\'.\')[0]+\'.wav\'\n                    real_name =row.original.split(\'.\')[0]+\'.wav\'\n                    S_fake,img_fake =create_spectogram(fake_name,sr)\n                    S_real,img_real =create_spectogram(real_name,sr)\n                    if not(np.array_equal(S_fake,S_real)):\n                        diff = np.sum(np.abs(S_real - S_fake))\n                        print(f"There is a difference in Audio : {diff}")\n                        plt.figure(figsize=(10, 4))\n                        plt.axis(\'off\')\n                        #librosa.display.specshow(S_fake, x_axis=\'time\',\n                        #          y_axis=\'mel\', sr=sr,\n                        #          fmax=8000)\n                        plt.imshow(img_fake,cmap=\'gray\')\n                        plt.colorbar(format=\'%+2.0f dB\')\n                        plt.title(f\'Mel-frequency spectrogram Fake name {fake_name}\')\n                        plt.tight_layout()\n                        plt.show()\n                        plt.figure(figsize=(10, 4))\n                        plt.axis(\'off\')\n                        #librosa.display.specshow(S_real, x_axis=\'time\',\n                        #          y_axis=\'mel\', sr=sr,\n                        #          fmax=8000)\n                        plt.imshow(img_real,cmap=\'gray\')\n                        plt.colorbar(format=\'%+2.0f dB\')\n                        plt.title(f\'Mel-frequency spectrogram Real name {real_name}\')\n                        plt.tight_layout()\n                        plt.show()\n            \n    i=i+1 ')


# ### Create ZIP file

# In[ ]:


with ZipFile(f'all_{output_format}s.zip', 'w') as zipObj:
   # Iterate over all the files in directory
   for folderName, subfolders, filenames in os.walk(f'./{output_format}s/'):
       for filename in filenames:
           #create complete filepath of file in directory
           filePath = os.path.join(folderName, filename)
           # Add file to zip
           zipObj.write(filePath)


# #### Cleanup

# In[ ]:


#Remove FFMPEG directory from output
shutil.rmtree("../working/ffmpeg-git-20191209-amd64-static")
#Remove directory of output files
shutil.rmtree(f'./{output_format}s/')


# In[ ]:




