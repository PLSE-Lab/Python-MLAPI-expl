#!/usr/bin/env python
# coding: utf-8

# ### Reference
# 
# Idea is from this notebook : https://www.kaggle.com/raghaw/panda-medium-resolution-dataset-25x256x256
# 
# Joblib Parallel form : https://www.youtube.com/watch?v=Ny3O4VpACkc&list=PL98nY_tJQXZnoCDfHLo58tRHUyNvrRVzn&index=4

# In[ ]:


import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import zipfile
import pandas as pd
import numpy as np
import shutil

from pydub import AudioSegment
from joblib import Parallel, delayed


# In[ ]:


get_ipython().system('mkdir -p /root/.kaggle/')
get_ipython().system('cp ../input/mykaggleapi/kaggle.json /root/.kaggle/')
get_ipython().system('chmod 600 /root/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('mkdir -p /tmp/birdcall_dataset')


# In[ ]:


get_ipython().system('ls /tmp')


# In[ ]:


data = '''{
  "title": "birdsong_recognition_wav_16000",
  "id": "gopidurgaprasad/birdsong-recognition-wav-16000",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
'''
text_file = open("/tmp/birdcall_dataset/dataset-metadata.json", 'w+')
n = text_file.write(data)
text_file.close()


# In[ ]:


TRAIN_CSV = "../input/birdsong-recognition/train.csv"
ROOT_PATH = "../input/birdsong-recognition/train_audio"
OUTPUT_DIR = "/tmp/birdcall_dataset/train_audio_wav_16000"


# In[ ]:


os.mkdir(OUTPUT_DIR)


# In[ ]:


def save_fn(bird_code, filename):
    
    path = f"{ROOT_PATH}/{bird_code}/{filename}"
    save_path = f"{OUTPUT_DIR}/{bird_code}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    if os.path.exists(path):
        try:
            sound = AudioSegment.from_mp3(path)
            sound = sound.set_frame_rate(16000)
            sound.export(f"{save_path}/{filename[:-4]}.wav", format="wav")
        except:
            print(path)


# In[ ]:


train = pd.read_csv(TRAIN_CSV)
bird_code_list = list(train.ebird_code.values)
filename_list = list(train.filename.values)


# In[ ]:


Parallel(n_jobs=8, backend="multiprocessing")(
    delayed(save_fn)(bird_code, filename) for bird_code, filename in tqdm(zip(bird_code_list, filename_list), total=len(bird_code_list))
)


# In[ ]:


get_ipython().system('sleep 10')


# In[ ]:


get_ipython().system('zip -r "/tmp/birdcall_dataset/train_audio_wav_16000.zip" "/tmp/birdcall_dataset/train_audio_wav_16000/"')


# In[ ]:


get_ipython().system('ls -l /tmp/birdcall_dataset/')


# In[ ]:


get_ipython().system('kaggle datasets create -p /tmp/birdcall_dataset')


# In[ ]:


get_ipython().system('rm -rf /tmp/birdcall_dataset')


# In[ ]:





# In[ ]:




