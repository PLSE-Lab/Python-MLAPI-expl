#!/usr/bin/env python
# coding: utf-8

# # This notebook enables the extraction of audio from the video files without internet.
# 
# If you find it useful, please upvote. Any suggestions to improve the code are welcome.
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import subprocess
import glob
import os
from pathlib import Path
import shutil
from zipfile import ZipFile


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


output_format = 'mp3'  # can also use aac, wav, etc

output_dir = Path(f"{output_format}s")
Path(output_dir).mkdir(exist_ok=True, parents=True)


# ### Get the list of videos to extract audio from

# In[ ]:


list_of_files = glob.glob('../input/deepfake-detection-challenge/train_sample_videos/*.mp4') 


# ### Extract the audio from files

# In[ ]:


for file in list_of_files:
    command = f"../working/ffmpeg-git-20191209-amd64-static/ffmpeg -i {file} -ab 192000 -ac 2 -ar 44100 -vn {output_dir/file[-14:-4]}.{output_format}"
    subprocess.call(command, shell=True)


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


# Remove FFMPEG directory from output
shutil.rmtree("../working/ffmpeg-git-20191209-amd64-static")
# Remove directory of output files
shutil.rmtree(f'./{output_format}s/')


# In[ ]:


os.listdir()


# In[ ]:




