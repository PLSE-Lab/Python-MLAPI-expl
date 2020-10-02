#!/usr/bin/env python
# coding: utf-8

# # Using GPT-2 to generate The Office Scenes
# 
# To use GPT-2 the quickest way is through the [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple) library
# 
# However, this requrires a lower version of tensorflow which is installed first and then a lower version of tensorflow GPU to get the Kaggle notebook to recognize the GPU.
# 
# We start by downgrading tensorflow and tensorflow-gpu

# In[ ]:


get_ipython().system('pip install -U tensorflow==1.15.0')


# In[ ]:


get_ipython().system('conda install tensorflow-gpu==1.14.0 -y')


# Before proceeding do a 'kernel restart' once both installations are done
# 
# > CMD-Shift-P > 'Confirm Kernal Restart'

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


import tensorflow as tf
tf.__version__


# In[ ]:


tf.test.is_gpu_available()


# In[ ]:


get_ipython().system('pip3 install gpt-2-simple')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gpt_2_simple as gpt2 # For GPT2
import os # For file
import re # For text cleanup


# In[ ]:


model_name = "124M"

# Download model
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/


# In[ ]:


# This is to prevent lines from being truncated when saving to a file
pd.set_option("display.max_colwidth", 10000000)


# In[ ]:


df = pd.read_csv('../input/the-office-us-complete-dialoguetranscript/The-Office-Lines-V2.csv')

df.head(10)


# In[ ]:


data = ["--Scene Start--"]
scene = 1

for index, row in df.iterrows():
    if scene != row['scene']:
        data.append("--Scene End--")
        data.append("")
        data.append("--Scene Start--")
        data.append(row['speaker'].strip() + ": " + row['line'].strip())
        scene += 1
    else:
        data.append(row['speaker'].strip() + ": " + row['line'].strip())

data.append("--Scene End--")
data.append("")
data.append("--Scene Start--")


# In[ ]:


data[0:8]


# In[ ]:


len(data)


# In[ ]:


# Saving all of the lines to a text file
with open('lines.txt', 'w') as filehandle:
    for listitem in data:
        filehandle.write('%s\n' % listitem)


# In[ ]:


office_sess = gpt2.start_tf_sess()


# In[ ]:


gpt2.finetune(office_sess,
              './lines.txt',
              model_name=model_name,
              steps=1000,
              print_every=100,
              sample_every=1000,
              save_every=500)   # steps is max number of training steps


# In[ ]:


# Generating a script based on an initial line

gpt2.generate(office_sess, length=250, temperature=0.8, prefix='Andy: This corona virus stuff is getting out of hand.')

