#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning GPT-2 model with Kaggle Lyrics data, generating some "Lyrics" from the trained model
# 
# This kernel/notebook takes a GPT-2 model, finetunes it with the Kaggle Lyrics dataset, and generates some text from the finetuned model.
# 
# Some base code/information:
# - https://github.com/nshepperd/gpt-2
# 
# - https://medium.com/@ngwaifoong92/beginners-guide-to-retrain-gpt-2-117m-to-generate-custom-text-content-8bb5363d8b7f
# 
# My fork of the GPT-2, allowing the type of training and also use of saved models:
# https://github.com/mukatee/gpt-2.git
# 
# ### most of the output this far will likely not print due to Kaggle kernel console output size limitations, so just look into the "output" of this kernel and the files "samples-XXXXX" for examples of generated text
# 

# In[ ]:




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('tail /kaggle/input/gtp-2-finetuning-lyrics/checkpoint/summary_110000.csv')


# In[ ]:


get_ipython().system('git clone https://github.com/mukatee/gpt-2.git')


# In[ ]:


get_ipython().system('ls -a gpt-2')


# In[ ]:


get_ipython().system('rm -rf ./gpt-2/.git')
get_ipython().system('rm -rf ./gpt-2/.gitattributes')
get_ipython().system('rm -rf ./gpt-2/.gitignore')


# In[ ]:


import tqdm


# In[ ]:


get_ipython().system('pip list | grep tqdm')


# In[ ]:


get_ipython().system('pip list | grep requests')


# In[ ]:


get_ipython().system('pip list | grep regex')


# In[ ]:


get_ipython().system('pip list | grep fire')


# In[ ]:


get_ipython().system('pip install fire')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls gpt-2')


# In[ ]:


get_ipython().system('python gpt-2/download_model.py 117M')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('ls models/117M')


# In[ ]:


get_ipython().system('ls gpt-2')


# In[ ]:


lyrics = pd.read_csv("/kaggle/input/every-song-you-have-heard-almost/Lyrics2.csv", error_bad_lines=False)
lyrics.head()


# Print a few lyrics just to see what it looks like:

# In[ ]:


for index, row in lyrics.iterrows():
    print(row['Band'], row['Lyrics'])
    if index > 2:
        break


# In[ ]:


processed_input_folder = "texts"
os.mkdir(processed_input_folder)


# In[ ]:


lyrics["filename"] = lyrics['Song'].str.replace('&','_')
#lyrics["filename"] = lyrics['filename'].str.replace('\r','')
lyrics["filename"] = lyrics['filename'].str.replace(' ','_')
lyrics["filename"] = lyrics['filename'].str.replace('[','_')
lyrics["filename"] = lyrics['filename'].str.replace(']','_')
lyrics["filename"] = lyrics['filename'].str.replace(']','_')
lyrics["filename"] = lyrics['filename'].str.replace("'",'_')
lyrics.head()


# In[ ]:


lyrics["filename"] = lyrics["filename"].astype("str")
lyrics["filename"].describe()


# In[ ]:


measurer = np.vectorize(len)
#max filename length
res1 = measurer(lyrics["filename"].astype(str)).max(axis=0)
res1


# In[ ]:


lyrics.dropna(subset=['Lyrics'], inplace=True)
lyrics.shape


# In[ ]:


from tqdm.auto import tqdm
#tqdm.pandas()

FILES_TO_READ = 20000*1000
#FILES_TO_READ = 2*1000
MAX_FILENAME_LEN = 100

for index, row in tqdm(lyrics.iterrows()):
    filename = row["filename"].replace("/", "_")
    if len(filename) > MAX_FILENAME_LEN:
        continue
    if index > FILES_TO_READ:
        #will run out of room on kernel disk space
        break
    filename = f"{filename}.txt"
    with open(processed_input_folder+f"/{filename}", "w") as f:
        definition = row["Lyrics"]
        #print(definition)
        f.write(definition)


# In[ ]:


get_ipython().system('ls')


# In[ ]:





# In[ ]:


get_ipython().system('ls gpt-2')


# In[ ]:


import os
old_python_path = os.environ["PYTHONPATH"]


# In[ ]:


os.environ["PYTHONPATH"] = old_python_path + ":/kaggle/working/gpt-2/src"
os.environ["PYTHONPATH"]


# In[ ]:





# In[ ]:


get_ipython().system('python gpt-2/encode.py texts texts.npz')


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('pip install toposort')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('python gpt-2/train.py --save_every 25000 --quiet 25000 --sample_every 25000 --sample_num 5 --max_time 30000 --max_batches 200000 --sample_length 300 --dataset texts.npz')


# In[ ]:


get_ipython().system('ls gpt-2')


# In[ ]:


get_ipython().system('ls checkpoint/')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system('rm -rf texts')


# # most of the output this far will likely not print due to Kaggle kernel console output size limitations, so just look into the "output" of this kernel and the files "samples-XXXXX" for examples of generated text

# In[ ]:




