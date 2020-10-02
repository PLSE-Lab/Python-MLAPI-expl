#!/usr/bin/env python
# coding: utf-8

# ### Copying scripts necessary into our working directory

# In[ ]:


import os
from shutil import copyfile as cp

SCRIPTS = ['data.py', 'config.py', 'models.py']
for f in SCRIPTS:
    cp(os.path.join('../input/myinput', f), f)
print('Scripts loaded!')


# ### Taking care of dependencies

# In[ ]:


print(' > Installing requirements...')
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install torch')
get_ipython().system('pip install textgrid')
get_ipython().system('apt-get install -y libsndfile-dev')
get_ipython().system('pip install soundfile')

print('\033[1;32mDone!\033[0m')


# ### Importing libraries

# In[ ]:


print(' > Importing...', end='')
import os
import data
import torch
import models
import pandas as pd
import soundfile as sf
from config import *
from tqdm import tqdm
print('\033[1;32mdone!\033[0m')


# ### Function definitions

# In[ ]:


def predict(wav):
    signal, _ = sf.read(wav)
    signal = torch.tensor(signal, device=device).float().unsqueeze(0)
    label = model.decode_intents(signal)
    return label

def set_label(category, intents):
    category = intents.loc[intents.intent == category]
    return UNSURE if category.empty else category.category.item()
print('Well defined!')


# ### Setting and reading configs

# In[ ]:


UNSURE = 31
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config = data.read_config('../input/myinput/no_unfreezing/no_unfreezing.cfg'); _,_,_=data.get_SLU_datasets(config)
model = models.Model(config).eval()
model.load_state_dict(torch.load('../input/myinput/no_unfreezing/model_state.pth', map_location=device)) # load trained model


# ### Predicting labels for each test set command, using a pre-trained model

# In[ ]:


TEST = '../input/myinput/test.csv'
SPEAKERS = '../input/myinput/speakers'
test = pd.read_csv(TEST)
df, paths = list(), list()
files = set(test['file'].apply(lambda f: f.replace('.png', '.wav')))
for i, speaker in enumerate(os.listdir(SPEAKERS)):
    speaker = os.path.join(SPEAKERS, speaker)
    for wav in os.listdir(speaker):
        if wav not in files:
            continue
        wav = os.path.join(speaker, wav)
        paths.append(wav)

df = pd.DataFrame({'file': paths})
tqdm.pandas(desc='Predicting command labels')
df['category'] = df['file'].progress_apply(lambda f: predict(f))

df = pd.DataFrame(df, columns=['file', 'category'])
df['category'] = df['category'].apply(lambda l: ','.join(l[0]))


# ### Map from intent lists to category IDs
# For example  `['activate', 'light', 'kitchen']` is associated with some numeric id from the `intents.csv` file.

# In[ ]:


INTENTS = '../input/myinput/intents.csv'
intents = pd.read_csv(INTENTS)
tqdm.pandas(desc='Mapping intent to category ID', total=df.shape[0])
df['category'] = df['category'].progress_apply(lambda c: set_label(c, intents))


# ### Mapping full .wav paths to relative .png paths in the test set
# 
# For example, the following row of `df` dataframe,
# 
# | file                                                                                	| category 	|
# |-------------------------------------------------------------------------------------	|----------	|
# | ../input/myinput/speakers/R3mXwwoaX9IoRVKe/b034cf00-454d-11e9-aa52-bf2189a03a60.wav 	| 25       	|
# 
# > 
# 
# would lead to the following update in `test` dataframe
# 
# | file                                      | category 	|
# |------------------------------------------	|----------	|
# | b034cf00-454d-11e9-aa52-bf2189a03a60.png 	| 25       	|
# 
# 

# In[ ]:


df['file'] = df['file'].apply(lambda file: os.path.basename(file).replace('.wav', '.png'))
tqdm.pandas(desc='Mapping files to category ID', total=test.shape[0])
test['category'] = test['file'].progress_apply(lambda file: df.loc[df.file == file]['category'].item())


# ### Convert to the appropriate format & submit

# In[ ]:


test['file'] = range(1, test['file'].shape[0] + 1)
test = test.rename(columns={'file': 'id'})

SUBMISSION = 'submission.csv'
test.to_csv(SUBMISSION, index=False)
print('Submission ready!!!')


# ### Evaluate comparing to the ground truth
# 
# `output/1.csv` contains the ground truth for all 3530 commands of `test.csv`. We will use that to measure the accuracy of the model.

# In[ ]:


# load our predictions as well as the ground truth and sort by id
LABELS = '../input/myinput/1.csv'
sub = pd.read_csv(SUBMISSION).sort_values(by='id')['category']
labels = pd.read_csv(LABELS).sort_values(by='id')['category']

# compare
correct = (sub == labels).sum()
total = labels.shape[0]
print(f'\033[1;32mAccuracy\033[0m: {correct/total:.6f}')


# ### Clean up

# In[ ]:


for f in SCRIPTS + [SUBMISSION]:
    os.remove(f)
print('All clean!')

