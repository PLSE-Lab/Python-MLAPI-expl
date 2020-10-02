#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install --upgrade transformers')
get_ipython().system('pip install simpletransformers')
# memory footprint support libraries/code
get_ipython().system('ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi')
get_ipython().system('pip install gputil')
get_ipython().system('pip install psutil')
get_ipython().system('pip install humanize')


# In[ ]:


import psutil
import humanize
import os
import GPUtil as GPU

GPUs = GPU.getGPUs()
gpu = GPUs[0]
def printm():
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
printm()


# In[ ]:


import numpy as np
import pandas as pd
#from google.colab import files
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
import gc
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import sklearn
from sklearn.metrics import log_loss
from sklearn.metrics import *
from sklearn.model_selection import *
import re
import random
import torch
pd.options.display.max_colwidth = 200

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

seed_all(2)


# In[ ]:


train=pd.read_csv('/kaggle/input/train-data/Train_BNBR.csv')
test=pd.read_csv('/kaggle/input/health-data/Test_health.csv')
sub = pd.read_csv('/kaggle/input/health-data/ss_health.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


train['label'] = train['label'].astype('category')  


# In[ ]:


train['label'] = train['label'].cat.codes
train.head()


# In[ ]:


train.label.value_counts()


# In[ ]:


print(train['text'].apply(lambda x: len(x.split())).describe())


# In[ ]:


print(test['text'].astype(str).apply(lambda x: len(x)).describe())


# In[ ]:


train1=train.drop(['ID'],axis=1)
test1=test.drop(['ID'],axis=1)
test1['target']=0


# In[ ]:


import math


# In[ ]:


#%%writefile setup.sh

#git clone https://github.com/NVIDIA/apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex


# In[ ]:


#!sh setup.sh


# # Roberta Large Model 1

# In[ ]:


get_ipython().run_cell_magic('time', '', 'err=[]\ny_pred_tot=[]\n\nfold=StratifiedKFold(n_splits=20, shuffle=True, random_state= 2)\ni=1\nfor train_index, test_index in fold.split(train1,train1[\'label\']):\n    train1_trn, train1_val = train1.iloc[train_index], train1.iloc[test_index]\n    model = ClassificationModel(\'roberta\', \'roberta-base\', use_cuda= False,num_labels= 4, args={\'train_batch_size\':32,\n                                                                         \'reprocess_input_data\': True,\n                                                                         \'overwrite_output_dir\': True,\n                                                                         "fp16": False,\n                                                                         #"fp16_opt_level": "O2",\n                                                                         \'do_lower_case\': True,\n                                                                         \'num_train_epochs\': 4,\n                                                                         \'max_seq_length\': 128,\n                                                                         #"adam_epsilon": 1e-4,\n                                                                         #"warmup_ratio": 0.06,\n                                                                         \'regression\': False,\n                                                                         #\'lr_rate_decay\': 0.4,\n                                                                         \'manual_seed\': 2,\n                                                                         "learning_rate":1e-4,\n                                                                         \'weight_decay\':0,\n                                                                         "save_eval_checkpoints": False,\n                                                                         "save_model_every_epoch": False,\n                                                                         "silent": True})\n    model.train_model(train1_trn)\n    raw_outputs_val = model.eval_model(train1_val)[1]\n    raw_outputs_val = softmax(raw_outputs_val,axis=1)   #[:,1]  \n    print(f"Log_Loss: {log_loss(train1_val[\'label\'], raw_outputs_val)}")\n    err.append(log_loss(train1_val[\'label\'], raw_outputs_val)) \n    raw_outputs_test = model.eval_model(test1)[1]\n    raw_outputs_test = softmax(raw_outputs_test,axis=1)  #[:,1]  \n    y_pred_tot.append(raw_outputs_test)\nprint("Mean LogLoss: ",np.mean(err))')


# In[ ]:


final=np.mean(y_pred_tot, 0)
final= pd.DataFrame(final)
final.head()


# In[ ]:


# Current best Mean LogLoss:  0.3973383729609846 (20fold_rbb_1_1e4_wd_0_32_128_epoch4_true), LB = 0.3726


# In[ ]:


final.to_csv('20fold_rbb_1_1e4_wd_0_32_128_epoch4_clean_data.csv',index=False)


# In[ ]:


#files.download("20fold_rbb_1_1e4_wd_0_32_40_epoch4_clean_data.csv")

