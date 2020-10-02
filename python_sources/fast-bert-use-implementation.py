#!/usr/bin/env python
# coding: utf-8

# ## How to make your inference faster

# This notebook shows how your inference can be faster.
# 
# The idea is to use the shortest max_len you can (instead of the standart 512)
# 
# To do it we re-order all sentences by their token length, each batch get the minimal max_len it can, and then sent to the model.
# 
# Later we reorder the outputs to the original order.
# 
# This kernel is for demonstration only. It is not a full competition solution

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
er=1


# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')


# In[ ]:


import sys
sys.path.insert(0, "../input/transformers/transformers-master/")
import warnings
warnings.filterwarnings(action='once')
import transformers
from collections import defaultdict
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pickle
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from functools import partial
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import os
pd.set_option('max_columns', 1000)
from tqdm import notebook


# The magic is in FastTokenIter it is an itaratore that is somewhat like DataLoader.
# 
# Just remember to reorder everything at the end using .reorder attribute
# 
# "fetch_vectors_full" is an example how to use it

# In[ ]:


def get_model_device(model):
    if not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        device_num = next(model.parameters()).get_device()
        if device_num<0:
            return torch.device('cpu')
        else:
            return torch.device("cuda:{}".format(device_num))

class FastTokenIter(D.Dataset):
    def __init__(self, ds,max_len=512, batch_size=16,shuffle = False,return_order=False):
        self.ds = ds
        self.max_len=max_len
        self.batch_size=batch_size
        self.num_items = ds.__len__()
        self.len=int(np.ceil(float(self.num_items)/self.batch_size))
        list_items=[ds.__getitem__(i) for i in notebook.tqdm(range(ds.__len__()) ,leave=False)]
        self.items=[torch.cat([item[i][None] for item in list_items]) for i in range(len(list_items[0]))]
        self.item_len=self.items[1].sum(1)
        self.item_order = np.argsort(self.item_len.numpy())
        self.reorder=np.argsort(self.item_order)
        self.batch_order =np.arange(self.len)
        self.len_tuple=len(self.items)
        if shuffle:
            np.rand.shuffle(self.batch_order)
        self.return_order=return_order or shuffle
        self.idx=0
            
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx>=self.len:
            raise StopIteration
        sidx=self.batch_order[self.idx]
        self.idx+=1
        mlen=min(self.item_len[self.item_order[sidx*self.batch_size:(1+sidx)*self.batch_size]].max(),self.max_len)
        ret =tuple([self.items[i][self.item_order[sidx*self.batch_size:(1+sidx)*self.batch_size]][:,:mlen] for i in range(self.len_tuple)])
        return (self.item_order[sidx*self.batch_size:(1+sidx)*self.batch_size],)+ret if self.return_order else ret

def fetch_vectors_full(ds,model,batch_size=64,num_workers=8):
    device = get_model_device(model)
    fin_features=[]
    dl = FastTokenIter(ds, batch_size=batch_size, shuffle=False)
    _=model.eval()
    with torch.no_grad():
        for batch in notebook.tqdm(dl,total=dl.len,leave=False):
            fin_features.append(model( input_ids=batch[0].to(device), attention_mask=batch[1].to(device))[0][:, 0, :].detach().cpu().numpy())
    return np.vstack(fin_features)[dl.reorder]   

def fetch_vectors_full_slow(ds,model,batch_size=64,num_workers=8):
    device = get_model_device(model)
    fin_features=[]
    dl = D.DataLoader(ds, batch_size=batch_size, shuffle=False)
    _=model.eval()
    with torch.no_grad():
        for batch in notebook.tqdm(dl,leave=False):
            fin_features.append(model( input_ids=batch[0].to(device), attention_mask=batch[1].to(device))[0][:, 0, :].detach().cpu().numpy())
    return np.vstack(fin_features)  

class TextDataset(D.Dataset):
    def __init__(self,text_list,tokenizer,max_len=512):
        self.text_list=text_list
        self.tokenizer = tokenizer
        self.max_len=max_len
    def __len__(self):
        return len(self.text_list)
    def __getitem__(self,idx):
        token_ids=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.text_list[idx]))[:self.max_len-2]
        token_ids = [self.tokenizer.cls_token_id]+token_ids+[self.tokenizer.sep_token_id]
        token_ids_tensor=torch.zeros(self.max_len,dtype=torch.long)
        mask_tensor=torch.zeros(self.max_len,dtype=torch.long)
        token_type_tensor=torch.zeros(self.max_len,dtype=torch.long)
        token_ids[:len(token_ids)]=token_ids
        mask_tensor[:len(token_ids)]=1
        return tuple((token_type_tensor,mask_tensor,token_type_tensor))


# In[ ]:


device='cuda'


# In[ ]:


test = pd.read_csv('../input/google-quest-challenge/test.csv').fillna(' ')
sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
er=2*er+1


# In[ ]:


tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
model.to(device)


# ## Banchmark 

# In[ ]:


question_ds= TextDataset(test.question_body.to_list(),tokenizer,512)
get_ipython().run_line_magic('time', 'question_features=fetch_vectors_full(question_ds,model,batch_size=16)')
get_ipython().run_line_magic('time', 'question_features=fetch_vectors_full_slow(question_ds,model,batch_size=16)')


# In[ ]:





# In[ ]:




