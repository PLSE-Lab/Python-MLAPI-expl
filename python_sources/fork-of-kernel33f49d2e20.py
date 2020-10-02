#!/usr/bin/env python
# coding: utf-8

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
#!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/
get_ipython().system('pip install ../input/efficientnet-pytorch/EfficientNet-PyTorch-master')


# In[ ]:


import torch
from torch import nn
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc


# In[ ]:


BATCH_SIZE=512
TEST_FILE_PATH='/kaggle/input/bengaliai-cv19/test.csv'
DATA_DIR='/kaggle/input/bengaliai-cv19/'
MODEL_PATH='/kaggle/input/12345678/checkpoint-epoch50.pth'


# In[ ]:


class test_dataset(torch.utils.data.Dataset):
    
    def __init__(self,test_dir):
        
        self.dir=test_dir
        self.data=pd.read_parquet(self.dir)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,idx):
        data=(self.data.iloc[idx].to_numpy()[1:].reshape([1, 137, 236])/255).astype('float')
        data=torch.from_numpy(data).cuda().float()
        return {'data':data,'id':torch.tensor(idx).cuda()}


# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
import torch
from efficientnet_pytorch import EfficientNet


class efficientnet_b1(nn.Module):

    def __init__(self):
        super(efficientnet_b1, self).__init__()
        model = EfficientNet.from_name('efficientnet-b1')
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3))
        self.layer0 = model._bn0
        self.layer1 = model._blocks
        self.layer2 = model._conv_head
        self.layer3 = model._bn1
        self.layer4 = model._avg_pooling
        self.layer5 = model._dropout
        self.fc1 = torch.nn.Linear(1280, 168)
        self.fc2 = torch.nn.Linear(1280, 11)
        self.fc3 = torch.nn.Linear(1280, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer0(x)
        for layer in self.layer1:
            x=layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return [x1, x2, x3]


# In[ ]:


torch.cuda.empty_cache()
with torch.no_grad():
    model1=efficientnet_b1()
    model1=torch.nn.DataParallel(model1,device_ids=[0])
    model1.eval()
    model1.load_state_dict(torch.load('/kaggle/input/all-models/checkpoint-epoch23.pth'))
    


# In[ ]:


gc.collect()
test_data=['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet']
#test_file=pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
test_id_answer={}
grapheme_root_list=[]
vowel_diacritic_list=[]
consonant_diacritic_list=[]
with torch.no_grad():
    for data_dir in test_data:
        dataset=pd.read_parquet(DATA_DIR+data_dir).iloc[:,1:].to_numpy().reshape([-1,1,137,236])
        dataset=np.array_split(dataset, 50210//128)
        for data in dataset:
            input_data=torch.from_numpy(data/255).float().cuda()
            print(input_data.size())
            output=model1(input_data)
            grapheme_root_list.extend(np.argmax(output[0].cpu().detach().numpy(),axis=1))
            vowel_diacritic_list.extend(np.argmax(output[1].cpu().detach().numpy(),axis=1))
            consonant_diacritic_list.extend(np.argmax(output[2].cpu().detach().numpy(),axis=1))
            del input_data,output
            gc.collect()
            torch.cuda.empty_cache()
        del dataset
        gc.collect()


# In[ ]:


z=len(consonant_diacritic_list)
row_id=[]
target=[]
for i in range(z):
    row_id.append('Test_{}_consonant_diacritic'.format(i))
    row_id.append('Test_{}_grapheme_root'.format(i))
    row_id.append('Test_{}_vowel_diacritic'.format(i))
    target.append(consonant_diacritic_list[i])
    target.append(grapheme_root_list[i])
    target.append(vowel_diacritic_list[i])


# In[ ]:


final_file=pd.DataFrame({'row_id':row_id, 'target':target},columns = ['row_id','target'] )


# In[ ]:


final_file.to_csv('submission.csv',index=False)


# In[ ]:


final_file


# In[ ]:




