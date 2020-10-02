#!/usr/bin/env python
# coding: utf-8

# 
# ** PLEASE UPVOTE BEFORE YOU FORK **
# 
# This is an inference kernel for training a 5-fold model for the APTOS 2019 competition. We see ~0.045 increase in LB score.
# 
# Note: due to my mistake, I messed up the name of the dataset. It is actually 5 folds though the directory says 4 folds.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.metrics import cohen_kappa_score
def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.argmax(y_hat,1), y, weights='quadratic'),device='cuda:0')


# In[ ]:


from fastai.vision import *
from shutil import copyfile
num_folds = 5
df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')
test_dir = os.path.join(base_image_dir,'test_images/')
df['path'] = df['id_code'].map(lambda x: os.path.join(test_dir,'{}.png'.format(x)))
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe
df.head(10)

for i in range(num_folds):
    copyfile("../input/aptos-4fold-models/model" + str(i)+".pkl","../model" + str(i)+".pkl")
        
test = ImageList.from_df(df,Path('.'),cols='path')
predictions = torch.from_numpy(np.zeros((len(df))))
for i in range(num_folds):
    learn = load_learner('../',file = 'model'+str(i)+'.pkl',test=test)
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    predictions = predictions + preds.argmax(dim=-1).double()


# In[ ]:


predictions = torch.round(predictions/num_folds)
print(predictions)


# In[ ]:


df = df.drop(columns=['path'])
df.diagnosis = predictions.numpy().astype(int)
df.head(10)
df.to_csv('submission.csv',index=False)

