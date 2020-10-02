#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


train.diagnosis.value_counts()


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


from tqdm import tqdm
from PIL import Image


# In[ ]:


#the func is from https://www.kaggle.com/toshik/image-size-and-rate-of-new-whale
def get_size_list(targets, dir_target):

    result = list()

    for target in tqdm(targets):

        img = np.array(Image.open(os.path.join(dir_target, target+'.png')))
        result.append(str(img.shape))

    return result


# In[ ]:


train['size_info'] = get_size_list(train.id_code.tolist(), dir_target='../input/train_images')


# In[ ]:


train.size_info.value_counts()


# In[ ]:


test['size_info'] = get_size_list(test.id_code.tolist(), dir_target='../input/test_images')


# In[ ]:


test.size_info.value_counts()


# In[ ]:


train_size_info = train.size_info.value_counts().to_frame().reset_index()
test_size_info = test.size_info.value_counts().to_frame().reset_index()
size_info = train_size_info.merge(test_size_info,on='index',how='outer').fillna(0)
size_info.columns = ['size','size_train','size_test']


# In[ ]:


size_info


# In[ ]:


tmp = train.groupby(['size_info','diagnosis']).size().unstack().fillna(0).reset_index()
train_size_info.columns = ['size_info','size']
tmp = tmp.merge(train_size_info,on='size_info',how='left')
tmp
# it seem that the size contain some information


# In[ ]:


for i in range(5):
    tmp.iloc[:,i+1] = tmp.iloc[:,i+1]/tmp['size']
tmp


# In[ ]:


train.diagnosis.value_counts().sort_index()/3662

