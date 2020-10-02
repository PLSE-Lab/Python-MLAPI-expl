#!/usr/bin/env python
# coding: utf-8

# I believe it is many people (including me) first time to deal with multilabel classification. I spent some time struggling converting the label columns to the matrix, after which I found out a much simpler, easier and faster way for the conversion. 

# In[1]:


import time
script_start_time = time.time()

import pandas as pd
import numpy as np
import json

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
import warnings
warnings.filterwarnings('ignore')

data_path = "../input"


# ## 1. Load data from json

# In[3]:


# 1. Load data =================================================================
print('%0.2f min: Start loading data'%((time.time() - script_start_time)/60))

train={}
test={}
validation={}
with open('%s/train.json'%(data_path)) as json_data:
    train= json.load(json_data)
with open('%s/test.json'%(data_path)) as json_data:
    test= json.load(json_data)
with open('%s/validation.json'%(data_path)) as json_data:
    validation = json.load(json_data)

print('Train No. of images: %d'%(len(train['images'])))
print('Test No. of images: %d'%(len(test['images'])))
print('Validation No. of images: %d'%(len(validation['images'])))

# JSON TO PANDAS DATAFRAME
# train data
train_img_url=train['images']
train_img_url=pd.DataFrame(train_img_url)
train_ann=train['annotations']
train_ann=pd.DataFrame(train_ann)
train=pd.merge(train_img_url, train_ann, on='imageId', how='inner')

# test data
test=pd.DataFrame(test['images'])

# Validation Data
val_img_url=validation['images']
val_img_url=pd.DataFrame(val_img_url)
val_ann=validation['annotations']
val_ann=pd.DataFrame(val_ann)
validation=pd.merge(val_img_url, val_ann, on='imageId', how='inner')

datas = {'Train': train, 'Test': test, 'Validation': validation}
for data in datas.values():
    data['imageId'] = data['imageId'].astype(np.uint32)

print('%0.2f min: Finish loading data'%((time.time() - script_start_time)/60))
print('='*50)


# In[4]:


train.head()


# In[5]:


validation.head()


# In[6]:


test.head()


# ## 2. MultiLabelBinarizer
# As the labelId is a list, we need to convert them to single label in a matrix to feed out classifiers. Luckily, sklearns provide such as tool.

# In[14]:


print('%0.2f min: Start converting label'%((time.time() - script_start_time)/60))
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
train_label = mlb.fit_transform(train['labelId'])
validation_label = mlb.transform(validation['labelId'])
dummy_label_col = list(mlb.classes_)
print(dummy_label_col)
print('%0.2f min: Finish converting label'%((time.time() - script_start_time)/60))

for data in [validation_label, train_label, test]:
    print(data.shape)


# I recommend you to save it as numpy for faster loading and the column name in an empty csv file for reconversion later to when submitting.

# In[15]:


# Save as numpy
dummy_label_col = pd.DataFrame(columns = dummy_label_col)
# dummy_label_col.to_csv('%s/dummy_label_col.csv'%'', index = False)
# np.save('%s/dummy_label_train.npy' % '', train_label)
# np.save('%s/dummy_label_val.npy' % '', validation_label)
dummy_label_col.head()


# In[16]:


# Save as csv if you prefer
train_label = pd.DataFrame(data = train_label, columns = list(mlb.classes_))
train_label.head()
validation_label = pd.DataFrame(data = validation_label, columns = list(mlb.classes_))
validation_label.head()


# This is my previous hardworking but stupid way of converting. I am not regretting as I also learn something during the process. Just for your reference and a comparision with the previous method.

# In[17]:


# print('%0.2f min: Start converting validation'%((time.time() - script_start_time)/60))
# validation_label = validation[['labelId']]
# validation_label['labelId'] = validation_label['labelId'].apply(lambda labels: str([int(l) for l in labels]).replace('[','').replace(']', ''))
# validation_label = validation_label['labelId'].str.get_dummies(sep=', ')
# validation_label = validation_label.astype(np.uint8)
# print('%0.2f min: Finish converting validation'%((time.time() - script_start_time)/60))


# print('%0.2f min: Start converting train'%((time.time() - script_start_time)/60))
# train_label = train[['labelId']]
# train_label['labelId'] = train_label['labelId'].apply(lambda labels: str([int(l) for l in labels]).replace('[','').replace(']', ''))
# train_label = train_label['labelId'].str.get_dummies(sep=', ')
# train_label = train_label.astype(np.uint8)
# print('%0.2f min: Finish converting train'%((time.time() - script_start_time)/60))

# validation_missing_labels = set(list(train_label.columns)).difference(set(list(validation_label.columns)))

# print(validation_missing_labels)
# for l in validation_missing_labels:
#     validation_label[str(l)] = 0
#     # print(validation[str(l)].sum())
# validation_label = validation_label.astype(np.uint8)


# Hope this help you a bit.
