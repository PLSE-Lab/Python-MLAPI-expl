#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from os.path import join as op
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image, ImageStat
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


PATH = '../input/'
df_train = pd.read_csv(PATH + 'train.csv')


# In[ ]:


print('Each image of the training set has maximum {} label'.format(df_train['Image'].value_counts().sort_values(ascending=False).max()))


# In[ ]:


plt.figure(figsize=(14,5))
plt.title('Number of images per Id')
df = df_train.groupby('Id').size().sort_values(ascending=False)
plt.plot([i for i in range(len(df))], df.values)
plt.yscale('log')
plt.xlabel('Number of Id')
plt.ylabel('Numbers of images per Id')
plt.show()


# In[ ]:


print("The Ids with more images are the following: \n \n",df.iloc[:10])


# # Color or Gray

# We can know if a photo is in gray if the three color channels are identical.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef grey_cv(row, dataset):\n    filename = op(PATH,dataset,row['Image'])\n    img = cv2.imread(filename)\n    if (img[:,:,0] == img[:,:,1]).all():\n        return img.shape[0], img.shape[1], True\n    else:\n        return img.shape[0], img.shape[1], False\n    \ndf_train['h'], df_train['w'], df_train['gray'] = zip(*df_train.apply(lambda row: grey_cv(row, 'train'), axis=1))")


# In[ ]:


df_train['gray'].value_counts()


# Half of the photos gray and half are in colors

# In[ ]:


df = df_train.groupby(['Id', 'gray', 'h', 'w']).size().sort_values(ascending=False).reset_index()
df['Number of images'] = df[0]
df.drop(0,axis=1, inplace=True)
df = df[df['Id'] != 'new_whale']


# In[ ]:


df.head(10)


# There are Id with several photos with the same conditions  (same size and in the same colors)

# In[ ]:


num_class = 10
fig, axr = plt.subplots(num_class,4, figsize=(15,30), sharex=True, sharey=True)
for i in range(num_class):
    df_tmp = df_train[(df.iloc[i:i+1]['h'].values[0] == df_train['h']) & (df.iloc[i:i+1]['w'].values[0] == df_train['w']) & (df.iloc[i:i+1]['gray'].values[0] == df_train['gray']) & (df.iloc[i:i+1]['Id'].values[0] == df_train['Id'])  ]
    for j, (_, row) in enumerate(df_tmp.iloc[:4].iterrows()):
        filename = op(PATH,'train',row['Image'])
        img = cv2.imread(filename)
        axr[i,j].set_ylabel(row['Id'])
        axr[i,j].imshow(img)
        axr[i,j].grid('off')


# # Predictions

# In[ ]:


get_ipython().run_cell_magic('time', '', "print('Find caracteristhics for the test images')\ndf_test = pd.DataFrame(os.listdir(PATH+'test'), columns=['Image'])\ndf_test['h'], df_test['w'], df_test['gray'] = zip(*df_test.apply(lambda row: grey_cv(row, 'test'), axis=1))")


# In[ ]:


def get_id(row):
    df_tmp = df[(df['h'] == row['h']) & (df['w'] == row['w']) & (df['gray'] == row['gray'])]
    mylist = list(set(df_tmp['Id']))
    if len(mylist) > 5:
        mylist = mylist[:5]
    if not df_tmp.shape[0]:
        mylist.append('new_whale')
    return ' '.join(mylist)

df_test['Id'] = df_test.apply(lambda row: get_id(row), axis=1)


# In[ ]:


num_class = 10
fig, axr = plt.subplots(num_class,4, figsize=(15,30), sharex=True, sharey=True)
for i in range(num_class):
    df_tmp = df_test[df_test['Id'].str.contains(df.iloc[i:i+1]['Id'].values[0])]
    for j, (_, row) in enumerate(df_tmp.iloc[:4].iterrows()):
        filename = op(PATH,'test',row['Image'])
        img = cv2.imread(filename)
        #axr[i,j].set_title('\n'.join(str(row['Id']).split(' ')))
        if len(str(row['Id'])) > 15:
            axr[i,j].set_title(str(row['Id'])[:15] + '- \n' + str(row['Id'])[15:], fontsize=8)
        else:
            axr[i,j].set_title(str(row['Id']))
        axr[i,0].set_ylabel(df.iloc[i:i+1]['Id'].values[0])
        axr[i,j].imshow(img)
        axr[i,j].grid('off')


# There are photos with similar conditions in the test set too. So we can add the same labels

# In[ ]:


df_test[['Image', 'Id']].to_csv('mysubmision.csv', index=False)

