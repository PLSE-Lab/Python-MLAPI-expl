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


# In[ ]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
# from fastai.model_selection import *
from fastai.metrics import error_rate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[ ]:


PATH = "../input/Kannada-MNIST/"


# In[ ]:


# pd.read_csv(PATH+"train.csv").head(100)


# In[ ]:


train = pd.read_csv(PATH+"train.csv")
train.head()#, len(train)


# In[ ]:


test = pd.read_csv(PATH+"test.csv")
test.head()# len(test)


# In[ ]:


sample = pd.read_csv(PATH+"sample_submission.csv")


# In[ ]:


#train images and labels
image = train.iloc[:,1:]
label = train.iloc[:,0:1]


# In[ ]:


#test images and labels
test_image = test.iloc[:,1:]


# In[ ]:


test_id = test.iloc[:,0:1]
# test_id


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


# In[ ]:


train['file_name'] = np.NaN


# In[ ]:


get_ipython().system('mkdir train')
get_ipython().system('mkdir test')
get_ipython().system('ls train')


# In[ ]:


#converting all pixels to image and saving it to to train folder
for i in tqdm(range(int(len(image)))):
    code = label.iloc[i,:].astype('int')[0]
    serial = i
    save_path = f'train/{serial}_train_{code}.png'
    train.iloc[i,785] = f'{serial}_train_{code}.png'
    
    temp_image = Image.fromarray(image.iloc[i,:].values.astype('uint8').reshape(28,28))
    temp_image.save(save_path)
    
    


# In[ ]:


test['file_name'] = np.NaN


# In[ ]:


#converting all pixels to image and saving it to to test folder
for i in tqdm(range(int(len(test_image)))):
    serial = i
    save_path = f'test/{serial}_test.png'
    test.iloc[i,785] = f'{serial}_test.png'
    
    temp_image = Image.fromarray(test_image.iloc[i,:].values.astype('uint8').reshape(28,28))
    temp_image.save(save_path)


# In[ ]:


train_dict = {'name':train['file_name'] , 'label': train['label']}
df = pd.DataFrame(train_dict)
df.head()


# In[ ]:


test_dict = {'name':test['file_name']}
df_test = pd.DataFrame(test_dict)
df_test.head()


# In[ ]:


tfms = get_transforms(do_flip=False)


# In[ ]:


src = (ImageList.from_df(path='train', df=df)
        .split_by_rand_pct()
        .label_from_df(cols='label')
       )


# In[ ]:


data = (src.transform(tfms, size=28)
        .databunch(bs=64).normalize(imagenet_stats)
       )


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints')
get_ipython().system('cp /kaggle/input/fastai-pretrained-models/resnet50-19c8e357.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth')

learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir = Path('../kaggle/working'),path = Path("."))


# In[ ]:


# arch = models.resnet50
# # arch
# learn = cnn_learner(data, arch, metrics=[accuracy])


# In[ ]:


# learn


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 0.01
learn.fit_one_cycle(4, lr)


# In[ ]:


learn.save('kannada-mnist-stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-4, lr/5))


# In[ ]:


learn.save('kanada-mnist-stage-2')


# In[ ]:


learn.export(file="/kaggle/working/export.pkl")


# In[ ]:


get_ipython().system('ls')


# In[ ]:


# pd.read_csv(PATH+"sample_submission.csv").head()


# In[ ]:


# train = ImageList.from_df(path='train', df=df)
# learn = load_learner(path="/kaggle/working", test=train)
# preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


# output = preds.argmax(dim=1)
# output[1:100]


# In[ ]:


dataframes = []
test = ImageList.from_df(path='test', df=df_test)
learn = load_learner(path="/kaggle/working/", test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


output= preds.argmax(dim=1)


# In[ ]:


# np.unique(output)
# len(test_id), len(output)


# In[ ]:


df_sub = test_id


# In[ ]:


df_sub['label'] = pd.Series(output)


# In[ ]:


get_ipython().system('rm -rf train')


# In[ ]:


get_ipython().system('rm -rf test')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


# df_sub.to_csv("submission.csv", index=False)


# In[ ]:


# submission = pd.DataFrame({ 'id': Id,
#                             'label': predictions })
df_sub.to_csv(path_or_buf ="submission.csv", index=False)


# In[ ]:


# pd.read_csv("submission.csv").head()


# In[ ]:


get_ipython().system('ls')


# In[ ]:




