#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import imageio


# In[ ]:


path = Path('../input/Kannada-MNIST')
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
test  =pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


train.head()


# **Preprocessing**

# In[ ]:


def to_img_shape(data_X, data_y=[]):
    data_X = np.array(data_X).reshape(-1,28,28)
    data_X = np.stack((data_X,)*3, axis=-1)
    data_y = np.array(data_y)
    return data_X,data_y


# In[ ]:


data_X, data_y = train.loc[:,'pixel0':'pixel783'], train['label']

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, test_size=0.01,random_state=7,stratify=data_y)


# In[ ]:


train_X,train_y = to_img_shape(train_X, train_y)
val_X,val_y = to_img_shape(val_X,val_y)


# In[ ]:


def save_imgs(path:Path, data, labels):
    path.mkdir(parents=True,exist_ok=True)
    for label in np.unique(labels):
        (path/str(label)).mkdir(parents=True,exist_ok=True)
    for i in range(len(data)):
        if(len(labels)!=0):
            imageio.imsave( str( path/str(labels[i])/(str(i)+'.jpg') ), data[i] )
        else:
            imageio.imsave( str( path/(str(i)+'.jpg') ), data[i] )

save_imgs(Path('/data/train'),train_X,train_y)
save_imgs(Path('/data/valid'),val_X,val_y)


# **Making the Learner**

# In[ ]:


tfms = get_transforms(do_flip=False )

data = (ImageList.from_folder('/data/') 
        .split_by_folder()          
        .label_from_folder()        
        .add_test_folder()          
        .transform(tfms, size=64)   
        .databunch())


# In[ ]:


data.show_batch(3,figsize=(6,6))


# In[ ]:


get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints')
get_ipython().system('cp /kaggle/input/fastai-pretrained-models/resnet50-19c8e357.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth')

learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir = Path('../kaggle/working'),path = Path("."))


# **Fitting the model**

# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = slice(2e-05)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5,lr)


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.fit_one_cycle(10,lr)


# In[ ]:


learn.save('stage-3')


# In[ ]:


learn.load('stage-3')


# **Getting the Predictions**

# In[ ]:


test_csv = pd.read_csv('../input/Kannada-MNIST/test.csv')
test_csv.drop('id',axis = 'columns',inplace = True)
sub_df = pd.DataFrame(columns=['id','label'])


# In[ ]:


test_data = np.array(test_csv)


# In[ ]:


def get_img(data):
    t1 = data.reshape(28,28)/255
    t1 = np.stack([t1]*3,axis=0)
    img = Image(FloatTensor(t1))
    return img


# In[ ]:


from fastprogress import progress_bar


# In[ ]:


mb=progress_bar(range(test_data.shape[0]))
for i in mb:
    timg=test_data[i]
    img = get_img(timg)
    sub_df.loc[i]=[i+1,int(learn.predict(img)[1])]


# In[ ]:


def decr(ido):
    return ido-1
sub_df['id'] = sub_df['id'].map(decr)
sub_df.to_csv('submission.csv',index=False)

