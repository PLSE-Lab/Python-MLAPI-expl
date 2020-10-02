#!/usr/bin/env python
# coding: utf-8

# **With Kaggle Kernels now GPU enabled it makes learning AI more democratic.For all those learners concerned with cost of GPU to learn AI this comes as an amazing news.FAST AI which is right now the best MOOC out there to learn AI coupled with kaggle kernels GPU enablement should be what the global AI learners community need right now.**

# Before we start we need to get fast ai as custom package .Go ti the packages options and  in the Github user repo use **"https://github.com/fastai/fastai"**.This should get fastai as one of the packages in your docker set up

# **First lets import all fastai needed tools**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# **Now we know that the train and test images are split across two folders lets  define them**

# In[ ]:


train='../input/dogs-vs-cats-redux-kernels-edition/train'
test='../input/dogs-vs-cats-redux-kernels-edition/test'


# **We will use a csv to get the Image data object so first lets get all the details of the training dataset on to a csv**

# In[ ]:


data = []
for files in sorted(os.listdir(train)):
    data.append((files))

df = pd.DataFrame(data, columns=['img'])
df['tag']=df['img'].astype(str).str[:3]
df['tag'].value_counts()


# **Lets store this dataframe as a csv somewhere**

# In[ ]:


get_ipython().system('mkdir ./data')
df.to_csv('./data/catdog.csv',index=False)


# **Lets get a validation sample that is 20% of all training data should be marked as validation which will be used to validate how our DL model looks**

# In[ ]:


label_csv = './data/catdog.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)


# In[ ]:


import gc
gc.collect()


# **Lets create a  data object using really cool fastai API**

# In[ ]:


PATH=''


# In[ ]:


arch=resnet34
sz=224
data = ImageClassifierData.from_csv(PATH,train,label_csv,tfms=tfms_from_model(arch, sz),test_name=test)


# **Now we don't have the model(Resnet34) we will use for transfer learning in this instance set up as fastai/pytorch expect it to be so lets do that**

# In[ ]:


#import pathlib
#data.path = pathlib.Path('.')
from os.path import expanduser, join, exists
from os import makedirs
cache_dir = expanduser(join('~', '.torch'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)


# In[ ]:


get_ipython().system('ls ~/.torch   #Now we have a .torch/models directory in the root folder ')


# In[ ]:


#Lets copy the weights to the folder

get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth')


# In[ ]:


#Change the true path of the model
import pathlib
data.Path = pathlib.Path('.')


# **Now we  pre-compute the weights of Resnet34 and fit the model**

# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(1e-2, 2)


# In[ ]:


#gc.collect()


# In[ ]:


#Two more epochs
learn.fit(0.01, 2)


# In[ ]:


#gc.collect()


# **Lets quickly look at the predictions before delving deeper into the details **

# In[ ]:


log_preds, y = learn.TTA(is_test=True)
probs = np.mean(np.exp(log_preds),0)
ds=pd.DataFrame(probs)
ds.columns=data.classes
ds.insert(0,'id',[o.rsplit('/', 1)[1] for o in data.test_ds.fnames])
subm=pd.DataFrame()
subm['id']=ds['id']
subm['cat']=ds['cat']


# In[ ]:


subm=pd.DataFrame()
subm['id']=ds['id']
subm['label']=ds['dog']


# In[ ]:


#subm.to_csv('../workingsubmission.csv',index=False)


# In[ ]:


#subm.to_csv('./data/submission.gz',compression='gzip',index=False)


# **Lets get the output as csv**

# In[ ]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

# create a link to download the dataframe
create_download_link(subm)


# In[ ]:


#!ls ./data


# In[ ]:


#subm


# In[ ]:




