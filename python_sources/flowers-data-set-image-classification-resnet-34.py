#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.getcwd()
#os.chdir('/kaggle/input/flowers')
# Any results you write to the current directory are saved as output.


# In[ ]:


os.chdir('/kaggle')
os.getcwd()
os.listdir()


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# In[ ]:


classes = ['daisy','dandelion','rose','sunflower','tulip']


# In[ ]:


for c in classes:
    path = Path('/kaggle/working/flowers/')
    path_old=Path('/kaggle/input/flowers/flowers')
    dest = path/c
    dest.mkdir(parents=True, exist_ok=True)
    
    


# In[ ]:


os.chdir('/kaggle/working/flowers')
os.listdir()


# In[ ]:


os.listdir('/kaggle/input/flowers/flowers')


# In[ ]:


os.listdir('/kaggle/input/flowers/flowers/daisy')


# In[ ]:


file_list=os.listdir('/kaggle/input/flowers/flowers/daisy')


# In[ ]:


import shutil

# move the test images from the images directory to the test directory
for c in classes:
    print(c)
    path='/kaggle/input/flowers/flowers'
    file_list=os.listdir(str(path)+'/'+str(c))
    for i in file_list:
        shutil.copy('/kaggle/input/flowers/flowers/'+str(c)+'/'+str(i),'/kaggle/working/flowers/'+str(c))
    


# In[ ]:


p_daisy=Path('/kaggle/working/flowers/daisy')
verify_images(p_daisy, delete=True, max_size=500)

len(os.listdir('/kaggle/working/flowers/daisy'))


# In[ ]:


path=Path('/kaggle/working/flowers/')
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)


# In[ ]:


path=Path('/kaggle/working/flowers/')
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)


# In[ ]:


print("Total ",data.c," classes in data namely",data.classes)


# In[ ]:


data.show_batch(rows=3, figsize=(5,5))


# In[ ]:


doc(cnn_learner)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


#! mkdir -p /kaggle/working/output
#os.listdir('/kaggle/working/')


# In[ ]:


# import pandas as pd
# pd.DataFrame(os.listdir('/kaggle/working/flowers'), columns=['file_nm']).to_csv('/kaggle/working/file_name.csv')


# In[ ]:


#learn.save('stage-1')
#learn.export('/kaggle/working/output/stage-1.pkl')


# In[ ]:


# learn.save('stage-1')


# In[ ]:


# interp = ClassificationInterpretation.from_learner(learn)

# losses,idxs = interp.top_losses()

# len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


# interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


# interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


# interp.most_confused(min_val=2)


# In[ ]:


# learn.unfreeze()


# In[ ]:


# learn.fit_one_cycle(1)


# In[ ]:


# learn.load('stage-1');


# In[ ]:


# learn.lr_find()


# In[ ]:


# learn.recorder.plot()


# In[ ]:


# learn.unfreeze()
# learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))


# In[ ]:




