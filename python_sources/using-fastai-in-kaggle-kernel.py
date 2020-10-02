#!/usr/bin/env python
# coding: utf-8

# # Important
# 
# **I have used only 16000 of the available 25000 images of the training data due to kaggle kernel limitations of Disk space. Please make the necessary changes in the code if you want to train on the complete dataset.**
# 
# **I realize this is not a well explained kernel, but I primarily wrote this to verify if it was possible to come up with an end-to-end solution using fastai inside kaggle kernels**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import glob
import shutil
import os
import gc
import pathlib
print(os.listdir("../input/"))
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


# In[ ]:


torch.cuda.is_available()


# In[ ]:


torch.backends.cudnn.enabled


# In[ ]:


dog_indexes = []
cat_indexes = []
count = 0
cat_count = 1
dog_count = 1


# **Indexes for Validation**

# In[ ]:


for name in os.listdir('../input/dogs-vs-cats-redux-kernels-edition/train/'):
    if 'cat' in name and cat_count <= 8000:
        cat_indexes.append(name.split('.')[1])
        cat_count += 1
    if 'dog' in name and dog_count <= 8000:
        dog_indexes.append(name.split('.')[1])
        dog_count += 1
    if dog_count > 8000 and cat_count > 8000:
        break


# In[ ]:


print ('Dog!\n',len(dog_indexes), '\nCat!\n', len(cat_indexes))


# **Randomly sampling indexes to add images to Validation set. Make changes to validation set if required.**

# In[ ]:


cat_val_list = random.sample(cat_indexes, 1200)
dog_val_list = random.sample(dog_indexes, 1200)


# **Folders to store the images in format that will allow to run fastai algorithms**

# In[ ]:


os.makedirs('../working/dogcats/valid/cats/')
os.makedirs('../working/dogcats/valid/dogs/')
os.makedirs('../working/dogcats/train/cats/')
os.makedirs('../working/dogcats/train/dogs/')
os.makedirs('../working/dogcats/test/')


# In[ ]:


train_dir = "../input/dogs-vs-cats-redux-kernels-edition/train/"
test_dir = "../input/dogs-vs-cats-redux-kernels-edition/test/"
cat_train_dir = "../working/dogcats/train/cats/"
cat_valid_dir = "../working/dogcats/valid/cats/"
dog_train_dir = "../working/dogcats/train/dogs/"
dog_valid_dir = "../working/dogcats/valid/dogs/"
dogcats_test = "../working/dogcats/test/"


# In[ ]:


PATH = "../working/dogcats/"
sz=224


# **Only moving 8000 images of each class due to size constraints. Make changes if required.**

# In[ ]:


for jpgfile in iglob(os.path.join(train_dir, "cat*.jpg")):
    if count >= 8000:
        break
    count += 1
    if jpgfile.split('.')[3] in cat_val_list:
        shutil.copy(jpgfile, cat_valid_dir)
    else:
        shutil.copy(jpgfile, cat_train_dir)

count = 0

for jpgfile in iglob(os.path.join(train_dir, "dog*.jpg")):
    if count >= 8000:
        break
    count += 1
    if jpgfile.split('.')[3] in dog_val_list:
        shutil.copy(jpgfile, dog_valid_dir)
    else:
        shutil.copy(jpgfile, dog_train_dir)
        


# In[ ]:


for jpgfile in iglob(os.path.join(test_dir, "*.jpg")):
    shutil.copy(jpgfile, dogcats_test)


# In[ ]:


cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# In[ ]:


get_ipython().system('cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth')


# In[ ]:


gc.collect()


# **Data Augmentation**

# In[ ]:


tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)


# **Transfer Learning**

# In[ ]:


arch=resnet34


# In[ ]:


data = ImageClassifierData.from_paths(PATH, tfms=tfms, test_name='test')
learn = ConvLearner.pretrained(arch, data, precompute=True)


# **Begin Training**

# In[ ]:


learn.fit(1e-2, 1)


# In[ ]:


learn.precompute=False


# In[ ]:


learn.fit(1e-2, 3, cycle_len=1)


# In[ ]:


learn.sched.plot_lr()


# In[ ]:


gc.collect()


# In[ ]:


learn.unfreeze()


# In[ ]:


lr=np.array([1e-4,1e-3,1e-2])


# In[ ]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.sched.plot_lr()


# In[ ]:


log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)


# In[ ]:


accuracy_np(probs, y)


# In[ ]:


preds = np.argmax(probs, axis=1)
probs = probs[:,1]


# **Prediction on Test Data**

# In[ ]:


temp = learn.predict(is_test=True)


# In[ ]:


temp.shape


# In[ ]:


pred_test = np.argmax(temp, axis=1)


# In[ ]:


pred_test[:20]


# In[ ]:


probs = np.exp(temp[:,1])


# In[ ]:


probs[:10]


# In[ ]:


os.listdir(f'{PATH}test')[:4]


# **Creation of submission CSV**

# In[ ]:


submission = pd.DataFrame({'id':os.listdir(f'{PATH}test'), 'label':probs})


# In[ ]:


get_ipython().system(' rm -rf ../working/dogcats/')


# In[ ]:


submission['id'] = submission['id'].map(lambda x: x.split('.')[0])


# In[ ]:


submission['id'] = submission['id'].astype(int)


# In[ ]:


submission = submission.sort_values('id')


# In[ ]:


submission.to_csv('../working/output.csv', index=False)

