#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Loading Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from pathlib import Path

from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate

import torchvision

import PIL

from sklearn.model_selection import train_test_split

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Loading path and reading in data

# In[ ]:


path = Path('/kaggle/input/digit-recognizer')


# In[ ]:


train = pd.read_csv(path/'train.csv')
test = pd.read_csv(path/'test.csv')


# In[ ]:


# Helper function
def convert_to_img(arr):
    px = arr.to_numpy()
    reshaped_pix = px.reshape(-1, 28, 28)
    reshaped_pix = np.stack((reshaped_pix, ) * 3, axis=1)
    tensor_pix = torch.tensor(reshaped_pix, dtype=torch.float) / 255
    
    images = [vision.image.Image(tensor) for tensor in tensor_pix]
    
    return images


# ## Initializing data and target

# In[ ]:


# Splitting data
data, target = convert_to_img(train.iloc[:, 1:]), train.iloc[:,0].to_numpy()
train_X, val_X, train_y, val_y = train_test_split(data, target, test_size=0.2)


# In[ ]:


# Custom PixelImageList class
class PixelImageList(ImageList):
    def get(self, i):
        img = self.items[i]
        self.sizes[i] = img.size
        return img


# In[ ]:


# LabelLists for training and valid
tx = PixelImageList(train_X)
ty = CategoryList(train_y, classes=list(map(str,range(10))))
vx = PixelImageList(val_X)
vy = CategoryList(val_y, classes=list(map(str, range(10))))

tll = LabelList(tx, ty)
vll = LabelList(vx, vy)

lls = LabelLists(path=".", train=tll, valid=vll)


# In[ ]:


# Test list
test_list = PixelImageList(convert_to_img(test))


# add test list to lls
lls.add_test(items=test_list)


# ## Prepare data for model

# First create a databunch

# In[ ]:


bs = 64


# In[ ]:


tfms = get_transforms(do_flip=False)
data = (ImageDataBunch.create_from_ll(lls, bs=bs, ds_tfms=tfms, num_workers=0, size=28)
                      .normalize(mnist_stats))


# In[ ]:


data.show_batch(3, figsize=(6, 6))


# ## Training model with ResNet50

# In[ ]:


learn2 = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn2.lr_find()
learn2.recorder.plot()


# In[ ]:


learn2.fit_one_cycle(4, 3e-3)


# In[ ]:


learn2.unfreeze()


# In[ ]:


learn2.lr_find()
learn2.recorder.plot()


# In[ ]:


learn2.fit_one_cycle(10, max_lr=slice(3e-6,3e-5))


# ## Interpretation from ClassificationInterpretation Object

# In[ ]:


interp50 = ClassificationInterpretation.from_learner(learn2)


# In[ ]:


interp50.plot_top_losses(12)


# In[ ]:


interp50.plot_confusion_matrix()


# In[ ]:


interp50.most_confused(min_val=2)


# ## Test Predictions

# In[ ]:


predictions = learn2.get_preds(ds_type=DatasetType.Test)


# In[ ]:


submission = pd.DataFrame({'ImageId': list(range(1, len(predictions[0]) + 1)), 'Label':predictions[0].argmax(1)})
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# ## Thank you.

# Much thanks to the kernel of [Pemtaira](https://www.kaggle.com/pemtaira/digit-recognizer-with-fast-ai-v3), [Nitron](https://www.kaggle.com/nitron/mnist-fastai), and [Chris Wallenwein](https://www.kaggle.com/christianwallenwein/beginners-guide-to-mnist-with-fast-ai).
# 
# Their notebooks were very helpful to figuring out `data_block` api, building custom `ItemList`, and the steps of completeing the kaggle competiton.
# 
# Wouldn't have gotten this far without them.
