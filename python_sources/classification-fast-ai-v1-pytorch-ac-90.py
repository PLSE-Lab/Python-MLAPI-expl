#!/usr/bin/env python
# coding: utf-8

# My first attempt at reusing the Fast ai with an image set.
# 
# Started with the original Fast ai library... had issues with bad images.
# I have cleaned up the dataset... but now my new problem is that fast ai has moved to 1.0...  and just trying t figure it out..
# 
# A strange learning rate graph which I would like to know more about..
# The accurate turned out about 90% and the confusion matrix looks good.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
# Added Resnet34 dataset


# In[ ]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This file contains all the main external libs we'll use
from fastai import *
from fastai.vision import *


# In[ ]:


PATH = "../input/art-images-drawings-painting-sculpture-engraving/art_dataset_cleaned/art_dataset/"
PATH_OLD = "../input/art-images-drawings-painting-sculpture-engraving/musemart/dataset_updated/"
TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model/"
sz=200


# In[ ]:


# GPU required
torch.cuda.is_available()


# In[ ]:


torch.backends.cudnn.enabled


# In[ ]:


os.listdir(PATH + 'training_set')


# In[ ]:


files = os.listdir(f'{PATH}training_set/engraving')[:5]
files


# In[ ]:


img = plt.imread(f'{PATH}training_set/engraving/{files[0]}')
plt.imshow(img);


# In[ ]:


img.shape


# In[ ]:


img[:4,:4]


# In[ ]:


# Fix to enable Resnet to live on Kaggle
cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


# In[ ]:


cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth


# In[ ]:


#arch=resnet34
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(PATH, train='training_set', valid='validation_set', ds_tfms=tfms, size=sz, num_workers=0)

#data = ImageClassifierData.from_paths(PATH, trn_name='training_set', val_name='validation_set', tfms=tfms_from_model(arch, sz))


# In[ ]:


# All the images that hit an OS error - Not needed with the cleaned data set
#imagesL = {}
#printIt = []

#for foldername in data.classes:
#    imagesL.update({foldername : os.listdir(PATH + 'training_set/' + foldername)})
#for key in imagesL:
#    for image in imagesL[key]:
#            try:
#                img = open_image(PATH + 'training_set/' + key + "/" + image) 
#            except OSError:
#                printIt.append("deleting " + PATH + 'training_set/' +  key + "/" + image + str(OSError))
#                #Read only data set so can't - os.remove(PATH + 'training_set/' + key + "/" + image, dir_fd=None)
#print(printIt)


# In[ ]:


data.show_batch(rows=3, figsize=(6,6))


# In[ ]:


#def simple_learner(): return Learner(data, simple_cnn((3,16,16,2)), metrics=[accuracy], model_dir=MODEL_PATH)
#learn = simple_learner()


# In[ ]:


#learn.lr_find(learn, start_lr=1e-6, end_lr=1e-1)


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics=accuracy, model_dir=MODEL_PATH)


# In[ ]:


learn.fit_one_cycle(4,1e-2)
learn.recorder.plot()


# This is as far as I got for now...

# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(data, preds, y, losses)
interp.plot_top_losses(9, figsize=(14,14))
## Showing the top losses -> worst predictions ##


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


## Showing the most confused preditions 
# - Engravings are often confused with drawings and vice versa
interp.most_confused(slice_size=10)

