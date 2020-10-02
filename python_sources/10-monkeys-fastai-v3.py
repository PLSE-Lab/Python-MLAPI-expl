#!/usr/bin/env python
# coding: utf-8

# # This code is from fastai and are used on the dataset 10 monkeys.

# Note: A few lines have been commented out since it causes "Commit Errors" in kaggle (The steps require manual correction of data which causes the kernel to stop running automatically). 
# Please uncomment the lines as you spot them

# # Creating your own dataset from 10 monkeys
# 
# *by: Francisco Ingham and Jeremy Howard. Inspired by [Adrian Rosebrock](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)*

# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
from glob import glob
import random
import cv2
import matplotlib.pylab as plt
import random as rand
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization, Input
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path
from keras.optimizers import Adam,RMSprop,SGD


# In[ ]:


df = pd.read_csv("../input/10-monkey-species/monkey_labels.txt")
df.head()


# In[ ]:


path = Path('../input/10-monkey-species')


# ## View data

# In[ ]:


(path/'').ls()


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(10,10))


# In[ ]:


learn = create_cnn(data, models.resnet34, metrics= accuracy, model_dir="/tmp/model/")   #metrics= error_rate


# In[ ]:


learn.fit_one_cycle(4) 


# In[ ]:


learn.save('stage-1') #save the model


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find() #finding best learning rate


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 0.01


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-2')


# ...And fine-tune the whole model:

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-4))


# In[ ]:


learn.save('stage-3')


# In[ ]:


#np.random.seed(42)
#src = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
#        ds_tfms=get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.), size=64, num_workers=0.1).normalize(imagenet_stats)


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=256, num_workers=0).normalize(imagenet_stats)

learn.data = data
data.train_ds[0][0].shape


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr=1e-2/2


# In[ ]:


learn.fit_one_cycle(5, slice(lr))


# In[ ]:


learn.save('stage-4')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5, lr/5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-5')


# ## Interpretation

# In[ ]:


learn.load('stage-3');  #load the model


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# ## Cleaning Up
# 
# Some of our top losses aren't due to bad performance by our model. There are images in our data set that shouldn't be.
# 
# Using the `ImageCleaner` widget from `fastai.widgets` we can prune our top losses, removing photos that don't belong.

# In[ ]:


#from fastai.widgets import *


# First we need to get the file paths from our top_losses. We can do this with `.from_toplosses`. We then feed the top losses indexes and corresponding dataset to `ImageCleaner`.
# 
# Notice that the widget will not delete images directly from disk but it will create a new csv file `cleaned.csv` from where you can create a new ImageDataBunch with the corrected labels to continue training your model.

# Note: Please Set the Number of images to a number that you'd like to view:
# ex: ```n_imgs=100```

# In[ ]:


#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)


# In[ ]:


#ImageCleaner(ds, idxs, path)


# Flag photos for deletion by clicking 'Delete'. Then click 'Next Batch' to delete flagged photos and keep the rest in that row. ImageCleaner will show you a new row of images until there are no more to show. In this case, the widget will show you images until there are none left from top_losses.ImageCleaner(ds, idxs)
# 
# You can also find duplicates in your dataset and delete them! To do this, you need to run .from_similars to get the potential duplicates' ids and then run ImageCleaner with duplicates=True. The API works in a similar way as with misclassified images: just choose the ones you want to delete and click 'Next Batch' until there are no more images left.

# In[ ]:


#ds, idxs = DatasetFormatter().from_similars(learn)


# Remember to recreate your ImageDataBunch from your cleaned.csv to include the changes you made in your data!

# # Prediction
# 

# In[ ]:


#get the image .jpg nummer 
image_path = "../input/10-monkey-species/training/training/"
images_dict = {}


for image in os.listdir(image_path):
    folder_path = os.path.join(image_path, image)
    images = os.listdir(folder_path)
    
    images_dict[image] = [folder_path, image]
    img_idx = rand.randint(0,len(image)-1)
    image_img_path = os.path.join(image_path, image, images[img_idx])
    #printing image
    img = cv2.imread(image_img_path)
    print(image_img_path) # to get the path of one image with the .jpg number; uncommen this line
    #plt.imshow(img);


# In[ ]:


import fastai
#fastai.defaults.device = torch.device('cpu')


# In[ ]:


img = open_image('../input/10-monkey-species/training/training/n4/n4146.jpg')
img


# In[ ]:


classes = ['n0', 'n1', 'n2', 'n3', 'n4','n5','n6','n7', 'n8','n9']


# In[ ]:


data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)


# In[ ]:


learn = create_cnn(data2, models.resnet34, model_dir="/tmp/model/").load('stage-3')


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))

