#!/usr/bin/env python
# coding: utf-8

# # Setting up the environment
# 
# Intall fast.ai

# In[ ]:


get_ipython().system('pip install fastai2')


# Import the vision library and the required metrics! <br>
# Fast.ai recommends importing the entire library.

# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# # Getting Started
# 
# We need to read the data and extract the labels from the folder name. <br>
# We created three lists, one each for the entire path per image, labels per image, and corresponding image name.  

# In[ ]:


bs = 64 #batch-size

import cv2

data = "/kaggle/input/flowers-recognition/flowers/"
folders = os.listdir(data)
print(folders)

image_names = []
train_labels = []
train_images = []

size = 64,64

for folder in folders:
    for file in os.listdir(os.path.join(data,folder)):
        if file.endswith("jpg"):
            image_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
        else:
            continue


# # Augmentation and Visualization
# 
# Data Augmentation using *get_transforms* method. <br>
# Create an *ImageDataBunch* object from the list of paths of the images.

# In[ ]:


tfms = get_transforms(do_flip = True, flip_vert = True, max_rotate = 30, max_zoom = 1.2, p_affine = 0.5)
data1 = ImageDataBunch.from_lists('',image_names, labels=train_labels, ds_tfms=tfms, size=224, bs=bs).normalize(imagenet_stats)
data1.classes


# Visualizing the data and the class size.

# In[ ]:


data1.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data1.classes)
len(data1.classes),data1.c


# # Training the Model
# 
# Using *cnn_learner* we define our Convolutional Neural Network of ResNet50 architechture. ResNet50 has a convolutional neural network backbone and a fully connected head with a single hidden layer as a classifier. 

# In[ ]:


learn = cnn_learner(data1, models.resnet50, metrics=error_rate)


# We will train for 4 epochs (4 cycles through all our data).

# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# Using *Confusion Matrix* to visualize loss and prediction error.
# <br> The model makes the same mistakes over and over again but it rarely confuses other categories. 

# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# # Fine Tuning
# .
# We use *lr_find* to find the best learning rate and plot it using a line graph.

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# Using the plot above we have to use the steepish slope in the graph, just before the lowest learning rate. 
# 
# <br>*Not the lowest point.*
# 
# <br> This is because the model needs to find that learning rate where it learns the most.
# 
# <br>We will unfreeze our model and train some more.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4, max_lr=slice(1e-6, 1e-4))


# # Results
# 
# Our accuracy has increased significantly. <br>
# This a pretty good model.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

interp.plot_top_losses(9, figsize=(15,11))


# We can see what we have achieved now.

# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:




