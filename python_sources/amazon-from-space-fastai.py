#!/usr/bin/env python
# coding: utf-8

# # Planet: Understanding the Amazon from Space
# ## Use satellite data to track the human footprint in the Amazon rainforest
# 
# Multiclass classification for classifying satelite images of the Amazon rainforest. 
# Exploring the data and Fast Ai. 

# ## Importing dependencies and setting file paths

# In[ ]:


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import random
import os
import glob
import cv2 
from fastai.vision import *
from fastai import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import plot_confusion_matrix


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')


# In[ ]:


# Set seed fol all
def seed_everything(seed=1358):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()


# In[ ]:


PATH = Path('../input/planets-dataset/planet/planet/')
train_img = PATH/'train-jpg'
train_folder = 'train-jpg'
test_img = PATH/'test-jpg'
model_dir = Path('/kaggle/working/')
bs = 64


# In[ ]:


PATH.ls()


# ## Little bit of EDA

# In[ ]:


train_df = pd.read_csv(os.path.join(PATH, 'train_classes.csv'))
# adding path to the image in our dataframe. 
train_df['image_name'] = train_df['image_name'].apply(lambda x: f'{train_folder}/{x}.jpg')
train_df.head()


# > We have 17 unique labels for this data

# In[ ]:


# Since this is a multi lable task and the labels are given as tags in a single dataframe series
biner = MultiLabelBinarizer()
tags = train_df['tags'].str.split()
y = biner.fit_transform(tags)

labels = biner.classes_
print('Number of labels: ', len(labels))
print(labels)


# In[ ]:


# Getting the labels into one hot encoded form for EDA ease. 
for label in labels:
    train_df[label] = train_df['tags'].apply(lambda x: 1 if label in x.split()  else 0)
    
train_df.head()


# The label primary appears the most in our dataset followed by clear and agriculture. 
# As stated in the data description, primary refers to primary rainforest.
# > Generally speaking, the "primary" label was used for any area that exhibited dense tree cover. 

# In[ ]:


train_df[labels].sum().sort_values(ascending=False).plot(kind='barh', figsize=(8,8))


# Looking at the co-ocurrance for these labels. 
# > The combination (primary, clear) has the highest co-ocurrance. Followed by (primary, agriculture)

# In[ ]:


df_asint = train_df.drop(train_df.columns[[0,1]], axis=1).astype(int)
coocc_df = df_asint.T.dot(df_asint)

coocc_df


# In[ ]:


# Confusion matrix. 


# Plotting a few random images with there labels to see how the data looks. 
# Choose 10 random images from the data. 

# In[ ]:


#reading images

random_imgs = train_df.ix[random.sample(list(train_df.index), 10)][['image_name', 'tags']]

to_read = random_imgs.loc[:, 'image_name'].values
tags = random_imgs.loc[:, 'tags'].values

images = [cv2.imread(os.path.join(PATH/file)) for file in to_read]
print("Number of images: ", len(images))
print("Size of an image: ", images[0].shape)


# In[ ]:


plt.figure(figsize=(25,15))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(image)
    plt.grid(False)
    plt.title(tags[i])


# In[ ]:





# ## Training 

# Using the usual fast ai to train and evaluate models. 

# In[ ]:


print(f"Size of Training set images: {len(list(train_img.glob('*.jpg')))}")
print(f"Size of Test set images: {len(list(test_img.glob('*.jpg')))}")


# Starting with an image size of 128*128 with a few transformations. 
# + Flipping the image vertically and horizontaly. 
# + Changing lighting and contrast. 
# + Rotations. 
# + Zooming. 

# In[ ]:


img_size = 128

tfms = get_transforms(do_flip=True,flip_vert=True,p_lighting=0.4,
                      max_lighting=0.3, max_zoom=1.05, max_rotate=360, xtra_tfms=[flip_lr()])


# The datablock API makes things very easy. 
# Im using 1% of the training data to validate the models. 

src = (ImageList.from_df(train_df, PATH, cols='image_name')
        .split_by_rand_pct(valid_pct=0.1)
        .label_from_df(label_delim=' '))


data = (src.transform(tfms,size=img_size,resize_method=ResizeMethod.CROP)
        .databunch(bs=bs,num_workers=4) 
        .normalize(imagenet_stats)      
       )


# 36432 images for training and 4047 for validating. 

# In[ ]:


data


# Train data has 36432 images of size 128x128x3. 

# In[ ]:


data.train_ds


# In[ ]:


data.valid_ds


# Picking a point from the train_ds from our databunch gives the Image object which includes its size. 
# It also outputs that the label is Multicategory coupled with its value. 

# In[ ]:


data.train_ds[0]


# A random point from the databunch. 
# This time i select the labels from **data.train_ds.y** and image from **data.train_ds.x**
# 
# The databunch containing the train dataset has both x and y components and we can index into them. 

# In[ ]:


print(data.train_ds.y[200])
data.train_ds.x[200]


# In[ ]:


data.show_batch(rows=2)


# > Every experiment would have two stages: 
# + 1st stage: Freezing early layers and only fine-tuning the last few newly added layers.
# + 2nd stage: Unfreezing all the layers and fine-tuning them. 
# 
# 
# Using the learning rate finder before every stage. 
# 

# ## Experiment 1
# 
# > For the first experiment: 
# > + Using resnet50 (pretrained on imagenet)
# > + fbeta with 0.2 threshold and accuracy as metrics. 
# 

# In[ ]:


model_1 = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)

learn = create_cnn(data, model_1, metrics=[acc_02, f_score], model_dir='/kaggle/working')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, 0.01)


# With 5 epochs able to get upto 92% fbeta. 

# In[ ]:


learn.model_dir = '/kaggle/working'
learn.save('resnet50-stage1')


# ## Experiment 2
# 
# > For the second experiment:
# + Using DenseNet121 (pretrained on imagenet)
# + Similar metrics as first experiment. 

# Fine-tuning last layers in Stage 1 

# In[ ]:


model_2 = models.densenet121
learn_dense = create_cnn(data, model_2, metrics=[acc_02, f_score], model_dir='/kaggle/working')


# In[ ]:


learn_dense.fit_one_cycle(5, 0.01)


# DenseNet201 performs slighly better than resnet50. 
# I ll use this for fine tuning the entire model. 

# In[ ]:


learn_dense.save('DenseNet121-stage1')


# Fine-tuning the entire model in stage 2. 

# In[ ]:


learn_dense.unfreeze()
learn_dense.lr_find()
learn_dense.recorder.plot()


# In[ ]:


learn_dense.fit_one_cycle(5, slice(1e-5, 1e-4))


# In[ ]:


learn_dense.save('DenseNet121-stage2')


# ## Experiment 3
# 
# Fine tuning the network further with larger image size. 

# In[ ]:


data_2 = (src.transform(tfms,size=256,resize_method=ResizeMethod.CROP)
        .databunch(bs=bs,num_workers=4) 
        .normalize(imagenet_stats)      
       )

data_2


# In[ ]:


model_2 = models.densenet121
learn_dense_2 = create_cnn(data_2, model_2, metrics=[acc_02, f_score], model_dir='/kaggle/working')


# In[ ]:


learn_dense_2.load('DenseNet121-stage2')


# In[ ]:


learn_dense_2.lr_find()
learn_dense_2.recorder.plot()


# In[ ]:


learn_dense_2.fit_one_cycle(10, 0.01)


# In[ ]:


learn_dense_2.unfreeze()
learn_dense_2.lr_find()
learn_dense_2.recorder.plot()


# In[ ]:


learn_dense_2.fit_one_cycle(10, slice(1e-5,1e-4))


# In[ ]:




