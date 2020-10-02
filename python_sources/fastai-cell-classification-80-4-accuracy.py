#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from glob import glob
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[13]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


from fastai.vision import *
from fastai.metrics import error_rate, accuracy


# **Loading Data**
# 
# For the purposes of this notebook, we shall be concerning ourselves with only image classification. As such, we require to load the images within the dataset.
# 
# An easy way to do so is to utilise glob to get the path of each image. A dict will be created with the key being each image name and the value the relative path to that image. This dict will become useful in comprising the dataframe we shall be working from later.
# 
# When working with images, it is advised to convert the labels into a one-hot encoding to allow for the model to perform better at its task of prediction. To do so, we will create a dict of the various legion types, with the used shorthand as their key and the actual type of lesion as the value.
# 
# This dict will serve as the means to which we will carry out one hot encoding.

# In[5]:


image_dir = "../input"

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(image_dir, '*', '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# With the dicts defined, let's create the dataframe that we will be utilising going forward. The metadata required to comprise the dataset is to be found in HAM10000_metadata.csv, so let's load that and get to work.

# In[6]:


image_dataframe = pd.read_csv(os.path.join(image_dir, 'HAM10000_metadata.csv'))
image_dataframe.head(5)


# Now we know what we're working with, it would serve us better to create a few columns which are derived from our dicts defined earlier.
# 
# Firstly, we shall create the column *path* whose data shall be a mapping from the associated image_id to the respective value in imageid_path_dict.
# 
# Secondly, we shall create the column *cell_type* whose data will be, like *path* above, a mapping between the *dx* column and the lesion_type_dict.
# 
# Finally, we shall create the column *cell_type_dx* which will be the categorical representation of the various cell types.
# Specifically, the lesion categories will be represented as:
# 
# * 'nv': 0
# * 'mel': 1
# * 'bkl': 2
# * 'bcc': 3
# * 'akiec': 4
# * 'vasc': 5
# * 'df': 6

# In[7]:


image_dataframe['path'] = image_dataframe['image_id'].map(imageid_path_dict.get)
image_dataframe['cell_type'] = image_dataframe['dx'].map(lesion_type_dict.get) 
image_dataframe['cell_type_idx'] = pd.Categorical(image_dataframe['cell_type']).codes
image_dataframe.head(5)


# With the dataframe complete for our purposes, let's see what the distribution is for each lesion type.

# In[8]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize = (12, 6))
image_dataframe['cell_type'].value_counts().plot(kind='bar', ax=ax)


# As can be observed, there's a distinct bias towards Melanocytic nevi lesions within the dataset. This is to be expected given the commonality of these kind of lesions. 
# 
# It's time to comprise the image dataset! We shall concatenate two columns, *path* and *cell_type*, renaming the columns to *name* and *label* respectively.

# In[20]:


image_dataset = pd.concat([image_dataframe['path'], image_dataframe['cell_type']], axis=1, keys=['name', 'label'])

image_dataset.head(5)


# As we are utilising FastAI for this classifier, it provides us with the means to split and transform the dataset. From the [documentation](http://https://docs.fast.ai/vision.transform.html), get_transforms
# 
# > returns a tuple of two list of transforms: one for the training set and one for the validation set (we don't want to modify the pictures in the validation set, so the second list of transforms is limited to resizing the pictures). This can be then passed directly to define a DataBunch object (see below) which is then associated with a model to begin training.
# 
# The resulting tuple from get_transforms is passed as the transformed dataset to [ImageDataBunch](http://https://docs.fast.ai/vision.data.html#ImageDataBunch).
# ImageDataBunch utilises factory methods to create an instance of ImageDataBunch, so for this instance as we've created a data frame representation of our dataset, we shall utilise [from_df](http://https://docs.fast.ai/vision.data.html#ImageDataBunch.from_df) which allows us to create an ImageDataBunch from a DataFrame.
# 
# Subsequently, we normalise the data using imagenet_stats. In the fast.ai library we have imagenet_stats, cifar_stats and mnist_stats so we can add normalization easily with any of these datasets.

# In[21]:


bs = 64

tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_df(".", image_dataset, ds_tfms=tfms, size=28, bs=bs).normalize(imagenet_stats)


# Let's review a selection of the images in the dataset.

# In[22]:


data.show_batch(rows=3, figsize=(7,6))


# # Model Selection
# 
# For the purposes of this classification task, we shall utilise a convolutional neural network (CNN) which is perfectly suited to this task.
# In respects to which particular model to utilise, instead of defining a model by ourselves and, more than likely, attempt to recreate what has already done by the field, we will utilise pretrained image classification models. Specifically, we shall utilise Resnet34, 50 and 161 to compare different complexity of neural network models to ascertain whether more neurons is better.
# 
# A brief overview on each Resnet mentioned above can be found here at [https://neurohive.io/en/popular-networks/resnet/](http://https://neurohive.io/en/popular-networks/resnet/)
# 
# **Training: resnet34**
# 
# With the data perpared for FastAI to be able to utilise, we can now start training our model. 
# To do so, we will use a convolutional neural network backbone and a fully connected head with a single hidden layer as a classifier; in this case Resnet34 (All models selected can be found in torchvision's [API](https://pytorch.org/docs/stable/torchvision/models.html)
# 
# We will train for 4 epochs (4 cycles through all our data).

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy)


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


learn.save('stage-1')


# **Results**
# 
# Let's see what results we have got.
# 
# Firstly, we will discover which of the classes of cell type that the model had difficulty classifying.
# Secondly, we will determine if what the model predicted was reasonable, i.e. the mistakes made were not naive. This is an indicator that our classifier is working correctly.
# 
# Furthermore, when we plot the confusion matrix, we can see that the distribution is heavily skewed: the model makes the same mistakes over and over again but it rarely confuses other categories. This suggests that it just finds it difficult to distinguish some specific categories between each other; this is normal behaviour.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


doc(interp.plot_top_losses)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# **Unfreezing, fine-tuning, and learning rates**
# 
# Since our model is working as we expect it to, we will unfreeze our model and train some more.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# In[ ]:


learn.load('stage-1')


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10, max_lr=slice(1e-2,1e-4))


# Not a bad start, but it's worth trying other models to compare and contrast.

# **Training: resnet50**
# 
# Now we will train in the same way as before but with one caveat: instead of using resnet34 as our backbone we will use resnet50 (resnet34 is a 34 layer residual network while resnet50 has 50 layers. It will be explained later in the course and you can learn the details in the resnet paper).
# 
# Basically, resnet50 usually performs better because it is a deeper network with more parameters. Let's see if we can achieve a higher performance here. To help it along, let's us use larger images too, since that way the network can see more detail. We reduce the batch size a bit since otherwise this larger network will require more GPU memory.

# In[ ]:


data = ImageDataBunch.from_df(".", image_dataset, ds_tfms=tfms, size=28, bs=bs//2).normalize(imagenet_stats)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.save('stage-1-50')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-4))


# **Training: ResNet152**
# 
# Now we will train in the same way as before but with one caveat: instead of using resnet34 or resnet50 as our backbone, we will use Resnet152. As the name suggests, it utilises 152 layers. The batch size remains reduced to limit the computation time.

# In[ ]:


learn = cnn_learner(data, models.resnet152, metrics=accuracy)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

