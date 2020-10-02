#!/usr/bin/env python
# coding: utf-8

# ## Imports and Params

# In[ ]:


from pathlib import Path
from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 128   # batch size
arch = models.resnet50


# ## Setup the data

# Setup the path to the images, caption, and label information. Create a DataBunch that is both the training and validation information wrapped in a class with transformations to be performed included.

# In[ ]:


path = Path(os.path.join('..', 'input', 'food41'))
path_h5 = path
path_img = path/'images'
path_meta = path/'meta/meta'
path_working = '/kaggle/working/'
path_last_model = Path(os.path.join('..', 'input', 'starter-food-images'))

get_ipython().system('ls {path}')


# In[ ]:


get_ipython().system('ls {path_last_model}')


# In[ ]:


# Modify the from folder function in fast.ai to use the dictionary mapping from folder to space seperated labels
def label_from_folder_map(class_to_label_map):
    return  lambda o: class_to_label_map[(o.parts if isinstance(o, Path) else o.split(os.path.sep))[-2]]


# In[ ]:


# Develop dictionary mapping from classes to labels
classes = pd.read_csv(path_meta/'classes.txt', header=None, index_col=0,)
labels = pd.read_csv(path_meta/'labels.txt', header=None)
classes['map'] = labels[0].values
classes_to_labels_map = classes['map'].to_dict()
label_from_folder_food_func = label_from_folder_map(classes_to_labels_map)


# In[ ]:


# Setup the training ImageList for the DataBunch
train_df = pd.read_csv(path_meta/'train.txt', header=None).apply(lambda x : x + '.jpg')
train_image_list = ImageList.from_df(train_df, path_img)

# Setup the validation ImageList for the DataBunch
valid_df = pd.read_csv(path_meta/'test.txt', header=None).apply(lambda x : x + '.jpg')
valid_image_list = ImageList.from_df(valid_df, path_img)


# In[ ]:


def get_data(bs, size):
    """Function to return DataBunch with different batch and image sizes."""
    # combine training and validation image lists into one ImageList
    data = (train_image_list.split_by_list(train_image_list, valid_image_list))
    
    tfms = get_transforms() # get all transformations

    # label with function defined above using the mapping from folder name to labels
    # perform transformations and turn into a DataBunch
    data = data.label_from_func(label_from_folder_food_func).transform(
        tfms, size=size).databunch(bs=bs,  num_workers = 0).normalize(
        imagenet_stats)
    return data

data = get_data(bs, 64)


# In[ ]:


# show a batch to get an idea of the images and labels
data.show_batch(rows=4, figsize=(10,9),)


# In[ ]:


# print all labels in the dataset
print(data.classes)


# ## Training the model

# Use a ResNet50 CNN network architecture that the weights are pre-trained on ImageNet as the starting point to train the model. Error-rate metric is used to get the Top-1 score.

# In[ ]:


# setup data, model architecture, and metrics
learn = cnn_learner(data, arch, metrics=error_rate)


# In[ ]:


# model_dir is set to the path where DataBunch is located
# Kaggle this needs to be set to '/kaggle/working/'
learn.model_dir = path_working


# In[ ]:


learn.data = get_data(bs//4, 412) #bs = 32
learn.purge();


# In[ ]:


learn.load(path_last_model/'stage-2-50-412');


# In[ ]:


learn.validate(learn.data.valid_dl)


# In[ ]:


learn.TTA()

