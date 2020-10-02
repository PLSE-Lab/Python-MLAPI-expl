#!/usr/bin/env python
# coding: utf-8

# **Implementation of Resnet152 to achieve accuracy of 0.99 in the public leaderboard**

# **Importing necessary libraries**

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from fastai.vision import *
from fastai.metrics import error_rate


# **Variables initialization**

# In[ ]:


SEED = 42
VALIDATION_PCT = 0.1
IMAGE_SIZE_64 = 64
IMAGE_SIZE_224 = 224
BATCH_SIZE_64 = 64
BATCH_SIZE_224 = 224
PATH = Path('../input')
TRAIN_PATH = PATH/'train'
TEST_FOLDER_PATH = "test/test"
SAMPLE_SUBMISSION_PATH = PATH/"sample_submission.csv"


# **Function using which you can load the data. It uses fastai's data_block api which helps in fetching the files from the folder structure.**

# In[ ]:


def load_data(path, image_size, batch_size, validation_pct = VALIDATION_PCT):
    data = (ImageList.from_folder(path)
                .split_by_rand_pct(validation_pct, seed = SEED) # Taking 10% of data for validation set
                .label_from_folder() # Label the images according to the folder they are present in
                .transform(get_transforms(), size = image_size) # Default transformations with the given image size 
                .databunch(bs = batch_size) # Using the given batch size
                .normalize(imagenet_stats)) # Normalizing the images to improve data integrity
    return data


# **Loading the data with low resolution.**

# In[ ]:


data_64 = load_data(TRAIN_PATH, IMAGE_SIZE_64, BATCH_SIZE_224, VALIDATION_PCT)
data_64


# **Initializing our CNN model with the required parameters. **

# In[ ]:


learn = cnn_learner(data_64, # training data with low resolution
                    models.resnet152, # Model which is pretrained on the ImageNet dataset 
                    metrics = [error_rate, accuracy], # Validation metrics
                    model_dir = '/tmp/model/') # Specifying a write location on the machine where the lr_find() can write 


# **Finding the best range for the Learning Rate. The aim is to find a steep range where our model can converge faster. 
# **

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# **Fitting the low resolution dataset into the model.**

# In[ ]:


learn.fit_one_cycle(1, 1e-2)


# **Fetching the sample submission file which contains the name of the images and a sample prediction class**

# In[ ]:


test_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
test_df.head()


# **Extracting the test images from the data frame**

# In[ ]:


test_images = ImageList.from_df(test_df, PATH/"", folder = TEST_FOLDER_PATH)


# In[ ]:


test_images[1]


# In[ ]:


test_images[192]


# **Loading the data with high resolution and adding the test images to it.**

# In[ ]:


data_256 = load_data(TRAIN_PATH, IMAGE_SIZE_224, BATCH_SIZE_64, VALIDATION_PCT)
data_256.add_test(test_images)


# **Putting the high resolution data as the training data for the model.
# Finding the best LR range so that our model converges fasters.**

# In[ ]:


learn.data = data_256
learn.lr_find()
learn.recorder.plot()


# **Unfreezing the layers of the Neural Net so that their weights can be updated with respect to our high resolution dataset. **

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(30, max_lr = slice(1.0e-4, 1.0e-3))


# **Finding the probabality of all the classes for the given Test Dataset. Finding the predictions for the dataset by choosing the class with the highest probability.**

# In[ ]:


test_probabalities, _ = learn.get_preds(ds_type=DatasetType.Test)
test_predictions = [data_256.classes[pred] for pred in np.argmax(test_probabalities.numpy(), axis=-1)]


# **Creating a predictions dataframe which consists of the imageId and the class.**

# In[ ]:


test_df.predicted_class = test_predictions
test_df.to_csv("submission.csv", index=False)
test_df.head()

