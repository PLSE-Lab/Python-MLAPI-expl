#!/usr/bin/env python
# coding: utf-8

# ### work in progress
# <br>
# To learn more about the fast.ai library check out the [documentation](https://docs.fast.ai) and the [fast.ai course 'Practical Deep Learning for Coders'](https://course.fast.ai/)

# ## Preparation
# #### Setup environment and import necessary modules

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *
import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image


# In[ ]:


# hide warnings
import warnings
warnings.simplefilter('ignore')


# #### Inspect and understand input data
# The first step in most competitions is to check out the input data. Let's do this:

# In[ ]:


input_dir = Path("../input/digit-recognizer")
os.listdir(input_dir)


# We found 3 interesting files:
# - sample_submission.csv
# - train.csv
# - test.csv
# 
# 'sample_submission.csv' will show us, how we have to structure our data at the end before we submit it to the competition. We will only need this file at the end.
# 
# 'train.csv' is a file that contains all necessary information for training the CNN
# 
# 'test.csv' is the file we later use to test how good our CNN is
# 
# Let's look at 'train.csv' and 'test.csv' to see how the data looks

# In[ ]:


train_df =  pd.read_csv(input_dir/"train.csv")
train_df.head(3)


# In[ ]:


test_df =  pd.read_csv(input_dir/"test.csv")
test_df.head(3)


# The data looks very interesting. Let's summarize what we got here:
# 
# What we know about 'train.csv':
# - Each row is one image
# - The first row of each image is the label. It tells us which digit is shown.
# - The other 784 rows are the pixel for each digit and should be read like this
# 
# `000 001 002 003 ... 026 027
# 028 029 030 031 ... 054 055
# 056 057 058 059 ... 082 083
#  |   |   |   |  ...  |   |
# 728 729 730 731 ... 754 755
# 756 757 758 759 ... 782 783`
# 
# What we know about 'test.csv':
# - The structure is the same as in train.csv, but there are no labels because it's our task to predict the labels
# 
# To read more about the data, read the ['Data' tab of the competition](https://www.kaggle.com/c/digit-recognizer/data)
# 
# #### Getting the data into the right format
# In this tutorial I want to use the [fast.ai library](https://docs.fast.ai/). Looking at the [documentation](https://docs.fast.ai/vision.data.html#ImageDataBunch) we can quickly see, that fast.ai only accepts image files as data and not the format we were offered in this competition. Therefore we have to create images from the data we have. Fast.ai accepts image data in different formats. We will use the from_folder function of the ImageDataBunch class to load in the data. To do this we need all images in the following structure:
# 
# `path\
#   train\
#     0\
#       ___.jpg
#       ___.jpg
#       ___.jpg
#     1\
#       ___.jpg
#       ___.jpg
#     2\
#       ...
#     3\
#       ...
#     ...
#   test\
#     ___.jpg
#     ___.jpg
#     ...
# `
# 
# Let's first create the folder structure!
# 
# (nice to know: the input folder of Kaggle Competitions is always read-only, so if we want to add data or create folders, we have to do so outside of the input folder)

# In[ ]:


train_dir = Path("../train")
test_dir = Path("../test")


# In[ ]:


# Create training directory
for index in range(10):
    try:
        os.makedirs(train_dir/str(index))
    except:
        pass


# In[ ]:


# Test whether creating the training directory was successful
sorted(os.listdir(train_dir))


# In[ ]:


#Create test directory
try:
    os.makedirs(test_dir)
except:
    pass


# Okay, all folders are created! The next step is to create the images inside of the folders from 'train.csv' and 'test.csv'. We will use the Image module from PIL to do this.
# 
# 
# we have to reshape each numpy array to have the desired dimensions of the image (28x28)
# 
# `000 001 002 003 ... 026 027
# 028 029 030 031 ... 054 055
# 056 057 058 059 ... 082 083
#  |   |   |   |  ...  |   |
# 728 729 730 731 ... 754 755
# 756 757 758 759 ... 782 783`
# 
# then we use the fromarray function to create a .jpg image from the numpy array and save it into the desired folder

# In[ ]:


# save training images
for index, row in train_df.iterrows():
    
    label,digit = row[0], row[1:]
    
    filepath = train_dir/str(label)
    filename = f"{index}.jpg"
    
    digit = digit.values
    digit = digit.reshape(28,28)
    digit = digit.astype(np.uint8)
    
    img = Image.fromarray(digit)
    img.save(filepath/filename)


# In[ ]:


# save testing images
for index, digit in test_df.iterrows():

    filepath = test_dir
    filename = f"{index}.jpg"
    
    digit = digit.values
    digit = digit.reshape(28,28)
    digit = digit.astype(np.uint8)
    
    img = Image.fromarray(digit)
    img.save(filepath/filename)


# Now that we have the right folder structure and images inside of the folders we can use the from_folder method of the ImageDataBunch class to create the dataset.
# 
# Whenever we train a CNN we need to split the data into 3 parts:
# - training set: used to modify weights of neural network
# - validation set: prevent overfitting
# - test set: test accuracy of fully-trained model
# 
# But right now we only have a training and a test set. That's why we split the test set to get a validation set. We do this with the 'valid_pct' parameter. This is one of the parameters you could tune to increase the accuracy. To learn more about this read [this stackoverflow post](https://stackoverflow.com/a/13623707)

# In[ ]:


tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(
    path = train_dir,
    test = test_dir,
    valid_pct = 0.2,
    bs = 32,
    size = 28,
    ds_tfms = tfms,
    num_workers = 0
).normalize(imagenet_stats)
print(data)
print(data.classes)
data.show_batch(figsize=(5,5))


# ## Training

# The next step is to select and create a CNN. In fast.ai creating a CNN is really easy. You just have to select one of the models from the [Computer Vision models zoo](https://docs.fast.ai/vision.models.html#Computer-Vision-models-zoo)

# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=accuracy, model_dir="/tmp/models")


# Now it's time to train the neural network using the fit_one_cycle() function. <br>Parameters to modify: the number of epochs to train and the learning rate

# In[ ]:


learn.fit_one_cycle(12)


# ## Evaluation
# Create a ClassificationInterpretation object to generate confusion matrices and visualizations of the most incorrect images

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# Plot the 9 images with the highest loss

# In[ ]:


interp.plot_top_losses(9, figsize=(7, 7))


# A good way to summarize the performance of a classification algorithm is to create a confusion matrix. Confusion Matricies are used to understand which classes are most easily confused.

# In[ ]:


interp.plot_confusion_matrix()


# **LET's make our model better, we will unfreeze, fine tuning and learning rates**

# In[ ]:


#let's unfreeze the whole model!
learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1)


# OOH woow this is so good! why? it is supposed to be BAD! because we just unfreeze it! OK let's commit this for V3 and submit it!

# OK let's try to make the model even better than 0.99 accuracy! By using learning lr_find!****

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-4))


# OK we will commit this as V7!

# ## Predict
# 
# Get the predictions on the test set

# In[ ]:


class_score, y = learn.get_preds(DatasetType.Test)
class_score = np.argmax(class_score, axis=1)


# The last step is creating the submission file. "sample_submission.csv" is showing us the desired format

# In[ ]:


sample_submission =  pd.read_csv(input_dir/"sample_submission.csv")
display(sample_submission.head(2))
display(sample_submission.tail(2))


# Columns the submission file has to have:
# - ImageId: index in the test set, starting from 1, going up to 28000
# - Label: the displayed digit

# In[ ]:


ImageId = []
for path in os.listdir(test_dir):
    # '456.jpg' to '456'
    path = path[:-4]
    path = int(path)
    # +1 because index starts at 1 in the submission file
    path = path + 1
    ImageId.append(path)


# In[ ]:


submission  = pd.DataFrame({
    "ImageId": ImageId,
    "Label": class_score
})
submission.sort_values(by=["ImageId"], inplace = True)
submission.to_csv("submission.csv", index=False)
submission[:10]


# OK this is for commit V7!

# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir="/tmp/models")


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5)


# In[ ]:


learn.save('stage1')


# In[ ]:


learn.load('stage1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-4))


# OK looks good ..let's do more epoch say 6 more

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(12)


# OK this looks good, I am lucky?? Is this a good method using fast.ai to generate really high accuracy??? I don't know, I hope so. let's commit this for V7 but I am not going to submit this! So, we are going up the cell to execute predict and submission process!****

# In[ ]:




