#!/usr/bin/env python
# coding: utf-8

# #  Plant Pathology 2020 - FGVC7
#  
# This is a starter kernel created for the competition ["Plant Pathology 2020 - FGVC7"](http://Plant Pathology 2020 - FGVC7).
# 
# ## Problem Statement
# 
# There are many benefits in identifying diseases in agricultural crops. These benefits include reduced time, input costs and less adverse environmental impact (due to overuse of chemicals).
# 
# In this competition, we are given 1821 apple leafs to create a model which can detect diseases.
# The aim is to classify the given test images into different categories like 'healthy', 'multiple_diseases','scab' and 'rust'.
# Please click on the [link](http://https://www.kaggle.com/c/plant-pathology-2020-fgvc7/overview) to find the details of the problem statement.
# 
# ## Data
# 
# Labels : healthy, scab, rust and multiple_diseases
# 
# 1.  **train.csv** contains image_id, healthy, scab, rust and multiple_diseases
# 2.  **test.csv**  contains image_id
# 
# 3. **images** : A folder containing the test and train images in jpg format. The name of files is in the below format
# 
# 
# * Test files   : Test_0 to Test_1820 
# * Train files  : Train_0 to Train_1820
# 
# ## Note
# 
# A simple example of Transfer Learning has been shown in the kernel where we use a pretrained model of ResNet152. 
# 
# The kernel uses [fastai2](http://https://dev.fast.ai/) APIs which is a wrapper on top of PyTorch.
# 
# I'd like to thank the author of the below kernel. The kernel was very useful which creating this kernel.
# 
# [Plant Pathology 2020 - EDA + training (fastai2)](http://https://www.kaggle.com/lextoumbourou/plant-pathology-2020-eda-training-fastai2)
# 

# ### Importing the required libraries

# In[ ]:


# Support operations for multi-dimentional arrays and matrices
import numpy as np
# Data manipulation and analysis
import pandas as pd
# Area Under the Receiver Operating Characteristic Curve (ROC AUC)
from sklearn.metrics import roc_auc_score


# ###  Installing the version 2 of fastai

# In[ ]:


# Installing the version 2 of fastai 
get_ipython().system('pip install -q fastai2')


# ### Importing all the required fastaiv2 functions

# In[ ]:


from fastai2.vision.all import *


# ### Defining all the variables, hyperparameters and loading the data.

# In[ ]:


# Defining the Path
PATH = Path('/kaggle/input/plant-pathology-2020-fgvc7/')
IMAGE_PATH = Path('/kaggle/input/plant-pathology-2020-fgvc7/images')

LABELS = ['healthy', 'multiple_diseases', 'rust', 'scab']

# Reading the train and test data
train_df = pd.read_csv(PATH / 'train.csv')
test_df = pd.read_csv(PATH / 'test.csv')

#Defining hyperparameters
VALIDATION_PCT = 0.2
SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 16


# ### Let's have a look at the training data.

# In[ ]:


train_df.head()


# ### Creating a function to fetch the label for a specific image.

# In[ ]:


def get_category(row):
    for key, value in row[LABELS].items():
        if value == 1:
            return key


# A new column called 'label' is created. 
# The func get_category() is used to fetch the values for this column from the training data.

# In[ ]:


train_df['label'] = train_df.apply(get_category, axis=1)


# ### Looking at the training data with the newly created column 'label'.

# In[ ]:


# train_df.head()
train_df.size


# # Data Visualization

# Defining a function which can be used to load the data. It uses fastai's **datablock api** which helps in fetching the files from the given Path.

# In[ ]:


def load_data():
    datablock = DataBlock(blocks=(ImageBlock, CategoryBlock(vocab=LABELS)),
                          getters=[ColReader('image_id', pref=IMAGE_PATH, suff='.jpg'),
                                   ColReader('label')],
                          splitter=RandomSplitter(valid_pct = VALIDATION_PCT, seed = SEED),
                          item_tfms=Resize(IMAGE_SIZE),
                          batch_tfms=aug_transforms(size = IMAGE_SIZE, max_rotate=40., min_scale=0.80, flip_vert=True, do_flip=True)
    )
    return datablock.dataloaders(source=train_df, bs=BATCH_SIZE)


# In[ ]:


data = load_data()
data.show_batch()


# ### Defining metric functions (ROC AUC) on which the predictions are evaluated

# In[ ]:


def comp_metric(preds, targs, labels=range(len(LABELS))):
    targs = np.eye(4)[targs]
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])

def healthy_roc_auc(*args):
    return comp_metric(*args, labels=[0])

def multiple_diseases_roc_auc(*args):
    return comp_metric(*args, labels=[1])

def rust_roc_auc(*args):
    return comp_metric(*args, labels=[2])

def scab_roc_auc(*args):
    return comp_metric(*args, labels=[3])


# # Model

# ### Initializing and training our CNN model with the required parameters.

# In[ ]:


learn = cnn_learner(data, resnet152, metrics=[
    AccumMetric(healthy_roc_auc, flatten=False),
    AccumMetric(multiple_diseases_roc_auc, flatten=False),
    AccumMetric(rust_roc_auc, flatten=False),
    AccumMetric(scab_roc_auc, flatten=False),
    AccumMetric(comp_metric, flatten=False)]
    )
learn.fine_tune(50)


# ### Interpretting the predictions of the model

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(4, nrows=4)


# ### Prediction of labels for the test data 

# In[ ]:


# Creating dataloader for the test data
test_dl = data.test_dl(test_df)
# Predicting the category of each image in test data using the trained model
test_preds, _ = learn.get_preds(dl=test_dl)


# # Prediction

# ### Creating the test prediction dataframe and csv file

# In[ ]:


# Adding the column for the labels to the test data frame
test_predictions = pd.concat([test_df, pd.DataFrame(np.stack(test_preds), columns=LABELS)], axis=1)
test_predictions.to_csv('submission.csv', index=False)
test_predictions.head()

