#!/usr/bin/env python
# coding: utf-8

# ## Version History
# 
# * V6
#     - Resnet with Stage 1 (only head with 4 epochs) and Stage 2 (full train with 8 epochs) + More aggresive transformation applied to see if generalization can be improved.
#     - <font color = red>Public Score: TBD Rank: TBD</font>
# * V5
#     - Reverting back to single kernel for train and prediction
#     - Resnet with Stage 1 (only head with 4 epochs) and Stage 2 (full train with 4 epochs) + More aggresive transformation applied to see if generalization can be improved.
#     - <font color = red>Public Score: 0.9680 Rank: Did not improve</font>
# * V4 
#     - Scoring is slow. Is it running the entire notebook again?
#     - **Idea**: Try separating the prediction from the training process so that scoring is faster (training does not have to be run again while scoring).
#     - However, this does not seem to work. Scoring is equally slow even after separating the training and prediction into separate notebooks.
# * V3 
#     - Resnet with Stage 1 (only head with 4 epochs) and Stage 2 (full train with 4 epochs) 
#     - <font color = green>**Public Score: 0.9742 Rank: 416/603 at time of submission**</font>
# * V2
#     - Resnet with Stage 1 (only head with 4 epochs) and Stage 2 (full train with 1 epochs) 
#     - <font color = red>Public Score: 0.9580 Rank: Did not improve</font>
# * V1
#     - Resnet with Stage 1 (only head with 4 epochs) and Stage 2 (full train with 2 epochs)
#     - <font color = green>Public Score: 0.9708 Rank: 453/603 at time of submission</font>
# 
# 
# * **Summary:**
#     - Getting a validation accuracy of more than 99% consistently.
#     - In general Stage 2 can be trained for some more epochs since the train error is still more than the validation error.
#     - Since the public score (accuracy) is lower, try to train with more severe transformations so that it generalizes a little better.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.


# ## Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import imageio

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import fastai
from fastai.vision import *
from fastai.metrics import error_rate
print(fastai.__version__) # Checking version


# In[ ]:


verbose = 0
train = True  ## DO NOT CHANGE 'train' to False - See NOTE below ##
# Idea was to mark 'train' as False before submission but this deoes not help to speed up submission.
# Hence not using 'train' anymore, but not deleting for now (set always to True)


# ## Read Data

# In[ ]:


root = '/kaggle/input/'
parent = "Kannada-MNIST/"

root_plus_parent = root + parent

if train:
    if verbose >= 1:
        print(root_plus_parent)

    if verbose >= 1:
        for dirname, _, filenames in os.walk(root):
            for filename in filenames:
                print(os.path.join(dirname, filename))


# In[ ]:


if train:
    df_train = pd.read_csv(root_plus_parent + "train.csv")
    
if train & verbose >= 1:
    print(df_train.head())


# In[ ]:


df_test = pd.read_csv(root_plus_parent + "test.csv")
if train & verbose >= 1:
    print(df_test.head())


# In[ ]:


if train & verbose >= 1:
    print(df_train.shape, df_test.shape)


# In[ ]:


if train:
    train_x = df_train.iloc[:,1:].values.reshape(-1,28,28)
    
if train & verbose >= 1:
    print(train_x.shape)


# In[ ]:


if train:
    train_y = df_train['label'].values

if train & verbose >= 1:
    print(train_y.shape)


# In[ ]:


test_x = df_test.iloc[:,1:].values.reshape(-1,28,28)
if train & verbose >= 1:
    print(test_x.shape)


# In[ ]:


if train & verbose >= 1:
    plt.imshow(train_x[0,:], cmap='gray')


# ## Save data in right format for fastai

# In[ ]:


# This function has been taken from another kernel: https://www.kaggle.com/demonplus/kannada-mnist-with-fast-ai-and-resnet
def save_imgs(path:Path, data, labels=[]):
    path.mkdir(parents=True,exist_ok=True)
    for label in np.unique(labels):
        (path/str(label)).mkdir(parents=True,exist_ok=True)
    for i in range(len(data)):
        if(len(labels)!=0):
            imageio.imsave( str( path/str(labels[i])/(str(i)+'.jpg') ), data[i] )
        else:
            imageio.imsave( str( path/(str(i)+'.jpg') ), data[i] )  # For test data which does not have a label


# In[ ]:


path = "../input/data"
train_path = Path('../input/data/train')
test_path = Path('../input/data/test')


# In[ ]:


if train:
    save_imgs(train_path,train_x,train_y)

if train & verbose >= 1:
    print((train_path).ls())


# In[ ]:


save_imgs(test_path,test_x)

if train & verbose >= 1:
    print((test_path).ls())


# ## Create DataBunch object

# In[ ]:


if train:
    np.random.seed(42)  # Make sure your validation set is reproducible
    # tfms = get_transforms(do_flip=False)  # We dont want to flip the images since numbers are not written flipped
    tfms = get_transforms(do_flip=False, max_rotate=30, max_zoom=1.2, max_lighting=0.4,
                          max_warp=0.3)  # We dont want to flip the images since numbers are not written flipped + more aggressive augmentation

    data = ImageDataBunch.from_folder(path=path,  
                                      valid_pct=0.2,
                                      ds_tfms=tfms,
                                      train='train',
                                      test='test',
                                      size=28,
                                      bs=64).normalize(imagenet_stats)


# In[ ]:


if train & verbose >= 1:
    print(len(data.train_ds))
    print(len(data.valid_ds))
    print(len(data.test_ds))


# In[ ]:


if train & verbose >= 1:
    data.show_batch(rows=6, figsize=(12,12))

# Subtle differences between the following which might be hard to detect
# - 9 and 6
# - 1 and 0
# - 7 and 3


# In[ ]:


if train & verbose >= 1:
    print(data.classes)


# ## Create Learner

# In[ ]:


# Kaggle comes with internet off. So have to copy over model to the location where fastai would have downloaded it.
# https://forums.fast.ai/t/how-can-i-load-a-pretrained-model-on-kaggle-using-fastai/13941/23

get_ipython().system('mkdir -p /tmp/.cache/torch/checkpoints')
get_ipython().system('cp ../input/resnet50/resnet50.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth')


# In[ ]:


if train:
    learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy])


# In[ ]:


if train & verbose >= 1:
    print(learn.model)


# ## Train Model

# In[ ]:


if train:
    learn.fit_one_cycle(4)


# In[ ]:


if train:
    learn.recorder.plot_losses()


# In[ ]:


if train:
    learn.save('resnet50-stage-1')  # Saves it in the directory where the images exist (under a folder called 'models' in there)


# ## Fine tune model

# In[ ]:


if train:
    learn.lr_find()
    learn.recorder.plot()


# In[ ]:


if train:
    learn.unfreeze()  # Train all layers
    learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-3))  # As per lr_finder (initial layers trained on smaller learning rate)


# In[ ]:


if train:
    learn.recorder.plot_losses()


# In[ ]:


if train:
    learn.save('resnet50-stage-2a')


# <font color = red>**Train Loss is still more than the validation loss, so we could train for some more time. --> Tried this with 4 more epochs, but did not seem to reduce train loss below the validation loss.**</font>[](http://)

# In[ ]:


if train:
    learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-3))  


# In[ ]:


if train:
    learn.recorder.plot_losses()


# In[ ]:


if train:
    learn.save('resnet50-stage-2b')


# <font color = red>**Train Loss is still more than validation loss, but we will use this for now (though we could train for some more time to make sure train error is below the validation error).**</font>

# In[ ]:


if train:
    learn.export(file = Path("/kaggle/working/export.pkl"))  # Not needed unless you want to download the model and then upload as a dataset


# ## Visualise Results and Metrics

# In[ ]:


if train:
    interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


if train:
    interp.plot_confusion_matrix()


# In[ ]:


if train:
    print(interp.most_confused())


# In[ ]:


if train:
    interp.plot_top_losses(9, figsize=(10,10))


# ## Predictions

# In[ ]:


# https://docs.fast.ai/tutorial.inference.html#A-classification-problem
if train:
    deployed_path = "/kaggle/working/"
else:
    deployed_path = "../input/kannada-mnist-resnet50/"  # If using a deployed model (not using right now)
    
print(deployed_path)


# In[ ]:


learn = load_learner(deployed_path, test=ImageList.from_folder(test_path))


# In[ ]:


# Adapted from another kernel: https://www.kaggle.com/demonplus/kannada-mnist-with-fast-ai-and-resnet
test_preds_probs, _ = learn.get_preds(DatasetType.Test)
test_preds = torch.argmax(test_preds_probs, dim=1)
if verbose >= 1:
    print(test_preds_probs)


# In[ ]:


num = len(learn.data.test_ds)
indexes = {}

for i in range(num):
    filename = str(learn.data.test_ds.items[i]).split('/')[-1]
    filename = filename[:-4] # get rid of .jpg
    indexes[(int)(filename)] = i


# In[ ]:


submission = pd.DataFrame({'id': range(0, num), 'label': [test_preds[indexes[x]].item() for x in range(0, num)] })
print(submission)


# In[ ]:


submission.to_csv(path_or_buf ="submission.csv", index=False)


# In[ ]:




