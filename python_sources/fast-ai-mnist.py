#!/usr/bin/env python
# coding: utf-8

# # Fast.AI.MNIST
# 
# After playing with plain `numpy` neutral network implementation in NNNN - a [Naive Neutral Network iNtroduction](https://www.kaggle.com/alukashenkov/nnnn-a-naive-neutral-network-introduction) kernel I intended to try to implement a similar model using a popular deep learning framework. However, I really got inspired by Fast.AI approach demonstrated in Lesson 1 of the [course](https://course.fast.ai). Also, it helps a lot in the learning process to take a concept and to extend it to a similar, but still different case.
# 
# So, here is my take on MNIST dataset with the Fast.AI toolbox.
# 
# As always, if you see a bug, please let me know at alukashenkov@gmail.com. If you find this piece of use and interest, please upvote.

# ## Acknowledgments
# 
# I took initial inspiration from [MNIST Classification using Fast AI V2](https://www.kaggle.com/vijaykris/mnist-classification-using-fast-ai-v2) kernel. 
# 
# [Practical Deep Learning for Coders](https://course.fast.ai) course materials were of great help. I guess I'm using inputs from first three lessons in this kernel. There are [incredible notes](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-1-602f73869197) for this course created by [Hiromi Suenaga](https://medium.com/@hiromi_suenaga) that make the content of video lectures "searchable."
# 
# [Data Augmentation Experimentation](https://towardsdatascience.com/data-augmentation-experimentation-3e274504f04b) article by [Amrit Virdee](https://medium.com/@pillview) helped a lot to understand how to work with data augmentation in Fast.AI.

# ## Environment

# Apparently, it is hard to manage the environment, I ran into this issue both here on Kaggle and on Google Colab. Although it is possible to change packages versions with `!pip`, it would work only for a particular session. However, having these installation steps in the script is not that practical, as it takes precious kernel runtime.

# In[ ]:


get_ipython().system('pip show fastai')


# ## Getting Data and Preparing Train/CV Sets
# Importing libraries.

# In[ ]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')

import gc
import datetime as dt


# In[ ]:


# Start timer
start = dt.datetime.now()


# Definig some useful constants.

# In[ ]:


#set the path
PATH = "../input"
os.listdir(PATH)


# In[ ]:


img_size = 28               # Known from dataset description
batch_size = 64             # Opting for a default value for now


# Load train and test data.

# In[ ]:


train_data = pd.read_csv(f'{PATH}/train.csv', header='infer')
test_data  = pd.read_csv(f'{PATH}/test.csv', header='infer')

# Getting number of training examples from train data shape
m = train_data.shape[0]
print("Number of samples in training data is:", m)


# The above training file contains both images and labels. These have to be split. The first column is a label.

# In[ ]:


#Pixels
X_train_data = train_data.iloc[:,1:]
X_train_data = X_train_data.values
X_test_data = test_data.values       # this one has no labels to be removed

# Labels
Y_train_data = train_data.iloc[:,0:1]
Y_train_data = Y_train_data.values

# Little clean up
del train_data
del test_data
gc.collect()


# Since the pre-trained model resnet has 3 channels, we will have to multiply the channels by 3 the test and train data.

# In[ ]:


# Converting to the square image
print("Original train image data shape:", X_train_data.shape)
X_train_data = X_train_data.reshape(-1, img_size, img_size)

#Add missing color channels to previously reshaped image
X_train_data = np.stack((X_train_data,) * 3, axis = -1).astype('uint8') 
X_train_data = X_train_data/255                                           # Normalizing data as we go, it seems to make a lot of difference
print("Resulting train image data shape:", X_train_data.shape)

# Same conversion for test data as well
X_test = X_test_data.reshape(-1, img_size, img_size)
X_test = np.stack((X_test,) * 3, axis = -1).astype('uint8')
X_test = X_test/255                                                      # Same processing for test as for training


# In[ ]:


def print_10_images(X, y):
    plt.rcParams['figure.figsize'] = (50.0, 50.0) # set default size of plots
    plt.axis('off')
    
    num_images_to_show = 10
    m = X.shape[0]
    
    for i in range(num_images_to_show):
        sample = np.random.randint(0, m - 1)
        
        plt.subplot(2, num_images_to_show, i + 1)
        plt.imshow(X[sample], interpolation = 'nearest')
        plt.rcParams['figure.figsize'] = (50.0, 50.0) # set default size of plots
        
        plt.title("Index: " + str(sample) + " \n Label: " + str(y[sample]), fontsize=25)
            
    plt.axis('on')
    plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
        
def plot_labels_distribution(Y_train, Y_cv, sharey = False):
    _, axes = plt.subplots(1, 2, figsize = (10, 3), sharey = sharey)
    sns.countplot(Y_train.flatten(), ax = axes[0])
    sns.countplot(Y_cv.flatten(), ax = axes[1])


# Let's see what we've got to work with.

# In[ ]:


print_10_images(X_train_data, Y_train_data)


# We need to split `train_data` into training and cross-validation sets.

# In[ ]:


test_size = 0.05      # Proportion of CV set to be extracted from training data

Y_train, Y_cv = train_test_split(Y_train_data, test_size = test_size, shuffle = True, stratify = None, random_state = 1974)
plot_labels_distribution(Y_train, Y_cv)


# The distribution of classes looks fine, but it is not perfect. Stratifying should help, so also including image data into split along with the labels.

# In[ ]:


X_train, X_cv, Y_train, Y_cv = train_test_split(X_train_data, Y_train_data, test_size = test_size, shuffle = True, stratify = Y_train_data, random_state = 1974)
plot_labels_distribution(Y_train, Y_cv)


# In[ ]:


# Final touches
Y_train = Y_train.flatten()
Y_cv = Y_cv.flatten()

del X_train_data
del Y_train_data
gc.collect()


# ## The Model
# 
# Now we are coming to the famous "three lines of code" model definition. Let's follow the steps outlined in the course to get it working.

# In[ ]:


# Loading data into the Fast.AI structure
arch = resnet34
data = ImageClassifierData.from_arrays('tmp/', (X_train, Y_train), (X_cv, Y_cv), classes = np.unique(Y_cv), bs = batch_size, tfms = tfms_from_model(arch, img_size))
learn = ConvLearner.pretrained(arch, data, precompute = True)


# In[ ]:


# Find learning rate
learn.lr_find()


# In[ ]:


learn.sched.plot()


# Learning rate of 0.01 seems to be a good default choice. It is important to notice that by default only the output layer of the model is affected by fitting.

# In[ ]:


learn.fit(lrs = 0.01, n_cycle = 3)


# All in all, accuracy is not that impressive. After all, the simplest possible model with one output layer (inputs pixels are directly mapped to softmax activations) would give accuracy at around 90%. However, with only the output layer being fitted by default we are very close to that.

# In[ ]:


# This gives prediction for validation set. Predictions are in log scale
log_preds = learn.predict()

# From log probabilities to 0 or 1
probs = np.exp(log_preds)  

# From probabilities to classes
preds = np.argmax(probs, axis=1)  


# Sanity check - what images the model gets wrong? In my case, most of the pictures I see still easily recognisable.

# In[ ]:


def print_10_mislabeled_images(X, y, p):
    
    plt.rcParams['figure.figsize'] = (50.0, 10.0) # set default size of plots
    plt.axis('off')
    
    num_images_to_show = 10
    
    a = p - y
    mislabeled_indices = np.asarray(np.where(a != 0))

    for i in range(num_images_to_show):
        sample = np.random.randint(0, mislabeled_indices.shape[1])
        index = mislabeled_indices[0][sample]
        
        plt.subplot(2, num_images_to_show, i + 1)
        plt.imshow(X[index], interpolation = 'nearest')
        plt.title("Prediction: " + str(p[index]) + " \n Label: " + str(y[index]), fontsize=25)
        
    plt.axis('on')
    plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
    
def print_all_mislabeled_images(X, y, p):
    
    plt.rcParams['figure.figsize'] = (50.0, 15.0) # set default size of plots
    plt.axis('off')
    
    num_images_in_row = 10
    
    a = p - y
    mislabeled_indices = np.asarray(np.where(a != 0))
    num_rows = mislabeled_indices.shape[1] // num_images_in_row + 1
    
    for i in range(mislabeled_indices.shape[1]):
        index = mislabeled_indices[0][i]
        
        plt.subplot(num_rows, num_images_in_row, i + 1)
        plt.imshow(X[index], interpolation = 'nearest')
        plt.title("Prediction: " + str(p[index]) + " \n Label: " + str(y[index]), fontsize=25)
        
    plt.axis('on')
    plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots


# In[ ]:


print_10_mislabeled_images(X_cv, Y_cv, preds)


# ## Improving the Model

# By default when we create a learner, it sets all but the last layer to *frozen*. That means that it's still only updating the weights in the last layer when we call `fit`.

# In[ ]:


learn.precompute = False
learn.fit(lrs = 1e-2, n_cycle = 3, cycle_len = 1)

learn.sched.plot_lr()


# Now that we have a good final layer trained, we can try fine-tuning the other layers. To tell the learner that we want to unfreeze the remaining layers, just call `unfreeze()`.

# In[ ]:


learn.unfreeze()


# Generally speaking, the earlier layers (as we've seen) have more general-purpose features. Therefore we would expect them to need less fine-tuning for new datasets. For this reason, we will use different learning rates for different layers. We refer to this as differential learning rates, although there's no standard name for this technique in the literature that we're aware of.

# In[ ]:


lr = 0.01
lrs = np.array([lr/9,lr/3,lr])                # Using bigger LR for early layers as our images are quite different from the ImageNet

learn.fit(lrs = lr, n_cycle = 3, cycle_len = 1, cycle_mult = 2)


# In[ ]:


learn.sched.plot_lr()


# Note that's what being plotted above is the learning rate of the final layers. The learning rates of the earlier layers are fixed at the same multiples of the last layer rates as we initially requested.
# 
# Loss graph for reference.

# In[ ]:


learn.sched.plot_loss()


# Let's see what we still get wrong? Actually, there are only a bunch of weird images in the validation set that the model couldn't classify correctly.

# In[ ]:


# This gives prediction for validation set. Predictions are in log scale
log_preds = learn.predict()

# From log probabilities to 0 or 1
probs = np.exp(log_preds)  

# From probabilities to classes
preds = np.argmax(probs, axis=1)  

print_all_mislabeled_images(X_cv, Y_cv, preds)


# Also, looking at `trn_loss` and `val_loss` it is clear that the model starts to overfit. So, I would assume that further training wouldn't yield much better results - we need to implement data augmentation.

# ## Data Augmentation

# Let's first look at the confusion matrix.

# In[ ]:


cm = confusion_matrix(Y_cv, preds)

plt.rcParams['figure.figsize'] = (8.0, 8.0) # set default size of plots
plot_confusion_matrix(cm, data.classes)
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots


# To implement data augmentation, we need to create new learner wish updated transformations structure `tfms`. As we are working with digits, we should be careful selecting augmentations as compared to simple images classification (e.g., we can't flip images). As we work with relatively small images, all augmentation parameters should be fairly subtle.

# In[ ]:


def show_img(ims, idx, figsize = (50, 50), ax = None):
    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    ims = np.rollaxis(to_np(ims), 1, 4)
    ax.imshow(np.clip(ims, 0, 1)[idx], interpolation = 'nearest')
    ax.axis('off')


# In[ ]:


# Loading data into the Fast.AI structure
arch = resnet34

aug_tfms = [RandomRotate(30, p = 0.75, mode = cv2.BORDER_REFLECT, tfm_y = TfmType.NO),
            AddPadding(pad = 2, mode = cv2.BORDER_REPLICATE)]

tfms = tfms_from_model(arch, img_size, aug_tfms = aug_tfms, max_zoom = 1.1)
data = ImageClassifierData.from_arrays('tmp/', (X_train, Y_train), (X_cv, Y_cv), classes = np.unique(Y_cv), bs = batch_size, tfms = tfms)


# Let's look at samples of augmented images.

# In[ ]:


batches = [next(iter(data.aug_dl)) for i in range(10)]

for pos in range(len(batches)):
    fig, axes = plt.subplots(1, 10, figsize=(25,8))
    for i,(x,y) in enumerate(batches):
        show_img(x, pos, ax = axes.flat[i])


# In[ ]:


learn = ConvLearner.pretrained(arch, data, precompute = True)

# Find learning rate
learn.lr_find()

learn.sched.plot()


# In[ ]:


lr = 0.02                                
learn.fit(lrs = lr, n_cycle = 3)


# In[ ]:


learn.precompute = False
learn.fit(lrs = lr, n_cycle = 3, cycle_len = 1)


# In[ ]:


learn.unfreeze()

learn.lr_find()
learn.sched.plot()


# In[ ]:


lr = 0.003
lrs = np.array([lr/9, lr/3, lr])
learn.fit(lrs = lrs, n_cycle = 3, cycle_len = 1, cycle_mult = 2)
learn.sched.plot_loss()


# It is clear that the model still can benefit from the time invested in further training. But first, let's check out TTA.

# In[ ]:


# This gives prediction for validation set. Predictions are in log scale
log_preds, y = learn.TTA()

# From log probabilities to 0 or 1
probs = np.exp(log_preds) 

# From probabilities to classes
preds = np.argmax(np.mean(probs, axis = 0), axis = 1)

print_all_mislabeled_images(X_cv, Y_cv, preds)


# In[ ]:


cm = confusion_matrix(y, preds)

plt.rcParams['figure.figsize'] = (8.0, 8.0) # set default size of plots
plot_confusion_matrix(cm, data.classes)
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots


# ## Putting everything together.
# 
# I now want to start from scratch and put together all I have learned. The goal is to get to accuracy > 0.999 or better.

# In[ ]:


arch = resnet152
batch_size = 64

def get_data(sz, bs, trn = (X_train, Y_train), val = (X_cv, Y_cv), test = X_test):
    aug_tfms = [RandomRotate(30, p = 0.75, mode = cv2.BORDER_REFLECT, tfm_y = TfmType.NO),
                RandomZoom(zoom_max = 0.05),
                #RandomStretch(max_stretch = 0.05),
                AddPadding(pad = 2, mode = cv2.BORDER_REPLICATE)]
    tfms = tfms_from_model(arch, sz, aug_tfms = aug_tfms, max_zoom = 1.1)
    data = ImageClassifierData.from_arrays('tmp/', trn, val, classes = np.unique(Y_cv), bs = bs, tfms = tfms, test = test)
    return data

data = get_data(img_size, batch_size, (X_train, Y_train), (X_cv, Y_cv), X_test)


# In[ ]:


# Using Adam optimizer
learn = ConvLearner.pretrained(arch, data, opt_fn = optim.Adam, precompute = True)

learn.lr_find()
learn.sched.plot()


# In[ ]:


lr = 0.001

learn.fit(lr, n_cycle = 1, cycle_len = 1)

learn.precompute = False
learn.fit(lr, n_cycle = 4, cycle_len = 1, cycle_mult = 2)
learn.save('pre_fit')


# I observed no material improvements in model accuracy after 4 cycles (15 epochs). So, let's now unfreeze and fit model with all layers, but first, let's see if any changes to the learning rate are needed.

# In[ ]:


learn.load('pre_fit')
learn.unfreeze()

learn.lr_find()
learn.sched.plot()


# I'm also using weight decay parameter (as described in https://www.fast.ai/2018/07/02/adam-weight-decay/)

# In[ ]:


lr = 0.0003
wd = 0.0001

learn.fit(lrs = [lr/9, lr/3, lr], n_cycle = 5, wds = [wd/100, wd/10, wd], use_wd_sched = True, cycle_len = 1, cycle_mult = 2)


# In[ ]:


learn.sched.plot_loss()


# In[ ]:


# This gives prediction for validation set. Predictions are in log scale
log_preds, y = learn.TTA()

# From log probabilities to 0 or 1
probs = np.exp(log_preds) 

# From probabilities to classes
preds = np.argmax(np.mean(probs, axis = 0), axis = 1)

print_all_mislabeled_images(X_cv, Y_cv, preds)


# In[ ]:


cm = confusion_matrix(Y_cv, preds)

plt.rcParams['figure.figsize'] = (8.0, 8.0) # set default size of plots
plot_confusion_matrix(cm, data.classes)
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots


# ## Testing and Submitting

# In[ ]:


# This gives prediction for validation set. Predictions are in log scale
log_preds, y = learn.TTA(is_test = True)

# From log probabilities to 0 or 1
probs = np.exp(log_preds) 

# From probabilities to classes
preds = np.argmax(np.mean(probs, axis = 0), axis = 1)

# Saving predictions
results = pd.Series(preds, name = "Label")
submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), results], axis = 1)
submission.to_csv("submission.csv", index = False)


# In[ ]:


# Remove temp files as it seems to help to avoid error on the kernel submission
get_ipython().system("rm -rf '../working/tmp'")


# In[ ]:


# Collect and report execution timings
end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))


# In[ ]:




