#!/usr/bin/env python
# coding: utf-8

# ## Effective augmentation - mixup & cutout/random erasing
# Thanks to [JihaoLiu](https://www.kaggle.com/jihaoliu), he introduced "mixup" in [his discussion post](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47730) at [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/).
# 
# And there's also similarly effective augmentation method cutout[2] or random erasing[3] introduced in [yu4u's github repository](https://github.com/yu4u/mixup-generator). These are the same basically.
# 
# Paper:
# - [1] [Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," in 
# arXiv:1710.09412, 2017 \[cs.LG\]](https://arxiv.org/abs/1710.09412)
# - [2] [T. DeVries and G. W. Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout," in arXiv:1708.04552, 2017.](https://arxiv.org/abs/1708.04552)
# - [3] [Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random Erasing Data Augmentation," in arXiv:1708.04896, 2017.](https://arxiv.org/abs/1708.04896)
# 
# I confirmed to be useful for this acoustic scene classification task as well.
# 
# Here's what's used in my code for mixup. Cutout/random erasing is also introduced at the bottom of this post.

# In[ ]:


import numpy as np

def mixup(data, one_hot_labels, alpha=1, debug=False):
    np.random.seed(42)

    batch_size = len(data)
    weights = np.random.beta(alpha, alpha, batch_size)
    index = np.random.permutation(batch_size)
    x1, x2 = data, data[index]
    x = np.array([x1[i] * weights [i] + x2[i] * (1 - weights[i]) for i in range(len(weights))])
    y1 = np.array(one_hot_labels).astype(np.float)
    y2 = np.array(np.array(one_hot_labels)[index]).astype(np.float)
    y = np.array([y1[i] * weights[i] + y2[i] * (1 - weights[i]) for i in range(len(weights))])
    if debug:
        print('Mixup weights', weights)
    return x, y


# ### Example
# First, prepare data.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
import keras

datadir = "../input/"

# Load data as is
X_train_org = np.load(datadir+'X_train.npy')
X_test = np.load(datadir+'X_test.npy')
y_labels_train = pd.read_csv(datadir+'y_train.csv', sep=',')['scene_label'].tolist()

# Make label list and converters
labels = sorted(list(set(y_labels_train)))
label2int = {l:i for i, l in enumerate(labels)}
int2label = {i:l for i, l in enumerate(labels)}

# Map y_train to int labels
y_train_org = keras.utils.to_categorical([label2int[l] for l in y_labels_train])

# Train/Validation split --> X_train/y_train, X_valid/y_valid
splitlist = pd.read_csv(datadir+'crossvalidation_train.csv', sep=',')['set'].tolist()
X_train = np.array([x for i, x in enumerate(X_train_org) if splitlist[i] == 'train'])
X_valid = np.array([x for i, x in enumerate(X_train_org) if splitlist[i] == 'test'])
y_train = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'train'])
y_valid = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'test'])


# In[ ]:


# Now augment by mixup --> X_train + mixup'ed X_train, y_train + mixup'ed y_train
# ** This is slight deviation from original idea in paper,
#    using mixup as preprocessing. **
tmp_X, tmp_y = mixup(X_train, y_train, alpha=1)
X_train, y_train = np.r_[X_train, tmp_X], np.r_[y_train, tmp_y]


# ### Visualize mixup results
# Example above shows double the entire dataset by mixup, here let's use just small number of samples and  see how it is mixed-up. 

# In[ ]:


# Here we pick first five samples, and mix-up.
five_X, five_y = X_train[:5], y_train[:5]
mixup_X, mixup_y = mixup(five_X, five_y, alpha=3, debug=True)


# In[ ]:


# Visualize them.
def plot_dataset(XYs, titles):
    for i, (x, y) in enumerate(XYs):
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
        axL.pcolormesh(x)
        axL.set_title('%s [%d]' % (titles[0], i))
        axL.grid(True)
        axR.pcolormesh(y)
        axR.set_title('%s [%d]' % (titles[1], i))
        axR.grid(True)
        plt.show()

plot_dataset(zip(five_X, mixup_X), ('Original data', 'Mixup result'))# Here we pick first two samples, and mix-up.


# In[ ]:


print(mixup_y)


# You can see both X and label y are mixed (except for 0th due to mixed itself).
# 
# ### How it worked
# - Effective to prevent overfitting. Train/valid accuracy gets closer.
# - Improves 1-2% so far in my attempts.
# 
# ### Deviation from original paper[1]
# I introduced mixup as preprocessing to augment training samples.
# But it is basically intended to use in every mini-batch.
# ```python
# for (x1, y1), (x2, y2) in zip(loader1, loader2):
#     lam = numpy.random.beta(alpha, alpha)
#     x = Variable(lam * x1 + (1. - lam) * x2)
#     y = Variable(lam * y1 + (1. - lam) * y2)
#     optimizer.zero_grad()
#     loss(net(x), y).backward()
#     optimizer.step()
# ```
# (quoted from paper)
# 
# ### For Keras users, better implementation there
# You can use the following with `ImageDataGenerator`, it's useful. It also contains cutout/random_erasing.
# 
# - [github/yu4u/mixup-generator, An implementation of "mixup: Beyond Empirical Risk Minimization"](https://github.com/yu4u/mixup-generator)
# 
# You can use them as follows (quoted from it's sample)
# ```python
# datagen3 = ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True,
#     preprocessing_function=get_random_eraser(v_l=0, v_h=255)) ## <==== Cutout/randome erasing part.
# 
# generator3 = MixupGenerator(x_train, y_train, alpha=1.0, datagen=datagen3)() ## <=== Plus mixup.
# ```
# 
# _CAUTION:_ Be careful to feed `v_l` & `v_h` to get_randome_eraser correctly, or your data will be wrongly augmented. The default 0/255 doesn't work and will generate wrong image.
# 
# Here's from my code.
# ```python
# datagen = ImageDataGenerator(
#     featurewise_center=True,  # set input mean to 0 over the dataset
#        :
#    preprocessing_function=get_random_eraser(v_l=np.min(X_train), v_h=np.max(X_train)) # Trainset's boundaries.
# )
# mygenerator = MixupGenerator(X_train, y_train, alpha=1.0, batch_size=batch_size, datagen=datagen)
#     :
# model.fit_generator(mygenerator(),
#                     steps_per_epoch=X_train.shape[0] // batch_size,
#                     epochs=epochs,
#                     validation_data=test_datagen.flow(X_valid, y_valid), callbacks=callbacks)
# ```
# 
# 

# In[ ]:




