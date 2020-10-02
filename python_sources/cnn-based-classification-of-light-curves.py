#!/usr/bin/env python
# coding: utf-8

# **UPDATE:**
# 
# I made a couple of changes since submitting my first set of predictions, which include scaling the flux observations and doing per-channel convolutions. Together, these two tweaks really improved my model, and ** halfed the test loss from 24 to 12**. I've shared the updates in Bold below (Also updated the TOC headers)

# Hey guys!
# This is my first real kernel for a Kaggle competition, so feedback is greatly appreciated!

# # Table of Contents
# 
# - [Introduction](#Introduction)
# - [Approach](#Approach)
# - [Implementation](#Implementation)
# - [DMDT Images](#DMDT-Images)
#     - [Generating the Images **UPDATED**](#Generating-the-Images)
#     - [Loading the Images](#Loading-the-Images)
# - [Training and Test set](#Training-and-Test-set)
# - [The CNN Model Architecture **UPDATED**](#The-CNN-Model-Architecture)
# - [Custom Loss Function](#Custom-Loss-Function)
# - [Training and Evaluation](#Training-and-Evaluation)

# # Introduction
# 
# A little bit about myself - by day, I'm just a regular software engineer, trying to eek my way into data science. But by night, I'm a huge astrophysics/cosmology buff. I try to keep up with what's happening by reading news, watching videos, and even going through some lectures as well as reading a few papers. So, this competition was literally the icing on the cake for me! It was fascinating to go through the two example kernels - to learn about the [motivation and the astronomical background](https://www.kaggle.com/michaelapers/the-plasticc-astronomy-starter-kit) (super fun stuff!) as well to look at a [sample approach taken by astronomers](https://www.kaggle.com/michaelapers/the-plasticc-astronomy-classification-demo). 
# 
# It was equally (if not more) fun to look at the different kernels posted by everyone here. I'm yet to go through a large chunk of them, so its possible my approach has already been shared by someone, but in general I found some of these particularly interesting/useful - 
# 
# - [The Astronomical (complete) EDA - PLAsTiCC dataset](https://www.kaggle.com/danilodiogo/the-astronomical-complete-eda-plasticc-dataset) - this does a better job at EDA than I could've done.
# - [All Classes Light Curve Characteristics](https://www.kaggle.com/mithrillion/all-classes-light-curve-characteristics)
# - [Using CNNs in time series analysis](https://www.kaggle.com/hrmello/using-cnns-in-time-series-analysis) - this one in particular since my approach was inspired by it.
# 
# I did look at a few basic options such Random Forests/XGBoost, Logistic Regression, etc. But the CNN-based approach seemed to interest me, given that I have some prior experience working with CNNs. Plus, the fact that I sort of started to get into it right away. A short google search yielded a paper that uses CNNs to classify light curves, which I describe below. 
# 

# # Approach
# 
# Based on [Mahabal, Ashish, et al. "Deep-learnt classification of light curves." ](https://arxiv.org/pdf/1709.06257.pdf)
# 
# 
# I came across this paper and I liked it because it directly tackles the problem at hand - classifying light curves based on time series data is a hard problem. Firstly, because there is often an irregular gap in the observations due to a variety of reasons. Secondly, and this is especially true with LSST, there is always the scope of coming across objects that you've never seen before (hence the training set isn't a good representation of the test set). This makes computing features challenging. Generic statistical features don't yield high accuracy. Sometimes, domain-level features are employed (the paper gives an example feature - 'fading profile of a single peaked fast transient'. These features are useful only for certain classes of objects and don't generalize well.
# 
# The authors present a different approach - to use neural networks to classify objects, since neural networks are good at extracting features from data. In particular, they use Convolutional Neural Networks, which have proven their metal in image classification. You can read more about CNNs [here](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/). 
# 
# In short, the idea is to encode the flux values into a matrix (or a tensor), and use that as input to a CNN (which I implement using Keras). 
# 
# The encoding approach is interesting, the idea is to capture changes in flux values at different magnitude and time scales. For each pair of points on the light curve, compute the difference in magnitude $dm$ and time $dt$ and then put them into bins. Each bin contains counts of all pairs of points that fall within it.  The resulting matrix/image is referred in the paper as a _dmdt_ image. There are $p = {n \choose 2}$ such pairs for n points on the light curve. The sample bins used in the paper (as well my code) are 
# $$dm = \pm [0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8]$$ 
# and 
# $$dt = [\frac{1}{145}, \frac{2}{145}, \frac{3}{145}, \frac{4}{145}, \frac{1}{25}, \frac{2}{25}, \frac{3}{25}, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 10, 20, 30, 60, 90, 120, 240, 600, 960, 2000, 4000]$$
# 
# The bins are in approximately in a semi-logarithmic fashion, so that smaller changes in magnitude and time are spread over many bins, whereas the corresponding larger changes are clubbed together. The flux counts are normalized and stretched to fit between 0 and 255 - $${norm}_{bin} = \lfloor{\frac{255 * {count}_{bin}}{p} + 0.99999}\rfloor$$
# 
# I also tried to create a kernel just to visualize these images and do some basic EDA stuff, but I haven't got around to finishing that, you can find the kernel here - [Plasticc DMDT EDA](https://www.kaggle.com/pankajb64/plasticc-dmdt-eda)
# 
# The authors talk about a single light curve, and hence creating a single 2D matrix, but in our case we have 6 different passbands, so 6 such matrices per object. The simplest approach, which is also the one that I use here, is to simply stack these images to create a 6-channel image and feed that to the CNN. Its not ideal, since different bands are measured at different time instants, and the total number of measurements vary across bands. On the other hand, treating each band as an individual input to the CNN isn't a great idea either (though I haven't tried it so can't be sure) since the different bands combine to uniquely identify the objects, and each band may potentially contain different information. Its not clear to me what the best way to deal with this, so feedback is greatly appreciated.
# 
# I also implemented the weighted multi-class log loss used in this competition as a custom loss function, since I wasn't sure categorical cross-entropy takes class imbalance into account. In my case I set the weight of each class to be 1, since I obviously don't know the actual weights used by the compeition judges.
# 
# So without further ado, let's just get to the code!

# # Implementation

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
import multiprocessing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras.backend as K #to define custom loss function
import tensorflow as tf #We'll use tensorflow backend here
import dask.dataframe as dd
import matplotlib.pyplot as plt

from tqdm import tnrange, tqdm_notebook
from collections import OrderedDict
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, AveragePooling1D, Reshape, DepthwiseConv2D, SeparableConv2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

print(os.listdir("../input/"))


# In[ ]:


df = pd.read_csv('../input/PLAsTiCC-2018/training_set.csv')


# In[ ]:


df.head(5)


# ## DMDT Images

# ### Generating the Images
# 
# As mentioned earlier, in this kernel, I'm not doing EDA, but just getting straight to how to generate the dmdt images, and run the model using them.
# 
# The functions below generate dmdt images (or "dmdtize" the input as I say). I decided to get around the large number of rows by saving the image for each object in a different file, and combining them while loading. That way, I didn't have to re-process images if I happened to stop the run mid-way.
# 
# **UPDATE**:
# 
# During my post-submission analysis, I got a chance to re-evaluate a larger set of dmdt images, and I realized a bulk of them looked very similar even though they were from different classes. So I decided to scale the observations for each individual object to be in the range defined by the flux magnitude bins, i.e. between -8 and 8.

# In[ ]:


def dmdtize_single_band(df, dm_bins, dt_bins, col):
    n_points = df.shape[0]
    dmdt_img = np.zeros((len(dm_bins), len(dt_bins)), dtype='int')
    for i in range(n_points):
        for j in range(i+1, n_points):
            dmi = df.iloc[i][col]
            dmj = df.iloc[j][col]
            dti = df.iloc[i]['mjd']
            dtj = df.iloc[j]['mjd']
            
            dm = dmj - dmi if dtj > dti else dmi - dmj
            dt = abs(dtj - dti)
            
            dm_idx = min(np.searchsorted(dm_bins, dm), len(dm_bins)-1)
            dt_idx = min(np.searchsorted(dt_bins, dt), len(dt_bins)-1)
            
            dmdt_img[dm_idx, dt_idx] += 1
    return dmdt_img


# In[ ]:


def dmdtize_single_object(args):
    (df, object_id, base_dir) = args
    key = '{}/{}_dmdt.pkl'.format(base_dir, object_id)
    if os.path.isfile(key):
        return
    num_bands = 6
    dm_bins = [-8, -5, -3, -2.5, -2, -1.5, -1, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8]
    dt_bins = [1/145, 2/145, 3/145, 4/145, 1/25, 2/25, 3/25, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 10, 20, 30, 60, 90, 120, 240, 600, 960, 2000, 4000]
    dmdt_img = np.zeros((len(dm_bins), len(dt_bins), num_bands), dtype='int')
    
    mms = MinMaxScaler(feature_range=(-8, 8))
    df['local_tr_flux'] = mms.fit_transform(df['flux'].values.reshape(-1,1))
    
    max_points = 0
    for band_idx in range(num_bands):
        df_band = df.loc[df['passband'] == band_idx]
        dmdt_img[:, :, band_idx] = dmdtize_single_band(df_band, dm_bins, dt_bins, 'local_tr_flux')
        if band_idx == 0 or df_band.shape[0] > max_points:
            max_points = df_band.shape[0] #store max points to scale the image later
    
    max_pairs = (max_points*(max_points-1))//2
    dmdt_img = np.floor(255*dmdt_img/max_pairs + 0.99999).astype('int')
    with open(key, 'wb') as f:
        pickle.dump(dmdt_img, f)        


# In[ ]:


def dmdtize(df, base_dir='train'):
    objects = df['object_id'].drop_duplicates().values
    nobjects = len(objects)
    dmdt_img_dict = {}
    pool = multiprocessing.Pool()
    df_args = []
    for obj in objects:
        df_obj = df.loc[df['object_id'] == obj]
        df_args.append((df_obj, obj, base_dir))
    pool.map(dmdtize_single_object, df_args)
    pool.terminate()


# To generate/save the dmdt images, convert thes cell below to code and execute. It takes a while (it took me about 20 minutes on a 16-core machine, using all cores), so I've attached the pre-processed images as a dataset, so they can be loaded up easily.

# ```
# dmdtize(df)
# ```

# ### Loading the Images
# 
# The cells below load the dmdt images from the dataset.

# In[ ]:


objects = df['object_id'].drop_duplicates().values


# In[ ]:


def load_dmdt_images(objects, base_dir='train'):
    dmdt_img_dict = OrderedDict()
    for obj in objects:
        key = '{}/{}_dmdt.pkl'.format(base_dir, obj)
        if os.path.isfile(key):
            with(open(key, 'rb')) as f:
                dmdt_img_dict[obj] = pickle.load(f)
    return dmdt_img_dict


# In[ ]:


dmdt_img_dict = load_dmdt_images(objects, '../input/plasticc_dmdt_images/train/data1/plasticc/input/train')


# ## Training and Test set
# 
# The images are already scaled and pre-processed, so we can just feed them to the CNN. Lets create the input and output vectors for training and test.

# In[ ]:


X = np.array(list(dmdt_img_dict.values()), dtype='int')

df_meta = pd.read_csv('../input/PLAsTiCC-2018/training_set_metadata.csv')
labels = pd.get_dummies(df_meta.loc[df_meta['object_id'].isin(dmdt_img_dict.keys()) , 'target'])

y = labels.values


# In[ ]:


df_meta


# In[ ]:


labels


# In[ ]:


#TODO split X and y into train/test set. (Maybe a val set ?)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
splits = list(splitter.split(X, y))[0]
train_ind, test_ind = splits


# In[ ]:


X_train = X[train_ind]
X_test  = X[test_ind]

y_train = y[train_ind]
y_test  = y[test_ind]


# In[ ]:


print(y_train.shape, y_test.shape)


# 6000 images isn't nearly enough for training a CNN (unless you're doing transfer learning), but we'll have to make do with what we have. Again, feedback appreciated.

# ## The CNN Model Architecture
# 
# The CNN Architecture is outined below. Now, I'm definitely not an expert in building models, but I tried a few different variations and found this to be most well-suited.
# 
# A couple of things things I noted -
# - The dimensions of the input image are unusually small even for a CNN (a typical image size is 256x256 per channel whereas we have a 23x24) and my gut feeling was that the first Convolutional layer should not be max pooled. This turned out to be right - I got better training layer in abscence of the max pooling layer.
# - I didn't try a whole bunch of activations, but using an Exponential Linear Unit (ELU) worked better than ordinary Rectilinear Unit (RELU). This is probably because the internal layers have negative values.
# 
# **UPDATE:**
# After visualizing a few sample images for each class, I noticed that the lack of synchronicity between the passband observations meant that especially for events like supernovae, only some of the bands would capture the event. The remaining bands are essentially noise for the purpose of classification.  In this case a regular convolution kernel involving multiplication across channels might not be able to capture the features. My hunch was if we try to instead convolve individual features separately, we'd stand a better chance to capture those features.
# Luckily, I found Keras had a layer called *Depthwise Convolution* which does exactly what we need. At some point we need to combine those features together, and that's where a *Separable Convolution* comes in, which is essentially Depthwise convolution followed by a regular pointwise convolution. See [Keras Documentation](https://keras.io/layers/convolutional/) for more information about the various Convolutional Layers.
# 

# In[ ]:


def build_model(n_dm_bins=23, n_dt_bins=24, n_passbands=6, n_classes=14):
    model = Sequential()
    model.add(DepthwiseConv2D(kernel_size=2, strides=1, depth_multiplier=8,
                     padding="valid", activation="elu",
                     input_shape=(n_dm_bins, n_dt_bins, n_passbands)))
    model.add(Dropout(0.1))
    model.add(SeparableConv2D(48, kernel_size=2, strides=1, depth_multiplier=2,
                    padding="valid", activation="elu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(32, activation="elu"))
    model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation="softmax"))
    print(model.summary())
    return model


# ## Custom Loss Function
# 
# The code below implements the weighted multi class log loss functions. This is similar to the one defined in the Evaluation section of the project description with one caveat - I have used the proportion of objects in a class instead of the actual counts, since I found it gave more readable loss values.
# 
# A couple of Keras-specific things - 
# - A custom loss function defined in Keras this way must only methods from the keras.backend interface to process its arguments (and not use numpy directly) to ensure that it can be smoothly translated into the appropriate backend (Tensorflow or theano) when compiling the model. 
# - I set Keras to use float32 as the default variable dtype, since float 64 wasn't working for some reason.

# In[ ]:


n_classes=14
#assumes weights to be all ones as actual weights are hidden
#UPDATE - settings weights for classes 15 (idx=1) and 64(idx=7) to 2 based on LB probing post 
#https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194#397153
weights = np.ones(n_classes, dtype='float32') 
weights[1], weights[7] = 2, 2
epsilon = 1e-7
#number of objects per class
class_counts = df_meta.groupby('target')['object_id'].count().values 
#proportion of objects per class
class_proportions = class_counts/np.max(class_counts)
#set backend to float 32
K.set_floatx('float32')


# In[ ]:


#weighted multi-class log loss
def weighted_mc_log_loss(y_true, y_pred):
    y_pred_clipped = K.clip(y_pred, epsilon, 1-epsilon)
    #true labels weighted by weights and percent elements per class
    y_true_weighted = (y_true * weights)/class_proportions
    #multiply tensors element-wise and then sum
    loss_num = (y_true_weighted * K.log(y_pred_clipped))
    loss = -1*K.sum(loss_num)/K.sum(weights)
    
    return loss


# This is just small test of the loss function to see if its compiles and runs fine.

# In[ ]:


y_true = K.variable(np.eye(14, dtype='float32'))
y_pred = K.variable(np.eye(14, dtype='float32'))
res = weighted_mc_log_loss(y_true, y_pred)
K.eval(res)


# ## Training and Evaluation
# 
# Lets build the model now.

# In[ ]:


model = build_model()


# I also print the model summary to get an idea of the number of parameters to be trained. 50000 is a lot of parameters for just 6k input rows, but its nothing compared to the CNNs used for image classification. I found higher number of parameters gave worse results on test.
# 
# Lets compile the model now, I used Adam as the optimizer based on the paper, but I increased the learning rate to 0.002 since it yielded faster convergence.

# In[ ]:


model.compile(loss=weighted_mc_log_loss, optimizer=Adam(lr=0.002), metrics=['accuracy'])


# Training for 20 epochs (I didn't do a validation set since the training set is already too small). I found that higher epochs led to overfitting.

# In[ ]:


checkPoint = ModelCheckpoint("./keras.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=1)


# In[ ]:


class ReduceLRWithEarlyStopping(ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super(ReduceLRWithEarlyStopping, self).__init__(*args, **kwargs)
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        super(ReduceLRWithEarlyStopping, self).on_epoch_end(epoch, logs)
        old_lr = float(K.get_value(self.model.optimizer.lr))
        if self.wait >= self.patience and old_lr <= self.min_lr:
            # Stop training early
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        super(ReduceLRWithEarlyStopping, self).on_epoch_end(logs)
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


# In[ ]:


reduce_lr = ReduceLRWithEarlyStopping(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)


# In[ ]:


history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=[X_test, y_test],shuffle=True,verbose=1,callbacks=[checkPoint, reduce_lr])


# In[ ]:


def plot_loss_acc(history):
    plt.plot(history['loss'][1:])
    plt.plot(history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()
    
    plt.plot(history['acc'][1:])
    plt.plot(history['val_acc'][1:])
    plt.title('model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()


# In[ ]:


plot_loss_acc(history.history)


# Lets evaluate it on the test set now.

# In[ ]:


loss_acc = model.evaluate(X_test, y_test, batch_size=32)
print(loss_acc)


# ~67% accuracy on the training set, and ~55% on the test is good but not great, so really any feedback here is truly appreciated.
# 
# Lets try and look at some of the predictions.

# In[ ]:


y_pred_test = model.predict(X_test)


# In[ ]:


classes = np.sort(df_meta['target'].drop_duplicates())


# In[ ]:


df_meta_test = df_meta.iloc[test_ind]
df_meta_test['pred_label'] = classes[np.argmax(y_pred_test, axis=1)]


# In[ ]:


df_meta_test.loc[df_meta_test.target == 15]


# In[ ]:


df_meta_test.loc[(df_meta_test.target == df_meta_test.pred_label) & (df_meta_test.target != 16) & (df_meta_test.target != 92) & (df_meta_test.target != 88)]


# Just eyeballing the predictions, it doesn't seem like there's any particular class for which its doing good/bad, its just doing decent on average, which is perhaps an effect of the loss function. More investigation is needed here.
# 
# For now, we'll save our model so we can reuse it later.

# In[ ]:


time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
save_file = 'model_{}.h5'.format(time_stamp)
model.save(save_file)


# That's all for now! Feedback appreciated!

# In[ ]:




