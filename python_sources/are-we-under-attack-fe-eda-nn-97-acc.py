#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Introduction
# 
# In this kernel, we will be looking at the NSL-KDD training and testing sets which provide significant information on internet traffic record data. These data sets contain the records of the internet traffic seen by a simple intrusion detection network and is considered to be a benchmark for modern-day internet traffic.
# 
# Let's first take a look at what is in these datasets.

# In[ ]:


init_train_df = pd.read_csv('../input/nslkdd/kdd_train.csv')
init_test_df = pd.read_csv('../input/nslkdd/kdd_test.csv')


# In[ ]:


init_train_df.head()


# In[ ]:


init_test_df.head()


# We see that in both, training and testing, sets there are 42 features per record, with 41 of the features referring to the traffic input itself and one label feature referring  to what type of activity the record is.
# 
# We also see that there are appear two be some distinct "groups" of features and give us some hints as how we could manipulate our data for model building; Specifically:
# * Categorical Features: Some features have named values identifying someting in the feature, such as *protocol_type* which tells us protocol is happening in the oberservation or *flag* which identifies what flag occurs during this record. For these features, we will most likely need to one-hot encode them.
# * Numeric Count Features: Features like *duration*, *src_bytes*, *dst_bytes*, etc seem to be integer counts of what they track, respectively. So far, we're not sure of the true scope of them, but they could have a high spanning distribution if we take *dst_bytes* as an example; From the head of the feature, we see it can be anywhere from 0 to 8153 or beyond. We will most likely have to normalize these features somehow.
# * Numeric Rate Features: Features with *_rate* as a tail seem to be float values ranging from 0 to 1.0, obviously signifying the rate of something they respectively represent. As these features are in a *benchmark* dataset, we will assume that these will not need much, if any, touch-up.
# 
# Let's confirm or refute our initial observations by doing some exploratory data analysis.

# # Exploratory Data Analysis
# 
# ## Dataset Quality
# 
# First things first, let's see if there are any null or missing values in our datasets.

# In[ ]:


init_train_df.isnull().sum()


# Seems we're all good with the training set!

# In[ ]:


init_train_df.isnull().sum()


# ...And we're good with the testing set as well!

# ## Types of Features and Scope

# To verify our ideas on the types of feature groups we have, we look at the summary of our dataframes and number of unique values in their features. 

# In[ ]:


init_train_df.info()
init_train_df.nunique()


# In[ ]:


init_test_df.info()
init_test_df.nunique()


# We see that our initial impressions had some truth behind them!
# 
# Both sets possess only four object/categorical features all with relatively reasonable amounts of levels. Specifically, these features are *duration*, *protocol_type*, *service* and *labels.* We also see two concerns: 
# * There is a some degree of mismatch in the levels of the *service* and *labels* features between our testing set. We will need to reconcile these disparities somehow, especially if we do one-hot encoding, to maintain dimensional fidelity during model training and predicting.
# * The *service* feature has up to 70 unique values. This level of cardinality may be problematic, but we will choose whether to minimize it or not depending on how well our model performs.
# 
# Both sets contain certain integer features with large ranges. Specifically, these features are:
# * *duration*
# * *src_bytes*
# * *dst_bytes*
# * *hot*
# * *num_compromised*
# * *num_root*
# * *count*
# * *srv_count*
# * *dst_host_count*
# * *dst_host_srv_count*
# 
# We also see we might be right about the *rate* features, as they have up to 101 features (zero to one?). 
# 
# Let's dig deeper by visualizing our data.

# ## Visualizations

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 


# For our first visualization, we'll graph the counts of each target label in our datasets.

# In[ ]:


fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)

fig.suptitle(f'Counts of Observation Labels', fontsize=25)

sns.countplot(x="labels", 
            palette="OrRd_r", 
            data=init_train_df, 
            order=init_train_df['labels'].value_counts().index,
            ax=ax1)

ax1.set_title('Train Set', fontsize=20)
ax1.set_xlabel('label', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)

sns.countplot(x="labels", 
            palette="GnBu_r", 
            data=init_test_df, 
            order=init_test_df['labels'].value_counts().index,
            ax=ax2)

ax2.set_title('Test Set', fontsize=20)
ax2.set_xlabel('label', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)

plt.show()


# Immediately we see that we have a significant skew in our data, which is mostly observations of normal behavior and neptune attacks, which is an attack where the attacker exploits flaws in the three-way-handshake of the TCP protocol and continuously sends a large number of successive spoofed SYN packets.
# 
# When we do our preprocessing, we will downsample these observations so that we use a number more inline with number of other types of attacks in our dataset.
# 
# Now we will visualize our integer type features with large ranges in histograms. To make it easier for us, we write up a simple plot/subplot function:

# In[ ]:


def plot_hist(df, cols, title):
    grid = gridspec.GridSpec(10, 2, wspace=0.5, hspace=0.5) 
    fig = plt.figure(figsize=(15,25)) 
    
    for n, col in enumerate(df[cols]):         
        ax = plt.subplot(grid[n]) 

        ax.hist(df[col], bins=20) 
        #ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{col} distribution', fontsize=15) 
    
    fig.suptitle(title, fontsize=20)
    grid.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.show()


# In[ ]:


hist_cols = [ 'duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
    
plot_hist(init_train_df, hist_cols, 'Distributions of Integer Features in Training Set')


# In[ ]:


hist_cols = [ 'duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']
    
plot_hist(init_test_df, hist_cols, 'Distributions of Integer Features in Testing Set')


# Although these initial plots are a bit hard to read for certain ranges (due to the skewed overall distribution), we clearly see how these features have bimodal (even multimodal) distributions wtih a large range. It's clear that to make these features more manageable, we will need to normalize them.
# 
# We repeat the same type of visualizations for our rate columns.

# In[ ]:


rate_cols = [ 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

plot_hist(init_train_df, rate_cols, 'Distributions of Rate Features in Training Set')


# In[ ]:


rate_cols = [ 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

plot_hist(init_test_df, rate_cols, 'Distributions of Rate Features in Testing Set')


# Again, in all these features, we see the same bimodal distributions we saw in the integer type features. Although, in this case, we did verify that the ranges of our rate features are from 0 to 1.

# # Data Preprocessing
# 
# Its obvious that with the current states of our sets, we will not be able to get any meaningful insight from visualizations or other methods until we remove the skew from our data.
# 
# So first we will downsample our *normal* and *neptune* observations in our training set, so that there are only 5000 observations, and in our testing set, so that there are only 1000 observations, of both.

# In[ ]:


random_state = 42
 
proc_train_df = init_train_df.copy()                                                                      # create a copy of our initial train set to use as our preproccessed train set.
proc_test_df = init_test_df.copy()                                                                        # create a copy of our initial test set to use as our preproccessed test set.

proc_train_normal_slice = proc_train_df[proc_train_df['labels']=='normal'].copy()                         # get the slice of our train set with all normal observations
proc_train_neptune_slice = proc_train_df[proc_train_df['labels']=='neptune'].copy()                       # get the slice of our train set with all neptune observations

proc_test_normal_slice = proc_test_df[proc_test_df['labels']=='normal'].copy()                            # get the slice of our test set with all normal observations
proc_test_neptune_slice = proc_test_df[proc_test_df['labels']=='neptune'].copy()                          # get the slice of our test set with all neptune observations

proc_train_normal_sampled = proc_train_normal_slice.sample(n=5000, random_state=random_state)             # downsample train set normal slice to 5000 oberservations
proc_train_neptune_sampled = proc_train_neptune_slice.sample(n=5000, random_state=random_state)           # downsample train set neptune slice to 5000 oberservations

proc_test_normal_sampled = proc_test_normal_slice.sample(n=1000, random_state=random_state)               # downsample test set normal slice to 1000 oberservations
proc_test_neptune_sampled = proc_test_neptune_slice.sample(n=1000, random_state=random_state)             # downsample test set neptune slice to 5000 oberservations

proc_train_df.drop(proc_train_df.loc[proc_train_df['labels']=='normal'].index, inplace=True)              # drop initial train normal slice
proc_train_df.drop(proc_train_df.loc[proc_train_df['labels']=='neptune'].index, inplace=True)             # drop initial train neptune slice

proc_test_df.drop(proc_test_df.loc[proc_test_df['labels']=='normal'].index, inplace=True)                 # drop initial test normal slice
proc_test_df.drop(proc_test_df.loc[proc_test_df['labels']=='neptune'].index, inplace=True)                # drop initial test neptune slice

proc_train_df = pd.concat([proc_train_df, proc_train_normal_sampled, proc_train_neptune_sampled], axis=0) # add sampled train normal and neptune slices back to train set
proc_test_df = pd.concat([proc_test_df, proc_test_normal_sampled, proc_test_neptune_sampled], axis=0)     # add sampled test normal and neptune slices back to test set


# In[ ]:


fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)

fig.suptitle(f'Counts of Observation Labels', fontsize=25)

sns.countplot(x="labels", 
            palette="OrRd_r", 
            data=proc_train_df, 
            order=proc_train_df['labels'].value_counts().index,
            ax=ax1)

ax1.set_title('Train Set', fontsize=20)
ax1.set_xlabel('label', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)

sns.countplot(x="labels", 
            palette="GnBu_r", 
            data=proc_test_df, 
            order=proc_test_df['labels'].value_counts().index,
            ax=ax2)

ax2.set_title('Test Set', fontsize=20)
ax2.set_xlabel('label', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)

plt.show()


# Looking much better!
# 
# However, we see that some target labels have very few observations and that some types of attacks are only observed in the test set. We will now kill two birds with one stone, by keeping the labels of attacks, with enough observations, as they are and changing the label values of all other attacks to *Other.*
# 
# Specifically, we will hold onto the *normal*, *neptune*, *satan*, *ipsweep*, *portsweep*, *smurf*, *nmap*, *back*, *teardrop*, *warezclient* values. 

# In[ ]:


keep_labels = ['normal', 'neptune', 'satan', 'ipsweep', 'portsweep', 'smurf', 'nmap', 'back', 'teardrop', 'warezclient']

proc_train_df['labels'] = proc_train_df['labels'].apply(lambda x: x if x in keep_labels else 'other')
proc_test_df['labels'] = proc_test_df['labels'].apply(lambda x: x if x in keep_labels else 'other')


# In[ ]:


fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)

fig.suptitle(f'Counts of Observation Labels', fontsize=25)

sns.countplot(x="labels", 
            palette="OrRd_r", 
            data=proc_train_df, 
            order=proc_train_df['labels'].value_counts().index,
            ax=ax1)

ax1.set_title('Train Set', fontsize=20)
ax1.set_xlabel('label', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)

sns.countplot(x="labels", 
            palette="GnBu_r", 
            data=proc_test_df, 
            order=proc_test_df['labels'].value_counts().index,
            ax=ax2)

ax2.set_title('Test Set', fontsize=20)
ax2.set_xlabel('label', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)

plt.show()


# Looks like we were successful in our discretization, however, we see that there are significantly more *Other* observations in the test set than the train set. We can solve this easily, however, by taking a sample of the *Other* observations in the test set and transferring them to the train set (specifically 80%)!

# In[ ]:


from sklearn.model_selection import train_test_split

seed_random = 718

proc_test_other_slice = proc_test_df[proc_test_df['labels']=='other'].copy()

proc_train_other_sampled, proc_test_other_sampled = train_test_split(proc_test_other_slice, test_size=0.2, random_state=seed_random)

proc_test_df.drop(proc_test_df.loc[proc_test_df['labels']=='other'].index, inplace=True)

proc_train_df = pd.concat([proc_train_df, proc_train_other_sampled], axis=0)
proc_test_df = pd.concat([proc_test_df, proc_test_other_sampled], axis=0)


# In[ ]:


fig,(ax1,ax2)= plt.subplots(ncols=2, figsize=(25, 7.5), dpi=100)

fig.suptitle(f'Counts of Observation Labels', fontsize=25)

sns.countplot(x="labels", 
            palette="OrRd_r", 
            data=proc_train_df, 
            order=proc_train_df['labels'].value_counts().index,
            ax=ax1)

ax1.set_title('Train Set', fontsize=20)
ax1.set_xlabel('label', fontsize=15)
ax1.set_ylabel('count', fontsize=15)
ax1.tick_params(labelrotation=90)

sns.countplot(x="labels", 
            palette="GnBu_r", 
            data=proc_test_df, 
            order=proc_test_df['labels'].value_counts().index,
            ax=ax2)

ax2.set_title('Test Set', fontsize=20)
ax2.set_xlabel('label', fontsize=15)
ax2.set_ylabel('count', fontsize=15)
ax2.tick_params(labelrotation=90)

plt.show()


# Everything looks good!
# 
# Now, we will normalize any integer type feature with more than 10 unique values by applying a log transform on them.

# In[ ]:


norm_cols = [ 'duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'num_root', 'num_file_creations', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

for col in norm_cols:
    proc_train_df[col] = np.log(proc_train_df[col]+1e-6)
    proc_test_df[col] = np.log(proc_test_df[col]+1e-6)
    
plot_hist(proc_train_df, norm_cols, 'Distributions in Processed Training Set')
plot_hist(proc_test_df, norm_cols, 'Distributions in Processed Testing Set')


# Awesome! Although these features still have multimodal distributions, they are now in more manageable ranges! 
# 
# As this dataset is a *benchmark* one, we will not do anything with the rate features to avoid any potential information loss from whatever manipulations we make.
# 
# Now, we will one-hot encode our object/categorical training features. Although we were able to reconcile our target feature levels mismatch by discretizing them, we will reconcile our training features level mismatch in a more naive way. We will first join our train and test sets, encode the totality of our object/categoical features, and then split the joined dataset back to the individual training and testing sets.
# 

# In[ ]:


proc_train_df['train']=1                                                                       # add train feature with value 1 to our training set
proc_test_df['train']=0                                                                        # add train feature with value 0 to our testing set

joined_df = pd.concat([proc_train_df, proc_test_df])                                           # join the two sets
 
protocol_dummies = pd.get_dummies(joined_df['protocol_type'], prefix='protocol_type')          # get one-hot encoded features for protocol_type feature
service_dummies = pd.get_dummies(joined_df['service'], prefix='service')                       # get one-hot encoded features for service feature
flag_dummies = pd.get_dummies(joined_df['flag'], prefix='flag')                                # get one-hot encoded features for flag feature

joined_df = pd.concat([joined_df, protocol_dummies, service_dummies, flag_dummies], axis=1)    # join one-hot encoded features to joined dataframe

proc_train_df = joined_df[joined_df['train']==1]                                               # split train set from joined, using the train feature
proc_test_df = joined_df[joined_df['train']==0]                                                # split test set from joined, using the train feature

drop_cols = ['train', 'protocol_type', 'service', 'flag']                                      # columns to drop

proc_train_df.drop(drop_cols, axis=1, inplace=True)                                            # drop original columns from training set
proc_test_df.drop(drop_cols, axis=1, inplace=True)                                             # drop original columns from testing set


# In[ ]:


proc_train_df.head()


# In[ ]:


proc_test_df.head()


# Looks like we were successful!
# 
# # Model Building
# 
# With a fully preprocessed dataset, we are now able to move onto model building. As we have two sets with 122 features and we want to predict what each observation is out of 11 target values, we think that the best model to take on this challenge is a Neural Network model as deep learning models are good at handling large numbers of parameters to provide generalized results.
# 
# First, we'll turn our training and testing dataframes into the x_features, y_target format.

# In[ ]:


y_buffer = proc_train_df['labels'].copy()
x_buffer = proc_train_df.drop(['labels'], axis=1)

y_test = proc_test_df['labels'].copy()
x_test = proc_test_df.drop(['labels'], axis=1)


# We will also split our training set into a training and validation set, so that we can use both resultant sets during model training to ensure it converges on adequate optima.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

seed_random = 315

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_buffer)

x_train, x_val, y_train, y_val = train_test_split(x_buffer, y_buffer, test_size=0.3, random_state=seed_random)


# Now for our model, we will construct a 5-layer model. I've constructed this model using a rule-of-thumb I've developed from prior exercises, where the first layer has twice as many nodes as the number of features in our sets and every subsequent layer has half as many until we reach the number of output values.

# In[ ]:


from keras.models import Sequential
from keras.optimizers import Adam, Nadam
from keras.layers import Dense, Dropout

input_size = len(x_train.columns)

deep_model = Sequential()
deep_model.add(Dense(256, input_dim=input_size, activation='softplus'))
#deep_model.add(Dropout(0.2))
deep_model.add(Dense(128, activation='relu'))
deep_model.add(Dense(64, activation='relu'))
deep_model.add(Dense(32, activation='relu'))
#deep_model.add(Dense(18, activation='softplus'))
deep_model.add(Dense(11, activation='softmax'))

deep_model.compile(loss='categorical_crossentropy', 
                   optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True),
                   metrics=['accuracy'])


# For a sequential neural network to perform categorical classification, we will also need to one-hot encode our target set. We do so below:

# In[ ]:


y_train_econded = label_encoder.transform(y_train)
y_val_econded = label_encoder.transform(y_val)
y_test_econded = label_encoder.transform(y_test)

y_train_dummy = np_utils.to_categorical(y_train_econded)
y_val_dummy = np_utils.to_categorical(y_val_econded)
y_test_dummy = np_utils.to_categorical(y_test_econded)


# Now, we can finally train our model!

# In[ ]:


deep_model.fit(x_train, y_train_dummy, 
               epochs=50, 
               batch_size=2500,
               validation_data=(x_val, y_val_dummy))


# We've got a good feeling about how well this model will perform when we use it to make predictions! We see how, by epoch 50, it was able to achieve a validation accuracy of 0.9747 and overall accuracy of 0.9649! Not only are those numbers impressive, but they also indicate the model was able to reach an optima!
# 
# So, we make some predictions:

# In[ ]:


deep_val_pred = deep_model.predict_classes(x_val)
deep_val_pred_decoded = label_encoder.inverse_transform(deep_val_pred)

deep_test_pred = deep_model.predict_classes(x_test)
deep_test_pred_decoded = label_encoder.inverse_transform(deep_test_pred)


# To help us better understand our model performance, we introduce a function to visualize the confusion matrix between our predictions and our targets.

# In[ ]:


import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, make_scorer 

# Showing Confusion Matrix
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_cm(y_true, y_pred, title):
    figsize=(14,14)
    #y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# In[ ]:


plot_cm(y_val, deep_val_pred_decoded, 'Confusion matrix for predictions on the validation set')
f1_score(y_val, deep_val_pred_decoded, average = 'macro')


# In[ ]:


plot_cm(y_test, deep_test_pred_decoded, 'Confusion matrix for predictions on the testing set')
f1_score(y_test, deep_test_pred_decoded, average = 'macro')


# # Conclusions
# 
# Wow! We achieved really impressive results with our model predictions on the validation and testing sets! We got F1 Scores of 95.873% and 95.330% respectively! 
# 
# For all types of attacks, except for *nmap* and *other* attacks, we got >95% accuracies. For *other* attacks, this is not suprising; That label is a conglomerate of various attacks, some with significantly few observations to adequately train on. If we wish to acheive better results with this label, or even predicting the specific attack, we just need to capture and add more such observations to our training set.
# 
# For the nmap attack, it is probably due to the fact that it has similar enough metrics to other attacks in the dataset, specifically *ipsweep*. This is understandable as both *nmap* and *ipsweep* activities, aim to figure out how many live *IP's* there are.
# 
# I hope this kernel helps you better understand the data and gives you some inspiration on how to tackle it yourself!
