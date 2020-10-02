#!/usr/bin/env python
# coding: utf-8

# Adversarial Validation CNN of PTP for VSB Power, Paul Nussbaum, Feb 2019

# 
# An Adversarial Validation approach may be useful when the TEST set may be very different from the TRAINING set. Simply choosing a subset of the training set to perform validation may not yield the best results. 
# 
# Traditional methods for creation of a validation set include stratified k-fold cross validation, stratified percentage split, and a simple percentage split as included in the "fit" function "validation_split" parameter.
# 
# This Python Jupyter Notebook demonstrates a different kind of validation split, popular in several Kaggle competitions, called the Adversarial Validation approach. In this approach, we create a machine learning algorithm to distinguish between the training set and the testing set. We then use that algorithm to find those training set examples that "most resemble" testing set examples, an we use those as our validation set. We then train our regognition algorithm as we normally would. 
# 
# In this example, I will use the same Peak to Peak (PTP) feature from three phases of non-overlapping windows of data, and the same Convolutional Neural Network (CNN) to both create the Adversarial Validation Set as well as to recognize when a fault has taken place in the VSB Power competition.
# 
# It works as follows:
# * first load all the training and testing data into one huge training set X, and replace y with 0 or 1 for train or test, respectively
# * use this huge training set to learn the difference between training and testing examples
# * use the learnings to classify the training set
# * sort the training by the "similar to the test set" score of y
# * finally, keep the first n% most test like training samples as the validation group for traditional training.
# * thanks go to http://fastml.com/adversarial-validation-part-one/ - Posted by Zygmunt Z. 2016-05-23
# * thanks go to adversarial_validation_and_lb_shakeup posted by Olivier https://www.kaggle.com/ogrellier/adversarial-validation-and-lb-shakeup
# * thanks go to http://manishbarnwal.com/blog/2017/02/15/introduction_to_adversarial_validation/ by Manish Barnwal

# In[ ]:


# LOADING UP PYTHON COMPONENTS
import numpy as np # linear algebra
from numpy import sort
from scipy import stats, histogram
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pyarrow.parquet as pq
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import * # Keras is the most friendly Neural Network library, this Kernel use a lot of layers classes
from keras.models import Model, Sequential, load_model
from tqdm import tqdm # Processing time measurement
from keras import backend as K # The backend give us access to tensorflow operations and allow us to create the Attention class
from keras import optimizers # Allow us to access the Adam class to modify some parameters
from keras.callbacks import * # This object helps the model to train in a smarter way, avoiding overfitting

from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing


# In[ ]:


# This function copied from https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694
#    courtesy Bruno Aquino - 5-fold LSTM Attention (fully commented)
# It is the official metric used in this competition
# below is the declaration of a function used inside the keras model, calculation with K (keras backend / thensorflow)
def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


# In[ ]:


# Some global parameters
filename = '../input/train.parquet'
# number of non-overlaping windows...
windows = 64 # 312.5 us at 50Hz or 12,500 samples
win_size = int(800000 / windows)
# number of features per time slice
win_feat = 1
phases = 3
# total number of features extracted from the signal
number_of_params = int(phases * win_feat)


# In[ ]:


# FEATURE EXTRACTION FUNCTION
# This function chops up three phases of data at a time into equal sized non-overlapping windows.
# For each phase and each window, it extracts the number of features (in this case, one feature called PTP, which is max - min)
def extract_features(begin_col, loc_num_to_use, filename) :  
    # Use "Batching" to avoid RAM limitations of Kaggle
    # loc_num_to_use must be a multiple of 3, and so does batch
    batch = 300 
    num_batches = int((loc_num_to_use) / batch)
    remainder = int((loc_num_to_use) % batch)
    ### Create a pandas data frame
    loc_X = np.zeros((int(loc_num_to_use), windows, win_feat, phases))
    if num_batches > 0:
        for ix in range (num_batches) :
            # load a batch of signals
            x1 = pq.read_pandas(filename, columns=[str(ix * batch + j + begin_col) for j in range(batch)]).to_pandas().values.transpose()
            # now look at all three phases at once
            for i in range (0, batch, phases) :
                idx = ix * batch + i
                for k in range (windows) :
                    # start and end of window in signal data
                    win_start = k * win_size
                    win_end = win_start + win_size - 1

                    # THIS PHASE is Phase 1 or Phase 2 for other phases
                    min0 =  float(x1[i,win_start:win_end].min())
                    max0 =  float(x1[i,win_start:win_end].max())
                    ptp0 =  max0 - min0
                    
                    loc_X[idx    , k, 0, 0] = ptp0 
                    loc_X[idx + 1, k, 0, 1] = ptp0
                    loc_X[idx + 2, k, 0, 2] = ptp0
                    
                    # PHASE 1 is this phase or Phase 2 for other phases
                    min0 =  float(x1[i + 1,win_start:win_end].min())
                    max0 =  float(x1[i + 1,win_start:win_end].max())
                    ptp0 =  max0 - min0

                    loc_X[idx    , k, 0, 2] = ptp0
                    loc_X[idx + 1, k, 0, 0] = ptp0
                    loc_X[idx + 2, k, 0, 1] = ptp0
                    
                    # PHASE 2 is this phase or Phase 1 for other phases
                    min0 =  float(x1[i + 2,win_start:win_end].min())
                    max0 =  float(x1[i + 2,win_start:win_end].max())
                    ptp0 =  max0 - min0
                    
                    loc_X[idx    , k, 0, 1] = ptp0  
                    loc_X[idx + 1, k, 0, 2] = ptp0 
                    loc_X[idx + 2, k, 0, 0] = ptp0 
    # Here we process what is left over from all of the Batches
    if remainder > 0 :
        ix = num_batches
        if 1 : # dummy indent so I don't have to keep changing indent on cut and paste
            # load a batch of signals
            x1 = pq.read_pandas(filename, columns=[str(ix * batch + j + begin_col) for j in range(remainder)]).to_pandas().values.transpose()
            for i in range (0,remainder, phases) :
                idx = ix * batch + i
                for k in range (windows) :
                    # start and end of window in signal data
                    win_start = k * win_size
                    win_end = win_start + win_size - 1

                    # THIS PHASE is Phase 1 or Phase 2 for other phases
                    min0 =  float(x1[i,win_start:win_end].min())
                    max0 =  float(x1[i,win_start:win_end].max())
                    ptp0 =  max0 - min0
                    
                    loc_X[idx    , k, 0, 0] = ptp0 
                    loc_X[idx + 1, k, 0, 1] = ptp0
                    loc_X[idx + 2, k, 0, 2] = ptp0
                    
                    # PHASE 1 is this phase or Phase 2 for other phases
                    min0 =  float(x1[i + 1,win_start:win_end].min())
                    max0 =  float(x1[i + 1,win_start:win_end].max())
                    ptp0 =  max0 - min0

                    loc_X[idx    , k, 0, 2] = ptp0
                    loc_X[idx + 1, k, 0, 0] = ptp0
                    loc_X[idx + 2, k, 0, 1] = ptp0
                    
                    # PHASE 2 is this phase or Phase 1 for other phases
                    min0 =  float(x1[i + 2,win_start:win_end].min())
                    max0 =  float(x1[i + 2,win_start:win_end].max())
                    ptp0 =  max0 - min0
                    
                    loc_X[idx    , k, 0, 1] = ptp0  
                    loc_X[idx + 1, k, 0, 2] = ptp0 
                    loc_X[idx + 2, k, 0, 0] = ptp0 

    return loc_X


# In[ ]:


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-    CREATE ADVERSARIAL DATA SET

# TRAINING DATA
# USE THIS VALUE TO RUN IT ON THE WHOLE TRAINING SET
num_train = 8712
# HERE IS A SMALLER VALUE FOR SPEED
# num_train = 300

print('loading training set')
X_train = extract_features(0, num_train, '../input/train.parquet')
print (X_train.shape)
print('swapping features and phases')
X_train = np.reshape(X_train[:,:,0:win_feat,0:phases], 
                     ((num_train),windows,phases,win_feat))
print(X_train.shape)

# TEST DATA
# USE THIS VALUE TO RUN IT ON THE WHOLE TEST SET
num_test = 20337
# HERE IS A SMALLER VALUE FOR SPEED
# num_test = 600

print('loading testing set')
X_test = extract_features(8712, num_test, '../input/test.parquet')
print(X_test.shape)
print('swapping features and phases')
X_test = np.reshape(X_test[:,:,0:win_feat,0:phases], 
                     ((num_test),windows,phases,win_feat))
print(X_test.shape)

print('combining the two sets')
X = np.concatenate((X_train,X_test))
print (X.shape)
print('creating target set')
y = np.zeros((num_train+num_test))
y[num_train:num_train + num_test] = 1
print (y.shape)
### 13.3 GB peak 13.4 GB


# In[ ]:


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-    CREATE ADVERSARIAL CNN
# This uses a Convolutional Neural Network to learn the difference between testing and training data
CNN_scale = 50

class_model = Sequential()
class_model.add(Conv2D(CNN_scale, (3,phases), strides = 3, activation='relu', input_shape=(windows,phases,win_feat)))
class_model.add(BatchNormalization())
class_model.add(Dropout(0.25))
class_model.add(Conv2D(CNN_scale, (3,1), strides = 3, activation='relu' ))
class_model.add(BatchNormalization())
class_model.add(Dropout(0.5))
class_model.add(BatchNormalization())
class_model.add(Dropout(0.5))
class_model.add(Flatten())
class_model.add(Dense(CNN_scale, activation='relu'))
class_model.add(BatchNormalization())
class_model.add(Dropout(0.5))
class_model.add(Dense(CNN_scale, activation='relu'))
class_model.add(Dense(1, activation='sigmoid', name='output'))

class_model.compile(loss='binary_crossentropy', optimizer='adam')
# Diplay the model summary
print("model summary")
class_model.summary()


# In[ ]:


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-    TRAIN ADVERSARIAL CNN
earlystopper = EarlyStopping(patience=25,
                             verbose=1) 
checkpointer = ModelCheckpoint('adv_classifier', verbose=1, save_best_only=True, monitor='val_loss', mode='min')
# results = class_model.fit(X, y, validation_split=0.2, batch_size=25, epochs=300, 
#                     callbacks=[earlystopper, checkpointer])
# Now using stratified validation split (same percentage of t, activation='relu'rue and false in both training and validation)
results = class_model.fit(X, y, validation_split = 0.33, 
                          batch_size=32, epochs = 300, 
                          callbacks=[earlystopper, checkpointer])
### 13.8 GB peak 14 GB


# In[ ]:


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-    USE ADVERSARIAL CNN to CLASSIFY TRAINING DATA
y_liketest = class_model.predict(X_train)
print(y_liketest.shape)
# also load the training set's correct classifications in the same sequence
# The second row of three shows the three phases of a power line that has a fault
meta_train = pd.read_csv('../input/metadata_train.csv')
# correct classifications from training set
y_train = np.zeros((num_train))
for i in range(0, int(num_train)):
    y_train[i] = meta_train.target[i]
print(y_train.shape)
### 13.8 GB 


# In[ ]:


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-    SORT TRAINING DATA BY TEST-LIKENESS (AND SORT Y)
y_sorted = np.sort(y_liketest, axis=0)
plt.plot(y_sorted)
# see what value would split the data 80% / 20%
eighty = int(num_train * .8)
twenty = num_train - eighty
thresh80 = y_sorted[eighty,0]
# copy dummy data just to dimension the arrays
X = X_train[0:eighty]
X_valid = X_train[0:twenty]
y = y_train[0:eighty]
y_valid = y_train[0:twenty]
# now put the TRAIN data that most resembles the TEST data into the validation set, and use the others for training
tidx = 0
vidx = 0
for i in range(num_train) :
    if (y_liketest[i] >= thresh80) and (vidx < twenty) :
        X_valid[vidx] = X_train[i]
        y_valid[vidx] = y_train[i]
        vidx = vidx + 1
    elif tidx < eighty : 
        X[tidx] = X_train[i]
        y[tidx] = y_train[i]
        tidx = tidx + 1
    else :
        X_valid[vidx] = X_train[i]
        y_valid[vidx] = y_train[i]
        vidx = vidx + 1

print(X.shape, X_valid.shape, y.shape, y_valid.shape)
        
num_to_use = num_train
# Quick and dirty RAM cleanup
y_liketest = 5

### 13.8 GB 


# In[ ]:


# quick and dirty RAM cleanup
meta_train = 5


# In[ ]:


#Show stats of TEST data
print("Was Adversarial creation of a Validation set worth it? Let's compare feature means and variances...")
print ("")
print("X mean (unscaled)")
Xmin = X.mean(axis=0).min()
Xmax = X.mean(axis=0).max()
print("X mean ranges from ", Xmin," to ", Xmax, " for a span of ", Xmax - Xmin)
print ("X Variance (unscaled)")
Xmin = X.var(axis=0).min()
Xmax = X.var(axis=0).max()
print("X var ranges from ", Xmin," to ", Xmax, " for a span of ", Xmax - Xmin)
print(" ")
print("Validation mean (unscaled)")
Xmin = X_valid.mean(axis=0).min()
Xmax = X_valid.mean(axis=0).max()
print("Validation mean ranges from ", Xmin," to ", Xmax, " for a span of ", Xmax - Xmin)
print ("Validation Variance (unscaled)")
Xmin = X_valid.var(axis=0).min()
Xmax = X_valid.var(axis=0).max()
print("Validation var ranges from ", Xmin," to ", Xmax, " for a span of ", Xmax - Xmin)
print("")
#Show stats of TEST data
print("TEST SET mean (unscaled)")
Xmin = X_test.mean(axis=0).min()
Xmax = X_test.mean(axis=0).max()
print("TEST SET mean ranges from ", Xmin," to ", Xmax, " for a span of ", Xmax - Xmin)
print ("TEST SET Variance (unscaled)")
Xmin = X_test.var(axis=0).min()
Xmax = X_test.var(axis=0).max()
print("TEST SET var ranges from ", Xmin," to ", Xmax, " for a span of ", Xmax - Xmin)


# In[ ]:


# Will use the same CNN to categorize the data
CNN_scale = 50

class_model = Sequential()
class_model.add(Conv2D(CNN_scale, (3,phases), strides = 3, activation='relu', input_shape=(windows,phases,win_feat)))
class_model.add(BatchNormalization())
class_model.add(Dropout(0.25))
class_model.add(Conv2D(CNN_scale, (3,1), strides = 3, activation='relu' ))
class_model.add(BatchNormalization())
class_model.add(Dropout(0.5))
class_model.add(BatchNormalization())
class_model.add(Dropout(0.5))
class_model.add(Flatten())
class_model.add(Dense(CNN_scale, activation='relu'))
class_model.add(BatchNormalization())
class_model.add(Dropout(0.5))
class_model.add(Dense(CNN_scale, activation='relu'))
class_model.add(Dense(1, activation='sigmoid', name='output'))

class_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
# Diplay the model summary
print("model summary")
class_model.summary()


# In[ ]:


### Train the residual compressor
earlystopper = EarlyStopping(patience=25,
                             verbose=1) 
checkpointer = ModelCheckpoint('VSB_classifier', verbose=1, save_best_only=True, monitor='val_matthews_correlation', mode='max')
# results = class_model.fit(X, y, validation_split=0.2, batch_size=25, epochs=300, 
#                     callbacks=[earlystopper, checkpointer])
# Now using stratified validation split (same percentage of t, activation='relu'rue and false in both training and validation)
results = class_model.fit(X, y, 
                          validation_data=[X_valid, y_valid], 
                          batch_size=32, epochs = 300, 
                          callbacks=[earlystopper, checkpointer])
### 13.2 GB peak 13.5 GB


# In[ ]:



class_model = load_model('VSB_classifier', custom_objects={'matthews_correlation':matthews_correlation})


# In[ ]:


# X = X_raw
# y = y_raw
X = X_train
y = y_train
# Quick and dirty RAM cleanup
X_train = 5
y_train = 5
X_valid = 5
y_valid = 5


# In[ ]:


pred_y = class_model.predict(X)
predicted_y = pred_y[:,0] #np.concatenate(pred_y)[0]
predicted_y.shape
### 13.2 GB 


# In[ ]:


# This function copied from https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694
#    courtesy Bruno Aquino - 5-fold LSTM Attention (fully commented)
# The output of this kernel must be binary (0 or 1), but the output of the NN Model is float (0 to 1).
# So, find the best threshold to convert float to binary is crucial to the result
# this piece of code is a function that evaluates all the possible thresholds from 0 to 1 by 0.01
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = K.eval(matthews_correlation(y_true.astype(np.float64), (y_proba > threshold).astype(np.float64)))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'matthews_correlation': best_score}
    return search_result


# In[ ]:


result = threshold_search(y, predicted_y)
### 13.2 GB peak 13.2 GB


# In[ ]:


best_thresh = result['threshold']
Best_MCC = result['matthews_correlation']
print('best MCC of ', Best_MCC, ' using best threshold ', best_thresh)
predicted_y = pred_y
# quick and dirty RAM cleanup
pred_y = 5


# In[ ]:


y = predicted_y
# See how many doubles and triples we got on the test data
# How often did we predict power line faults occur on all three phases simultaneously, versus on fewer than all 3?
triples = 0
doubles = 0
singles = 0
for i in range(0,int(num_to_use),3) : 
    y[i] = int(y[i] > best_thresh) 
    y[i+1] = int(y[i+1] > best_thresh)
    y[i+2] = int(y[i+2] > best_thresh)
    num_phases_faulty = y[i] + y[i+1] + y[i+2]
    if (num_phases_faulty == 3):
        triples = triples + 1
    elif (num_phases_faulty == 2):
        doubles = doubles + 1
    elif (num_phases_faulty == 1):
        singles = singles + 1

print('triples', triples, 'doubles', doubles, 'singles', singles)
print('sanity check: ', 'total faults', y.sum(), ' sum of above ', 3 * triples + 2 * doubles + singles)
### 13.3 GB 


# In[ ]:


X = X_test
num_to_use = num_test
print(X.shape)


# In[ ]:


# quick and dirty RAM cleanup
X_unscaled = 5
print (X.shape)
### 12.8 GB


# In[ ]:


y = class_model.predict(X)

# How often did we predict power line faults occur on all three phases simultaneously, versus on fewer than all 3?
triples = 0
doubles = 0
singles = 0
for i in range(0,int(num_to_use),3) : 
    y[i] = int(y[i] > best_thresh) 
    y[i+1] = int(y[i+1] > best_thresh)
    y[i+2] = int(y[i+2] > best_thresh)
    num_phases_faulty = y[i] + y[i+1] + y[i+2]
    if (num_phases_faulty == 3):
#        print('triple',meta_train.signal_id[i], meta_train.phase[i] )
        triples = triples + 1 
    elif (num_phases_faulty == 2):
#        print('double',meta_train.signal_id[i], meta_train.phase[i])
        doubles = doubles + 1
    elif (num_phases_faulty == 1):
#        print('single',meta_train.signal_id[i], meta_train.phase[i])
        singles = singles + 1

print('triples', triples, 'doubles', doubles, 'singles', singles)
print('sanity check: ', 'total faults', y.sum(), ' sum of above ', 3 * triples + 2 * doubles + singles)
# plt.plot(meta_train.target)
### 12.2 GB


# In[ ]:


meta_test = pd.read_csv('../input/metadata_test.csv')
output = pd.DataFrame({"signal_id":meta_test.signal_id[0:int(num_to_use)]})
# Use this one for NN 
output["target"] = pd.Series(y[:,0]) 
# Use this one for Random Forest
# output["target"] = pd.Series(y[:]) 

output['signal_id'] = output['signal_id'].astype(np.int64)
output['target'] = output['target'].astype(np.int64)
output.to_csv("submission.csv", index=False)
output
### 13.2 GB

