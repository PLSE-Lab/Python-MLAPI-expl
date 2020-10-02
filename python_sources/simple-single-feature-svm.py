#!/usr/bin/env python
# coding: utf-8

# Hi everyone!
# 
# This is my first Kaggle Competition and Kernel. I tried working with Support Vector Machines, and achieved very high F1 macro score with the same. I am sharing my results below.
# Dataset used : https://www.kaggle.com/cdeotte/data-without-drift
# 
# I have used 5 different SVM models. For more details and detailed plots, go here: https://www.kaggle.com/cdeotte/one-feature-model-0-930/output
# 
# The above link explains how 5 different models were used to create synthetic data.
# 
# I do not have much experience with Machine Learning, so I have naturally explained in a very simpler manner. Enjoy!

# In[ ]:


import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score


# The below cell will input data and store them in numpy arrays. There are 5 models: Model 0, Model 1...Model 4. They are our estimations of the original respective models used to generate respective batches:
# 
# 1. Model 0: 
#     * **Training Batches** 0,1
#     * **Testing Batches** 0,3,8,10,11,12,13,14,15,16,17,18,19
#     * **Maximum Open Channels**: 1
# 2. Model 1: 
#     * **Training Batches** 2,6
#     * **Testing Batches** 4
#     * **Maximum Open Channels**: 1
# 3. Model 2: 
#     * **Training Batches** 3,7
#     * **Testing Batches** 1,9
#     * **Maximum Open Channels**: 3
# 4. Model 3: 
#     * **Training Batches** 4,9
#     * **Testing Batches** 5,7
#     * **Maximum Open Channels**: 10
# 5. Model 4: 
#     * **Training Batches** 5,8
#     * **Testing Batches** 2,6
#     * **Maximum Open Channels**: 5
# 

# In[ ]:


data_path = '/kaggle/input/data-without-drift/'
train_data_file = data_path + 'train_clean.csv'
test_data_file = data_path + 'test_clean.csv'

def get_data(filename, train=True):
  
    if(train):
        with open(filename) as training_file:
            split_size = 10
            data = np.loadtxt(training_file, delimiter=',', skiprows=1)
            signal = data[:,1]
            channels = data[:,2]
            signal = np.array_split(signal, split_size)
            channels = np.array_split(channels, split_size)
            data = None
        return np.array(signal), np.array(channels)
    else:
       with open(filename) as training_file:
            split_size = 4
            data = np.loadtxt(training_file, delimiter=',', skiprows=1)
            signal = data[:,1]
            signal = np.array_split(signal, split_size)
            data = None
       return np.array(signal)

train_signal , train_channels = get_data(train_data_file)
test_signal = get_data(test_data_file, train=False)

test_model_signal = np.zeros((5,1000000))
test_model_channel = np.zeros((5,1000000))
test_model_signal[0][:500000] = train_signal[0].flatten()
test_model_signal[0][500000:] = train_signal[1].flatten()
test_model_signal[1][:500000] = train_signal[2].flatten()
test_model_signal[1][500000:] = train_signal[6].flatten()
test_model_signal[2][:500000] = train_signal[3].flatten()
test_model_signal[2][500000:] = train_signal[7].flatten()
test_model_signal[3][:500000] = train_signal[4].flatten()
test_model_signal[3][500000:] = train_signal[9].flatten()
test_model_signal[4][:500000] = train_signal[5].flatten()
test_model_signal[4][500000:] = train_signal[8].flatten()


test_model_channel[0][:500000] = train_channels[0].flatten()
test_model_channel[0][500000:] = train_channels[1].flatten()
test_model_channel[1][:500000] = train_channels[2].flatten()
test_model_channel[1][500000:] = train_channels[6].flatten()
test_model_channel[2][:500000] = train_channels[3].flatten()
test_model_channel[2][500000:] = train_channels[7].flatten()
test_model_channel[3][:500000] = train_channels[4].flatten()
test_model_channel[3][500000:] = train_channels[9].flatten()
test_model_channel[4][:500000] = train_channels[5].flatten()
test_model_channel[4][500000:] = train_channels[8].flatten()


# Specs below refers to specifications of SVM model, namely C and gamma. You need to have a basic understanding of what an SVM is to understand the math behind the specifications. These were evaluated using a grid search for hyperparameter tuning. Refer to documentation of sklearn.svm.svc for more details. 
# 
# Below, the model is trained on the first 400000 entries and validated on the next 100000 entries. The remaining 500000 is unused. You can do undersampling and upsampling to generate a well balanced data but the below also works.

# In[ ]:


from sklearn.svm import SVC
models = []

specs = [[1.2,1],[0.1,1],[0.5,1],[7,0.01],[10,0.1]]

for k in range (5):
    print("starting training model no: ", k)
    x = test_model_signal[k].flatten()
    y = test_model_channel[k].flatten()
    y = np.array(y).astype(int)
    x = np.expand_dims(np.array(x),-1)
    model = SVC(kernel = 'rbf', C=specs[k][0],gamma = specs[k][1])
    samples= 400000
    #trains by splitting into 10 batches for faster training
    for i in range(10):
        model.fit(x[i*samples//10:(i+1)*samples//10],y[i*samples//10:(i+1)*samples//10])
    y_pred = model.predict(x[400000:500000])
    y_true = y[400000:500000]
    print(f1_score(y_true, y_pred, average=None))
    print(f1_score(y_true, y_pred, average='macro'))
    models.append(model)


# The following is the testing process. Each batch is of length 100000, which can be easily seen from plotting the signal values. The model for each batch can be manually determined, or by calculating the average of all the entries on each batch and matching the same with the average of training batches.

# In[ ]:


model_ref = [0,2,4,0,1,3,4,3,0,2,0,0,0,0,0,0,0,0,0,0]
y_pred_all = np.zeros((2000000))
for pec in range(20):
  print("starting prediction of test batch no: ", pec)
  x_test = test_signal.flatten()[pec*100000:(pec+1)*100000]
  x_test = np.expand_dims(np.array(x_test),-1)
  test_pred = models[model_ref[pec]].predict(x_test)
  y_pred_1 = np.array(test_pred).astype(int)
  y_pred_all[pec*100000:(pec+1)*100000] = y_pred_1

y_pred_all = np.array(y_pred_all).astype(int)


# The following is a good estimation of LB. As it is known that the first 600000 or the first 6 batches of testing data are used for the public leaderboard. So, we evaluate the results for 6 batches of validation data from similar models.

# In[ ]:


model_ref = [0,0,1,2,3,4]
y_valid = np.zeros((1000000))
y_pred = np.zeros((1000000))
for k in range(6):
  x = train_signal[k].flatten()
  y = train_channels[k].flatten()
  y = np.array(y).astype(int)
  x = np.expand_dims(np.array(x),-1)
  model = models[model_ref[k]]
  y_pred[k*100000:(k+1)*100000] = model.predict(x[400000:500000])
  y_valid[k*100000:(k+1)*100000]=y[400000:500000]

print(f1_score(y_valid, y_pred, average=None))
print(f1_score(y_valid, y_pred, average='macro'))


# The following writes the testing predictions into csv file for submission:

# In[ ]:


import pandas as pd
sub = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
sub.iloc[:,1] = y_pred_all
sub.to_csv('submission.csv',index=False,float_format='%.4f')
print("saved the file")

