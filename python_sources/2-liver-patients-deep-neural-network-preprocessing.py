#!/usr/bin/env python
# coding: utf-8

# # Indian Lever Patients Analysis-Deep Neural Networks-Preprocessing

# We are going to perform a Supervised Learnind on the dataset. The last column contains the targets
# 
# We do the following tasks here:
# 1. Preporcess the data
#     a. Removed records with null values
#     b. Changed the value '2' under columnt 'Dataset' to 0
#     c. Changed Male to 0 and Female to 1 under 'Gender' coulumn.
#     d. Remoed the headers from the file
# 
# 2. Balance the dataset
# 3. Create 3 datasets: training, validation, and test
#     a. We can create datasets with different proportions (80:10:10, 70:20:10)
# 4. Save those 3 datasets in a tensor friendly format (*.npz)
#     a. We use tensorflow to create the ML algorithm
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Extract the data from the csv

# In[ ]:


import numpy as np
from sklearn import preprocessing

raw_csv_data = np.loadtxt('../input/indian_liver_patient-NN.csv', delimiter = ',')

unscaled_inputs_all = raw_csv_data[:,:-1]
targets_all = raw_csv_data[:,-1]


# ### Balance the dataset

# In[ ]:


def balance_dataset(y_n):
    if y_n == True:
        num_one_targets = 0
        num_two_targets = 0
        for i in range(targets_all.shape[0]):
            if targets_all[i] == 1:
                num_one_targets += 1
            if targets_all[i] == 0:
                num_two_targets += 1

        if num_one_targets > num_two_targets:
            equal_prior = num_two_targets
        else:
            equal_prior = num_one_targets

        print('Equal_prior: ', equal_prior)

        indices_to_remove = []
        num_one_counter = 0
        num_two_counter = 0
        for j in range(targets_all.shape[0]):
            if targets_all[j] == 1:
                num_one_counter += 1
                if num_one_counter > equal_prior:
                      indices_to_remove.append(j)
            if targets_all[j] == 0:
                if num_two_counter > equal_prior:
                      indices_to_remove.append(j)

        unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all,indices_to_remove, axis = 0)
        targets_equal_priors = np.delete(targets_all, indices_to_remove, axis = 0)
    else:
        unscaled_inputs_equal_priors = unscaled_inputs_all
        targets_equal_priors = targets_all
    return [unscaled_inputs_equal_priors,targets_equal_priors]


# In[ ]:


[unscaled_inputs_pre_proc,targets_pre_proc] = balance_dataset(True) # True: for a balanced dataset, False: for the dataset as such


# In[ ]:


print(unscaled_inputs_pre_proc.shape)
print(targets_pre_proc.shape)


# In[ ]:


#unscaled_inputs_pre_proc


# ### Standardize the inputs

# In[ ]:


# We standardize the variables to reduce the weight of higher numbers on the model. 
scaled_inputs = preprocessing.scale(unscaled_inputs_pre_proc)


# In[ ]:


#scaled_inputs


# ### Shuffle the data

# In[ ]:


shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_pre_proc[shuffled_indices]
shuffled_inputs.shape


# In[ ]:


#shuffled_targets


# ### Split the dataset into train, validation, and test

# In[ ]:


# We split the data into train and validation to prevent overfitting. 'test' dataset is for calculating the accuracy of the model.
# We will see the test accuracy in the next notebook file.

samples_count = shuffled_inputs.shape[0]

#You can create datasets with different proportions (80:10:10, 70:20:10)
train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# To see if how the values in target dataset distributed. From a balanced dataset, we expect 1s and 0s in an approximate 50:50 proportion
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)


# ### Save the three datasets in *.npz

# In[ ]:


np.savez('Liver_disease_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Liver_disease_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Liver_disease_data_test', inputs=test_inputs, targets=test_targets)

