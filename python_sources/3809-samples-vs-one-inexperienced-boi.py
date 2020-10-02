#!/usr/bin/env python
# coding: utf-8

# So what exactly is this assignment? This is a simple classification task with a minor hiccup: how exactly should the inputs be vectorized? 
# 
# This will be a public kernel, so feel free to offer suggestions/yell at me if you're reading this

# In[22]:


#Getting the basic libraries set up

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Machine learning-specific
import tensorflow as tf
from tensorflow import keras

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

y_train = pd.read_csv('../input/y_train.csv')
x_train = pd.read_csv('../input/X_train.csv')
x_test = pd.read_csv('../input/X_test.csv')

# Any results you write to the current directory are saved as output.


# First things first, let's get the inputs and outputs set up.
# 
# For outputs, this is a pretty simple use of label encoding.
# 
# For inputs, this implementation crafts a couple of features from each measurement series: mean, stdev, and range. I suspect that the feature selection is the first thing that needs to be improved to make an accurate model, but I'm not entirely sure what else to add

# In[23]:


#Function for setting up training & testing data
def processor(input,out_name,num_samples):
    #Input should be an input csv turned into a data frame
    #out_name is a string ending in .csvorientations = ["or_x_mean","or_x_std","or_x_range","or_y_mean","or_y_std","or_y_range","or_z_mean","or_z_std","or_z_range","or_w_mean","or_w_std","or_w_range"]
    orientations = ["or_x_mean","or_x_std","or_x_range","or_y_mean","or_y_std","or_y_range","or_z_mean","or_z_std","or_z_range","or_w_mean","or_w_std","or_w_range"]
    ang_vels = ["an_x_mean","an_x_std","an_x_range","an_y_mean","an_y_std","an_y_range","an_z_mean","an_z_std","an_z_range"]
    accels = ["acc_x_mean","acc_x_std","acc_x_range","acc_y_mean","acc_y_std","acc_y_range","acc_z_mean","acc_z_std","acc_z_range"]
    columns = orientations + ang_vels+accels
    processed = pd.DataFrame(index = range(0,num_samples+1), columns = columns)
    #What you are about to see is an affront to good code, but I'm not entire sure how to make this manageable
    for i in range(0,num_samples+1):
        curr = input.loc[input["series_id"]==i]
        
        or_x_mean = np.mean(curr["orientation_X"])
        processed["or_x_mean"][i] = or_x_mean
        or_x_std = np.std(curr["orientation_X"])
        processed["or_x_std"][i] = or_x_std
        or_x_range = max(curr["orientation_X"])-min(curr["orientation_X"])
        processed["or_x_range"][i] = or_x_range
        
        
        or_y_mean = np.mean(curr["orientation_Y"])
        processed["or_y_mean"][i] = or_y_mean
        or_y_std = np.std(curr["orientation_Y"])
        processed["or_y_std"][i] = or_y_std
        or_y_range = max(curr["orientation_Y"])-min(curr["orientation_Y"])
        processed["or_y_range"][i] = or_y_range
        
        or_z_mean = np.mean(curr["orientation_Z"])
        processed["or_z_mean"][i] = or_z_mean
        or_z_std = np.std(curr["orientation_Z"])
        processed["or_z_std"][i] = or_z_std
        or_z_range = max(curr["orientation_Z"])-min(curr["orientation_Z"])
        processed["or_z_range"][i] = or_z_range
        
        or_w_mean = np.mean(curr["orientation_W"])
        processed["or_w_mean"][i] = or_z_mean
        or_w_std = np.std(curr["orientation_W"])
        processed["or_w_std"][i] = or_z_std
        or_w_range = max(curr["orientation_W"])-min(curr["orientation_W"])
        processed["or_w_range"][i] = or_z_range
        
        an_x_mean = np.mean(curr["angular_velocity_X"])
        processed["an_x_mean"][i] = an_x_mean
        an_x_std = np.std(curr["angular_velocity_X"])
        processed["an_x_std"][i] = an_x_std
        an_x_range = max(curr["angular_velocity_X"])-min(curr["angular_velocity_X"])
        processed["an_x_range"][i] = an_x_range
        
        
        an_y_mean = np.mean(curr["angular_velocity_Y"])
        processed["an_y_mean"][i] = an_y_mean
        an_y_std = np.std(curr["angular_velocity_Y"])
        processed["an_y_std"][i] = an_y_std
        an_y_range = max(curr["angular_velocity_Y"])-min(curr["angular_velocity_Y"])
        processed["an_y_range"][i] = an_y_range
        
        an_z_mean = np.mean(curr["angular_velocity_Z"])
        processed["an_z_mean"][i] = an_z_mean
        an_z_std = np.std(curr["angular_velocity_Z"])
        processed["an_z_std"][i] = an_z_std
        an_z_range = max(curr["angular_velocity_Z"])-min(curr["angular_velocity_Z"])
        processed["an_z_range"][i] = an_z_range
        
        acc_x_mean = np.mean(curr["linear_acceleration_X"])
        processed["acc_x_mean"][i] = acc_x_mean
        acc_x_std = np.std(curr["linear_acceleration_X"])
        processed["acc_x_std"][i] = acc_x_std
        acc_x_range = max(curr["linear_acceleration_X"])-min(curr["linear_acceleration_X"])
        processed["acc_x_range"][i] = acc_x_range
        
        acc_y_mean = np.mean(curr["linear_acceleration_Y"])
        processed["acc_y_mean"][i] = acc_y_mean
        acc_y_std = np.std(curr["linear_acceleration_Y"])
        processed["acc_y_std"][i] = acc_y_std
        acc_y_range = max(curr["linear_acceleration_Y"])-min(curr["linear_acceleration_Y"])
        processed["acc_y_range"][i] = acc_y_range
        
        acc_z_mean = np.mean(curr["linear_acceleration_Z"])
        processed["acc_z_mean"][i] = acc_z_mean
        acc_z_std = np.std(curr["linear_acceleration_Z"])
        processed["acc_z_std"][i] = acc_z_std
        acc_z_range = max(curr["linear_acceleration_Z"])-min(curr["linear_acceleration_Z"])
        processed["acc_z_range"][i] = acc_z_range
        
    processed.to_csv(out_name)
    return processed


# Now the first payoff- Building the trainining dataframes

# In[24]:


train = processor(x_train,'training_processed.csv',3809)
test = processor(x_test,'testing_processed.csv',3815)

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train['surface'])


# With this taken care of, we can get to building the actual model
# This is another place where I think improvements can be made in shaping the model. This is my first idea for doing this

# X = train.values
# 
# model = keras.Sequential([
# 
# keras.layers.Dense(25, activation=tf.nn.relu),
# keras.layers.BatchNormalization(), #Absolutely no idea if this will do anything, throwing data science at the wall and seeing what sticks
# keras.layers.Dense(25, activation=tf.nn.relu),
# keras.layers.Dense(9, activation=tf.nn.softmax)
# ])
# 
# stop = keras.callbacks.EarlyStopping(monitor='loss')
# callbacks = [stop]
# 
# model.compile(optimizer='adam', 
# loss='sparse_categorical_crossentropy',
# metrics=['accuracy'])
# 
# 
# model.fit(X, y_train, epochs=1000,batch_size = 32)
# model.save_weights('attempt.hd5')

# ...But what if we did this another way? Time for me to find out what random forests are

# X = train.values
# 
# clf = RandomForestClassifier(n_estimators = 200)
# 
# clf.fit(X,y_train)

# In[29]:


#You know what, lets try this XGBoost thing they've been talking about
model = XGBClassifier(n_estimators = 2000)
X = train.values
model.fit(X,y_train,early_stopping_rounds=5,eval_set = [(X,y_train)]) #Too laxy to split r/n,just seeing if this works


predictions = model.predict(test.values)
predictions = le.inverse_transform(predictions)

indexes = range(0,3816)
columns = ["series_id","surface"]
submission = pd.DataFrame(index = indexes, columns = columns)
submission["series_id"]= range(0,3816)
submission["surface"]=predictions

submission.to_csv('predictions.csv', index = False)


# Now we apply the model to the test set

# #This is the predictions using the NN implementation
# predictions = model.predict(test)
# indexes = range(0,3816)
# true_predictions = np.zeros((3816,)) #Ask Mike about this one
# 
# #convert the number arrays to actual surfaces
# for i in range(len(predictions)):
#     true_predictions[i]= np.argmax(predictions[i])
# 
# true_predictions = true_predictions.astype(int)
# true_predictions = le.inverse_transform(true_predictions)  
#     
# columns = ["series_id","surface"]
# submission = pd.DataFrame(index = indexes, columns = columns)
# 
# submission["series_id"]= range(0,3816)
# submission["surface"]=true_predictions
# 
# submission.to_csv('predictions.csv', index = False)

# #Random forest implementation
# 
# predictions=clf.predict(test.values)
# 
# predictions = le.inverse_transform(predictions)
# 
# indexes = range(0,3816)
# columns = ["series_id","surface"]
# submission = pd.DataFrame(index = indexes, columns = columns)
# submission["series_id"]= range(0,3816)
# submission["surface"]=predictions
# 
# submission.to_csv('predictions.csv', index = False)

# #Random forest implementation
# 
# predictions=clf.predict(test.values)
# 
# predictions = le.inverse_transform(predictions)
# 
# indexes = range(0,3816)
# columns = ["series_id","surface"]
# submission = pd.DataFrame(index = indexes, columns = columns)
# submission["series_id"]= range(0,3816)
# submission["surface"]=predictions
# 
# submission.to_csv('predictions.csv', index = False)
