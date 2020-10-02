#!/usr/bin/env python
# coding: utf-8

# # Data
# **Hi, today I will show you how to classify Human Actions per sample with Deep Learning Methods.In previous notebooks that I shared, I already show you how to classify Human Action Data with classical Machine Learning Technics.
# ![action_samples_surface_800.jpg](attachment:action_samples_surface_800.jpg)**

# # There are 13 actions:
# **1	Jumping in place	
# 2	Jumping jacks	
# 3	Bending - hands up all the way down	
# 4	Punching (boxing)	
# 5	Waving - two hands	
# 6	Waving - one hand (right)
# 7	Clapping hands	
# 8	Throwing a ball
# 9	Sit down then stand up	
# 10	Sit down	
# 11	Stand up	
# 12	T-pose**

# # Data Storage Format
# 
# - The files in database are in format of "txt".In txt files there are joint coordinates of 43 markers in human body in order.
# 

# # Let's start code!
# **First, we import pandas and os library.Then, we iterate over all txt files and collect all joint data with labels in a dataframe.Every row of dataframe is a action in a sample, and there are 129 xyz joint coordinates and 1 class label named 'classs' column.**

# In[ ]:


import pandas as pd
import os
path = '/kaggle/input/berkeley-multimodal-human-action-database/'

full_data = pd.DataFrame()

for entry in sorted(os.listdir(path)):
    if os.path.isfile(os.path.join(path, entry)):
        if entry.endswith('.txt'):
            data = pd.read_csv(path+entry,sep=' ',header=None)
            data.drop([129,130],inplace=True,axis=1)
            data['classs'] = entry[-10:-8]
            full_data = pd.concat([full_data,data],ignore_index=True)


# **Let's check the data size of data.There are 2401920 rows(samples) and 130 features, last feature named 'classs' is label of that row.**

# In[ ]:


full_data.shape


# **When we check data types of the dataframe we can clearly see that first 129 rows are x,y,z features in format of float and the last column named 'classs' is object in string format which is labels.**

# In[ ]:


full_data.dtypes


# **Let's have a look to data.**

# In[ ]:


full_data.head()


# **In this section, I want to have a look to memory usage of dataframe.You know that kaggle provide us 16GB memory and this dataframe is quaite big.I just wanted to see that "can kaggle hardware handle this?"**

# In[ ]:


full_data.info()


# **In this section, we split data as features and label."x" variable is features and "y" variable is label of that features in order.**

# In[ ]:


x = full_data.drop(["classs"],axis=1)
y = full_data.classs.values
x.head()


# **("y") Label data is in format of string,for 01-11 classes are basicly can be convertable to integer.But 12th class is like "t-" which is a string.So in this section we replace "t-" with "12" and convert it to integer.**

# In[ ]:


y = pd.DataFrame(y)
y.iloc[:,0] = y.iloc[:,0].str.replace('t','1')
y.iloc[:,0] = y.iloc[:,0].str.replace('-','2')
y.astype('int32')


# **Now, we can split data as train and test data.**

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=True)
print('Shape of train data is : ',x_train.shape)
print('Shape of label data is : ',y_train.shape)


# **"Y" label data in format of integer, but we have to convert it to categorical to use in keras "categorical crossentropy" loss.**

# In[ ]:


from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# # Model Build
# 
# - You can see the neuron number for all denses, I used 'relu' activation for this time, but you can change it to tune hyperparameters.The input shape is 129 which is features(columns) for all rows.
# - I choose optimizer 'adam', loss function 'categorical crossentropy'.
# - I also define a basic EarlyStopping to prevent overtrain.
# - Also, in model summary you can see the model clearly.In fact, I choose neuron numbers randomly with power of 2's.

# In[ ]:


from keras.callbacks import EarlyStopping
from tensorflow import keras

early_stop = EarlyStopping(monitor='loss', patience=2)
model = keras.Sequential()

model.add(keras.layers.Dense(128, activation='relu', input_shape=(129,)))

model.add(keras.layers.Dense(256, activation='relu'))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(256, activation='relu'))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(64, activation='relu'))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(256, activation='relu'))

model.add(keras.layers.Dense(128, activation='relu'))

model.add(keras.layers.Dense(64, activation='relu'))

model.add(keras.layers.Dense(13, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# # Train Area
# - I choose %20 of train data as validation data.
# - Batch size is 128.
# - 6 epochs used.

# In[ ]:


hist = model.fit(x_train , y_train , epochs=6, validation_split=0.20, batch_size= 128,callbacks=[early_stop])


# **Let's look train and validation loss.In the graph we can clearly see that validation loss is lower which is a very good result.**

# In[ ]:


from matplotlib import pyplot as plt
print(hist.history.keys())

plt.plot(hist.history['loss'],label = 'Train loss')
plt.plot(hist.history['val_loss'],label = 'Val loss')
plt.legend()
plt.show()


# # Model Test
# 
# - In this section, I tried my model in test data which the model hasn't seen yet.As you see, the accuracy is %98 which is better result than train and validation accuracy.so model works well.

# In[ ]:


hist2 = model.evaluate(x_test,y_test)


# **Thanks for read my notebook.If you encounter a problem in this notebook, please contact with me.If you like my notebook, please upvote to reach more people.Thanks :)**
