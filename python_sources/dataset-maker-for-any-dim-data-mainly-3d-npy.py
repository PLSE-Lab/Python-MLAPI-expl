#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Create any dataset for any type of input by Meltem Atay
#Prepare input packages
import os
import numpy as np
from os import listdir
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
### In case of images from scipy.misc import  imread, imresize could be necessary


# In[ ]:


# Folder structure
#-input_data
#--label1
#---class1data1
#---class1data2
#---class1data3
#... (similarly all data of label 1 goes like that)
#--label2
#---class2data1
#---class2data2
#---class2data3
#... (similarly all data of label 2 goes like that)
#--label3
#---class3data1
#---class3data2
#---class23data3
#... (similarly all data of label 3 goes like that)
#..... same structure until the end of your labels and dataset


# In[ ]:


# Give data dimensions in case of 3D data there would be 3 dimensions to provide, 
# if they are the same just repeat one dimension 3 times...
data_dim1= # 1st dimension of your data
data_dim2= # 2nd dimension of your data
data_dim3= # 3rd dimension of your data
channels = 1 # 1: Grayscale, 3: RGB
labels = 2 # how many classes in dataset?
test_ratio = 0.2 # split 20% of data as test set the 80 % would be the training set
input_folder = 'alldata'
data_folder = 'data1'


# In[ ]:


def get_data(input_folder):
    my_data = np.load(input_folder)
    #my_data = imread(input_folder, flatten= True if channels == 1 else False) # to make all the color layers into a single grayscale layer
    #my_data = imresize(my_data, (dim1, dim2, channels)) 
    # if your data is a picture uncomment the code above 
    # so you can read your data using imread and resize your data using imresize
    return my_data


# In[ ]:


labels = listdir(input_folder) 
X, Y = [], []


# In[ ]:


# now merging data with labels
for i, label in enumerate(labels):
    label_of_folder = input_folder+'/'+label
    for my_data_labeled in listdir(label_of_folder):
        my_data = get_data(label_of_folder+'/'+my_data_labeled)
        X.append(my_data)
        Y.append(i)


# In[ ]:


X = np.array(X).astype('float32')/255.
print(X.shape)


# In[ ]:


#X = X.reshape(data_dim1, data_dim2, data_dim3, channels) 
#X = X.reshape(X.shape[0], data_dim2, data_dim3, channels) #same outputs...


# In[ ]:


X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])  #I know X.shape[3] is the channel info
# instead of providing additional dimensions if data in its form 


# In[ ]:


Y = np.array(Y).astype('float32')
print(Y.shape)


# In[ ]:


Y = to_categorical(Y, labels)


# In[ ]:


if not os.path.exists(data_folder+'/'):
    os.makedirs(data_folder+'/')


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio, random_state=42)


# In[ ]:


x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)


# In[ ]:


np.save(data_folder+'/x_train.npy', x_train)


# In[ ]:


np.save(data_folder+'/x_test.npy', x_test)


# In[ ]:


np.save(data_folder+'/x_val.npy', x_test)


# In[ ]:


np.save(data_folder+'/y_train.npy', y_train)


# In[ ]:


np.save(data_folder+'/y_test.npy', y_test)


# In[ ]:


np.save(data_folder+'/y_val.npy', y_test)

