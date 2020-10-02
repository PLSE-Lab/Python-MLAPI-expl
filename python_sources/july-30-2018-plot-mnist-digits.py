'''
Dated: July26-2018 
Author: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for #display 5th digit of mnist dataset

Results:

'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist

(train_data,train_labels),(test_data,test_labels) = mnist.load_data()

print("train_data.shape")
print(train_data.shape)

print("displaying the 5th image")
plt.imshow(train_data[4],cmap=plt.cm.binary)
plt.show()

#selecting 10th image to the 100th image
my_slice = train_data[10:15]
print("my_slice.shape")
print(my_slice.shape)

print("train_data.dtype")
print(train_data.dtype)