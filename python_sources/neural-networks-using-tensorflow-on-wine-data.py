#!/usr/bin/env python
# coding: utf-8

# # Neural Network on Winedata using Tensorflow

# This a simple neural network classification on the winedata, I have kept it simple, not much of the data preprocessing and visualization. 

# Frankly speaking, this dataset is not a worth of Neural Networks, but I chose this dataset, since this is my first attempt on NN's and i want to keep it simple.

# The dataset is available at 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#  The Column names and other details are at 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names'

# In[ ]:


colnames = ['Class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','dilute','Proline']


# In[ ]:


df = pd.read_csv('../input/wine.data.txt',names = colnames,index_col = False)


# Snapshot of the data

# In[ ]:


df.head()


# Check if it has any null values

# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# To know about the class distribution.

# In[ ]:


df.Class.value_counts()


# Check which columns are correlated 

# In[ ]:


df.corr()


# As shown above Ash has very very low correlation in detecting the class, so it can be removed to improve our model. I'm dropping the ash columns in the following code.

# # We convert the Class labels into the Onehot format.

# In[ ]:


df = pd.get_dummies(df, columns=['Class'])


# Now only take the labels into to new dataframe

# In[ ]:


labels = df.loc[:,['Class_1','Class_2','Class_3']]


# For Neural Nets the data should be in numpy arrays, so convert them,

# In[ ]:


labels = labels.values


# Now collect the features dataframe

# In[ ]:


features = df.drop(['Class_1','Class_2','Class_3','Ash'],axis = 1)


# Convert the feature dataframe to numpy arrays

# In[ ]:


features = features.values


# In[ ]:


print(type(labels))
print(type(features))


# Check the shape of the arrays

# In[ ]:


print(labels.shape)
print(features.shape)


# Split into the training and testing sets, We have just 178 columns, which is a very very very small data for NN's

# In[ ]:


train_x,test_x,train_y,test_y = train_test_split(features,labels)


# Print the shapes of the split datasets.

# In[ ]:


print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)


# Everything looks good, so lets go further.

# Neural Networks perform better if the data is scaled between (0,1). So, lets do it.

# In[ ]:


scale = MinMaxScaler(feature_range = (0,1))


# In[ ]:


train_x = scale.fit_transform(train_x)
test_x = scale.fit_transform(test_x)


# Snapshot of the features and labels

# In[ ]:


print(train_x[0])


# In[ ]:


print(train_y[0])


# # The Neural Network part begins here.

# In[ ]:


X = tf.placeholder(tf.float32,[None,12]) # Since we have 12 features as input
y = tf.placeholder(tf.float32,[None,3])  # Since we have 3 outut labels


# Lets create our model with 2 hidden layers with 80 and 50 nodes respectively.
# It was suggested(by online tutor) to use xavier_initializer for weights and zeros initializer for biases, but not mandatory.( I have been used to it, so i continued with this)
# I have also ran the model with random weights, they eventually optimize.

# In[ ]:


weights1 = tf.get_variable("weights1",shape=[12,80],initializer = tf.contrib.layers.xavier_initializer())
biases1 = tf.get_variable("biases1",shape = [80],initializer = tf.zeros_initializer)
layer1out = tf.nn.relu(tf.matmul(X,weights1)+biases1)

weights2 = tf.get_variable("weights2",shape=[80,50],initializer = tf.contrib.layers.xavier_initializer())
biases2 = tf.get_variable("biases2",shape = [50],initializer = tf.zeros_initializer)
layer2out = tf.nn.relu(tf.matmul(layer1out,weights2)+biases2)

weights3 = tf.get_variable("weights3",shape=[50,3],initializer = tf.contrib.layers.xavier_initializer())
biases3 = tf.get_variable("biases3",shape = [3],initializer = tf.zeros_initializer)
prediction =tf.matmul(layer2out,weights3)+biases3


# Define the loss function, softmax_cross_entropy_with_logits_v2 is suggested over softmax_cross_entropy_with_logits because of label backpropagation, and then optimize the loss function. I have choose the learning rate as 0.001
# 

# In[ ]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# I'm expecting that everyone has an idea about the below process so I'm not going to elaborate much on this.

# Matches is a list(tensor) which takes 1, if the index of largest element in prediction and y are equal and 0 it the indices are not equal.
# Accuracy is calculated by taking the mean of those matches.

# In[ ]:


acc = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(201):
        opt,costval = sess.run([optimizer,cost],feed_dict = {X:train_x,y:train_y})
        matches = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(matches, 'float'))
        acc.append(accuracy.eval({X:test_x,y:test_y}))
        if(epoch % 100 == 0):
            print("Epoch", epoch, "--" , "Cost",costval)
            print("Accuracy on the test set ->",accuracy.eval({X:test_x,y:test_y}))
    print("FINISHED !!!")


# We can see that they cost(loss) is reducing and the Accuracy is increasing, which shows that our model is training.

# Lets plot the Accuracy over epochs

# In[ ]:


plt.plot(acc)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")


# The graph of the accuracy over training steps(Epochs)

# # Last words.
# This is not the best dataset for the neural networks. I would suggest something huge, like really huge.
# The accuracy we got might not the best we can get.We can tune the model more by changing the epochs, learning rate.
# There is no much significane of this kernel, better accuracies might be achieved by using sklearn's logisticregression etc.This is just for my understanding of the neural networks.
# Thankyou!
