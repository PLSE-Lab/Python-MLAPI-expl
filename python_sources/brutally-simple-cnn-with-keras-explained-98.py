#!/usr/bin/env python
# coding: utf-8

# Hi all, hope this will help you in your ML journey
# 
# So a bearbone ML flow having this components
# 1. data
# 2. model
# 3. prediction
# 
# Explaination alone the way
# 
# this notebook won't have anything fancy, but hopefully get you interested.

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow.keras as keras


# Explaination for the data we get
# 
# ## data
# ### train.csv
# In this case we have a 28 * 28 photo of a hand writing digit, which is represented by 784 columns for each pixel.
# In additional of the 784 columns, the first column (named "label"), holds the correct solution of the prediction.
# <span style="color:blue">        
# note: The first column ("label" column) needed to be seperated before we feed it into the model, but it's a simple task.
# </span>
# 
# <span style="color:brown">  
# fun note: so this is like the homework for the model, where it have the question and solution. The model does the homework over and over again, knowing how well it did and improve itself next time.
# </span>
#         
# ### test.csv
# This section contains only the 28 * 28 photos, with no label columns. Feed the photos to the model and submit the models prediction to the competion. 
# 
# <span style="color:brown">
# fun note: so this is the final exam for the model, only questions the model never saw and will be a good estimate on how well they perform in the real world.
# </span>

# In[ ]:


#reading both files
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
val_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# ## processing the data before feeding it into the model
# 
# 1. seperating the label and the actual data in train.csv
# 2. change the label from 0 - 9 to **One Hot Encoding**
# 
# One Hot encoding with 10 classes
# ```
# solution -> One Hot version of the solution
# 0 -> [1,0,0,0,0, 0,0,0,0,0]
# 1 -> [0,1,0,0,0, 0,0,0,0,0]
# 2 -> [0,0,1,0,0, 0,0,0,0,0]
# 3 -> [0,0,0,1,0, 0,0,0,0,0]
# 4 -> [0,0,0,0,1, 0,0,0,0,0]
# 5 -> [0,0,0,0,0, 1,0,0,0,0]
# 6 -> [0,0,0,0,0, 0,1,0,0,0]
# 7 -> [0,0,0,0,0, 0,0,1,0,0]
# 8 -> [0,0,0,0,0, 0,0,0,1,0]
# 9 -> [0,0,0,0,0, 0,0,0,0,1]
# ```
# 
# I think you see what one hot encoding is here. Only one bit are "hot" down there, and each bit represents a possible answer (0 to 9 in our case)
# 
# <span style="color:brown">  
# why One Hot Encoding?
# <span style="color:brown">  
# mainly an easy way to let the model know how wrong it's prediction are. Think about confident level for each solution that the model returns. We reward confidents level for the correct solution and pushish any confidence on the incorrect solutions.
# </span>
# 
# <span style="color:brown"> 
# let's say you gave the model a hand-writen photo of 0, and it returns it's confident level for each class.
# ```
# confident level we got back from the model:
# number 0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8  , 9
#       [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3]
# ```
# <span style="color:blue"> 
# (what I am gonna talk about the main idea of loss function, in actuality there's so many different better way to do so).
#     
# <span style="color:brown"> 
# Here we got some pretty nice prediction, where the model is 90% sure the solution is a 0, so we reward the model 0.9 points, but we don't like the fact that the model is hassitaing that it might be a 6 or 9. Eventhough they might look similar, we would still like the model to be more confident on it's answer, so we will take off 0.2 points and 0.3 points respectively. The Final score for this prediction is 0.9 - (0.2+0.3) = **0.4**
# 
# <span style="color:brown"> 
# So the model have 2 ways to improve the score, one is to increase the condifence level of the correct solution, another is to reduce the confidentce level of other numbers. The model and go back and adjust it's nural network to achieve it.

# In[ ]:


NUM_CLASS = 10

#making one hot encoding for the label
label = data['label']
label_one_hot = np.zeros((label.size, NUM_CLASS))
for i in range(label.size):
    label_one_hot[i,label[i]] = 1
#remove the label column, so the remaining 784 columns can form a 28*28 photo
del data['label']

#changing data from DataFrame object to a numpy array, cause I know numpy better :p
data = data.to_numpy()
print(data.shape)

#making data to 28*28 photo
data = data.reshape(-1,28,28,1)


# In[ ]:


#checking out data shape
print(' data shape: {} \n one hot lable shape: {}'.format(
    data.shape, label_one_hot.shape))


# Okay, Here we finish preparing our data
# 
# On the cell above, we can have data and label_one_hot, where one is the question (28 * 28 photo) and one is the solution
# 
# and it also shows that we have 42000 photos and solution pairs to work with.

# # Model
# 
# ## it's magic 
# 
# joking, personally I think there are blog post that can explain way better and deeper than I can.
# 
# Here is my attempt to explain CNN (convolutional nueral net)
# 
# so everthing after the `flatten()` is 1 layer nueral net, trying to reduce all the features into confident level for each number.
# 
# ### Convolution layer
# this layer uses a 3 x 3 kernal (`kernal_size = (3,3)`) that scans through the photo overlappingly and extract 5 different feature maps (`filters = 5`). A 28 x 28 x 1 photo goes in will come out as a 26 x 26 x 5 feature map. (26 x 26 is becuse a 3 x 3 filter can only have 26 position in a 28 pixel edge)
# 
# <span style="color:brown"> 
# Main take away: it extracts features
# 
# ### MaxPooling layer
# this layer uses a 2 x 2 kernal (`pool_size = (2,2)`) that scans through the photo non-overlappingly, selects the largest value and output it to the next level. So a 26 x 26 x 5 feature map from the last layer will become 13 x 13 x 5 for the next layer (it doesn't reduce the z axis). It's main function is to reduce the input size for the next layer with some other benifit of like less sensitive to local variation, etc, etc.
# 
# <span style="color:brown"> 
# Main take away: it reduce the size. smaller the photo, trains faster.
#     
# ### Batch Normalization layer
# it Normalize the photo by the mean and variance of the photo. Making it more uniform for the next layer to work on. Usually performed after the pooling layers (or any upscale or downscale layers).
# 
# <span style="color:brown"> 
# Main take away: it conditions the photo. Making it easier for the next layer to work on the data.
#     
# ### Flatten layer
# the layer squizes everything into a 1 dimension array. for example, a 13 x 13 x 5 feature map will turn into a array of 845. for preperation of the regular neural net.

# In[ ]:


#simple CNN model with Keras
import keras
from keras.models import Model, Sequential
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Flatten, Activation

model = Sequential([
    
    Convolution2D(filters = 5, kernel_size = (3,3), activation = 'relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    BatchNormalization(),
    #do a drop out here if interested
    
    Convolution2D(filters = 25, kernel_size = (3,3), activation = 'relu'),
    MaxPooling2D(pool_size=(2,2)),
    BatchNormalization(),
    #do a drop out here if interested
    
    Flatten(),
    Dense(10),
    Activation('softmax')
])


# ## Compile model
# 
# Here we can see we complieled it with adam optimizer and a loss function of catagorical crossentropy.
# 
# The metrics is optional, we added accuracy here for use extra information.
# 
# <span style="color:brown"> 
# fun note: basically telling what the model how it should do better next time (optimizer) and how performance is measured (loss function). Also asking it to show up some other performance metrics during training for us to see (metrics).

# In[ ]:


model.compile('adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
             )


# ## Training model
# 
# Here we provide the model with training set (data and label_one_hot), and we ask it to go through the set 10 times (epochs = 10).
# But we ask the model to only train on 90% of the data, and the remaining 10% as a validation data set. (validation_split = 0.1)
# 
# <span style="color:brown"> 
# fun note: validation_split is like reserving some question as a mid-term or quiz, it's used as a rough estimation of real world performce for us to see the model's traning progress.
#     
# If the validation data accuracy starts to decrease, while the training accuracy still increase. It is a good indication of over-fitting occuring and should reduce the epoches.
# <span style="color:blue">
# There's a automatic way to do so. It's called early stopping, you should check it out ;)
# 
# <span style="color:brown"> 
# fun note: Over-fitting is like when the student did the same question sets for too many times and kinda remembers the solution instead of learning.

# In[ ]:


#Starts training the model over the data 10 times.
#Here nothing fancy added for keeping it really really simple.

history = model.fit(data, label_one_hot, epochs = 10, validation_split = 0.1)


# # End of traning
# 
# here is a graph on how the model did for each epoches. Just some visulization for how the model performs, but it can also a critical part to see further improve your model.

# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()


# # Prediction
# 
# The model are given a set of 28 * 28 photos that it haven't seen before and are ask to give a prediction on what number it is.
# 
# <span style="color:brown"> 
# fun note: Well, it's the final exam for the model. Wish the model luck and wish you luck too.

# In[ ]:


#we read the csv before, but just read it again here.
val_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

#the same way to process the training data after seperating the label
val_data = val_data.to_numpy()
val_data = val_data.reshape(-1,28,28,1)

#here we ask the model to predict what the class is
raw_result = model.predict(val_data)

#note: model.predict will return the confidence level for all 10 class,
#      therefore we want to pick the most confident one and return it as the final prediction
result = np.argmax(raw_result, axis = 1)

#generating the output, remember to submit the result to the competition afterward for your final score.
submission = pd.DataFrame({'ImageId':range(1,len(val_data) + 1), 'Label':np.argmax(raw_result, axis = 1)})
submission.to_csv('SimpleCnnSubmission.csv', index=False)


# It's a fun ride, hope you learn something here and there :)
