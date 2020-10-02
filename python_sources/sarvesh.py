#!/usr/bin/env python
# coding: utf-8

# ## Team Name: Sarvesh
#   **Sarvesh** **Mayilvahanan**
#   
# ## Table 58

# ## What it does
# The model takes in a wide variety of data from 107 sensors to predict the equipment failure.
# ## How I built it
# I built the model using Google Colab and the libraries listed above. I first converted the data from the csv file into an array and performed some initial refining of the data, including converting the "na" values for the sensors into a negative number since none of the sensors are able to read negative values. The data was then randomly shuffled and passed into the model. The model consists of a Dense layer that takes the input tensor of size 170 and converts it to a tensor of size 16. This is then followed by a dropout layer with a dropout rate of 0.15, which drops the values of about 15% of the input values, preventing the model from overfitting. This is then passed into another Dense layer which converts the tensor of size 16 to a tensor of size 32. This is finally passed into a Dense layer of size 1 with a sigmoid activation. This returns a value between 0 and 1 that represents the probability that the target value is 1. The model was optimized using a RMSprop optimizer. The training data was split into training and validation data, with training data taking about 80% of the data and validation taking the rest. The model was then tested on the test data, and the predictions were uploaded to Kaggle. The model achieved a Mean F1 score of 0.99125.
# ## Challenges I ran into
# Some of the challenges I ran into was trying to find a good model architecture. I first started with several Dense layers, and realized that the model was a little too complex and could be simplified. I then tested several other model architectures such as a Recurrent Neural Network and a simple Dense layer model. I finally tested a simple Dense layer model with dropout and it increased the performance somewhat, and it was the model that I finally settled on.
# ## Accomplishments that I'm proud of
# I'm proud of the wide variety of models that I tested and was able to get to work.
# ## What I learned
# I learned how certain ML techniques can vastly effect how the model learns and performs.
# ## What's next
# The model could definitely be further improved to achieve a better Mean F1 score and gain better accuracy. This could be done by trying even more model architectures and playing around with tensor sizes and dropout rates.

# # Setting Up Data
# 

# In[ ]:


cd ../input/equipfails


# In[ ]:


data = open("equip_failures_training_set.csv")

features  = [] #Array to hold all the feature values

firstline = data.readline() #Takes the header names for all the columns

for line in data:
  event = [] #Contains data for each individual event
  
  linedata = line.split(",")
  
  for i in range(len(linedata)):
    if (linedata[i] == "na" or linedata[i] == "na\n"): #Converting na values to a negative number (none of the sensors can read negative values)
      linedata[i] = -100000
  
  id = float(linedata[0]) #id number for the event
  target = float(linedata[1]) #target values for the event
  
  event.append(id)
  
  for j in range(170): #Looping through the sensor values
    event.append(float(linedata[j+2])) #adding sensor values to event array
    
  event.append(target)
  
  features.append(event) #Adding the event array to the total features array


# In[ ]:


print("There are ", len(event)-2, " features in each event.")
print("There are ", len(features), " events in the training dataset.")


# In[ ]:


import numpy as np

total_data = np.array(features)

np.random.shuffle(total_data) #shuffle the data to be randomly ordered

targets = []
ids = []

for i in range(len(total_data)): #remove the ids and targets from the features data
  ids.append(total_data[i][0])
  targets.append(total_data[i][-1])
  
total_data = np.delete(total_data, -1, 1)
total_data = np.delete(total_data, 0, 1)

all_targets = np.array(targets)


# In[ ]:


print("The tensor shape of the features is: ", total_data.shape)
print("The tensor shape of the targets is: ", all_targets.shape)


# In[ ]:


#Split the training data to also have validation data
#80% of the training data is used for training and the remaining 20% is used for validation
train_data = total_data[:48000]
validation_data = total_data[48000:]

train_targets = all_targets[:48000]
validation_targets = all_targets[48000:]


# In[ ]:


#Normalizing all the feature data using a z scale

mean = train_data.mean(axis=0)
train_data -= mean

std = train_data.std(axis=0)
train_data /= std

validation_data -= mean
validation_data /= std


# # Building Model Architecture and Training

# In[ ]:


from keras import layers
from keras import models

#Build a simple 3 layer Dense model ending with a sigmoid layer that will provide a value between 0 and 1 that represents the probability that the value is 1.
model = models.Sequential()
#The input shape is based on the shape of the input tensor
model.add(layers.Dense(16, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(32, activation='relu'))
#returns the probability that the target value is 1
model.add(layers.Dense(1, activation='sigmoid'))


# In[ ]:


model.summary() #Breakdown of the model's layers


# In[ ]:


from keras import optimizers

#Define the loss function and optimizer. Only take the accuracy of the model
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-2), metrics=['acc'])


# In[ ]:


#Train the model
history = model.fit(train_data, train_targets, epochs=20, batch_size=512, validation_data=(validation_data, validation_targets))


# In[ ]:


#Plot the loss of the model versus epochs run
import matplotlib.pyplot as plt
history_dict = history.history
acc_values = history_dict['acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


#Plot the accuracy of the model versus epochs run
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'r', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# # Testing the model

# In[ ]:


#Prepare the test data
data = open("equip_failures_test_set.csv")

features  = [] #Array to hold all the feature values

firstline = data.readline() #Takes the header names for all the columns

for line in data:
  event = [] #Contains data for each individual event
  
  linedata = line.split(",")
  
  for i in range(len(linedata)):
    if (linedata[i] == "na" or linedata[i] == "na\n"): #Converting na values to a negative number (none of the sensors can read negative values)
      linedata[i] = -100000
  
  id = float(linedata[0]) #id number for the event
  
  event.append(id)
  
  for j in range(170): #Looping through the sensor values
    event.append(float(linedata[j+1])) #adding sensor values to event array
  
  features.append(event) #Adding the event array to the total features array


# In[ ]:


print("There are ", len(event)-1, " features in each event.")
print("There are ", len(features), " events in the test dataset.")


# In[ ]:


test_data = np.array(features)

ids = []

for i in range(len(test_data)): #remove the ids and targets from the features data
  ids.append(test_data[i][0])
  
test_data = np.delete(test_data, 0, 1)


# In[ ]:


print("The tensor shape of the features is: ", test_data.shape)
print("The number of events is: ", len(ids))


# In[ ]:


#Normalizing all the feature data using the mean and standard deviation from the training data.

#mean = test_data.mean(axis=0)
test_data -= mean

#std = test_data.std(axis=0)
test_data /= std


# In[ ]:


print(test_data[0])


# In[ ]:


print(len(test_data))
print(len(ids))


# In[ ]:


#Test the model on the test data and put the calculated values into an array
predictions = model.predict(test_data)


# In[ ]:


print(predictions)


# file = open("submission.csv", "w+") #create the submission file
# 
# file.write("id,target\n") #Header
# 
# for i in range(len(ids)):
#   if (predictions[i] >= 0.5): #if the prediction is above 0.5, then assume the value 1
#     target = 1
#   else:
#     target = 0 #otherwise, assume the value 0
#   file.write("%d,%d\n" %(ids[i], target))
#   
# file.close()

# # Testing a RNN

# In[ ]:


data = open("equip_failures_training_set.csv")

features  = [] #Array to hold all the feature values

firstline = data.readline() #Takes the header names for all the columns

for line in data:
  event = [] #Contains data for each individual event
  
  linedata = line.split(",")
  
  for i in range(len(linedata)):
    if (linedata[i] == "na" or linedata[i] == "na\n"): #Converting na values to a negative number (none of the sensors can read negative values)
      linedata[i] = -100000
  
  id = float(linedata[0]) #id number for the event
  target = float(linedata[1]) #target values for the event
  
  event.append(id)
  
  for j in range(170): #Looping through the sensor values
    event.append(float(linedata[j+2])) #adding sensor values to event array
    
  event.append(target)
  
  features.append(event) #Adding the event array to the total features array


# In[ ]:


print("There are ", len(event)-2, " features in each event.")
print("There are ", len(features), " events in the training dataset.")


# In[ ]:


import numpy as np

total_data = np.array(features)

np.random.shuffle(total_data) #shuffle the data to be randomly ordered

targets = []
ids = []

for i in range(len(total_data)): #remove the ids and targets from the features data
  ids.append(total_data[i][0])
  targets.append(total_data[i][-1])
  
total_data = np.delete(total_data, -1, 1)
total_data = np.delete(total_data, 0, 1)

all_targets = np.array(targets)


# In[ ]:


print("The tensor shape of the features is: ", total_data.shape)
print("The tensor shape of the targets is: ", all_targets.shape)


# In[ ]:


#Split the training data to also have validation data
#75% of the training data is used for training and the remaining 25% is used for validation
train_data = total_data[:45000]
validation_data = total_data[45000:]

train_targets = all_targets[:45000]
validation_targets = all_targets[45000:]


# In[ ]:


#Normalizing all the feature data

mean = train_data.mean(axis=0)
train_data -= mean

std = train_data.std(axis=0)
train_data /= std

validation_data -= mean
validation_data /= std


# In[ ]:


print(train_data[0])
print(train_targets)

print(validation_data[0])
print(validation_targets)


# In[ ]:


from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(170, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


from keras import optimizers

#Define the loss function and optimizer. Only take the accuracy of the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


# In[ ]:


#Train the model
history = model.fit(train_data, train_targets, epochs=20, batch_size=512, validation_data=(validation_data, validation_targets))


# In[ ]:


import matplotlib.pyplot as plt
history_dict = history.history
acc_values = history_dict['acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'r', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


#New model with less epochs to reduce overfitting

model = Sequential()
model.add(Embedding(170, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])


# In[ ]:


#Train the model
history = model.fit(train_data, train_targets, epochs=5, batch_size=512, validation_data=(validation_data, validation_targets))


# In[ ]:


import matplotlib.pyplot as plt
history_dict = history.history
acc_values = history_dict['acc']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'r', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Testing of this RNN model showed that it was not as accurate or efficient as the basic Dense layer model with dropout
