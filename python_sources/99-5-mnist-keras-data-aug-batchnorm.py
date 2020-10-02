#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import keras


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


train.shape, test.shape


# In[ ]:


#Randomly shuffling the training set (incase consecutive digits are same)
indices = np.arange(len(train))
np.random.shuffle(indices)
trainShuffled = train.loc[indices,:]


# In[ ]:


#A visual plot of the images
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.imshow(np.array(trainShuffled.iloc[i,1:]).reshape(28,28), cmap = 'gray')
  plt.axis('off')


# **Convolution Neural Network**
# It has the following architecture:
# Conv2D(32) -> MaxPool -> BN -> Conv2D(64) -> MaxPool -> BN -> Conv2D(128) -> Dropout(50%) -> Dense 10 (output layer)
# 
# **Batch Normalization:** it improves the convergence rate in optimization problem, when we are trying to minimize our loss. It has same idea like when we divide the features (input) by 255 (because images are from 0 to 255) or when we use:
# x = (x-mean)/standard deviation
# So, doing this for all hidden units, sometimes is benefitial.
# 
# **Dropout:** it involves randomly turning off certain neurons in the hidden layer it is specified for (0.5 means 50% after every batch, because we are using mini-batch gradient descent). This prevents overfitting, as the network would not rely on certain neurons always to get the class.

# In[ ]:



#Creating Keras Model, consiting of Conv2d + Pooling blocks followed by one Dense layer with 10 hidden units
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), input_shape = (28,28,1,), activation = 'relu', name = 'Conv2D1'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', name = 'Conv2D2'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, (2,2),activation = 'relu', name = 'Conv2D3'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation = 'softmax'))


# In[ ]:


model.summary()


# **Data Augmentation**
# It is another method like Dropout, to prevent overfitting. One way to deal with overfitting is more data. This is not always possible, so what we do here is we create new data from the given data. How? We have images, so we randomly rotate the images by some degrees (not too extreme) and we shear it, or scale the height and width. We can also zoom it. There are many more parameters that can be experiemented with, like flipping the image (which I found not to be useful), but there are more.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


generator = ImageDataGenerator(rescale = 1./255, rotation_range = 15, width_shift_range = 0.1, height_shift_range = 0.1,
                              shear_range = 0.1, zoom_range = 0.1)


# **Validation or Hold-out Set**
# To know if our model is overfitting or not, and to have a rough idea about the number of epochs it would require to get our desired result, we use a validation data that is seaparate from the train data and the model hasn't seen this data before.
# It is somewhat equivalent to test data, but since we sometimes tune the hyperparameters of our network based on results obtained on validation data, we are optimizing our model for validation data and it might not perform that well on test data, but comparison should be somewhat similar.

# In[ ]:


#Validation set and train set split
train_X = trainShuffled.iloc[:35000, 1:]
train_y = trainShuffled.iloc[:35000, 0]
test_X = trainShuffled.iloc[35000:, 1:]
test_y = trainShuffled.iloc[35000:, 0]
train_X.shape, train_y.shape, test_X.shape, test_y.shape


# In[ ]:


train_X = np.array(train_X).reshape(-1,28,28,1) #Networks expects a 4D input
                                        #(samples, height ,width, channel), since not RGB channel = 1
test_X = np.array(test_X).reshape(-1,28,28,1)


# In[ ]:


train_X.shape, test_X.shape


# In[ ]:


model.compile(optimizer = keras.optimizers.Adam(.001), loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']) #Our y_train is numerical, so instead of categorical_crossentropy
                                      #we use sparse_categorical_cross_entropy, and a learning rate 0.001


# In[ ]:


myDataGen = generator.flow(train_X,train_y, batch_size = 100) #We get data augmented images


# In[ ]:


test_X = test_X/255. #Also normalize test data, since we did it with train data


# In[ ]:


hist  = model.fit_generator(myDataGen, steps_per_epoch = 350, epochs = 40, validation_data = (test_X, test_y))


# In[ ]:


hist.history.keys()


# In[ ]:


val_loss = np.array(hist.history['val_loss'])
val_acc = np.array(hist.history['val_acc'])
train_loss = np.array(hist.history['loss'])
train_acc = np.array(hist.history['acc'])
epochs = len(train_acc)


# In[ ]:


accuracies = pd.DataFrame(train_acc, columns = ['train_acc'])
accuracies['val_acc'] = pd.DataFrame(val_acc)
losses = pd.DataFrame(train_loss, columns = ['train_loss'])
losses['val_loss'] = pd.DataFrame(val_loss)
losses['epochs'] = pd.DataFrame(list(range(epochs)))
accuracies['epochs'] = pd.DataFrame(list(range(epochs)))


# In[ ]:


accuracies.shape


# In[ ]:



plt.scatter(x = 'epochs', y = 'train_acc', data = accuracies, marker = 'x', label = 'train_acc')
plt.scatter(x = 'epochs', y = 'val_acc', s = 5,data = accuracies, label = 'val_acc', color = 'r')
plt.plot(accuracies.iloc[:,0:2])
plt.legend()
plt.show()


# In[ ]:



plt.scatter(x = 'epochs', y = 'train_loss', data = losses, marker = 'x', label = 'train_loss')
plt.scatter(x = 'epochs', y = 'val_loss', data = losses, s = 5,label = 'val_loss', color = 'r')
plt.plot(losses.iloc[:,0:2])
plt.legend()
plt.show()


# In[ ]:


test = np.array(test)/255. #Need also to normalize data that we need to predict
test = test.reshape(-1,28,28,1)


# In[ ]:


#Combine train and validation set for final model training, from scratch
totalData_X = np.concatenate((train_X, test_X))
totalData_y = np.concatenate((train_y, test_y))
myDataGen = generator.flow(totalData_X,totalData_y, batch_size = 100)


# In[ ]:


#Redefining model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), input_shape = (28,28,1,), activation = 'relu', name = 'Conv2D1'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', name = 'Conv2D2'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(128, (2,2),activation = 'relu', name = 'Conv2D3'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation = 'softmax'))


# In[ ]:


model.compile(optimizer = keras.optimizers.Adam(.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


hist  = model.fit_generator(myDataGen, steps_per_epoch = 450, epochs = 40)


# In[ ]:


preds = model.predict(test)


# In[ ]:


predict = pd.DataFrame([preds[x].argmax() for x in range(len(preds))], columns = ['Label'])
predict.head(2)


# In[ ]:


sample = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


predict['ImageId'] = sample['ImageId']
predict = predict[['ImageId','Label']]


# In[ ]:


predict.head()


# In[ ]:


predict.to_csv('predictionsMy.csv', index = False)

