#!/usr/bin/env python
# coding: utf-8

# <h1>Deep Neural Networks and Data Augmentation</h1>
# In this tutorial I will explain how to classify the MNIST dataset using DNNs (Deep Neural Network). The code will be written using python with the help of Keras library which is a high level library and i am going to use TensorFlow as it's backend.
# 
#   Now let's import Keras and some other useful libraries that we are gonna use later and also we will load the data from keras databases for later use.

# In[ ]:


from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist
import numpy as np
np.random.seed(0)

x_train = pd.read_csv('../input/train.csv')
label = x_train['label']
x_train.drop(['label'], inplace = True, axis = 1 )

x_test = pd.read_csv('../input/test.csv')
x_train = x_train.values
y_train = label.values
x_test = x_test.values
x_train ,  y_train = shuffle(x_train, label , random_state=0)


# <h1>Data Exploration</h1>
# Let's explore the data we have as this will give us a hint on the algorithm we will use if we have to choose. Exploring data is also very important because it will tell you which accuracy metric you are going to use, if the data is balanced which means all the classes have fair contribution in the dataset regarding its numbers then we can easily use accuracy, But if the data is skewed then we won't be able to use accurace as it's results will be misleading and we may use F-beta score instead.

# In[ ]:


print(x_train.shape)


# In[ ]:


print("the number of training examples = %i" % x_train.shape[0])
print("the number of classes = %i" % len(np.unique(y_train)))
print("Flattened Image dimentions = %d x %d  " % (x_train.shape[1], 1)  )

#This line will allow us to know the number of occurrences of each specific class in the data
print("The number of occuranc of each class in the dataset = %s " % label.value_counts(), "\n" )


X_train = x_train.reshape(-1, 28, 28).astype('float32')
images_and_labels = list(zip(X_train,  y_train))
for index, (image, label) in enumerate(images_and_labels[:12]):
    plt.subplot(5, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('label: %i' % label)


# In[ ]:


type(x_train)


# From the previous results we can see that the dataset consists of 60000 training example each is an image of dimention 28 * 28. We can see that the number of occurances of each class is almost balanced and based on that it is safe to use accuracy as our metric later.

# <h1>Algorithm Choice</h1>
# In this tutorial we will use the Deep Neural Networks Algorithm. Deep Neural Networks consist of levels. Each level of the  neural network consists of neurons. A neuron in the NN layer outputs discrete values for a classification task so it even fires or doesn't fire.
# ![Deep Neural Network example](https://raw.githubusercontent.com/MoghazyCoder/Machine-Learning-Tutorials/master/assets/Deep.png)
# 
# Neurons use the equation that determines whether or not it will fire. each neuron outputs the result from applying the function a(z) where a() is the activation function and z is the linear function WX + b and passes it to the next layer neurons. One of the mostly used activation function is the Relu function that is because it solves the problem of the exploding gradient, You can read more about that [here](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/).

# <h1>Parameter and Model Selection</h1>
# Now Let's fit the model. We will make a sequential model which is a stack of layers, each layer passes the output to the next layer. we must reshape the input data to make the image a 1d vector instead to be able to pass it to the Deep Neural Network.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout, Conv2D
from keras import regularizers

from keras.utils import np_utils

#reshape the inputs
# I will change the size of the training and testing sets to be able to use ImageDataGenerator wich accepts inputs in the following shape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print(x_train.shape )
print(x_train.shape )


#Makine the outputs 1-hot vector of 10 elements
y_train = np_utils.to_categorical(y_train)

model = Sequential()
# The first layer doesn't have significant importance in the code.
# THe conv layer is used only to get the 3d images from the fit generator in the 2d format and flatten it using flatten layer
# THe layer will not affect the layer since i am only using feature Pooling _ 1*1 convolution with only 1 feature map
model.add(Conv2D(1, kernel_size=1, padding="same",input_shape=(28, 28, 1), activation = 'relu'))
model.add(Flatten())

# model.add(Dense(units=800, activation='relu', input_dim= 784 ,  kernel_regularizer=regularizers.l2(0.001) ) )

model.add(Dense(units=100, activation='relu'  ))
model.add(Dropout(0.1))
model.add(Dense(units=100, activation='relu'  ))
model.add(Dropout(0.1))
model.add(Dense(units=100, activation='relu'  ))
model.add(Dropout(0.1))

#and now the output layer which will have 10 units to
#output a 1-hot vector to detect one of the 10 classes
model.add(Dense(units=10, activation='softmax'))


# Let's configure the learning process and choose the suitable parameters.we will use [catagorical cross entropy as the loss finction](http://neuralnetworksanddeeplearning.com/chap3.html), adam optimizer which  is an efficient gradient descent algorithm that proved to work well and our performance metric will be the accuracy.

# In[ ]:


from keras import optimizers

# optimizer = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])


# # Data Augmentation
# I will use ImageDataGenerator from keras to augment the images. Augmenting the images makes the model more robust and more generalizable when using newly unseen data like the data in the test set of the competition. There are many ways to augment the images like centering the images, normalization, rotation, shifting, and flipping and i will use some of them here .

# In[ ]:


from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
x_train2 = np.array(x_train, copy=True) 
y_train2 = np.array(y_train, copy=True) 

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    fill_mode='nearest',
    validation_split = 0.2
    )


# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)

datagen.fit(x_train)

print(type(x_train))

earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min')

validation_generator = datagen.flow(x_train2, y_train2, batch_size=60, subset='validation')
train_generator = datagen.flow(x_train2, y_train2, batch_size=60, subset='training')


# # fits the model on batches with real-time data augmentation:
history = model.fit_generator(generator=train_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    steps_per_epoch = len(train_generator) / 60,
                    validation_steps = len(validation_generator) / 60,
                    epochs = 300,
                    workers=-1, callbacks = [earlystopping])


# In[ ]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([-1,1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([-1,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# Now it is time to train the Network. We will use an early stopping function to stp the training if the validation loss doesn't change with patience of 50 epochs

# Now we should evaluate the model on the test set

# In[ ]:


res = model.predict(x_test)
res = np.argmax(res,axis = 1)
res = pd.Series(res, name="Label")
submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   res],axis = 1)
submission.to_csv("solution.csv",index=False)
submission.head(10)

