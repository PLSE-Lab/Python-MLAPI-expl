#!/usr/bin/env python
# coding: utf-8

# <img src="https://www.katacoda.com/basiafusinska/courses/deep-learning-with-tensorflow/mnist-dataset/assets/MNIST.png" />

# # MNIST Dataset

# ### Importing data and libraries

# In[ ]:


import os


# In[ ]:


os.listdir()


# In[ ]:


os.chdir('/kaggle/input')


# In[ ]:


os.listdir()


# In[ ]:


import pandas as pd


# In[ ]:


traindatafile = pd.read_csv('train.csv')
testdatafile = pd.read_csv('test.csv')


# In[ ]:


traindatafile.head()


# In[ ]:


y = traindatafile.pop('label')


# In[ ]:


import numpy as np # linear algebra
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ### Data Wrangling

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(traindatafile, y, test_size=0.1, random_state=42)


# In[ ]:


traindatafile.shape


# In[ ]:


# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
x_train = x_train.values.reshape(-1,28,28,1)
x_val = x_val.values.reshape(-1,28,28,1)
x_test= testdatafile.values.reshape(-1,28,28,1)


# In[ ]:


print(x_train.shape)
print(x_val.shape)


# In[ ]:


x_train[0]


# In[ ]:


x_train = x_train.astype("float32")/255.
x_val = x_val.astype("float32")/255.
x_test = x_test.astype("float32")/255.


# In[ ]:


x_train[0]


# In[ ]:


print(x_train.shape)
x_val.shape


# In[ ]:


x_test.shape


# #### Importing Keras libraries

# In[ ]:


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard


# In[ ]:


y_train = to_categorical(y_train) 
y_val = to_categorical(y_val) 


# In[ ]:


print(y_train.shape)
y_val.shape


# Train and test images (28px x 28px) has been stock into pandas.Dataframe as 1D vectors of 784 values. We reshape all data to 28x28x1 3D matrices.
# 
# Keras requires an extra dimension in the end which correspond to channels. MNIST images are gray scaled so it use only one channel. For RGB images, there is 3 channels, we would have reshaped 784px vectors to 28x28x3 3D matrices.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

g = plt.imshow(x_train[0][:,:,0])


# ### Model CNN 

# <img src= "https://iq.opengenus.org/content/images/2018/11/cnn.png" />

# In[ ]:


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# In[ ]:


model.summary()


# this is to for changing images properties 

# In[ ]:


datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                                 featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
                               zca_whitening=False,  # apply ZCA whitening
        rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
                              zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                               height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
                               vertical_flip=False)  # randomly flip images)


# <img src="https://i.ytimg.com/vi/tRsSi_sqXjI/maxresdefault.jpg" />

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"]) #1e-4, means the 1 is four digits the other way, so 1e-4 = 0.0001.


# In[ ]:


learning_rate_min = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
#tensor= TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq=1000)


# ## Fitting the data in model

# In[ ]:


hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                           steps_per_epoch=500,
                           epochs=20, #Increase this when not on Kaggle kernel
                           verbose=2,  #1 for ETA, 0 for silent
                           validation_data=(x_val[:400,:], y_val[:400,:]), #For speed
                           callbacks=[learning_rate_min]) 


# In[ ]:


final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))


# In[ ]:


plt.plot(hist.history['loss'], color='r',label='Training loss')
plt.plot(hist.history['acc'], color='b',label='Training accuracy')
plt.title('Training loss and Training accuracy')
plt.show()

plt.plot(hist.history['val_loss'], color='r', label='Validation loss')
plt.plot(hist.history['val_acc'], color='b', label='Validation accuracy')
plt.title('validation loss and validation accuracy')
plt.show()


# Confusion matrix example

# <img src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/03/confusion-matrix-mercari.png" />

# In[ ]:


y_predicted = model.predict(x_val)
y_predictedint = np.argmax(y_predicted, axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_predictedint)
print(cm)


# In[ ]:


y_hat = model.predict(x_test ,batch_size=64)


# In[ ]:


print(y_hat)


# In[ ]:


y_pred = np.argmax(y_hat,axis=1)


# In[ ]:


y_pred


# In[ ]:


print(y_pred)


# In[ ]:


y_pred.shape


# #####  another way to submit 
# with open("submission.csv", 'w') as f :
#     f.write('ImageId,Label\n')
#     for i in range(len(y_pred)) :
#         f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))

# submissions=pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),
#                          "Label": y_pred})
# submissions.to_csv("submission.csv", index=False, header=True)
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




