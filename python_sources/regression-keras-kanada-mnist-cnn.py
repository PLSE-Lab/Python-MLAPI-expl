#!/usr/bin/env python
# coding: utf-8

# Forked from this nice kernel https://www.kaggle.com/shahules/indian-way-to-learn-cnn

# In[ ]:


import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ReduceLROnPlateau


# ### Loading data

# In[ ]:


train=pd.read_csv('../input/Kannada-MNIST/train.csv')
test=pd.read_csv('../input/Kannada-MNIST/test.csv')
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')


# Before jumping to all complex stuff about Convolutions and all,we will simply understand our data.We will learn and gain basic understanding about this data.

# In[ ]:


print('The Train  dataset has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
print('The Test  dataset has {} rows and {} columns'.format(test.shape[0],test.shape[1]))


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)
test=test.drop('id',axis=1)


# ### Checking Target class distribution..
# 

# In[ ]:


y=train.label.value_counts()
sns.barplot(y.index,y)


# Now we can see that all of the classes has equal distribution.There are 6000 examples of each numbers in kannada in the the training dataset.Cool !

# ## Data preparation <a id='2'></a>

# In[ ]:


X_train=train.drop('label',axis=1)
Y_train=train.label


# ### Normalize Pixel Values
# 
# For most image data, the pixel values are integers with values between 0 and 255.
# 
# Neural networks process inputs using small weight values, and inputs with large integer values can disrupt or slow down the learning process. As such it is good practice to normalize the pixel values so that each pixel value has a value between 0 and 1.
# 
# It is valid for images to have pixel values in the range 0-1 and images can be viewed normally.
# 
# This can be achieved by dividing all pixels values by the largest pixel value; that is 255. This is performed across all channels, regardless of the actual range of pixel values that are present in the image.

# In[ ]:


X_train=X_train/255
test=test/255


# ### Reshape

# In[ ]:


X_train=X_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)


# In[ ]:


print('The shape of train set now is',X_train.shape)
print('The shape of test set now is',test.shape)


# All Set,We have our data reshape into 60000 examples of height 28 and width 28 and 1 channel.

# ### Splitting train and test

# Now we will split out training data into train and validation data.15percent of the training data will be used for validation purpose.

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=42,test_size=0.15)


# In[ ]:


plt.imshow(X_train[0][:,:,0])


# It's Nine in Kannada
# 

# ### More data !

# In order to avoid overfitting problem, we need to expand artificially our handwritten digit dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations occuring when someone is writing a digit.
# 
# For example, the number is not centered The scale is not the same (some who write with big/small numbers) The image is rotated...
# 
# Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more.
# 
# By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.
# 
# 

# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)


# For the data augmentation, i choosed to :
# 
#    - Randomly rotate some training images by 10 degrees
#    - Randomly Zoom by 10% some training images
#    - Randomly shift images horizontally by 10% of the width
#    - Randomly shift images vertically by 10% of the height
# 
# I did not apply a vertical_flip nor horizontal_flip since it could have lead to misclassify symetrical numbers such as 6 and 9.

# ## Modelling <a id='4' ></a>

# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization(momentum=.15))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization(momentum=0.15))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization(momentum=.15))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(1))


# In[ ]:


model.summary()


# In[ ]:


optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)


# In[ ]:


import keras.backend as K
def Acc(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred=K.round(y_pred)
    return K.mean(K.cast(K.equal(y_true,y_pred),y_pred.dtype))

model.compile(optimizer=optimizer,loss='mae',metrics=['accuracy',Acc])


# ### Learning rate reduction

# In[ ]:


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# ### Fitting our model <a id='5'></a>

# In[ ]:


epochs=40 
batch_size=64


# In[ ]:


# Fit the model
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_test,y_test),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# ## Evaluating our approach <a id='6'></a>

# In[ ]:


fig,ax=plt.subplots(2,1)
fig.set
x=range(1,1+epochs)
ax[0].plot(x,history.history['loss'],color='red')
ax[0].plot(x,history.history['val_loss'],color='blue')

ax[1].plot(x,history.history['accuracy'],color='red')
ax[1].plot(x,history.history['val_accuracy'],color='blue')
ax[0].legend(['trainng loss','validation loss'])
ax[1].legend(['trainng acc','validation acc'])
plt.xlabel('Number of epochs')
plt.ylabel('accuracy')


# We have plotted the performance of our model.We can see the number of epochs in the X axis and change in model performance in Y axis.

# In[ ]:


y_pre_test=model.predict(X_test)
y_pre_test=np.round(y_pre_test)
y_pre_test[y_pre_test>9]=9
y_pre_test[y_pre_test<0]=0
y_pre_test.shape


# In[ ]:


print("Correct predictions",np.sum((y_pre_test.flat==y_test).astype(int)))
print("Incorrect predictions",np.sum((y_pre_test.flat!=y_test).astype(int)))


# ## Making a Submission <a id='7'></a>
# 

# In[ ]:


test=pd.read_csv('../input/Kannada-MNIST/test.csv')


# In[ ]:


test_id=test.id
test=test.drop('id',axis=1)
test=test/255
test=test.values.reshape(-1,28,28,1)


# In[ ]:


test.shape


# We will make our prediction using our CNN model.

# In[ ]:


y_pre=model.predict(test)     ##making prediction
y_pre=np.round(y_pre).astype(int) ##changing the prediction intro labels
y_pre[y_pre>9]=9
y_pre[y_pre<0]=0


# In[ ]:


sample_sub['label']=y_pre.flat
sample_sub.to_csv('submission.csv',index=False)


# In[ ]:


sample_sub.head()


# Things to try:
# 1. Different Thresholds
# 2. Other loss fuctions
