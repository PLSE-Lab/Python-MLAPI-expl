#!/usr/bin/env python
# coding: utf-8

# # MNIST: Simple CNN with Keras (top 9%)
# ### A top 9% submission with an accuracy of 0.99685
# 
# * **1. Import**
# * **2. Data preparation**
#     * 2.1 Load data
#     * 2.2 Inspect data
#     * 2.3 Visualize data
#     * 2.4 Normalise and reshape data
#     * 2.5 Split training and valdiation set
# * **3. Convolutional Neural Network**
#     * 3.1 Define the model
#     * 3.2 Data augmentation
#     * 3.3 Fit model
# * **4. Evaluate model**
#     * 4.1 Confusion matrix
#     * 4.2 Training and validation curves
# * **5. Prediction and submission**
#     * 5.1 Prediction 
#     * 5.2 Submission
#     * 5.3 Save model
# * **6. References**
# 
# 

# ## 1. Import
# Import necessary libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #for plotting
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


# ## 2. Data preparation
# ### 2.1 Load data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ### 2.2 Inspect data

# In[ ]:


print(train.shape)
print(test.shape)
train.head()


# ### 2.3 Visualize data
# Number of observations per label

# In[ ]:


print(Counter(train['label']))
sns.countplot(train['label'])


# Properly define pixels and labels

# In[ ]:


x_train = (train.iloc[:,1:].values).astype('float32')
y_train = train.iloc[:,0].values.astype('int32')

x_test = test.values.astype('float32')


# Observe some digit examples

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12,6))
x, y = 10, 4
for i in range(40):
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i].reshape((28,28)), interpolation='nearest')
plt.show()


# See an example of how pixel values are defined on the gray scale

# In[ ]:


def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')

x_train=(train.iloc[:,1:].values).astype('int32')
fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
visualize_input(x_train[1].reshape(28,28), ax)
x_train=(train.iloc[:,1:].values).astype('float32')


# ### 2.4 Normalise and reshape data

# In general, neural nets perform better if we normalise data

# In[ ]:


x_train = x_train/255.0
x_test = x_test/255.0
y_train


# Reshape data to match Keras' expectations

# In[ ]:


print('x_train shape:', x_train.shape)


# In[ ]:


X_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# ### 2.5 Split training and validation set

# In[ ]:


num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)

X_train, X_val, Y_train, Y_val= train_test_split(
    X_train, y_train, test_size = 0.1, random_state = 42)


# ## 3. Convolutional Neural Network
# ### 3.1 Define the model
# There is no single right way to define your CNN - number of epochs, batch size and other hyperparameters vary from problem to problem, though ReLU seems to be chosen as the standard activation function now, as the ReLU function avoids the vanishing gradients problem in neural networks. Furthermore, the Adam optimizer seems to be the most widely used optimizer at the moment.

# In[ ]:


batch_size = 64
epochs = 20
input_shape = (28, 28, 1)

model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', 
                kernel_initializer = 'he_normal', input_shape = input_shape))
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', 
                kernel_initializer = 'he_normal'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation='relu',
                 padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu',
                 padding='same',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',
                 padding='same',kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation = 'softmax'))
          
model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer = keras.optimizers.Adam(),
                         metrics = ['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                           patience = 3,
                                           verbose = 1,
                                           factor = 0.5,
                                           min_lr = 0.0001)


# ### 3.2 Data augmentation
# 
# We augment data e.g. by randomy rotating the pictures. This expands our dataset and adds noise which helps us avoid overfitting.

# In[ ]:


datagen = ImageDataGenerator(featurewise_center=False, # set input mean to 0 over the dataset
                            samplewise_center = False, # set each sample mean to 0
                            featurewise_std_normalization = False, # divide inputs by std of the dataset
                            samplewise_std_normalization = False, # divide each input by its std
                            zca_whitening = False, # apply ZCA whitening
                            rotation_range = 15, # randomly rotate images in the range (degrees, 0 to 180)
                            zoom_range = 0.1, # Randomly zoom image 
                            width_shift_range = 0.1, # randomly shift images horizontally (fraction of total width)
                            height_shift_range = 0.1, # randomly shift images vertically (fraction of total height)
                            horizontal_flip = False, # randomly flip images
                            vertical_flip = False) # randomly flip images - we do not want this as it e.g. messes up the digits 6 and 9


# In[ ]:


model.summary()


# ### 3.3 Fit model

# In[ ]:


datagen.fit(X_train)
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size),
                       epochs = epochs,
                       validation_data = (X_val, Y_val),
                       verbose = 1,
                       steps_per_epoch = X_train.shape[0] // batch_size,
                       callbacks = [learning_rate_reduction],)


# In[ ]:


final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)
print("validation loss: {0:.6f}, validation accuracy: {1:.6f}".format(final_loss, final_acc))


# ## 4. Evaluate model
# ### 4.1 Confusion matrix
# Code taken from the sklearn website: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

# In[ ]:


def plot_confusion_matrix(cm, classes,
                         normalize = False,
                         title = 'Confusion matrix',
                         cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(num_classes))


# ### 4.2 Training and validation curves

# In[ ]:


print(h.history.keys())


# In[ ]:


accuracy = h.history['acc']
val_accuracy = h.history['val_acc']
loss = h.history['loss']
val_loss = h.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label = 'training accuracy')
plt.plot(epochs, val_accuracy, 'b', label = 'validation accuracy')

plt.title('training and validation accuracy')

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'training loss')
plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.title('training and validation loss')


# ## 5. Prediction and submission
# ### 5.1 Prediction

# In[ ]:


predicted_classes = model.predict_classes(X_test)


# ### 5.2 Submission

# In[ ]:


submissions = pd.DataFrame({"ImageId": list(range(1, len(predicted_classes)+1)),
                           "Label": predicted_classes})
submissions.to_csv("mnistSubmission.csv", index = False, header = True)


# ### 5.3 Save model

# In[ ]:


model.save('my_model_1.h5')

json_string = model.to_json()


# ## 6. References
#  - A great deal of inspiration taken from: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
#  - Sklearn confusion matrix: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
