#!/usr/bin/env python
# coding: utf-8

# <h1>A Keras Intro, working through the MNIST dataset</h1>
# 1. Introduction
# 2. Data Preparation  
#     2.1 Loading in Dataset    
#     2.2 Visualize numbers  
#     2.3 Scale data  
#     2.4 Reshape data 
#     2.5 One hot encode labels
# 3. Build Model  
#     3.1 Build Model structure
#     3.2 Define optimizer, callbacks  
#     3.3 Augment Images  
#     3.4 Compile Model  
#     3.5 Train Model
# 4. Evaluate Model
# 5. Making Predictions

# <h2>1. Introduction</h2>

# In this Kernal you will learn how to use Keras to create a Convolutional Neural Network for recognizing digit images. First of I will prepare the data, then I will visualize it and build an Convolutional Neural Network to make predictions on unseen data.

# In[ ]:


import numpy as np # linear algebra libary
import pandas as pd # data processing libary
import matplotlib.pyplot as plt # visualization libary
import os # So we can see if we already saved a model

# Deep Learning Libary
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical


# <h2>2. Data Preparation</h2>

# <h4>2.1 Loading in Dataset</h4>

# In[ ]:


training_dataset = pd.read_csv('../input/train.csv')
testing_dataset = pd.read_csv('../input/test.csv')


# In[ ]:


training_dataset.head()


# In[ ]:


X_train = np.array(training_dataset.drop(['label'], axis=1))
y_train = np.array(training_dataset['label'])
X_test = np.array(testing_dataset)


# <h4>2.2 Visualize numbers</h4>

# In[ ]:


def visualize_digits(data, n, true_labels, predicted_labels=[]):
    fig = plt.figure()
    plt.gray()
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(data[i].reshape(28, 28))
        # disable axis
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if len(predicted_labels)!=0:
            ax.set_title('True: ' + str(true_labels[i]) + ' Predicted: ' + str(np.argmax(predicted_labels[i])))
        else:
            ax.set_title('True: ' + str(true_labels[i]))
    fig.set_size_inches(np.array(fig.get_size_inches()) * n)
    plt.show()


# In[ ]:


visualize_digits(X_train, 10, y_train)


# <h4>2.3 Scale data</h4>

# In[ ]:


X_train = X_train / 255
X_test = X_test / 255


# <h4>2.4 Reshape data</h4>
# We need to reshape our image data so we can use it in an convolutional layer.

# In[ ]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# <h4>2.5 One hot encode labels</h4>

# In[ ]:


y_train = to_categorical(y_train, num_classes=10)


# <h2>3. Build Model</h2>

# <h4>3.1 Build Model structure</h4>

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='softmax'))


# <h4>3.2 Define optimizer, callbacks</h4>

# In[ ]:


optimizer = RMSprop(lr=0.001)
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.3, min_lr=0.00001)


# <h4>3.3 Augment Images</h4>

# In[ ]:


datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1)


# <h4>3.4 Compile Model</h4>

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# <h4>3.5 Train Model</h4>

# In[ ]:


history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=64), epochs=10,
                              verbose=2, steps_per_epoch=X_train.shape[0]//64, 
                              callbacks=[learning_rate_reduction])


# In[ ]:


plt.plot(history.history['loss'])


# In[ ]:


plt.plot(history.history['acc'])


# <h2>4. Evaluate Model</h2>

# In[ ]:


scores = model.evaluate(X_train, y_train)
scores


# In[ ]:


predictions = model.predict(X_train)

visualize_digits(X_train, 10, training_dataset['label'], predictions)


# <h2>5. Making Predictions</h2>

# In[ ]:


predictions = model.predict(X_test)
predictions = [np.argmax(x) for x in predictions]
image_id = range(len(predictions))
solution = pd.DataFrame({'ImageId':image_id, 'Label':predictions})
solution.head()


# <h2>Conclusion</h2>
# That's all from this Kernel we got pretty good results but you could surely get better results by training on more epochs and using a bigger network.
