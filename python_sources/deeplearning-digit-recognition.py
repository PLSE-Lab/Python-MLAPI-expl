#!/usr/bin/env python
# coding: utf-8

# ## Below are the steps that we will follow to gain heighr accuracy.
# 1. Load all require modules
# 2. Import datasets
# 3. Analyse and visualise some given digits from training set
# 4. Split Data set into
# * training set
# * Validation set
# 5.  Make our CNN Model
# 6. Plot Loss and accuracy Graph
# 7. test the test data
# 8. Summit the model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
warnings.filterwarnings('ignore') # Ignore comman warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ## Check Directory whare data is located:
# Here we can see that Data is located in "digit-recognizer"

# In[ ]:


print(os.listdir("../input/"))


# ## Set some configuration which can be used in algorithm;
# It is a good practice to make setting dictionary for all constants which are used multiple times

# In[ ]:


settings = {
    'EPOCHS':10,
    'BATCH_SIZE' : 30
}


# 

# ## Import data
# Import given data in train and test set

# In[ ]:


train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")


# ## Check Dataset

# In[ ]:


train_data.columns


# In[ ]:


train_data.shape


# ## Display some Images:
# We are displaying some images from traing set
# 

# In[ ]:


def show_image(train_image, label, index):
    image_shaped = train_image.values.reshape(28,28)
    plt.subplot(4, 5, index+1)
    plt.imshow(image_shaped, cmap=plt.cm.gray)
    plt.title(label)


plt.figure(figsize=(18, 8))
sample_image = train_data.sample(20).reset_index(drop=True)
print(len(sample_image))
label = sample_image['label']
image_pixel = sample_image.drop('label', axis = 1)
for index, row in sample_image.iterrows():
    label = row['label']
    image_pixels = row.drop('label')
    show_image(image_pixels, label, index)
plt.tight_layout()


# ## Build Our Convolutional Network:
# The fundamental difference between a densely connected layer and a convolutional layer is this: Dense layers learn global patterns in their input feature space whereas convolutional layers learn local patterns.
# Note, a convent takes as input tensor of shape(image_height, image_width,image_channel),so we will configure the convent to process inputs of size(28,28,1) and pass it in first layer which is configured in 32 output_depth,and (3,3) hieght and width windows.example:**Conv2D**(output_depth,(window_height,window_width)).
# MaxPooling: Maxpooling consists of extracting windows from the input feature map and outputting the max value of each channel.we We can see in model.summary that the size of the feature map is halved after every maxpooling.
# **Flatten** Platten layer convent tensor into a vector that can be fed into a fully connected neural network classifier.
# **Softmax**: In the last layer we use softmax, because softmax is used for multiclass
# 

# In[ ]:


from keras import layers, models

def prepareModel():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(10, activation = 'softmax'))
    
    return model


# ## Model Summary

# In[ ]:


model = prepareModel()
model.summary()


# ## Data Preprocessing
# split data into train and test set. we are alos deviding data by 255 in order to make all pixel in same range

# In[ ]:


from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
Xtrain = train_data.drop(columns=['label']).values.reshape(train_data.shape[0],28,28,1).astype('float')/255
Ytrain = to_categorical(train_data['label'])

x_validation = Xtrain[:1000]
x = Xtrain[1000:]
y_validation = Ytrain[:1000]
y = Ytrain[1000:]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
x_validation.shape


# In[ ]:


model.compile(
optimizer = 'rmsprop',
loss = 'categorical_crossentropy',
metrics = ['accuracy']
)


# In[ ]:


trainedModel = model.fit(
    x_train,y_train,
    epochs = settings['EPOCHS'],
    batch_size = settings['BATCH_SIZE'],
    validation_data = (x_validation,y_validation )
)


# In[ ]:


trainedModel.history


# ## Now Plot loss and accuracy Graph: Evaluate

# In[ ]:


test_loss,test_acc = model.evaluate(x_test, y_test)


# In[ ]:


print("Test Loss:{loss},Accuracy :{acc}".format(loss = test_loss,acc = test_acc))


# # Amazing Accuracy:Hurrrrrrrrrrrrrraaaaaaaaaaaaaaaaaaaaaaaah

# In[ ]:


loss = trainedModel.history['loss']
val_loss = trainedModel.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'g', label = 'Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


acc = trainedModel.history['accuracy']
val_acc = trainedModel.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'g', label = 'Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# ## Set Data for Testing pupose

# In[ ]:


test_digit_data = test_data.values.reshape(test_data.shape[0],28,28,1).astype("float32") / 255
predictions = model.predict(test_digit_data)
results = np.argmax(predictions, axis = 1) 


# # Set how is our prediction

# In[ ]:


plt.figure(figsize=(10, 8))
sample_test = test_data.head(10)
for index, image_pixels in sample_test.iterrows():
    label = results[index]
    show_image(image_pixels, label, index)
plt.tight_layout()


# ## Now Submit our Model

# In[ ]:


submissions = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
submissions['Label'] = results
submissions.to_csv('submission.csv', index = False)

