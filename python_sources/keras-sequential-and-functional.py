#!/usr/bin/env python
# coding: utf-8

# # Classifying the finger images
# 
# Goal: Our goal is to classify the unknown images into clean house or messy house label. For this purpose we will use Keras Sequential model and Functional API , with Categorical Crossentropy and softmax activation as we have 12 target categories.
# 
# 1. [Reading the data](#Reading-the-data)
# 2. [Verifying the images](#Verifying-the-images)
# 3. [Creating train , test and validation set](#Creating-train-,-test-and-validation-set)
# 4. [Creating the Sequential model](#Creating-the-Sequential-model)
# 5. [Training the model](#Training-the-model)
# 6. [Storing the results](#Storing-the-results)
# 7. [Analysing the results](#Analysing-the-results)

# # Reading the data

# Importing all the libraries required first

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras.layers.merge import concatenate

import os
import cv2
from sklearn import preprocessing
from pathlib import Path


# Retrieving the path is relatively easy. The only challenge is to retrieve the label. The label is stored in the image name. Example: The following is a path:
# 
# *../input/fingers/train/00048bba-979b-4f84-b833-5bbbb082b582_0L.png*
# 
# The last two characters before .png are labels. Here 0L is a label which indicates 0 fingers from left hand. Thus, it requires to step to retrieve 0L.
# 
# 1. Split the image name with '_' and take the second half (0L.png)
# 2. Split the result obtained in step 1 with '.' and take the first half
# 

# In[ ]:



train_path = []
label_train = []

path_train = "../input/fingers/train/"

for filename in os.listdir(path_train):
    
    train_path.append(path_train+filename)
    whole_label = filename.split('_')[1]
    useful_label = whole_label.split('.')[0]
    label_train.append(useful_label)

print("Number of train images: ", len(train_path))
print("First 6 labels: ", label_train[:6])


# In[ ]:


test_path = []
label_test = []

path_train = "../input/fingers/test/"

for filename in os.listdir(path_train):
    
    test_path.append(path_train+filename)
    whole_label = filename.split('_')[1]
    useful_label = whole_label.split('.')[0]
    label_test.append(useful_label)

print("Number of test images: ", len(test_path))
print("First 6 labels: ", label_train[:6])


# In[ ]:


train_path[0]


# # Verifying the images #

# In[ ]:


# checking train path
image = cv2.imread(train_path[0]) 

# the first image bleongs to clean directory under train
plt.imshow(image)
plt.title(label_train[0], fontsize = 20)
plt.axis('off')
plt.show()


# In[ ]:


# checking train path
image = cv2.imread(test_path[95]) 

# the first image bleongs to clean directory under train
plt.imshow(image)
plt.title(label_test[95], fontsize = 20)
plt.axis('off')
plt.show()


# # Creating train , test and validation set #

# Keras model works on numpy arrays and thus, images are required to convert to numpy. Same goes for the labels

# In[ ]:


X_train = []
X_test = []

# reading images for train data
for path in train_path:
    
    image = cv2.imread(path)        
    image =  cv2.resize(image, (50,50))    
    X_train.append(image)
    
# reading images for test data
for path in test_path:
    
    image = cv2.imread(path)        
    image =  cv2.resize(image, (50,50))    
    X_test.append(image)

X_test = np.array(X_test)
X_train = np.array(X_train)


# In[ ]:


print("Shape of X_train: ", X_train.shape)
print("Shape of X_test: ", X_test.shape)


# Divding each pixel value in range 0 - 255 with 255 to get numbers in range 0 - 1

# In[ ]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


# Label encoding all the labels and then converting to categorical values

# In[ ]:


lable_encoder = preprocessing.LabelEncoder()
y_train_temp = lable_encoder.fit_transform(label_train)
y_test_temp = lable_encoder.fit_transform(label_test)

print("Integer encoded values for train: ", y_train_temp)
print("Integer encoded values for test: ", y_test_temp)


# In[ ]:


y_train = keras.utils.to_categorical(y_train_temp, 12)
y_test = keras.utils.to_categorical(y_test_temp, 12)

print("Categorical values for y_train:", y_train)
print("Categorical values for y_test:", y_test)


# Inputs for functional model keras

# In[ ]:


X_train_A , X_train_B = X_train[:9000], X_train[-9000:]
y_train_A , y_train_B = y_train[:9000], y_train[-9000:]


# In[ ]:


print("Shape of X_train_A: ", X_train_A.shape, ", shape of X_train_B: ", X_train_B.shape)


# In[ ]:


# uncomment to check if they are different or not
# X_train_A == X_train_B


# # Creating the Sequential model #

# In[ ]:


model_seq = Sequential()

# input shape for first layer is 50,50,3 -> 50 * 50 pixles and 3 channels
model_seq.add(Conv2D(32, (3, 3), padding='same', input_shape=(50, 50, 3), activation="relu"))
model_seq.add(Conv2D(32, (3, 3), activation="relu"))

# maxpooling will take highest value from a filter of 2*2 shape
model_seq.add(MaxPooling2D(pool_size=(2, 2)))

# it will prevent overfitting
model_seq.add(Dropout(0.25))

model_seq.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model_seq.add(Conv2D(64, (3, 3), activation="relu"))
model_seq.add(MaxPooling2D(pool_size=(2, 2)))
model_seq.add(Dropout(0.25))

model_seq.add(Flatten())
model_seq.add(Dense(512, activation="relu"))
model_seq.add(Dropout(0.5))

# last layer predicts 12 labels
model_seq.add(Dense(12, activation="softmax"))

# Compile the model
model_seq.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

model_seq.summary()


# In[ ]:


keras.utils.plot_model(model_seq, "keras_seq_model.png", show_shapes=True)


# In[ ]:


# two inputs
input_1 = keras.Input(shape=(50, 50, 3))
input_2 = keras.Input(shape=(50, 50, 3))


# for input 1
conv_1_1 = Conv2D(32, (3, 3), padding='same', activation="relu")(input_1)
conv_1_2 = Conv2D(32, (3, 3), activation="relu")(conv_1_1)
max_1_1 = MaxPooling2D(pool_size=(2, 2))(conv_1_2)
drop_1_1 = Dropout(0.25)(max_1_1)
conv_1_3 = Conv2D(64, (3, 3), padding='same', activation="relu")(drop_1_1)
conv_1_4 = Conv2D(64, (3, 3), activation="relu")(conv_1_3)
max_1_2 = MaxPooling2D(pool_size=(2, 2))(conv_1_4)
drop_1_2 = Dropout(0.25)(max_1_2)
flat_1 = Flatten()(drop_1_2)

# for input 2
conv_2_1 = Conv2D(32, (3, 3), padding='same', activation="relu")(input_2)
conv_2_2 = Conv2D(32, (3, 3), activation="relu")(conv_2_1)
max_2_1 = MaxPooling2D(pool_size=(2, 2))(conv_2_2)
drop_2_1 = Dropout(0.3)(max_2_1)
conv_2_3 = Conv2D(64, (3, 3), padding='same', activation="relu")(drop_2_1)
conv_2_4 = Conv2D(64, (3, 3), activation="relu")(conv_2_3)
max_2_2 = MaxPooling2D(pool_size=(2, 2))(conv_2_4)
drop_2_2 = Dropout(0.3)(max_2_2)
flat_2 = Flatten()(drop_2_2)

# merge both falt layers
merge = concatenate([flat_1, flat_2])

dense = Dense(512, activation="relu")(merge)
drop = Dropout(0.5)(dense)

output = Dense(12, activation="softmax")(drop)

# creating model
model_fun = Model(inputs = [input_1,input_2], outputs = output, name="functional_model")

# compile the model
model_fun.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

model_fun.summary()


# In[ ]:


keras.utils.plot_model(model_fun, "keras_func_model.png", show_shapes=True)


# # Training the model

# In[ ]:


# training the model
history_seq = model_seq.fit(
    X_train,
    y_train,
    batch_size=50,
    epochs=30,
    validation_split=0.2,
    shuffle=True
)


# In[ ]:


history_fun = model_fun.fit(
    [X_train[:9000], X_train[-9000:]],
    y_train,
    batch_size=50,
    epochs=30,
    validation_split=0.2,
    shuffle=True
)


# # Storing the results #

# Storing the results for future use

# In[ ]:


# for sequential model
# saving the structure of the model
model_structure = model_seq.to_json()
f = Path("model_seq_structure.json")
f.write_text(model_structure)

# saving the neural network's trained weights
model_seq.save_weights("model_seq_weights.h5")



# for functional model
# saving the structure of the model
model_structure = model_fun.to_json()
f = Path("model_fun_structure.json")
f.write_text(model_structure)

# saving the neural network's trained weights
model_fun.save_weights("model_fun_weights.h5")


# # Analysing the results #

# Displaying the model accuracy and loss using graphs

# In[ ]:


# displaying the model accuracy

fig, axs = plt.subplots(1, 2 , figsize = [10,5])

plt.suptitle("For Sequential Model", fontsize = 20)

axs[0].plot(history_seq.history['accuracy'], label='train', color="red")
axs[0].plot(history_seq.history['val_accuracy'], label='validation', color="blue")
axs[0].set_title('Model accuracy')
axs[0].legend(loc='upper left')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')

axs[1].plot(history_seq.history['loss'], label='train', color="red")
axs[1].plot(history_seq.history['val_loss'], label='validation', color="blue")
axs[1].set_title('Model loss')
axs[1].legend(loc='upper left')
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('loss')

plt.show()


# In[ ]:


fig, axs = plt.subplots(1, 2 , figsize = [10,5])

plt.suptitle("For functional api model", fontsize = 20)

axs[0].plot(history_fun.history['accuracy'], label='train', color="red")
axs[0].plot(history_fun.history['val_accuracy'], label='validation', color="blue")
axs[0].set_title('Model accuracy')
axs[0].legend(loc='upper left')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')

axs[1].plot(history_fun.history['loss'], label='train', color="red")
axs[1].plot(history_fun.history['val_loss'], label='validation', color="blue")
axs[1].set_title('Model loss')
axs[1].legend(loc='upper left')
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('loss')

plt.show()


# Calculating the accuracy achieved

# In[ ]:


print("For sequential model: ")
score, accuracy = model_seq.evaluate(X_test, y_test)
print('Test score achieved: ', score)
print('Test accuracy achieved: ', accuracy)


# In[ ]:


print("For functional API model: ")
score, accuracy = model_fun.evaluate([X_test[:9000], X_test[-9000:]], y_test)
print('Test score achieved: ', score)
print('Test accuracy achieved: ', accuracy)


# ## Analysing with sequential model first, then we will use functional model ##

# In[ ]:


pred = model_seq.predict(X_test)
pred[:10]


# In[ ]:


y_test[:10]


# At last displaying the model prediction for first 10 images of test set

# In[ ]:


fig, axs= plt.subplots(2,5, figsize=[24,12])


count=0
for i in range(2):    
    for j in range(5):  
        
        img = cv2.imread(test_path[count])
        
        results = np.argsort(pred[count])[::-1]
      
        labels = lable_encoder.inverse_transform(results)
        
        axs[i][j].imshow(img)
        axs[i][j].set_title(labels[0], fontsize = 20)
        axs[i][j].axis('off')

        count+=1
        
plt.suptitle("Sequential Model : all predictions are shown in title", fontsize = 24)        
plt.show()


# ### Let's take images from range 20-30 for functional model api anaylsis

# In[ ]:


pred2 = model_fun.predict([X_test[:9000], X_test[-9000:]])
pred2[20:30]


# In[ ]:


fig, axs= plt.subplots(2,5, figsize=[24,12])


count=20
for i in range(2):    
    for j in range(5):  
        
        img = cv2.imread(test_path[count])
        
        results = np.argsort(pred2[count])[::-1]
      
        labels = lable_encoder.inverse_transform(results)
        
        axs[i][j].imshow(img)
        axs[i][j].set_title(labels[0], fontsize = 20)
        axs[i][j].axis('off')

        count+=1
        
plt.suptitle("Functional Model : all predictions are shown in title", fontsize = 24)        
plt.show()


# # Visualising output of each layer in sequential model and functional api model
# 
# 

# Source code for the following visualization is taken from this amazing documentation [Visualize layer outputs of your Keras classifier with Keract](https://www.machinecurve.com/index.php/2019/12/02/visualize-layer-outputs-of-your-keras-classifier-with-keract/)

# In[ ]:


get_ipython().system('pip install keract')


# ## For sequential model

# In[ ]:


from keract import get_activations, display_heatmaps
keract_inputs = X_test[:1]
keract_targets = y_test[:1]
activations = get_activations(model_seq, keract_inputs)
display_heatmaps(activations, keract_inputs, save=False)


# # Thank you #

# # References: #
# 
# 1. [Keras Guide train and evaluate](https://www.tensorflow.org/guide/keras/train_and_evaluate)
# 2. [Machine Learning Mastery](https://machinelearningmastery.com/keras-functional-api-deep-learning/)
# 3. [Keras API Models](https://keras.io/api/models/)
# 4. [Keras multiple inputs and mixed data](https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/)
# 5. [Visualize layer outputs of your Keras classifier with Keract](https://www.machinecurve.com/index.php/2019/12/02/visualize-layer-outputs-of-your-keras-classifier-with-keract/)

# In[ ]:




