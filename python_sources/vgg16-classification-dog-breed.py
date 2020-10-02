#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# I found this dataset rather cute so I tried to make a deep learning model to predict the breed of the dog
# 
# I used transfer learning, basically I took an already trained model and added layers to it, and trained it on this dataset.
# 
# The model I chose is the VGG16, trained on the Imagenet dataset.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt

import cv2


# ## Loading the images and preprocessing
# 
# I chose to load all images into memory, but if you don't have enough memory (around 9.3GB) you can use the generator from keras https://keras.io/preprocessing/image/.
# 
# Further we have to preprocess the image to fit the VGG16 model. The model expects a 224 by 224 RGB image.
# 

# In[ ]:


BASEPATH = "../input/images/Images/"

LABELS = set()

paths = []
    
for d in os.listdir(BASEPATH):
    LABELS.add(d)
    paths.append((BASEPATH+d, d))


# In[ ]:


# resizing and converting to RGB
def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224,224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# In[ ]:


X = []
y = []

for path, label in paths:
    for image_path in os.listdir(path):
        image = load_and_preprocess_image(path+"/"+image_path)
        
        X.append(image)
        y.append(label)


# ### Encoding the labels
# 
# Every model I know of, can only work with numbers, therefore we need to create our label (n02096294-Australian_terrier) to an integer array. This is called **one-hot-encoding**

# In[ ]:


encoder = LabelBinarizer()

X = np.array(X)
y = encoder.fit_transform(np.array(y))

print(y[0])


# In[ ]:


print(X.shape)
print(y.shape)
plt.imshow(X[0])


# ## Training and evaluating
# 
# Now comes to most interesting part. Training and evaluating our model
# 
# First step is to create a training and testing dataset
# 
# After this I have 2 approaches.
# - Creating our own model, which I tried
# - Using transfer learning on the VGG16 model

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)


# This is the own model I tried. I have to say I did not have a great succes with it. Maybe it needs more finetuning.

# In[ ]:


model = Sequential()

model.add(Conv2D(64,(3,3),activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation="relu", padding="same"))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation="relu", padding="same"))
model.add(MaxPooling2D())

model.add(Conv2D(128,(3,3),activation="relu", padding="same"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(len(LABELS),activation="softmax"))


# This is the transfer learning model
# 
# I used the VGG16 with the Imagenet weights and did not include the top, so I can add my own top. Being 3 Dense Layers with relu activation, and the last one with softmax and number of neurons is the amount of labels we have.
# 
# I set the last 5 layers to trainable, which are the Dense layers I added myself. I don't want to train any layers I get from the pretrained VGG16 model.

# In[ ]:


base_model=VGG16(weights='imagenet',include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(512,activation='relu')(x)
preds=Dense(len(LABELS),activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:-5]:
    layer.trainable=False
for layer in model.layers[-5:]:
    layer.trainable=True
    
model.compile("adam",loss="categorical_crossentropy",metrics=["accuracy"])

print(model.summary())


# In[ ]:


early_stopping = EarlyStopping(patience=5, verbose=1,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3,verbose=1)


# ### Model training
# 
# I used the Adam optimizer, typically my first choice.
# 
# I trained the model on 50 epochs, with a batch size of 64, I used two callbacks. Early stopping to stop the training of the validation loss does not improve for 5 epochs, and Reduce LR on plateau to reduce the learning rate to 10% if the validation loss doesn't improve for 3 epochs.

# In[ ]:


model.fit(X_train,y_train,batch_size=64,epochs=50,validation_data=(X_test,y_test), callbacks=[early_stopping, reduce_lr])


# ### Evaluation
# 
# Last step should always be to evaluate your model

# In[ ]:


loss, acc = model.evaluate(X_test,y_test,verbose=0)
print(f"loss on the test set is {loss:.2f}")
print(f"accuracy on the test set is {acc:.3f}")


# As we can see the accuracy on the test set is over 50% maybe by tuning the model and the hyperparameters we could increase this.

# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


label_predictions = encoder.inverse_transform(predictions)


# In[ ]:


rows, cols = 5, 3
size = 25

fig,ax=plt.subplots(rows,cols)
fig.set_size_inches(size,size)
for i in range(rows):
    for j in range (cols):
        index = np.random.randint(0,len(X_test))
        ax[i,j].imshow(X_test[index])
        ax[i,j].set_title(f'Predicted: {label_predictions[index]}\n Actually: {encoder.inverse_transform(y_test)[index]}')
        
plt.tight_layout()

