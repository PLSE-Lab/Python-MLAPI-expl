#!/usr/bin/env python
# coding: utf-8

# # Mango Classification

# ### using EfficientNet B7

# * Website: https://aidea-web.tw/topic/72f6ea6a-9300-445a-bedc-9e9f27d91b1c
# * Forum : https://www.facebook.com/groups/184056976135023/

# ## Dataset : AICUP 2020 Mango sample data
# ![Taiwan%20AICUP%202020%20Mango.JPG](attachment:Taiwan%20AICUP%202020%20Mango.JPG)

# In[ ]:


import os
print(os.listdir('../input/aicup-2020-mango-c1-p1/C1-P1_Train Dev'))


# In[ ]:


print(os.listdir('../input/aicup-2020-mango-c1-p1/C1-P1_Train Dev/C1-P1_Train'))


# In[ ]:


trainPath = '../input/aicup-2020-mango-c1-p1/C1-P1_Train Dev/C1-P1_Train/'
devPath   = '../input/aicup-2020-mango-c1-p1/C1-P1_Train Dev/C1-P1_Dev/'

trainCSV = '../input/aicup-2020-mango-c1-p1/C1-P1_Train Dev/train.csv'
devCSV   = '../input/aicup-2020-mango-c1-p1/C1-P1_Train Dev/dev.csv'


# In[ ]:


import pandas as pd
trainDF = pd.read_csv(trainCSV, header=None)
print(trainDF)


# In[ ]:


trainFiles = trainDF[0].tolist()
trainClasses = trainDF[1].tolist()


# In[ ]:


devDF = pd.read_csv(devCSV, header=None)
print(devDF)


# In[ ]:


devFiles = devDF[0].tolist()
devClasses = devDF[1].tolist()


# In[ ]:


labels = ['A', 'B', 'C']


# ## Dataset Equilibre 

# In[ ]:


# plot the circle of value counts in dataset
import matplotlib.pyplot as plt

def plot_equilibre(equilibre, labels, title):
    plt.figure(figsize=(5,5))
    my_circle=plt.Circle( (0,0), 0.5, color='white')
    plt.pie(equilibre, labels=labels, colors=['red','green','blue'],autopct='%1.1f%%')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.title(title)
    plt.show()


# In[ ]:


equilibreTrain = []
[equilibreTrain.append(trainClasses.count(label)) for label in labels]
print(equilibreTrain)
plot_equilibre(equilibreTrain, labels, 'Train Data')
del equilibreTrain


# In[ ]:


equilibreDev = []
[equilibreDev.append(devClasses.count(label)) for label in labels]
print(equilibreDev)
plot_equilibre(equilibreDev, labels, 'Development Data')
del equilibreDev


# ## Import Libraries

# In[ ]:


import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

from IPython.display import Image
import matplotlib.pyplot as plt


# In[ ]:


#TargetSize = (192, 144) # image ratio = 4:3
TargetSize = (112,112)
def prepare_image(filepath):
    img = cv2.imread(filepath)
    # get image height, width
    (h, w) = img.shape[:2]
    if (w<h): # rotate270
        # calculate the center of the image
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, 270, 1.0)
        img = cv2.warpAffine(img, M, (h, w))
    img_resized = cv2.resize(img, TargetSize, interpolation=cv2.INTER_CUBIC)
    img_result  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_result


# ### Display Image file

# In[ ]:


plt.imshow(prepare_image(trainPath+trainFiles[1]))


# In[ ]:


plt.imshow(prepare_image(devPath+devFiles[1]))


# ## Prepare Data

# ### Training Data

# In[ ]:


trainX = []
[trainX.append(prepare_image(trainPath+file)) for file in trainFiles]
trainX = np.asarray(trainX) 
print(trainX.shape)

# data normalisation
trainX = trainX / 255.0


# In[ ]:


# Convert Y_data from {'A','B','C'} to {0,1,2}
trainY = []
[trainY.append(ord(trainClass) - 65) for trainClass in trainClasses]
#print(trainY)

# one-hot encoding
trainY = to_categorical(trainY)


# ### Development Data (for Validation)

# In[ ]:


validX = []
[validX.append(prepare_image(devPath+file)) for file in devFiles]
validX = np.asarray(validX)    
print(validX.shape)

# data normalisation
validX = validX / 255.0


# In[ ]:


# Convert Y_data from char to integer
validY = []
[validY.append(ord(devClass) - 65) for devClass in devClasses]
#print(validY)

# One-hot encoding
validY = to_categorical(validY)


# ### Shuffle Training Data

# In[ ]:


from sklearn.utils import shuffle
trainX,trainY = shuffle(trainX,trainY, random_state=42)


# In[ ]:


num_classes = 3


# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, BatchNormalization, Activation, LeakyReLU, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


get_ipython().system('pip install -q efficientnet')
import efficientnet.tfkeras as efn


# ## Build Model

# In[ ]:


input_shape = trainX.shape[1:]
print(trainX.shape[1:])


# In[ ]:


# Build Model
net = efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)

# add two FC layers (with L2 regularization)
x = net.output
x = GlobalAveragePooling2D()(x)
#x = BatchNormalization()(x)

#x = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
x = Dense(256)(x)
#x = Dropout(0.2)(x)

#x = Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
x = Dense(32)(x)
#x = Dropout(0.2)(x)

# Output layer
out = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=net.input, outputs=out)
model.summary()


# In[ ]:


# Compile Model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


## set Checkpoint : save best only, verbose on
#checkpoint = ModelCheckpoint("mango_classification.hdf5", monitor='accuracy', verbose=0, save_best_only=True, mode='auto', save_freq=1)


# ## Train Model

# In[ ]:


batch_size = 16
num_epochs = 90


# In[ ]:


# Train Model
history = model.fit(trainX,trainY,batch_size=batch_size,epochs=num_epochs, validation_data=(validX,validY)) #, callbacks=[checkpoint])


# ## Save Model

# In[ ]:


## Save Model
model.save('mango_efficientnet.h5')


# In[ ]:


## load best model weights if using callback (save-best-only)
#model.load_weights("mango_classification.hdf5")


# ## Evaluate Model

# ### predict Validation set for a Confusion Matrix

# In[ ]:


predY = model.predict(validX)
y_pred = np.argmax(predY,axis=1)
y_actual = np.argmax(validY,axis=1)
#y_label= [labels[k] for k in y_pred]
cm = confusion_matrix(y_actual, y_pred)
print(cm)


# In[ ]:


print(classification_report(y_actual, y_pred, target_names=labels))


# ## Test Model

# ### prepare Test data

# In[ ]:


testPath  = '../input/aicup-2020-mango-c1-p1/C1-P1_Test/C1-P1_Test/'
testFiles = os.listdir(testPath)
testFiles.sort()
print(testFiles)


# In[ ]:


testX = []
[testX.append(prepare_image(testPath+file)) for file in testFiles]
testX = np.asarray(testX)    
print(testX.shape)
testX = testX/255.0


# ### model prediction

# In[ ]:


testY = model.predict(testX)
print(testY[0])


# In[ ]:


# create a list of the predicted label
pred_y = []
for y in testY:
    maxindex = int(np.argmax(y))
    pred_y.append(labels[maxindex])
print(pred_y)


# ## Generate Submission

# In[ ]:


# create dataframe with each image id & its predicted label
dfTest = pd.DataFrame(columns = ['image_id','label'])
dfTest['image_id'] = testFiles
dfTest['label']=pred_y

# output a .csv file
dfTest.to_csv('test_submission.csv',index=False)


# ## Versions
# *Note. training accuracy did not converge if using transfer learning*
# 
# V1. FCx2 (1024,64) epochs=90, Train = 99.11%, Valid = 76.38% <br />
# V2. FCx2 (256,32)  epochs=90, Train = 99.25%, Valid = 74.25% <br />

# In[ ]:




