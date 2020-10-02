#!/usr/bin/env python
# coding: utf-8

# # Mango Classification

# ## training EfficientNet-B7

# * Website: https://aidea-web.tw/topic/72f6ea6a-9300-445a-bedc-9e9f27d91b1c
# * Forum : https://www.facebook.com/groups/184056976135023/

# ## Dataset : AICUP 2020 Mango sample data
# ![Taiwan%20AICUP%202020%20Mango.JPG](attachment:Taiwan%20AICUP%202020%20Mango.JPG)

# In[ ]:


import os
print(os.listdir('../input'))


# In[ ]:


trainPath = '../input/Mango/train'
devPath   = '../input/Mango/dev'
testPath  = '../input/Mango/test'

trainCSV = '../input/Mango/train.csv'
devCSV   = '../input/Mango/dev.csv'


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


testFiles = os.listdir(testPath+'/unknown')
testFiles.sort()
print(testFiles)


# In[ ]:


del testFiles[1600]


# ### display image of a training data

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[ ]:


plt.imshow(mpimg.imread(trainPath+'/'+trainClasses[0]+'/'+trainFiles[0]))


# In[ ]:


plt.imshow(mpimg.imread(devPath+'/'+devClasses[0]+'/'+devFiles[0]))


# ## Dataset Equilibre 

# In[ ]:


labels = ['A', 'B', 'C']


# In[ ]:


# plot the circle of value counts in dataset
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


# ## Data Augmentation

# In[ ]:


import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical

from IPython.display import Image
import matplotlib.pyplot as plt


# In[ ]:


target_size=(112,112)
batch_size = 16


# In[ ]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    trainPath,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',    
    shuffle=True,
    seed=42,
    class_mode='categorical')


# In[ ]:


valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_directory(
    devPath,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical')


# In[ ]:


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    testPath,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='rgb',    
    class_mode='categorical')


# ## Build Model

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


# In[ ]:


num_classes = 3


# In[ ]:


input_shape = (112,112,3)


# In[ ]:


# Build Model
net = efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)

# add two FC layers (with L2 regularization)
x = net.output
x = GlobalAveragePooling2D()(x)
#x = BatchNormalization()(x)

#x = Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
x = Dense(512)(x)
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


num_train = len(trainFiles)
num_valid = len(devFiles)
num_epochs = 10
print(num_train, num_valid)


# In[ ]:


# Train Model
history = model.fit_generator(train_generator,steps_per_epoch=num_train // batch_size,epochs=num_epochs, validation_data=valid_generator, validation_steps=num_valid // batch_size) #, callbacks=[checkpoint])


# ## Save Model

# In[ ]:


## Save Model
model.save('mango_efficientnet.h5')


# In[ ]:


## load best model weights if using callback (save-best-only)
#model.load_weights("mango_classification.hdf5")


# ## Evaluate Model

# In[ ]:


score = model.evaluate_generator(valid_generator, steps=num_valid//batch_size)
print(score)


# ## Confusion Matrix (validation set)

# In[ ]:


predY=model.predict_generator(valid_generator)
y_pred = np.argmax(predY,axis=1)
#y_label= [labels[k] for k in y_pred]
y_actual = valid_generator.classes
cm = confusion_matrix(y_actual, y_pred)
print(cm)


# ### report confusion matrix

# In[ ]:


print(classification_report(y_actual, y_pred, target_names=labels))


# ### plot confusion matrix

# In[ ]:


import itertools
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()        


# In[ ]:


plot_confusion_matrix(cm, 
                      normalize=False,
                      target_names = labels,
                      title="Confusion Matrix, not Normalized")


# ## Test Model

# ### try a test image

# In[ ]:


plt.imshow(mpimg.imread(testPath+'/unknown/'+testFiles[0]))


# In[ ]:


#TargetSize = (192, 144) # ratio = 4:3
TargetSize = (112, 112)
def prepare_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, TargetSize, interpolation=cv2.INTER_CUBIC)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[ ]:


testData = prepare_image(testPath+'/unknown/'+testFiles[0]).reshape(1,112,112,3)
testData = testData / 255.0
print(testData.shape)
predictions = model.predict(testData)
print(predictions[0])


# In[ ]:


maxindex = int(np.argmax(predictions[0]))
print('Predicted: %s, Probability = %f' %(labels[maxindex], predictions[0][maxindex]) )


# ### model prediction

# In[ ]:


testY = model.predict_generator(test_generator)
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


# ## Version History

# In[ ]:




