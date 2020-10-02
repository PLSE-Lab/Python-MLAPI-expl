#!/usr/bin/env python
# coding: utf-8

# ## Robust digit training with ImageDataAugmentor
# 
# > Data augmentation is an integral part in most of the of the contemporary ML pipelines. Using augmentations an image dataset can be artificially expanded by creating modified versions of the original images. Augmentations can be used in making a ML algorithm more robust by creating variations of the training set of images that are likely to be seen by the model in the real use case.
# 
# > This kernel demonstrates the usage of ImageDataAugmentor [https://github.com/mjkvaak/ImageDataAugmentor/blob/master/README.md], which is a custom image data generator for Keras supporting the use of modern augmentation modules (e.g. imgaug and albumentations). For reference, albumentations outperforms the Keras build in augmentation module in speed in EVERY transformation task - plus includes a huge number of augmentations that don't exist in the latter [https://github.com/albu/albumentations#benchmarking-results]. 

# In[ ]:


## Fix the seeds
from numpy.random import seed
seed = 2019

from tensorflow import set_random_seed
set_random_seed(2019)


# In[ ]:





# ## Import necessary packages and functions

# In[ ]:


## Install ImageDataAugmentor
get_ipython().system('git clone https://github.com/mjkvaak/ImageDataAugmentor')
get_ipython().system('rm -r ImageDataAugmentor/.git/ #<- To get rid of Kaggle\'s "Output path contains too many nested subdirectories"-error')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import cv2
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from keras.utils.np_utils import to_categorical
from keras.utils import Sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau

from ImageDataAugmentor.image_data_augmentor import *
import albumentations


# In[ ]:





# ## Fetch the data

# In[ ]:


IMG_SIZE = 28


# In[ ]:


import os
os.listdir('../input/')


# In[ ]:


## Fetch the prepared data
train_df = pd.read_csv('../input/train.csv')
print("Data fetched!")

## Constants
num_classes = train_df.label.nunique()
cols = train_df.columns[1:]


# In[ ]:


## Define train and test arrays
Y_train = train_df['label']
X_train = train_df.drop('label', axis = 1)


# In[ ]:


## Split train into train & validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=seed, stratify=Y_train)


# In[ ]:


## Visualize the data
plt.figure(figsize=(16,8))
plt.subplot(121)
sns.countplot(Y_train)
plt.title('Train')

plt.subplot(122)
sns.countplot(Y_val)
plt.title('Validation')
plt.show()


# In[ ]:


## Visualize the images
tmp = train_df.sample(frac=1).sort_values('label')
tmp = tmp.groupby('label').head(10)
tmp = tmp.set_index('label')
rows = len(tmp)//10+1

plt.figure(figsize=(16,16))
for idx in range(len(tmp)):
    plt.subplot(rows, 10, idx+1)
    letter = tmp.iloc[idx].values.reshape(IMG_SIZE,IMG_SIZE)
    plt.imshow(letter)
    plt.title(tmp.index.values[idx])
    plt.axis('off')

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()


# In[ ]:


## Reshape data into tensor format
X_train = X_train.values.reshape(-1,IMG_SIZE, IMG_SIZE, 1)
X_val = X_val.values.reshape(-1,IMG_SIZE, IMG_SIZE, 1)


# In[ ]:


## Transform targets into categorical vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = num_classes)
Y_val = to_categorical(Y_val, num_classes = num_classes)


# In[ ]:


## Define the model
model = None
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (IMG_SIZE,IMG_SIZE,1)))
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
model.add(Dense(num_classes, activation = "softmax"))


# In[ ]:


## Define the optimizer & compile the model
optimizer = Adam(lr=0.001, decay=0.001)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()


# In[ ]:


## Reduce the learning rate in case the learner hits a plateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


## Fix some parameters (adjust for best performance)
epochs = 30
batch_size = 500


# In[ ]:


## Data augmentation: just picked some from https://albumentations.readthedocs.io/en/latest/api/augmentations.html that sounded relevant
AUGMENTATIONS = albumentations.Compose([
    albumentations.ShiftScaleRotate(
        p=0.8, #apply these to 80% of the images
        shift_limit=0.1, #translate by (-0.1%, 0.1%)
        scale_limit=0.1, #zoom by (-0.1%, 0.1%)
        rotate_limit=20, #rotate by (-20, 20)
        border_mode=cv2.BORDER_CONSTANT, value = [0,0,0], #fill emptyness with (0,0,0)
        ),
    albumentations.Blur(blur_limit=1, p=0.2),
    albumentations.ElasticTransform(alpha=0.1, sigma=5, alpha_affine=2,
                                     border_mode=cv2.BORDER_CONSTANT, value = [0,0,0], #fill emptyness with (0,0,0)
                                    ),
    albumentations.ToFloat(max_value=255),
])


# In[ ]:


## Define the train & validation generators
train_data_gen = ImageDataAugmentor(augment=AUGMENTATIONS)
training_generator = train_data_gen.flow(X_train,Y_train, 
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         seed = seed,
                                         )

val_data_gen = ImageDataAugmentor(augment=albumentations.ToFloat(max_value=255))
validation_generator = val_data_gen.flow(X_val,Y_val, 
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         seed = seed
                                         )


# In[ ]:


## Visualize the augmented data
training_generator.show_batch(rows=10)


# In[ ]:


## Train the model
history = model.fit_generator(training_generator,
                              steps_per_epoch=len(training_generator)*10, #<-expand the dataset 10-fold with random augmentations
                              epochs = epochs, 
                              validation_data = validation_generator,
                              validation_steps = len(validation_generator),
                              verbose = 1, 
                              callbacks=[learning_rate_reduction]
                             )


# In[ ]:


## Visualize the training history
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot(title='Losses')
history_df[['acc', 'val_acc']].plot(title='Accuracies')
plt.show()


# In[ ]:


## Calculate the accuracy from validation
fl,fac = model.evaluate(X_val/255.,Y_val,verbose=0)
print("Final Loss =",fl)
print("Final Accuracy =",fac)


# In[ ]:


## ## Uncomment to save the model
## model.save("digit_model.h5")


# In[ ]:


## Make predictions from test set
Y_pred_ohv = model.predict(X_val/255.)

## Flatten the predicted categorical labels
Y_pred = np.argmax(Y_pred_ohv, axis = 1) 

## Flatten the real categorical labels
Y_true = np.argmax(Y_val, axis = 1) 


# In[ ]:


## Define the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues): 
    ''' 
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`
    ''' 
 
    if normalize: 
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap) 
    plt.title(title) 
    plt.colorbar() 
    tick_marks = np.arange(len(classes)) 
    plt.xticks(tick_marks, classes, rotation=45) 
    plt.yticks(tick_marks, classes) 

 
    thresh = cm.max() / 2. 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): 
        plt.text(j, i, cm[i, j], 
                 horizontalalignment="center", 
                 color="white" if cm[i, j] > thresh else "black") 
 
    plt.tight_layout() 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')
    #plt.plot()


# In[ ]:


## Determine the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred) 

## Plot
plt.figure(figsize=(16,8)) 
plt.subplot(121)
plot_confusion_matrix(confusion_mtx, classes = range(num_classes))
plt.subplot(122)
plot_confusion_matrix(confusion_mtx, classes = range(num_classes), normalize=True)
plt.show()


# In[ ]:


## Checking the errors and visualizing them
errors = (Y_true- Y_pred != 0)

Y_true_errors = Y_true[errors]
Y_pred_errors = Y_pred[errors]
X_val_errors = X_val[errors]
Y_ohv_errors = Y_pred_ohv[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    n = 0
    nrows = 3
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True, figsize=(12,12))
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((IMG_SIZE,IMG_SIZE)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

## Probability of wrong prediction
y_pred_errors_prob = np.max(Y_ohv_errors,axis = 1)

## Probability of true values in error set
true_prob_errors = np.diagonal(np.take(Y_ohv_errors, Y_pred_errors, axis=1))

## Difference between true and error set
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

## Top 9 errors 
most_important_errors = sorted_dela_errors[-9:]

## Show the top 9 errors
display_errors(most_important_errors, X_val_errors, Y_true_errors, Y_pred_errors)


# In[ ]:





# ## Predict on custom data
# > This could be your own data 

# In[ ]:


## ## Uncomment to load the previously saved the model
## model = None
## model = load_model("digit_model.h5")


# In[ ]:


## Fetch the test data
test_df = pd.read_csv('../input/test.csv')
print("Data fetched!")


# In[ ]:


## Reshape data into tensor format
X_test = test_df.values.reshape(-1,IMG_SIZE, IMG_SIZE, 1)


# In[ ]:


## Make predictions
Y_test = model.predict(X_test/255.)

## Flatten the predicted categorical labels
Y_test = np.argmax(Y_test, axis = 1) 


# In[ ]:


## Prepare the submission
submit = pd.read_csv('../input/sample_submission.csv')
submit['Label'] = Y_test
submit.to_csv('submission.csv', index=False)
submit.head()

