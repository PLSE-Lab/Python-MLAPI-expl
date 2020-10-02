#!/usr/bin/env python
# coding: utf-8

# Importing Essential pyhton Libraries

# In[21]:


import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import fnmatch
from glob import glob
import gc
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing Machine learning Libraries

# In[57]:


import keras
import tensorflow
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization,Conv2D,MaxPool2D,MaxPooling2D
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score


# In[23]:


ls ../input/IDC_regular_ps50_idx5/ 


# In[24]:


ls ../input/IDC_regular_ps50_idx5/14305


# In[25]:


images=glob('../input/IDC_regular_ps50_idx5/**/*.png',recursive=True)
for fileindex,filename in enumerate(images):
    if fileindex==10:
        break
    print(filename)


# In[26]:


image_9075=cv2.imread('../input/IDC_regular_ps50_idx5/9075/1/9075_idx5_x1751_y351_class1.png')


# In[27]:


plt.figure(figsize=(15,15))
plt.imshow(image_9075)


# As opencv uses BGR format initially, So we have to convert the image to RGB format

# In[28]:


plt.imshow(cv2.cvtColor(image_9075,cv2.COLOR_BGR2RGB))


# Now lets look at all the other sample images

# In[29]:


plt.rcParams['figure.figsize'] = (10.0, 10.0)
for singleIndex,singleImage in enumerate(images[:25]):
    im = cv2.imread(singleImage)
    im = cv2.resize(im, (50, 50)) 
    plt.subplot(5, 5, singleIndex+1) #.set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.axis('off')


# Now lets filter and differentiate different classes by using fnmatch library which supports wildcards

# In[30]:


patternZero = '*class0.png'
patternOne = '*class1.png'
classZero = fnmatch.filter(images, patternZero)
classOne = fnmatch.filter(images, patternOne)
print("IDC(-)\n\n",classZero[0:5],'\n')
print("IDC(+)\n\n",classOne[0:5])


# Now let's resize the images using cv2 library

# In[31]:


def process_images(lowerIndex,upperIndex):
    x = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    for img in images[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    return x,y

X,Y=process_images(0,90000)


# In[32]:


X[:6]


# In[33]:


X1=np.array(X)


# In[34]:


X1.shape


# In[35]:


df = pd.DataFrame()
df["images"]=X
df["labels"]=Y


# In[36]:


X2=df["images"]
Y2=df["labels"]


# In[37]:


X2=np.array(X2)


# In[38]:


X2[:5]


# In[39]:


X2.shape


# Now, lets separate images 

# In[40]:


# Separation of classes of images
imgs0=[]
imgs1=[]
imgs0 = X2[Y2==0]
imgs1 = X2[Y2==1] 


# Now let's look at the Statisitcs of all the images we have

# In[41]:


print('Total number of images: {}'.format(len(X2)))
print('Number of Class 0 images: {}'.format(np.sum(Y2==0)))
print('Number of class 1 Images: {}'.format(np.sum(Y2==1)))
print('Image shape (Width, Height, Channels): {}'.format(X1[0].shape))


# Now's lets split the data into train test split

# In[42]:


X=np.array(X)

#Standarizing the data for the model 
X=X/255.0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[43]:


del X,X1,X2,Y2
gc.collect()


# In[44]:


X_train.shape,X_test.shape


# In[45]:


df['labels'].value_counts()


# In[46]:


sns.countplot(df['labels'])


# As we can conculde from the above figure that there is an uneven distribution in the labels, as the number of sample images of class 0 is greater

# Now, lets Implement the one hot encoding for the categorical variables

# In[47]:


y_trainCat=to_categorical(Y_train,num_classes=2)
y_testCat=to_categorical(Y_test,num_classes=2)


# In[48]:


# Helper Functions  Learning Curves and Confusion Matrix

class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
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

def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')


# In[49]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


# In[50]:


batch_size = 128
num_classes = 2
epochs = 8
img_rows,img_cols=50,50
input_shape = (img_rows, img_cols, 3)
e = 2


# In[51]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape,strides=e))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# In[52]:


model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


# In[53]:


# model.fit_generator(datagen.flow(X_train,y_trainCat,batch_size=16),steps_per_epoch=X_train/32,epochs=epochs,validation_data=[X_test,y_testCat])


# In[55]:


history = model.fit_generator(datagen.flow(X_train,y_trainCat, batch_size=32),
                        steps_per_epoch=len(X_train) / 32, 
                              epochs=epochs,validation_data = [X_test, y_testCat],
                              callbacks = [MetricsCheckpoint('logs')])


# In[56]:


y_pred=model.predict(X_test)


# In[ ]:





# In[ ]:




