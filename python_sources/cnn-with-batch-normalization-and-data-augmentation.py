#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau


# In[ ]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[ ]:


train = pd.read_csv("../input/Kannada-MNIST/train.csv")
test = pd.read_csv("../input/Kannada-MNIST/test.csv")
dig = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")
sample = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")


# In[ ]:


train.head()


# In[ ]:


y = train.label.value_counts()
x = y.index
plt.bar(x,y)
plt.xticks(x, labels=x)
plt.xlabel('Kannada Numbers')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


print(f"Train data shape {train.shape}")
print(f"Test data shape {test.shape}")


# In[ ]:


x_train = train.drop(columns=['label'])  #largest pixel value
y_train = to_categorical(train.label) #one-hot 
x_test = test.drop(columns="id")


# In[ ]:


x_train[train.label == 0].iloc[0].to_numpy().astype(np.uint8).reshape(28, 28)


# In[ ]:


fig, ax = plt.subplots(ncols=10, figsize=(15,15))
for i in range(10):
    kannada = x_train[train.label == i]
    ax[i].set_title(i)
    ax[i].axis('off')
    ax[i].imshow(kannada.iloc[0, :].to_numpy().astype(np.uint8).reshape(28, 28))


# In[ ]:


x_train = x_train /255
x_test = x_test / 255


# In[ ]:


x_train = x_train.values.reshape(-1,28,28,1) # 28 * 28 = 784
x_test = x_test.values.reshape(-1,28,28,1)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)


# In[ ]:


# CREATE MORE TRAINING IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.1,  
        width_shift_range=0.1, 
        height_shift_range=0.1)


# In[ ]:


datagen.fit(X_train)


# In[ ]:


EPOCHS=50
BATCH_SIZE=64


# In[ ]:


callbacks = [ReduceLROnPlateau(monitor='val_accuracy',
                               patience=5,
                               verbose=1,
                               factor=0.5,
                               min_lr=0.00001)]


# In[ ]:


#build based on :
#https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist#1.-How-many-convolution-subsambling-pairs?
model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu')) 
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


print(model.summary())


# In[ ]:


history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=BATCH_SIZE),
                              epochs = EPOCHS,
#                               batch_size=BATCH_SIZE,
                              validation_data = (X_test,Y_test),
                              verbose = 1,
                              steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                              callbacks=callbacks)


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(history.history['val_accuracy'], label = 'validation')
plt.plot(history.history['accuracy'], label = 'training')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
axes = plt.gca()
axes.set_ylim([0.98,1])

plt.figure(figsize=(15,5))
plt.plot(history.history['val_loss'], label = 'validation')
plt.plot(history.history['loss'], label = 'training')
plt.title('model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()

plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'pred = model.predict(X_test)\npred = np.argmax(pred,axis=1)\ntarget = np.argmax(Y_test,axis=1)')


# In[ ]:


cm = confusion_matrix(target, pred)
print(cm)


# In[ ]:


#source: https://www.kaggle.com/agungor2/various-confusion-matrix-plots 

df_cm = pd.DataFrame(cm, columns=np.unique(target), index = np.unique(target))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (16,10))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="BuPu", annot=True, fmt='g',annot_kws={"size": 16}, vmax=10)# font size
plt.show()


# In[ ]:


accuracy_score(target, pred)


# In[ ]:


print(classification_report(target, pred))


# In[ ]:


submission_pred = model.predict(x_test)


# In[ ]:


submission_pred.shape


# In[ ]:


submission_pred = np.argmax(submission_pred, axis=1)


# In[ ]:


df_submission = pd.DataFrame({'id': np.arange(0,submission_pred.shape[0]),
                             'label':submission_pred})


# In[ ]:


df_submission.head(10)


# In[ ]:


df_submission.to_csv('submission.csv',index=False)


# In[ ]:




