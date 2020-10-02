#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img,load_img
from keras.models import Sequential
from keras.layers import Dense , Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD,Adam 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras import backend as K
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
import shutil
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight


# In[ ]:


data_dir = Path('../input/chest_xray/chest_xray')
train_dir = data_dir / 'train'
val_dir = data_dir / 'val'
test_dir = data_dir / 'test'
class_names=['NORMAL','PNEUMONIA']
print(test_dir)


# In[ ]:


def read_data(data_directory):
    imgs = []
    labs = []
    for filename in os.listdir(data_directory):
        if filename.endswith('.jpeg'):
            filename = Path(data_directory) / filename
            img = cv2.imread(str(filename),1)
            img = cv2.resize(img, (224,224))
#                 img = img.astype('float32')/255.0
            imgs.append(img)
#                 labs.append(label)   
#        import pdb; pdb.set_trace()
    imgs = np.array(imgs)
#         labs = np.array(labs)  
    
    return imgs


# In[ ]:


def image_augment(imgs,batch_size,num_iterations):
    
    datagen = ImageDataGenerator(rotation_range=10,width_shift_range=0.2,
                                 height_shift_range=0.2,shear_range=0.2,
                                 zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
    i =0
    
    
    aug_imgs = []
#     datagen.fit(imgs)
    for batch in datagen.flow(imgs,batch_size=batch_size):
        i+=1
        aug_imgs.append(batch)

        if i>=num_iterations:
            break
    imgs =np.array(aug_imgs[0])  
#    imgs = np.squeeze(imgs)
    for index in range(num_iterations-1):
        img =np.array(aug_imgs[index+1])
        #img = np.squeeze(img)
        imgs = np.concatenate((imgs,img),axis=0)
        print(imgs.shape)
    

   
    return imgs


# In[ ]:


x_train_normal = read_data('../input/chest_xray/chest_xray/train/NORMAL/')
x_train_normal = image_augment(x_train_normal,batch_size=256,num_iterations=16)
y_train_normal = np.zeros(x_train_normal.shape[0])

x_train_pneumonia = read_data('../input/chest_xray/chest_xray/train/PNEUMONIA/')
x_train_pneumonia = image_augment(x_train_pneumonia,batch_size=256,num_iterations=16)
y_train_pneumonia = np.ones(x_train_pneumonia.shape[0])

x_train = np.concatenate((x_train_normal,x_train_pneumonia),axis=0)
y_train = np.concatenate((y_train_normal,y_train_pneumonia),axis=0)

print(x_train.shape)


# In[ ]:


y_train = to_categorical(y_train,2)


# In[ ]:


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),activation="relu", padding="same",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(3,3),strides=2, padding="same"))

model.add(Conv2D(filters=64, kernel_size=(3,3),activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(3,3),strides=2, padding="same"))

model.add(Conv2D(filters=64, kernel_size=(3,3),activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(3,3),strides=2, padding="same"))

model.add(Conv2D(filters=64, kernel_size=(3,3),activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(3,3),strides=2, padding="same"))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(rate=0.3))


model.add(Dense(64, activation="relu"))
model.add(Dropout(rate=0.3))

# model.add(Dense(2, activation="sigmoid"))
# model.add(Dense(2, activation="softmax"))
model.add(Dense(2))
model.add(Activation('softmax'))


# In[ ]:


from keras.optimizers import SGD,Adam 
model.compile(Adam(0.001),loss="categorical_crossentropy", metrics=["accuracy"])#categorical_crossentropy
print(model.summary())


# In[ ]:


x_test_normal = read_data('../input/chest_xray/chest_xray/test/NORMAL/')
x_test_normal = image_augment(x_test_normal,batch_size=234,num_iterations=1)
y_test_normal = np.zeros(x_test_normal.shape[0])

x_test_pneumonia= read_data('../input/chest_xray/chest_xray/test/PNEUMONIA/')
x_test_pneumonia = image_augment(x_test_pneumonia,batch_size=195,num_iterations=2)
y_test_pneumonia = np.ones(x_test_pneumonia.shape[0])
    
x_test = np.concatenate((x_test_normal,x_test_pneumonia),axis=0)
y_test = np.concatenate((y_test_normal,y_test_pneumonia),axis=0)
print(x_test.shape)


# In[ ]:


y_test = to_categorical(y_test,2)


# In[ ]:


from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.utils import to_categorical

file_name="./model_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath=file_name, monitor='val_acc', verbose=3, 
                             save_best_only=True,save_weights_only=False, mode='auto', period=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8, verbose=1, 
                                 mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0001)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, 
                              mode='auto', baseline=None, restore_best_weights=False)
callback_lists=[checkpoint,earlystop]


#history= model.fit_generator(train_batches,steps_per_epoch=31,class_weight={0:3.0,1:1.0},epochs=10)

# history= model.fit_generator(train_batches,validation_data=valid_batches,validation_steps=1,
#                             callbacks=callback_lists, steps_per_epoch=31, epochs=10)
# history= model.fit_generator(train_batches,validation_data=valid_batches,validation_steps=1,
#                             steps_per_epoch=25,class_weight={0:3.0,1:1.0},epochs=13)
# validation_data=(x_test,y_test)
history = model.fit(x_train,y_train,batch_size=256,epochs=30,validation_split=0.05,callbacks=callback_lists)
# history = model.fit(x_train,y_train,batch_size=256,epochs=10,class_weight=d_class_weights,validation_data=(x_test,y_test),callbacks=callback_lists)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,precision_score

# from keras.models import load_model
# model2 = load_model('./model_11-0.92.hdf5')
pred=model.predict(x_test, batch_size=32, verbose=1, steps=None)
pred = np.argmax(pred,axis = 1) 
print(pred)
y_true = np.argmax(y_test,axis = 1)


# In[ ]:


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_true, pred)
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))
plt.show()


# In[ ]:


recall = recall_score(y_true,pred)
precision = precision_score(y_true,pred)
print('accuracy is %.4f:'% (accuracy_score(y_true,pred)))
print('recall is %.4f:' % (recall))
print('precision is %.4f:'% (precision))
print('F1 score is %.4f:'% (2*recall*precision/(recall+precision)))


# In[ ]:




