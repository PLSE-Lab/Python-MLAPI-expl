#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# initiating gpu using tensorflow.
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)


# In[ ]:


#importing libraries for the data processing and model.
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from keras.models import load_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# defining the path and classes.
directory = '../input/state-farm-distracted-driver-detection/imgs/train'
test_directory = '../input/state-farm-distracted-driver-detection/imgs/test/'
random_test = '../input/driver/'
classes = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']


# In[ ]:


# defining a shape to be used for our models.
img_size1 = 240
img_size2 = 240


# In[ ]:


# Train class image for display.
for i in classes:
    path = os.path.join(directory,i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break


# In[ ]:


# Test class image for display.
test_array = []
for img in os.listdir(test_directory):
    img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_GRAYSCALE)
    test_array = img_array
    plt.imshow(img_array, cmap='gray')
    plt.show()
    break


# In[ ]:


# checkking image size using shape.
print(img_array.shape)


# In[ ]:


# trying out the resize image functionality
new_img = cv2.resize(test_array,(img_size2,img_size1))
plt.imshow(new_img,cmap='gray')
plt.show()


# In[ ]:


# creating a training dataset.
training_data = []
i = 0
def create_training_data():
    for category in classes:
        path = os.path.join(directory,category)
        class_num = classes.index(category)
        
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(img_array,(img_size2,img_size1))
            training_data.append([
                new_img,class_num])


# In[ ]:


# Creating a test dataset.
testing_data = []
i = 0
def create_testing_data():        
    for img in os.listdir(test_directory):
        img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_array,(img_size2,img_size1))
        testing_data.append([img,
            new_img])


# In[ ]:


create_training_data()


# In[ ]:


create_testing_data()


# In[ ]:


print(len(training_data))
print(len(testing_data))


# In[ ]:


random.shuffle(training_data)


# In[ ]:


x = []
y = []


# In[ ]:


for features, label in training_data:
    x.append(features)
    y.append(label)


# In[ ]:


x[0].shape


# In[ ]:


len(x)


# In[ ]:


#X  = np.array(x[1]).reshape(-1,img_size2,img_size1,1)
#i = 1
#for i in range(len(x)):
X = np.array(x).reshape(-1,img_size2,img_size1,1)
#    X = np.append(X,Y,axis = 0)
X.shape,X[0].shape


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=50)


# In[ ]:


Y_train = np_utils.to_categorical(y_train,num_classes=10)
Y_test = np_utils.to_categorical(y_test,num_classes=10)


# In[ ]:


model = Sequential()


# In[ ]:


model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(240,240,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))


# In[ ]:


model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))


# In[ ]:


model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.5))


# In[ ]:


model.add(Flatten())
model.add(Dense(units = 512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units = 128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[ ]:


callbacks = [EarlyStopping(monitor='val_acc',patience=5)]


# In[ ]:


batch_size = 50
n_epochs = 20


# In[ ]:


results = model.fit(x_train,Y_train,batch_size=batch_size,epochs=n_epochs,verbose=1,validation_data=(x_test,Y_test),callbacks=callbacks)


# In[ ]:


# Plot training & validation accuracy values
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:


preds = model.predict(np.array(testing_data[0][1]).reshape(-1,img_size2,img_size1,1))


# In[ ]:


model.save_weights('./driverdistraction_lr_weights.h5', overwrite=True)


# In[ ]:


model.save('./driverdistraction_lr_weights.h5')


# In[ ]:


loaded_model = load_model('../input/driver-distraction/driverdistraction_lr_weights.h5')


# In[ ]:


test_data = np.array(testing_data[1001][1]).reshape(-1,img_size2,img_size1,1)


# In[ ]:


preds = model.predict(test_data)
preds= np.argmax(preds)
preds


# In[ ]:




classes = {0: "safe driving",
1: "texting - right",
2: "talking on the phone - right",
3: "texting - left",
4: "talking on the phone - left",
5: "operating the radio",
6: "drinking",
7: "reaching behind",
8: "hair and makeup",
9: "talking to passenger",
}


for key,value in classes.items():
    if preds==key:
        predicted = value

predicted     


# In[ ]:


print(predicted)
new_img = cv2.resize(testing_data[1001][1],(img_size2,img_size1))
plt.imshow(new_img,cmap='gray')
plt.show()


# 
#     c0: safe driving
#     c1: texting - right
#     c2: talking on the phone - right
#     c3: texting - left
#     c4: talking on the phone - left
#     c5: operating the radio
#     c6: drinking
#     c7: reaching behind
#     c8: hair and makeup
#     c9: talking to passenger
# 

# In[ ]:


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

