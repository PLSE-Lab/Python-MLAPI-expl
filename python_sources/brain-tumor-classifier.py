#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# To remove deprecated warnings from the tensorflow
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


pwd


# In[ ]:


import os
os.chdir("/kaggle/input/brain-mri-images-for-brain-tumor-detection")


# In[ ]:


pwd


# In[ ]:


PATH = os.getcwd()


# In[ ]:


DATA_PATH = os.path.join(PATH, "brain_tumor_dataset")
data_dir_list = os.listdir(DATA_PATH)


# In[ ]:


print(data_dir_list)


# In[ ]:


import cv2

classes_names_list=[]
img_data_list=[]

for dataset in data_dir_list:
    classes_names_list.append(dataset) 
    print ('Loading images from {} folder\n'.format(dataset)) 
    img_list=os.listdir(DATA_PATH+'/'+ dataset)
    for img in img_list:
        input_img=cv2.imread(DATA_PATH + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(224, 224))
        (b, g, r)=cv2.split(input_img_resize) 
        img=cv2.merge([r,g,b])
        img_data_list.append(img)


# In[ ]:


num_classes = len(classes_names_list)
print(num_classes)


# In[ ]:


import numpy as np

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255


# In[ ]:


print (img_data.shape)


# In[ ]:


#show one training sample
from matplotlib import pyplot as plt
plt.imshow(img_data[97])
plt.show()


# In[ ]:


num_of_samples = img_data.shape[0]
input_shape = img_data[0].shape


# In[ ]:


classes = np.ones((num_of_samples,), dtype='int64')

classes[0:98]=0
classes[98:]=1


# In[ ]:


from keras.utils import to_categorical

classes = to_categorical(classes, num_classes)


# In[ ]:


from sklearn.utils import shuffle

X, Y = shuffle(img_data, classes, random_state=456)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=456)


# In[ ]:


# Check the number of images in each dataset split
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# In[ ]:


#### Build the model
model = Sequential()

model.add(Conv2D(32, (3, 3),activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'hist = model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, y_test))')


# In[ ]:


score = model.evaluate(X_test, y_test, batch_size=32)

print('Test Loss:', score[0])
print('Test Accuracy:', score[1])


# In[ ]:


from sklearn.metrics import confusion_matrix

Y_pred = model.predict(X_test)


# In[ ]:


y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)


# In[ ]:


print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))


# ## Data Augementation

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.1, 
    zoom_range=0.4, 
    vertical_flip=True,
    brightness_range=[0.5, 1.5],
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

test_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
        DATA_PATH,
        target_size=(224, 224), 
        batch_size=32,
        class_mode='binary',
        color_mode='rgb', 
        shuffle=True,  
        #save_to_dir='Train_Augmented_Images', 
        #save_prefix='TrainAugmented', 
        #save_format='jpeg'
)

test_generator = test_data_gen.flow_from_directory(
        DATA_PATH,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        color_mode='rgb',
        shuffle=True, 
        seed=None, 
        #save_to_dir='Test_Augmented_Images', 
        #save_prefix='TestAugmented', 
        #save_format='jpeg'
)


# In[ ]:


train_generator.class_indices


# In[ ]:


test_generator.class_indices


# In[ ]:


#### Build the model
model = Sequential()

model.add(Conv2D(32, (3, 3),activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])


# In[ ]:


model.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit_generator(train_generator, epochs=20, validation_data=test_generator)')


# ### TL - VGG16

# In[ ]:


from keras.layers import Input, Dense
from keras.models import Model


# In[ ]:


image_input = Input(shape=(224, 224, 3))


# In[ ]:


from keras.applications.vgg16 import VGG16

model = VGG16(input_tensor=image_input, include_top=False, weights='imagenet')


# In[ ]:


last_layer = model.get_layer('block5_pool').output
x = Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model = Model(image_input, out)


# In[ ]:


custom_vgg_model.summary()


# In[ ]:


# freeze all the layers except the dense layers
for layer in custom_vgg_model.layers[:-3]:
    layer.trainable = False


# In[ ]:


custom_vgg_model.summary()


# In[ ]:


custom_vgg_model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'hist = custom_vgg_model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(X_test, y_test))')


# In[ ]:


(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=32, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


# In[ ]:


Y_train_pred = custom_vgg_model.predict(X_test)


# In[ ]:


y_train_pred = np.argmax(Y_train_pred, axis=1)
print(y_train_pred)


# In[ ]:


print(confusion_matrix(np.argmax(y_test, axis=1), y_train_pred))


# In[ ]:




