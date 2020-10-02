#!/usr/bin/env python
# coding: utf-8

# This Notebook is an attempt to apply transfer Learning on Plant Village Dataset problem , it uses same kernel as **Tairu Oluwafemi Emmanuel.**
# Only Exception is that here I have used Transfer Learning using Inception Net which is initialized with  ImageNet weights , also Fine Tuning is done in later layers in order to ensure that the network solves our problem.

# In[ ]:


import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[ ]:


EPOCHS = 25
#INIT_LR = 0.01
BS = 100
default_image_size = tuple((256, 256))
image_size = 0
directory_root = '../input/plantvillage/'
width=256
height=256
depth=3


# In[ ]:


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


# In[ ]:


image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :
            # remove .DS_Store from list
            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")


# In[ ]:


image_size = len(image_list)


# In[ ]:


label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)


# In[ ]:


print(label_binarizer.classes_)


# In[ ]:


np_image_list = np.array(image_list, dtype=np.float16) / 225.0


# In[ ]:


print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42)


# In[ ]:


aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")


# In[ ]:


from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications import inception_v3
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


# In[ ]:


base_model= inception_v3.InceptionV3(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
#we add dense layers so that the model can learn more complex functions and classify for better results.

x = Dense(1024, activation='relu')(x)

preds=Dense(n_classes,activation='softmax')(x) #final layer with softmax activation


# In[ ]:


model=Model(inputs=base_model.input,outputs=preds)


# In[ ]:


model.summary()


# In[ ]:


for layer in model.layers[:249]:
    layer.trainable=False
for layer in model.layers[249:]:
    layer.trainable=True


# In[ ]:


# distribution
model.compile(loss="categorical_crossentropy", optimizer='Adam',metrics=["accuracy"])
# train the network
print("[INFO] training network...")


# In[ ]:


history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )


# Training for More Epochs might give better results!

# In[ ]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


# In[ ]:


print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")


# 60 % accuracy using transfer learning and fine tuning in later layers , to improve these results definitely need to play with parameters.

# **References**
# https://towardsdatascience.com/plant-ai-plant-disease-detection-using-convolutional-neural-network-9b58a96f2289
# https://www.kaggle.com/emmarex/plant-disease-detection-using-keras
# https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/
# https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e
# https://towardsdatascience.com/transfer-learning-946518f95666
# https://github.com/kofmangregory/CrowdAI-Plant-Village-Model/blob/master/Plant%20Diseases.ipynb

# Any Suggestion to improve the test accuracy will be highly appreciated

# In[ ]:




