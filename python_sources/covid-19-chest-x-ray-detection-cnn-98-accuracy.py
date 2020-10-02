#!/usr/bin/env python
# coding: utf-8

# # **Stay safe and healthy.**

# i hope you all are safe. here i am going to make notebook of covid-19 prediction with the help of chesr x-ray.
# as you can see now in the world the corona virus is spreading more and more and due to that lots of people are dying.

# here, i have 3 folders in which train,test and original test set.train and test have sub folders called covid and normal so covid indicating that the x-ray of that person having corona virus and normal folder have images of x-ray that person do not have corona virus. and the last folder which is original test set which includes mix images of covid and normal people x-ray so by that folder's images we will predict by individual images to find out that the x-ray that image is having covid or not. 

# i have copied all the test folder's images covid and normal and put them in one folder called original test set so we will see how our model perform on that. so,test and original test set folders both have same images but i have also added 35 random images of covid and normal people's x-ray from internet for better understanding.  

# ***Coronavirus is a family of viruses that are named after their spiky crown. The novel coronavirus, also known as SARS-CoV-2, is a contagious respiratory virus that first reported in Wuhan, China. On 2/11/2020, the World Health Organization designated the name COVID-19 for the disease caused by the novel coronavirus. This notebook aims at exploring COVID-19 through data analysis and projections.***

# ![Coronavirus-CDC-645x645.jpg](attachment:Coronavirus-CDC-645x645.jpg)

# # Feel free to provide me suggestions for better accuarcy.

# **here i am going to use convolutional neural network and after this notebook i will publish another notebook in which i have same dataset but i will use transfer learning(vgg16,vgg19 and resnet50).**

# In[ ]:


# Importing the Keras libraries and packages
import tensorflow as tf

# Initialising the CNN
classifier = tf.keras.models.Sequential()

classifier.add(tf.keras.layers.Convolution2D(filters=32, kernel_size=3, padding="same", input_shape=(224, 224, 3),
                                             activation='relu'))

classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

classifier.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))


classifier.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))


classifier.add(tf.keras.layers.Convolution2D(filters=128, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

classifier.add(tf.keras.layers.Convolution2D(filters=256, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

classifier.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=3, padding="same", activation="relu"))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))


classifier.add(tf.keras.layers.Flatten())

classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.summary()


# In[ ]:



# Part 2 - Fitting the CNN to the images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('/kaggle/input/corona-chest-xray-prediction/Data/train',
                                                 target_size=(224,224),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('/kaggle/input/corona-chest-xray-prediction/Data/test',
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='binary')


# In[ ]:


history = classifier.fit(training_set,
                         steps_per_epoch=4,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=4)

classifier.save('my_model.h5')


# In[ ]:


# evaluation on test set
loaded_model = tf.keras.models.load_model('my_model.h5')
loaded_model.evaluate(test_set)


# In[ ]:


#plot accuracy and loss
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# **below code is use to predict only one image at a time** 

# **since our model's accuracy is 98% so some image will not show proper output.
# below i have plotted confusion matrix for better understanding.**

# In[ ]:


# for only one prediction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('/kaggle/input/corona-chest-xray-prediction/Data/test/Normal/IM-0283-0001.jpeg',target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Normal'
else:
    prediction = 'Covid'
print(prediction)


# In[ ]:


# for only one prediction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('/kaggle/input/corona-chest-xray-prediction/Data/train/Covid/16654_1_1.png',target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Normal'
else:
    prediction = 'Covid'
print(prediction)


# In[ ]:


# for whole test set  {'Covid': 0, 'Normal': 1}
from keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

# image folder
folder_path = '../input/corona-chest-xray-prediction/original test set'
# path to model
model_path = './my_model.h5'

# load the trained model
classifier = tf.keras.models.load_model('./my_model.h5')
classifier.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

# load all images into a list
images = []
for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)
classes = classifier.predict_classes(images, batch_size=10)
print(classes)


# **as we can see the confusion matrix only 7 image prediction is failed and 93 images are correct that's seems like our model is perfectly fitted on the dataset.**

# In[ ]:


# plot confusion metrix
y_pred = []
y_test = []
import os

for i in os.listdir("../input/corona-chest-xray-prediction/Data/test/Normal"):
    img = image.load_img("../input/corona-chest-xray-prediction/Data/test/Normal/" + i, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    p = classifier.predict_classes(img)
    y_test.append(p[0, 0])
    y_pred.append(1)

for i in os.listdir("../input/corona-chest-xray-prediction/Data/test/Covid"):
    img = image.load_img("../input/corona-chest-xray-prediction/Data/test/Covid/" + i, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    p = classifier.predict_classes(img)
    y_test.append(p[0, 0])
    y_pred.append(0)

y_pred = np.array(y_pred)
y_test = np.array(y_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)
import seaborn as sns

sns.heatmap(cm, cmap="plasma", annot=True)


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_pred, y_test))


# # **please do upvote if you like**
