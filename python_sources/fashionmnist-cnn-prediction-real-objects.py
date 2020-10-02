#!/usr/bin/env python
# coding: utf-8

# # **Data Description**
# * This dataset provided by zalandoresearch.
# * The dataset consists of fashion objects' grayscale images and converted into numbers.
# * The training dataset has 785 columns and 60000 rows and test dataset has same number of columns and 10000 rows.
# * First column is labels that provide class information.
# * The rest of the 784 columns contain pixel values. 
# * The pictures are suppose to be 28x28 resulation. So, it is gonna be reshaped into this resolution.
# 
# 
# 

# # The Images Classes
# * 0 - T-shirt/top 
# * 1 - Trouser
# * 2 - Pullover
# * 3 - Dress
# * 4 - Coat
# * 5 - Sandal
# * 6 - Shirt
# * 7 - Sneaker
# * 8 - Bag
# * 9 - Ankle boot

# In[ ]:


# Importing the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt 
from keras.utils import to_categorical           # Library for One Hot Encoding
import keras.preprocessing.image as img


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # **Import the Dataset**
# 
# Although Kaggle has already provided the data, the data can be loaded by tensorflow.

# In[ ]:


df_training = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
df_testing = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

# Second way to load the data
# fashion_mnist_data = tf.keras.datasets.fashion_mnist_data           
# (training_images, training_labels), (testing_images, testing_labels) = fashion_mnist_data.load_data()


# # Prepraing the Data
# The data distinguished from label and converted into numpy array.
# 
# Each row reshaped from vector(784 pixels) to 28x28 resolution and reformatted as 'float32'.
# 
# Labels have converted into vectors by One Hot Encoding.

# In[ ]:


training_images = np.array(df_training.iloc[0:,1:])
training_images = training_images.reshape(len(training_images), 28,28)        # Reshaping the pictures
training_images = training_images.astype('float32')
training_labels = np.array(df_training.iloc[:,0])
training_labels = to_categorical(training_labels)                # One hot encoded for labels

testing_images = np.array(df_testing.iloc[0:,1:])
testing_images = testing_images.reshape(len(testing_images), 28,28)
testing_images = testing_images.astype('float32')
testing_labels = df_testing.iloc[:,0]

test_for_pred = testing_images              # For testing we copied the testing images and labels
test_labels_for_pred = testing_labels
testing_labels = to_categorical(testing_labels)


# In[ ]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[ ]:


training_images.shape


# # Example of Images

# In[ ]:


fig, ax = plt.subplots(10,5, figsize = (10,25))
for i,c_name in enumerate(class_names):
    temp = df_training[df_training.iloc[:,0] == i].head(5)
    for k in range(5):
        ax[i,k].imshow(np.array(temp.iloc[k,1:]).reshape(28,28))
        ax[i,k].axis('off')
        ax[i,k].grid(False)
        ax[i,k].title.set_text(str(c_name)) 


# The 28x28 pixels array and images representation.

# In[ ]:


pixel_img = img.load_img('../input/pixel-pictures/training_images1.PNG')
np.set_printoptions(linewidth = 200)
fig, ax = plt.subplots(1,2, figsize = (20,8))
ax[0].imshow(training_images[1], aspect = 'auto')
ax[1].imshow(pixel_img,aspect = 'auto')


# The one more dimension should be added for color. Since the pictures are grayscale, 3rd dimension depth is 1. If the images were colorful, depth would be 3.

# In[ ]:


training_images = np.expand_dims(training_images, axis = 3)
testing_images = np.expand_dims(testing_images, axis = 3)

training_images = training_images / 255.0
testing_images = testing_images / 255.0


# In[ ]:


print(training_images.shape)
print(training_labels.shape)


# # Model 1 Training with Image Augmentation

# In[ ]:


# Image Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_datagen = ImageDataGenerator(zoom_range=0.1,
                                      shear_range = 0.1,
                                      rotation_range = 0.1,
                                      horizontal_flip=True,
                                      fill_mode = 'nearest')

training_datagen.fit(training_images)


# In[ ]:


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation='relu', padding= 'same', input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.20),
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu', padding= 'same'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(10 , activation = 'softmax')])

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[ ]:


history = model.fit(training_datagen.flow(training_images, training_labels), verbose = 1, epochs = 30, batch_size = 50, steps_per_epoch= len(training_images)/ 50, validation_data= (testing_images, testing_labels))


# # Model Evaluation

# In[ ]:


model.summary()


# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(30)

plt.plot(epochs, acc, 'r', label = 'training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'validation accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs , val_loss, 'b', label = 'Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()


# In[ ]:


test_for_pred = np.expand_dims(test_for_pred, axis = 3)
test_predictions = model.predict_classes(test_for_pred)

from sklearn.metrics import classification_report
print(classification_report(test_labels_for_pred, test_predictions, target_names = class_names))


# The name list of real object images

# In[ ]:


pic_names = os.listdir('../input/test-object-pics')   
print(pic_names[0:2])


# # Loading the Real Pictures
# * The pictures found from random webpages and loaded.
# * These pictures are loaded by keras image method.
# * It also helps to convert real colorful different images to grayscale and resize them into 28x28 resolution. 

# In[ ]:


real_images = []
fig, ax = plt.subplots(6,6, figsize = (15,15))
i = 0
k = 0
for f in pic_names:
    path = '../input/test-object-pics/'
    image = img.load_img(path + f)     
    if i>5:
        k+=1
        i=0
    ax[i,k].imshow(image)
    ax[i,k].grid(False)
    ax[i,k].axis('off')
    i+=1


# # Real pictures Prediction 1
# * Converting real colorful different images to grayscale and resizing them into 28x28 resolution. 

# In[ ]:


import keras.preprocessing.image as img
images = []
for f in pic_names:
    path = '../input/test-object-pics/'
    image = img.load_img(path + f, grayscale=True, target_size=(28,28))      # Convert into grayscle and 28x28
    x = img.img_to_array(image)
    images.append(x.reshape(28,28))


# The pictures are predicted as is, but the results were disaster. Almost all of them were predicted as a bag.

# In[ ]:


obj_images = np.expand_dims(images, axis = 3)
obj_images = obj_images.astype('float32')
obj_images = obj_images / 255.0
predicted_images = model.predict(obj_images)

fig, ax = plt.subplots(6,6, figsize = (15,15))
pic = 0

for i in range(6):
  for k in range(6):
    ax[i,k].imshow(images[pic])
    #ax[i,k].add_subplot(gs[i,k])
    ax[i,k].axis('off')
    ax[i,k].grid(False)
    file_name = pic_names[pic].split('.')[0]
    ax[i,k].title.set_text(f'Image file: {file_name} \n predicted: {class_names[np.argmax(predicted_images[pic])]}')
    pic +=1 
plt.tight_layout()
plt.show()


# In[ ]:


true_pred = 0
for i,img in enumerate(predicted_images):
    pred = class_names[np.argmax(img)]
    pred = list(pred)[0].lower() + list(pred)[1].lower()
    out = list(pic_names[i])[0].lower() + list(pic_names[i])[1].lower()
    if pred == out:
        true_pred += 1
print(f'{round((true_pred/len(predicted_images))*100,2)} of the real pictures are predicted correctly.')


# After I looked the pictures closely, I realized that all the training pictures has black background. Howeveer, all real pictures in the fashion sector have white background.
# 
# So, all background pixel values are around 250 instead of 0.

# In[ ]:


#The real pictures' background is converted to 0.
for pic in range(len(images)):
  for i in range(28):
    for k in range(28):
      if images[pic][i][k] >= 235:
        images[pic][i][k] = 0
      else:
        images[pic][i][k] =images[pic][i][k]


# In[ ]:


import keras.preprocessing.image as img
pixel_img2 = img.load_img('../input/pixel-pictures/test_image_values.PNG')
converted_pixel_img2 = img.load_img('../input/pixel-pictures/converted_test_image_values.PNG')

fig, ax = plt.subplots(1,2, figsize = (20,8))
ax[0].imshow(pixel_img2,aspect = 'auto')
ax[1].imshow(converted_pixel_img2, aspect = 'auto')


# # Real Pictures Prediction 2
# The prediction after pictures' background converted.

# In[ ]:


import matplotlib.gridspec as gridspec
obj_images = np.expand_dims(images, axis = 3)
obj_images = obj_images.astype('float32')
obj_images = obj_images / 255.0
predicted_images1 = model.predict(obj_images)

fig, ax = plt.subplots(6,6, figsize = (15,15))
pic = 0
for i in range(6):
  for k in range(6):
    ax[i,k].imshow(images[pic])
    ax[i,k].axis('off')
    ax[i,k].grid(False)
    file_name = pic_names[pic].split('.')[0]
    ax[i,k].title.set_text(f'Image file: {file_name} \n predicted: {class_names[np.argmax(predicted_images1[pic])]}')
    pic +=1 
plt.tight_layout()
plt.show()


# In[ ]:


# Correct prediction rate
true_pred = 0
for i,img in enumerate(predicted_images1):
    pred = class_names[np.argmax(img)]
    pred = list(pred)[0].lower() + list(pred)[1].lower()
    out = list(pic_names[i])[0].lower() + list(pic_names[i])[1].lower()
    if pred == out:
        true_pred += 1
print(f'{round((true_pred/len(predicted_images1))*100,2)} of the real pictures are predicted correctly.')
    


# In[ ]:


pic = 0
fig, ax = plt.subplots(4,4, figsize =(30,15))
for i in range(4):
  for k in range(4):
    if k % 2 == 0:
      ax[i,k].imshow(images[pic])
      ax[i,k].axis('off')
      ax[i,k].grid(False)
    elif k % 2 == 1:
      ax[i,k].bar(class_names, predicted_images1[pic-1]* 100 )
      ax[i,k].tick_params(rotation =45)
      for a,p in enumerate(ax[i,k].patches):
        if p.get_height() > 12:
            
            ax[i,k].annotate(f"{class_names[a]} - "+format( p.get_height(),'.2f') + "%", (p.get_x() + p.get_width() / 2., p.get_height()), bbox=dict(boxstyle="round", alpha=0.2),size = 20, ha = 'center', va = 'bottom', xytext = (15,15), textcoords = 'offset points')
    pic +=1
    
plt.tight_layout()


#  # Model 2 Training without Img Augmentation

# In[ ]:


model2 = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation='relu', padding= 'same', input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.20),
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu', padding= 'same'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation = 'relu'),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Dense(10 , activation = 'softmax')])

model2.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model2.fit(training_images, training_labels, verbose = 1, epochs = 30, batch_size = 50, steps_per_epoch= len(training_images)/ 50, validation_data= (testing_images, testing_labels))


# # Model Comparasion
# The overall accuracy is higher in the model without image augmentation.
# 
# However, model with augmentation predict the real objects better because the capability of the distinguishing classes improved for similar products with image augmentation.

# In[ ]:


test_predictions = model.predict_classes(test_for_pred)
from sklearn.metrics import classification_report
print('Model 1 with augmentation\n')
print(classification_report(test_labels_for_pred, test_predictions, target_names = class_names))

test_predictions2 = model2.predict_classes(test_for_pred)
from sklearn.metrics import classification_report
print('Model 2 without augmentation\n')
print(classification_report(test_labels_for_pred, test_predictions2, target_names = class_names))


# In[ ]:


import matplotlib.gridspec as gridspec
obj_images = np.expand_dims(images, axis = 3)
obj_images = obj_images.astype('float32')
obj_images = obj_images / 255.0
predicted_images2 = model2.predict(obj_images)

fig, ax = plt.subplots(6,6, figsize = (15,15))
pic = 0

for i in range(6):
  for k in range(6):
    ax[i,k].imshow(images[pic])
    #ax[i,k].add_subplot(gs[i,k])
    ax[i,k].axis('off')
    ax[i,k].grid(False)
    file_name = pic_names[pic].split('.')[0]
    ax[i,k].title.set_text(f'Image file: {file_name} \n predicted: {class_names[np.argmax(predicted_images2[pic])]}')
    pic +=1 
plt.tight_layout()
plt.show()


# In[ ]:


true_pred = 0
for i,img in enumerate(predicted_images2):
    pred = class_names[np.argmax(img)]
    pred = list(pred)[0].lower() + list(pred)[1].lower()
    out = list(pic_names[i])[0].lower() + list(pic_names[i])[1].lower()
    if pred == out:
        true_pred += 1
print(f'{round((true_pred/len(predicted_images2))*100,2)} of the real pictures are predicted correctly.')
    


# In[ ]:


pic = 0
fig, ax = plt.subplots(4,4, figsize =(30,15))
for i in range(4):
  for k in range(4):
    if k % 2 == 0:
      ax[i,k].imshow(images[pic])
      ax[i,k].axis('off')
      ax[i,k].grid(False)
    elif k % 2 == 1:
      ax[i,k].bar(class_names, predicted_images2[pic-1]* 100 )
      ax[i,k].tick_params(rotation =45)
      for a,p in enumerate(ax[i,k].patches):

        if p.get_height() > 12:
            
            ax[i,k].annotate(f"{class_names[a]} - "+format( p.get_height(),'.2f') + "%", (p.get_x() + p.get_width() / 2., p.get_height()), bbox=dict(boxstyle="round", alpha=0.2),size = 20, ha = 'left', va = 'bottom', xytext = (15,-5), textcoords = 'offset points')
    pic +=1
    
plt.tight_layout()


# # Conclusion
# * The fashion dataset pictures have black background, so the model is not able to predict real objects.
# * Just around 10% of the real pictures are predicted correctly the pictures because of white bacground of pictures.
# 
# After processed the pictures:
# 
# * The dataset has trained two times with img augmentation and without.
# * Even though the model without img augmentation performs better, in the real images dataset 
# * Every time Model 1 (with img augmentation) is performed about 10% better than model 2 (without img augmentation) in real images dataset.
# * Therefore, model with augmentation predicts the real objects better because the capability of the distinguishing classes improved for similar products thanks to image augmentation.
