#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files


# In[ ]:


#after this runs upload just the training data.csv. The testing-data is reserved for submission only

#this will not run in a kaggle notebook only in google colab
from google.colab import files
uploaded = files.upload()


# In[ ]:


#this with statement will iterate through each row of the csv file and seperate the image pixels and the associated label
#for each image, and then reshape them in to a 28,28 array to be passed into the model.

with open(filename) as training_file:
        reader = csv.reader(training_file, delimiter=',')    
        imgs = []
        labels = []

        next(reader, None)
        
        for row in reader:
            label = row[0]
            data = row[1:]
            img = np.array(data).reshape((28, 28))

            imgs.append(img)
            labels.append(label)

        images = np.array(imgs).astype(float)
        labels = np.array(labels).astype(float)
    return images, labels


#upload the csv and run get_data function to seperate out the images and labels
images, labels = get_data('hwtrain.csv')
training_images, training_labels = images, labels

#This will index the arrays to split them up into training and testing data
training_images, training_labels = images[:37800, :, :], labels[:37800]
testing_images, testing_labels = images[37800:, :, :], labels[37800:]

# In this section you will have to add another dimension to the data
# So, for example, if your array is (10000, 28, 28)
# You will need to make it (10000, 28, 28, 1)
# Hint: np.expand_dims

training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)


# In[ ]:


# Create an ImageDataGenerator and do Image Augmentation. This block will shrink the size of the pixel values in the array
#to speed up learning before doing various image augmentation to help with creating a more diverse training set. Do not run this
#on the validation data, it will not work!
train_datagen = ImageDataGenerator(
      rescale = 1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255.)
    
    


# In[ ]:


# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), padding = 'same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (5,5), padding = 'same', activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (1,1), activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation= 'relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.summary()
#After every conv layer I added batch normalization to help with controlling the means and variances of the activations 
#after the conv layer. This helps with speeding up learning for the network and also adds a slight regularization effect
#(which is not the reason for using it, just an added benefit)


# In[ ]:


# Compile Model. 
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

train_generator = train_datagen.flow(training_images, training_labels,
  batch_size=16
)

validation_generator = validation_datagen.flow(testing_images, testing_labels,
  batch_size=16
)

# Train the Model
history = model.fit(train_generator,
                          epochs=10,
                          validation_data=validation_generator, verbose = 1, validation_steps = 3)

model.evaluate(testing_images, testing_labels)


# In[ ]:


# Plot the chart for accuracy and loss on both training and validation

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(10,8))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure(figsize=(10,8))
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


#run this to upload the hwtest.csv and the sample_submission.csv
uploaded=files.upload()


# In[ ]:


import pandas as pd 

#this is for testing the network on the test set given by kaggle for the submission portion.
test_set = pd.read_csv('hwtest.csv')
submission = pd.read_csv('sample_submission.csv')

test_set = np.asarray(test_set).reshape([len(test_set), 28, 28, 1])
test_set = test_set/255

#predicting the data
predicted_value=model.predict(test_set)
classes=[0,1,2,3,4,5,6,7,8,9]
list1=[]
for index in range(0, len(predicted_value)):
    list1.append(classes[np.argmax(predicted_value[index])])
results= pd.DataFrame(list1, columns=['Label'])
    
#Conveting the output to a csv file 
submission['Label']=pd.DataFrame(results, columns=['Label'])
submission.to_csv('finalsubmission.csv', index= False)

