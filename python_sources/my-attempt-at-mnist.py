#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Basic import of libraries to support our analysis
import numpy as np 
import pandas as pd

# Visualization Libs
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import tensorflow as tf


# In[ ]:


# Check which version of TF is hosted 
print(tf.__version__)


# ### Exploring the Dataset
# Before we begin to visualize the dataset lets see the shape and size of the data we are working with.

# In[ ]:


train=pd.read_csv("../input/digit-recognizer/train.csv")
test=pd.read_csv("../input/digit-recognizer/test.csv")

print("Train Shape: {}".format(train.shape))
print("Test Shape: {}".format(test.shape))


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print("Train Nulls: {}".format(train.isna().any().sum()))
print("Test Nulls: {}".format(test.isna().any().sum()))


# Looks like we are working wth 42000 images in the training set and 28000 images in the test set. The test set is missing the label column. Interesting thing to note is the images should be 28 px x 28 x 1 channel (black and white) but the pixels were unrolled into 784 columns. We need to reshape the columns to 28x28x1 in order to visualize the image and also feed it into the neural network we will build. 

# ## Data Visualization
# 
# Lets explore what our data looks like. Lets see what each number.

# In[ ]:


# Training set
fig, axs = plt.subplots(2, 5, figsize=(16,6))
for i, ax  in zip(range(0,10), axs.flat):
    ax.imshow(train[train['label'] == i].drop(columns=['label']).iloc[0].values.reshape(28, 28), cmap='gray')


# In[ ]:


fig, axs = plt.subplots(2, 5, figsize=(16,6))
for i, ax  in zip(range(0,10), axs.flat):
    ax.imshow(test.iloc[i].values.reshape(28, 28), cmap='gray')


# We can see that there are 10 classes of numbers, from 0 to 9. They all look handwritten with varying density and curves. There does not seem to be any difference between the test set and train set.
# 
# The next step is to see the distribution of labels. We need to pay attention to any labels that are over represented or underrepresented in out dataset. A model predictions power decreases when it encounters any label that was underrepresented in train set

# In[ ]:


# Plot the distribution of each label
plt.figure(figsize=(24, 7))
plt.hist(train['label'], color='c', rwidth=0.5, align='mid')
plt.xlabel('Digits')
plt.ylabel('Frequency')
plt.title('Distribution of Labels')
plt.show()


# There doesn't seem to be any class over/under represented in the train set.
# 
# ## Data Preprocessing
# 
# Lets begin to setup the data so we can feed it into a neural network. We need to reshape X from 42000x28x28 to 42000x28x28x1. The extra dimension represents the black and white colour channel. If this was a coloured picture dataset then we would have 3 channels to work with. We are also going to downcast the dataset to variable types that take up less memory. This makes it easier to train larger Neural Networks (NN) since we are using up less memory.

# In[ ]:


# Downcasting all the values to save memory
y = train['label'].astype('int8')

# Downcast to float16 for every column except label
X = train.drop(columns=['label']).astype('float16').values

# Reshape the arrays so they are easier to visualize and input to NN
X = X.reshape(42000, 28,28, 1)


# We will create a new training and a validation set. The size of the validation set will be 20% of the training set which is the standard.

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)


# We are going to use the ImageDataGenerator class to load all out data. This class provides us the flexibility to apply many builtin data augmentation technqiues. But for now we are just going to divide all the pixels by 255 to normalize the values between 0 and 1.0 and see how the NN performs. Normalized values makes it easier for our NN to converge.

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create our image generator
train_image_generator = ImageDataGenerator(
    rescale=1./255
)

validation_image_generator = ImageDataGenerator(
    rescale=1./255
)

# Create instance of image generator attach to the dataset
train_image_gen = train_image_generator.flow(
    x=X_train, 
    y=y_train,
)

validation_image_gen = validation_image_generator.flow(
    x=X_validation, 
    y=y_validation,
)


# We are going to use a combination of Conv2D nets and MaxPooling2D to help determine what the most important features are.
# We are then going to feed the features to a Flatten() layer and then to a large dense layer. A dropout rate of 0.1 is used to lower the chances of overfitting. BatchNorm is also used to help the NN train more efficently. 

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.1),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.1),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation='softmax')
])


# We will train for 5 epochs and plot the loss and accuracy between the two datasets. 

# In[ ]:


tf.keras.backend.clear_session()  # For easy reset of notebook state.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_image_gen,
    epochs=5,
    validation_data=validation_image_gen
)


# In[ ]:


# Plot the loss and accuracy of the validation and training set
fig, axs = plt.subplots(1,2, figsize=(20,5))
axs[0].plot(history.history['loss'], label='Train')
axs[0].plot(history.history['val_loss'], label='Validation')
axs[0].set_title("Loss of Validation and Train")
axs[0].legend()

axs[1].plot(history.history['accuracy'], label='Train')
axs[1].plot(history.history['val_accuracy'], label='Validation')
axs[1].set_title("Accuracy of Validation and Train")
axs[1].legend()

fig.show()


# Juding by the loss and accuracy we can see that there doesnt seem to be much bias or variance between the two datasets. This is good meaning out NN does not over or underfit and in theory perform well in predicting the test set. Before we actually make predictions on our test set, lets visualize the results of the worst predictions and see which pictures the model has trouble predicting.

# In[ ]:


# Lets predict the values on the validation set
validation_predict = model.predict(X_validation)

# Create a dataframe to store the label and the confidence in their predictions
low_predictions = pd.DataFrame()
low_predictions['label'] = np.argmax(validation_predict, axis=1)
low_predictions['Confidence'] = np.max(validation_predict, axis=1)

low_index = low_predictions.sort_values(by=['Confidence'])[:10].index
low_labels = low_predictions.sort_values(by=['Confidence'])['label']

fig, axs = plt.subplots(2, 5, figsize=(24,9.5))
for i, low_label, ax  in zip(low_index,low_labels, axs.flat):
    image = X_validation[i].astype('float32').reshape(28, 28)
    ax.imshow(image, cmap='gray')
    ax.set_xlabel("True: {}".format(y_validation.iloc[i]))
    ax.set_title("Guessed:{} ".format(low_label))


# Looking at the predictions the model was not very confident in that aren't drawn that well. Images are not drawn well, or drawn at a slant are difficult for the model to classify since it most likely does not encounter these kind of images that often.
# 
# In order to improve our model lets recreate the ImageDataGenerator this time lets add rotations and some horizontal and vertical shift to stimulate the distortions. We will then retrain the network and evaluate the accuracy. Lets also run it for 20 epochs since we want to see if more training time leads to better results.

# In[ ]:


# Create our image generator
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=.1,
    height_shift_range=.1,
    zoom_range=0.1
)

validation_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=.1,
    height_shift_range=.1,
    zoom_range=0.1
)

# Create the dataset
train_image_gen = train_image_generator.flow(
    x=X_train, 
    y=y_train,
)

validation_image_gen = validation_image_generator.flow(
    x=X_validation, 
    y=y_validation,
)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.001,
    decay_steps=250,
    decay_rate=1,
    staircase=False
)

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)

tf.keras.backend.clear_session()  # For easy reset of notebook state.

model.compile(optimizer=get_optimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_image_gen,
    epochs=20,
    validation_data=validation_image_gen
)


# In[ ]:


# Plot the loss and accuracy of validation and train
fig, axs = plt.subplots(1,2, figsize=(20,5))
axs[0].plot(history.history['loss'], label='Train')
axs[0].plot(history.history['val_loss'], label='Validation')
axs[0].set_title("Loss of Validation and Train")
axs[0].legend()

axs[1].plot(history.history['accuracy'], label='Train')
axs[1].plot(history.history['val_accuracy'], label='Validation')
axs[1].set_title("Accuracy of Validation and Train")
axs[1].legend()

fig.show()


# Looks there is not much overfitting or underfitting. Lets make predictions on the test set and see the score.

# In[ ]:


X_test = test.astype('float16').values
X_test = X_test / 255.0
X_test = X_test.reshape(len(X_test), 28, 28, 1)


# In[ ]:


label_pred = model.predict_classes(X_test, verbose=0)

submission = pd.DataFrame()
submission['Label'] = label_pred
submission['ImageId'] = submission.index + 1
submission.to_csv('../working/output.csv', index=False)

