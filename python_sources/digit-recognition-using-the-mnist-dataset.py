#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing MNIST dataset
df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


y_train = df_train['label']
y_train


# In[ ]:


x_train = df_train.drop('label', axis=1)
x_train = x_train.values
x_train


# In[ ]:


x_train.shape


# In[ ]:


# Reshaping array to visualize the images
plt.imshow(x_train[0].reshape(28,28), cmap='Greys')
plt.show()


# In[ ]:


x_train = x_train.reshape(42000,28,28)


# In[ ]:


x_train.shape


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=19)


# In[ ]:


x_train.shape


# In[ ]:


x_valid.shape


# **Note:** The targets in y_train and y_test are simply numerical values. We need to perform one-hot encoding for our model as this is a multi-class classification problem.

# In[ ]:


from tensorflow.keras.utils import to_categorical
y_cat_valid = to_categorical(y_valid, num_classes=10)
y_cat_train = to_categorical(y_train, num_classes=10)


# In[ ]:


# Sampling first image visually
first_image = x_train[0]
plt.imshow(first_image, cmap='Greys')
plt.show()


# In[ ]:


print('Actual image:')
print(y_train[0])


# In[ ]:


first_image.max()


# In[ ]:


first_image.min()


# **Note:** We can simply divide the training and test data by the maximum value of 255 to scale the data.

# In[ ]:


# Scaling values on train and test data
x_train_scaled = x_train/255
x_valid_scaled = x_valid/255


# In[ ]:


# Rechecking image after scaling
plt.imshow(x_train_scaled[0], cmap='Greys')
plt.show()


# In[ ]:


# Reshaping final training and testing data to prep for training
# (batch_size, width, height, color_channels)
x_train_final = x_train_scaled.reshape(37800,28,28,1)
x_valid_final = x_valid_scaled.reshape(4200,28,28,1)


# # Building the CNN Model
# - This models consists of only one convolution layer followed by a pooling layer to speed up computation while still retaining the meaningful information of the data.
# - The image was flattened and then fed into a dense layer followed by the output layer.
# - A softmax activation function was used for the output layer as this is a multi-class classification problem.

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


# In[ ]:


# Building CNN Model

# Instantiate model
model = Sequential()

# Convolution layer 1
model.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))
# Convolution layer 2
model.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))
# Pooling layer (selected half of kernel_size)
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Convolution layer 3
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))
# Convolution layer 4
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), padding='Same', activation='relu'))
# Pooling layer (selected half of kernel_size)
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Flattening image
model.add(Flatten())
# Dense layer
model.add(Dense(256, activation='relu'))

# Output layer
model.add(Dense(10, activation='softmax'))

# Compiling model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', patience=2)


# # Data Augmentation

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


image_gen = ImageDataGenerator(rotation_range=10,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=0.1,
                              fill_mode='nearest')


# In[ ]:


train_image_gen = image_gen.fit(x_train_final)


# In[ ]:


train_image_gen


# In[ ]:


model.fit_generator(image_gen.flow(x_train_final, y_cat_train), epochs=10, validation_data=(x_valid_final, y_cat_valid), callbacks=[early_stop])


# # Model Evaluation
# - The loss of the training and validation sets were plotted to ensure we are not overfitting our model
# - Accuracy of the training and validation data were also plotted
# - Overall accuracy was obtained from a classification report and confusion matrix

# In[ ]:


metrics = pd.DataFrame(model.history.history)
metrics


# In[ ]:


metrics[['loss', 'val_loss']].plot()
plt.show()


# In[ ]:


metrics[['accuracy', 'val_accuracy']].plot()
plt.show()


# In[ ]:


model.evaluate(x_valid_final, y_cat_valid, verbose=0)


# In[ ]:


y_pred = model.predict_classes(x_valid_final)
y_pred


# In[ ]:


# Model evaluation
from sklearn.metrics import classification_report, confusion_matrix
print('Classification Report:')
print(classification_report(y_valid, y_pred))
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_valid, y_pred))


# Our model performs quite well based on the above evaluation, resulting in an overall accuracy of 99%.

# # Sample Predictions
# - Random samples from the testing set were selected and our model was used to predict each sample target.
# - The actual image is displayed followed by the prediction.

# In[ ]:


np.random.seed(19)
random_selection = np.random.randint(0, 4201, size=1)
random_sample = x_valid_final[random_selection]
plt.imshow(random_sample.reshape(28,28), cmap='Greys')
plt.show()


# In[ ]:


print('Prediction:')
print(model.predict_classes(random_sample.reshape(1,28,28,1))[0])


# In[ ]:


np.random.seed(20)
random_selection_2 = np.random.randint(0, 4201, size=1)
random_sample_2 = x_valid_final[random_selection_2]
plt.imshow(random_sample_2.reshape(28,28), cmap='Greys')
plt.show()


# In[ ]:


print('Prediction:')
print(model.predict_classes(random_sample_2.reshape(1,28,28,1))[0])


# In[ ]:


np.random.seed(22)
random_selection_3 = np.random.randint(0, 4201, size=1)
random_sample_3 = x_valid_final[random_selection_3]
plt.imshow(random_sample_3.reshape(28,28), cmap='Greys')
plt.show()


# In[ ]:


print('Prediction:')
print(model.predict_classes(random_sample_3.reshape(1,28,28,1))[0])


# # Generating Predictions from Test Data

# In[ ]:


# Reshaping test data
x_test = df_test.values
x_test = x_test.reshape(28000,28,28)
x_test.shape


# In[ ]:


# Scaling test data
x_test_scaled = x_test/255


# In[ ]:


# Generating preditions
test_predictions = model.predict_classes(x_test_scaled.reshape(28000,28,28,1))


# In[ ]:


test_predictions


# In[ ]:


# Save test predictions to file
output = pd.DataFrame({'ImageId': df_test.index + 1,
                       'Label': test_predictions})
output.to_csv('submission.csv', index=False)

