#!/usr/bin/env python
# coding: utf-8

# # Digit Recognizer (MNIST)

# In this notebook we use sparse categorical that doesn't need to transform the label into one hot.
# [The Dataset used in this Notebook](https://www.kaggle.com/c/digit-recognizer/data)

# ## Data Description: 
# 
# The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.
# 
# Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
# 
# The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.
# 
# Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).
# 
# For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.
# 
# Visually, if we omit the "pixel" prefix, the pixels make up the image like this:
# 
# > 000 001 002 003 ... 026 027 <br>
# 028 029 030 031 ... 054 055 <br>
# 056 057 058 059 ... 082 083 <br>
#  |   |   |   |  ...  |   |  <br>
# 728 729 730 731 ... 754 755 <br>
# 756 757 758 759 ... 782 783 
# 
# The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.
# 
# Your submission file should be in the following format: For each of the 28000 images in the test set, output a single line containing the ImageId and the digit you predict. For example, if you predict that the first image is of a 3, the second image is of a 7, and the third image is of a 8, then your submission file would look like:
# 
# > ImageId,Label <br>
# 1,3 <br>
# 2,7 <br>
# 3,8  <br>
# (27997 more lines)
# 
# The evaluation metric for this contest is the categorization accuracy, or the proportion of test images that are correctly classified. For example, a categorization accuracy of 0.97 indicates that you have correctly classified all but 3% of the images.

# # The Data

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


len(train), len(test)


# In[ ]:


train.columns


# In[ ]:


train['label'].unique()


# # Visualize the data

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


tr_sample = train.drop('label', axis=1).values.reshape(-1,28,28)[0]
ts_sample = test.values.reshape(-1,28,28)[0]


# In[ ]:


test['pixel345'].value_counts()


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16,8))

ax[0].imshow(tr_sample, cmap='gray')
ax[0].set(title='Train', xticks=[], yticks=[])

ax[1].imshow(ts_sample, cmap='gray')
ax[1].set(title='Test', xticks=[], yticks=[]);


# # Preprocessing

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# In[ ]:


ts_sample.min(), ts_sample.max()


# In[ ]:


x_test = test.values.reshape(-1,28,28,1)
x_test = x_test/255

x_train_full = train.drop('label', axis=1).values.reshape(-1,28,28,1)
y_train_full = train['label'].values

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)


# In[ ]:


batch_size=64

img_gen = ImageDataGenerator(rescale=1/255, 
                             rotation_range=30, 
                             zoom_range=.1,
                             shear_range=.1,
                             width_shift_range=.25,
                             height_shift_range=.25)

train_gen = img_gen.flow(x_train, y_train, 
                         batch_size=batch_size)

valid_gen = img_gen.flow(x_val, y_val, 
                         batch_size=batch_size,
                         shuffle=False)


# # Modelling

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min',restore_best_weights=True)

check_point = ModelCheckpoint('digit_reg_mnist_z.h5', monitor='val_accuracy', save_best_only=True)

lr_plateau = ReduceLROnPlateau(monitor='val_accuracy', 
                               patience=2,
                               factor=.2, 
                               min_lr=1e-6)


# In[ ]:


model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

# model.add(MaxPool2D(pool_size=(2,2)))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(Conv2D(128, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(Conv2D(256, kernel_size=(3,3), padding='same'))
model.add(BatchNormalization(momentum=.9, epsilon=1e-5))
model.add(Activation('relu'))

model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

model.summary()


# In[ ]:


model.fit(train_gen,
          epochs=100,
          steps_per_epoch=250,
          validation_data=valid_gen,
          callbacks=[lr_plateau, early_stop])


# # Evaluation

# In[ ]:


import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


eval_df = pd.DataFrame(model.history.history)
length = len(eval_df)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15,6))

eval_df[['loss','val_loss']].plot(ax=ax[0])
ax[0].set(title='Loss', xlabel='Epoch', xticks=range(0,length,2))

eval_df[['accuracy','val_accuracy']].plot(ax=ax[1])
ax[1].set(title='Accuracy', xlabel='Epoch', xticks=range(0,length,2));


# In[ ]:


pred = np.argmax(model.predict(valid_gen), axis=1)
pred


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(confusion_matrix(y_val, pred), annot=True, fmt='d', lw=.3, cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('True Label')


# In[ ]:


print(classification_report(y_val, pred))


# # Finishing

# In[ ]:


x_test.shape


# In[ ]:


lr_plateau = ReduceLROnPlateau(monitor='accuracy', 
                               patience=2,
                               factor=.2, 
                               min_lr=1e-6)

full_train_gen = img_gen.flow(x_train_full, y_train_full, 
                              batch_size=batch_size)


# In[ ]:


model.fit(full_train_gen,
          epochs=22,
          steps_per_epoch=250,
          callbacks=[lr_plateau])


# In[ ]:


real_pred = np.argmax(model.predict(x_test), axis=1) # Prediction on real test data


# In[ ]:


submit_df = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submit_df['Label'] = real_pred
submit_df.to_csv('submission.csv', index=False)

