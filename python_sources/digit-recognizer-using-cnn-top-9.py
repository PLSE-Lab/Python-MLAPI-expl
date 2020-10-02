#!/usr/bin/env python
# coding: utf-8

# ## Digit Recognizer
# 
# In this notebook, I used an ensemble of **Convolutional Neural Network** models based on the **LeNet-5** architecture and the architecture inspired from this [notebook](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist) to recognize the hand-written digits. I also used **Data Augmentation** or **Artificial Data Synthesis** technique to synthesize images in addition to the training data to boost the model's performance on the test data. 
# 
# Achieved an accuracy of **99.7%** (Leaderboard) on the test data by using the images in the Kaggle's training set and not on the entire MNIST dataset externally available.
# 
# Please **upvote** this notebook if you like the implementation and share your valuable feedback to improve.
# 
# You can find my other notebooks below:
# 
# * [Disaster Tweets Classification](https://www.kaggle.com/gauthampughazh/disaster-or-not-plotly-use-tfidf-h2o-ai-automl)
# * [House Sales Price Prediction](https://www.kaggle.com/gauthampughazh/house-sales-price-prediction-svr)
# * [Titanic Survival Classification](https://www.kaggle.com/gauthampughazh/titanic-survival-prediction-pandas-plotly-keras)
# * [Digit Recognition using KNN](https://www.kaggle.com/gauthampughazh/digit-recognition-using-knn)

# In[ ]:


import os
import json
import numpy as np # Linear algebra
import pandas as pd # For data manipulation
import matplotlib.pyplot as plt # For visualization
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold # For evaluation and hyperparameter tuning
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report # For evaluation
from scipy.ndimage import rotate, shift, zoom # For data augmentation
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from IPython.display import FileLink # For downloading the output file

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Data Exploration

# Loading the datasets into dataframes

# In[ ]:


train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
submission_df = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")


# Knowing about the features in the datasets

# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# Converting the train and test dataframes into numpy arrays

# In[ ]:


X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.values

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")


# Visualizing a digit from the training data as a 28 X 28 image

# In[ ]:


some_digit = X_train[40]

some_digit_image = some_digit.reshape(28, 28)
print(f"Label: {y_train[40]}")
plt.imshow(some_digit_image, cmap="binary")
plt.show()


# ### Data Augmentation

# Each image in the training set is 
# 
# * shifted down, up, left and right by one pixel
# * rotated clockwise and anti-clockwise 
# * clipped and zoomed at two different ranges
# 
# generating eight different images. The image is clipped before zooming to preserve the image size.

# In[ ]:


def shift_in_one_direction(image, direction):
    """
    Shifts an image by one pixel in the specified direction
    """
    if direction == "DOWN":
        image = shift(image, [1, 0])
    elif direction == "UP":
        image = shift(image, [-1, 0])
    elif direction == "LEFT":
        image = shift(image, [0, -1])
    else:
        image = shift(image, [0, 1])

    return image


def shift_in_all_directions(image):
    """
    Shifts an image in all the directions by one pixel
    """
    reshaped_image = image.reshape(28, 28)

    down_shifted_image = shift_in_one_direction(reshaped_image, "DOWN")
    up_shifted_image = shift_in_one_direction(reshaped_image, "UP")
    left_shifted_image = shift_in_one_direction(reshaped_image, "LEFT")
    right_shifted_image = shift_in_one_direction(reshaped_image, "RIGHT")

    return (down_shifted_image, up_shifted_image,
            left_shifted_image, right_shifted_image)


def rotate_in_all_directions(image, angle):
    """
    Rotates an image clockwise and anti-clockwise
    """
    reshaped_image = image.reshape(28, 28)
    
    rotated_images = (rotate(reshaped_image, angle, reshape=False),
                      rotate(reshaped_image, -angle, reshape=False))
    
    return rotated_images


def clipped_zoom(image, zoom_ranges):
    """
    Clips and zooms an image at the specified zooming ranges
    """
    reshaped_image = image.reshape(28, 28)
    
    h, w = reshaped_image.shape
    
    zoomed_images = []
    for zoom_range in zoom_ranges:
        zh = int(np.round(h / zoom_range))
        zw = int(np.round(w / zoom_range))
        top = (h - zh) // 2
        left = (w - zw) // 2
        
        zoomed_images.append(zoom(reshaped_image[top:top+zh, left:left+zw],
                                  zoom_range))
    
    return zoomed_images

def alter_image(image):
    """
    Alters an image by shifting, rotating, and zooming it
    """
    shifted_images = shift_in_all_directions(image)
    rotated_images = rotate_in_all_directions(image, 10)
    zoomed_images = clipped_zoom(image, [1.1, 1.2])
            
    return np.r_[shifted_images, rotated_images, zoomed_images]

X_train_add = np.apply_along_axis(alter_image, 1, X_train).reshape(-1, 784)
y_train_add = np.repeat(y_train, 8)

print(f"X_train_add shape: {X_train_add.shape}")
print(f"y_train_add shape: {y_train_add.shape}")


# Combining the original images and the synthesized images to form a new dataset

# In[ ]:


X_train_combined = np.r_[X_train, X_train_add]
y_train_combined = np.r_[y_train, y_train_add]

del X_train
del X_train_add
del y_train
del y_train_add

print(f"X_train_combined shape: {X_train_combined.shape}")
print(f"y_train_combined shape: {y_train_combined.shape}")


# ### Modelling

# Building a custom transformer to reshape the scaled images as required by the KerasClassifier

# In[ ]:


class ImageReshaper(BaseEstimator, TransformerMixin):
    """
    Reshapes the data to the shape required by the KerasClassifier
    """
    def __init__(self, shape):
        self.shape = shape
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.reshape(self.shape)


# Function to build a model based on LeNet-5 architecture

# In[ ]:


def build_lenet5_model():
    """
    Builds and returns the model based on LeNet-5 architecture
    """
    model = Sequential()
    # Adding layers to the model
    model.add(Conv2D(6, kernel_size=5, activation='relu',
                     input_shape=(28,28,1)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(16, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    model.add(Dense(400, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # Specifying the loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    
    return model


# Function to build a model based on a modified LeNet-5 architecture from the above mentioned notebook

# In[ ]:


def build_custom_lenet5_model():
    """
    Builds and returns the model based on a modified LeNet-5 architecture
    """
    model = Sequential()
    # Adding layers to the model
    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    
    # Specifying the loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    
    return model


# Using **StratifiedKFold** to ensure that the test data represents samples from all classes (digits) and for cross-validating the model. Using the classification report and confusion matrix to understand the model's performance on each fold.

# In[ ]:


stratified_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, indices in enumerate(stratified_fold.split(X_train_combined, y_train_combined)):
    # Creating datasets for training and testing the model 
    X_train_, y_train_ = X_train_combined[indices[0]], y_train_combined[indices[0]]
    X_test_, y_test_ = X_train_combined[indices[1]], y_train_combined[indices[1]]
    
    model_pipeline = Pipeline([
        ('min_max_scaler', MinMaxScaler()),
        ('image_reshaper', ImageReshaper(shape=(-1, 28, 28, 1))),
        ('model', KerasClassifier(build_lenet5_model, epochs=5, batch_size=32))
    ])
    
    model_pipeline.fit(X_train_, y_train_)
    predictions = model_pipeline.predict(X_test_)
    
    print(f"Classification report for Fold {fold + 1}:")
    print(classification_report(y_test_, predictions, digits=3), end="\n\n")
    
    print(f"Confusion Matrix for Fold {fold + 1}:")
    print(confusion_matrix(y_test_, predictions), end="\n\n")
    
    del X_train_
    del X_test_
    del y_train_
    del y_test_


# Fitting models to the combined dataset with custom pipelines

# In[ ]:


lenet5_model = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('image_reshaper', ImageReshaper(shape=(-1, 28, 28, 1))),
    ('model', KerasClassifier(build_lenet5_model, epochs=5, batch_size=32))
])

custom_lenet5_model = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('image_reshaper', ImageReshaper(shape=(-1, 28, 28, 1))),
    ('model', KerasClassifier(build_custom_lenet5_model, epochs=20, batch_size=32))
])


lenet5_model.fit(X_train_combined, y_train_combined)
# Getting the estimated probabilities for each class
lenet5_model_predictions = lenet5_model.predict_proba(X_test)

custom_lenet5_model.fit(X_train_combined, y_train_combined)
# Getting the estimated probabilities for each class
custom_lenet5_model_predictions = custom_lenet5_model.predict_proba(X_test)


# ### Result Generation

# Combining the model results

# In[ ]:


predictions = lenet5_model_predictions + custom_lenet5_model_predictions

predictions = np.argmax(predictions, axis=1)


# Generating the submission file

# In[ ]:


submission_df["Label"] = predictions
submission_df.to_csv('submissions.csv', index=False)
FileLink('submissions.csv')


# In[ ]:


submission_df.head()

