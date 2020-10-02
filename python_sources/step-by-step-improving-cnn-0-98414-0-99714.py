#!/usr/bin/env python
# coding: utf-8

# # Step-by-Step: Improving CNN (0.98414 -> 0.99714)
# I created this Notebook to share my experience how can you get better performance from a Deep-Learning model. 
# 
# 
# * **1. Overview**
# * **2. Data preparation**
# * **3. Models**
#     * 3.1 Simple CNN (cnn_1)
#     * 3.2 Simple CNN (cnn_2)
#     * 3.3 Simple CNN (cnn_3)
#     * 3.4 Simple CNN (cnn_4)
#     * 3.5 Simple CNN (cnn_5)
#     * 3.6 Simple CNN (cnn_6)
#     * 3.7 Simple CNN (cnn_7)
#     * 3.8 CNN with data augmentation (data_aug_1)
#     * 3.9 CNN with data augmentation (data_aug_2)
#     * 3.10 Optimizing Learning Rate (lr_1)
#     * 3.11 Optimizing Learning Rate (lr_2)
#     * 3.12 Final model (final_1)
# * **4. Further Improvements**
# * **5. Resources**
# 

# # 1. Overview
# 
# 
# | **#** | **Git tag** | **Epochs**  |  **Validation Accuracy** | **Kaggle Score** | **Training Time** |
# | ----- :|----------------------- |:---------------:| -------------------------------:| ---------------------:| ----------------------:|
# | 1        | cnn_1                        | 20                  | 0.983714                               | 0.98414                 | 1 minute                  |
# | 1        | cnn_1                        | 100                | 0.987048                              | 0.98671                  | 5 minutes               |
# | 2       | cnn_2                       | 20                  | 0.983619                               | 0.98457                 | 2 minutes               |
# | 2       | cnn_2                       | 100                | 0.986381                               | 0.98628                 | 8 minutes               |
# | 3       | cnn_3                       | 20                  | 0.989333                              | 0.98614                  | 1.5 minutes            |
# | 3       | cnn_3                       | 100                | 0.988476                               | 0.98824                 | 7 minutes              |
# | 4       | cnn_4                       | 20                  | 0.990190                              | 0.98842                 | 2 minutes               |
# | 4       | cnn_4                       | 100                | 0.991905                               | 0.99071                 | 8.5 minutes            |
# | 5       | cnn_5                       | 20                  | 0.991143                               | 0.99100                 | 1 minute               |
# | 5       | cnn_5                       | 100                | 0.992286                               | 0.99214                | 5 minutes              |
# | 6       | cnn_6                       | 20                  | 0.991619                                | 0.99142                 | 1 minute               |
# | 6       | cnn_6                       | 100                | 0.992762                               | 0.99171                 | 5 minutes              |
# | 7       | cnn_7                       | 20                   | 0.993810                               | 0.99314                 | 1.5 minutes              |
# | 7       | cnn_7                       | 100                 | 0.993810                               | 0.99371                  | 7 minutes              |
# | 8       | data_aug_1              | 20                   | 0.993143                               | 0.99085               | 2 minutes               |
# | 8       | data_aug_1              | 100                 | 0.994762                               | 0.99400                 | 9 minutes              |
# | 8       | data_aug_1              | 800                | 0.994762                               | 0.99542                 | 1 hour 15 minutes     |
# | 9       | data_aug_2              | 20                   | 0.993016                               | 0.99414                | 2.5 minutes               |
# | 9       | data_aug_2              | 100                 | 0.995238                               | 0.99428                | 10 minutes              |
# | 9       | data_aug_2             | 800                | 0.995714                              | 0.99457                 | 1 hour 20 minutes     |
# | 10     | lr_1                             | 100                 | 0.996032                               | 0.99557                | 15 minutes              |
# | 10     | lr_1                             | 400                 | 0.996032                               | 0.99685                | 57 minutes              |
# | 11     | lr_2                             | 100                 | 0.996190                               | 0.99571                | 12 minutes              |
# | 11     | lr_2                             | 400                 | 0.996667                               | 0.99657                | 42 minutes              |
# | 12     | final_1                        | 100                 | 0.995556                               | 0.99628                | 14 minutes              |
# | 12     | final_1                       | 250                 | 0.996667                               | **0.99714**                | 35 minutes              |
# 
# You can find the full source code [here](https://github.com/pestipeti/KaggleDigitRecognizer).
# 
# 

# # 2. Data preparation
# In this Notebook, I'd like to focus on the differences between the versions of the model. I won't give you many details about the preprocessing. If you have any questions, leave me a comment, and I will try to answer.
# 
# ```python
# train = pd.read_csv(FOLDER_INPUT + '/train.csv')
# test = pd.read_csv(FOLDER_INPUT + '/test.csv')
# 
# features_train = train.iloc[:, 1:].values
# features_test = test.values
# labels_train = train.iloc[:, 0].values
# 
# features_train = features_train.reshape(len(features_train), IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)
# features_test = features_test.reshape(len(features_test), IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)
# 
# features_train = features_train.astype('float32')
# features_test = features_test.astype('float32')
# 
# features_train /= 255.0
# features_test /= 255.0
# 
# # Convert the labels.
# labels_train = to_categorical(labels_train, num_classes=10)
# 
# # Splitting the training set to training and validation subset
# features_train, features_validation, labels_train, labels_validation = train_test_split(
#     features_train, labels_train, test_size=0.25, random_state=0)
# ```

# # 3. Models
# I used the Keras Sequential API for my models. You can find more details about the API [here](https://keras.io/models/sequential/#the-sequential-model-api).

# ## 3.1. Simple CNN (cnn_1)
# My first model was a very simple Convolutional Neural Network. I will use this model as a starting point, and I will try to improve its accuracy.
# 
# ```python
# from keras.models import Sequential
# from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
#         
# km = Sequential()
# 
# # Convolutional layer with 32 filters
# km.add(Convolution2D(32, (3, 3), input_shape=input_shape, activation='relu'))
# # Downsampling
# km.add(MaxPooling2D(pool_size=(2, 2)))
# # Prevent the model from overfitting
# km.add(Dropout(0.25))
# 
# # Convert the feature maps to a single vector
# km.add(Flatten())
# # Add a fully-connected layer
# km.add(Dense(units=64, activation='relu'))
# km.add(Dropout(0.25))
# 
# # And the output layer with 10 possible outcome.
# km.add(Dense(units=10, activation='softmax'))
# 
# # Compile the model using Keras's build-in classes.
# km.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# ```
# 

# ## 3.2. Simple CNN (cnn_2)
# Let's see what happen if we double the number of the filters in the convolutional (Conv2D) layer.
# 
# ```diff
#         km = Sequential()
# 
# -        km.add(Convolution2D(32, (3, 3), input_shape=input_shape, activation='relu'))
# +        km.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
#         km.add(MaxPooling2D(pool_size=(2, 2)))
#         km.add(Dropout(0.25))
# 
# ```
# ### Results
# 
# With 20 iterations the final accuracy is a bit worse (-0.01%) than the previous version. The Kaggle Score improved a bit (+0.04%), but it is not significant. I would say that this modification is not an improvement. Let's try something else.
# 
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 20                 | cnn_1             | 0.983714     | 0.983619        | -0.01%| 0.98414            | 0.98457             | **+0.04%**|
# | 100               | cnn_1             | 0.987048     | 0.986381        | **+0.07%**| 0.98671   | 0.986280          | -0.04%|

# ## 3.3. Simple CNN (cnn_3)
# I increased the size of the kernel from **(3, 3)** to **(5, 5)**.
# 
# ```diff
#         km = Sequential()
# 
# -        km.add(Convolution2D(32, (3, 3), input_shape=input_shape, activation='relu'))
# +        km.add(Convolution2D(32, (5, 5), input_shape=input_shape, activation='relu'))
#         km.add(MaxPooling2D(pool_size=(2, 2)))
#         km.add(Dropout(0.25))
# ```
# ### Results
# Better. Both accuracy and score improved.
# 
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 20                 | cnn_1             | 0.983714     | 0.989333        | **+0.57%** | 0.98414  | 0.98614             | **+0.20%**|
# | 100               | cnn_1             | 0.987048     | 0.988476        | **+0.14%**| 0.98671   | 0.98824            | **+0.16%**|

# ## 3.4 Simple CNN (cnn_4)
# In this version I added an extra convolutional (Conv2D) layer (the MaxPooling and the Dropout layers too).
# 
# ```diff
#         km.add(Convolution2D(32, (5, 5), input_shape=input_shape, activation='relu'))
#         km.add(MaxPooling2D(pool_size=(2, 2)))
#         km.add(Dropout(0.25))
#         
# +        km.add(Convolution2D(16, (3, 3), input_shape=input_shape, activation='relu'))
# +        km.add(MaxPooling2D(pool_size=(2, 2)))
# +        km.add(Dropout(0.25))
# 
#         km.add(Flatten())
# ```
# 
# ### Results
# 
# 
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 20                 | cnn_3             | 0.989333    | 0.990190        | **+0.09%** | 0.98614  | 0.98842           | **+0.23%**|
# | 100               | cnn_3             | 0.988476     | 0.991905        | **+0.35%**| 0.98824   | 0.99071           | **+0.25%**|

# ## 3.5 Simple CNN (cnn_5)
# In this version I increased the number of the filters (2nd convolutional layer).
# 
# ```diff
#         km.add(Convolution2D(32, (5, 5), input_shape=input_shape, activation='relu'))
#         km.add(MaxPooling2D(pool_size=(2, 2)))
#         km.add(Dropout(0.25))
#         
# -        km.add(Convolution2D(16, (3, 3), input_shape=input_shape, activation='relu'))
# +        km.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
#         
#         km.add(Flatten())
# ```
# 
# ### Results
# 
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 20                 | cnn_4             | 0.990190    | 0.991143        | **+0.1%** | 0.98842     | 0.99100            | **+0.26%**|
# | 100               | cnn_4             | 0.991905     | 0.992286      | **+0.04%**| 0.99071   | 0.99214            | **+0.14%**|

# ## 3.6 Simple CNN (cnn_6)
# I changed the units of the Dense layer (and I added one more)
# 
# ```diff
#         km.add(Flatten())
# -        km.add(Dense(units=64, activation='relu'))
# +       km.add(Dense(units=128, activation='relu'))
#         km.add(Dropout(0.25))
# +
# +        km.add(Dense(units=64, activation='relu'))
# +        km.add(Dropout(0.25))
# 
#         km.add(Dense(units=10, activation='softmax'))
# ```
# 
# ### Results
# 
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 20                 | cnn_5             | 0.991143     | 0.991619        | **+0.05%** | 0.99100  | 0.99142             | **+0.04%**|
# | 100               | cnn_5             | 0.992286    | 0.992762       | **+0.05%**| 0.99214   | 0.99171              | -0.04%|

# ## 3.7 Simple CNN (cnn_7)
# In this version I simply duplicated the convolutional layers.
# 
# ```diff
#         km = Sequential()
# 
#         km.add(Convolution2D(32, (5, 5), input_shape=input_shape, activation='relu'))
# +       km.add(Convolution2D(32, (5, 5), input_shape=input_shape, activation='relu'))
#         km.add(MaxPooling2D(pool_size=(2, 2)))
# 
#         km.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
# +        km.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
#         km.add(MaxPooling2D(pool_size=(2, 2)))
# 
# ```
# 
# ### Results
# 
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 20                 | cnn_6             | 0.991619     | 0.993810        | **+0.22%** | 0.99142  | 0.99314            | **+0.17%**|
# | 100               | cnn_6             | 0.992762    | 0.993810        | **+0.11%**   | 0.99171   | 0.99371            | **+0.20%**|

# ## 3.8 CNN with data augmentation (data_aug_1)
# Let's implement some of the common data augmentation strategies (translation, rotation, flipping, etc). By applying the transformations to our training data we can reduce overfitting, we can add more images to our training set and hopefully we can increase our score.
# 
# ```python
# from keras.preprocessing.image import ImageDataGenerator
# 
# # [ ... ]
# 
# generated_data = ImageDataGenerator(rotation_range=10,
#                                     zoom_range=0.1,
#                                     shear_range=0.1)
# 
# generated_data.fit(features_train)
# 
# # model = same as the previous one.
# model.fit_generator(generated_data.flow(features_train, labels_train,
#                                         batch_size=self._batch_size),
#                     epochs=self._epochs,
#                     callbacks=[self._history],
#                     validation_data=(features_validation, labels_validation),
#                     steps_per_epoch=features_train.shape[0] / self._batch_size,
#                     verbose=self._verbose)
# 
# ```
# 
# ### Results
# 
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 20                 | cnn_7             | 0.993810     | 0.993143        | -0.07% | 0.99314         | 0.99085           | -0.23%|
# | 100               | cnn_7             | 0.993810     | 0.994762       | **+0.16%**| 0.99371    | 0.99400    | **+0.03%**|
# | 800               | data_aug_1 (100) | 0.994762 | 0.994762   | **+0.0%**| 0.99400    | 0.99542     | **+0.14%**|
# 

# ## 3.9 CNN with data augmentation (data_aug_2)
# I added some more of the transformation strategies.
# 
# ```diff
# generated_data = ImageDataGenerator(rotation_range=10,
# 									zoom_range=0.1,
# -									shear_range=0.1)
# +									shear_range=0.1,
# +									height_shift_range=0.1,
# +									width_shift_range=0.1)
# 
# ```
# 
# ### Results
# 
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 20                 | data_aug_1    | 0.993143     | 0.993016        | -0.01% | 0.99085         | 0.99414           | **+0.33%**|
# | 100               | data_aug_1    | 0.994762     | 0.995238       | **+0.06%**| 0.99400    | 0.99428    | **+0.03%**|
# | 800               | data_aug_1   | 0.994762     | 0.995714        | **+0.1%**| 0.99542    | 0.99457     | -0.09%|
# I modified the *rotation_range* to *15*, but I forgot to save the data.

# ## 3.10 Fine tuning the learning rate (lr_1)
# The learning rate is one of the most important hyper-parameters to tune when you are training CNNs. In this version I added *ReduceLROnPlateau* callback to the model.
# 
# ```diff
# @@ -59,6 +61,16 @@ def create_model(self, input_shape):
#  
#          self._set_model(km)
# 
# +    def get_optimizer(self):
# +        return Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=True)
# +
# +    def get_learning_rate_optimizer_callback(self):
# +        return ReduceLROnPlateau(monitor='val_loss',
# +                                 patience=3,
# +                                 verbose=self._verbose,
# +                                 factor=0.9,
# +                                 min_lr=0.00001)
# +
#                                  
# @@ -72,7 +84,7 @@ def fit(self, features_train, labels_train, features_validation, labels_validati
#          model.fit_generator(generated_data.flow(features_train, labels_train,
#                                                  batch_size=self._batch_size),
#                              epochs=self._epochs,
# -                            callbacks=[self._history],
# +                            callbacks=[self.get_learning_rate_optimizer_callback(), self._history],
# 
# ```
# 
# 
# ### Results
# 
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 100               | data_aug_2    | 0.995238     | 0.996032          | **+0.08%**| 0.99428    | 0.99557   | **+0.13%**|
# | 400               | data_aug_2 (800)   | 0.995714   | 0.996032   | **+0.05%**| 0.99457    | 0.99685   | **+0.23%**|
# 
# ### Overfitting
# As you can see, after ~90 epochs the accuracy is not improving and the model is overfitted.
# 
# | **100 epochs** | **400 epochs**  |
# | ------------------ :|:---------------------:|
# |![100 epochs](https://raw.githubusercontent.com/pestipeti/KaggleDigitRecognizer/master/data/lr_1_100_accuracy.jpg)                | ![400 epochs](https://raw.githubusercontent.com/pestipeti/KaggleDigitRecognizer/master/data/lr_1_400_accuracy.jpg)    |
# 
# 

# ## 3.11 Fine tuning the learning rate (lr_2)
# I experimented with the learning rate parameters; these were the best I found.
# 
# ```diff
# @@ -62,14 +62,15 @@ def create_model(self, input_shape):
#          self._set_model(km)
#  
#      def get_optimizer(self):
# -        return Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=True)
# +        return Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=True)
#  
#      def get_learning_rate_optimizer_callback(self):
#          return ReduceLROnPlateau(monitor='val_loss',
# -                                 patience=3,
# +                                 patience=4,
#                                   verbose=self._verbose,
#                                   factor=0.9,
# -                                 min_lr=0.00001)
# +                                 min_lr=0.0000075,
# +                                 cooldown=2)
# ```
# 
# ### Results
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 100               | lr_1              | 0.996032     | 0.996190   | **+0.02%**| 0.99557    | 0.99571   | **+0.01%**|
# | 400              | lr_1              | 0.996032     | 0.996667   | **+0.06%**| 0.99685    | 0.99657   | -0.03%|
# 
# ### Overfitting
# Its better, but still not good enough.
# 
# | **100 epochs** | **400 epochs**  |
# | ------------------ :|:---------------------:|
# |![100 epochs](https://raw.githubusercontent.com/pestipeti/KaggleDigitRecognizer/master/data/lr_2_100_accuracy.jpg)                | ![400 epochs](https://raw.githubusercontent.com/pestipeti/KaggleDigitRecognizer/master/data/lr_2_400_accuracy.jpg)    |
# 

# ## 3.12 Final version of the model (final_1)
# In order to reduce overfitting and improve the accuracy I added a few *BatchNormalization* layers.
# 
# ```diff
#          km.add(Convolution2D(32, (5, 5), input_shape=input_shape, activation='relu'))
# +        km.add(BatchNormalization())
#          km.add(Convolution2D(32, (5, 5), input_shape=input_shape, activation='relu'))
#          km.add(MaxPooling2D(pool_size=(2, 2)))
# +        km.add(BatchNormalization())
#          km.add(Dropout(0.25))
#  
#          km.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
# +        km.add(BatchNormalization())
#          km.add(Convolution2D(64, (3, 3), input_shape=input_shape, activation='relu'))
#          km.add(MaxPooling2D(pool_size=(2, 2)))
# +        km.add(BatchNormalization())
#          km.add(Dropout(0.25))
#  
#          km.add(Flatten())
# +        km.add(BatchNormalization())
#          km.add(Dense(units=128, activation='relu'))
# +        km.add(BatchNormalization())
#          km.add(Dropout(0.25))
#  
#          km.add(Dense(units=64, activation='relu'))
# +        km.add(BatchNormalization())
#          km.add(Dropout(0.25))
# ```
# 
# ### Results
# | **Epochs** | **Old Ver.**  | **Old Acc** | ** New Acc** | **%** | **Old Score** | **New Score** | **%** |
# | -------------- |:---------------:| --------------:| ----------------:| -------:| -----------------:| ------------------:| -------:|
# | 100               | lr_2              | 0.996190     | 0.995556   | -0.06% | 0.99571    | 0.99628   | **+0.06%**|
# | 250              | lr_2 (400)    | 0.996667     | 0.996667   | **+0.00%** | 0.99657    | 0.99714   | **+0.06%**|
# 
# ### Overfitting
# I did not find any better solution. Please leave me a comment if you have any ideas. Maybe I should try to increase the rate of the dropouts.
# I will add an *EarlyStopping* layer, the training should stop after 105-110 epochs.
# 
# Anyway, this version has the highest public score (from my submissions): **0.99714**.
# 
# | **100 epochs** | **250 epochs**  |
# | ------------------ :|:---------------------:|
# | ![100](https://raw.githubusercontent.com/pestipeti/KaggleDigitRecognizer/master/data/final_1_100_accuracy.jpg) | ![250](https://raw.githubusercontent.com/pestipeti/KaggleDigitRecognizer/master/data/final_1_250_accuracy.jpg)    |
# 

# # 4. Further Improvements
# 
# * Implement popular CNN architectures
# * Ensebling
# * Using different sizes of the images (and ensembling)
# 
# 

# # 5. Resources
# 
# ### Hardware
# I'm using *Paperspace* for all of my machine learning tasks. For this competition I used their *p4000* machine:
# 
# * **CPUs**: 8
# * **GPU**: nVidia Quadro p4000 8GB
# * **Memory**: 32GB
# 
# [Use this link](https://www.paperspace.com/&R=SBQM8B) and they give you $10 in account credit once you sign up and you add a valid payment method.
# 
# You have to install the GPU driver, TensorFlow, etc. Leave me a comment if you need help.
# 
# 
# ### Source code
# You can find the full source code [here](https://github.com/pestipeti/KaggleDigitRecognizer).

# **If you found this notebook helpful, an upvote would be much appreciated, thanks.**
