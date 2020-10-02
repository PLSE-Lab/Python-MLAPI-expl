#!/usr/bin/env python
# coding: utf-8

# # MNIST digit recognizer: CNN, grid search and data augmentation
# I am using the MNIST digit classificaton problem as an exercice to implement some intermediate technics for image processing using Keras, Tensorflow and Scikit Learn. We will: 
# * start by **implementing a simple MultiLayer Perceptron** with some default values for hyper parameters 
# * improve this model by doing a **grid search over some hyperparameters**
# * move on on a more complex model by implementing a **Convolutional Neural Network**
# * improve the CNN model using again a **grid search**
# * use **data augmentation** to reduce overfit and further improve performance
# 
# This kernel is designed for intermediate users having some knowledge and experience of neural networks and optimization of hyperparameters in Scikit Learn. You can find [here](https://victorzhou.com/blog/keras-neural-network-tutorial/) an excellent introduction to MLP. The same autor has also a very good [introduction](https://victorzhou.com/blog/keras-cnn-tutorial/) to CNN. You can also refer to this Kaggle [kernel](https://www.kaggle.com/anebzt/mnist-with-cnn-in-keras-detailed-explanation) for detailed information on implementing a CNN.
# 
# For hyperparameter optimization, this amazing [kernel](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist) is an excellent demonstration of manual tuning. Thanks also to the authors of this [post](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/) which got me started on using grid search for Keras hyperparameters, and of this [kernel](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6) for data augmentation.
# 
# We will follow the following steps to run this kernel:
# 1. EDA: initial look at the data
# 1. Implementing a simple MLP
# 1. Optimize hyperparameters and add dropout to reduce overfit
# 1. Implementing a Convolutional Neural Network
# 1. Optimize the hyperparameters of the CNN thru GridSearch
# 1. Use data augmentation to improve performance

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# keras import
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# hyperparameter optimization
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# data augmentation
from keras.preprocessing.image import ImageDataGenerator

# visualisation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
#set figure size
plt.rcParams['figure.figsize'] = 12, 6
sns.set_style('white')

# others
from random import randrange
from time import time


# I define here global constants for the number of epochs when training an individual model (n_epochs) or doing a grid search with cross validation (n_epochs_cv). Reduce these 2 values if you want to reduce the running time of the kernel.  
# 
# I also define the number of runs for cross validation (when using Scikit Learn GridSearchCV) and the size of the validation set for train_test_split.
# 

# In[ ]:


n_epochs = 30 # 30 
n_epochs_cv = 10 # 10  # reduce number of epochs for cross validation for performance reason

n_cv = 3
validation_ratio = 0.10


# ## 1. EDA: initial look at the data
# We load the data and display some sample observations to understand the data structure.  

# In[ ]:


# load dataset and check dimension
data_set = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
print(data_set.shape)


# In[ ]:


data_set.sample(3)


# We check the distribution of the 10 classes of digits. They are roughly equivalently represented, therefore we do not need to use stratify when splitting the data set into training and validation sets.

# In[ ]:


# segregate training data set in pixel features and label
y = data_set['label']
X = data_set.drop(labels = ['label'], axis=1) 
# free memory
del data_set

# check distribution of the handwritten digits
sns.countplot(y, color='skyblue');


# Next, let's plot a random sample of 60 images to get a *visual feeling* of the classification task. 

# In[ ]:


# show multiple images chosen randomly 
fig, axs = plt.subplots(6, 10, figsize=(10, 6)) # 6 rows of 10 images

for ax in axs.flat:
    i = randrange(X.shape[0])
    ax.imshow(X.loc[i].values.reshape(28, 28), cmap='gray_r')
    ax.set_axis_off()


# In[ ]:


# Normalize pixel value to range 0 to 1
X = X / 255.0

# extract train and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = validation_ratio)


# ## 2. Implementing a simple MLP
# Let's start with a simple multilayer perceptron with only 1 hidden layer:
# * the input layer consists in 128 units, with relu
# * the hidden layer is made of 64 units, with sigmoid activation function
# * the output layer contains one unit per expected class, that is 10 units, and uses a softmax activation function to output probabilities  
# [](http://)
# How many parameters has this model?
# * the input layer is taking values from 28x28 images: the number of parameters is 28x28 (input size) x 128 (output size) weights + 128 bias values = 100,480
# * for the hidden layer, a similar caculation gives 128 x 64 + 64 = 8,256
# * and for the output layer, we have 64 x 10 + 10 = 650 parameters  
# 
# OK, this was easy, but we'll see below that counting parameters of CNN is a bit more tricky. 

# In[ ]:


# define model
mlp = Sequential()
mlp.add(Dense(128, activation='relu', input_shape=(784,)))
mlp.add(Dense(64, activation='sigmoid'))  
mlp.add(Dense(10, activation='softmax'))

mlp.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

mlp.summary()


# Now, let's train the model. I am using a early stop callback to reduce training time.  
# Look at the accuracy on the validation set and compare it with the accuracy on the training set. What can you say about the performance and limits of this simple model?

# In[ ]:


# Train the model

#define callbacks
early_stop = EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience=5, restore_best_weights=True)

history = mlp.fit(
    X_train,
    to_categorical(y_train),
    epochs = n_epochs,  
    validation_data = (X_val, to_categorical(y_val)),
    batch_size = 32,
    callbacks = [early_stop]
)


# ## What can we say from the results of our training:
# * the performance of our model is given by the best accuracy obtained on the validation set: around 0.975 (depending on the run)
# * the best scores are already obtained after more or less 10 epochs, meaning that there is probably no or little gain to increase number of epochs
# * and there is clear sign of overfitting as the loss for the validation set (around 0.1) is roughly 10 times the loss on the training set (around 0.01).  
# 
# Let's plot the accuracy for the training and validation to confirm this last point.  

# In[ ]:


# compare accuracy accuracy on training and validation data
df_history = pd.DataFrame(history.history)
sns.lineplot(data=df_history[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);


# ## 3. Optimize hyperparameters and add dropout to reduce overfit
# As our MLP model shows sign of overfit, we will now add dropout layers to try to fix this. The dropout rate will be determined thru a grid search, along with the batch size.  
# For this, we will use the KerasClassifier wrapper for Scikit Learn, which gives us a Scikit Learn estimator that we can optimize with GridSearchCV (cf Keras [documentation](https://keras.io/scikit-learn-api/)).  
# 
# Results are:
# * no more overfit from dropout rate = 0.2 and above 
# * no or minimal improvement of the accuracy on the validation set: **we are reaching the limit of a simple MLP model**
# * noticeable degradation of results (both on training and validation sets) for rate of 0.4 and above 

# In[ ]:


start=time()

# define a function to create model, required for KerasClassifier
# the function takes drop_out rate as argument so we can optimize it  
def create_mlp_model(dropout_rate=0):
    # create model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(784,))) 
    # add a dropout layer if rate is not null
    if dropout_rate != 0:
        model.add(Dropout(rate=dropout_rate))        
    model.add(Dense(64, activation='sigmoid')) 
    # add a dropout layer if rate is not null    
    if dropout_rate != 0:
        model.add(Dropout(rate=dropout_rate))           
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    model.compile( 
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )    
    return model

# define function to display the results of the grid search
def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))    
    
# create model
model = KerasClassifier(build_fn=create_mlp_model, verbose=1)
# define parameters and values for grid search 
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [n_epochs_cv],
    'dropout_rate': [0.0, 0.10, 0.20, 0.30],
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=n_cv)
grid_result = grid.fit(X, to_categorical(y))  # fit the full dataset as we are using cross validation 

# print out results
print('time for grid search = {:.0f} sec'.format(time()-start))
display_cv_results(grid_result)


# The best score (around 0.973) is obtained for a drop out rate of 0.10 and a batch size of 16 or 32 (depending on the run).  
# Looking at the mean accuracy for each set of parameters:
# * batch sizes of 16 or 32 give similar results, but we see a degradation for batch size = 64
# * droprout rate of 0.10 consistently gives better results, regardless of the batch size
# 
# One point that puzled me is that the best score is lower than the accuracy that we obtained before without hyperparameter optimization (!?). My guess is that this best score value is an average of the score of the best estimator over all epochs (I'd appreciate a lot if someone can point me to the exact reason). For this reason, I reload the best estimator and train it on the full dataset. Compared to the first network that we trained, there is a slight improvement of the accuracy at 0.978.

# In[ ]:


# reload best model
mlp = grid_result.best_estimator_ 

# retrain best model on the full training set 
history = mlp.fit(
    X_train,
    to_categorical(y_train),
    validation_data = (X_val, to_categorical(y_val)),
    epochs = n_epochs,
    callbacks = [early_stop]    
)


# In[ ]:


# get prediction on validation dataset 
y_pred = mlp.predict(X_val)
print('Accuracy on validation data = {:.4f}'.format(accuracy_score(y_val, y_pred)))

# plot accuracy on training and validation data
df_history = pd.DataFrame(history.history)
sns.lineplot(data=df_history[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);


# In[ ]:


# load test data and make prediction
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
y_test = mlp.predict(test)


# In[ ]:


# convert prediction to df
submission = pd.DataFrame(data=y_test)

# set label as the 0-9 class with highest value 
submission['Label'] = submission.idxmax(axis=1)
submission['ImageId'] = np.asarray([i+1 for i in range(submission.shape[0])])

submission.to_csv('submission-mlp_dropout.csv', 
                  columns=['ImageId','Label'],
                  header=True,
                  index=False)


# ## 4. Implementing a Convolutional Neural Network
# We implement now a CNN since MLP model is limited at 0.975 accuracy. I selected the architecture thru trials based on several examples. 

# In[ ]:


# Reshape the images
img_size = 28
X_cnn = X.values.reshape(-1, img_size, img_size, 1)
# check 
print(X_cnn.shape)

X_train, X_val, y_train, y_val = train_test_split(X_cnn, y, test_size = validation_ratio)


# In[ ]:


# function to create the model for Keras wrapper to scikit learn
# we will optimize the type of pooling layer (max or average) and the activation function of the 2nd and 3rd convolution layers 
def create_cnn_model(pool_type='max', conv_activation='sigmoid', dropout_rate=0.10):
    # create model
    model = Sequential()
    
    # first layer: convolution
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1))) 
        
    # second series of layers: convolution, pooling, and dropout
    model.add(Conv2D(32, kernel_size=(5, 5), activation=conv_activation))  
    if pool_type == 'max':
        model.add(MaxPooling2D(pool_size=(2, 2)))
    if pool_type == 'average':
        model.add(AveragePooling2D(pool_size=(2, 2)))
    if dropout_rate != 0:
        model.add(Dropout(rate=dropout_rate))     
    
    # third series of layers: convolution, pooling, and dropout    
    model.add(Conv2D(64, kernel_size=(3, 3), activation=conv_activation))   # 32   
    if pool_type == 'max':
        model.add(MaxPooling2D(pool_size=(2, 2)))
    if pool_type == 'average':
        model.add(AveragePooling2D(pool_size=(2, 2)))
    if dropout_rate != 0:
        model.add(Dropout(rate=dropout_rate))     
      
    # fourth series
    model.add(Flatten())         
    model.add(Dense(64, activation='sigmoid')) # 64
    # add a dropout layer if rate is not null    
    if dropout_rate != 0:
        model.add(Dropout(rate=dropout_rate)) 
        
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    model.compile( 
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )    
    return model

cnn = create_cnn_model()

cnn.compile(
  optimizer='adam',
  loss='categorical_crossentropy',  
  metrics=['accuracy'],
)

cnn.summary()


# Let's check the number of parameters:
# * the first layer is a convolution layer with 16 kernels of size 5x5. Each kernel has 26 parameters (25 weigths plus bias). Total is 16 x 26 = 416
# * the second layer is also a convolution layer with 32 kernels of size 5x5. There are 16 input images from the output of the previous layer. For every kernel, the 16 images are combined together, with 16 (input size) x 25 (kernel size) weights plus bias (that is 401 parameters for each kernel). Total number of parameters for this layers is 401 x 32 = 12,832
# * using same calculation logic, the number of parameters for the 3rd layer is ( 32 (input size) x 9 (kernel size) + 1 ) x 64 (number of kernels) = 18,4896
# * for the dense layer, its input is the result of the flatten layer (taking as input the 64 x 16 images from the pooling layer, and mapping them to a flat array of 64 x 16 = 1024). Output size is 64, therefore number of parameters is 64 x (1024 weights + bias) = 65,600
# * same way, the final layer has (64 x 1) x 10 = 650 parameters

# In[ ]:


# Train the default CNN model
history = cnn.fit(
    X_train,
    to_categorical(y_train),
    epochs=n_epochs,  
    validation_data=(X_val, to_categorical(y_val)), 
    batch_size=32,
    callbacks = [early_stop]
)


# We gained about 1% of accuracy with the CNN! But there is a cost: training the CNN model takes 45 seconds per epoch, about 10 times what the MLP  required!  
# Let's optimize now some hyperparameters (pooling type and activation function) to further improve the performance of this CNN. 

# ## 5. Optimize the hyperparameters of the CNN thru GridSearch

# In[ ]:


# optimize model 
start = time()

# create model
model = KerasClassifier(build_fn=create_cnn_model, verbose=1)
# define parameters and values for grid search 
param_grid = {
    'pool_type': ['max', 'average'],
    'conv_activation': ['sigmoid', 'tanh'],    
    'epochs': [n_epochs_cv],
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=n_cv)
grid_result = grid.fit(X_train, to_categorical(y_train))

# summarize results
print('time for grid search = {:.0f} sec'.format(time()-start))
display_cv_results(grid_result)


# ## 6. Use data augmentation to improve performance

# In[ ]:


# optimize parameters of the fit method 
cnn_model = create_cnn_model(pool_type = grid_result.best_params_['pool_type'],
                             conv_activation = grid_result.best_params_['conv_activation'])

# With data augmentation 
datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        fill_mode='constant', cval = 0.0)

datagen.fit(X_train)

history = cnn_model.fit_generator(datagen.flow(X_train,to_categorical(y_train), batch_size=32),
                                  epochs = n_epochs, 
                                  validation_data = (X_val,to_categorical(y_val)),
                                  verbose = 1, 
                                  steps_per_epoch = X_train.shape[0] / 32,
                                  callbacks = [early_stop])

# plot accuracy on training and validation data
df_history = pd.DataFrame(history.history)
sns.lineplot(data=df_history[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);


# The accuracy on the validation is consistently higher than on the training set. The model is too much constrained: let's retrain it with data augmentation but without dropout and see how it performs. 

# In[ ]:


# optimize parameters of the fit method 
cnn_model = create_cnn_model(pool_type = grid_result.best_params_['pool_type'],
                             conv_activation = grid_result.best_params_['conv_activation'], 
                            dropout_rate=0.0)

#define early stop on the accuracy as this is the metric we want to improve
early_stop = EarlyStopping(monitor = 'accuracy', mode = 'max', patience=5, restore_best_weights=True)
history = cnn_model.fit_generator(datagen.flow(X_train,to_categorical(y_train), batch_size=32),
                                  epochs = n_epochs, 
                                  validation_data = (X_val,to_categorical(y_val)),
                                  verbose = 1, 
                                  steps_per_epoch = X_train.shape[0] / 32,
                                  callbacks = [early_stop])

# plot accuracy on training and validation data
df_history = pd.DataFrame(history.history)
sns.lineplot(data=df_history[['accuracy','val_accuracy']], palette="tab10", linewidth=2.5);


# In[ ]:


# save weights
cnn_model.save_weights('mnist_cnn.h5')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

X_test = test.values.reshape(-1, img_size, img_size, 1)
y_test = cnn_model.predict(X_test)

# convert to df
submission = pd.DataFrame(data=y_test)

# set label as the 0-9 class with highest value 
submission['Label'] = submission.idxmax(axis=1)
submission['ImageId'] = np.asarray([i+1 for i in range(submission.shape[0])])

submission.to_csv('submission-cnn.csv', 
                  columns=['ImageId','Label'],
                  header=True,
                  index=False)

