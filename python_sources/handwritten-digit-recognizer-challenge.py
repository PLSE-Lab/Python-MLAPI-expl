#!/usr/bin/env python
# coding: utf-8

# # Handwritten Digit Recognizer - A Image Classification Problem
# This is my commented walkthrough for this Image Classification problem. Sections of this notebook were inspired on the bechmark solution [CNN Keras](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6) from [Yassine Ghouzam](https://www.kaggle.com/yassineghouzam) which achieved amazingly 99,7% Accuracy! <br><br>
# The proposal of this notebook is to discuss the best practices for Image Classification with Neural Networks on Keras. The main concepts presented are:
# * Data Reshaping 
# * Neural Network Architecture & Optimizations 
# * Data Augmentation 
# 
# 

# ### Load packages
# Let's start off by loading the proper packages.

# In[ ]:


# Data Wrangling
import numpy as np
import pandas as pd

# Visualizations
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import clear_output
plt.rcParams['figure.figsize'] = (12,6)
get_ipython().run_line_magic('matplotlib', 'inline')

# Neural Network Model (Keras with TensorFlow backend)
from keras.models import Sequential
from keras.optimizers import RMSprop#, Adadelta # removed due to bad performance
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

# Model Evaluation
np.random.seed(2)
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import scikitplot as skplt

clear_output()

print('Packages properly loaded!')


# ### Load and check data

# In[ ]:


# Load the data
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

print('train.shape, test.shape')
train.shape, test.shape


# In[ ]:


print('train dataset')
display(train.head(2))
print('test dataset')
display(test.head(2))


# 70,000 samples with 784 pixels resolution each plus a 'label' column on our train dataset.<br><br>
# It's still not possible to visualize each digit like this, we need first to reshape each sample. Let's do it using the numpy reshape function!

# In[ ]:


i = 0
n_rows, n_cols = 2, 6
fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12,4))
print(f'First {n_rows*n_cols} Digits of Training Dataset')
for row in range(n_rows):
    for col in range(n_cols):
        ax[row, col].imshow(train.iloc[i,1:].values.reshape(28,28))
        ax[row, col].set_title(f"Label {train.label.iloc[i]}")
        i+=1
plt.tight_layout(pad=2)


# So now we can understand a little bit better how our data is stored and represents. Here we plotted to first 12 digits of our training dataset.<br><br>
# It was easily done due to the numpy reshape function. It takes a given data, a numpy array, and changes the position of each point according to the arguments specified. Here we used 28x28 in order to encompass all 784 pixels into a squared matrix for image plotting. 

# In[ ]:


test.describe()


# In[ ]:


print('Pixel missing values')
print(f'test set: {test.isna().any().sum()}')
print(f'train set: {train.isna().any().sum()}')


# A describe method shows us some parameters of our data such as mean, min and max. Here we can see that the borders are always dark due to it's null values on head and tail pixels. Closer to the center there are some pixels activated (e.gg. pixels 774 to 778, above). There also no missing values for any pixels.<br><br>
# So now that we understood and checked our data, let's move forwared to process it for our Neural Network model.

# In[ ]:


# Let's check memory usage
train.info()


# It seems like the dataframes require a lot of memory (251,5 MB). Let's process and dump it so our cache get's cleared up.

# ### Process data

# In[ ]:


# If condition to ensure cell execution to be idempotent
if 'train' in globals(): 
    y_train = train["label"]
    X_train = train.drop(labels = ["label"], axis=1) 
    del train 

_ = y_train.value_counts().sort_index().plot(kind='bar', title='Sample Count per Digit', figsize=(12,4))


# Our sample distribution is well balanced. No need to worry with any resampling or stratification in order to avoid dominant classification with a biased class.
# 
# Let's hot encode the classes for our CNN. The idea is to have a 10 position vector instead of a label. Example: 1 = \[1 0 0 0 0 0 0 0 0 0\]

# In[ ]:


y_train = to_categorical(y_train, num_classes = 10)
print(f'Vector for first digit of training set (label 1): {y_train[0]}')
print(f'y_train.shape                                   : {y_train.shape}')


# Let's now normalize the data and reshape it.

# In[ ]:


# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

# Reshape for CNN
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

X_train.shape, test.shape


# Now our data has turned into a four dimensional matrix. We have 70,000 samples from train (42k) and test (28k) with each observation being a squared matrix of 28x28 and the last layer a unidimensional array containg the activation value for the pixel which has been normalized from 0 to 1.<br><br>
# Let's quickly check if our first train digit is properly stored by plotting it.

# In[ ]:


display(X_train[0][:,:,0].shape)
_ = plt.imshow(X_train[0][:,:,0])


# In[ ]:


X_train.shape, y_train.shape


# Data has been processed, now let's split it into train and test for fitting the model.

# In[ ]:


# Random seed for reproducibility
random_seed = 2

# Split into train and validation (since test definition is already attributed to submission data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=random_seed)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# ### Neural Network Architecture

# Now we should define our CNN architecture. Since there's a lot of tweaking going on here I based my solution on [CNN Keras](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6) from [Yassine Ghouzam](https://www.kaggle.com/yassineghouzam) due to it's high accuracy (99,7%) and best practices for image classification. Below my explanations.
# 
# 10 Layer Convolution Neural Network Archtecture:
# 1. Conv2D with 'relu' - contains 32 filters to extract image embeddings and 'relu' adds non-linearity to the model
# 2. MaxPool2D - reduces computational cost and avoids overfitting via image downsampling
# 3. Dropout - avoids overfitting by turning off neurons randomly 
# 4. Conv2D with 'relu'
# 5. MaxPool2D
# 6. Dropout
# 7. Flatten - turns data into 1D vector for combining global and local features
# 8. Dense - classifies data with perdicted labels
# 9. Dropout 
# 10. Softmax - turns classified data into output probabilty vector for calculating loss

# In[ ]:


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding='Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding='Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding='Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding='Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# Now we set a very important function, the optimizer RMSprop. It gives a better strategy for calculating loss and learning rate than the usual Gradient Descent. I have also tried a very popular optimizer, Adadelt, but RMSprop is much more efficient. For comparision purposes, I achieved 22% accuracy in 3 epochs with Adadelta whereas RMSprop gave an astonishing 97%! Better to keep with best practices in such a sensitive parameters.
# 
# We define it and compile the model.

# In[ ]:


optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 10
batch_size = 86


# ### Data Augmentation

# Prior to running the model let's make use of a good practice for improving quality of Predictive Models: Data Augmentation. This is a strategy to synthetically enlarge training set by applying slight changes to the data. This easily multiplies the labeled dataset by two to three folds. The main concept is to tilt, shift or zoom some images so the model gets more robust. It also avoids overfilling in some sense.

# In[ ]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # turns off centering (mean=0) by feature
        samplewise_center=False,  # turns off centering (mean=0) by sample
        featurewise_std_normalization=False,  # turns off normalization by feature
        samplewise_std_normalization=False,  # turns off normalization by sample
        rotation_range=10,  # randomly rotates image from 0 to 10 degrees
        zoom_range = 0.1, # randomly zooms image by 10%
        width_shift_range=0.1,  # randomly shifts image horizontally by 10% of total width
        height_shift_range=0.1,  # randomly shifts image vertically by 10% of total height
        horizontal_flip=False,  # turns off random image flipping
        vertical_flip=False)  # turns off random image flipping

datagen.fit(X_train)


# Now that the data has been preprocessed and the model is defined, let's fit it to and check results.

# In[ ]:


history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# ### Model Evaluation

# We can see how accuracy and loss evolved along the epochs. There's a very good performance since the beginning due to the proper Neural Network architecture and its hyperparameter definition.

# In[ ]:


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# Let's check the confusion matrix to have a better understanding of the model's predictive performance. That's a very useful way to spot missclassifications and then straight-foreward correcting them.

# In[ ]:


y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred,axis = 1) 
y_true = np.argmax(y_val,axis = 1) 

skplt.metrics.plot_confusion_matrix(y_true, y_pred_classes)


# As expected from the accuracy and loss metrics the model is performing very well. There are some missclassifications for 0, 1 and 4 digits.
# 
# Let's have a closer look.

# In[ ]:


# Errors are difference between predicted labels and true labels
errors = (y_pred_classes - y_true != 0)

y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

y_pred_errors_prob = np.max(y_pred_errors,axis = 1)
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_dela_errors[-6:]
display_errors(most_important_errors, X_val_errors, y_pred_classes_errors, y_true_errors)
plt.tight_layout(pad=2)


# We can see that that the miss classifications are reasonable, since even humans would have a hard time figuring out what those digits stand for.

# Let's wrap the results into the submission csv and close this challange.

# In[ ]:


results = model.predict(test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)


# ### Conclusions
# We were able to walkthrough some of the best practice for Image Classification problemas. Data Reshaping, Augmentation and Neural Network Archtecture were the main concepts described here. Thus yielding a result of over 99% accuracy with relatively little effort.
# 
# The ideas and concepts here may be better tweaked for production models, but they already give a very good sense of how powerful Machine Learning can be.
