#!/usr/bin/env python
# coding: utf-8

# A quick walkthrough to apply Covolutional Neural Network (CNN) via Keras on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset in the [Kaggle Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer).
# 
# If you want to begin somewhere, with clearly annotated code but fewer paragraphs, try here ;)
# 
# Kudos for [vishwas](https://www.kaggle.com/vishwasgpai)'s [nice starter](https://www.kaggle.com/vishwasgpai/guide-for-creating-cnn-model-using-csv-file) I came across where this model is based. Please do also read & upvote his if you find this notebook userful.
# 
# I rewritten the explanations and did the following code edits:
# * adding `train_test_split` to test & improve results
# * update to [Keras 2 syntax](https://github.com/keras-team/keras/issues/6006)
# * wrap up the kernel to generate and save output
# * adding colored visualization check for label prediction
# 
# To focus on the application, I catalog links to relevant documentation/details for fellow beginners. <br/>
# 
# This is my first Kaggle kernel. Please let me know if you have any suggestions :) <br/>
# 
# *Let's go!*
# 
# > #keras #cnn #conv2d #mnist #concise #beginner

# ---
# # Import modules
# Import the usual modules
# * [numpy](http://www.numpy.org): linear algebra, e.g matrix calculation __"all-the-math"__
# * [pandas](http://pandas.pydata.org/pandas-docs/stable/): __"tables"__ ("DataFrame") I/O & manipulation
# * [matplotlib](https://matplotlib.org): data visualization, ie __"draw plots"__
# * [sklearn](http://scikit-learn.org) ("sci-kit learn"): __"data mining toolset"__  usu. for supervised machine learning and data processing, e.g.
#    - data preprocessing 
#    - classification
#    - regression
# * [keras](https://keras.io): __neural network__ library
#    - uses [tensorflow](https://www.tensorflow.org) by default, [theano](https://deeplearning.net/software/theano/), or [CNTK](https://github.com/Microsoft/CNTK/wiki) backend for __tensor manipulation__
#    -  Sequential() is among the most commonly used models to pipeline operations in each layer
# 
# tensor: take it as scalar, vector, or matrix - vector of vector for now. Look at [here](https://math.stackexchange.com/questions/412423/differences-between-a-matrix-and-a-tensor/412429#412429) and scroll up for the (much) more mathematically accurate and detailed answer. What we really "see" in notebooks are matrix tables as Pandas DataFrames. 

# In[ ]:


# Basic data processing
import numpy as np
import pandas as pd 

# Try visualize pixel values as an image
import matplotlib.pyplot as plt

# Split labelled dataset into training and testing data to test & improve our model
from sklearn.model_selection import train_test_split 
# Change label formats between input/human-readable/output-required & better for model training formats
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# To build our CNN sequential model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D # CNN
from keras import backend as K

# List the data files in store
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))


# ---
# # Loading data

# Not all raw data format are as standardized as in Kaggle. <br/>
# I have a habit to use `!head {yourfile}` to inspect input data before loading. <br/>
# You will then know the right arguments for delimiters, header, comments etc for pd.[read_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)

# In[ ]:


# Read in 2 lines of the training data to inspect
get_ipython().system('head -n2 ../input/train.csv')


# Read the CSV tables and check the data dimensions.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)


# ---
# # Data preprocessing

# We saw that the data are shaped as linear arrays of pixel intensities, e.g. <br/>
# oooooooxoooo#ooooxooooooo
# 
# But images are 2D matrix of pixel values (, ie a 3D matrix), like
# 
# ooooo<br/>
# ooxoo<br/>
# oo#oo<br/>
# ooxoo<br/>
# ooooo<br/>
# 
# Also, it's faster to store and calculate with [0,1] decimals than [0,255] (8-bit) values.
# 
# So, we
# 1. Reshape the matrix
# 2. Normalize the values from [0,255] to [0,1]

# In[ ]:


# Reshape and normalize training data
# drop the "label" column for training dataset
X = train.drop("label",axis=1).values.reshape(-1,1,28,28).astype('float32')/255.0
y = train["label"]

# Reshape and normalize test data
X_test1 = test.values.reshape(-1,1,28,28).astype('float32')/255.0


# Let's take a look at the one of the images.

# In[ ]:


plt.imshow(train.drop("label",axis=1).iloc[0].values.reshape(28,28),cmap=plt.get_cmap('binary'))
plt.show()


# # Prepare training dataset

# Split out some data from the training set to test and improve our model. <br/>
# (`random_state` set for better chance of reproducibility only)

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## LabelBinarizer
# Binarize the labels from digits to ["one-hot"](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science) encoding. (linked to the most concise explanation for OneHot seen so far)

# In[ ]:


# Before binarization
y_train.head()


# In[ ]:


lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)


# In[ ]:


# After binarization
y_train


# # Training model

# ## Designing a Conv2D sequential model

# In[ ]:


# Start a Keras sequential model
model = Sequential()
# Before Keras 2: K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')
# Before Keras 2: model.add(Convolution2D(30,5,5, border_mode= 'valid', input_shape=(1,28,28),activation= 'relu' ))
model.add(Convolution2D(30, (5,5),padding='valid',input_shape=(1,28,28),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Drop out 20% of training data in each batch
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation= 'relu' ))
model.add(Dense(50, activation= 'relu' ))
model.add(Dense(10, activation= 'softmax' ))
  # Compile model
model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train,
          epochs=25,
          batch_size= 128)


# In[ ]:


score = model.evaluate(X_test, y_test, batch_size=128)


# In[ ]:


score


# ---
# # Predict test dataset
# We're almost there. Let's label the test dataset.

# In[ ]:


y_test1 = model.predict(X_test1)
y_test1


# Convert the predictions from probabilities to be each of the labels (1-10 here) into the most probably label.

# In[ ]:


y_test1 = lb.fit_transform(np.round(y_test1))
y_test1


# Convert label from one-hot to digits, e.g. [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] to 2.

# In[ ]:


predicted_labels = np.argmax(y_test1, axis=1)
predicted_labels


# # See how we're doing
# Visualize some test images and see if our labels match.
# 
# The following code shows the first 100 images in the test data set, each colored according to their label.
# We can easily spot out if any particular number is more easily misrecognized.[](http://)

# In[ ]:


cmaps = ['binary','gray','summer','YlOrRd_r','Set3','BuGn_r','spring','tab20b','PuRd_r','winter']
fig,axarr = plt.subplots(10,10)

for i,ax in enumerate(axarr.flat):
    ax.imshow(test.iloc[i].values.reshape(28,28),cmap=plt.get_cmap(cmaps[predicted_labels[i]]))
plt.setp(axarr, xticks=[], yticks=[])
plt.subplots_adjust(hspace=-0.2,wspace=-0.2)
plt.show()


# ---
# # Save output
# Save your output for submission

# In[ ]:


np.savetxt('submission_kagglekernel_cnn25epochs.csv', 
           np.c_[range(1,len(X_test1)+1),predicted_labels], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')


# # What next?
# Before further, why not also try tune the different parameters?
# * dropout (enlarge training dataset vs overfitting)
# * change (silly, but yes, why always waste a particular set of data
# * number of layers
# * number of nodes in each layers
# * number of epochs (if you have the equipment and/or time)
# * loss function
# * optimizer
# * metric for optimization
# 

# ---
# Thank you for reading! Hope you enjoyed the kernel. :) <br/>
# See you next time!

# In[ ]:




