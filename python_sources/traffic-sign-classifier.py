#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[ ]:


import pickle
from pathlib import Path
import numpy as np
import math
#import pandas as pd
# TODO: Fill this in based on where you saved the training and testing data

PATH = Path('.')

training_file = '../input/german-traffic-sign/train.p'
validation_file = '../input/german-traffic-sign/valid.p' 
testing_file = '../input/german-traffic-sign/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[ ]:


n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[ ]:


import pandas as pd
signnames = pd.read_csv('../input/signnames/signnames.csv')


# In[ ]:


signnames


# In[ ]:


classID_signames = list(signnames['SignName'])


# In[ ]:


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import numpy as np
# Visualizations will be shown in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_unique_indexs = list(np.unique(y_train, return_index=True)[1])
rows = len(train_unique_indexs)//4 + 1
f = plt.figure(figsize=(20, 16))
for i, index in enumerate(train_unique_indexs, 1):
    plt.subplot(rows, 4, i)
    plt.imshow(X_train[train_unique_indexs[i-1]])
    plt.axis('off')
    plt.title(classID_signames[i-1])
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.4)


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[ ]:


X_train = (X_train)/255
X_valid = (X_valid)/255
X_test = (X_test)/255


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[ ]:


from keras.utils import np_utils
from keras.layers import (Conv2D, MaxPooling2D,
                          Input, Flatten, Dense, 
                          BatchNormalization, 
                          Activation, AveragePooling2D,
                          GlobalAveragePooling2D,LeakyReLU, Dropout, Add)
from keras.models import Model
from keras import layers
from keras.regularizers import l2
from keras.callbacks import Callback


# ### Normalization

# In[ ]:


y_train = np_utils.to_categorical(y_train)
y_valid = np_utils.to_categorical(y_valid)
y_test = np_utils.to_categorical(y_test)


# In[ ]:


input_shape = (32, 32, 3)
classes = 43
X_input = Input(input_shape)


# ### Terminate Training after Validation Accuracy reaches above 97%

# In[ ]:


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='acc', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True


# In[ ]:


callbacks = [TerminateOnBaseline(monitor='val_acc', baseline=0.97)]


# In[ ]:


### Resnet block o be used in the model


# In[ ]:


def resnet(X, channel):
    X_short = X
    X = Conv2D(channel, (1, 1), strides = (1, 1), kernel_initializer='he_normal',use_bias=False, kernel_regularizer=l2(1e-4))(X)
    X = Conv2D(channel, (1, 1), strides = (1, 1), kernel_initializer='he_normal',use_bias=False, kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization()(X)
    X = Add()([X, X_short])##############
    X = LeakyReLU(alpha=0.1)(X)
    return X


# In[ ]:


def simple_conv(X, channel, f, s):
    X = Conv2D(channel, (f, f), strides = (s, s), kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.1)(X)
    return X

def conv(X, channel, f, s):
    X = Conv2D(channel, (f, f), strides = (s, s), kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(channel, (1, 1), strides = (1, 1), kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    return X


# ### Model Architecture

# In[ ]:


X = simple_conv(X_input, 64, 3, 2)

X = resnet(X, 64)
X = conv(X, 128, 3, 2)
X = resnet(X, 128)
X = conv(X, 256, 1, 1) # test
X = resnet(X, 256) # test
X = conv(X, 512, 3, 2)
X = resnet(X, 512)
X = conv(X, 1024, 3, 2)
X = resnet(X, 1024)

X = simple_conv(X, 128, 1, 1)
X = simple_conv(X, 128, 1, 1)

X = GlobalAveragePooling2D()(X)
X = BatchNormalization()(X) # imp
output = Dropout(0.25)(X)
output = Dense(512, activation='relu')(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
out = Dense(43, activation='softmax')(output)


# ### Train, Validate and Test the Model

# In[ ]:


model = Model(inputs = X_input, outputs = out)


# In[ ]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=20, batch_size=32, callbacks=callbacks)


# ### Final Validation Score 

# In[ ]:


valid = model.predict(X_valid)
valid_score = len(y_valid[y_valid.argmax(axis=1)==valid.argmax(axis=1)])/len(y_valid)
print(f"Validation Score = {valid_score*100:0.2f}%")


# ### Final Test Score

# In[ ]:


y_test_predict = model.predict(X_test)
test_score = len(y_test[y_test.argmax(axis=1)==y_test_predict.argmax(axis=1)])/len(y_test)
print(f"Test Score = {test_score*100:0.2f}%")


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign nam

# ### Load and Output the Images

# In[ ]:


import glob


# ### Loading and preprocessing

# In[ ]:


img_internet = glob.glob('../input/internet-images/*.jpg')
img_internet = np.array([plt.imread(i) for i in img_internet])
img_internet = img_internet/255


# In[ ]:


f = plt.figure(figsize=(20, 16))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(img_internet[i])
    plt.axis('off')
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.4)


# ### Predict the Sign Type for Each Image

# In[ ]:


predict_internet = model.predict(img_internet)


# In[ ]:


predict_internet_id = predict_internet.argmax(axis=1)


# In[ ]:


f = plt.figure(figsize=(20, 16))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(img_internet[i])
    plt.title(f'predicted = {classID_signames[predict_internet_id[i]]}')
    plt.axis('off')
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.4)


# ### Analyze Performance

# In[ ]:


total_images = 5
correct_prediction = 2
accuracy = 2/5*100
print(f"accuracy of images found on internet = {accuracy} %")


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[ ]:


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
np.sort(predict_internet, axis=1)[:,::-1][:,:5]


# In[ ]:


np.max(predict_internet, axis=1)

