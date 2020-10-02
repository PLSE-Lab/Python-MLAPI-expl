#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Softmax Regression to Classify Numbers
# 
# This problem is more complicated than the regular classification/regression Machine Learning problems like House Prices or Titanic Survivors. In this problem, we have to classify a number, given the pixels. We are given a data set with pixels at different locations of the entire grid of the number. This is known as a Deep Learning problem, a more advanced application of Machine Learning. In this kernel, we will build a TensorFlow Softmax Regression model with a SGD optimizer to help us with the classification. 
# 
# Before we get started, however, you should know the basics of Machine Learning beforehand. TensorFlow has a larger library than sklearn and is more complicated to use. Be sure to read my kernel on [regression](https://www.kaggle.com/samsonqian/predicting-house-prices-with-regression) and [classification](https://www.kaggle.com/samsonqian/titanic-guide-with-sklearn-and-eda) before this one!
# 
# *Please upvote if this kernel helps you! Feel free to fork this notebook to play with the code yourself.* If you may have any questions about the code, or any step of the process, please comment and I will clear up any confusion.

# # Contents
# 1. [Importing Packages](#p1)
# 2. [Exploring and Visualizing Data](#p2)
# 3. [Preprocessing Data](#p3)
# 4. [TensorFlow Modelling](#p4)
# 5. [Submission](#p5)

# <a id="p1"></a>
# # 1. Importing Packages
# These will look familiar if you've seen my previous kernels. We will use the below Python packages to inspect and manipulate our data. We can visualize with seaborn and matplotlib. We always use these same packages when we are doing any type of Machine Learning. 

# In[ ]:


import numpy as np 
import pandas as pd 

from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")


# <a id="p2"></a>
# # 2. Exploring and Visualizing Data
# We will now load and inspect our data. Since it's data of the pixels of the numbers, we don't need to do any visual analysis of the features, but we can take a peek at the visuals that the data generates. We should however, inspect the values of the data to make sure there are no extraneous values.

# In[ ]:


training = pd.read_csv("../input/train.csv")
testing = pd.read_csv("../input/test.csv")


# In[ ]:


training.head()


# Looks like a lot of zeros. This doesn't tell us much about the what the numbers look like, so let's take a look at what this pixel data generates.

# In[ ]:


print(training.shape)


# In[ ]:


X_train = training.drop("label", axis=1)
y_train = training["label"]


# In[ ]:


ax = plt.hist(y_train)


# Seems like a pretty uniform distribution of all numbers 1-9! Let's take a look at what the numbers themselves actually look like.

# In[ ]:


plt.figure(figsize=(14, 10))

def show_images(numbers):
    for i in range(1, numbers + 1):
        plt.subplot(5, 10, i)
        image = X_train.iloc[i].as_matrix()
        image = image.reshape((28,28))
        plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title(y_train[i])

show_images(50)


# This is just a sample of what the different numbers that we're trying to classify look like. The label is on top of each number, and you can see the contrast between the black and white pixels, which is highlighted by our data.

# In[ ]:


max(training["pixel99"])


# So it seems that the pixel values go from 0 all the way to 255. This may cause problems for us, and we should definitely seek to normalize the data so that our results are better. We'll do the preprocessing in the next section.

# <a id="p3"></a>
# # 3. Preprocessing Data
# Before we build our TensorFlow model, we should prepare the data first. We'll define our features and label and we'll scale all the data so it's more representative of what we're trying to predict.

# First, we need to define the label and feature. We will convert these to float values so we can scale them by dividing by the max value, 255. We'll do this for both the training and test set.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def rescale_table(table):
    table = table.astype("float32")
    for i in table:
        reshaped_array = np.array(table[i]).reshape(-1, 1)
        table[i] = scaler.fit_transform(reshaped_array)
    return table

X_train = rescale_table(X_train)

X_test = testing
X_test = rescale_table(X_test)

#delete old sets to free space
del training
del testing


# Great! Now let's take another peek at our data.

# In[ ]:


X_train.sample(10)


# In[ ]:


max(X_train["pixel99"])


# Mostly zeros, but it's fine because they're float values and there are many values not shown that display a value greater than zero. As we can now see, the max value for our pixels is 1. Now, our data is normalized and we are ready to go.

# ## One-Hot-Encoding
# Now that we're done preprocessing our features, let's work with our label. The fact that the labels go from 1-9 is going to be troublesome because it implies that 8 is more similar to 9 than 2, just because the numbers are closer. This is not true, so we should remove this bias somehow. One-Hot-Encoding does a good job for this, since we can make a numpy array that describes each number with a 1. If this is confusing, look below and you will understand.

# In[ ]:


set(y_train)


# Pandas has a get_dummies method that helps us with one-hot-encoding, so we can use it on y_train!

# In[ ]:


def one_hot_encode(series):
    return pd.get_dummies(series).as_matrix()

y_train = one_hot_encode(y_train)


# In[ ]:


y_train


# Now, as you can see, the numbers are all represented by 1s instead of the actual number. This gets rid of the bias we had before and we are ready to use this data in our model. Let's get to work building our model!

# <a id="p4"></a>
# # 4. TensorFlow Modelling
# In this section, we will actually build our model and go over step-by-step of the process of doing so. If you are new to TensorFlow, this may be confusing at first, but you will eventually pick up on it. 

# First things first, we should always create a validation data set to evaluate our model and avoid overfitting. We use sklearn's train_test_split.

# In[ ]:


from sklearn.model_selection import train_test_split

X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


# Of course we're also going to need to import TensorFlow.

# In[ ]:


import tensorflow as tf


# The special thing about TesnorFlow is that you can define operations and computations as variables. TensorFlow doesn't run these computations until you actually tell it to during a session, which we will get to later. For now, let's define our learning rate and epochs (number of times we want the model to iterate). These are constants that can be played around with and adjusted for better model performance.
# 
# Also, we need to define our prediction function. This will be in the form of y = W*x + b (bias). We need to define W, x, and bias first. Since we have the data we are going to feed the model, we define x as a placeholder where we will asign its value later. W and bias, however, are variables that will be changed when the model iterates. We also have the values of y_true, which are the actual labels, so we define that as a placeholder as well.

# In[ ]:


learning_rate = 0.3
epochs = 100

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_prediction = tf.nn.softmax(tf.matmul(x, W) + bias)


# We need a goal for our model that we're building. If you read my previous kernel on regression, the goal was to minimize the RMSE. In this case, however, ince we are doing Softmax Regression, a common metric to use is cross-entropy, defined below. We would want to minimize our cross-entropy.

# In[ ]:


#our loss function is cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_prediction, labels=y_true))


# We also need a way to evaluate our model's performance for classifying the numbers. We can do this with the accuracy since it's a classification problem. 

# In[ ]:


correct_predictions = tf.equal(tf.argmax(y_prediction,1), tf.argmax(y_true,1))
accuracy_measure = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


# Here is our optimizer. We will use Gradient Descent since it is a very common and powerful method to minimize cross-entropy. We also give it a learning rate that we defined earlier to define how fast it should learn. We need to find a balance in the learning rate since both too low and high will yield bad results.

# In[ ]:


#optimizer we will use
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


# Let's launch our session! In TensorFlow, you always launch a session to run computations and assign placeholder values. We also will train our model during the session. Let's initialize all the variables we defined.

# In[ ]:


sess = tf.InteractiveSession()

init = tf.global_variables_initializer()
sess.run(init)


# Now let's train our model several times, depending on what we set epochs to be. We need a feed dictionary to tell the the function what the placeholder values are. 

# In[ ]:


#training the model with Gradient Descent
for i in range(epochs + 1):
    sess.run(training_step, feed_dict={x: X_train, y_true: y_train})
    print("Epoch " + str(i) + " accuracy: " + str(sess.run(accuracy_measure, feed_dict={x: X_valid, y_true: y_valid})))


# Here is the accuracy of our model for the validation set:

# In[ ]:


sess.run(accuracy_measure, feed_dict={x: X_valid, y_true: y_valid})


# Ok, now let's make some predictions with the test set. We will submit these predictions to the competition.

# In[ ]:


predictions = tf.argmax(y_prediction, 1)
predicted_labels = predictions.eval(feed_dict={x: X_test})
print(predictions.eval(feed_dict={x: X_test}))


# Always, Always, Always remember to close the session after you are done with the computations. 

# In[ ]:


sess.close()


# <a id="p5"></a>
# # 5. Submission
# We will create a Data Frame to submit our predictions we made in the previous section.

# In[ ]:


np.savetxt('submission_softmax.csv', 
           np.c_[range(1,len(X_test)+1), predicted_labels], 
           delimiter = ',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt = '%d')

