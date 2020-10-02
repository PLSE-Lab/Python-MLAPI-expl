#!/usr/bin/env python
# coding: utf-8

# # Intro to Deep Learning and Keras
# Jake Lee, TA for COMS 4701 Fall 2019
# 
# *Note: this is absolutely, 100%, OPTIONAL.*
# 
# ## Introduction
# Neural networks and deep learning are only covered in the last two lectures, and they're not a major part of the AI curriculum (there are entire courses dedicated to DL). However, due to the effectiveness of the method for computer vision, we knew that if we held a image classification competition, students would want to use deep learning. Therefore, I'm putting together this short tutorial to **give everyone a fair chance at trying out deep learning and seeing how it performs** against traditional ML techniques for the MNIST classification task.
# 
# This will also be covered on December 3rd's lecture.
# 
# ## Background
# **Lecture 25 Slides 2-6**
# 
# In the Nov. 19th lecture, we discussed Perceptrons and how they learn a linear separator given training data (assuming a binary case). However, most problems are not linearly separable.
# 
# It was determined that *multiple layers* of *multiple perceptrons* could, infact, learn non-linear separators for such problems. A simple example is the [XOR problem](https://towardsdatascience.com/perceptrons-logical-functions-and-the-xor-problem-37ca5025790a), which cannot be modeled by a single perceptron, but can be modeled by several perceptrons in multiple layers. In fact, the Universal Approximation Theorem suggests that a network of layers of perceptrons (a neural network) can approximate any continuous function (although it doesn't say anything about whether it can be efficiently learned).
# 
# While the concept of neural networks has been around since the 40s, recent improvements in computation, data storage, and memory have made them viable for many different applications. A famous example is Krizhevsky et al.'s AlexNet in 2012, which won an image object classification challenge (ILSVRC2012), beating the next best method by 10% (a significant leap at the time).

# # A Typical CNN Image Classifier
# CNN stands for Convolutional Neural Network.
# 
# ## Architecture
# A typical convolutional neural network looks like this. Feel free to research each layer to learn their function, but I'm providing this as a template for those that just want to try it out. This is also very hand-wavy - if you're interested in reading into this more thoroughly, feel free to look up tutorials, guides, and lectures online. A good resource for more in-depth descriptions is [CS231n](http://cs231n.github.io/convolutional-networks/).
# 
# - An input layer (for MNIST, 28x28x1)
#   - This is where the data comes into the network.
# - A convolutional layer
#   - These are perceptrons good at capturing spatial information.
#   - As covered in class, each perceptron needs an activation function. We'll use ReLU (Rectified Linear Units), which is the most popular/effective.
# - A few more conv layers
#   - A deeper network can learn better, but it's also more complex and takes longer to train.
# - A max pooling layer
#   - This is like resizing an image to half its length and width. It helps us go from 28x28 input to 10 output.
# - A dense or "fully-connected" layer
#   - These perceptrons are connected to every node before it, similar to the neural net covered during lecture. Spatial information is lost, but more abstract class information is learned.
# - A dense layer that equals the number of classes
#   - This is the output layer. Each node represents a class, and the highest value is the prediction.
# - A softmax layer
#   - This layer scales the values so they add up to 1, so that they may be interpreted as "confidence percentages".
#   
# Essentially, we input an image, get some spatial information out of it with convolutional layers, abstract that information to class information with dense layers, then predict which class that image must be in with the last layer.
# 
# Obviously, it's far more complicated and nebulous, but this is enough to get started.
#   
# ## Training
# We train this network exactly like how we train a single perceptron. We give it *iterations* of training data until it updates its weights and it learns the task. We call each update of the weights an *iteration*. We may also want to keep repeating training data. Every time we exhaust the training data and start repeating, we call it an *epoch*.
# 
# You may be wondering how you can update the weights when nodes are chained to each other. This is called backpropagation - I don't have space to explain it here, but feel free to research online. Also, courses such as NLP (Prof. Collins), Deep Learning (Prof. Drori), and other DL courses will cover backprop in detail.
# 
# I'm partial to *Deep Learning* by Goodfellow, Bengio, and Courville, which is available for free online. [Section 6.5](http://www.deeplearningbook.org/contents/mlp.html) covers backpropagation in detail.
# 
# ## Testing
# After training, after all of the weights have been updated, we can feed the network an input (just like a single perceptron) to see what it classifies it as.

# # Keras Implementation
# Enough of the boring (but important!) stuff, let's get coding. Thankfully, over the years, frameworks such as Keras, Tensorflow, Pytorch, etc. have made it very easy to build your own neural network.
# 
# We'll use Keras since it's the easiest. https://keras.io/examples/mnist_cnn/
# 
# We'll also just use MNIST, without the student-collected data. You'll have to merge the two datasets by yourself.

# In[ ]:


import numpy as np
import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# This is a quirk of keras; since the images are grayscale,
# we need to add an axis so the shape is (60000, 28, 28, 1)
# instead of (60000, 28, 28)

x_train = x_train[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]

# We're also going to convert 0~255 to 0~1 float.
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)
x_train /= 255
x_test /= 255

# Finally, the classes need to be one-hot encoded.
# That is:
# 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0]
# etc.
# This is to match what the network will output - 
# there are 10 nodes at the end, each with its own
# confidence of its class. The ground truth should be
# 100% confidence of the true label.

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# OK, Keras is imported and the data is loaded. Now, let's build our neural network. I'm going to follow the template above.

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
#                        Remember these?

# By the way, we really like powers of 2 for the number
# of nodes at each layer.
model = Sequential([
    # input layer, 16 conv (spatial) perceptrons of size (3,3)
    # image shape is (28, 28, 1). If it was color it'd be (28, 28, 3)
    Conv2D(8, (3,3), activation='relu', input_shape=(28, 28, 1)),
    # Now for the max pooling to make the size smaller
    MaxPooling2D(pool_size=(2,2)),
    # Flatten before sending to Dense (2D to 1D)
    Flatten(),
    # Output layer with 10 nodes for 10 classes, with softmax
    Dense(10, activation='softmax')
])


# That wasn't so bad! Now, one more step to tell Keras how to train the model:
# 
# - We'll use the Categorical Crossentropy loss function, a standard for classification tasks.
# - We'll use Stochastic Gradient Descent, which is just gradient descent with randomness included.

# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.SGD(),
             metrics=['accuracy'])


# And finally, we can train it! We can also give Keras the validation data so that we can see how it's doing on both data that it has seen before (train), as well as data it hasn't seen before (validation).

# In[ ]:


import time
start = time.time()
model.fit(x_train,        # training data
          y_train,        # training labels
          batch_size=16,  # how many training examples you want to give at once
          verbose=1,      # print progress in console
          validation_data=(x_test, y_test),  # validation data to check generalization
          epochs=5)       # how many times to go through the entire training set
end = time.time()
print("Training took", end-start, "seconds.")


# In just 67 seconds, our (comparatively) tiny CNN was able to reach 97.97% validation accuracy. This is great! Larger, more complex neural networks are able to reach >99% accuracy. 
# 
# Feel free to take the neural network provided and modify it - add more layers, more nodes at each layer, add regularization, data augmentation, learning rate scheduling, etc. To try and get as close to 100% as possible!
# 
# Also, this task will be more difficult as we collect more data ourselves (instead of just using MNIST). Also, note that for the actual Kaggle competition, you are testing on the collected dataset, not the provided MNIST test set.

# # Conclusion
# We discussed what deep learning and neural networks are, and we implemented a very simple network that did very well. Now try and get to the top of the leaderboard!
# 
# As always, if you enjoyed the writeup, click below to upvote!
