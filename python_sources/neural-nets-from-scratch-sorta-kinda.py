#!/usr/bin/env python
# coding: utf-8

# **Neural Nets from Scratch with Numpy**
# 
# The newest generation of machine learning researchers are inheriting an incredible set of tools. An incredible set of tools which can obscure the elegant mathematics driving it all. Additionally, there are many who just need code to play with to understand. In the end, neural nets are nothing more than fancy approximation machines. This notebook tries to address both problems by collecting a few simple ideas and examples for you to play with. My goal here is not for you to have a compete theoretical understand of neural nets, but to gain an end-to-end perspective on all their moving parts. So for some, this will be a programming challenge, for others a mathematical one. I do not think it's necessary to understand these concepts 100%, but I hope they will provide you with intuition as your deep learning applications grow more complex. 
# 
# We will start with the XOR problem. For context, check out this article.
# http://mnemstudio.org/neural-networks-multilayer-perceptrons.htm
# 
# **If you haven't used a notebook before**
# 
# All you really need to know is that, shift + enter evaluates a cell!
# For the programmers out there, the notebook scope acts like a package, so you'll see me coding as if we're in a class definition. With some sloppiness.
# 
# **If you are a notebook pro**
# 
# There's a lil button in the top right that'll let you download the `.ipynb`, and then click on the **Data** tab above to download the `.csv`'s you need!
# 
# **Outline**
# * What is a neural net?
# * Your first neural net in two minutes
# * The XOR Problem
# * Exercise
# * MNIST Handwritten Digit Data Set
# * Getting the data
# * Setting up to train
# * Results
# 
# # What is a neural net?
# 
# An approximation machine! Okay that might be a bit tongue-in-cheek. But, first, let's actually talk about what a neural net is NOT:
# - a neural net is not AI
# - a neural net is not a black box
# - a neural net is not something only grad students and math experts can understand
# - a neural net is not the solution to all current hard problems in computer science
# 
# That said, what is a neural net? Class?
# 
# A neural net is a *statistical model*. Statistical models are mathematical constructs which allow us to understand large amounts of data quickly, and hopefully generalize so that we can use our understanding for *predictive modeling*. Neural nets happen to be highly flexible and powerful statistical models, allowing them to do all the cool things you hear about them doing. The biggest downside to neural nets in industry is their *non-interpretability*. That means even when trained, it's hard to explain exactly why it makes a particular decision. With other statistical models, there is a parameter, or a simple equation we can point to, but with neural nets, all you get is big matrices. This shortcoming is generally overlooked in domains where they outperform any other model so thoroughly (like computer vision).
# 
# # Your first neural net in two minutes
# 
# Our only dependency provides an easy interface for matrix algebra:

# In[ ]:


import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
		e = nonlin(x)
		return e*(1-e)
	return 1/(1+np.exp(-x))


# Our training data, which is the XOR map:

# In[ ]:


X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
		[1],
		[1],
		[0]])


# This is a collection of four data points. Each row of `X` corresponds to the row in `y`. You might be tempted to say there are three variables, but the third column of `X` is the same for all rows, so for the purpose of guessing the correct `y` value, it doesn't help. 
# 
# This collection of data actually defines something very important to computer science, and we'll get to that next. First.... we train a network:

# In[ ]:


np.random.seed(1)

# randomly initialize our network weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):
	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


# **BOOM**
# A neural net. And yes, it should say error over and over with a wee small number (that gets smaller). But what did we even do? Well, since there are plenty of other resources out there on neural network theory, we're gonna take the hackers' approach and work from the inside out. If you want information on theory, find the links from iamtrask in the references section. This code is reproduced from his [excellent gist](http://iamtrask.github.io/2015/07/12/basic-python-network/), and I don't want to copy **all** his content. So, from here on out, we answer our questions not with history or theory, but by **playing with the code**. This is an excellent habit to get in for machine learning and data science, as the examples available online are numerous, but data scientists writing (and code commenting) skills are dubious at best. It's hard to hold it against the community, when it's an international group, pushing technology to its limits every day. 
# 
# That said, we did just train a neural net, probably sucessfully. We don't, however, have any of the handy features provided by neural net frontends like Keras. So, we need to make it all ourselves. But, let's start by figuring out what we just learned
# 
# # The XOR Problem
# 
# So, note that `y` takes the value `1` only when exactly one input variable is `1`. Otherwise, `y` is `0`. In a true machine learning context, you might call this the data, although in truth it's actually a map that defines the XOR function. XOR means "exclusive or" and is a major part of computer architecture. When neural nets were first explored, Minksy and Papert (1969) showed that this was a major issue for them (then simply called perceptrons). It was not until 1985 that the issue was resolved with the addition of the **hidden layer**. You'll have a chance to see what that is in a bit.  

# # Exercise 1: Line By Line

# ## Setup
# Notice, that we created the objects ```syn0``` and ```syn1``` outside the loop, then iterated a whole bunch of times, modifying them as we go. So what do they look like? 

# In[ ]:


syn0


# In[ ]:


syn1


# That's still pretty obtuse. We have a bunch of numbers. So what? Well, each iteration, we did stuff to each of these things with the values of ```l1``` and ```l2```. What are their types?

# In[ ]:


type(syn0), type(syn1)


# Okay, they are "`numpy.ndarray`s". So let's check those lines out again, outside of the loop.
# 
# ## Gettin' in the loop

# In[ ]:


# Feed forward through layers 0, 1, and 2
l0 = X
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))
# how much did we miss the target value?
l2_error = y - l2
print(l2_error)


# All of these are very small numbers - which is good, because we've already trained our network, and this variable is called ```l2_error```! So, what about where that comes from? ```y``` and ```l2```? Well, ```y``` was the labels in our training set. So, ```l2``` must be our predictions. Let's look:

# In[ ]:


l2


# These values approximately represent the probability that a given data point is a ```1```.  They're not truly probabilities, but they act sort of like probabilities as the mathematics forces these numbers to stay between 0 and 1. What's clear here is that if we round all these values, we get the array ```[0, 1, 1, 0]```, precisely the values of ```y```. So, our neural net has learned our dataset about as well as can be expected. But, you may still be uncomfortable about how we got here. Good! We've already covered a lot. So our next steps will small. We're going to investigate some of those `numpy` functions we're using. 
# 
# ## Our lil friend numpy
# 
# The first call to `numpy` is `np.random.seed(1)`, which is effectively the same as Pythons `math.random.seed()`. Next, we call `np.random.random()` twice. So let's play with that!

# In[ ]:


np.random.random()


# Okay, that may have been a bit predictable. Just for fun, re-run that cell a few times. Now, above, we used some parameters. You might have a guess as to what they're for, but let's confirm your intuition with experimentation!

# In[ ]:


np.random.random((10, 10))


# My personal choice of parameters may have been excessive, but it illustrates the point nicely. That just made us a nice 10x10 table of random numbers. None of them reach 1, or go below 0, so they must be in (0, 1). Finally, the actual line is `2*np.random.random((3,4)) - 1`, so that's going to move the range to (-1, 1). Neat!
# 
# Ok, the next interesting call is `l1 = nonlin(np.dot(l0,syn0))`. We know what nonlin is, it was a defined mathematical function that we actually don't really need to worry about. Just read the definition! But what about the `np.dot(...)` part? Anyone who has used linear algebra should be familiar with the dot product, but if you're not, that's about to change!

# In[ ]:


m = np.random.random((2,2))
n = np.random.random((2,2))
k = np.dot(m,n)
print("m:", m)
print("n:", n)
print("k:", k)


# So, our dot product took two 2x2 matrices and produced another 2x2 matrix. Try placing different parameters for `m` and `n`. When do you get errors? When does it work? What changes the size of `k`? Also, while we're here, let's try something with `nonlin()`:

# In[ ]:


j = nonlin(k)
print("j:", j)
print(nonlin(5))


# So, `nonlin()` can take or matrices or just plain numbers, and happily computes away. To get some intuition for the function, swap out the values here.
# 
# The next calls to `numpy` are `np.mean` and `np.abs` in the same line, but these should be readily interpretable. One gives the mean of a vector, the second gives the absolute value of a matrix. If you're still confused, play with the code! There's a more interesting call for us to move on to: `l1_error = l2_delta.dot(syn1.T)`. This *should* still run, unless you've messed around a lot:

# In[ ]:


l2_delta.dot(syn1.T)


# Okay, well... What? Numbers numbers numbers....

# In[ ]:


syn1.T


# In[ ]:


syn1


# Ah! So, that `.T` sort of rotates our matrix? Let's do another experiment.

# In[ ]:


m


# In[ ]:


m.T


# What changed? Only two values? They swapped....? Hmmmm....

# In[ ]:


m = np.random.random((3,3))


# In[ ]:


m


# In[ ]:


m.T


# Aaaah, so now everything but the diagonal flipped! So, `.T` is a funny little matrix-flipping operation. 
# 
# This is the last of the `numpy` functions used. You shouldn't expect to be able to define them all clearly, but now you have seen the moving parts. So let's forge ahead, like the fearless hackers we are! For further information on the math and theory this example, [visit the gist by iamtrask](http://iamtrask.github.io/2015/07/12/basic-python-network/) in the references section. We're gonna take a lil break, then start using our new knowledge!
# 
# **BREAK**

# # Exercise 2: Augmenting the Code
# 
# Now, armed with the knowledge we gained above, we're going to build out some utility functions to make training the neural net easier. 
# 
# ## Initializer
# 
# This just does the same thing as the first couple lines of code:

# In[ ]:


def initialize_net(inputs, hiddens, outputs):
    # randomly initialize our weights with mean 0
    syn0 = 2*np.random.random((inputs,hiddens)) - 1
    syn1 = 2*np.random.random((hiddens,outputs)) - 1
    
    return syn0, syn1


# ## Predictions
# This will give us our predictions, from the first couple lines inside the loop:

# In[ ]:


def predict_proba(x):
    # Feed forward through layers 0, 1, and 2
    l0 = x
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    
    return l2, l1

# don't use this with a single output network!
def predict(x):
    return predict_proba(x)[0].argmax()


# # The Trainin Loop
# Finally, we'll put the training loop into a function so we can call it easily:

# In[ ]:


def fit(X, y, syn0, syn1, cycles = 6000, alpha = 0.01):
    for j in range(cycles):
        predicts, hiddens = predict_proba(X)

        # how much did we miss the target value?
        error = y - predicts

        if (j% 100) == 0:
            print("Error:" + str(np.mean(np.abs(error))))

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        predicts_delta = alpha*error*nonlin(predicts,deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        hiddens_error = predicts_delta.dot(syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        hiddens_delta = hiddens_error * nonlin(hiddens,deriv=True)

        syn1 += hiddens.T.dot(predicts_delta)
        syn0 += X.T.dot(hiddens_delta)


# ## Test our new functions
# Let's just train the net on our XOR data before moving on to make sure everything is working nicely

# In[ ]:


# training variables
alpha = 0.005
cycles = 6000

syn0, syn1 = initialize_net(3, 4, 1)
    
fit(X, y, syn0, syn1, cycles, 1)


# That's a lot of error prints... Heh. Oh well. **TALLY HO**

# # Learning The MNIST Handwritten Digit Data Set
# 
# But first, I must admit a small lie. We need a few more packages to handle and display data. Our neural net will still only depend on `numpy`.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
data = pd.read_csv("../input/train.csv")[:50]


# Pandas dataframes have many handy built ins:

# In[ ]:


data.head()


# In[ ]:


data.describe()


# So, our data comes in rows where the first entry is the label of the image, and all following entries are pixel values. Check out this competitions data page for more info on the data format. Now, let's take a look.

# In[ ]:


image_width,image_height = 28,28

images = data.iloc[:,1:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))


# In[ ]:


# display image
def display(img):
    
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)

# output image     
IMAGE_TO_DISPLAY = 42
display(images[IMAGE_TO_DISPLAY])


# Well, that's nice. 

# In[ ]:


labels = data['label'].values
labels[42]


# Also nice! 
# 
# So, we've got our images, we can peek at them if we want, and we have a nice MLP example. Time to get dirty. There's one more important thing... Our class labels are in digital (0-9) format. We don't actually want this. Because the ordering of our classes is not relevant to classifying the images, trying to predict the numerical values 0-9 injects a lot of unecessary difficulty into the task. Instead, we'll one-hot encode the labels, and build a network that has one neuron for each class. Then, we'll label our prediction with the neuron that has the highest output value. 

# In[ ]:


# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels, 10)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))
print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))


# There we are. Our labels are in the format we want. 
# 
# ## Time to train!

# In[ ]:


X = images
y = labels
np.random.seed(42)

# training variables
alpha = 0.005
cycles = 1000

syn0, syn1 = initialize_net(784, 15, 10)
    
fit(X, y, syn0, syn1, cycles, alpha)


# ## Frickin' sweet new bike! Now let's validate
# No but really, awesome! Something good happened! Now we need to find out how well we did. We do this with a *validation set*. Validation sets are segments of our training data we don't let the neural net train on, so we get a really good idea of how well it has learned.

# In[ ]:


val_data = pd.read_csv("../input/train.csv")[50:100]
val = val_data.iloc[:,1:].values
val = val.astype(np.float)

# convert from [0:255] => [0.0:1.0]
val = np.multiply(val, 1.0 / 255.0)

val_labels = val_data['label'].values
val_labels_oh = dense_to_one_hot(val_labels, 10)
val_labels_oh = val_labels_oh.astype(np.uint8)

print('val({0[0]},{0[1]})'.format(val.shape))


# ## Another handy utility function to make validation easier

# In[ ]:


def display_predict(i):
    IMAGE_TO_DISPLAY = i
    print ('val_labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,val_labels[IMAGE_TO_DISPLAY]))
    print("prediction: ", predict(val[i]))
    display(val[IMAGE_TO_DISPLAY])


# In[ ]:


display_predict(np.random.randint(50))


# In[ ]:


def score(imgs, labels, labels_oh):
    l0 = imgs
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l2_error = labels_oh - l2
    predicts = [probs.argmax() for probs in l2]
    num_correct = sum([p == l for (p, l) in zip(predicts, labels)])
    print("Validation Error: {0}, {1} of {2} correctly labeled".format(str(np.mean(np.abs(l2_error))), num_correct, len(predicts)))


# In[ ]:


score(val, val_labels, val_labels_oh)


# **Results**
# 
# This may be very disappointing. But, actually, consider, what did you expect? We effectively took "black box" code and sucessfully adapted it to a totally different data set (from three input values to over seven hundred). How much accuracy can we expect? Well, for a given classification, a random selection has a 10% of getting a correct label. So, that's our baseline score. If we had gotten less than 5 correctly labelled images, then we would have truly messed something up. If we got 10-15, we would know that our network had definitely started learning, but something might be wrong. Since we got exactly 50% correct, we know that the algorithm is working fine, and we really just need more data and more training time. Remember that our training and validation sets only contained 50 images each. For thoroughness, let's conclude by testing our network with the full remaining training data. **Be warned** this could take a while to run.

# In[ ]:


test_data = pd.read_csv("../input/train.csv")[50:]
test = test_data.iloc[:,1:].values
test = test.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test = np.multiply(test, 1.0 / 255.0)

print('test({0[0]},{0[1]})'.format(test.shape))

test_labels = test_data['label'].values
test_labels_oh = dense_to_one_hot(test_labels, 10)
test_labels_oh = test_labels_oh.astype(np.uint8)


# So now we're testing with all the training data we didn't see. To compete in this competition, you'll train with the whole ```train.csv``` file, and submit your predictions from the ```test.csv``` file.

# In[ ]:


score(test, test_labels, test_labels_oh)


# So, we got almost 50% correctly labeled. Our take away is that even with only 50 training images, we can get decent generalization. 
# 
# This concludes this workshop!

# **Thank you**
# Utility functions from [fellow kaggler](https://www.kaggle.com/kakauandme/tensorflow-deep-nn)
# 
# With many thanks to [iamtrask](https://github.com/iamtrask)
# 
# The intro content below is based on his article, [A Neural Net in 11 Lines of Python](https://iamtrask.github.io/2015/07/12/basic-python-network/)
# 
# To learn about another neural network architecture, check out his other article, [Anyone Can Learn To Code an LSTM-RNN in Python](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)
# 
# More example problems, check out this handy [list of machine learning research data sets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)
# 
# [And an awesome paper with awesome visualizations](http://scs.ryerson.ca/~aharley/vis/)
# 

# In[ ]:




