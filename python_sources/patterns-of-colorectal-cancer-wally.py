#!/usr/bin/env python
# coding: utf-8

# ## What can you find in this kernel?
# 
# This kernel is part of a series of notebooks that correspond to my personal deep learing journey. Using the dataset of colorectal cancer images I like to understand how convolutional neural network learn and what I can do to improve their success. My acutal level is what you call a beginner in this field and I'm currently interested in details, like...
# 
# * How do the kernel weights look like that were learnt by a CNN?
# * How do they change when we add more layers?
# * Can I analyse exploding or vanishing gradients?
# * Why are some network architectures better than others?
# * How should I preprocess the images given this specific medical problem?
# 
# Etc... This kernel has not the goal to answer all these questions, but by writing it I will hopefully gain more insights and more intuition of the learning process and perhaps find some answers by the way. 
# 
# ### Table of contents
# 
# 1. [Loading packages and data](#load) 
# 2. [Setting up the loss with maximum likelihood](#loss)
# 3. [Exploring the data](#explore)
# 4. [Building a model with tensorflow - Wally](#wally)
# 5. [Data preprocessing](#preprocessing)
# 6. [Going live - A robot starts to learn](#live)
# 7. [What has Wally learnt?](#results)
# 8. [Going deeper - Can we improve Wallys visual range?](#bigeye)
# 9.  [Reduce overfitting - It's not about details, Wally!](#regularisation)
# 
# 
# ### In progress or planned:
# 
# * Regularisation.
# * Improve reproducibility
# * Explore Classification results 

# 
# ### Short overview
# 
# * 8 classes of cancer tissues
# * multiclass classification
# * Kather_texture_2016_image_tiles_5000
#     * 150 x 150 pixel in size
#     * 5000 samples
#     
# ### Sense and meanings of a baseline model
# 
# A general strategy to solve complex problems is to start simple and grow complex during the analysis. This way it is much easier to figure out how to improve and which steps are necessary to build a simple but powerful model that is able to generalize as well. In our case this means to **begin with grayscaled data** instead of colored and a **middle ranged resolution** that is sufficient to work with but does not necassarily lead to computation performance issues. In addition we need a very simple model to play with. 
# 
# * Build and setup a model quicky that can already detect some patterns
# * Understand key concepts and difficulties of your task
# * Experimental platform to increase performance step by step

# ## Loading packages and data <a class="anchor" id="load"></a>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="darkgrid")

from os import listdir
print(listdir("../input"))

import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# For the baseline model we may start with the 64x64 grayscaled images given in *hmnist_64_64_L.csv*. As we have already seen during dataset overview the resolution of them is sufficient to recognize different tissues and cancer classes. 

# In[ ]:


data = pd.read_csv('../input/colorectal-histology-mnist/hmnist_64_64_L.csv')
data.head()


# In[ ]:


print("The data consists of {} samples".format(data.shape[0]))


# ## Setting up the loss <a class="anchor" id="loss"></a>
# 
# Before we start with building a small baseline model, let's discover the data we are working with. As the target distribution is often the origin to setup an objective function or loss, it's worth to look at it. 
# 
# ### The cancer target distribution
# 
# For each image there is only 1 possible true target class $t_{k}$ out of 8 classes. Hence the target per sample is following a one-hot-coding scheme: All elements of t are zero except from the true class that holds one. Consequently we can **describe the target distribution by a multinomial distribution** this way:
# 
# $$ p(t) = \prod_{k=1}^{8} y_{k}^{t_{k}}$$
# 
# Given an image our model tries to predict which cancer class is true and for each possible class it yields a probability $y_{k}$. Looking at the equation you can see that all classes that are not the target class yield $y_{k}^{0} = 1$. Only the probability of the true class contributes with $y_{k}^{1} = y_{k}$. Consequently $p(t)$ reaches its maximum value of 1 when $y_{k} = 1$. This is the case of a perfect prediction! Then our model is sure with 100% that this image belongs to the true class. Using our multinomial distribution we are able to describe how well the model predictions fit to our target values. By fitting a model to our data we are trying to maximize how likely it is that our predictions suite to target values. 
# 
# As a take-away we can say that we can setup a so called likelihood function by probability distributions that tell us how well predictions $Y$ fit to our targets $T$. The function we build this way is often called an objective function. But before we can do this for cancer classification we should describe all data samples, all images, their predictions and target cancer classes.

# ### The likelihood function
# 
# Now we are given a data set of observed images $X = \{ x_{1}, ... ,x_{N} \}$ and target cancer classes $T = \{t_{1}, ... ,t_{N} \}$ whereby each $x_{n}$ stands for one image and $t_{n}$ for its corresponding target vector. The latter has $k$ elements with one hot and all others zero. Let's assume that all $N$ samples were drawn independently from the same distribution. With this independence assumption we can split into $n$ factors and describe the target distribution of our data set as follows:
# 
# $$ p(T|Y) = \prod_{n=1}^{N} \prod_{k=1}^{K} y_{n,k}^{t_{n,k}} $$
# 
# Our goal now is to maximize this function by computing nice predictions that fit well to the targets. In case of neural networks this is done by adjusting the model parameters, the weights between neurons. Let's describe all weights (and biases) by one parameter $\theta$. Then we compute all predictions $Y$ given the images $X$ and weights $\theta$: 
# 
# $$Y = Y(X,\theta)$$
# 
# By using a neural network we try to find weights that are able to describe the mapping between images, predictions and targets for all samples. Hence weights are shared over all samples regardless if we are computing predictions for the first image of our dataset or for the last one. Once we find - estimated - its weights $\theta^{MLE}$ by learning they are fixed and the neural network doesn't change.  
# 
# $$ \theta^{MLE} = \max_{\theta} p(T|X, \theta)  $$
# 
# Probably you know that optimizing a function with respect to some variable means that we need to compute its first and second dervative. We would set the further to zero and would check the solution found given the second derivative. This way we could say if the solution found is a maximum, minimum or a saddle point for example. Regardless if we can do this procedure analytically by hand or using numerical optimization, we need to find the first derivative of our likelihood function with respect to neural network parameters $\theta$ for each layer $l$ and each connection between neurons $i$ and $j$:
# 
# $$\partial_{\theta_{i,j}^{l}} p(T|Y) = \frac{\partial} {\partial \theta_{i,j}^{l}} \prod_{n=1}^{N} \prod_{k=1}^{K} y_{n,k}^{t_{n,k}}  $$
# 
# Due to the products this job is infeasible. To make it tractable we can use the natural logarithm. As it's a monotonically increasing function the maximization of the log-likelihood is equivalent to maximization of the likelihood. In addition it subpresses underflow of numerical precision of the computer caused by products of a large number of very small single sample probabilites. Using the log everything turns out to be nice and smart :-) ...
# 
# $$ \partial_{\theta_{i,j}^{l}} \ln p(T|Y) = \frac{\partial} {\partial \theta_{i,j}^{l}} \sum_{n=1}^{N} \sum_{k=1}^{K} t_{n,k} \cdot \ln y_{n,k} $$
# 
# Now the derivative acts on sums which can be done for each summand separately. Cool! You might have already detected that the log-likelihood looks like a cross entropy apart from a missing minus sign. With adding this minus sign our task changes to minimization of the negative log-likelihood which is our well known loss function. :-) 
# 
# $$ E = - \ln p(T|Y) = - \sum_{n=1}^{N} \sum_{k=1}^{K} t_{n,k} \cdot \ln y_{n,k}  $$
# 

# ### Why is it good to know something about likelihood functions?
# 
# Within this kernel you have seen that one can build a loss function out of a distribution that describes the data. One can see that the choice of the cross entropy below is not arbitrary and has some deeper motivation. Knowing this way can be an entry point to build own likelihoods that suite better to your current problem. One example: What if you have a mutlilabel classification problem to solve but with couplings between target variables? What if independence assumption between observations is not true? If you like you can but something on top and customise your objective as you need it. 
# 
# ### Where to go next?
# 
# For our case study and baseline model we are almost done. We have found a suitable loss function. Where to go next? If you are new to the likelihood concept this way probably full of new stuff. To cool down let's switch do data exploration. After that we will go one step back to find out how the derivatives - alias gradients - of our model will look like and how the model will be influenced during learning. 

# ## Exploring the data <a class="anchor" id="explore"></a>

# ### How many samples per cancer class are present?

# In[ ]:


class_names = {1: "Tumor", 2: "Stroma", 3: "Complex", 4: "Lympho",
               5: "Debris", 6: "Mucosa", 7: "Adipose", 8: "Empty"}
class_numbers = {"Tumor": 1, "Stroma": 2, "Complex": 3, "Lympho": 4,
               "Debris": 5, "Mucosa": 6, "Adipose": 7, "Empty": 8}
class_colors = {1: "Red", 2: "Orange", 3: "Gold", 4: "Limegreen",
                5: "Mediumseagreen", 6: "Darkturquoise", 7: "Steelblue", 8: "Purple"}

label_percentage = data.label.value_counts() / data.shape[0]
class_index = [class_names[idx] for idx in label_percentage.index.values]

plt.figure(figsize=(20,5))
sns.barplot(x=class_index, y=label_percentage.values, palette="Set3");
plt.ylabel("% in data");
plt.xlabel("Target cancer class");
plt.title("How is cancer distributed in this data?");


# ### Take-Away
# 
# * We can see that the **target distributions are balanced**. For each class there are 12.5 % of samples with the corresponding type of cancer. Consequently we should not suffer under model confusion caused by imbalanced classes. 
# * If we like we could split the data such that we will enforce imbalanced classes. This way we could try out different strategies to solve problems that are caused by such imbalance. 

# ### How bright or dark are images per cancer class?
# 
# As the cancer types are probably located in different tissue types the images between classes could differ in their intensities. Thus let's have a look at the overall intensitiy distribution per class:

# In[ ]:


fig, ax = plt.subplots(2,4, figsize=(25,10))
for n in range(2):
    for m in range(4):
        class_idx = n*4+(m+1)
        sns.distplot(data[data.label == class_idx].drop("label", axis=1).values.flatten(),
                     ax=ax[n,m],
                     color=class_colors[class_idx])
        ax[n,m].set_title(class_names[class_idx])
        ax[n,m].set_xlabel("Intensity")
        ax[n,m].set_ylabel("Density")


# ### Take-away
# 
# * The intensity values of all samples in the data are distributed very differently.
# * Whereas the distributions of tumor, stroma, complex and lympho are still similar, they are completely different from debris, mucosa, adipose and empty.
# * Debris and mucosa have broad distributions. The one of debris looks even trimodal. 
# * Images of adipose and empty are of very high intensity. We should find out their meanings as aidpose may be very tough to distinguish from empty. And does empty meen, no image at all?
# * We should try to unterstand if each sample is distributed that way of if we have high variance between samples. 
# * **The variation of image intensity distributions between classes motivates further analysis based on image statistics**. As this is a topic of its own, is covered by a new kernel notebook: [image clustering.](https://www.kaggle.com/allunia/patterns-of-colorectal-cancer-image-clustering)    

# ## Split into train, test and validation

# In[ ]:


from sklearn.model_selection import train_test_split

temp_data, val = train_test_split(data, test_size=0.2, stratify=data.label.values, random_state=2019)
train, test = train_test_split(temp_data, test_size=0.2, stratify=temp_data.label.values, random_state=2020)


label_counts = pd.DataFrame(index=np.arange(1,9), columns=["train", "test", "val"])
label_counts["train"] = train.label.value_counts().sort_index()
label_counts["test"] = test.label.value_counts().sort_index()
label_counts["val"] = val.label.value_counts().sort_index()


# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(label_counts.transpose(), cmap="YlGnBu", annot=True, cbar=False, fmt="g");
plt.xlabel("Cancer label number")


# ## Building a model with tensorflow - Wally <a class="anchor" id="wally"></a>
# 
# Following this [nice tutorial ](http://cs231n.github.io/convolutional-networks/) I like to start with a very reduced network architecture:
# 
# 1. Input Layer
# 2. (Convolutional Layer & Activation) $\cdot $ N
# 3. (Fully connected Layer & Activation) $\cdot$ K
# 4. Fully connected Layer (Output)
# 
# As this is our first session with our baseline model and start simply, we will set N and K to 1. During improvement of the baseline we will explore different additional layers and concepts like dropout, batch normalization, weight regularization, different activation layers and much more. Our baseline model to discover and learn will be a small working unit, nothing amazing. It somehow reminds me of the [small robot WALL-E](https://en.wikipedia.org/wiki/WALL-E), so let's name it similar. 
# 

# ### Wallys Code
# 
# If you unfold the hidden code you can find the network as described above. If you have never written software in tensorflow you might be confused as you can find two different parts in Wally:
# 
# 1. A build Wally method that calls all important methods that we need to build up our tensorflow model. What you obtain is like a **skeleton, a dead shell without life. It describes how things have to be computed but the compution is not done.** This skeleton is called the tensorflow graph.  
# 2. A learn method that calls some attributes of our class that act like **ports to breathe life into Wally**. If you perform session.run (self.loss_op) the code knows that all parts of our skeleton that are needed to compute the loss have to be run. Calling this so-called **tensorflow operations via session.run fills in life to our robot**. 
# 
# I like this imagination of some aritifical organism that you can feed with data and fill with life by running specific functions. :-) I hope this makes it easier to understand for you as well. 

# In[ ]:


class Wally:
    
    def __init__(self, num_classes,
                 n_features,
                 image_width,
                 image_height,
                 image_color_channels,
                 learning_rate):
        # The number of pixels per image
        self.n_features = image_width * image_height * image_color_channels
        # Image dimensions
        self.image_width = image_width
        self.image_height = image_height
        self.color_channels = image_color_channels
        # The number of unique cancer classes 
        self.num_classes = num_classes
        # The learning rate of our loss optimizer
        self.learning_rate = learning_rate
    
    def create_placeholder(self):
        with tf.name_scope("placeholder"):
            self.inputs = tf.placeholder(tf.float32, shape=[None, self.n_features], name="inputs")
            self.targets = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="targets")
    
    # Let's define a method to feed Wally with data and fill in placeholders:
    def feed(self, x, t):
        food = {self.inputs: x, self.targets: t}
        return food
    
    def build_wally(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.create_placeholder()
            self.body()
            self.heart_beat()
            self.wonder()
            self.blood()
    
    def blood(self):
        self.init_op = tf.initialize_all_variables()
    
    def body(self):
        with tf.name_scope("body"):
            
            image = tf.reshape(self.inputs, shape=[-1,
                                                   self.image_height,
                                                   self.image_width,
                                                   self.color_channels]
                              )
        
            self.conv1, self.conv1_weights, self.conv1_bias = self.get_convolutional_block(
                image, self.color_channels, 5, "convolution1"
            )
            self.active_conv1 = tf.nn.relu(self.conv1)
        
            flatten1 = tf.reshape(self.active_conv1, [-1, self.image_height * self.image_width * 5])
            fc1 = self.get_dense_block(
                flatten1, "fullyconnected1", self.image_height * self.image_width * 5, 20
            )
        
            self.logits = self.get_output_block(fc1, "output", 20)
            self.predictions = tf.nn.softmax(self.logits)
    
    def heart_beat(self):
        with tf.name_scope("heartbeat"):
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.targets))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)
    
    def wonder(self):
        with tf.name_scope("wonder"):
            correct_pred = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.targets, 1))
            self.evaluation_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    def sleep(self):
        self.sess.close()
    
    def learn(self, x_train, t_train, x_test, t_test, max_steps):
        self.train_losses = []
        self.test_losses = []
        self.train_scores = []
        self.test_scores = []
        
        x_train = x_train/255 - 0.5
        x_test = x_test/255 - 0.5
        
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)
        
        for step in range(max_steps):
            _, train_loss, train_score = self.sess.run(
                [self.train_op, self.loss_op, self.evaluation_op],
                feed_dict=self.feed(x=x_train, t=t_train)
            )
            
            if step % 5 == 0:
                test_loss, test_score = self.sess.run(
                    [self.loss_op, self.evaluation_op],
                    feed_dict=self.feed(x=x_test, t=t_test)
                )
                train_loss = np.round(train_loss, 4)
                self.train_losses.append(train_loss)
                test_loss = np.round(test_loss, 4)
                self.test_losses.append(test_loss)
                train_score = np.round(train_score, 2)
                self.train_scores.append(train_score)
                test_score = np.round(test_score, 2)
                self.test_scores.append(test_score)
                
                print("Learning step {}".format(step))
                print("Train loss: {}, and train score: {}.".format(train_loss, train_score))
                print("Test loss: {}, and test score: {}.".format(test_loss, test_score))
        
        self.first_kernel, self.first_hidden_neurons, self.first_neurons = self.sess.run(
            [self.conv1_weights, self.conv1, self.active_conv1],
            feed_dict=self.feed(x=x_train, t=t_train)
        )
         
        
    def get_convolutional_block(self, images, in_channel, out_channel, blockname):
        with tf.variable_scope(blockname):
            weights = tf.Variable(
                tf.truncated_normal(shape=[3,3,in_channel,out_channel], mean=0, stddev=0.01, seed=0),
                name="weights")
            bias = tf.Variable(
                tf.truncated_normal(shape=[out_channel], mean=0, stddev=0.01, seed=0),
                name="bias")
            conv_neurons = tf.nn.conv2d(images, weights,
                                        strides=[1,1,1,1],
                                        padding="SAME",
                                        data_format='NHWC',
                                        name="conv_neurons")
            hidden_neurons = tf.nn.bias_add(conv_neurons, bias, name="hidden_neurons")
        return hidden_neurons, weights, bias
    
    def get_dense_block(self, flatten, blockname, n_inputs, n_outputs):
        with tf.variable_scope(blockname):
            weights = tf.Variable(
                tf.truncated_normal(shape=[n_inputs, n_outputs], mean=0, stddev=0.01, seed=1),
                name="weights")
            bias = tf.Variable(
                tf.truncated_normal(shape=[n_outputs], mean=0, stddev=0.01, seed=1),
                name="bias")
            fc_neurons = tf.add(tf.matmul(flatten, weights), bias)
        return fc_neurons
    
    def get_output_block(self, flatten, blockname, n_inputs):
        with tf.variable_scope(blockname):
            weights = tf.Variable(
                tf.truncated_normal(shape=[n_inputs, self.num_classes], mean=0, stddev=0.01, seed=2),
                name="weights")
            bias = tf.Variable(
                tf.truncated_normal(shape=[self.num_classes], mean=0, stddev=0.01, seed=2),
                name="bias")
            hidden_output = tf.add(tf.matmul(flatten, weights), bias, name="logits")
        return hidden_output
    
    def tell_fortune(self, x):
        x = x/255 - 0.5
        predictions = self.sess.run(self.predictions, feed_dict={self.inputs: x})
        return predictions
    


# ## Data Preprocessing <a class="anchor" id="preprocessing"></a>

# Before we start, let's transform all labels to a binary representation with one-hot-encoding as only one cancer class is true for each image.

# ### One-hot-encoding of cancer targets

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

t_train = train.label.values
t_test = test.label.values
t_val = val.label.values

x_train = train.drop("label", axis=1).values
x_test = test.drop("label", axis=1).values
x_val = val.drop("label", axis=1).values

encoder = OneHotEncoder(sparse=False)
t_train = encoder.fit_transform(t_train.reshape(-1,1))
t_test = encoder.transform(t_test.reshape(-1,1))
t_val = encoder.transform(t_val.reshape(-1,1))


# ### Image normalization
# 
# We should normalize the images to improve convergence speed. But how to do it? 
# 
# I often find these kind of techniques but the motivation for them often seem to be unclear:
# 
# * Subtract the mean of each image and scale to unit variance.
# * Just bring the image into a new range. Hence instead of values from 0 up to 255, bring them to 0 up to 1. 
# * Or choose some other min-max-range like -0.5 and 0.5. 
# * Subtract the overall-mean and scale to unit variance from this mean.
# * Class-wise normalization
# 
# Let's take a look at each kind of normalization and let's try to figure out what they are doing and if it makes sense to use them to improve speed or model prediction performance. 

# In[ ]:


your_cancer = "Tumor"
seed = 0


# In[ ]:


def norm_image(image):
    return (image - np.mean(image)) / np.std(image)

def min_max_scaling(image, new_min=-0.5, new_max=0.5):
    return new_min + (image - np.min(image)) * (new_max - new_min) /(np.max(image) - np.min(image))


# Choose a scaling/normalisation method you like to discover:

# In[ ]:


f = min_max_scaling


# In[ ]:


image_ids = data[data.label == class_numbers[your_cancer]].index.values
selected_ids = np.random.RandomState(seed).choice(image_ids, 4)
sns.set()

fig, ax = plt.subplots(4,4, figsize=(20,22))
for n in range(4):
    image = data.loc[selected_ids[n]].drop("label").values
    original = image.reshape((64,64))
    ax[0,n].imshow(original, cmap="gray")
    ax[0,n].set_title("Original image of class \n {}, Id:{}".format(your_cancer, selected_ids[n]))
    
    sns.distplot(image, ax=ax[1,n], color="midnightblue")
    ax[1,n].axvline(np.mean(image), c="r")
    ax[1,n].set_title("Original intensity distribution")
    
    normed_image = f(image)
    
    ax[2,n].imshow(np.reshape(normed_image, (64,64)), cmap="gray", vmin=-2, vmax=5)
    sns.distplot(normed_image, ax=ax[3,n], color="cornflowerblue")
    #ax[3,n].set_xlim([-2,5])
    ax[3,n].axvline(np.mean(normed_image), c="r")
    ax[3,n].set_title("Per-Image normalized \n intensity distribution")
    


# ### Take-Away
# 
# ** under construction **
# 
# #### Per Image Mean-Centering and Unit Variance
# We can see that this kind of normalization causes some problems. Take a look at the first two images. Both have similar skewed intensity distributions in their originals. After normalization we can see that the bright regions (> 200) of the first image now have lower values higher than 3. In contrast the bright values (> 200) of the second image are now given by values higher than 4. **Hence our normalization has caused a shift! Similar bright regions of the originals are now different from each other. I think this does not make sense.** This shift was caused by the skewness of the distribution as the mean is not robust towards outliers and shifted towards higher values. Consequently the mean of the first image is higher than of the second. Consequently images with bright regions always will be normalized to lower values and their distributions are clinched. This comes especially clear for the third image. It's not a good idea to clinch bright images and to strech dark ones. We don't know if cancer cells are always similar bright. If we do this normalization we would end up with different valued cancers that were similar once. But we are dependent on this similarity as our kernel weights only can work this way. 
# 
# #### Min-Max Scaling
# Scaling to min of -0.5 and max of 0.5 does only act on the value range but leaves the shape of the distribution the same as in the original cases. Why is it good to use this scaling method? To find an answer we should take a look at the activation function we are working with. Wally uses relu as activation function. 

# ## Going live! A robot starts to learn... <a class="anchor" id="live"></a>

# ### Setup information
# 
# And let's define some further information Wally needs to learn like:
# 
# * How many pixels / features does an image have?
# * How is the height and width of each image? How many color channels does it have?
# * How big should the learning rate / step size be?
# * How many learning steps (epochs) should be performed?

# In[ ]:


num_classes = len(train.label.unique())
n_features = x_train.shape[1]
image_height = 64
image_width = 64
image_color_channels = 1
eta = 0.01
max_steps = 150


# Ok, now, let's wake up Wally:

# In[ ]:


robot = Wally(num_classes = num_classes,
              n_features = n_features,
              image_width = image_width,
              image_height = image_height,
              image_color_channels = image_color_channels,
              learning_rate = eta)


# ... build it and start learning....

# In[ ]:


robot.build_wally()
robot.learn(x_train, t_train, x_test, t_test, max_steps)


# ## What has Wally learnt? <a class="anchor" id="results"></a>
# 
# We can see from the print output that Wally has learnt something as loss was decreasing for train and test. But the accuracy scores are still low. Hence our first question is...
# 
# ### Do we need more learning steps?

# In[ ]:


sns.set()
plt.figure(figsize=(20,5))
plt.plot(np.arange(0,max_steps,5), robot.train_losses, '+--', label="Train loss")
plt.plot(np.arange(0,max_steps,5), robot.test_losses, '+--', label="Test loss")
plt.xlabel("Learning steps")
plt.ylabel("Loss")
plt.legend();


# ### Take-Away
# 
# We can see that we started to overfit badly after roughly 50 steps. The training loss is still decreasing if you ignore th jumps. In contrast the test loss has sattled down.

# ### How do the learnt convolutional kernels look like?
# 
# If you take a look into the code, you can see that I stored the weight values of the first convolutional layer after learning. This way we can visualize for some example images, what wally tried to extract!

# In[ ]:


example_kernel = np.squeeze(robot.first_kernel)

fig, ax = plt.subplots(1,5,figsize=(20,5))
for n in range(5):
    ax[n].imshow(example_kernel[:,:,n], cmap="coolwarm")
    ax[n].set_title("Weight kernel {}".format(n+1))


# ### Take-Away
# 
# * We can see that all kernels are different and weight spacial positions differently. It looks balanced somehow but perhaps this was not enough and we need more feature maps in our first convolutional layer.
# * What about a bigger receptive field?! Perhaps we need to stack a second convolutional layer on top of the first. 

# ### How good is Wally in classifying cancer? 

# In[ ]:


p_val = robot.tell_fortune(x_val)


# In[ ]:


robot.sleep()


# ## Going deeper - Can we improve Wallys visual range? <a class="anchor" id="bigeye"></a>

# In[ ]:


class BigEyeWally(Wally):
    
    def __init__(self, num_classes,
                 n_features,
                 image_width,
                 image_height,
                 image_color_channels,
                 learning_rate):
        super().__init__(num_classes,
                 n_features,
                 image_width,
                 image_height,
                 image_color_channels,
                 learning_rate)
        
    def body(self):
        with tf.name_scope("body"):
            
            image = tf.reshape(self.inputs, shape=[-1,
                                                   self.image_height,
                                                   self.image_width,
                                                   self.color_channels]
                              )
        
            self.conv1, self.conv1_weights, self.conv1_bias = self.get_convolutional_block(
                image, self.color_channels, 10, "convolution1"
            )
            self.active_conv1 = tf.nn.relu(self.conv1)
            
            self.conv2, self.conv2_weights, self.conv2_bias = self.get_convolutional_block(
                self.active_conv1, 10, 5, "convolution2"
            )
            
            self.active_conv2 = tf.nn.relu(self.conv2)
            
            flatten1 = tf.reshape(self.active_conv2, [-1, self.image_height * self.image_width * 5])
            fc1 = self.get_dense_block(
                flatten1, "fullyconnected1", self.image_height * self.image_width * 5, 20
            )
        
            self.logits = self.get_output_block(fc1, "output", 20)
            self.predictions = tf.nn.softmax(self.logits)


# In[ ]:


robot = BigEyeWally(num_classes = num_classes,
              n_features = n_features,
              image_width = image_width,
              image_height = image_height,
              image_color_channels = image_color_channels,
              learning_rate = eta)


# In[ ]:


robot.build_wally()
robot.learn(x_train, t_train, x_test, t_test, max_steps)


# In[ ]:


robot.sleep()


# In[ ]:


sns.set()
plt.figure(figsize=(20,5))
plt.plot(np.arange(0,max_steps,5), robot.train_losses, '+--', label="Train loss")
plt.plot(np.arange(0,max_steps,5), robot.test_losses, '+--', label="Test loss")
plt.xlabel("Learning steps")
plt.ylabel("Loss")
plt.legend();


# That's pretty interesting! The last time we obtained a score of ~0.6 for test images, but this time we only reached ~0.4. Consequently building a deeper network was not the key! This time overfitting is even more bad!

# ## Reduce overfitting - It's not about details, Wally! <a class="anchor" id="regularisation"></a>
# 
# Ok, going deeper was not advantegous. But we have observed jumps in the loss and overfitting on the training images in both cases. Hence before going deeper, it might be better to reduce overfitting. Let's start simple by forcing the weights of the neural net to be small. For this purpose we can add regularization terms to our loss:

# ## What features are highlighted by the weight kernels?
# 
# You probably know that the weight kernels we learnt for our convolutional layers act like kernels one can use for image preprocessing. Take a look at [these kernels](https://en.wikipedia.org/wiki/Kernel_(image_processing) for example. If you compare the structure of edge detecting kernels with our learnt kernels, we can concluce that ours are different. Thus my first questions is... what are Wallys kernels doing? 

# In[ ]:


sample_maps = robot.first_hidden_neurons

fig, ax = plt.subplots(10,5,figsize=(20,50))
for m in range(10):
    for n in range(5):
        ax[m, n].imshow(sample_maps[m,:,:,n], cmap="coolwarm")
    


# Well.... so far I can't conclude something out of it. :-)
