#!/usr/bin/env python
# coding: utf-8

# # This notebook summarizes some basics of NN-building with Keras
# 
# Includes some notes that might be useful for newbies.

# In[ ]:


################################################################################################
#                                      Used packages:                                          #
#             python 3.7.3, numpy 1.16.4, pandas 0.24.2, matplotlib 3.1.0, keras 2.2.4         #
################################################################################################


# **Import necessary libraries**

# In[ ]:


import numpy as np   # linear algebra
import pandas as pd  # data processing

# The subprocess module enables you to start new applications from your Python program.
# subprocess command to run command with arguments and return its output as a byte string.
from subprocess import check_output


# In[ ]:


import matplotlib.pyplot as plt  # figures
get_ipython().run_line_magic('matplotlib', 'inline')


# **Read train and test image datasets.**
# 
# Each image is $28$ pixels in height and $28$ pixels in width, giving $784$ pixels in total. 
# Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel,
# with higher numbers meaning darker. This pixel-value is an integer between $0$ and $255$, inclusive.

# In[ ]:


# Read data from files. Data belongs to <class 'pandas.core.frame.DataFrame'>
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv") # DataFrame shape (42000, 785), DataFrame type int64
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")   # DataFrame shape (28000, 784), DataFrame type int64

# return numpy representation of DataFrame object
test_data = test.to_numpy(dtype='float32') # does exactly the same as the line above but for the test dataset


# **Split train DataFrame into labels (answers) and actual data with intensities at each pixel.** 
# 
# Perform indexing of our data. DataFrame.iloc() is a purely integer-location based indexing for selection by position.

# In[ ]:


train_data = train.iloc[:, 1:]                         # pixel data are indexed
train_labels = train['label'].values.astype('float32') # just a one-dimensional ndarray, with answers for 42000 samples.


# **Show structure of train_data object**
# 
# Each column represent an image from the training set with $42000$ images overall.
# Each row consists of $784$ columns representing an intensity, that is an $int64$ integer from $0$ to $255$, at each pixel.

# In[ ]:


#train_data


# **Show some training images with labels**
# 
# Show $5$ images in a row starting from image number $205$.

# In[ ]:


# original train_data shape is (42000, 784) We reshape it into s square form: (42000, 28*28)
# In order to reshape we must convert the DataFrame into Numpy array.
train_data_image = train_data.to_numpy(dtype='float32').reshape(train_data.shape[0], 28, 28)

# show num_image images from the train set starting from startin_image 
starting_image = 204
num_image = 5 

for idx,i in enumerate(range(starting_image,starting_image+num_image)):
    plt.subplot(1,num_image,idx+1)
    # show an image. cmap is a colormap = {'gray','hot', 'autumn', 'winter', 'bone'} 
    plt.imshow(train_data_image[i], cmap=plt.get_cmap('hot'))
    # show a label (answer) 
    plt.title(train_labels[i])


# **Neural nretwork data must be renormalized into $[0,1]$ interval**
# 
# Initial intensities are between $0$ and $255$. 

# In[ ]:


train_data = train_data/255  
test_data = test_data/255    


# **Convert a class vector to binary class matrix for use with categorical_crossentropy loss function.**

# In[ ]:


from keras.utils import to_categorical

train_labels = to_categorical(train_labels)


# # Define the layers of your Neural Network
# 
# Keras models can be built either in a **sequentional** layer-by-layer way, as it is done in this notebook, or by using **the functional API**. The latter method provides user with more functionalities enabling one to set up a complex topology of the network. For example, one can construct a network where a layer is connected to more than just the previous and next layer, or where some layers are shared or have multiple inputs or outputs. Here we biuld our network as a sequeance of fully connected layers. To do this, we create an instance of the *Sequential* class of Keras.
# 
# *Dense* defines a layer of a Neural Network. A layer consists of certain number of neurons given by a user while building a network. Each neuron recieves some input signals that are either input data or outputs of neurons from the previous layer. These signals are then transformed in an output. The following mathematical expression demonstrates the work of a layer: 
# 
# \begin{equation}
# \mathbf{a_1} = f(W\cdot\mathbf{a_0}+\mathbf{b})
# \end{equation}
# 
# where $\mathbf{a_1}$ is an **output** vector, $W$ is the input matrix, $f$ is the **activation** function, $\mathbf{a_0}$ is a **kernel** weights matrix created by the layer, and $\mathbf{b}$ stands for the **bias** vector.
# 
# **units** parameter in *Dense* is a positive integer defining dimensionality of the output space. Essentially it is the total number of neurons in the next layer.
# 
# First layer takes an input image array of *input_dim=*$28\times28$ pixels and transforms it into a vector of dimension *units* = 300. The next two layers consist of $300$ and $100$ neurons correspondingly. These intermediate layers are called "hidden" layers. The last fourth layer has 10 neurons corresponding to 10 possible answers $0,1,..,9$. 
# 
# Each neuron is connected to all neurons in the previous layer. It recieves signals $a_1,a_2,...,a_n$ each of which has a weight $w_i$. The resulting signal arriving at a neuron is just a linear combination of all signals from the previous layer to a given neuron plus a bias parameter $b$
# 
# $$w_1a_1+w_2a_2+...+w_na_n + b$$
# 
# To activate a bias parameter one needs to set up **use_bias = True**, it tells whether the layer uses a bias vector.
# The weights and biaseses are tuned during the machine learning. This linear combination is then scaled to $[0,1]$ interval by means of the **activation** function $f$: 
# 
# \begin{equation}
# a^{(m+1)} = f(w_1a_1^{(m)}+w_2a_2^{(m)}+...+w_na_n^{(m)} + b^{(m)})
# \end{equation}
# 
# Here *m* is a layrer number. The matrix form of this equation for a layer is given above, Some examples of activation functions include *elu,softmax,selu,softplut,softsign,relu,tanh,sigmoid,hard_sigmoid,exponential,linear*.
# There are also advanced activation functions which can be found in *keras.layers.advanced_activations*. In the example above we used *relu* - rectified linear unit which returns $max(x,0)$ -, and *softmax* - softmax activation function.

# In[ ]:


# import some machine learning libraries from keras and sklearn
from keras.models import Sequential
from keras.layers import Dense

# We use the Sequential model that is a linear stack of layers.
# Create an empty Sequential model object
model = Sequential()

# We add 3 fully-connected layers one by one: 

# The first layer processes the inputs. Thus, one needs to specify the dimensions input_dim: 
model.add(Dense(units=300, use_bias=True, activation='relu', input_dim=(28*28)))
model.add(Dense(units=100, use_bias=True, activation='relu'))
model.add(Dense(units=10, use_bias=True, activation='softmax'))


# **Now let's visualize the network model using plot_model.**
# 
# You need to install pydot first: *conda install pydot*

# In[ ]:


from keras.utils import plot_model

# save model architechture graph into file
plot_model(model, show_shapes=True,to_file='model.png')


# # Compile your neural network model
# 
# The goal of machine learning is to reduce the difference between the predicted output and the actual output. This difference is also called as a **cost function** or **loss function**. The loss function could be a mean squared error, categorical crossentropy, Poisson loss or many [other functions](https://keras.io/losses/). Our goal is to **minimize** this function by finding the **optimized** value for neural network parameters (weights and biasis). Minimization is done by updating the network parameters in the negative gradient direction.  Loss function acts as guides to the terrain telling optimizer if it is moving in the right direction to reach the bottom of the valley, the global minimum. The **gradient descent** can be done with a variety of algorithms including Stochastic Gradient Descent (**SGD**), Root Mean Square propagation (**RMSprop**), Adaptive Gradient Algorithm (**Adagrad**) and many [other methods](https://keras.io/optimizers/). [Here](https://medium.com/datadriveninvestor/overview-of-different-optimizers-for-neural-networks-e0ed119440c3) and [here](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) you can find more details about gradient descent methods. Moreover, the performance of a model can be evaluated by means of one of the [metrics](https://keras.io/metrics/). For example, accuracy metrics calculates how often predictions matches labels.
# 
# **Compile** method configures the learning process before training a model. It receives three arguments:
# 
# - **optimizer** is the gradient descent <u>**method**</u> used for optimization.
# 
# - **loss** is the objective <u>**function**</u> that the model will try to minimize. It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse), or it can be an objective function.
# 
# - **metrics** is a function that is used to judge the <u>**performance**</u> of your model. 
# 
# An important hyperparameter [to tune](https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/) is a **learning rate**. It defines how much we are adjusting the weights of our network with respect the loss gradient (see [here](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10) and [here](https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b) )
# 
# In the model below, we use the Stochastic Gradient Descent (SGD) algorithm with learning rate $lr = 0.001$ to optimize the categorical crossentropy loss function and evaluate the accuracy of the model with the *accuracy* metric.
# 

# In[ ]:


from keras.losses import categorical_crossentropy, categorical_hinge
from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adamax

model.compile(optimizer=Adagrad(lr=0.005), loss=categorical_crossentropy, metrics=['categorical_accuracy'])


# # Fitting (learning). Find the NN parameters to minimize the loss function
# 
# After defining the NN architecture (**model**) and the optimization algorithm (**compile**), we will train our network on the train dataset. Trains the model for a fixed number of epochs (iterations on a dataset).
# 
# **Epoch** is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE. Since one epoch is too big to feed to the computer at once we divide it in several smaller **batches** with **batch_size** each. Each epoch requires a number of iterations that is equal to nymber of training examples devided by batch size. As the number of epochs increases, more number of times the weight are changed in the neural network and the curve goes from underfitting to optimal to overfitting curve.
# 
# - **x**: Input data
# - **y**: Target data
# - **validation_split**: Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. 
# - **epochs**: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
# - **batch size**: Number of samples per gradient update.
# 
# **Output** has a history attribute where a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable) are stored.

# In[ ]:


fit = model.fit(x=train_data, y=train_labels, validation_split=0.05, epochs=30, batch_size=50)#, shuffle=True)


# # Analysis of learning metrics
# 
# There are two main metrics: **loss function** and **accuracy**. The former must be minimized while the latter must be maximized. Both metrics are computed on the main part of your training dataset as well as on a small validation set that is set apart from the main set from the beginning. Validation dataset is used to estimate your prediction. The splitting of the original training dataset into two has been done in the fit method. In our example the validation dataset consists of $5\%$ of the whole training set (i.e. *validation_split=0.05*). The metrics change at each epoch *i.e.* each run of the main training set through the neural network. The metrics evolution can be accessed *via* the **history** callback.

# In[ ]:


# list of computed metrics names
metrics = list(fit.history.keys())
metrics


# **First we analyze the loss function**

# In[ ]:


loss_values = fit.history[metrics[2]]
val_loss_values = fit.history[metrics[0]]

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo')
plt.plot(epochs, val_loss_values, 'b+')

plt.xlabel('Epochs')
plt.ylabel('Loss')


# **Then we analyze the accuracy**

# In[ ]:


acc_values = fit.history[metrics[3]]
val_acc_values = fit.history[metrics[1]]

plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()


# **Print final validation metrics**

# In[ ]:


print("Final validation loss function is", val_loss_values[-1])
print("Final validation accuracy is", val_acc_values[-1])


# In[ ]:


#score = model.evaluate(X_test, Y_test, verbose=0)


# # Run the model on the test dataset and make predictions
# 
# Note that we want to predict the class but we don't need to know the probability to know which class the test image belongs to. For this classification purpose we should use **predict_classes** method intead of **predict** method which is used for regression tasks. 

# In[ ]:


predictions = model.predict_classes(test_data, batch_size=64, verbose=0)


# # Write output data into the ready-for-submission CSV file
# 
# Your submission file should be in the following format: For each of the $28000$ images in the **test set**, output a single line containing **the ImageId** and **the digit** you predict. In addition, we include the header.

# In[ ]:


submissions = pd.DataFrame({'ImageId':list(range(1,len(predictions) + 1)), "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)

