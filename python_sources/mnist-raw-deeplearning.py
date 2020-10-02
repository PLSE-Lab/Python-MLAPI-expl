#!/usr/bin/env python
# coding: utf-8

# ## Breaking the Magician's Code!
# ### MNIST Digit Recognizer using raw Deep Learning (No ML libs used).
# This is a good place to understand the ground-up setup of ANNs using BackProp (Gradient Descent), and how things work under-the-hood to train your network. <br />
# If you have any suggestions or questions, please feel free to leave in comments. 

# We will start by importing the required packages and the data.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[ ]:



x_train = pd.read_csv('../input/train.csv', dtype='uint8', header=0)
#x_test = pd.read_csv('../input/test.csv', dtype='uint8', header=0)  # Not used


# **Memory Optimization: **Using uint8 takes 4 times less memory as we know pixel vales are b/w [0,255]. <br />
# PS: Although the dataframe will be converted back to float64, during scaling later however, this is still a good technique to use if you need to optimize memory consumption while loading data, among other approaches.

# Here's a sneak peek at the data.

# In[ ]:


x_train.shape


# In[ ]:


x_train.head()


# There are 784 (flattened 28x28) pixel values (0-255) for each of the 42000 training digits.
# 
# Let's extract the label column.

# In[ ]:


y_train = x_train.label
x_train.drop('label', axis=1, inplace=True)


# In[ ]:


y_train.head()


# **Scaling** the pixel values to [0, 1] in the training data makes it easier for our activation functions to converge (using Gradient Descent) faster, by bringing them within the output range.

# In[ ]:


# Scale values to be friendlier with our Sigmoid activation
scaler = preprocessing.MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train))


# In[ ]:


x_train.describe()


# **One-hot encoding of labels: ** It is important to binarize the labels as we will use a 10 unit ouput layer to predict the digits from 0 to 9.

# In[ ]:


lb = preprocessing.LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)


# In[ ]:


# let's check our vector shapes
x_train.shape, y_train.shape


# # The ANN
# 
# 'forward_pass()' performs the forward pass with Backpropagation using Gradient Descent. <br />
# Brackpropagation is a loss optimization (error minimization) technique that uses Gradient Descent (fine steps in the negative direction of the gradient/slope of the error function), to train/tune the best weights (coefficents) for the function variables (features). 
# 
# I know, pithy statement with quite a few parentheses. But fret not, here is a good [resource](https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e) to understand or brush up on the concept if the water looks murky!

# ### Setting up the Neural Network
# 
# Here, I will be using a two-layer NN with Sigmoid activation for the hidden layer units, and Softmax for output. <br /> 
# Here is another [resource](http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/) to understand about these choices.
# 
# Simply put, Softmax generates probabilistic output (much like Scikit-learn's predict.proba()) for our multi-class labels.

# In[ ]:


# Hidden layer activation function (good as our inputs are b/w 0 and 1)
def sigmoid(x):
    """
    Sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-x))

# Output layer activation function
def softmax(z):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

# Using log loss
def loss_fn(y_true, y_pred, eps=1e-16):
    """
    Loss function we would like to optimize (minimize)
    We are using Logarithmic Loss (Categorical Cross Entropy)
    http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
    """
    y_pred = np.maximum(y_pred,eps)
    y_pred = np.minimum(y_pred,(1-eps))  # Preventing inf
    return -(np.sum(y_true * np.log(y_pred)) + np.sum((1-y_true)*np.log(1-y_pred)))/len(y_true)

# Setup processor for two layer NN
def forward_pass(W_1, W_2, x, y):
    """
    Does a forward computation of the neural network
    Also produces the gradient of the log loss function
    """
    # First, compute the new predictions `y_pred`
    z_2 = np.dot(x, W_1)  # ILF1  (Induced Local Field to layer 2)
    a_2 = sigmoid(z_2)
    z_3 = np.dot(a_2, W_2)  # ILF2
    y_pred = softmax(z_3)
    
    # Now compute the gradient: BackProp
    J_z_3_grad = -y + y_pred
    J_W_2_grad = np.dot(a_2.T, J_z_3_grad)
    a_2_z_2_grad = sigmoid(z_2)*(1-sigmoid(z_2))
    J_W_1_grad = (np.dot(J_z_3_grad, W_2.T)*a_2_z_2_grad).T.dot(x).T
    gradient = (J_W_1_grad, J_W_2_grad)
    
    # return
    return y_pred, gradient


def plot_loss_accuracy(loss_vals, accuracies, header):
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Log Loss and Accuracy over {} iterations'.format(header))
    
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss_vals)
    ax.grid(True)
    ax.set(xlabel='iterations', title='Log Loss')
    
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracies)
    ax.grid(True)
    ax.set(xlabel='iterations', title='Accuracy');


# ## Preparing ANN for training

# In[ ]:


def train_nn(W_1, W_2, eta, num_iter, x, y, x_test, y_test, eps=0.1):
    loss_vals, accuracies = [], []
    loss_vals_test, accuracies_test = [], []
    y_pred = None
    l_prev = 100  # To control eta
    for i in range(num_iter):
        ### Do a forward computation, and get the gradient
        y_pred, (g_w_1, g_w_2) = forward_pass(W_1, W_2, x, y)
                
        ## Update the weight matrices using Gradient Descent
        W_1 = W_1 - eta*g_w_1
        W_2 = W_2 - eta*g_w_2

        ### Compute the loss and accuracy
        l = loss_fn(y, y_pred)
#         print(y,y_pred,l)
        
        # Adjust eta to help convergence
        if l >= l_prev:
            eta = eta * 0.9
#             print("eta updated to: ", eta)
        l_prev = l
        match = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        acc = match/len(y)
        
        loss_vals.append(l)
        accuracies.append(acc)
        
        test_acc, test_l = test_nn(x_test, y_test, W_1, W_2)
        loss_vals_test.append(test_l)
        accuracies_test.append(test_acc)

        ## Print the loss and accuracy for every 200th iteration
        if i%200 == 0:
            print(f'Epoch = {i}')
            print("Loss={}, Acc={}".format(l, acc))
            print("Validation Loss={}, Validation Acc={}".format(test_l, test_acc))
            
        if l <= eps:  # Epsilon threshold
            print("Epoch: {}. Breaking as desired acc ({}) achieved..".format(i, acc))
            break
        
    plot_loss_accuracy(loss_vals, accuracies, "train")
    plot_loss_accuracy(loss_vals_test, accuracies_test, "test")
    return y_pred, W_1, W_2

def test_nn(x, y, W_1, W_2):
    z_2 = np.dot(x, W_1)  # ILF1  (Induced Local Field to layer 2)
    a_2 = sigmoid(z_2)
    z_3 = np.dot(a_2, W_2)
    y_pred = softmax(z_3)
    
    # Calculate loss
    l = loss_fn(y, y_pred)
    # Calculate acc
    match = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
    acc = match/len(y)
    return (acc, l)

def predict(x, W_1, W_2):
    z_2 = np.dot(x, W_1)  # ILF1  (Induced Local Field to layer 2)
    a_2 = sigmoid(z_2)
    z_3 = np.dot(a_2, W_2)  # ILF2
    y_pred = softmax(z_3)
    
    return np.argmax(y_pred, axis=1)


# ## Finally, set up the ANN parameters and initialize the training

# In[ ]:


#### Initialize the network parameters

# Split into train/validation
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.10)

np.random.seed(1241)
input_units = X_train.shape[1]
h1_units = 128       # Hidden layer 1 units
out_units = y_train.shape[1]
n = len(X_train)           # No. of training samples to use in training

W_1 = np.random.uniform(-1,1,size = (input_units,h1_units))  # Include bias
W_2 = np.random.uniform(-1,1,size = (h1_units,out_units))
num_iter = 1000  # Epochs
eta = .0001

y_pred, W_1, W_2 = train_nn(W_1, W_2, eta, num_iter, X_train[:n], Y_train[:n], X_test[:n], Y_test[:n])


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = predict(X_test, W_1, W_2)
accuracy_score(np.argmax(Y_test, axis=1), y_pred)


# Note that I have reduced the number of neurons in the hidden layer to 128. Using 256 neurons yielded 96% accuracy but with longer training time. 

# Hope this simplified outline helped you get the gist of how things really work under-the-hood in a Deep Neural Network. If it did, please don't forget to vote.

# In[ ]:




