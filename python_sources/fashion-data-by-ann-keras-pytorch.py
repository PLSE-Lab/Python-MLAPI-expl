#!/usr/bin/env python
# coding: utf-8

# ANN is tried with Fashion Dataset of MNIST
# 
# 2019 - November

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # visualization tool
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv',dtype = np.float32)
data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv',dtype = np.float32)


# In[ ]:


X_train = data_train.iloc[:,1:785].values
X_test = data_test.iloc[:,1:785].values


# In[ ]:


print('X_train', X_train.shape)
print('X_test', X_test.shape)


# In[ ]:


y_train = data_train.iloc[:,0].values
y_test = data_test.iloc[:,0].values


# In[ ]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[ ]:


print('y_train', y_train.shape)
print('y_test', y_test.shape)


# In[ ]:


x_train = X_train.T
x_test = X_test.T
Y_train = y_train.T
Y_test = y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",Y_train.shape)
print("y test: ",Y_test.shape)


# In[ ]:


#normalization 
x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))


# <a id="10"></a> <br>
# # Artificial Neural Network (ANN)
# * It is also called deep neural network or deep learning.
# * **What is neural network:** It is basically taking logistic regression and repeating it at least 2 times.
# * In logistic regression, there are input and output layers. However, in neural network, there is at least one hidden layer between input and output layer.
# * **What is deep, in order to say "deep" how many layer do I need to have:** When I ask this question to my teacher, he said that ""Deep" is a relative term; it of course refers to the "depth" of a network, meaning how many hidden layers it has. "How deep is your swimming pool?" could be 12 feet or it might be two feet; nevertheless, it still has a depth--it has the quality of "deepness". 32 years ago, I used two or three hidden layers. That was the limit for the specialized hardware of the day. Just a few years ago, 20 layers was considered pretty deep. In October, Andrew Ng mentioned 152 layers was (one of?) the biggest commercial networks he knew of. Last week, I talked to someone at a big, famous company who said he was using "thousands". So I prefer to just stick with "How deep?""
# * **Why it is called hidden:** Because hidden layer does not see inputs(training set)
# * For example you have input, one hidden and output layers. When someone ask you "hey my friend how many layers do your neural network have?" The answer is "I have 2 layer neural network". Because while computing layer number input layer is ignored. 
# * Lets see 2 layer neural network: 
# <a href="http://ibb.co/eF315x"><img src="http://preview.ibb.co/dajVyH/9.jpg" alt="9" border="0"></a>
# * Step by step we will learn this image.
#     * As you can see there is one hidden layer between input and output layers. And this hidden layer has 3 nodes. If yoube curious why I choose number of node 3, the answer is there is no reason, I only choose :). Number of node is hyperparameter like learning rate. Therefore we will see hyperparameters at the end of artificial neural network.
#     * Input and output layers do not change. They are same like logistic regression.
#     * In image, there is a tanh function that is unknown for you. It is a activation function like sigmoid function. Tanh activation function is better than sigmoid for hidden units bacause mean of its output is closer to zero so it centers the data better for the next layer. Also tanh activation function increase non linearity that cause our model learning better.
#     * As you can see with purple color there are two parts. Both parts are like logistic regression. The only difference is activation function, inputs and outputs.
#         * In logistic regression: input => output
#         * In 2 layer neural network: input => hidden layer => output. You can think that hidden layer is output of part 1 and input of part 2.
# * Thats all. We will follow the same path like logistic regression for 2 layer neural network.

# <a id="11"></a> <br>
# ## 2-Layer Neural Network
# * Size of layers and initializing parameters weights and bias
# * Forward propagation
# * Loss function and Cost function
# * Backward propagation
# * Update Parameters
# * Prediction with learnt parameters weight and bias
# * Create Model

# <a id="12"></a> <br>
# ## Size of layers and initializing parameters weights and bias
# * For x_train that has 348 sample $x^{(348)}$:
# $$z^{[1] (348)} =  W^{[1]} x^{(348)} + b^{[1] (348)}$$ 
# $$a^{[1] (348)} = \tanh(z^{[1] (348)})$$
# $$z^{[2] (348)} = W^{[2]} a^{[1] (348)} + b^{[2] (348)}$$
# $$\hat{y}^{(348)} = a^{[2] (348)} = \sigma(z^{ [2] (348)})$$
# 
# * At logistic regression, we initialize weights 0.01 and bias 0. At this time, we initialize weights randomly. Because if we initialize parameters zero each neuron in the first hidden layer will perform the same comptation. Therefore, even after multiple iterartion of gradiet descent each neuron in the layer will be computing same things as other neurons. Therefore we initialize randomly. Also initial weights will be small. If they are very large initially, this will cause the inputs of the tanh to be very large, thus causing gradients to be close to zero. The optimization algorithm will be slow.
# * Bias can be zero initially.

# In[ ]:


# intialize parameters and layer sizes
def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3,1)),
                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0],1))}
    return parameters


# In[ ]:


print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",Y_train.shape)
print("y test: ",Y_test.shape)


# <a id="13"></a> <br>
# ## Forward propagation
# * Forward propagation is almost same with logistic regression.
# * The only difference is we use tanh function and we make all process twice.
# * Also numpy has tanh function. So we do not need to implement it.

# In[ ]:


# calculation of z
#z = np.dot(w.T,x_train)+b
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


# In[ ]:



def forward_propagation_NN(x_train, parameters):

    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# <a id="14"></a> <br>
# ## Loss function and Cost function
# * Loss and cost functions are same with logistic regression
# * Cross entropy function
# <a href="https://imgbb.com/"><img src="https://image.ibb.co/nyR9LU/as.jpg" alt="as" border="0"></a><br />

# In[ ]:


# Compute cost
def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost


# <a id="15"></a> <br>
# ## Backward propagation
# * As you know backward propagation means derivative.
# * If you want to learn (as I said I cannot explain without talking bc it is little confusing), please watch video in youtube.
# * However the logic is same, lets write code.

# In[ ]:


# Backward Propagation
def backward_propagation_NN(parameters, cache, X, Y):

    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads


# <a id="16"></a> <br>
# ## Update Parameters 
# * Updating parameters also same with logistic regression.
# * We actually do alot of work with logistic regression

# In[ ]:


# update parameters
def update_parameters_NN(parameters, grads, learning_rate = 0.01):
    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}
    
    return parameters


# <a id="17"></a> <br>
# ## Prediction with learnt parameters weight and bias
# * Lets write predict method that is like logistic regression.

# In[ ]:


# prediction
def predict_NN(parameters,x_test):
    # x_test is a input for forward propagation
    A2, cache = forward_propagation_NN(x_test,parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


# <a id="18"></a> <br>
# ## Create Model
# * Lets put them all together

# In[ ]:


# 2 - Layer neural network
def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):
    cost_list = []
    index_list = []
    #initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        A2, cache = forward_propagation_NN(x_train,parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
         # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
         # update parameters
        parameters = update_parameters_NN(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    # predict
    y_prediction_test = predict_NN(parameters,x_test)
    y_prediction_train = predict_NN(parameters,x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(x_train, Y_train,x_test,Y_test, num_iterations=2500)


# # Evaluating the (L-Layer) ANN with KERAS Library

# <a id="19"></a> <br>
# # L Layer Neural Network
# * **What happens if number of hidden layer increase:** Earlier layerls can detect simple features.
# * When model composing simple features together in later layers of neural network that it can learn more and more complex functions. For example, lets look at our sign one.
# <a href="http://ibb.co/dNgDJH"><img src="http://preview.ibb.co/mpD4Qx/10.jpg" alt="10" border="0"></a>
# * For example first hidden layer learns edges or basic shapes like line. When number of layer increase, layers start to learn more complex things like convex shapes or characteristic features like forefinger.
# * Lets create our model
#     * There are some hyperparameters we need to choose like learning rate, number of iterations, number of hidden layer, number of hidden units, type of activation functions. Woww it is too much :)
#     * These hyperparameters can be chosen intiutively if you spend a lot of time in deep learning world.
#     * However, if you do not spend too much time, the best way is to google it but it is not necessary. You need to try hyperparameters to find best one.
#     * In this tutorial our model will have 2 hidden layer with 8 and4 nodes, respectively. Because when number of hidden layer and node increase, it takes too much time. 
#     * As a activation function we will use relu(first hidden layer), relu(second hidden layer) and sigmoid(output layer) respectively.
#     * Number of iteration will be 100.
# * Our way is same with previous parts however as you learn the logic behind deep learning, we can ease our job and use keras library for deeper neural networks.
# * First lets reshape our x_train, x_test, y_train and y_test.
# 

# In[ ]:


# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, Y_train.T, Y_test.T


# In[ ]:


print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# In[ ]:



print("input dimension: ",x_train.shape[1])


# <a id="22"></a> <br>
# ## Implementing with keras library
# Lets look at some parameters of keras library:
# * units: output dimensions of node
# * kernel_initializer: to initialize weights
# * activation: activation function, we use relu
# * input_dim: input dimension that is number of pixels in our images (785 px)
# * optimizer: we use adam optimizer
#     * Adam is one of the most effective optimization algorithms for training neural networks.
#     * Some advantages of Adam is that relatively low memory requirements and usually works well even with little tuning of hyperparameters
# * loss: Cost function is same. By the way the name of the cost function is cross-entropy cost function that we use previous parts.
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$
# * metrics: it is accuracy.
# * cross_val_score: use cross validation. If you do not know cross validation please chech it from my machine learning tutorial. https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners
# * epochs: number of iteration

# In[ ]:


# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 1)  # burda egitim yapiliyor cross validation cv=3
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))


# In[ ]:


history = classifier.fit(x =x_train, y = y_train, validation_split=0.25, epochs=50, batch_size=16, verbose=1)


# In[ ]:


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# ## Implementing with Pytorch Library

# Steps of ANN:
# 1. Import Libraries
#     * In order to show you, I import again but we actually imported them at previous parts.
# 2. Prepare Dataset
#     * Totally same with previous part(logistic regression).
#     * We use same dataset so we only need train_loader and test_loader.
#     * We use same batch size, epoch and iteration numbers.
# 3. Create ANN Model
#     * We add 3 hidden layers.
#     * We use ReLU, Tanh and ELU activation functions for diversity.
# 4. Instantiate Model Class
#     * input_dim = 2828 # size of image pxpx
#     * output_dim = 10 # labels 0,1,2,3,4,5,6,7,8,9
#     * Hidden layer dimension is 150. I only choose it as 150 there is no reason. Actually hidden layer dimension is hyperparameter and it should be chosen and tuned. You can try different values for hidden layer dimension and observe the results.
#     * create model
# 5. Instantiate Loss Class
#     * Cross entropy loss
#     * It also has softmax(logistic function) in it.
# 6. Instantiate Optimizer Class
#     * SGD Optimizer
# 7. Traning the Model
# 8. Prediction
#     * As a result, as you can see from plot, while loss decreasing, accuracy is increasing and our model is learning(training).
#     * Thanks to hidden layers model learnt better and accuracy(almost 95%) is better than accuracy of logistic regression model.

# In[ ]:


# Import Libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:


# Prepare Dataset
# load data

# data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv',dtype = np.float32)  # yukarda yapildi


# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = data_train.label.values
features_numpy = data_train.loc[:,data_train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42) 

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

# visualize one of the images in data set
plt.imshow(features_numpy[11].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[11]))
plt.savefig('graph.png')
plt.show()


# In[ ]:


# Create ANN Model
class ANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()
        
        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.tanh2 = nn.Tanh()
        
        # Linear function 3: 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.elu3 = nn.ELU()
        
        # Linear function 4 (readout): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.tanh2(out)
        
        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.elu3(out)
        
        # Linear function 4 (readout)
        out = self.fc4(out)
        return out

# instantiate ANN
input_dim = 28*28
hidden_dim = 150 #hidden layer dim is one of the hyper parameter and it should be chosen and tuned. For now I only say 150 there is no reason.
output_dim = 10

# Create ANN
model = ANNModel(input_dim, hidden_dim, output_dim)

# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.01  # 0.001 ile %67 dogruluk verdi
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[ ]:


# ANN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        
        # Calculating gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Predict test dataset
            for images, labels in test_loader:

                test = Variable(images.view(-1, 28*28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)

                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))


# In[ ]:


# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("ANN: Loss vs Number of iteration")
plt.show()

# visualization accuracy 
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("ANN: Accuracy vs Number of iteration")
plt.show()

