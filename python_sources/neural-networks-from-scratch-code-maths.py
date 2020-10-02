#!/usr/bin/env python
# coding: utf-8

# Historically the idea of a Artificial Neural Networks (ANN) has been there since the 40s (pretty old eh), it was inspired by the neurons in the human brain, even though the first functional perceptron with many layers was first published only in 60s, but the research on NN were not frequent because it required a lot of computational power and in the meanwhile the machine learning researches were increasing a lot.
# <img src="http://krisbolton.com/assets/images/posts/2018/neuron-annotated.jpg" alt="Human NN vs ANN" height="200" width="400"/>
# 
# This was until the backpropagation algorithm was discovered in 1975 by Paur Werbos, which actually revolutionalized the ANN world and made the Multi Layer Perceptron (MLP) practical, this algorithm consisted in changing the weights of the nodes based on the output errors. 
# <img src="https://i.pinimg.com/originals/e5/6e/8a/e56e8a055bfbcbeafaf413a70c911876.jpg" alt="First Meme" height="200" width="400"/>

# And then came the parallel computation in the industry, which will make the MLP even more practical. And finally in the 90s thanks to the GPUs they started becoming what we use today, a perfect black box algorithm that works almost everywhere, from the numerical values to Signals, form Text Mining to Images, from supervised to unsupervised and improved even the reinforcement learning. <br>
# Of course they do not always work perfectly as expected, and truth to be told we are still far away from perfection but the improvement is happening very rapidly.
# <img src="https://i.redd.it/qx6rsm2tqb741.jpg" alt="Second Meme" height="200" width="400"/>
# 
# Enough with the history, the main question is why do they work so well, I mean why are they usually as good as the best ML algorithm for that problem, if not better? 
# Well the simple answer is MATHS!!
# <img src="https://media.makeameme.org/created/math-math-everywhere-599f77.jpg" alt="Third Meme" height="200" width="400"/>
# We will see how maths gives us a certain guarantee on the performances of the ANN. Let us start by considering a simple architecture with one hidden layer.
# 
# We will not dive deep into the maths but try to give a general idea, since it requires quite advanced mathematical concepts such as Lesbegue Integrals (thus Measure Theory), some Topological concepts and some basic Calculus (normed spaces, $\varepsilon$-Balls) and Fourier Analysis, of course for all of those who would love to look into the complete maths here is the [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.441.7873&rep=rep1&type=pdf).

# The Paper I just mentioned explains, the actual reason of their perfomance, let $x \in \mathbb{R}^n$ a single layer perceptron can be seen as:
# \begin{align}
# y = \sum_{i=1}^N \alpha_i \sigma(y_i^Tx + \theta_i) \label{eq1}\tag{eq. 1}
# \end{align}
# 
# where $\sigma$ is a sigmoidical function, which mean that 
# $$\sigma(t) = \begin{cases} 1 & \text{for} & t \to +\infty  \\ 0 & \text{for} & t \to -\infty \end{cases}$$
# it basically represents the activation function you use today in the Neural Networks.
# And then the paper proofs basically how a function of the form (\ref{eq1}) can approximate any given continuous function (or atleast with a finite integral value of the absolute function) in a compact subspace of $\mathbb{R}^n$.
# 
# This explains why the MLP works well, they can approximate mostly any given function, the problem is using only one layer the search space is very big, so you need to find the perfect parameters to fit the function, while using more layers, the (\ref{eq1}) becomes nested since x is recieved not from the input but from a previous layer, this way it can approximate even more complex function very easily and rapidly, of course there is a trade off, adding more layer you increase the number of parameters, which in result might overfit the data, so the number of layer becomes a hyper parameter to be optimized.
# 
# There are a few hypthosis that the MLP makes, first of all, it assumes that the function which connects output to the input actually exists, so if you see that the ANN are not working well, it might be because the input are not actually explaining the output, and one more thing since we are trying to approximate a global function the training data must be as representative as possible of the whole population.

# Finally before implementing the NN with a single layer from scatch, we will explain a little bit of the Back Propagation. Let the current layer be L then we have that:
# \begin{align}
# z_L &= W_L \cdot a_{L-1} + \theta_L \label{eq2}\tag{eq. 2}\\
# a_L &= \sigma(z_L) \qquad \text{with } \sigma(x) = \dfrac{1}{1+e^{-x}} 
# \end{align}
# where $W_L$ are the weights for the L-th layer, the more layers there are the more nested the equations become. So the ANN can be seen as matrix multiplications (with bias addition) where the matrix is the parameter that we need to estimate (toghether with the bias).
# 
# Let's take the MSE as the loss function and the sigmoid function as the activation (it's derivative is pretty easy :D), given the true value $y$ and the prediction $\hat{y}$ (output of the final layer, so if we have L layers $\hat y = a_L$) we have:
# 
# \begin{align}
# L(y,\hat{y}) &= (\hat y - y)^2
# \end{align}
# 
# We need to minimize the Loss Function, so let's calculate the derivative of the Loss function:
# \begin{align}
# \dfrac{\partial L(y,\hat y)}{\partial W_L} &= \text{Using the chain Rule} \\
#                                      &= \dfrac{\partial L(y,\hat y)}{\partial \hat y} \dfrac{\partial \hat y}{\partial z_L} \dfrac{\partial z_L}{\partial W_L} 
# \end{align}
# Now let's compute each one of them:
# \begin{align}
# \dfrac{\partial L(y,\hat y)}{\partial \hat y} &= \dfrac{\partial (\hat y - y)^2}{\partial \hat y} = 2(\hat y - y) \\
# \dfrac{\partial \hat y}{\partial z_L} &= \sigma'(z_L) = \sigma(z_L)(1-\sigma(z_L)) \text{ if $\sigma$ is the sigmoid function}\\
# \dfrac{\partial z_L}{\partial W_L} &= \dfrac{\partial (W_L a_{L-1} + b_L)}{\partial W_L} = a_{L-1} 
# \end{align}
# While for the precedent layer it becomes nesting the derivative even more:
# 
# \begin{align}
# \dfrac{\partial L(y,\hat y)}{\partial W_{L-1}} &= \dfrac{\partial L(y,\hat y)}{\partial \hat y} \dfrac{\partial \hat y}{\partial z_L} \dfrac{\partial z_L}{\partial a_{L-1}} \dfrac{\partial a_{L-1}}{\partial z_{L-1}} \dfrac{\partial z_{L-1}}{\partial W_{L-1}} \\
#         &= 2(\hat y - y)\sigma(z_L)(1-\sigma(z_L))W_L\sigma(z_{L-1})(1-\sigma(z_{L-1}))a_{L-2}
# \end{align}

# Here is a simple implementation of a MLP, with variable number of neurons and layers for basic numerical tasks. It helps a lot to understand how actually a MLP works, instead of using the existing libraries as black box models, well of course don't expect this simple implementation to work on a decent dataset, it is nowhere optimized nor completely customizable, as you will see it has been simplified quite a lot to make it more understandable, for example it has only one activation function which is activating all the layers (to simplify and not do all the derivatives for all the activation functions), then it can be only used for binary classification because it has only one output neuron with the sigmoid activation function and finally it doesn't work in batch but with one element at a time, otherwise the code would get a lot more complex to manage all those vectors and finally uses only the MSE loss (it has an easier derivative :D)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BasicNeuralNetwork:
    def __init__(self, input_shape, hidden_neurons):
        """
        Initialization of a simple MLP with only a single output only with 
        the sigmoid activation function (so only for binary classification)
        
        input_shape: int, represents the input layer size
        hidden_neurons: array-like giving the number of neurons 
                        for each hidden layer.
        
        Example for a 2 layer perceptron with 2,4 neurons respectively and 
        with 3 input neurons:
        model = BasicNeuralNetwork(3, (2,4))
        """
        self.input_shape = input_shape
        self.n_layers = len(hidden_neurons)
        self.hidden   = hidden_neurons
            
        self.init_layers()
        
    def init_layers(self):
        "Initializes and creates the layers with random weights"
        self.weights  = dict()
        self.layers = list()
        
        for i in range(1,self.n_layers+1):
            weight = f"W_{i}"
            self.layers.append(weight)
            np.random.seed(i*41)
            if i==1:
                self.weights[weight] = np.random.rand(self.input_shape, 
                                                      self.hidden[0])
            else:
                self.weights[weight] = np.random.rand(self.hidden[i-2], 
                                                      self.hidden[i-1])
        
        out_layer = f"W_{self.n_layers+1}" #We call the output layer with W_{L+1}
        self.weights[out_layer] = np.random.rand(self.hidden[self.n_layers-1], 
                                                              1)
        self.layers.append(out_layer)

    def forward(self, X):
        """
        Given the input X, it basically is the implementation
        of the formula:
           y_hat = \sigma(W_L*\sigma(W_{L-1}*\sigma(W_{L-2}*\sigma(...))))
        and it stores the output value both with and without the
        activation funtion (it simplifies the backpropagation function)
        """
        self.forward_z = dict()
        self.forward_a = dict()
        self.forward_z["W_0"] = X
        self.forward_a["W_0"] = self.sigmoid(self.forward_z["W_0"])
        for i, l in enumerate(self.layers):
            prev_layer = f"W_{i}"
            self.forward_z[l] = np.dot(self.forward_a[prev_layer], 
                                       self.weights[l])
            self.forward_a[l] = self.sigmoid(self.forward_z[l])
                
        out = self.forward_a[f"W_{self.n_layers+1}"]
                
        return out

    def backprop(self, y, y_hat, lr):
        """
        The core of the MLP, without it nothing would work, this simply
        backpropagates the error, it requires the true y and the prediction
        and the learning rate and with the Gradient Descendent adjusts the weights,
        It is basically the implementation of the theory described before.
        """
        dpartial = dict()
        dweights = dict()
        for i in range(self.n_layers+1,0,-1):
            prev_l = f"W_{i-1}"
            current_l = f"W_{i}"
            next_l = f"W_{i+1}"
            if i == self.n_layers+1:
                dpartial[current_l] = 2*(y_hat - y)*self.sigmoid_derivative(self.forward_z[current_l])
                dweights[current_l] = np.dot(self.forward_a[prev_l].T, dpartial[current_l])
            else:
                dpartial[current_l] = np.dot(dpartial[next_l],
                                             self.weights[next_l].T)*\
                                                self.sigmoid_derivative(self.forward_z[current_l])
                dweights[current_l] = np.dot(self.forward_a[prev_l].T, dpartial[current_l])
                 
        #Gradient descendent updates all the layers
        for l in self.layers:
            self.weights[l] = self.weights[l] - lr * dweights[l]
            
    @staticmethod    
    def sigmoid(x):
        "Sigmoid activation function"
        return 1/(1+np.exp(-x))
    @staticmethod
    def sigmoid_derivative(x):
        "Derivative of the sigmoid"
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        return sigmoid(x) * (1-sigmoid(x))


# Let's take a very simple dataset and try to fit it and see if it actually works (otherwise it could be considered a big time waste)!

# In[ ]:


X1 = [0,0,1,1,1]
X2 = [1,0,0,1,1]
X3 = [0,1,0,0,1]
y  = [1,0,0,1,1] 

df = pd.DataFrame({"X1":X1,"X2":X2,
                   "X3":X3,"y":y})

df.head()


# In[ ]:


X = df.drop("y", axis=1).values
y = df["y"].values.reshape(-1,1)


# In[ ]:


model = BasicNeuralNetwork(input_shape=3,
                           hidden_neurons=(4,4))


# In[ ]:


# The training is done manually instead of creating another method
# so it's easier to understand what is happening
NUM_EPOCH = 3000
LR = 1
avg_loss = []
for epoch in range(NUM_EPOCH):
    loss = []
    for i in range(X.shape[0]):
        y_hat = model.forward(X[i].reshape(1,3))
        model.backprop(y[i].reshape(1,1), y_hat, LR)
        loss.append(((y[i].reshape(1,1) - y_hat)**2)[0])
    avg_loss.append(np.mean(loss))


# In[ ]:


plt.plot(avg_loss)
plt.title("Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()


# Let's compare it to the Tensforflow equivalent.

# In[ ]:


import tensorflow as tf
from tensorflow.keras import models, layers, optimizers


# In[ ]:


tf_model = models.Sequential()
tf_model.add(layers.Input(shape=(3,)))
tf_model.add(layers.Activation("sigmoid"))
tf_model.add(layers.Dense(4, activation="sigmoid"))
tf_model.add(layers.Dense(4, activation="sigmoid"))
tf_model.add(layers.Dense(1, activation="sigmoid"))

tf_model.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(1), 
                 loss="mse")

hist = tf_model.fit(X,y, epochs=3000, verbose=0)


# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(avg_loss)
plt.title("Our Model Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")

plt.subplot(122)
plt.plot(hist.history["loss"])
plt.title("Tensorflow Loss Plot")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()


# In[ ]:


print("Prediction Comparision:")
print("Ours \t TFs \t Actual")
for x in range(X.shape[0]):
    y_h  = round(model.forward(X[x].reshape(1,3)).reshape(1,)[0],3)
    y_tf = round(float(tf_model.predict(X[x].reshape(1,3)).reshape(1,)[0]),3)
    print(f"{y_tf} \t {y_tf} \t {y[x][0]}")


# <img src="https://memegenerator.net/img/instances/58852927/yes-it-works.jpg" alt="Last Meme" width="300"/>

# <center><h2>
# The creation of this notebook required a lot of time and dedication, please UPVOTE if you find it useful.
# </h2></center>
# And of course if there is something you find confusing or wrong don't hestitate to comment :)
# 
# For future if people like this notebook, I would love to explain also Gradient Descendent and other non linear optimization techniques with code, may be not on the NN backpropagation but with actual functions so it's easier to animate etc.
