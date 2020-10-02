#!/usr/bin/env python
# coding: utf-8

# **Creating neural network from scratch using python.**
# 
# using sigmoid function as an activation function in every layer.

# In[ ]:


import numpy as np #importing numpy for handling array's
import matplotlib.pyplot as plt #for plot


# Using Sigmoid Function for squashing the output values between 0 and 1.
# ![Sigmoid function and its derivative](http://miro.medium.com/max/4384/1*6A3A_rt4YmumHusvTvVTxw.png)

# It is a four layer network i.e one input layer and two hidden layers and one final output layer.
# ![image.png](attachment:image.png)
# 

# In[ ]:


X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float) #input value's
y=np.array(([0],[1],[1],[0]), dtype=float) #actual output value's

def sigmoid(t):
    return 1/(1+np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork: #Class of neural network
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],4)
        #print(self.weights1)
        self.weights2 = np.random.rand(4,3)
        #print(self.weights2)
        self.weights3=np.random.rand(3,1)
        #print(self.weights3)
        self.y = y
        self.output = np. zeros(y.shape)
        
    def feed(self): #feedforward into next layers
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2,self.weights3))
      
        return self.output
        
    def backpropagation(self): #backpropagating to previous layers
        d_weights3 = np.dot(self.layer2.T,2*(self.y-self.output)*sigmoid_derivative(self.output))
        d_weights2 = np.dot(self.layer1.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights3.T)*sigmoid_derivative(self.layer2))
        d_weights1 = np.dot(self.input.T, np.dot(np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights3.T)*sigmoid_derivative(self.layer2),self.weights2.T)*sigmoid_derivative(self.layer1))
        #updating weights
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3

    def train(self, X, y):
        self.output = self.feed()
        self.backpropagation()
    
    def save(self):
        weights=[self.weights1,self.weights2,self.weights3]
        np.save('weights.npy',weights)
        return "Saved"
    
    def load(self):
        self.wih = np.load('weights.npy',allow_pickle=True)
        #for i in self.wih:print(i)
        pass
      
    def testing(self,arr): #for testing nn on new data
        self.arr=arr
        self.layer1 = sigmoid(np.dot(self.arr, self.wih[0]))
        self.layer2 = sigmoid(np.dot(self.layer1, self.wih[1]))
        self.layer3 = sigmoid(np.dot(self.layer2,self.wih[2]))
        return self.layer3   

NN = NeuralNetwork(X,y)
losses=[]
for i in range(1,1501):
    if i%100==0:
        print ("for iteration # " + str(i) + "\n")
        print("Predicted Value:"+"\n"+str(NN.feed())+"\n")
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feed()))))
        print ("\n")
    losses.append(np.mean(np.square(y - NN.feed())))
    NN.train(X, y)

NN.save() #saving weights of the neural network


# In[ ]:


NN.load()
NN.testing([0,0,0])


# In[ ]:


y_points=losses
x_points=[i for i in range(1,1501)]
plt.plot(x_points,y_points) 
plt.xlabel("No. of Iterations")
plt.ylabel("Loss")
plt.title("Loss per Iteration")
plt.show()

