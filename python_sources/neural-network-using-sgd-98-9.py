#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
np.random.seed(1070)


# In[ ]:


df = pd.read_csv("../input/train.csv")
data = df.as_matrix()
np.random.shuffle(data)
data_y = data[:,0].astype('float32')
data_x = data[:,1:].astype('float32')


# In[ ]:


data_y = pd.get_dummies(data_y).as_matrix()

meio = 255/2
data_x = (data_x-meio)/meio


# In[7]:


VALID_SIZE = round(data_x.shape[0]*0.15)


x_train = data_x[VALID_SIZE:]
x_valid = data_x[:VALID_SIZE]

d_train = data_y[VALID_SIZE:]
d_valid = data_y[:VALID_SIZE]

data_x = None
data_y = None

x_train.shape


# In[ ]:


def initializationWeights():
    ##Initialization of the Weights and the Biases with the random gaussian function with mean zeron, and variance between 1/sqtr(num_inputs_layer)
    
    ninputs = 784
    wl1 = 128
    wl2 = 64
    nclass = 10
    
    mean = 0
    
    #layer1
    variance = 1.0/np.sqrt(ninputs)
    w1 = np.random.normal(mean, variance, [ninputs,wl1])
    b1 = np.random.normal(mean, variance, [1,wl1])
    dw1 = np.zeros([ninputs,wl1])
    db1 = np.zeros([1,wl1])
    
    #Layer2
    variance = 1.0/np.sqrt(wl1)
    w2 = np.random.normal(mean, variance, [wl1,wl2])
    b2 = np.random.normal(mean, variance, [1,wl2])
    dw2 = np.zeros([wl1,wl2])
    db2 = np.zeros([1,wl2])

    #Layer3
    variance = 1.0/np.sqrt(wl2)
    w3 = np.random.normal(mean, variance, [wl2,nclass])
    b3 = np.random.normal(mean, variance, [1,nclass])
    dw3 = np.zeros([wl2,nclass])
    db3 = np.zeros([1,nclass])
    
    return w1,w2,w3,b1,b2,b3,dw1,dw2,dw3,db1,db2,db3


# In[ ]:


##Activation Function's and Cross-entropy Function

def ReLu(x, derivative=False):
    if(derivative==False):
        return x*(x > 0)
    else:
        return 1*(x > 0)
    
def LReLu(x, derivative=False):
    if(derivative==False):
        return x*(x > 0) + 0.1*x*(x<0)
    else:
        return 1*(x > 0) + 0.1*(x<0)

def sigmoid(x, derivative=False):
    if(derivative==False):
        return 1/(1+np.exp(-x))
    else:
        return x*(1-x)
       
def softmax(x):
        if(x.ndim==1 or x.ndim==0):
            e_x = np.exp(x - np.max(x))
            return e_x/e_x.sum(axis=0)
        else:
            k = 0
            x3 = np.empty((0,0))
            for x2 in x:
                if(k==0):
                    x3 = np.array([softmax(x2)])
                    k=1
                else:
                    x4 = softmax(x2)
                    x3 = np.concatenate((x3,[x4]), axis=0)
            return np.array(x3)

def cost(Y_predict, Y_right):
    Loss = -np.mean(Y_right*np.nan_to_num(np.log(Y_predict)),keepdims=True)
    return Loss


# In[10]:


def accuracy(output, y):
    hit = 0
    output = np.argmax(output, axis=1)
    y = np.argmax(y, axis=1)
    for y in zip(output, y):
        if(y[0]==y[1]):
            hit += 1

    p = (hit*100)/output.shape[0]
    return p


# In[ ]:


def run(x_train, y_train, x_valid, y_valid, epochs = 10, nbatchs=25, alpha = 1e-3, decay = 0, momentum = 0, l2 = 0.001, DROPOUT = 0):
  
    pross = x_train.shape[0]*0.05
    w1,w2,w3,b1,b2,b3,dw1,dw2,dw3,db1,db2,db3  = initializationWeights()
    index = np.arange(x_train.shape[0])
    
    print("Train data: %d" % (x_train.shape[0]))
    print("Validation data: %d \n" % (x_valid.shape[0]))
    
    for j in range(epochs):
        np.random.shuffle(index)
        t = 0
        iterations = round(x_train.shape[0]/nbatchs)
        prog = ""
        sacurr = 0
        sloss = 0
        sys.stdout.write("\nEpochs: %2d \ %2d \n"% (j+1,epochs))
        for i in range(iterations):
         
            f = i*nbatchs
            l = f+nbatchs
            
            if(l>(x_train.shape[0]-1)):
                l = x_train.shape[0]
                
            x = x_train[index[f:l]]
            y = y_train[index[f:l]]

            #1-Hidden Layer
       
            first = ReLu(x.dot(w1)+b1)
            if(DROPOUT!=1):
                first *= np.random.binomial([np.ones_like(first)],1-DROPOUT)[0]  /(1-DROPOUT)
            #2-Hidden Layer
            second = ReLu(first.dot(w2)+b2)
            if(DROPOUT!=1):
                second *= np.random.binomial([np.ones_like(second)],1-DROPOUT)[0] / (1-DROPOUT)
            #Output Layer
            output = softmax(second.dot(w3)+b3)
         
            loss = cost(output, y)
            
            error = y-output
            
            accuracy_t = accuracy(output, y)
            
            sacurr += accuracy_t
            sloss += loss
            
            accuracy_train = sacurr/(i+1)
            loss_train = sloss/(i+1)
            
            w3_delta = error
            
            w2_error = w3_delta.dot(w3.T)
            w2_delta = w2_error * ReLu(second,derivative=True)
            
            w1_error = w2_delta.dot(w2.T)
            w1_delta = w1_error * ReLu(first,derivative=True)
            
            mew3 = np.mean(w3)
            meb3 = np.mean(b3)
            mew2 = np.mean(w2)
            meb2 = np.mean(b2)
            mew1 = np.mean(w1)
            meb1 = np.mean(b1)
        
            w3 += alpha * (momentum*w3 + second.T.dot(w3_delta)) - l2 * mew3
            b3 += alpha * (momentum*b3 + second.T.dot(w3_delta).sum(axis=0)) - l2 * meb3
            w2 += alpha * (momentum*w2 + first.T.dot(w2_delta)) - l2 * mew2
            b2 += alpha * (momentum*b2 + first.T.dot(w2_delta).sum(axis=0)) - l2 * meb2
            w1 += alpha * (momentum*w1 + x.T.dot(w1_delta)) - l2 * mew1
            b1 += alpha * (momentum*b1 + x.T.dot(w1_delta).sum(axis=0)) - l2 * meb1
            
            
            
            t+= x.shape[0]
            
            qtd = round(t/pross)
            prog = "["
            for p in range(20):
                if(p<qtd-1):
                    prog += "-"
                elif(p==qtd-1):
                    prog += ">"
                else:
                    prog += " "
            prog += "]"

           
            sys.stdout.write("\r%5d : %5d %s Train Acc.: %.4f - Train Loss: %.4f\n" % (x_train.shape[0],t, prog, accuracy_train, loss_train))
        
        alpha = alpha - (alpha*decay)
        #1-Hidden Layer
        first = ReLu(x_valid.dot(w1)+b1)
        #2-Hidden Layer
        second = ReLu(first.dot(w2)+b2)
        #Output Layer
        output = softmax(second.dot(w3)+b3)
        
        loss_valid = cost(output, y_valid)
        accuracy_valid = accuracy(output, y_valid)
        
        sys.stdout.write("\r%5d : %5d %s  Train Acc: %.4f - Train Loss: %.4f - Valid Acc: %.4f - Valid Loss: %.4f\n" % ( x_train.shape[0],t, prog, accuracy_train, loss_train, accuracy_valid, loss_valid))


# In[ ]:


alpha = 1e-3
epochs = 10
run(x_train, d_train, x_valid, d_valid, epochs = epochs, nbatchs=25, alpha = alpha, decay = alpha/epochs, momentum = 1e-9, l2 = 0.01, DROPOUT = 0.25)


# In[ ]:




