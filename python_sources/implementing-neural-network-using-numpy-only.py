#!/usr/bin/env python
# coding: utf-8

# 1. In this notebook, we will implement a neural network from scratch using only NumPy
# 2. Dataset comprises of 10 features and binary target variable.
# 3. We will try to predict given house features whether it will be above or below market median price value.
# 4. We will implement:
# 
#     * Train- Test Split
#     * Standardization
#     * Binary Cross-entrophy loss
#     * Sigmoid
#     * ReLU
#     * He initialization
#     * Forward Propagation
#     * Backward Propagation
#     * Dropout
#     * L2 Regularization
#     * Learning Rate Deacay
#     * Adam optimizer with hyperparameters
#     * Precision, Recall, f1 score, Accuracy
#     
# 
# 5. Note : This is vectorized implementation
# 6. Positive Criticsm and suggestion to improve tutorial are heartly-welcomed.

# In[ ]:


import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
import os


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


dataset = pd.read_csv('/kaggle/input/housepricedata.csv')
dataset


# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# In[ ]:


data_orig = np.genfromtxt('/kaggle/input/housepricedata.csv',delimiter=',',skip_header=1)
print("Dataset : \n\n"+ str(data_orig))
print("\nDimensions of dataset : "+str(data_orig.shape))


# In[ ]:


#Extacting Y
y_orig = data_orig[:,-1]
print("Output Y   :"+str(y_orig))
print("Shape of Y : "+str(y_orig.shape))


# In[ ]:


#Removing Rank 1 array
Y = np.reshape(y_orig,(y_orig.shape[0],1)).T    
print("Shape of Y: "+ str(Y.shape))
print((np.sum(Y)/1460)*100)


# In[ ]:


#Extracting vectorized input feature X (transposed)
X = data_orig[:,0:-1].T


# In[ ]:


print(X.shape)


# In[ ]:


#Splitting into Train, Test sets ( with a fixed seed )
train_split_percent = 80
test_split_percent = 20

train_X , test_X = X[:, : int( (train_split_percent/100)*X.shape[1])] , X[:,int( (train_split_percent/100)*X.shape[1]) : ]
train_Y , test_Y = Y[:, : int( (train_split_percent/100)*X.shape[1])] , Y[:,int( (train_split_percent/100)*X.shape[1]) : ]
print("\nShape of Training set X : "+str(train_X.shape))
print("Shape of Training set Y : "+str(train_Y.shape))
print("\nShape of Test set   X   : "+str(test_X.shape))
print("Shape of Test set Y     : "+str(test_Y.shape))


# In[ ]:


m_train = train_X.shape[1]
m_test  = test_X.shape[1]
print("No of training examples : "+str(m_train))
print("No of test example      : "+str(m_test))
print((np.sum(1-test_Y)/292)*100)


# In[ ]:


def standardize(x):
    x_mean = np.mean(x,axis=1, keepdims=True)
    x_std = np.std(x, axis=1, keepdims=True)+0.0000001

    X = (x - x_mean)/x_std   #Python Broadcasting

    return X


# In[ ]:


train_X = standardize(train_X)
test_X  = standardize(test_X)


# In[ ]:


def sigmoid(Z):
    sigz= 1/(1+np.exp(-Z))
    sigz[sigz==1] = 0.99999999999
    sigz[sigz==0] = 0.000000000001
    return sigz        

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;


# In[ ]:


def initialize_parameters(n_x, n_h, n_y):
    
    W1 = np.random.randn(n_h,n_x)*0.1
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.1
    b2 = np.zeros((n_y,1))
    
    p = {"W1": W1,"b1": b1,   "W2": W2, "b2": b2}
    
    return p 


# In[ ]:


def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)            

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*(2/layer_dims[l-1])**0.5
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
    return parameters


# In[ ]:


def linear_forward(A, W, b):
   
    Z = np.dot(W,A)+b
    #Z = standardize(Z) Batch-Normalize with u,var=1
    cache = (A, W, b)
    
    return Z, cache


# In[ ]:


def linear_activation_forward(A_prev, W, b, activation,layer):
    
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z), sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z), relu(Z)
        dropout_cache = A
        D = np.random.rand(A.shape[0],A.shape[1]) 
        if layer==1:
            D[:,:]=1
        else:
            D = (D < keep_prob).astype(int)                                         
            A = A*D                                         
            A = A/keep_prob 
        global Dcache 
        Dcache = D
    
    cache = (linear_cache, activation_cache,Dcache)

    return A, cache


# In[ ]:


def forwardprop(X, parameters):

    caches = []
    D = []
    A = X
    L = len(parameters) // 2                
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)],"relu",l)
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)],"sigmoid",l)
    caches.append(cache)
            
    return AL, caches


# In[ ]:


def compute_cost(AL, Y,parameters):
    
    m = Y.shape[1]
    #print(AL)
    cost = (-1/m)*(np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL)))
    sumW = 0
    L = len(parameters) // 2 
    for l in range(1, L):
        sumW= sumW + np.sum(parameters["W"+str(l)])
        
    L2_cost= lambd*(sumW)/(2*m)
    cost = cost + L2_cost
    cost = np.squeeze(cost)     
   
    return cost


# In[ ]:


def linear_backward(dZ, linear_cache,keep_prob):
    
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev, dW, db


# In[ ]:


def linear_activation_backward(dA, cache, activation,keep_prob):

    linear_cache, activation_cache, dropout_cache = cache
    global dA_prev, dW, db
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache,keep_prob)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache,keep_prob=1)
    
    return dA_prev, dW, db


# In[ ]:


def backprop(AL, Y, caches):
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    #print(caches[-2][-1].shape)
    #print(L)
    
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,"sigmoid",keep_prob=1)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        global Dprev_cache
        D_prev = caches[l-1][2]
        global dA_prev_temp, dW_temp, db_temp
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                "relu",keep_prob)
        if l > 0:
            dA_prev_temp = np.multiply(dA_prev_temp,D_prev)
            dA_prev_temp = dA_prev_temp/keep_prob
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# In[ ]:


def initialize_adam(parameters) :

    L = len(parameters) // 2 
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0],parameters["W" + str(l+1)].shape[1]))
        v["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0],parameters["b" + str(l+1)].shape[1]))
        s["dW" + str(l+1)] = np.zeros((parameters["W" + str(l+1)].shape[0],parameters["W" + str(l+1)].shape[1]))
        s["db" + str(l+1)] = np.zeros((parameters["b" + str(l+1)].shape[0],parameters["b" + str(l+1)].shape[1]))
   
    return v, s


# In[ ]:


def update_parameters(parameters, grads, v, s, t,m, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8,):

    L = len(parameters) // 2                 
    v_corrected = {}                         
    s_corrected = {}                        
    
    # Perform Adam update on all parameters
    for l in range(L):
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)]+(1-beta1)*grads['dW'+str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)]+(1-beta1)*grads['db'+str(l+1)]
       
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-pow(beta1,t)) 
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-pow(beta1,t))
        
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)]+(1-beta2)*np.power(grads['dW'+str(l+1)],2)
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)]+(1-beta2)*np.power(grads['db'+str(l+1)],2)

        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-pow(beta2,t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-pow(beta2,t))

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*np.divide(v_corrected["dW" + str(l+1)],np.sqrt(s_corrected["dW" + str(l+1)])+epsilon)
        parameters["W" + str(l+1)] = parameters["W"+ str(l+1)] +(lambd/m)*parameters["W" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*np.divide(v_corrected["db" + str(l+1)],np.sqrt(s_corrected["db" + str(l+1)])+epsilon)

    return parameters, v, s


# In[ ]:


def model(X, Y, layers_dims, learning_rate0 = 0.003, epocs = 3000,
                  beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, print_after=1):

    costs = []                      
    
    parameters = initialize_parameters_deep(layers_dims)
    v, s = initialize_adam(parameters)
    
    t = 0
    m=X.shape[1]
    
    for i in range(0, epocs):
        AL, caches = forwardprop(X, parameters)
        cost = compute_cost(AL, Y,parameters)
        grads = backprop(AL, Y, caches)
        
        t = t + 1
        learning_rate = learning_rate0/(1+decay_rate*i)
        
        parameters, v, s = update_parameters(parameters, grads, v, s,
                                                               t,m, learning_rate, beta1, beta2,  epsilon,)
        if i % print_after == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if  i % print_after == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per '+str(print_after)+')')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[ ]:


def evaluate(Y,Yhat,Set):
    spos=0
    
    for i in range(Y.shape[1]): 
        if Y[0,i]==1 and Yhat[0,i]==1:
            spos = spos+1
            
    p = spos /np.sum(Yhat == 1)
    r = spos/ np.sum( Y == 1)
    acc = np.mean(Y == Yhat)
    f1score = 2*p*r/(p+r)
    
    #print(Set+" :       "+str(p) + "  "+str(r)+"  "+str(f1score)+"  "+str(acc))
    data = [{'Precision': p, 'Recall': r, 'Accuracy': acc,'F-score': f1score}] 
    df = pd.DataFrame(data)
    print('\n'+Set+'\n')
    print(df)
    #error = (1-acc)*100
    #print(Set+" :       "+'%0.3f'%error+" %" +'\t'+str(f1score))
    
    return


# In[ ]:


global dropout_cache
global keep_prob
global lambd

np.random.seed(3)
keep_prob=1
lambd=0
decay_rate=0

p = model(train_X, train_Y, layers_dims = [10,64,32,16,8,4,1], epocs =901, 
                           learning_rate0 = 0.000088,  beta1 = 0.9, beta2 = 0.9,  epsilon = 1e-8, 
                           print_after = 50)

def predict(X,p):
    keep_prob=1
    AL = forwardprop(X, p)[0]
    Y_prediction = AL
    for i in range(AL.shape[1]):
          Y_prediction[0, i] = 1 if AL[0, i] > 0.5 else 0
   
    return Y_prediction 

test_Yhat = predict(test_X,p)
train_Yhat = predict(train_X,p)


#print("    "+" :       "+ "\t Precision " + "  "+ "     \tRecall" +"  "+"          F-score "+"  "+"         Accuracy")
evaluate(train_Y,train_Yhat,"Train")
evaluate(test_Y,test_Yhat,"Test ")

print("\nSeed of initialization : "+str(3))


# In[ ]:


predict_me = test_X[:,:5]
predict_me.shape


# In[ ]:


predict(predict_me,p)


# In[ ]:




