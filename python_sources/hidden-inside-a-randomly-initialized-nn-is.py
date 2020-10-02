#!/usr/bin/env python
# coding: utf-8

# # What is hidden in a randomly weighted neural network?
# 
# This is a small implementation of a fully connected neural network in numpy/cupy ( i switched to the latter since it took a long time on CPU, pretty interchangeable save for some parts of the code where the cupy array has to be converted to numpy array)
# 
# TL ; DR of the paper : Chonky neural nets contains wisdom. 
# 
# The paper explores the posibility of finding a good subnetwork inside a randomly initialized neural network that could perform well on a certain task. How the authors approach finding this subnetwork is by assigning a score corresponding to each weights, and during feedforward, only weights with a top K % score is considered. Optimization is done on the score, while weights are kept constant.
# 
# In this implementation, i observe how both approaches perform on a similar task : Digits classification in MNIST.
# 
# Sadly as of today (12/13/2019) sometimes i have convergence issues in either of both, depending on the random state (even after setting up the random seed!). Probably i will implement a better optimizer that isn't as dependant on initial distribution.
# 
# **TODO : ** (as of 12/13/2019)
# - Deal with the variability depending on random things
# - Implement this in an actual framework ASAP :p

# In[ ]:


# import numpy as np
import cupy as np
import matplotlib.pyplot as plt


# In[ ]:


np.random.seed(69420)


# # Implementation : Baseline
# 
# Since we need a comparison, we will first see how would a fully connected NN trained with normal backprop perform.

# In[ ]:


def glorot_init(shape):
    fan_out,fan_in = shape
    limit = np.sqrt(6/(fan_in+fan_out))
    params = np.random.uniform(low=limit*-1,high=limit,size=shape)
    return params


# In[ ]:


def kaiming_uniform(shape):
    fan_out, fan_in = shape
    params = np.random.normal(loc=0,scale=np.sqrt(2/fan_in),size=shape)
    return params


# In[ ]:


def relu(a,prime=False):
    if not prime:
        mask = a > 0
        return a * mask
    else:
        mask = a > 0
        return mask.astype(np.float32)


# In[ ]:


def softmax(a,prime=False):
    a = np.exp(a)
    a = a / np.sum(a,axis=0)
    return a


# In[ ]:


def targets_to_onehot(targets,numclass):
    onehot = np.zeros((numclass,len(targets)))
    onehot[targets,np.arange(len(targets))] = 1
    return onehot


# In[ ]:


def crossentropy_with_logits(logits_out, y,prime=False):
    num_classes, num_samples = logits_out.shape
    y_onehot = targets_to_onehot(y,num_classes)
    assert logits_out.shape == y_onehot.shape
    softmaxed = softmax(logits_out)
    if prime:
        softmaxed[y,np.arange(num_samples)] -=1
        return softmaxed
    else:
        return np.sum(np.nan_to_num(y_onehot * np.log(softmaxed) + (1-y_onehot) * np.log((1-softmaxed))))


# In[ ]:


def SGDOptimizer(gradients,parameters,learningrate):
    parameters = [p - learningrate * g for p,g in zip(parameters,gradients)]
    return parameters


# In[ ]:


def MomentumSGD(lr,m):
    vl = None
    def optimizer(grads,params):
        nonlocal vl
        if vl is None:
            vl = [np.zeros(p.shape) for p in params]
        vl = [m * v + lr * g for v,g in zip(vl,grads)]
        params = [p-v for p,v in zip(params,vl)]
        return params    
    return optimizer


# In[ ]:


class FCNN(object):
    
    def __init__(self, layers_size,initializer_weight,initializer_bias,activation,loss_function,optimizer_w,optimizer_b):
        self.layers_size = layers_size
        self.weights = [initializer_weight((k,j)) for j,k in zip(layers_size[:-1],layers_size[1:])]
        self.biases = [initializer_bias((k,1)) for k in layers_size[1:]]
        self.activation = activation
        self.loss_function = loss_function
        self.optimizer_w = optimizer_w
        self.optimizer_b = optimizer_b
    
    def feedforward(self,X):
        a = X
        zs = []
        acs = [a]
        for w,b in zip(self.weights,self.biases):
            a = np.dot(w,a) + b
#             print(a.shape)
            zs.append(a)
            a = self.activation(a)
            acs.append(a)
        return a, zs, acs
    
    def feedforward_softmax(self,X):
        final_a, _, _ = self.feedforward(X)
        final_a = np.exp(final_a)
#         print(final_a.shape)
        final_a = final_a / np.sum(final_a,axis=0)
        return final_a
    
    def backpropagation(self,X,y):
        input_dim, batch_size = X.shape
        final_activation, zs, acs = self.feedforward(X)
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        error = self.loss_function(final_activation,y,prime=True) * self.activation(zs[-1],True)
        delta_w[-1] += np.einsum('ik,jk->ijk',error,acs[-2]).mean(axis=2)
        delta_b[-1] += np.expand_dims(error,1).mean(axis=2)
        for i in range(2,len(self.layers_size)):
            error = np.einsum('ik,ij->jk',error,self.weights[-i+1]) * self.activation(zs[-i],prime=True)
            delta_w[-i] += np.einsum('ik,jk->ijk',error,acs[-i-1]).mean(axis=2)
            delta_b[-i] += np.expand_dims(error,1).mean(axis=2) 
        return delta_w, delta_b
    
    def fit(self,X,y,batch_size,epochs):
        # expecting X and y to be already numpy array with the num_samples as 
        # 1st dim
        history = np.zeros(epochs + 1)
        history -= 1
        index = 0
        for epoch in range(epochs):
            indices = np.arange(X.shape[0],dtype=np.int32)
            np.random.shuffle(indices)
            X = X.copy()[indices,:]
            y = y.copy()[indices]
            i = 0
            batches_X = [X[k:k+batch_size] for k in range(0,X.shape[0],batch_size)]
            batches_y = [y[k:k+batch_size] for k in range(0,y.shape[0],batch_size)]
            losses = 0
            if index == 0:
                for Xb, yb in zip(batches_X,batches_y):
                    logits,_,_ = self.feedforward(Xb.T)
                    history[0] += self.loss_function(logits,yb.T) / (Xb.shape[0] * len(batches_X))
            for Xb, yb in zip(batches_X,batches_y):
                delta_w, delta_b = self.backpropagation(Xb.T, yb.T)
#                 print(delta_w)
                self.weights = self.optimizer_w(delta_w,self.weights)
                self.biases = self.optimizer_b(delta_b,self.biases)
                logits,_,_ = self.feedforward(Xb.T)
                losses += self.loss_function(logits,yb.T) / (Xb.shape[0] * len(batches_X))
#                 print((Xb.shape[0] * len(batches_X)))
            history[index] = losses
            index += 1
            print(f"Epoch {epoch} done. Loss : {losses}")
        return history
    
    def predict(self,X):
        softmaxed = self.feedforward_softmax(X.T)
        preds = np.argmax(softmaxed,axis=0)
        return preds
    
    def score(self, X, y):
        preds = self.predict(X)
        assert preds.shape == y.shape
        return np.mean(preds == y) * 100


# In[ ]:


import numpy as pp


# In[ ]:


mnist_data = pp.loadtxt('../input/digit-recognizer/train.csv',delimiter=',', skiprows=1)


# In[ ]:


mnist_data = np.array(mnist_data)


# In[ ]:


mnist_label = mnist_data[:,0]
mnist_pixels = mnist_data[:,1:] / 255


# In[ ]:


smol_batch_pixels = mnist_pixels[:32,:]
smol_batch_label = mnist_label[:32].astype(np.int32)


# In[ ]:


weight_optimizer = MomentumSGD(0.001,0.9)
bias_optimizer = MomentumSGD(0.001,0.9)
nn_overfit = FCNN([784,100,100,10],kaiming_uniform,kaiming_uniform,relu,crossentropy_with_logits,weight_optimizer,bias_optimizer)


# The loss did decrease, but not all the way to 0 :/ no idea if it is the learning rate or i just need more epochs, but at least we know the training iteration works and the loss decreases.

# In[ ]:


hist = nn_overfit.fit(smol_batch_pixels,smol_batch_label,32,1000)


# In[ ]:


samples = mnist_pixels[:120,:]
samples_y =  mnist_label[:120]


# Yeah, not good, just curious how badly it would perform

# In[ ]:


nn_overfit.score(smol_batch_pixels,smol_batch_label)


# In[ ]:


plt.figure()
plt.plot(-1 * np.asnumpy(hist[hist != -1]))


# ## Now for the actual training

# In[ ]:


weight_optimizer = MomentumSGD(0.001,0.9)
bias_optimizer = MomentumSGD(0.001,0.9)
nn = FCNN([784,100,100,10],kaiming_uniform,kaiming_uniform,relu,crossentropy_with_logits,weight_optimizer,bias_optimizer)


# In[ ]:


indices = np.arange(len(mnist_label),dtype=np.int32)
np.random.shuffle(indices)


# #### Data is p big lets save 10% for test (around 420 for each classes)

# In[ ]:


cutoff = int(len(indices) * 0.9)


# In[ ]:


indices_train = indices[:cutoff]
indices_test = indices[cutoff:]


# In[ ]:


train_pix, train_lab = mnist_pixels[indices_train], mnist_label[indices_train].astype(np.int32)
test_pix, test_lab = mnist_pixels[indices_test], mnist_label[indices_test].astype(np.int32)


# In[ ]:


fit_history = nn.fit(train_pix,train_lab,32,30)


# Nice and smooth

# In[ ]:


plt.figure()
plt.plot(np.asnumpy(fit_history[fit_history != -1]) * -1)


# ## Performance of FCNN trained with normal backpropagation

# This is our performance using normal FCNN

# In[ ]:


print(f"Performance on training data : {nn.score(train_pix,train_lab)} %")


# In[ ]:


print(f"Performance on test data : {nn.score(test_pix,test_lab)} %")


# **Not good, but eh, not our main goal.**

# # Score-trained NN

# Just remember to make it extra wide.

# In[ ]:


class ScoreFCNN(object):
    
    def __init__(self, layers_size,topKPercentage,initializer_weight,initializer_score,activation,loss_function,optimizer):
#         assert len(layers_size) == (len(topKPercentage)+1)
        self.layers_size = layers_size
        self.weights = [initializer_weight((k,j)) for j,k in zip(layers_size[:-1],layers_size[1:])]
        self.scores = [initializer_score(w.shape) for w in self.weights]
        self.topK = [int(j*k*topKPercentage) for j,k in zip(layers_size[:-1],layers_size[1:])]
        assert len(self.weights) == len(self.scores)
        self.activation = activation
        self.loss_function = loss_function
        self.optimizer = optimizer
    
    def feedforward(self,X):
        a = X
        zs = []
        acs = [a]
        for w,s,k in zip(self.weights,self.scores,self.topK):
            ss = s.copy()
            ww = w.copy()
#             assert not np.shares_memory(ss,s)
#             assert not np.shares_memory(ww,w)
#             ss = np.abs(ss)
            ss_flat = ss.ravel()
            sort_idx = ss_flat.argsort()
            ss_flat[sort_idx[-k:]] = 1
            ss_flat[sort_idx[:-k]] = 0
            assert ss.shape == w.shape
            ww = ww * ss
#             print(ww)
#             print(ww.shape)
            a = np.dot(ww,a)
            zs.append(a)
            a = self.activation(a)
            acs.append(a)
        return a, zs, acs
    
    def feedforward_softmax(self,X):
        final_a, _, _ = self.feedforward(X)
        final_a = np.exp(final_a)
        final_a = final_a / np.sum(final_a,axis=0)
        return final_a
    
    def backpropagation(self,X,y):
        input_dim, batch_size = X.shape
        final_activation, zs, acs = self.feedforward(X)
        delta_s = [np.zeros(s.shape) for s in self.scores]
        error = self.loss_function(final_activation,y,prime=True) * self.activation(zs[-1],True)
        delta_s[-1] += np.einsum('ik,jk->ijk',error,acs[-2]).mean(axis=2)
        for i in range(2,len(self.layers_size)):
            error = np.einsum('ik,ij->jk',error,self.weights[-i+1]) * self.activation(zs[-i],prime=True)
            delta_s[-i] += np.einsum('ik,jk->ijk',error,acs[-i-1]).mean(axis=2) * self.weights[-i]
        return delta_s
    
    def fit(self,X,y,batch_size,epochs):
        # expecting X and y to be already numpy array with the num_samples as 
        # 1st dim
        history = np.zeros(epochs + 1)
        history -= 1
        index = 1
        for epoch in range(epochs):
            indices = np.arange(X.shape[0],dtype=np.int32)
            np.random.shuffle(indices)
            X = X.copy()[indices,:]
            y = y.copy()[indices]
            i = 0
            batches_X = [X[k:k+batch_size] for k in range(0,X.shape[0],batch_size)]
            batches_y = [y[k:k+batch_size] for k in range(0,y.shape[0],batch_size)]
            losses = 0
            if index == 0:
                for Xb, yb in zip(batches_X,batches_y):
                    logits,_,_ = self.feedforward(Xb.T)
                    history[0] += self.loss_function(logits,yb.T) / (Xb.shape[0] * len(batches_X))
            for Xb, yb in zip(batches_X,batches_y):
                delta_s = self.backpropagation(Xb.T, yb.T)
#                 print(delta_w)
                self.scores = self.optimizer(delta_s,self.scores)
                logits,_,_ = self.feedforward(Xb.T)
                losses += self.loss_function(logits,yb.T) / (Xb.shape[0] * len(batches_X))
#                 print((Xb.shape[0] * len(batches_X)))
            history[index] = losses
            index += 1
            print(f"Epoch {epoch} done. Loss : {losses}")
        return history
    
    def predict(self,X):
        softmaxed = self.feedforward_softmax(X.T)
        preds = np.argmax(softmaxed,axis=0)
        return preds
    
    def score(self, X, y):
        preds = self.predict(X)
        assert preds.shape == y.shape
        return np.mean(preds == y) * 100


# **First i overfit the model to make sure everything works**

# In[ ]:


optimizer = MomentumSGD(0.1,0.9)
scorenn = ScoreFCNN([784,200,200,10],0.5,kaiming_uniform,kaiming_uniform,relu,crossentropy_with_logits,optimizer)


# In[ ]:


hist = scorenn.fit(smol_batch_pixels,smol_batch_label,32,1000)


# #### Seems nice, not all the way to 0 but still

# I cringed a bit seeing the loss, anyone knows what could be the cause?

# In[ ]:


plt.figure()
plt.plot(np.asnumpy(hist[1:]) * -1)


# #### Probs have something to do with the depth / width, at least we know it works

# In[ ]:


scorenn.score(smol_batch_pixels,smol_batch_label)


# Testing on a deeper and wider model.
# 
# According to the paper, wider is better, even at constant parameter count.
# 
# For starters, lets keep the 3 layers model, and make the hidden layers 1.25 times larger

# In[ ]:


newoptimizer = MomentumSGD(0.1,0.9)
newscorenn = ScoreFCNN([784,250,250,10],0.5,kaiming_uniform,kaiming_uniform,relu,crossentropy_with_logits,newoptimizer)


# In[ ]:


history = newscorenn.fit(train_pix,train_lab,32,30)


# Ignore the first epoch error i messed up something for sure xd

# In[ ]:


plt.figure()
plt.plot(np.asnumpy(history[1:]) * -1)


# # Performance of Score NN

# This is our performance using scoreNN. Pretty good, huh?

# In[ ]:


print(f"Performance on training data : {newscorenn.score(train_pix,train_lab)} %")


# In[ ]:


print(f"Performance on test data : {newscorenn.score(test_pix,test_lab) } %")


# ## Summary
# 
# In short, it works! But to what extent? What is the practical use? How does this differ from normal weight optimization?
# 
# In a competitive environment like kaggle, such questions are important! But unfortunately, i have yet to experiment further to see how much does it differ and what kind of advantage it can give me.
# 
# Time will tell, who knows..
