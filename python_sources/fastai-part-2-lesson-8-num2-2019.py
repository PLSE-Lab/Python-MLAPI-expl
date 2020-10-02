#!/usr/bin/env python
# coding: utf-8

# ## This is last part of fastai part 2 lesson 8, though it is mixed with my notes, eksperiments and what I found usefull, if you want the pure version, check fastai github or the following link: https://github.com/fastai/course-v3/tree/master/nbs/dl2

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# # The forward and backward passes

# In[ ]:


#export
# from exp.nb_01 import *

def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s): return (x-m)/s  #since we know about boardcasting we make a funktion that our tensor(x) subrat with the mean(m) and divided with standard diviation(s)
def test_eq(a,b): test(a,b,operator.eq,'==')


# In[ ]:


#export test if the floats are neer since we cant just compare floats with eachother 
def near(a,b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)
def test_near(a,b): test(a,b,near)


# In[ ]:


#export
from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'


# In[ ]:


path = datasets.download_data(MNIST_URL, ext='.gz'); path


# In[ ]:


with gzip.open(path, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')


# In[ ]:


x_train,y_train,x_valid,y_valid = map(tensor, (x_train,y_train,x_valid,y_valid))
n,c = x_train.shape
x_train, x_train.shape, y_train, y_train.shape, y_train.min(), y_train.max()


# In[ ]:


# get the mean and standard diviation (std()) for the normalize function
train_mean,train_std = x_train.mean(),x_train.std()
train_mean,train_std #note we want then to be 0 or 1 


# In[ ]:


x_train = normalize(x_train, train_mean, train_std)
# NB: Use training, not validation mean for validation set
x_valid = normalize(x_valid, train_mean, train_std)


# In[ ]:


train_mean,train_std = x_train.mean(),x_train.std()
train_mean,train_std #note that after the normalizer we get it really close to 0 and 1


# In[ ]:


#export # to test if the mean is near 0 and std is near 0 
def test_near_zero(a,tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"


# In[ ]:


test_near_zero(x_train.mean())
test_near_zero(1-x_train.std()) #note it is usally 1 so we subrat 1 from the std, so it should be 0 now, when we test


# In[ ]:


n,m = x_train.shape #the size og the training set
c = y_train.max()+1 #the number of activations we will need in our model
n,m,c


# # Foundations version

# ## Basic architecture

# There is one hidden layer, and usally we would wan 10 activations since there are 10 tensor a the output, but we will 
# use MAE, so it means we only need one activation a the end.

# In[ ]:


# num hidden
nh = 50


# for the model there are to layers so we need two weight matrices and two bias vectors 

# In[ ]:


# the below code is a simplified kaiming method ( init / he init )
w1 = torch.randn(m,nh)/math.sqrt(m) #randn = normal random numbers of size m(784) by nh(50), 
# /math.sqrt(m) we do this because we want t.mean(),t.std() between 0 and 1 
b1 = torch.zeros(nh)
w2 = torch.randn(nh,1)/math.sqrt(nh) #randn = normal random numbers of size nh(50) by 1
b2 = torch.zeros(1)


# In[ ]:


test_near_zero(w1.mean())
test_near_zero(w1.std()-1/math.sqrt(m))


# In[ ]:


# This should be ~ (0,1) (mean,std)...
x_valid.mean(),x_valid.std()


# In[ ]:


def lin(x, w, b): return x@w + b #this is not how our first layer are defined because first layer will have a relu


# In[ ]:


t = lin(x_valid, w1, b1)


# In[ ]:


#...so should this, because we used kaiming init, which is designed to do this
t.mean(),t.std()


# In[ ]:


def relu(x): return x.clamp_min(0.) #this means take our data (x) and replace any negative data with 0 


# In[ ]:


t = relu(lin(x_valid, w1, b1)) #this is our first layer 


# In[ ]:


#but unfornunally this does not give us a mean on 0 and a std on 1 and therefor we have to use a different technic 
t.mean(),t.std()


# The reason the above code wont be optimal is because that we remove every negative number under the actiovation. So efter a few runs most of the data will be gone. 

# ![image.png](attachment:image.png)

# the graf above show the blue dots as being points in our dataset with a std of 1. Here we remove haf and this is a problem when we train models since we have half as much data to train on and as the number of runs it has the more the data is removed

# 
# From pytorch docs: a: the negative slope of the rectifier used after this layer (0 for ReLU by default)
# 
# $$\text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan_in}}}$$
# This was introduced in the paper that described the Imagenet-winning approach from He et al: Delving Deep into Rectifiers, which was also the first paper that claimed "super-human performance" on Imagenet (and, most importantly, it introduced resnets!)

# numpy. std() in Python. numpy. std(arr, axis = None) : Compute the standard deviation of the given data (array elements) along the specified axis(if any).. Standard Deviation (SD) is measured as the spread of data distribution in the given data set

# In[ ]:


# kaiming init / he init for relu
w1 = torch.randn(m,nh)*math.sqrt(2/m)


# In[ ]:


w1.mean(),w1.std()


# In[ ]:


t = relu(lin(x_valid, w1, b1))
t.mean(),t.std()


# In[ ]:


#export
from torch.nn import init


# In[ ]:


w1 = torch.zeros(m,nh)
init.kaiming_normal_(w1, mode='fan_out')  #this is the same as the above formular though with a forwards pass
t = relu(lin(x_valid, w1, b1))


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'init.kaiming_normal_')


# In[ ]:


w1.mean(),w1.std()


# In[ ]:


t.mean(),t.std()


# In[ ]:


w1.shape


# In[ ]:


import torch.nn


# In[ ]:


torch.nn.Linear(m,nh).weight.shape #this goes to show that ...


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'torch.nn.Linear.forward')


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'torch.nn.functional.linear')


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'torch.nn.Conv2d')


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'torch.nn.modules.conv._ConvNd.reset_parameters')


# In[ ]:


# what if...?
def relu(x): return x.clamp_min(0.) - 0.5 #take the data set and set all the negative to 0, and then subrat 0.5 


# In[ ]:


# kaiming init / he init for relu 
w1 = torch.randn(m,nh)*math.sqrt(2./m )
t1 = relu(lin(x_valid, w1, b1))
t1.mean(),t1.std() #this shows that the mean and std is much better and closer to 0 and 1 


# In[ ]:


# a model in pytoch can just be a function, so this model is the forward pass  
def model(xb):
    l1 = lin(xb, w1, b1) #one linear layer
    l2 = relu(l1) #one relu (activation) layer
    l3 = lin(l2, w2, b2) #and one more linear layer
    return l3


# In[ ]:


get_ipython().run_line_magic('timeit', '-n 10 _=model(x_valid) #9.45 ms is fast enoth')


# In[ ]:


assert model(x_valid).shape==torch.Size([x_valid.shape[0],1]) #check if the shape is still the same 


# # Loss function: MSE

# The next we need is a loss function to check how well we are doing 

# In[ ]:


model(x_valid).shape # we can see the model have to more then a singel vector, so we have to remove the unit axsis 
# we do this by using squeeze 


# In[ ]:


#export
def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean() # you can also write squeeze(1) wich is the same axsis in this chase
#so we take our output and subrat from target (targ) take it to the power of 2 and find the mean of that. This is the MSE (mean qsure error)


# In[ ]:


y_train,y_valid = y_train.float(),y_valid.float()  #to use the loss function the data has to be floats 


# In[ ]:


preds = model(x_train) #make a prediction 


# In[ ]:


preds.shape #the shape of our prediction


# In[ ]:


mse(preds, y_train) #use the above funktion for mse


# # Gradients and backward pass

# the backward pass is the one of which that tells us how to update our parameters 
# for this we use gradients 

# ![image.png](attachment:image.png)

# the above picture show to formulars, that tells the same thing. The thing it tells is how we go from input(x) though the model and to mse to make the target prediction. 

# In[ ]:


def mse_grad(inp, targ): 
    # gradient of loss function with respect to output of previous layer
    inp.g = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]
    #the mse is just the input squared subratet with the target and the derevitive of that is 2 times input minus target 
    #we have to store the gradient somewhere since for the chain rule, we have to multiply all the these things together (see above picture)
    #so we can store it in the dot g (.g) of the previous layer. So the input of the mse is the same as the output of the previous layer


# ![image.png](attachment:image.png)

# In[ ]:


def relu_grad(inp, out):
    # gradient of relu with respect to input activations. 
    inp.g = (inp>0).float() * out.g
    #the above picture shows relu activation, where the gradient is either 1 or 0, so we can write (inp>0) so the input is greater then 0
    #.float() to make it a float. though we still need to use the chain rule, so we have to multiply it with the gradient of the next layer
    #which we stored away in (.g). 


# 

# In[ ]:


def lin_grad(inp, out, w, b):
    # gradient of matmul with respect to input
    inp.g = out.g @ w.t()  # gradient of output with respect to input
    w.g = (inp.unsqueeze(-1) * out.g.unsqueeze(1)).sum(0) # gradient of output with respect to weight 
    b.g = out.g.sum(0)  # gradient of output with respect to bias
    
    #inp.g = out.g @ w.t()  = we do the same thing for the linear layer but we do a gradient of a matrix product where is simply a matrix product with a transpos (.t())
    


# In[ ]:


def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    # we don't actually need the loss in backward!
    loss = mse(out, targ)
    
    # backward pass: #note we do it in "omvendt" order since we use the chain rule 
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)
    
    #backpropagation is just the chain rule where we save away the intermidet calulation (.g)so we dont have to calulate them again
    


# In[ ]:


forward_and_backward(x_train, y_train)


# In[ ]:


# Save for testing against later
w1g = w1.g.clone()
w2g = w2.g.clone()
b1g = b1.g.clone()
b2g = b2.g.clone()
ig  = x_train.g.clone()


# We cheat a little bit and use PyTorch autograd to check our results.

# In[ ]:


xt2 = x_train.clone().requires_grad_(True)
w12 = w1.clone().requires_grad_(True)
w22 = w2.clone().requires_grad_(True)
b12 = b1.clone().requires_grad_(True)
b22 = b2.clone().requires_grad_(True)


# In[ ]:


def forward(inp, targ):
    # forward pass:
    l1 = inp @ w12 + b12
    l2 = relu(l1)
    out = l2 @ w22 + b22
    # we don't actually need the loss in backward!
    return mse(out, targ)


# In[ ]:


loss = forward(xt2, y_train)


# In[ ]:


loss.backward()


# In[ ]:


# test_near(w22.grad, w2g)
# test_near(b22.grad, b2g)
# test_near(w12.grad, w1g)
# test_near(b12.grad, b1g)
# test_near(xt2.grad, ig )


# # Refactor model

# ## Layers as classes

# now we use the code from before and put it into classes. More directly we are taking each of our layers, and making them into classes, here is 'Relu' a activation and 'Lin' is the linear layer 

# In[ ]:


class Relu():
    def __call__(self, inp): #__call__ means dondercall and it does so we can treat the Relu class as a function. so if you call the Relu() without any parameters it return __Call__ function
        self.inp = inp  #save the input 
        self.out = inp.clamp_min(0.)-0.5  #save the output, where all negative numbers are turned to 0 and minus 0.5, see the code longer up in this notebook for deeper eksplination if needed
        return self.out #and let us return the output 
    
    def backward(self): self.inp.g = (self.inp>0).float() * self.out.g #this was the activation in the backward pass and we save it in self.inp.g 


# In[ ]:


class Lin():
    def __init__(self, w, b): self.w,self.b = w,b #__init__ is a special Python method that is automatically called when memory is allocated for a new object. 
        #The sole purpose of __init__ is to initialize the values of instance members for the new object
        
    #forward pass 
    def __call__(self, inp):
        self.inp = inp #save the input
        self.out = inp@self.w + self.b #save the output, where the input is a matrix multiplication with the weight matrix added with the bias vector
        return self.out #and let us return the output 
    
    def backward(self):
        self.inp.g = self.out.g @ self.w.t()  # gradient of output with respect to input
        # Creating a giandient outer product, just to sum it, is inefficient!
        self.w.g = (self.inp.unsqueeze(-1) * self.out.g.unsqueeze(1)).sum(0) # gradient of output with respect to weight 
        self.b.g = self.out.g.sum(0) # gradient of output with respect to bias


# In[ ]:


class Mse():
    def __call__(self, inp, targ):
        self.inp = inp  #save the input 
        self.targ = targ #save the target
        self.out = (inp.squeeze() - targ).pow(2).mean() #calulate the mean "kvadratrod" error
        return self.out #return the output
    
    def backward(self):
        self.inp.g = 2. * (self.inp.squeeze() - self.targ).unsqueeze(-1) / self.targ.shape[0] # and here are our gradient see code longer up in this notebook for a deeper explornation


# In[ ]:


class Model():
    def __init__(self, w1, b1, w2, b2): 
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)] #list of all our layers, note it calls an other classes defined above
        self.loss = Mse() #define loss funcion 
        
    def __call__(self, x, targ):
        for l in self.layers: x = l(x) #call are gonna go through each layer x = l(x) here we call and the function on the result of the previuas thing 
        return self.loss(x, targ) #and then we call self.loss on that 
    
    def backward(self):
        self.loss.backward() #here we do the 'direkte modsatte' 
        for l in reversed(self.layers): l.backward() #here we go through the reversed layers and call backwards on each one an remember that the backward pass are gonna
            # save the gradien away inside the (.g) -- se function above 


# In[ ]:


w1.g,b1.g,w2.g,b2.g = [None]*4 #let save all our gradients to None, so we know we are not "snyder"
model = Model(w1, b1, w2, b2) #then we create our model 


# In[ ]:


get_ipython().run_line_magic('time', "loss = model(x_train, y_train) #and we can call it as if it was a function because '__call__'")


# In[ ]:


get_ipython().run_line_magic('time', 'model.backward() ##call our backward')


# In[ ]:


# test_near(w2g, w2.g)
# test_near(b2g, b2.g)
# test_near(w1g, w1.g)
# test_near(b1g, b1.g)
# test_near(ig, x_train.g)


# # Module.forward()

# we can now rewrite the above code so it looks better and cleaner 

# In[ ]:


class Module():
    def __call__(self, *args):
        self.args = args 
        self.out = self.forward(*args) #have something that calls forward 
        return self.out #return output
    
    def forward(self): raise Exception('not implemented') #which we will set to 'not implemented' 
    def backward(self): self.bwd(self.out, *self.args) #backward are gonna take in the output we just saved 


# In[ ]:


class Relu(Module):
    def forward(self, inp): return inp.clamp_min(0.)-0.5 #so relu have somethin called forward which uses the same code as before
    def bwd(self, out, inp): inp.g = (inp>0).float() * out.g # and backward just have this smaller version of the code as before 


# In[ ]:


class Lin(Module):
    def __init__(self, w, b): self.w,self.b = w,b
        
    def forward(self, inp): return inp@self.w + self.b
    
    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = torch.einsum("bi,bj->ij", inp, out.g) # we can rewrite the  self.w.g = (self.inp.unsqueeze(-1) * self.out.g.unsqueeze(1)).sum(0) with einsum
        self.b.g = out.g.sum(0)


# In[ ]:


class Mse(Module):
    def forward (self, inp, targ): return (inp.squeeze() - targ).pow(2).mean()
    def bwd(self, out, inp, targ): inp.g = 2*(inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]


# In[ ]:



class Model():
    def __init__(self):
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]
        self.loss = Mse()
        
    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x, targ)
    
    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()


# In[ ]:


w1.g,b1.g,w2.g,b2.g = [None]*4
model = Model()


# In[ ]:


get_ipython().run_line_magic('time', 'loss = model(x_train, y_train)')


# In[ ]:


get_ipython().run_line_magic('time', 'model.backward()')


# In[ ]:


# test_near(w2g, w2.g)
# test_near(b2g, b2.g)
# test_near(w1g, w1.g)
# test_near(b1g, b1.g)
# test_near(ig, x_train.g)


# # Without einsum

# In[ ]:


class Lin(Module):
    def __init__(self, w, b): self.w,self.b = w,b
        
    def forward(self, inp): return inp@self.w + self.b
    
    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ out.g #but we can also rewrite this again from  self.w.g = torch.einsum("bi,bj->ij", inp, out.g). Because it is realy jus the input.transpos matrix multiplied with the output gradient 
        self.b.g = out.g.sum(0)


# In[ ]:


w1.g,b1.g,w2.g,b2.g = [None]*4
model = Model()


# In[ ]:


get_ipython().run_line_magic('time', 'loss = model(x_train, y_train)')


# In[ ]:


get_ipython().run_line_magic('time', 'model.backward()')


# In[ ]:


# test_near(w2g, w2.g)
# test_near(b2g, b2.g)
# test_near(w1g, w1.g)
# test_near(b1g, b1.g)
# test_near(ig, x_train.g)


# # nn.Linear and nn.Module

# This is how Pytorch implement the same code as above 

# In[ ]:



#export
from torch import nn


# In[ ]:


class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out)]
        self.loss = mse
        
    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x.squeeze(), targ)


# In[ ]:


model = Model(m, nh, 1)


# In[ ]:


get_ipython().run_line_magic('time', 'loss = model(x_train, y_train)')


# In[ ]:


get_ipython().run_line_magic('time', 'loss.backward()')

