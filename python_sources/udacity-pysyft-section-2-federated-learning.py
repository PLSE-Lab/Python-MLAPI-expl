#!/usr/bin/env python
# coding: utf-8

# # Section 2: Federated Learning
# from Kaggle
# 
# A notebook from the Udacity course about Secure and Private AI
# https://eu.udacity.com/course/secure-and-private-ai--ud185

# # Lesson: Introducing Federated Learning
# 
# Federated Learning is a technique for training Deep Learning models on data to which you do not have access. Basically:
# 
# Federated Learning: Instead of bringing all the data to one machine and training a model, we bring the model to the data, train it locally, and merely upload "model updates" to a central server.
# 
# Use Cases:
# 
#     - app company (Texting prediction app)
#     - predictive maintenance (automobiles / industrial engines)
#     - wearable medical devices
#     - ad blockers / autotomplete in browsers (Firefox/Brave)
#     
# Challenge Description: data is distributed amongst sources but we cannot aggregated it because of:
# 
#     - privacy concerns: legal, user discomfort, competitive dynamics
#     - engineering: the bandwidth/storage requirements of aggregating the larger dataset

# # Lesson: Introducing / Installing PySyft
# 
# In order to perform Federated Learning, we need to be able to use Deep Learning techniques on remote machines. This will require a new set of tools. Specifically, we will use an extensin of PyTorch called PySyft.
# 
# ### Install PySyft
# 
# The easiest way to install the required libraries is with [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/overview.html). Create a new environment, then install the dependencies in that environment. In your terminal:
# 
# ```bash
# conda create -n pysyft python=3
# conda activate pysyft # some older version of conda require "source activate pysyft" instead.
# conda install jupyter notebook
# pip install syft
# pip install numpy
# ```
# 
# If you have any errors relating to zstd - run the following (if everything above installed fine then skip this step):
# 
# ```
# pip install --upgrade --force-reinstall zstd
# ```
# 
# and then retry installing syft (pip install syft).
# 
# If you are using Windows, I suggest installing [Anaconda and using the Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/) to work from the command line. 
# 
# With this environment activated and in the repo directory, launch Jupyter Notebook:
# 
# ```bash
# jupyter notebook
# ```
# 
# and re-open this notebook on the new Jupyter server.
# 
# If any part of this doesn't work for you (or any of the tests fail) - first check the [README](https://github.com/OpenMined/PySyft.git) for installation help and then open a Github Issue or ping the #beginner channel in our slack! [slack.openmined.org](http://slack.openmined.org/)

# In[ ]:


import torch as th
import syft as sy


# In[ ]:


x = th.tensor([1,2,3,4,5])
x


# In[ ]:


y = x + x


# In[ ]:


print(y)


# In[ ]:


hook = sy.TorchHook(th)


# In[ ]:


th.tensor([1,2,3,4,5])


# # Lesson: Basic Remote Execution in PySyft

# ## PySyft => Remote PyTorch
# 
# The essence of Federated Learning is the ability to train models in parallel on a wide number of machines. Thus, we need the ability to tell remote machines to execute the operations required for Deep Learning.
# 
# Thus, instead of using Torch tensors - we're now going to work with **pointers** to tensors. Let me show you what I mean. First, let's create a "pretend" machine owned by a "pretend" person - we'll call him Bob.

# In[ ]:


bob = sy.VirtualWorker(hook, id="bob")


# In[ ]:


bob._objects


# In[ ]:


x = th.tensor([1,2,3,4,5])


# In[ ]:


x = x.send(bob)


# In[ ]:


bob._objects


# ''' performing on x: 
# it is going to send a message to self.location, 
# the worker finds the tensor with the ID x.id_at_location
# execute the command'''

# In[ ]:


x.location


# In[ ]:


x.id_at_location


# In[ ]:


x.id


# In[ ]:


x.owner


# In[ ]:


hook.local_worker


# In[ ]:


x


# In[ ]:


x = x.get()
x


# In[ ]:


bob._objects


# # Project: Playing with Remote Tensors
# 
# In this project, I want you to .send() and .get() a tensor to TWO workers by calling .send(bob,alice). This will first require the creation of another VirtualWorker called alice.

# In[ ]:


# try this project here!
bob = sy.VirtualWorker(hook, 'bob')
ada = sy.VirtualWorker(hook, 'ada')

mytensor = th.Tensor([2,3,5,7])


# In[ ]:


tensor_pointer = mytensor.send(bob, ada)


# In[ ]:


print(bob._objects)
print(ada._objects)


# In[ ]:


tensor_pointer


# In[ ]:


mytensor = tensor_pointer.get()
mytensor


# In[ ]:





# In[ ]:





# In[ ]:





# # Lesson: Introducing Remote Arithmetic

# In[ ]:


x = th.tensor([1,2,3,4,5]).send(bob)
y = th.tensor([1,1,1,1,1]).send(bob)


# In[ ]:


x


# In[ ]:


y


# In[ ]:


z = x + y


# In[ ]:


z


# In[ ]:


z = z.get()
z


# In[ ]:


z = th.add(x,y)
z


# In[ ]:


z = z.get()
z


# In[ ]:


x = th.tensor([1.,2,3,4,5], requires_grad=True).send(bob)
y = th.tensor([1.,1,1,1,1], requires_grad=True).send(bob)


# In[ ]:


z = (x + y).sum()


# In[ ]:


z.backward()


# In[ ]:


x = x.get()


# In[ ]:


x


# In[ ]:


x.grad


# In[ ]:





# # Project: Learn a Simple Linear Model
# 
# In this project, I'd like for you to create a simple linear model which will solve for the following dataset below. You should use only Variables and .backward() to do so (no optimizers or nn.Modules). Furthermore, you must do so with both the data and the model being located on Bob's machine.

# In[ ]:


# try this project here!
myinput = th.tensor([[0.,0], [1,0], [0,1], [1,1]], requires_grad=True)
input_ptr = myinput.send(ada)
target = th.tensor([[0.], [1], [0], [1]], requires_grad=True).send(ada)
weights = th.tensor([[0.], [0.]], requires_grad=True).send(ada)


# In[ ]:


for i in range(10):
    prediction = input_ptr.mm(weights)
    loss = ((prediction - target)**2).sum()
    loss.backward()
    weights.data.sub_(weights.grad * 0.15)
    weights.grad *= 0

    print(loss.get().data)


# In[ ]:


ada._objects


# In[ ]:



del input_ptr
ada.clear_objects()
ada._objects


# In[ ]:





# In[ ]:





# # Lesson: Garbage Collection and Common Errors
# 

# In[ ]:


bob = bob.clear_objects()


# In[ ]:


bob._objects


# In[ ]:


x = th.tensor([1,2,3,4,5]).send(bob)


# In[ ]:


bob._objects


# In[ ]:


del x


# In[ ]:


bob._objects


# In[ ]:


x = th.tensor([1,2,3,4,5]).send(bob)


# In[ ]:


bob._objects


# In[ ]:


x = "asdf"


# In[ ]:


bob._objects


# In[ ]:


x = th.tensor([1,2,3,4,5]).send(bob)


# In[ ]:


x


# In[ ]:


bob._objects


# In[ ]:


x = "asdf"


# In[ ]:


bob._objects


# In[ ]:


del x


# In[ ]:


bob._objects


# In[ ]:


bob = bob.clear_objects()
bob._objects


# In[ ]:


for i in range(1000):
    x = th.tensor([1,2,3,4,5]).send(bob)


# In[ ]:


bob._objects


# In[ ]:


x = th.tensor([1,2,3,4,5]).send(bob)
y = th.tensor([1,1,1,1,1]).send(bob)


# In[ ]:


z = x + y
z


# # Lesson: Toy Federated Learning
# 
# Let's start by training a toy model the centralized way. This is about a simple as models get. We first need:
# 
# - a toy dataset
# - a model
# - some basic training logic for training a model to fit the data.

# In[ ]:


from torch import nn, optim


# In[ ]:


# A Toy Dataset
data = th.tensor([[1.,1],[0,1],[1,0],[0,0]], requires_grad=True)
target = th.tensor([[1.],[1], [0], [0]], requires_grad=True)

bob = sy.VirtualWorker(hook, 'bob')
ada = sy.VirtualWorker(hook, 'ada')


# In[ ]:


# A Toy Model
model = nn.Linear(2,1)


# In[ ]:


opt = optim.SGD(params=model.parameters(), lr=0.1)


# In[ ]:


def train(iterations=20):
    for iter in range(iterations):
        opt.zero_grad()

        pred = model(data)

        loss = ((pred - target)**2).sum()

        loss.backward()

        opt.step()

        print(loss.data)
        
train()


# In[ ]:


data_bob = data[0:2].send(bob)
target_bob = target[0:2].send(bob)


# In[ ]:


data_ada = data[2:4].send(ada)
target_ada = target[2:4].send(ada)


# In[ ]:


datasets = [(data_bob, target_bob), (data_ada, target_ada)]


# In[ ]:


def train(iterations=20):

    model = nn.Linear(2,1)
    opt = optim.SGD(params=model.parameters(), lr=0.1)
    
    for iter in range(iterations):

        for _data, _target in datasets:

            # send model to the data
            model = model.send(_data.location)

            # do normal training
            opt.zero_grad()
            pred = model(_data)
            loss = ((pred - _target)**2).sum()
            loss.backward()
            opt.step()

            # get smarter model back
            model = model.get()

            print(loss.get())


# In[ ]:


train()


# In[ ]:





# # Lesson: Advanced Remote Execution Tools
# 
# In the last section we trained a toy model using Federated Learning. We did this by calling .send() and .get() on our model, sending it to the location of training data, updating it, and then bringing it back. However, at the end of the example we realized that we needed to go a bit further to protect people privacy. Namely, we want to average the gradients BEFORE calling .get(). That way, we won't ever see anyone's exact gradient (thus better protecting their privacy!!!)
# 
# But, in order to do this, we need a few more pieces:
# 
# - use a pointer to send a Tensor directly to another worker
# 
# And in addition, while we're here, we're going to learn about a few more advanced tensor operations as well which will help us both with this example and a few in the future!

# In[ ]:


bob.clear_objects()
ada.clear_objects()


# In[ ]:





# In[ ]:


x = th.tensor([1,2,3,4,5]).send(bob)


# In[ ]:


x = x.send(ada)


# In[ ]:


bob._objects


# In[ ]:


ada._objects


# In[ ]:


y = x + x


# In[ ]:


y


# In[ ]:


bob._objects


# In[ ]:


ada._objects


# In[ ]:


jon = sy.VirtualWorker(hook, id="jon")


# In[ ]:


bob.clear_objects()
ada.clear_objects()

x = th.tensor([1,2,3,4,5]).send(bob).send(ada)


# In[ ]:


bob._objects


# In[ ]:


ada._objects


# In[ ]:


x = x.get()
x


# In[ ]:


bob._objects


# In[ ]:


ada._objects


# In[ ]:


x = x.get()
x


# In[ ]:


bob._objects


# In[ ]:


bob.clear_objects()
ada.clear_objects()

x = th.tensor([1,2,3,4,5]).send(bob).send(ada)


# In[ ]:


bob._objects


# In[ ]:


ada._objects


# In[ ]:


del x


# In[ ]:


bob._objects


# In[ ]:


ada._objects


# In[ ]:





# In[ ]:





# # Lesson: Pointer Chain Operations

# In[ ]:


bob.clear_objects()
ada.clear_objects()


# In[ ]:


x = th.tensor([1,2,3,4,5]).send(bob)


# In[ ]:


bob._objects


# In[ ]:


ada._objects


# In[ ]:


x.move(ada)


# In[ ]:


bob._objects


# In[ ]:


ada._objects


# In[ ]:





# In[ ]:


x = th.tensor([1,2,3,4,5]).send(bob).send(ada)


# In[ ]:


bob._objects


# In[ ]:


ada._objects


# In[ ]:


x.remote_get()


# In[ ]:


bob._objects


# In[ ]:


ada._objects


# In[ ]:


x.move(bob)


# In[ ]:


x


# In[ ]:


bob._objects


# In[ ]:


ada._objects

