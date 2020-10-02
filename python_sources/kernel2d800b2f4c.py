#!/usr/bin/env python
# coding: utf-8

# <nav>
#     <h1>Table of Content</h1>
#     <h5 style="padding-left:5%;"><a href="#overview">1.Problem overview</a></h5>
#     <h5 style="padding-left:5%;"><a href="#data">2.Inspecting and cleaning the Data</a></h5>
#     <h5 style="padding-left:5%;"><a href="#setup">3.Setting up the Data set</a></h5>
#     <h5 style="padding-left:5%;"><a href="#selection">4.Parameters selection</a></h5>
#     <h5 style="padding-left:5%;"><a href="#train">5.training the model</a></h5>
#     <h5 style="padding-left:5%;"><a href="#conclusion">6.Conclusion</a></h5>
# </nav>

# <a id="overview"></a>
# <h2>1.Overview</h2>

# <p>
# Welcome, in this notebook we will be treating a linear regression problem which deals with predicting the price of houses we will solve this using gradient descent</p>
# <p>This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.</p>

# In[ ]:


from shutil import copyfile


# In[ ]:


copyfile(src="../input/regressionmpdel/regressionModel.py",dst="../working/regressionModel.py")


# In[ ]:


import regressionModel as rm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm


# <a id="data"></a>
# <h2>2.Inspecting and cleaning the Data</h2>

# <p>Let's first start with loading and inspecting the data</p>

# In[ ]:


data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# We see here that except the date all the other features are real numbers</p>
# Since we don't need the if and date, let's remove them

# In[ ]:


data = data.drop(["id","date"],axis=1)


# <a id="setup"></a>
# <h2>3.Setting up the Data set</h2>

# Now, we will convert the data which is a pandas dataframe into a numpy matrix</p>
# We will then seperate our prediction target from the set

# In[ ]:


x = data.values
y = np.expand_dims(x[:,0],axis=1)


# In[ ]:


x = np.delete(x,0,1)


# Here, preProcess is used to normalize the features so that the convergence is faster

# In[ ]:


x = rm.preProcess(x)


# Now, we will seperate the set into 3 sets:
# <br>A training set with 60%
# <br>A cross validation set with 20%
# <br>A test set with 20%

# In[ ]:


a = int(x.shape[0] * 0.6)
b = int(x.shape[0] * 0.2)
xt = x[:a]
yt = y[:a]
xc = x[a:a+b]
yc = y[a:a+b]
xte = x[a+b:]
yte = y[a+b:]
theta0 = np.random.random((x.shape[1],1))
l = 0


# <a id="selection"></a>
# <h2>4.Parameters selection</h2>

# Here will plot the cost against the number of iterations which also gives us the best learning rate,the cost against the number of examples and the cost against the regularization parameter

# In[ ]:


alphas = [0.001,0.01,0.1]
nIterations = range(200)
for i,alpha in enumerate(alphas,1):
    c = []
    for nIter in tqdm(nIterations):
        t = rm.gradientDescent(xc,yc,theta0,alpha,l,nIter)
        c.append(rm.cost(xc,yc,t,0))
    plt.plot(nIterations,c)
plt.legend(alphas)
plt.show()


# We see here that 0.1 is the best learning rate with less than 25 iteration

# In[ ]:


c1,c2 = [],[]
nExp = range(1,a)
for i in tqdm(nExp):
  t = rm.gradientDescent(xt[:i],yt[:i],theta0,0.1,l,25)
  c1.append(rm.cost(xc,yc,t,0))
  c2.append(rm.cost(xt,yt,t,0))


# In[ ]:


plt.plot(nExp,c1)
plt.plot(nExp,c2)
plt.legend(["cross validation set","training set"])
plt.show()


# We see here that the cost stabilizes after a 1000 examples or so

# <a id="train"></a>
# <h2>5.Training the model</h2>

# In[ ]:


theta = rm.gradientDescent(xt,yt,theta0,0.1,l,25)


# <a id="conclusion"></a>
# <h2>6.Conclusion</h2>

# In[ ]:


print("error: ","{:.2f}".format(np.mean(100 * np.absolute(rm.predict(xc,theta) - yc)/yc)),"%")

