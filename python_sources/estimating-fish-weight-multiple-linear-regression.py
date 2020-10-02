#!/usr/bin/env python
# coding: utf-8

# # Estimating Fish Weight- Using Multiple Linear Regression
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#lg-skt">Multiple Linear Regression Using Scikit Learn Library</a></li>
# <li><a href="#lg-grd">Multiple Linear Regression Using Gradient Descent Implementation</a></li>
# <li><a href="#lg-ne">Multiple Linear Regression Using Normal Equation</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ### Introduction

# Using the fish market data set we will try to create a model to estimate fish weight, this model will be based on multiple linear regression.
# 
# We will be trying out three approaches:
#     1. Using the Scikit library.
#     2. Using the gradient descent implementation which gives us a way to fine tune hyper parameters like learning rate.
#     3. Using Normal equation implementation, this equation is helpful in a way that is does not require us to go through iterations and chossing learning rate as is the case with gradient descent.

# <a id='eda'></a>
# ### Exploratory Data Analysis

# In[ ]:


import pandas as pd
import os


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/fish-market/Fish.csv')
print(df.shape)
df.sample(10)


# Species -->	species name of fish
# 
# Weight -->	weight of fish in Gram g
# 
# Length1 --> vertical length in cm
# 
# Length2 -->	diagonal length in cm
# 
# Length3 -->	cross length in cm
# 
# Height -->	height in cm
# 
# Width -->	diagonal width in cm

# In[ ]:


print(df.Species.unique())
print(df.info())


# In[ ]:


print(df.describe())


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


for specie in df.Species.unique():
    df_spe=df.query('Species==@specie')
    print(specie)
    for col in df_spe.columns:
        if(col =='Species'):continue
        df_spe.boxplot(col)
        plt.show()


# Looking at the Specie wise box plots, we observe that for "Roach" specie, there are two records which are having anamoly.
#     1. In one record the Weight is zero.
#     2. In second record, the Weight, Length1, Length2, Length3, Height and Width measurements are all outliers.

# In[ ]:


df.query('Species=="Roach" & (Weight ==0 | Weight>350)')


# So, we will drop the record with Index 54, since in this record all the parameters are outliers.

# In[ ]:


df= df.drop([54])
df.query('Species =="Roach"').describe().T


# For the record where the weight is zero, looking at this record we see that the Length1,Length2,Length3 and width for this record lies around the 1st Quartile, and since this is just one record, we chose to replace the weight 0 with the 1st quartile value of weight.

# In[ ]:


df.iloc[40,1]=df.query('Species =="Roach"').describe().T['25%'].Weight


# In[ ]:


df.query('Species =="Roach"').describe().T


# Now we will look at the attributes in a consolidated way.

# In[ ]:


for col in df.columns:
    if(col =='Species'):continue
    df.boxplot(col)
    plt.show()


# Here we observe that there are three outliers in 'Weight' attribute.

# In[ ]:


df.query('Weight> 1500')


# In[ ]:


df= df.query('Weight<= 1500')


# In[ ]:


df.query('Weight> 1500')


# In[ ]:


sb.pairplot(df, kind='scatter', hue='Species');


# <a id='lg-skt'></a>
# ### Multiple Linear Regression Using Scikit Learn Library

# In[ ]:


from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[ ]:


# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
df['Species']= label_encoder.fit_transform(df['Species']) 

df['Species'].unique()


# In[ ]:


X=df.drop(['Weight'] , axis=1, inplace=False)
X.head()


# In[ ]:


y= df[df.columns[1:2]]


# In[ ]:


lg = LinearRegression()
lstSeed=[]
lstRMSQ=[]
lstRSq=[]
for seed in range(0,150,10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    lg.fit(X_train, y_train) #training the algorithm
    pred = lg.predict(X_test)
    root_mean_sq = np.sqrt(metrics.mean_squared_error(y_test,pred))
    r_sq = metrics.r2_score(y_test,pred)
    lstRSq.append(r_sq)
    lstSeed.append(seed)
    lstRMSQ.append(root_mean_sq)


# In[ ]:


df_metric=pd.DataFrame({
    'Seed': lstSeed, 
    'RMSQ': lstRMSQ,
    'RSQ': lstRSq})
df_metric.head()


# In[ ]:


ax=df_metric.plot('Seed', 'RMSQ',legend=False)
ax2 = ax.twinx()
df_metric.plot('Seed', 'RSQ', ax=ax2,color="r",legend=False)
ax.figure.legend()
plt.show()


# Looking at the plot we can choose the seed value to be 10

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
lg.fit(X_train, y_train) #training the algorithm
pred = lg.predict(X_test)
print('root mean sq:',np.sqrt(metrics.mean_squared_error(y_test,pred)))
print('r squared:',metrics.r2_score(y_test,pred))


# <a id='lg-grd'></a>
# ### Multiple Linear Regression Using Gradient Descent Implementation
# 
# In this section we will explicitly implement gradient descent and cost function, we will tune various parameters like learning rate, iterations etc.

# In[ ]:


X=df.drop(['Weight'] , axis=1, inplace=False)
y= df[df.columns[1:2]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train_inter = np.ones((X_train.shape[0],1))
X_train = np.concatenate((X_train_inter, X_train), axis = 1)

X_test_inter = np.ones((X_test.shape[0],1))
X_test = np.concatenate((X_test_inter, X_test), axis = 1)
print(X_train.shape)
print(X_test.shape)


# In[ ]:


def computeCost(X,y,theta):
    #number of training examples
    m= len(y)
    hypothesis= X.dot(theta)
    #Take a summation of the squared values
    delta=np.sum(np.square(hypothesis-y))
    J=(1/(2*m))*delta
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    #number of training examples
    m, n = np.shape(X)
    x_t = X.transpose()
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        hypothesis = np.dot(X,theta)-y
        gradient = np.dot(x_t, hypothesis) / m
        #update the theta
        theta = theta- alpha*gradient
        J_history[i]=np.sum(hypothesis**2) / (2*m)
    return theta,J_history

def predict(x_test,theta):
    n = len(x_test)
    predicted_vals=[]
    for i in range(0,n):
        predicted_vals.append(np.matmul(theta.T,x_test[i,:]))
    return predicted_vals

def runEpoch(X,y,theta,alpha,iterations,epochs):
    dicGradient={}
    dicRMSQ={}
    dicRSQ={}
    dicJ_Hist={}
    J_hist=[]
    X_t_act, X_valid, y_t_act, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)
    for epoch in range(epochs):
        print('Running Epoch {}'.format(epoch))
        theta,J_History=gradientDescent(X_t_act,y_t_act,theta,alpha,iterations)
        dicGradient[epoch]=(theta,J_History)
        J_hist.extend(J_History)
        pred_vals=predict(X_valid,theta)
        root_mean_sq = np.sqrt(metrics.mean_squared_error(y_valid,pred_vals))
        r_sq = metrics.r2_score(y_valid,pred_vals)
        dicRMSQ[epoch]=root_mean_sq
        print('Epoch {0}: RMSQ {1}'.format(epoch,root_mean_sq))
        dicRSQ[epoch]=r_sq
    key_min = min(dicRMSQ.keys(), key=(lambda k: dicRMSQ[k]))
    return dicGradient[key_min][0],J_hist


# In[ ]:


n=X_train.shape[1]
theta=np.zeros((n, 1))
theta,J_History=runEpoch(X_train,y_train,theta,0.00065,4000,25)
print(theta)
plt.plot(J_History);
plt.show();


# In[ ]:


pred_vals=predict(X_test,theta)
preds=[]
for pred in pred_vals:
    preds.append(abs(pred[0]))


# In[ ]:


root_mean_sq = np.sqrt(metrics.mean_squared_error(y_test,preds))
r_sq = metrics.r2_score(y_test,preds)
print('root mean sq:',root_mean_sq)
print('r squared:',r_sq)


# <a id='lg-ne'></a>
# ### Multiple Linear Regression Using Normal Equation
# 
# Normal equation, will for some linear regression (usually when features are less than aprroximately 10000) problems gives us a much better way to solve the optimal value of $\theta$
# 
# **Normal Equation**
# 
# $\theta = (X^{T}X)^{-1}X^{T}y$

# In[ ]:


def normalEquation(X,y):
    x_trans=X.T
    inv=np.linalg.pinv(np.dot(x_trans,X))
    theta=np.dot(np.dot(inv,x_trans),y)
    return theta


# In[ ]:


theta_ne= normalEquation(X_train,y_train)
print(theta_ne)


# In[ ]:


pred_vals=predict(X_test,theta_ne)
preds=[]
for pred in pred_vals:
    preds.append(abs(pred[0]))
root_mean_sq = np.sqrt(metrics.mean_squared_error(y_test,preds))
r_sq = metrics.r2_score(y_test,preds)
print('root mean sq:',root_mean_sq)
print('r squared:',r_sq)


# <a id='conclusions'></a>
# ### Conclusions

# In context of this data set we can see that the gradient descent implementation gives a bit better result than Scikit library or using Normal equation, this may be attributed to tuning parameters available with gradient descent.
# But gradient descent has drawback with respect to Normal Equation, that it has to go through lot more iterations (time consuming) and we need to choose learning rate.
