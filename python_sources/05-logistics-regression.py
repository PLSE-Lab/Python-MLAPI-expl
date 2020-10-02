#!/usr/bin/env python
# coding: utf-8

# <h1>Logistics Regression (Classification) </h1>
# <p>
# Trying to sparate the classes by line or Curve using Gradient Descent Algorithm

# <h3> Gradient Descent Algorithm (Logistics)</h3>
# <p>
# We start with assumpution equation (Called hypothesis) which can separte above data in two classes. 
#     <img src="https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg" width=200 align="right" />
# <p>
#  
#  $h(x) =g(w_0 + w_1 x_1+ w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2)$
# 
#  OR
#   
#  <p>
#  $h(x) =g(f(x))$  where $f(x) =w_0 + w_1 x_1+ w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2$
# 
#  where $g$ is called "sigmoid" or "logistics" function
#  $g(z) =\displaystyle\frac{1}{1+e^{-z}}$
# 
# </p>
# The coefficients with initial guess (i.e. $w_0$, $w_1$...) of $h(x)$ will be fed into the algorithm.
# Then Program will start from initial guess and then iterate steps to find the best fit.
# <p> We predict $\hat{Y}= 1$ if
#     $h(x)>=0.5$ i.e.  $f(x)>=0$ 
# <p> We predict $\hat{Y}= 0$  if
#     $h(x)<0.5$ i.e. $f(x)<0$
# <p>
#  Our objective is to minimize Error in predicted values.
#     <p>
#  $ Error=   \hat{Y}-Y$  Where  $\hat{Y}=h(X)$
#  </p>
# Since Loss involve propablities between 0 and 1. we define loss function differently. we define Loss/Cost function as follows

# <h3>Cost/Loss Function</h3>
# Loss funnction is defined as
# <p>$L(W) = \dfrac {-1}{n} \displaystyle \sum _{i=1}^n \left [ Y_{i} log(\hat{Y}_{i})+ (1-Y_{i}) log(1-\hat{Y}_{i}) \right]$
# <p> and gradient update is same as it was in case of linear. 
#     <p>$W :=  W - \alpha \frac{1}{n} \sum\limits_{i=1}^{n}(h(X) - Y)X$ 
#     
# 

# <h1> Derivation of Logistics Loss Function and Gradient Updates

# <h3>Cost/Loss Function(logistics)</h3>
# <p>
# <p>$f(x) =W^T X$  where $W^T X=w_0 + w_1 x+ w_2 y + w_3 x^2 + w_4 x y + w_5 y^2$
# <p>$g(z) =\displaystyle\frac{1}{1+e^{-z}}$
# <p>$\implies h(X)=g(f(X))=g(W^T X)=\displaystyle\frac{1}{1+e^{-W^T X}}$
# 

#  <p>We calculate loss, 
#       
#    <p>$Loss= \begin{cases} 
#               -\log(h(X)) & Y=1 \\
#               -\log(1- h(X)) & Y=0
#                \end{cases}$
#  <p> Therefore we can simplify above discrete funciton into following loss function     
#  <p>$L(W)=\frac{-1}{n} \displaystyle \sum_{i=1}^n[Ylog(h(X)) +(1-Y)log(1-h(X))]$

# <h3>Derivative of Cost/Loss Function(logistics)</h3>
#   <p>Now, 
#       
#    <p>$\log(h(X))=\displaystyle\log(\frac{1}{1+e^{-W^T X} })$  
#     <p> $\hspace{20mm}  =  -\log ( 1+e^{-W^T X} ) $
#     
#  <p> $\log(1- h(X))=\displaystyle\log(1-\frac{1}{1+e^{-W^T X}})$
#   <p>  $\hspace{25mm}=\displaystyle\log(\frac{e^{-W^T X}}{1+e^{-W^T X}})$
#   <p>  $\hspace{25mm}=\log (e^{-W^T X} )-\log ( 1+e^{-W^T X} )$
#   <p>  $\hspace{25mm}=e^{-W^T X}-\log ( 1+e^{-W^T X} )$
#     <p>$L(W)=\frac{-1}{n} \displaystyle \sum_{i=1}^n[Ylog(h(X)) +(1-Y)log(1-h(X))]$
#     <p> $\hspace{15mm}=\frac{-1}{n}\displaystyle\sum_{i=1}^n \left[Y(\log ( 1+e^{-W^T X})) + (1-Y)(-W^T X-\log ( 1+e^{-W^T X} ))\right]$    
#  <p>$\hspace{15mm}=\frac{-1}{n}\displaystyle\sum_{i=1}^n \left[YW^T X-W^T X-\log(1+e^{-W^T X})\right]$    
#  <p>$\hspace{15mm}=\frac{-1}{n}\displaystyle\sum_{i=1}^n \left[YW^T X-\log e^{W^T X}- \log(1+e^{-W^T X})\right]$  
#       $\hspace{15mm}\text{using}\hspace{15mm} \log(e^{W^T X})  = W^T X $
#  <p>$\hspace{15mm}=\frac{-1}{n}\displaystyle\sum_{i=1}^n \left[YW^T X-\log(1+e^{W^T X})\right]$ 
#     $\hspace{15mm}\text{using}\hspace{15mm} \log(X) + \log(Y) = log(X Y) $  
# 
# 
# <p>$\frac{\partial}{\partial W} L(W)$
#     $=\frac{\partial}{\partial W}(-YW^T X +\displaystyle\log(1+e^{W^T X}))$
#    <p> $\hspace{20mm}=\frac{\partial}{\partial W}(-YW^T X) +\frac{\partial}{\partial W}(\log(1+e^{W^T X})))$
#    <p> $\hspace{20mm}=-YX+\displaystyle\frac{e^{W^T X}}{1+e^{W^T X}} X$
#     <p> $\hspace{20mm}=(-Y+\displaystyle\frac{e^{W^T X}}{1+e^{W^T X }}) X$
#     <p> $\hspace{20mm}=(-Y+\displaystyle\frac{1}{1+e^{-W^T X }}) X$
#     <p> $\hspace{20mm}=(-Y+h(X)) X$
#    <p>$ Finally$
#     <p> $\implies\frac{\partial}{\partial W} L(W)=(h(X)-Y) X$
#     

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import itertools


# <H1>Data Generate

# In[ ]:


X1=[]
X2=[]
Y1=[]

for i,j in itertools.product(range(50),range(50)):
    if abs(i-j)>5 and abs(i-j)<40 and np.random.randint(5,size=1) >0:
        X1=X1+[i/2]
        X2=X2+[j/2]
        if (i>j):
            Y1=Y1+[1]
        else:
            Y1=Y1+[0]
            
X=np.array([X1,X2]).T
Y=np.array([Y1]).T


# <h5> Visualize Data

# In[ ]:


cmap = ListedColormap(['blue', 'red'])                    
plt.scatter(X1,X2, c=Y1,marker='.', cmap=cmap)
plt.show()


# <h5>Normalize Input   

# In[ ]:


SMean=np.min(X,axis=0)    #using Min-Max Normalization
SDev=np.max(X,axis=0)
def NormalizeInput(X,SMean,SDev):   
    XNorm=(X-SMean)/SDev
    return XNorm


# In[ ]:


XNorm=NormalizeInput(X,SMean,SDev)


# <h5>Add Polynomial Features

# In[ ]:


def mapFeature(X,degree):
    
    sz=X.shape[1]
    if (sz==2):
        sz=(degree+1)*(degree+2)/2
        sz=int(sz)
    else:
         sz=degree+1
    out=np.ones((X.shape[0],sz))     #Adding Bias W0

    sz=X.shape[1]
    if (sz==2):
        X1=X[:, 0:1]
        X2=X[:, 1:2]
        col=1
        for i in range(1,degree+1):        
            for j in range(0,i+1):
                out[:,col:col+1]= np.multiply(np.power(X1,i-j),np.power(X2,j))    
                col+=1
        return out
    else:
        for i in range(1,degree+1):        
            out[:,i:i+1]= np.power(X,i)
    
    return out


# In[ ]:


degree=2
inputX=mapFeature(XNorm,degree) 


# <h1>Training

# In[ ]:


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# In[ ]:


def computeCost(weights,X,Y):
    n = X.shape[0]
    fx=np.matmul( X,weights)                      #Hypothesis
    hx=sigmoid(fx)
    term1=np.sum(np.multiply(Y,np.log(hx)))
    term2=np.sum(np.multiply(np.subtract(1,Y),np.log(1-hx)))    
    J=(-1/n)*(term1+term2)
    return J


# <h5> Initialization

# In[ ]:


batchSize=len(Y)         #no of Examples
iterations = 10000
alpha = 0.9
beta1=0.99
beta2=0.999
learningDecayRate=0.999998
epsilon=0.0000000001
featureCount=inputX.shape[1] 
weights=np.zeros((featureCount, 1)) #initialize Weight Paramters
vDW=np.zeros((featureCount, 1))
sDW=np.zeros((featureCount, 1))
lossList=np.zeros((iterations,1),dtype=float)  #for plotting loss curve


# <h5> Gradient Descent Updates

# In[ ]:



for k in range(iterations):
    #nth iteration
    t=k+1
    
    #Hypothesis
    fx=np.matmul( inputX,weights)           
    
    hx=sigmoid(fx)
    
    #Loss
    loss=hx-Y  
    
    
    #derivative
    dW=np.matmul(inputX.T,loss)  #Derivative
   
    #learning Rate decrease as training progresses 
    alpha=alpha*learningDecayRate
    
    #Moment Update
    vDW = (beta1) *vDW+ (1-beta1) *dW        #Momentum  
    sDW = (beta2) *sDW+ (1-beta2) *(dW**2)   #RMSProp
    
    #Bias Correction
    vDWc =vDW/(1-beta1**t)       
    sDWc =sDW/(1-beta2**t)
    
    #gradient Update
    #weights=weights - (alpha/batchSize)*dW                           #Simple
    weights=weights - (alpha/batchSize)*vDW                          #Momentum   
    #weights=weights - (alpha/batchSize)*dW/np.sqrt(csDW+epsilon)     #RMSProp 
    #weights=weights - (alpha/batchSize)*(vDWc/(np.sqrt(sDWc)+epsilon)) #Adam          
    
    
    #Compute Loss for Plotting
    lossList[k]=computeCost(weights,inputX,Y)

print("{0:.15f}".format(lossList[iterations-1][0]))


# <h1>Plot Loss

# In[ ]:


plt.plot(lossList,color='r')
plt.show


# <h1> Prediction/Accuracy Evaluation

# In[ ]:


def predict(X,weights,SMean,SDev,degree):
    XNorm=NormalizeInput(X,SMean,SDev)
    inputX=mapFeature(XNorm,degree)
    fx=np.matmul(inputX, weights)
    hx=sigmoid(fx)
    PY=np.round(hx) 
    return PY


# In[ ]:


def accurracy(Y1,Y2):
    m=np.mean(np.where(Y1==Y2,1,0))    
    return m*100


# <h5>Accurracy on Training Data

# In[ ]:


pY=predict(X, weights,SMean,SDev,degree) 
print(accurracy(Y, pY))


# <h1>Plotting Hypothesis

# In[ ]:


plt.figure(figsize=(12,8))
plt.scatter(X[:,0],X[:,1], c=Y[:,0], cmap=cmap) 
###########################################################################
#Predict for each X1 and X2 in Grid 
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
u = np.linspace(x_min, x_max, 50) 
v = np.linspace(y_min, y_max, 50) 

U,V=np.meshgrid(u,v)
UV=np.column_stack((U.flatten(),V.flatten())) 
W=predict(UV, weights,SMean,SDev,degree) 
plt.scatter(U.flatten(), V.flatten(),  c=W.flatten(), cmap=cmap,marker='.', alpha=0.1)

###########################################################################
#Exact Decision Boundry can be plot with contour
z = np.zeros(( len(u), len(v) )) 
for i in range(len(u)): 
    for j in range(len(v)): 
        uv= np.column_stack((np.array([[u[i]]]),np.array([[v[j]]])))               
        z[i,j] =predict(uv, weights,SMean,SDev,degree) 
z = np.transpose(z) 
plt.contour(u, v, z)
###########################################################################
plt.show()


# <h3> Visualize Sigmoid for given Data points

# In[ ]:


XNorm=NormalizeInput(X,SMean,SDev)
inputX=mapFeature(XNorm,degree)
fx=np.matmul(inputX, weights)
hx=sigmoid(fx)
plt.figure(figsize=(12,8))
plt.scatter(fx,hx,c=np.round(hx), cmap=cmap)

x = np.arange(-18, 18, 0.1)
g = sigmoid(x)
plt.plot(x, g,color='g' ,linewidth=2,alpha=1)
plt.plot(x, x*0,color='k',linewidth=1,alpha=0.2)
plt.plot([-2,0,2], [0.5,0.5,0.5],color='r',alpha=0.8)
plt.plot([0,0], [-0.1,1],color='k',linewidth=1,alpha=0.2)
plt.xlabel('x')
plt.ylabel('$\sigma(z)$')
plt.show()

