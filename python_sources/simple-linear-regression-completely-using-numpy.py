#!/usr/bin/env python
# coding: utf-8

# # Implementing Simple Linear Regression from scratch.
# 

# # Importing Libraries
# * numpy, also known as numerical python. It is a package that deals with multidimentional array.
# * seaborn and matplotlib are libraries that helps in data visualisation.

# In[ ]:


#Simple Linear Regression

#Importing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Generating Data
# * For performing Simple Linear Regression we need to generate data.
# * x -> input
# * y -> output
# * w and b -> parameters
# * To generate random data we use np.random.randn().
# * np.random.seed() is used to repeatedly generate same random data.

# In[ ]:


#generating data for simple linear regression
np.random.seed(1)

x=np.random.randn(1000,1)  
w=np.random.randn()
b=np.random.randn()

y=x*w+b +(np.random.randn(1000,1)*0.09)  #this last term of np.random.randn() is to add noise in data


# # Visualising generated data

# In[ ]:


# plt.title('Generated data')
plt.figure(figsize=(12,8))
plt.title('Generated data')
plt.xlabel('input(x)')
plt.ylabel('output(y)')
sns.scatterplot(x=x[:,0],y=y[:,0])
plt.grid(True)
plt.show()


# # Creating model
# * SimpleLinearRegression class is created which has three functions:-
# 
#     1. predict() -> takes input 'x' and returns predicted output 'y_pred'.
#     2. cost() -> takes input 'true y' and 'predicted y' and returns the cost.
#     3. fit() ->takes input 'x' 'y' and optional arguments learning_rate and interations, it works as follows:-
#         * initailizes parameters with random values.
#         * predicts values.
#         * computes cost.
#         * uses gradient descent algorithm for optimisation.
#         * plots cost vs number of iterations.

# In[ ]:


class SimpleLinearRegression():
    def __init__(self):
        self.parameters={}
        self.m=None
    
    
    def predict(self,X):
        self.m=X.shape[0]
        try:
            y_pred=self.parameters['w1']*X+self.parameters['b']
            return y_pred
        except KeyError:
            print('First fit your data and then predict the values')

            
    def cost(self,y_pred,y_true):
        return (np.sum(np.square(y_pred-y_true)))/(2*self.m)
     
        
    def fit(self,X,y,learning_rate=0.01,iterations=1000):
        self.parameters['w1']=np.random.randn()
        self.parameters['b']=np.random.randn()
        cost=[]
        y_pred=self.predict(X)
        cost.append(self.cost(y_pred,y))
        print('Cost at iteration number '+str(1)+' is : ',cost[0])
        for i in range(iterations):
            dw1=np.sum((y_pred-y)*X)/self.m
            db=np.sum(y_pred-y)/self.m
            
            self.parameters['w1']=self.parameters['w1']-(learning_rate*dw1)
            self.parameters['b']=self.parameters['b']-(learning_rate*db)
            
            y_pred=self.predict(X)
            cost.append(self.cost(y_pred,y))
            if (i+1)%100==0:
                print('Cost at iteration number '+str(i+1)+' is : ',cost[i])
        plt.title('cost v/s iteration')
        plt.plot(cost)
        plt.xlabel('no. of iterations')
        plt.ylabel('cost')
        plt.show()


# In[ ]:


model=SimpleLinearRegression()


# In[ ]:


model.fit(x,y)


# In[ ]:


y_pred=model.predict(x)


# In[ ]:


cost=model.cost(y_pred,y)
print(cost)


# # Visualising best fit line and the data

# In[ ]:


plt.figure(figsize=(12,8))
plt.title('Simple Linear Regression')
sns.scatterplot(x=x[:,0],y=y[:,0],alpha=0.4)
sns.lineplot(x=x[:,0],y=y_pred[:,0])
plt.xlabel('x-input')
plt.ylabel('y-input')
plt.show()


# ### If this notebook was helpful to you please do upvote and comment suggestions if any.
