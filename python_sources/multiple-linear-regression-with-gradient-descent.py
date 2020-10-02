#!/usr/bin/env python
# coding: utf-8

# This post we look at some pseudo code for simple linear regression and multiple linear regression <br>
# This is continuation of the [simple linear regression post](https://www.kaggle.com/rakend/simple-linear-regression-using-gradient-descent)

# Take home from this pseudo code can be observing how the vector size of the feature vector and parameter vector changes for SLR and MLR.  <br>
# This can be little confusing and frustating at times 

# ![](https://drive.google.com/uc?id=1LIaxxO9MuJUvKSruCkzxBHaY_upb0LI5)

# ![](https://drive.google.com/uc?id=14HYJy_z_hJjZiNIQ_c81Ec3ePcg4cgG2)
# 

# ![](https://drive.google.com/uc?id=1hGL590sluMBRr_5tntG8Qo_PV7VxaNo3)

# ![](https://drive.google.com/uc?id=1rFbWrjPfT1CMoCw9hR9C6QWtwAvPPHAt)

# ![](https://drive.google.com/uc?id=1Y6XiDWfzdCsUkVytA_7wcFrw40FAON05)

# ![](https://drive.google.com/uc?id=15CR0HEwHlVqZHFg66uY0dHkt5tKgqJ9k)

# ![](https://drive.google.com/uc?id=1u6kHk2ujLmFZG89Wo8h-n27OSp2JR_CG)

# ![](https://drive.google.com/uc?id=1OAI7GmHHTCwOGpxVUaZQyi7xnw_nNM4U)

# ![](https://drive.google.com/uc?id=1ENyf4CtlHwFshz8szsEvsO6qFvbsgNv8)

# ### Working

# Like, in the earlier post, here too we'll try solving MLR from the scratch and also from sci-kit learn package. <br>
# Now we'll jump in to see how 

# In[1]:


from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


# In[2]:


boston = datasets.load_boston()


# In[3]:


boston.keys()


# In[4]:


df = pd.DataFrame(data = boston.data)


# In[5]:


df.columns = boston.feature_names


# In[6]:


df.head()


# In[7]:


df['Target'] = boston.target


# In[8]:


df = df.rename(columns = {'Target':'Price'})


# In[9]:


corr = df.corr()


# In[10]:


print(boston.DESCR)


# In[11]:


corr['Price'].sort_values(ascending = False)


# After RM (average number of rooms), LSTAT (lower status of the population) is well (negatively) correlated with Price than the other 
# features. <br>
# 

# So it's like if the low status people are living more at a particular house, then its price is low

# We'll just look at the absolute values of the correlation

# In[12]:


corr_values = corr['Price'].abs().sort_values(ascending = False)
corr_values


# So, we'll try to do multivariate linear regression analysis with these two features 

# ### Data Standardization

# In[13]:


from sklearn import preprocessing


# In[14]:


x_RM = preprocessing.scale(df['RM'])
x_LSTAT = preprocessing.scale(df['LSTAT'])
y = preprocessing.scale(df['Price'])


# In[15]:


from pylab import rcParams
rcParams['figure.figsize'] = 12,8


# Plotting both the features in same plot

# In[16]:


plt.scatter(y, x_RM, s=5, label = 'RM')
plt.scatter(y, x_LSTAT, s=5, label = 'LSTAT')
plt.legend(fontsize=15)
plt.xlabel('Average number of rooms & Low status population', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.legend()
plt.show()


# ### Trying from the scratch

# Adding column of ones to x vector

# In[17]:


x = np.c_[np.ones(x_RM.shape[0]),x_RM, x_LSTAT]


# In[18]:


# Parameters required for Gradient Descent
alpha = 0.0001   #learning rate
m = y.size  #no. of samples
np.random.seed(10)
theta = np.random.rand(3)  #initializing theta with some random values


# In[19]:


def gradient_descent(x, y, m, theta, alpha):
    cost_list = []   #to record all cost values to this list
    theta_list = []  #to record all theta_0 and theta_1 values to this list 
    prediction_list = []
    run = True
    cost_list.append(1e10)    #we append some large value to the cost list
    i=0
    while run:
        prediction = np.dot(x, theta)   #predicted y values theta_0*x0+theta_1*x1
        prediction_list.append(prediction)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)   #  (1/2m)*sum[(error)^2]
        cost_list.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))   # alpha * (1/m) * sum[error*x]
        theta_list.append(theta)
        if cost_list[i]-cost_list[i+1] < 1e-9:   #checking if the change in cost function is less than 10^(-9)
            run = False

        i+=1
    cost_list.pop(0)   # Remove the large number we added in the begining 
    return prediction_list, cost_list, theta_list


# In[20]:


prediction_list, cost_list, theta_list = gradient_descent(x, y, m, theta, alpha)
theta = theta_list[-1]


# In[21]:


plt.title('Cost Function J', size = 30)
plt.xlabel('No. of iterations', size=20)
plt.ylabel('Cost', size=20)
plt.plot(cost_list)
plt.show()


# Using equation of hyperplane

# In[22]:


yp = theta[0] +theta[1]*x[:,1] + theta[2]*x[:,2]


# ### Mean square of residuals

# From Gradient descent prediction list 

# In[23]:


MSE_equ = ((yp-y)**2).mean()  #Using yp from equation of hyperplane
MSE_GD = ((prediction_list[-1]-y)**2).mean()  #From Gradient Descent


print('Mean Square Error using equation of hyperplane : {}'.format(round(MSE_equ,3)))
print('Mean Square Error from Gradient Descent prediction : {}'.format(round(MSE_GD,3)))


# # Using sci-kit learn

# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


ys = df['Price']
xs = np.c_[df['RM'],df['LSTAT']]


# Now if you check the shape of ys and xs vectors

# In[26]:


ys.shape, xs.shape


# ### Data Standardizing

# In[27]:


xs = preprocessing.scale(xs)
ys = preprocessing.scale(ys)


# In[28]:


lm = LinearRegression()

#Fitting the model
lm = lm.fit(xs,ys)


# In[29]:


pred = lm.predict(xs)


# This predicted vector will be of same size as of ys

# In[30]:


pred.shape


# We'll check the parameters from sci-kit learn

# In[31]:


intercept = lm.intercept_
Theta_0 = lm.coef_[0]
Theta_1 = lm.coef_[1]

print('Intercept : {}'.format(round(intercept,3)))
print('Theta_0 : {}'.format(round(Theta_0,4)))
print('Theta_1 : {}'.format(round(Theta_1,4)))


# From Gradient descent doing from the scratch

# In[32]:


print('Intercept : {}'.format(round(theta[0],3)))
print('Theta_0 : {}'.format(round(theta[1],4)))
print('Theta_1 : {}'.format(round(theta[2],4)))


# We got almost same values by both ways

# ### Model performance
R square
# In[33]:


r2_sk = lm.score(xs,ys)
print('R square from sci-kit learn: {}'.format(round(r2_sk,4)))


# From Gradient descent doing from the scratch

# In[34]:


r2 = 1 - (sum((y - prediction_list[-1])**2)) / (sum((y - y.mean())**2))
print('R square doing from the scratch: {}'.format(round(r2,4)))


# ### Animation

# In[35]:


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D



# Create a figure and a 3D Axes
fig = plt.figure(figsize=(12,10))
ax = Axes3D(fig)
ax.set_xlabel('Rooms', fontsize = 15)
ax.set_ylabel('Population', fontsize = 15)
ax.set_zlabel('Price', fontsize = 15)

plt.close()


# In[36]:


def init():
    ax.scatter(xs[:,0], xs[:,1], ys, c='C6', marker='o', alpha=0.6) 
    x0, x1 = np.meshgrid(xs[:,0], xs[:,1])
    yp = Theta_0 * x0 + Theta_1 * x1
    ax.plot_wireframe(x0,x1,yp, rcount=200,ccount=200, linewidth = 0.5,color='C9', alpha=0.5)
    ax.legend(fontsize=15, labels = ['Data points', 'Hyperplane'])
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,


# Animate

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)

# plt.legend(fontsize=15, labels = [''])
anim.save('animation.gif', writer='imagemagick', fps = 30)
plt.close()


# In[37]:


#Display the animation...
import io
import base64
from IPython.display import HTML

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))


# ### 2D view

# We'll see how this looks in 2D view

# In[38]:


# Function for getting the 2D view

def plot_view(elev_given, azim_given):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Scatter plot
    ax.scatter(xs[:,0], xs[:,1], ys, c='C6', marker='o', alpha=0.6, label='Data points')

    #Plane 

    x0, x1 = np.meshgrid(xs[:,0], xs[:,1])
    yp = Theta_0 * x0 + Theta_1 * x1
    ax.plot_wireframe(x0,x1,yp, rcount=200,ccount=200, linewidth = 0.5, color='C9', alpha=0.5, label='Hyperplane')

    ax.set_xlabel('Rooms', fontsize = 15)
    ax.set_ylabel('Population', fontsize = 15)
    ax.set_zlabel('Price', fontsize = 15)
    plt.legend(fontsize=15)
    ax.view_init(elev=elev_given, azim=azim_given)
    
    


# How our plane fits with data points along the feature 'Average no. of rooms'

# In[39]:


plot_view(-23,91)
plt.show()


# How our plane fits with data points along the feature 'Proportional of residential land'

# In[40]:


plot_view(158,-172)
plt.show()


# #### Between different features

# We can check the model performance by choosing other features which are not well correlated with Price feature <br>
# Say, we can consider ZN - proportion of residential land, this has something like 0.36 correlationg with Price

# In[41]:


x_ZN = df['ZN']


# In[42]:


xs = np.c_[df['RM'],df['ZN']]
xs = preprocessing.scale(xs)


# In[43]:


lm = lm.fit(xs,ys)


# In[44]:


lm.score(xs,ys)


# Here we got higher $R^2$ using RM and LSTAS than using RM and ZN <br>
# Higher $R^2$ tells you how reliable the fit of your hyperplane to the dataset. 

# ### Simple vs Multivariate

# We' ll check model performance simple vs multivariate

# Checking with single feature, using RM

# In[45]:


xsingle = preprocessing.scale(df['RM'])


# In[46]:


xsingle = xsingle.reshape(-1,1)
lm = lm.fit(xsingle,ys)


# In[47]:


lm.score(xsingle,ys)
print('R square from sci-kit learn using single feature: {}'.format(round(lm.score(xsingle,ys),4)))


# Whereas for multiple features we got 0.638 <br>
# One thing we need to keep in mind here is, everytime we add a feature to the model, $R^2$ will increase <br>
# So, model with more terms may appear to have a better fit simply because it has more terms or more predictors

# We can address this by using Adjusted R squred ($R^2_{adj}$)  instead of $R^2$

# It is given by <br>
# $$R^2_{adj} = 1 - \dfrac{(1-R^2)(m-1)}{m-n-1}$$

# m is the number of samples in the dataset <br>
# and n is the number of features we considered here 

# In[48]:


# Adjusted R square : 
1 - (1-r2_sk)*(df.shape[0]-1)/(df.shape[0]-2-1)


# Adjusted R squared is still higher than R square from single feature

# We can see the multiple linear regression model performs better than simple linear regression model.  <br>
# So, overall this linear regression model performs better when used two features RM and ZN instead of using just RM

# With the same above approach, we can include other features and try to fit our hyperplane for multiple features. 
