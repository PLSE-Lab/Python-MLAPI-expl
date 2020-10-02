#!/usr/bin/env python
# coding: utf-8

# # LINEAR REGRESSION IN ONE VARIABLE (GRADIENT DESCIENT AND COST FUNCTION

# WE WILL NOT BE USING THE SKKIT LIBRARY INITIALLY ,BUT WILL CALCULATE THE GRADIENT DESCENT AND COST FUNCTION OURSELVES AND TRY TO PREDICT THE HOUSING PRICES OR ANY OTHER SINGLE VARIABLE REGRESSION PROBLEM BY REDUCING OUR COST FUNCTION AND ARRIVING AT OUR PREDICTED FUNCTION . WE WILL USE THE SKKIT LIBRARY IN THE END TO  PERFORM LINEAR REGRESSION IN ONE VARIABLE AND MULTIPLE VARIABLE AS WELL. LET'S GET STARTED :)
# 

# In[ ]:


#import these
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import seaborn as sns
import pandas as pd 
import numpy as np
from sklearn import linear_model


# In[ ]:


#NoW suppose we have a dataset
x=[1,2,3,4,5]
y=[5,7,9,11,13]


# Here's what our cost function looks like-

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
img = mpimg.imread('../input/linear-regression-analysis/cost_function.png')
imgplot = plt.imshow(img)


# In[ ]:


#Let's plot it
plt.scatter(x,y,marker='o',color='r',label='skitscat')
plt.xlabel('X')
plt.ylabel('Y')
#plt.grid(False,color='k')
plt.title('Scatter')
plt.legend()
plt.show()


# Now we want to reduce the m and b and we do that using learning rate and derivatives of m and b

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
img = mpimg.imread('../input/linear-regression-analysis/learningrate.png')
imgplot = plt.imshow(img)


# Here's what the MEAN SQUARED ERROR FUNCTION(COST FUNCTION) is and also derivative of m and b ,so that we minimize it

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
img = mpimg.imread('../input/linear-regression-analysis/MSEvsb.png')
imgplot = plt.imshow(img)


# In[ ]:


import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
get_ipython().run_line_magic('matplotlib', 'inline')
img = mpimg.imread('../input/linear-regression-analysis/learningrate.png') 
imgplot = plt.imshow(img)


# In[ ]:


#now let's create a gradient descent function
n=len(x)
x=np.array(x)
y=np.array(y)
learningrate=0.08
def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=10000
    
    for i in range(iterations):
        y_predic=m_curr*x + b_curr
        cost=(1/n)*sum([val**2 for val in (y-y_predic)])
        md=-(2/n)*sum(x*(y-y_predic))
        bd=-(2/n)*sum(y-y_predic)
        m_curr=m_curr-learningrate*md
        b_curr=b_curr-learningrate*bd
        print(cost,m_curr,b_curr)
gradient_descent(x,y)
        
    


# As you can see the cost function reduces , we have to alter the iterations and the learning rate ourselves.

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
img = mpimg.imread('../input/linear-regression-analysis/Learningr.png')
imgplot = plt.imshow(img)


#  ##NOW LET'S USE SKLEARN
# 

# In[ ]:


reg=linear_model.LinearRegression()

df=pd.DataFrame(columns=['x','y'])
df['x']=x
df['y']=y
df.head()


# In[ ]:


df[['x']]


# In[ ]:


df['y']


# In[ ]:


reg.fit(df[['x']],df.y)
reg.predict([[2.5]])


# In[ ]:


print(reg.coef_,reg.intercept_)


# # LINEAR REGRESSION WITH TWO VARIABLES

# In[ ]:


homeprices=pd.read_csv('../input/linear-regression-analysis/homeprices.csv')
homeprices
df=homeprices
df.fillna(3.0,inplace=True)
df
#You need to fill the nan values with exploratory data analytics
#WE are using 3.0 here


# In[ ]:


reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
reg.predict([[3000,2000,3.0]])

