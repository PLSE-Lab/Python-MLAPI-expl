#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # H1
# ## H2
# ### H3
# #### H4
# ##### H5
# ###### H6
# 
# Alternatively, for H1 and H2, an underline-ish style:
# 
# Alt-H1
# ======
# 
# Alt-H2
# ------

# In[ ]:


print ("Hello")


# In[ ]:


import numpy as np

x=np.array([2,4,3,5,6])
y=np.array([10,5,9,4,3])

E_x=np.mean(x)
E_y=np.mean(y)

cov_xy=np.mean(x*y)- E_x*E_y

y_0= E_y- cov_xy / np.var(x)* E_x
m= cov_xy/np.var(x)

y_pred=m*x+y_0

print("E[(y_pred-y_actual)^2]=", np.mean(np.square(y_pred-y)))


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Create data
N = 500
x = np.random.rand(N)
y = np.random.rand(N)
colors = (0,0,0)
area = np.pi*3

# Plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# # **Homework1 Method 1** : *linear regression prediction* (Using Python Matplotlib)
# ## X = [2,4,3,5,6]
# ## Y = [10,5,9,4,3]
# ## Y_Prediction = M * X + B

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

x=np.array([2,4,3,5,6])
y=np.array([10,5,9,4,3])

E_x=np.mean(x)
E_y=np.mean(y)

cov_xy=np.mean(x*y)- E_x*E_y

y_0= E_y- cov_xy / np.var(x)* E_x
m= cov_xy/np.var(x)

y_pred=m*x+y_0


N = 500
colors = (0,0,0)
area = np.pi*3

# Plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Homework: linear regression prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y_pred, color='Green', alpha=0.5)
plt.show()


# # **Homework1 Method 2** : * Random linear regression prediction* (Using Javascript TensorFlow.js)
# ## X = [2,4,3,5,6]
# ## Y = [10,5,9,4,3]
# ## Y_Prediction = M * X + B
# ## M (slope of the line) is randomized tf.variable(tf.scalar(Math.random()))
# ## B (scalar offset along y) is randomized tf.variable(tf.scalar(Math.random()))
# 
# ###[Live sample](http://www.nur-tech.net/uaeu/trends/homework1/random-linear-regression/) (The page will produce a random linear regression prediction every time you open it)
# ###[Source code download](http://www.nur-tech.net/uaeu/trends/homework1/random-linear-regression/index.zip)
# 
# ![1](http://www.nur-tech.net/uaeu/trends/homework1/random-linear-regression/1.jpg)
# ![2](http://www.nur-tech.net/uaeu/trends/homework1/random-linear-regression/2.jpg)
# 
# #### Citation : [Basic Tutorial with TensorFlow.js: Linear Regression](https://medium.com/@tristansokol/basic-tutorial-with-tensorflow-js-linear-regression-aa68b16e5b8e)
# 

# # **Homework1 Method 3** : * linear regression prediction* (Using Javascript TensorFlow.js)
# ## X = [2,4,3,5,6]
# ## Y = [10,5,9,4,3]
# ## Y_Prediction = M * X + B
# ## To optimize the linear regression model from method 2, a predict function is added along with a tidy helper function for cleaning X tensor values.
# ## A loss function is also added to determin how fit is the data (this method will be used in the next function)
# ## Finally, a train function is added that will use the loss function and train the predicted data to optmize our model.
# 
# ###[Live sample](http://www.nur-tech.net/uaeu/trends/homework1/linear-regression/) (Train the data with a click)
# ###[Source code download](http://www.nur-tech.net/uaeu/trends/homework1/linear-regression/index.zip)
# 
# ![1](http://www.nur-tech.net/uaeu/trends/homework1/linear-regression/1.jpg)
# 
# #### Citation : [Basic Tutorial with TensorFlow.js: Linear Regression](https://medium.com/@tristansokol/basic-tutorial-with-tensorflow-js-linear-regression-aa68b16e5b8e)
# 
