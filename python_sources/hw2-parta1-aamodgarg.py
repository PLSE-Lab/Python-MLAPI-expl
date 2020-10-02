#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model as lm


# ** 1.	With the data points, fit a linear model.**

# In[ ]:


o = pd.read_csv("../input/listings_detail_uploaded.csv")
o1 = o[["accommodates","price"]]
# print(o.dtypes)
# print("Correlation:\n",o1.corr())


# Explanatory variable - 'accommodates'
# Outcome Variable - 'price'

# In[ ]:


y=o1[["price"]]
x1=o1[["accommodates"]]
# print(y)
# print(x1)
d = pd.DataFrame(np.hstack((x1,y)))
d.columns = ["x1","y"]
print(d)


# #Linear Regression - model fitting

# In[ ]:



model = lm.LinearRegression()
results = model.fit(x1,y)
print(model.intercept_, model.coef_)


# #Result: Scikit-Learn

# In[ ]:


yp2 = model.predict(x1)
print(yp2)


# **2.	Draw a scatterplot with the linear model as a line.**

# In[ ]:


#Linear Regression representation using scatter plot
plt.scatter(x1,y)
plt.plot(x1,yp2, color="blue")
plt.xlabel('Number of people accommodated')
plt.ylabel('Price')
plt.title('Accommodates vs. Price')
plt.show()


# **4.	Using the model, predict the outcomes for the same data points **

# In[ ]:


#Result: Scikit-Learn
yp2 = model.predict(x1)
print(np.round(yp2[0:10],2))


# In[ ]:


print(y[0:10])
print(x1[0:10])


# **5.	Compare the outcomes with the actual values of the data set.**

# In[ ]:


# Residuals = Difference of the predictions and the original values.
a1 = yp2-y
print(a1)


# **6. Calculate the sum of squares of residuals for your model**

# In[ ]:


# calculate squares of residuals
b1 = np.square(a1)
print(b1)


# In[ ]:


# Sum the square of residuals
c1 = np.sum(b1)
print(c1)

