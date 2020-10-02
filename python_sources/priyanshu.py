#!/usr/bin/env python
# coding: utf-8

# Here we work on machine learing algoritham named "Linear Regression" using covid-19 dataset. And draw a graph between totel confirm cases and cured cases.

# In which we use some library files like numpy,pandas,matpotlib.pyplot and sklearn.
# numpy used for perform a number of mathematical operations on arrays such as trigonometric, statistical and algebraic routines.
# pandas used for data analysis and it provides highly optimized performance.
# matpolotlip.pyplot used of data visualization, its amazing library for ploting 2d graphs.
# and sklearn(Scikit-learn) is used for various algorithms like SVM(support vector machine),random forests and k-neighbours. It also support other numerical and scientific libraries.  

# In[ ]:



import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Here we read dataset by function pd.read_csv, this function used for read coumma sepreted values and we stor in the variable df here df denotes datafream.

# In[ ]:


df = pd.read_csv("../input/complete.csv")
df


# we use drop function for drop some columns which we don't need.

# In[ ]:


df = df.drop("ConfirmedcasesIndia",axis=1)
df = df.drop("ConfirmedcasesForeign",axis=1)
df


# this function used for showing columns name.

# In[ ]:


df.keys()


# Here we select column name StatesAndUT in  which we select Maharastra and store in the new variable name new_data. 

# In[ ]:


new_data = df[df["StateaAndUT"] == "Maharashtra"]
new_data


# Here we store our data of totel confirmedcases using object new_data in the x and same in y we store totel number of cured case.

# In[ ]:


x=new_data["TotalConfirmedcases"]
y=new_data["Cured"]


# In[ ]:


x=new_data.iloc[:,5:6].values


# Now we ues Linear Regression function and fit function in which we fit x and y.

# In[ ]:


LR=LinearRegression()
LR.fit(x,y)


# Now we find y_pred and print x,y and y_pred.
# 

# In[ ]:


y_pred=LR.predict(x)
y_pred
print(y)
print(x)
print(y_pred)


# Here plot the graph between x and y where x is Totel Confirmed Cases and y is cured.

# In[ ]:


plt.figure(figsize=(20,10)) 
plt.scatter(x,y)
plt.plot(x, y_pred, color="red")
plt.xlabel("TotalConfirmedcases")
plt.ylabel("Cured")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




