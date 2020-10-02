#!/usr/bin/env python
# coding: utf-8

# **Joseph Ho - Homework 2**

# For Part A, I did two sets of analysis on the NBA Data sets that I researched. I was really interested in one of the capabilities I mentioned in the Project 2 Data Exploration document but it did not have an outcome variable that was a continuous numeric variable. Because of this, the first part of Part A in this notebook corresponds to certain calculations for the analysis and hypothesis I made in the Project 2 document and the second part of Part A is where the Homework 2 truly begins. 

# *Part A.1*: This part consists of calculations done for the Project 2 document. See Part A.2 for Homework 2.
# 

# In[18]:


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns

print(os.listdir("../input"))


# In[60]:


d1=pd.read_csv("../input/2017-18_teamBoxScore.csv")
df = d1[["teamAbbr","teamFIC40","teamRslt"]]
df = df.replace({'Win': 1, 'Loss': 0})
df.head(10)


# In[42]:


df.isnull().values.any()


# In[43]:


df.hist()


# In[62]:


print(df)


# In[63]:


gs = df.loc[df['teamAbbr'] == 'GS']
gs.head()


# In[64]:


gs.corr()


# In[65]:


inputDF = gs[["teamFIC40"]]
outcomeDF = gs['teamRslt']
model = lm.LinearRegression()
results = model.fit(inputDF,outcomeDF)

print(model.intercept_, model.coef_)


# ***Part A.2*: Player Height(in) and Field Goal Percentage **

# In[45]:


d2=pd.read_csv("../input/2017-18_playerBoxScore.csv")
df = d2[["playHeight","playFG%"]]
df.columns = ["height","fgPercent"]
print(df)


# In[46]:


df.isnull().values.any()


# In[47]:


df.hist()


# In[48]:


df.corr()


# 1. Fit the linear model.

# In[49]:


h = df[["height"]]
f = df[["fgPercent"]]


# In[50]:


model = lm.LinearRegression()
results = model.fit(h,f)
print(model.intercept_, model.coef_)


# 2. Create scatterplot

# In[51]:


plt.scatter(h,f, alpha=0.50)
plt.show()


# 3. Analyze the linear model fitted and examine whether predictor variables seems to have a significant influence on the outcome (No wrong answer here)

# Looking at the linear model fitted and variables relationship, the predictor variables look like they do not have a significant influence on the outcome. 

# 4. Using the model, predict the outcomes for the same data points 

# In[52]:


y = model.predict(h)
print(y)


# In[53]:


plt.scatter(h,f)
plt.plot(h,y, color="blue")
plt.show()


# 5. Compare the outcomes with the actual values of the test data set.

# In[ ]:


print(df)


# Comparing to the actual values of the test data set, the predicted outcomes are fairly similar to actual values of the test data set.

# 6. Use your numpy and Pandas skills to calculate the sum of squares of residuals for your model

# In[54]:


mean = (df.sum(axis = 0, skipna = True))/26109
print(mean)


# In[55]:


m = pd.DataFrame(data=df["height"])
a = pd.DataFrame(data=df["height"])
for col in m.columns:
    m[col].values[:] = 78.984488
s = np.sum((m[:]-a[:])**2)
print(s)


# ***Part B***
# 

# 1. Select 5 variables from your dataset. For each, draw a boxplot and analyze your observations.

# In[32]:


d1=pd.read_csv("../input/2017-18_playerBoxScore.csv")
d1.head(10)


# In[20]:


plt.boxplot(d1["playPTS"])
plt.show()


# In[21]:


plt.boxplot(d1["playAST"])
plt.show()


# In[56]:


plt.boxplot(d1["playSTL"])
plt.show()


# In[57]:


plt.boxplot(d1["playBLK"])
plt.show()


# In[58]:


plt.boxplot(d1["playTO"])
plt.show()


# 2. Select four pairs of variables from your dataset. Draw a scatterplot for each pair and make your visual observations.

# In[35]:


a1=pd.read_csv("../input/2017-18_playerBoxScore.csv")
a1.head(10)


# In[37]:


plt.scatter(a1["playHeight"], a1["playWeight"], alpha=0.50)
plt.show()


# In[39]:


plt.scatter(a1["playHeight"], a1["playPTS"], alpha=0.50)
plt.show()


# In[40]:


plt.scatter(a1["playHeight"], a1["playAST"], alpha=0.50)
plt.show()


# In[41]:


plt.scatter(a1["playHeight"], a1["playBLK"], alpha=0.50)
plt.show()

