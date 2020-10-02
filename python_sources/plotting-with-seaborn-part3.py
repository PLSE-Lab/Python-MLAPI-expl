#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string as st # to get set of characters


# In[ ]:


col = [a for a in st.ascii_uppercase[:10]]
tmp = np.random.randint(1,30,1000).reshape(100,10)
df = pd.DataFrame(tmp,columns=col)


# In[ ]:


df.head(2)


# In[ ]:


df.describe()


# In[ ]:


df["categ"] = np.random.choice(col[:3],100)


# In[ ]:


date = pd.date_range("1/1/2018",periods=100)
df["date"] = date


# In[ ]:


df.set_index(date,inplace=True)


# In[ ]:


df.drop("date",axis=1,inplace=True)


# In[ ]:


df.head(2)


# # My random dataset is ready with some categorical value as well as time series info

# # Joint Plot

# In[ ]:


df.head(1)


# In[ ]:


sns.jointplot(df.A,df.D) # Default joint plot with scatter plot


# In[ ]:


sns.jointplot(df.A,df.D,kind="kde") # Default joint plot with scatter plot


# In[ ]:


sns.jointplot(df.A,df.D,kind="reg") # Default joint plot with scatter plot


# In[ ]:


sns.jointplot(df.A,df.D,kind="resid") # Default joint plot with scatter plot


# In[ ]:


sns.jointplot(df.A,df.D,kind="hex") # Default joint plot with scatter plot


# In[ ]:


sns.jointplot(df.A,df.D,kind="reg",color="g") # Default joint plot with scatter plot


# # Regression (best fit line) plot

# In[ ]:


sns.regplot(df.A,df.B)


# In[ ]:


sns.regplot(df.A,df.B,x_jitter=2.9)


# In[ ]:


sns.regplot(df.A,df.B,x_jitter=2.9,marker="^",color="r")


# In[ ]:


sns.regplot(df.A,df.B,x_jitter=2.3,marker=">",color="r",line_kws={"color":"m","linewidth":5.1},ci=100,x_estimator=np.median)


# In[ ]:


sns.regplot(df.A,df.B,x_jitter=2.3,marker=">",color="r",line_kws={"color":"m","linewidth":5.1},ci=100,x_estimator=np.std)


# In[ ]:


sns.regplot(df.A,df.B,x_jitter=2.3,marker=">",color="r",line_kws={"color":"m","linewidth":5.1},ci=100,x_estimator=np.mean)


# # Pair plot

# In[ ]:


df.head(1)


# In[ ]:


sns.set_style("darkgrid")
sns.pairplot(df[["A","B","C","categ"]],hue="categ",kind="reg")


# In[ ]:


sns.set_style("darkgrid")
sns.pairplot(df,hue="categ")


# In[ ]:


sns.set_style("darkgrid")
sns.pairplot(df[["A","B","C","categ"]],hue="categ",diag_kind="hist")


# In[ ]:


sns.set_style("darkgrid")
sns.pairplot(df[["A","B","C","categ"]],hue="categ",diag_kind="hist",palette="husl")


# In[ ]:


sns.set_style("darkgrid")
sns.pairplot(df[["A","B","C","categ"]],hue="categ",diag_kind="hist",palette="husl",markers=["D","<",">"])


# In[ ]:


sns.set_style("darkgrid")
sns.pairplot(df,vars=["A","B","C"],hue="categ",diag_kind="hist",palette="husl",markers=["D","<",">"])


# In[ ]:


sns.set_style("darkgrid")
sns.pairplot(df,vars=["A","B","C"],hue="categ",diag_kind="hist",palette="husl",markers=["D","<",">"])


# In[ ]:


sns.set_style("darkgrid")
sns.pairplot(df,x_vars=["A","B"],y_vars=["C","D"],hue="categ",palette="husl",markers=["^","<",">"])

