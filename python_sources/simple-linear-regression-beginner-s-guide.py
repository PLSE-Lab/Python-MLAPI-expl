#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


import plotly
from plotly.offline import iplot
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


# In[8]:


df=pd.read_csv("../input/Salary_Data.csv")


# In[ ]:


df.head()


# In[ ]:


df.corr()


# In[ ]:


trace=go.Scatter(
x=list(df['YearsExperience']),
y=list(df['Salary']) ,
    
)

layout=dict(
    title='Salary vs Experience',
xaxis=dict(title='Exp'),
yaxis=dict(title='Sal')
)

data=[trace]
fig = dict(data=data, layout=layout)


# In[ ]:


iplot(fig)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


from sklearn.model_selection import train_test_split
X=df.iloc[:,:-1].values
print(X)
y=df.iloc[:,1].values
print(y)


# In[ ]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=1/3, random_state=100)
X_train


# In[ ]:


lr= LinearRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


ypred=lr.predict(X_test)


# In[ ]:


ypred


# In[ ]:


trace=go.Box(
y=list(df['Salary']),
    boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
)

data=[trace]
fig = dict(data=data)
iplot(fig)


# In[ ]:




