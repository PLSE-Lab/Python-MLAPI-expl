#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/USA_Housing.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.distplot(df['Price'])


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


df.columns


# In[ ]:


X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[ ]:


y=df['Price']


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


print(lm.intercept_)


# In[ ]:


print(lm.coef_)


# In[ ]:


pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[ ]:


pred = lm.predict(X_test)


# In[ ]:


pred


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
import matplotlib
plt.scatter(y_test,pred)


# In[ ]:





# In[ ]:





# In[ ]:




