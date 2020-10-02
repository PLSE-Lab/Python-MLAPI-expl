#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''Age Vs Salary Classification either obove 50k or less 50k through logistic regression classification'''


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv(r'../input/adult.csv')
df.head()


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.rename(columns={'hours-per-week':'Working_Hour'}, inplace = True)
df.columns


# In[ ]:


dummies = pd.get_dummies(df.income)
dummies.head()


# In[ ]:


df_dummies = pd.concat([df,dummies],axis ='columns')
df_dummies.head()


# In[ ]:


df_dummies.drop(columns = ['fnlwgt','educational-num','marital-status','relationship','capital-gain','capital-loss'], inplace = True)
df_dummies.head()


# In[ ]:


df_dummies.rename(columns = {'<=50K':'Less_than_50K','>50K':'More_than_50K'}, inplace = True)
   


# In[ ]:


df_dummies.head()


# In[ ]:


plt.xlabel('Age')
plt.ylabel('Working Hour')
plt.scatter(df.age,df.Working_Hour,marker='+',color='red')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_dummies[['age']],df_dummies[['income']],train_size=0.8)


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# In[ ]:


y_train.head()


# In[ ]:


y_test.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


y_predicted = model.predict(X_test)
y_predicted


# In[ ]:


model.predict_proba(X_test)


# In[ ]:


model.score(X_test,y_test)

