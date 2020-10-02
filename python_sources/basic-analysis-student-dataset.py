#!/usr/bin/env python
# coding: utf-8

# In[137]:


import xgboost
import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns
sns.set(style="white", color_codes=True)

import plotly.plotly as py
import plotly.graph_objs as go


import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


# In[68]:


train = pd.read_excel("../input/research_student (1).xlsx")


# In[70]:


train.head()


# In[71]:


train


# In[72]:


train.tail()


# In[73]:


train.describe()


# In[74]:


train = train.drop([0,221,222])


# In[75]:


train


# In[9]:


train.head()


# In[10]:


train.Branch.value_counts()


# In[76]:


train = train.fillna(0)


# In[77]:


train[['Branch','Board[10th]']]


# **VISUALISE DATA**

# In[88]:



train.plot(kind ='scatter',x='Marks[10th]',y='CGPA')


# In[ ]:





# In[81]:


train.plot(kind ='scatter',x='Marks[12th]',y='CGPA')


# In[82]:


train.plot(kind ='scatter',x='Rank',y='CGPA')


# In[90]:


train.head()


# In[130]:


plt.hist(train['CGPA'],rwidth=0.8)
plt.title("CGPA")
plt.show()


# In[129]:


plt.hist(train['Branch'],rwidth=0.8)
plt.title("Branch")
plt.show()


# In[105]:


plt.hist(train['Category'])
#plt.hist(histtype="bar")
plt.title("Category")
plt.show()


# In[106]:


plt.hist(train['Gender'])
plt.title("Gender")
plt.show()


# In[107]:


train.head()


# In[113]:


cgpa = train["CGPA"]

cgpa.describe()


# In[128]:


plt.hist(cgpa, range = (4,10),bins=16,rwidth=0.8)
plt.title("CGPA")


# In[143]:


from pandas.tools.plotting import radviz
#radviz(train.drop("CGPA", axis=1), "Branch")


# In[145]:


sns.boxplot(x="Branch", y="CGPA", data=train)


# In[146]:


sns.boxplot(x="Gender", y="CGPA", data=train)


# In[148]:


sns.boxplot(x="Category", y="CGPA", data=train)


# In[ ]:


train = sns.load_dataset("train")
train = train.pivot("Branch", "Gender", "CGPA")
ax = sns.heatmap(train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[78]:


train.columns


# In[15]:


scale_list = [ 'Marks[10th]', 'Marks[12th]', 'GPA 1', 'Rank', 'Normalized Rank', 'CGPA',
       'Current Back', 'Ever Back', 'GPA 2', 'GPA 3', 'GPA 4', 'GPA 5',
       'GPA 6', 'Olympiads Qualified', 'Technical Projects', 'Tech Quiz',
       'Engg. Coaching', 'NTSE Scholarships', 'Miscellany Tech Events']
sc = train[scale_list]


# In[16]:


sc.head()


# In[17]:


sc.tail()


# In[18]:


sc.fillna(0)


# In[19]:


scaler = StandardScaler()
sc = scaler.fit_transform(sc)


# In[20]:


train[scale_list] = sc
train[scale_list].head()


# In[21]:


train


# In[23]:


train.head()


# In[24]:


train.info()


# In[ ]:


encoding_list = ['Branch','Gender','Board[10th]','Board[12th]','Category']
train[encoding_list] = train[encoding_list].apply(LabelEncoder().fit_transform)


# In[26]:


train.head()


# In[27]:


train.info()


# **Linear Regression**

# In[ ]:





# In[28]:


train['CGPA']


# In[30]:


y = train['CGPA']
x = train.drop('CGPA', axis=1)


# In[31]:


x.head()


# In[32]:


x.info()


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)


# In[34]:


X_train.info()


# In[35]:


X_train.shape


# In[36]:



X_test.shape


# In[39]:


y_test.shape


# **LINEAR REGRESSION**

# In[37]:


logreg=LinearRegression()


# Training

# In[38]:


logreg.fit(X_train,y_train)


# In[40]:


y_pred=logreg.predict(X_test)


# In[41]:


y_test


# In[42]:


y_pred


# In[ ]:





# In[43]:


print(metrics.mean_squared_error(y_test, y_pred))


# **XBOOST ALGORITHM**

# In[44]:


xgb = xgboost.XGBRegressor(n_estimators=2500, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)


# In[45]:


xgb.fit(X_train,y_train)


# In[46]:


train.head()


# In[ ]:




