#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/HR_comma_sep.csv')
df.head()
               


# In[ ]:


df.shape


# ### overview of dataset
# 
# The dataset has:
#  - About 14,999 employee observations and 10 features 

# In[ ]:


#checking data types of features
df.dtypes


# In[ ]:


df=df.rename(columns={'sales':'department',
                      'promotion_last_5years':'promotion'})


# In[ ]:


#left rate
left_rate=df.left.value_counts()/len(df)
left_rate


# ** looks like 76% of employee stayed and 24 % employess left **

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
#left_rate.plot(kind='bar');
locations=[1,2]
labels=['stayed','left']
plt.bar(locations,left_rate,tick_label=labels,alpha=0.7);
plt.title('stayed vs left');


# In[ ]:


#overview for those who left  vs those who stayed
left_summary=df.groupby('left')
left_summary.mean()


# In[ ]:


# correlation matrix
corr=df.corr()
corr


# In[ ]:


#heatmap ofcorrelation
sns.heatmap(corr);


# ## salary vs leaving comapny

# In[ ]:


sns.countplot(x='salary',hue='left',data=df).set_title('Distribution of salary');


# ## Department vs leaving company

# In[ ]:


sns.countplot(x='department',data=df).set_title('distribution of employees department wise');
plt.xticks(rotation=90);


# In[ ]:


sns.countplot(x='department',hue='left',data=df).set_title('distribution of employee who left department wise');
plt.xticks(rotation=90);


# ## Relation between satisfaction level and leaving company
# 
# 

# In[ ]:


x=df.groupby('left')['satisfaction_level']
x1=x.mean()
x1


# ## project count and leaving company

# In[ ]:


sns.countplot(x='number_project', hue='left', data=df);


# # using ML to find which features are main factor for leaving company job

# **comapring accuracy score of different algorithm**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#mapping of string value to integer categorical value
df['department']=df['department'].astype('category').cat.codes
df['salary']=df['salary'].astype('category').cat.codes
df.head()


# In[ ]:


Y=df['left']
X=df.drop('left',axis=1)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

#using logistic Regression
reg=LogisticRegression()
reg=reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print ('logistic regression accuracy score:',accuracy_score(y_test,y_predict))

#using DecisionTree
clf=DecisionTreeClassifier(random_state=42)
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
print('Decision Tree accuracy score',accuracy_score(y_test,y_predict))

#using Random Forest 
Rtree=RandomForestClassifier()
Rtree.fit(x_train,y_train)
y_predict=Rtree.predict(x_test)
print('Random forest accuracy score',accuracy_score(y_test,y_predict))


# ### random forest accuarcy is high among all .

# ## finding feature importance

# In[ ]:


importances=Rtree.feature_importances_
indices=np.argsort(importances)[::-1]
print ('feature ranking:')
for i in range(X.shape[1]):
     print ("feature no. {}: {} ({})".format(i+1,X.columns[indices[i]],importances[indices[i]]))


# In[ ]:


plt.figure(figsize=(10,6));
plt.bar(range(len(indices)),importances[indices],color='red',alpha=0.5,tick_label=X.columns[indices]);
plt.xticks(rotation='-75');


# **The  above graph show main factor for leaving the job**
# 
# 1. satisfaction level
# 2. time spend in comapny
# 3. average monthly hour work done
# 4. number of project
# 5. evaluation. 

# 
