#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 


# In[ ]:


data = pd.read_csv('/kaggle/input/suv-data/suv_data.csv')


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


df = pd.DataFrame(data.info())
df


# In[ ]:


sns.countplot(x='Gender',data=data)


#  #  As we can see from above countplot this data contains more number of Female than male

# In[ ]:


sns.countplot(x='Gender',data=data[data['Purchased']==1])


# female customer purchased car more than male 

# In[ ]:


plt.figure(figsize = (5,5))
sns.distplot(data['Age'],color='maroon')


# 

# In[ ]:


plt.figure(figsize = (5,5))
sns.distplot(data[data['Purchased']==1]['Age'],color='maroon')


# number customers who purchased cars are in between age range of 45-55 

# In[ ]:


sns.pairplot(data=data,hue='Gender')


# In[ ]:


plt.figure(figsize = (15,7))
sns.barplot(x=data['Age'],y=data['Purchased'])


# **customers having age 18-26 have not purchased car**

# In[ ]:


plt.figure(figsize = (15,7))
sns.lineplot(x=data['EstimatedSalary'],y=data['Purchased'])


# In[ ]:


data['EstimatedSalary'].max()


# In[ ]:


data['EstimatedSalary'].min()


# In[ ]:


data[data['Purchased'] ==1 ]['EstimatedSalary'].max()


# In[ ]:


data[data['Purchased'] ==1 ]['EstimatedSalary'].min()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Gender"] = le.fit_transform(data['Gender'])


# In[ ]:


data.head()


# # Creating Model 

# In[ ]:


x = data.drop(['Purchased'],axis=1).values
y = data['Purchased'].values


# In[ ]:


x_train ,x_test,y_train,y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr =LogisticRegression()
lr.fit(x_train,y_train)


# In[ ]:


pred = lr.predict(x_test)


# In[ ]:


pred


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cm = confusion_matrix(y_test,pred)
sns.heatmap(cm,annot=True)


# In[ ]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)

