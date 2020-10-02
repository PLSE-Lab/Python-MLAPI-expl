#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/diabetes.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # Exploring Data Analysis Visually 

# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.countplot(x='Outcome',hue='Outcome',data=df,palette='RdBu_r')


# In[ ]:


sns.pairplot(data=df,hue='Outcome')


# In[ ]:


plt.scatter(x='Outcome',y='Age',data=df)
plt.xlabel('Outcome')
plt.ylabel('Age')


# In[ ]:


sns.distplot(df['Age'],kde=False,color='darkblue',bins=20)


# In[ ]:


df['Age'].hist(bins=40,color='Green',alpha=0.6)


# In[ ]:


sns.distplot(df['BloodPressure'],kde=False,color='darkblue',bins=20)


# In[ ]:


sns.jointplot(x='Age',y='BloodPressure',data=df)


# # Cufflinks for Plots

# In[ ]:


import cufflinks as cf 
cf.go_offline()


# In[ ]:


df['BMI'].iplot(kind='hist',bins=40,color='red')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


df.head()


# In[ ]:


X = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
y = ['Output']


# In[ ]:


df2 = pd.DataFrame(data=df)
df2.head()


# # Building Logistics Regression: Train Test Split 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome',axis=1),df['Outcome'],
                                                    test_size=0.30, random_state=101)


# # Training and Predicting 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# # Evaluation 

# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))


# # Outcome
# From the outcome we can say that the results are ~79-80% confident on the results which I see not a bad evaluation. 
