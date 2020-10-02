#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe().transpose()


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.countplot(df['banking_crisis'])


# In[ ]:


sns.scatterplot(x='year',y='systemic_crisis',data=df)


# In[ ]:


sns.scatterplot(x='domestic_debt_in_default',y='year',data=df)


# In[ ]:


sns.scatterplot(x='sovereign_external_debt_default',y='year',data=df)


# In[ ]:


sns.countplot(df['cc3'])


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:



sns.jointplot(x='year',y='exch_usd',data=df)


# In[ ]:


sns.pairplot(df)


# In[ ]:


df['banking_crisis'].value_counts()


# In[ ]:


df1=df[df['banking_crisis']=='no_crisis'].head(100)


# In[ ]:


df2=df[df['banking_crisis']=='crisis'].head(94)


# In[ ]:


new_df=pd.concat([df1,df2]).sample(frac=1)


# In[ ]:


new_df.head()


# In[ ]:


new_df.drop(['cc3','country'],axis=1,inplace=True)


# In[ ]:


new_df['banking_crisis']=pd.get_dummies(new_df['banking_crisis'],drop_first=False)


# In[ ]:


new_df['banking_crisis']


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


x=new_df.drop('banking_crisis',axis=1)
y=new_df['banking_crisis']


# In[ ]:


scaler=StandardScaler()


# In[ ]:


x=scaler.fit_transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=32)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier()


# In[ ]:


rfc.fit(x_train,y_train)


# In[ ]:


predictions=rfc.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[ ]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


# In[ ]:




