#!/usr/bin/env python
# coding: utf-8

# **ML Evaluative Lab-I**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


df.describe()


# In[ ]:


df.fillna(df.mean(),inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


# df['feature7'].value_counts()


# In[ ]:


df[df.rating==6].count()


# In[ ]:


sns.boxplot(x='rating',y='feature6',data=df)


# In[ ]:


# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# Converting the 'type' attribute which is categorical to numerical value

# In[ ]:


df['type']=df['type'].astype('category').cat.codes


# In[ ]:


sns.countplot(x = df['rating'])


# In[ ]:


corr_values=corr['rating'].sort_values(ascending=False)
corr_values=abs(corr_values).sort_values(ascending=False)
print("Correlation of mentioned features wrt outcome in ascending order")
print(abs(corr_values).sort_values(ascending=False))


# In[ ]:


sns.distplot(df['feature1'],kde = False)


# In[ ]:


features=['feature6','feature8','feature11','type','feature4','feature2']
#           ,'feature7','feature3','feature5','feature8','feature10','feature1','feature9','feature11']


# In[ ]:


X_data=df.drop(['id','rating'],axis=1)


# In[ ]:


Y_data=df['rating']


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data) 


# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X_scaled,Y_data,test_size=0.35,random_state=42) 
# X_train,X_val,y_train,y_val = train_test_split(x,y,test_size=0.35,random_state=42) 


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=250)
model=rfc.fit(X_train, y_train)


# In[ ]:


from math import sqrt
from sklearn.metrics import mean_squared_error
rfc_pred = model.predict(X_val)
amae_lr = sqrt(mean_squared_error(rfc_pred,y_val))

print("Mean Squared Error of Linear Regression: {}".format(amae_lr))


# **Approach 2:**

# Used random forest for this approach also, however changed the number of estimators to 200 and didnot scale the features first

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.35,random_state=42) 


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc1 = RandomForestClassifier(n_estimators=200)
model1=rfc1.fit(X_train, y_train)


# In[ ]:


rfc_pred1 = model1.predict(X_val)
amae_lr = sqrt(mean_squared_error(rfc_pred1,y_val))

print("Mean Squared Error of Linear Regression: {}".format(amae_lr))


# **Tested on the test dataset, used all the train dataset for training the model**

# In[ ]:


test=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')


# In[ ]:


# data1=test[features]
# data1['type']=data1['type'].astype('category').cat.codes
data1=test.drop(['id'],axis=1)


# In[ ]:


data1.fillna(data1.mean(),inplace=True)


# In[ ]:


data1['type']=data1['type'].astype('category').cat.codes


# In[ ]:


data1.isnull().sum()


# In[ ]:


# data1 = scaler.transform(data1)


# In[ ]:


rfc_pred = model1.predict(data1)
# amae_lr = sqrt(mean_squared_error(rfc_pred,y_val))

# print("Mean Squared Error of Linear Regression: {}".format(amae_lr))


# In[ ]:


rfc_pred


# In[ ]:


compare = pd.DataFrame({'id': test['id'], 'rating' : rfc_pred})


# In[ ]:


compare.to_csv('submission1.csv',index=False)


# In[ ]:




