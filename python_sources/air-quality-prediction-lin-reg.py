#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split


# In[69]:


train = pd.read_csv('../input/Train.csv')
test = pd.read_csv('../input/Test.csv')


# In[70]:


train.describe()


# In[71]:


test.describe()


# In[72]:


train.isnull().sum()


# In[73]:


test.isnull().sum()


# ### So we got cleaned data
# 
# All are contineous

# In[74]:


#plotting correlations
num_feat=train.columns[train.dtypes!=object]
num_feat=num_feat[1:-1] 
labels = []
values = []
for col in num_feat:
    labels.append(col)
    values.append(np.corrcoef(train[col].values, train.target.values)[0,1])

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,8))
rects = ax.barh(ind, np.array(values), color='red')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation Coefficients w.r.t target");


# In[75]:


#Heatmap
corrMatrix=train[num_feat].corr()
sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))
sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');


# In[76]:


plt.hist((train.target),bins=152)
plt.show()


# In[77]:


# Let's see how the numeric data is distributed.

train.hist(bins=10, figsize=(20,15), color='#E14906')
plt.show()


# # Linear Regression 
# 
# By merging train.csv and test.csv data and then splitting in 80:20 ratio
# 
# then creating model
# 
# and calculating score

# In[78]:


combined = train.append(test)
combined.reset_index(inplace=True)


# In[79]:


combined.describe()


# In[80]:


y = combined['target']
x = combined.drop('target', axis=1)


# In[81]:


y.describe()


# In[82]:


x.describe()


# ### Splitting into 80:20

# In[83]:


X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.2)


# In[84]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[85]:


# Implimenting Linear Regression from inbuilt function of sklearn
from sklearn.linear_model import LinearRegression
reg=LinearRegression(fit_intercept=True)
model = reg.fit(X_train,y_train)
predict = model.predict(X_test)


# In[86]:


#Predicting Score
from sklearn.metrics import r2_score
print(r2_score(y_test,predict))


# In[87]:


print("Training Score %.4f"%reg.score(X_train,y_train))
print("Testing Score %.4f"%reg.score(X_test,y_test))


# # Cross Validation

# In[88]:


from sklearn.model_selection import cross_val_score


# In[89]:


scores=cross_val_score(reg,X_train,y_train,cv=10,scoring='r2')


# In[90]:


print(scores)


# In[91]:


print(scores.mean())


# In[92]:


print(scores.std())


# ### Calculating loss
# 

# In[93]:


scores=cross_val_score(reg,X_train,y_train,cv=10,scoring='neg_mean_squared_error')


# In[94]:


print(scores)


# In[95]:


#Average loss
print(scores.mean())


# In[96]:


print(scores.std())


# ## Linear Regression fit

# In[97]:


y_pred = reg.predict(X_test)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual target')
plt.ylabel('Predicted target')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],'r')
plt.tight_layout()


# ### Predicted Values

# In[98]:


y_pred.shape


# In[99]:


print("predicted values for 400 test data are", y_pred)


# In[119]:


df = pd.DataFrame(y_pred)


# In[120]:


df.index.names = ["index"]


# In[124]:


df.columns = ["Predicted values"]


# In[125]:


df.head()


# In[127]:


df.to_csv('predictedres.csv')


# In[ ]:




