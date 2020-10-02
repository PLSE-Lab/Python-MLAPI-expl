#!/usr/bin/env python
# coding: utf-8

# ![Crowede](https://jtf.org/wp-content/uploads/2016/05/syrian_refugees_keleti_railway_station.jpg)
# 
# **We might think that most of our social welfare rarely help those people who need help indeed. Therefore, here comes the power of data science!**
# 
# ### Outline
# 
# * Overlook
# 
# * Cleaning
# 
# * Basic EDA
# 
# * Modeling 
# 
# * Validation

# ## Overlook

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.shape


# In[ ]:


train.head(3)


# **Here, since I don't have too much time, I'll start it with some columns that I think are really important to the target. I'll add on other columns gradually if time allows**
# 
# Here are some features that I think it's important:
# 
# * v2a1, Monthly rent payment
# * hacdor, =1 Overcrowding by bedrooms
# * rooms,  number of all rooms in the house
# * hacapo, =1 Overcrowding by rooms
# * v14a, =1 has bathroom in the household
# * refrig, =1 if the household has refrigerator
# * v18q, owns a tablet
# * r4h1, Males younger than 12 years of age
# * r4h3, Total males in the household
# * r4m1, Females younger than 12 years of age
# * r4m3, Total females in the household
# * tamhog, size of the household
# * tamviv, number of persons living in the household
# * pisonotiene, =1 if no floor at the household
# * cielorazo, =1 if the house has ceiling
# * abastaguano, =1 if no water provision
# * noelec, =1 no electricity in the dwelling
# * sanitario1, =1 no toilet in the dwelling
# * epared1, =1 if walls are bad
# * epared2, =1 if walls are regular
# * epared3, =1 if walls are good
# * etecho1, =1 if roof are bad
# * etecho2, =1 if roof are regular
# * etecho3, =1 if roof are good
# * eviv1, =1 if floor are bad
# * eviv2, =1 if floor are regular
# * eviv3, =1 if floor are good
# * dis, =1 if disable person
# * idhogar, Household level identifier
# * instlevel1, =1 no level of education
# * instlevel2, =1 incomplete primary
# * instlevel3, =1 complete primary
# * instlevel4, =1 incomplete academic secondary level
# * instlevel5, =1 complete academic secondary level
# * instlevel6, =1 incomplete technical secondary level
# * instlevel7, =1 complete technical secondary level
# * instlevel8, =1 undergraduate and higher education
# * instlevel9, =1 postgraduate higher education
# * bedrooms, number of bedrooms
# * overcrowding, # persons per room

# In[ ]:


#Check the ratio of hacdor & hacapo
len(train.loc[train.hacdor == 1])/len(train.loc[train.hacapo == 1])


# In[ ]:


# Slicing the dataset
train = train[['v2a1','hacdor','rooms','hacapo','v14a','refrig','v18q','r4h1','r4h3','r4m1','r4m3','tamhog',
               'tamviv','pisonotiene','cielorazo','abastaguano','noelec','epared1',
               'epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3','dis','idhogar','instlevel1',
               'instlevel2','instlevel3','instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9',
               'bedrooms','overcrowding','Target']]


# **And there are some columns that actually can be ordinal. I'll put them together''
# 
# epared1, =1 if walls are bad   
# epared2, =1 if walls are regular   
# epared3, =1 if walls are good   
# 
# etecho1, =1 if roof are bad   
# etecho2, =1 if roof are regular   
# etecho3, =1 if roof are good  
# 
# eviv1, =1 if floor are bad   
# eviv2, =1 if floor are regular   
# eviv3, =1 if floor are good  

# In[ ]:


df = train[['epared1','epared2','epared3']]
x = df.stack()
train['epared'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
train['epared'] = train['epared'].apply(lambda x : 1 if x == 'epared1' else (2 if x == 'epared2' else 3))


# In[ ]:


df = train[['etecho1','etecho2','etecho3']]
x = df.stack()
train['etecho'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
train['etecho'] = train['epared'].apply(lambda x : 1 if x == 'etecho1' else (2 if x == 'etecho2' else 3))


# In[ ]:


df = train[['eviv1','eviv2','eviv3']]
x = df.stack()
train['eviv'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
train['eviv'] = train['epared'].apply(lambda x : 1 if x == 'eviv1' else (2 if x == 'eviv2' else 3))


# In[ ]:


train.drop(['epared1','epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3'],axis=1,inplace=True)


# ## Cleaning

# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.idhogar.value_counts()


# In[ ]:


train.drop('idhogar',axis=1,inplace=True)


# In[ ]:


pd.isnull(train).sum()


# In[ ]:


train.v2a1.describe()


# In[ ]:


sns.boxplot(train.v2a1)


# In[ ]:


train['unavailable_v2a1'] = train.v2a1.apply(lambda x: 1 if pd.isnull(x) else 0)


# In[ ]:


train['v2a1'] = train['v2a1'].fillna(130000)


# In[ ]:


sns.countplot(train.Target)
plt.title('Distribution of Target')


# In[ ]:


corr = train.corr()
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


X = train.drop('Target',axis=1)
y = train[['Target']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


# Create the model and assign it to the variable model.
model = DecisionTreeClassifier()

# Fit the model.
model.fit(X_train,y_train)
# Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X_test)

# Calculate the accuracy and assign it to the variable acc.
print('The accuracy for the model is:', accuracy_score(y_test,y_pred))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='2.0f')
plt.xlabel('Predicted label')
plt.ylabel('True label')


# In[ ]:


features = X_train.columns[:X_train.shape[1]]
importances = model.feature_importances_
indices = np.argsort(importances)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


model = DecisionTreeClassifier()

param_dist = {"max_depth": [3,5,7,9,10,15,20,None],
              "min_samples_split": [2,5,10,15],
              "min_samples_leaf": [1,3,5]}

Search = RandomizedSearchCV(model, param_distributions=param_dist)

# Fit the model on the training data
Search.fit(X_train, y_train)

# Make predictions on the test data
preds = Search.best_estimator_.predict(X_test)

print('The accuracy for the model is:', accuracy_score(y_test,preds))


# In[ ]:


sns.heatmap(confusion_matrix(y_test,preds),annot=True,fmt='2.0f')
plt.xlabel('Predicted label')
plt.ylabel('True label')


# In[ ]:


features = X_train.columns[:X_train.shape[1]]
importances = Search.best_estimator_.feature_importances_
indices = np.argsort(importances)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


test_df = test[['v2a1','hacdor','rooms','hacapo','v14a','refrig','v18q','r4h3','r4m1','r4m3','tamhog',
               'tamviv','pisonotiene','cielorazo','abastaguano','noelec','epared1','bedrooms','overcrowding','r4h1',
               'epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3','dis','instlevel1',
               'instlevel2','instlevel3','instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9']]

df = test_df[['epared1','epared2','epared3']]
x = df.stack()
test_df['epared'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
test_df['epared'] = test_df['epared'].apply(lambda x : 1 if x == 'epared1' else (2 if x == 'epared2' else 3))

df = test_df[['etecho1','etecho2','etecho3']]
x = df.stack()
test_df['etecho'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
test_df['etecho'] = test_df['epared'].apply(lambda x : 1 if x == 'etecho1' else (2 if x == 'etecho2' else 3))

df = test_df[['eviv1','eviv2','eviv3']]
x = df.stack()
test_df['eviv'] = np.array(pd.Categorical(x[x!=0].index.get_level_values(1)))
test_df['eviv'] = test_df['epared'].apply(lambda x : 1 if x == 'eviv1' else (2 if x == 'eviv2' else 3))


test_df.drop(['epared1','epared2','epared3','etecho1','etecho2','etecho3','eviv1','eviv2','eviv3'],axis=1,inplace=True)

test_df['unavailable_v2a1'] = test_df.v2a1.apply(lambda x: 1 if pd.isnull(x) else 0)
test_df['v2a1'] = test_df['v2a1'].fillna(0)


# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train,y_train)
prediction = model.predict(test_df)
submission1 = pd.read_csv('../input/sample_submission.csv')
submission1['Target'] = prediction
submission1.to_csv('submission1.csv',index=False)


# In[ ]:


model = DecisionTreeClassifier()

param_dist = {"max_depth": [3,5,7,9,10,15,20,None],
              "min_samples_split": [2,5,10,15],
              "min_samples_leaf": [1,3,5]}

Search = RandomizedSearchCV(model, param_distributions=param_dist)

# Fit the model on the training data
Search.fit(X_train, y_train)

# Make predictions on the test data
preds = Search.best_estimator_.predict(test_df)

submission2 = pd.read_csv('../input/sample_submission.csv')
submission2['Target'] = prediction
submission2.to_csv('submission2.csv',index=False)

