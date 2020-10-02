#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


train = pd.read_csv("../input/train.csv")
train.head(5) 


# The first step is to make data cleansing. Check for the null values and clean the data.

# In[43]:


train.isnull().sum()


# We see that, there are two colums which has null values. Cabin is a column where in we don't require for data visualization, so it's betterto drop the column."

# In[44]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis');


# In[45]:


train.drop('Cabin',axis=1,inplace=True)
train['Age'].fillna((train['Age'].median()), inplace=True)


# In[46]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis');


# Now the data is clean without NaN values, lets do some visualization and try to find the insights.

# In[47]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r');


# In[48]:


sns.set_style('darkgrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='ocean');


# From the above graph, it stats that the most of the males are not survided.

# In[49]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='winter');


# This graph clearly indicaes that the 3rd class passengers are not surivided much.

# In[50]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=20);


# We can observe that most of the passengers are in the age group of 20 to 35.

# In[51]:


sns.countplot(x='SibSp',data=train,palette='ocean');


# 

# SibSp stands for number of siblings and spouse travelled along with the passenger. Most of the passengers are travelled in single without siblings and spouse along with them.

# In[52]:


sns.countplot(x='Parch',data=train,palette='ocean');


# In[53]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter');


# Parch stands for number of Parent-children combination travelled along with the passenger. Even we see that the count is less. now let's look at the data and check for the textual colums and convert them to numerical in order to design a model.

# In[54]:


train.info()


# Now drop unnecessary columns.
# The dataset has PassengerId,Name and Ticket which are really not required. Hence drop the columns. Convert the sex and Embarked columns to numerical respectively.

# In[55]:


# create the dummy variables and drop one column as there is no need of 2 columns in order to differentiate the values.
sex = pd.get_dummies(train['Sex'],drop_first=True)
# similarly for this colimn as well. If there are n dummy columns, consider n-1
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[56]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)


# In[57]:


train.head()


# Now the data is ready! let's build the model.

# ### Train Test Split
# 

# In[58]:


from sklearn.model_selection import train_test_split
X = train.drop("Survived",axis=1)
y = train['Survived']
#X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.20,random_state=5)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=5)


# Let us train the data and predict the scores using logistic regression

# ## Logistic Regression

# In[59]:


from sklearn.linear_model import LogisticRegression
# create an instance
logmodel = LogisticRegression()
# pass the values and build the model
logmodel.fit(X_train,y_train)


# In[60]:


# preditcing the test models
predictions = logmodel.predict(X_test)


#  ## Evaluation :  Logistic Regression

# Let's evaluate using confusion  matrix and F1 score

# In[61]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(confusion_matrix(y_test,predictions))


# In[62]:


print(classification_report(y_test,predictions))


# In[63]:


print(accuracy_score(y_test,predictions)*100)


# ## Evaluation : Decision Tree Classifiction
# 

# In[64]:


from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
print(confusion_matrix(y_test,dt_pred))


# In[65]:


print(classification_report(y_test,dt_pred))


# In[66]:


print(accuracy_score(y_test,dt_pred)*100)


# The accuracy score is comparitively less to the logistic regression and it is still the best model as of now, let us try another model.

# ## Random Forest Classification
# 

# In[67]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)


# In[68]:


rf_pre=rf.predict(X_test)
print(confusion_matrix(y_test,rf_pre))


# In[69]:


print(classification_report(y_test,rf_pre))


# In[70]:


print(accuracy_score(y_test,rf_pre)*100)


# Random forest and logistic regressions are almost equal in predictions, you can use either of them for the model.

# Finally, the model is complete. Now let's take a random data and check the survival rate of the passegers.

# In[71]:


test = pd.read_csv("../input/test.csv")
test.head(5) 


# Observe that there is no " survived " column in the dataset, which has to be predicted now.

# In[72]:


sns.heatmap(test.isnull());


# In[73]:


test.info()


# There are missing values just like the train data which we performed before, perform the same steps and clean the data.

# In[74]:


test.drop('Cabin',axis=1,inplace=True)
test['Age'].fillna((test['Age'].median()), inplace=True)
test['Fare'].fillna((test['Fare'].median()), inplace=True)
sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test= pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex_test,embark_test],axis=1)
test.head()


# Now apply any one of the regression technique and check for the results.

# In[75]:


prediction = logmodel.predict(test)
prediction


# The prediction values are stored in numpy array, in order to add this column to the test dataset, convert to dataframe.

# In[76]:


test_pred = pd.DataFrame(prediction, columns= ['Survived'])


# In[77]:


Survived_dataset = pd.concat([test, test_pred], axis=1, join='inner')


# In[78]:


Survived_dataset.head()


# In[83]:


dataset = Survived_dataset[['PassengerId','Survived']]
dataset.head(10)


# From the above dataset, we have successfully predicted the passengers who are survived Vs not survived.

# In[84]:


data_to_submit = pd.DataFrame(Survived_dataset[['PassengerId','Survived']])


# In[85]:


data_to_submit.to_csv('csv_to_submit.csv', index = False)


# In[ ]:




