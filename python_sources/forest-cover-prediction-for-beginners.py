#!/usr/bin/env python
# coding: utf-8

# ## Thank You for opening this notebook!!!
# This is my second attempt to help beginners start with ML and Data Science. I hope you'll like it.

# ## Importing libraries

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plot


# ## Loading Datasets 

# In[4]:


testx = pd.read_csv("../input/test.csv")
testx.head()


# In[5]:


df = pd.read_csv("../input/train.csv")
print(df.shape)
df.head()
df.corr()
df.columns.get_loc("Wilderness_Area1")


# In[6]:


df.head()


# In[7]:


train_y = df.pop('Cover_Type')
train_y.head()


# In[8]:


print(type(train_y))


# In[9]:


df.isnull().sum()


# ## Combining Binary features wilderness & soil type into single column 

# In[10]:


df["Wilderness"] = df.iloc[:,11:15].idxmax(axis =1)
df["Wilderness"]


# In[11]:


df["Wilderness"] = df["Wilderness"].apply(lambda x: x[15])
df["Wilderness"]


# In[12]:


df["Soil_Type"]= df.iloc[:,15:55].idxmax(axis =1)
df.head()


# In[13]:


df["Soil_Type"] = df["Soil_Type"].apply(lambda x:x[9:])
df["Soil_Type"]


# In[14]:


df.tail()


# In[15]:


df.shape


# In[16]:


def remove_feature(df):
    dfg = df.columns[15:55]
    dfg2 = df.columns[11:15]
    df.drop(dfg ,axis =1, inplace= True )
    df.drop(dfg2 , axis = 1, inplace= True)
    df.pop("Id")
    df.pop("Horizontal_Distance_To_Roadways")
remove_feature(df)
print(df.shape)
print(testx.shape)


# In[17]:


df.head()


# ## Combining similarly for test set 

# In[18]:


testx["Wilderness"] = testx.iloc[:,11:15].idxmax(axis =1)
testx["Wilderness"] = testx["Wilderness"].apply(lambda x: x[15])
testx["Wilderness"]


# In[19]:


testx.shape


# In[20]:


testx["Soil_Type"]= testx.iloc[:,15:55].idxmax(axis =1)
testx["Soil_Type"] = testx["Soil_Type"].apply(lambda x:x[9:])
testx.head()


# In[21]:


dfg = testx.columns[15:55]
dfg2 = testx.columns[11:15]
testx.drop(dfg ,axis =1, inplace= True )
testx.drop(dfg2 , axis = 1, inplace= True)
testx.pop("Id")
testx.pop("Horizontal_Distance_To_Roadways")


# In[22]:


testx.head()


# In[23]:


print(testx.shape)
print(df.shape)


# In[24]:


df.head()


# In[25]:


train_y.shape


# # ML Methods 

# ## Spliting Data into x_train , y_train , x_test , y_test

# In[26]:


from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(df ,train_y ,test_size = .34, random_state = 234)


# # Defining Classifiers 

# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[28]:


clf = RandomForestClassifier()


# ## Finding proper parameters from GridSearchCV

# In[29]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer , accuracy_score


# In[30]:


parameters = {
    'n_estimators': [4, 6, 9], 
    'max_features': ['log2', 'sqrt','auto'], 
    'criterion': ['entropy', 'gini'],
    'max_depth': [2, 3, 5, 10], 
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1,5,8]
}

acc_scorer =make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf , parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(x_train,y_train)

clf = grid_obj.best_estimator_
clf.fit(x_train, y_train)
predict = clf.predict(testx)
print(predict)


# ## k Fold Validation 

# In[31]:


from sklearn.cross_validation import KFold 


# In[32]:


def validation_ml(clf):
    
    kf = KFold(df.shape[0] , 10 )
    outcome =[]
    for train_i, test_i in kf :
        train_x, test_x = df.values[train_i] , df.values[test_i]
        trainy, test_y = train_y.values[train_i] , train_y.values[test_i]
        
        clf.fit(train_x,trainy)
        prediction = clf.predict(test_x)
        accuracy =accuracy_score(test_y, prediction)
        outcome.append(accuracy)
        print(accuracy)
    print(" Mean Accuracy = " , np.mean(outcome))
        
        


# In[33]:


validation_ml(clf)


# In[34]:


validation_ml(clf)


# ## Output

# In[35]:


test_original = pd.read_csv("../input/test.csv")
ids = test_original['Id']
output = pd.DataFrame({ 'Id' : ids, 'Cover_Type' : predict })
output.shape


# In[36]:


output.tail()


# In[37]:


columnsTitles=["Id","Cover_Type"]
output=output.reindex(columns=columnsTitles)


# In[38]:


output.tail()


# In[39]:


output.to_csv('Forest-cover-prediction_new.csv', index = False)


# In[ ]:




