#!/usr/bin/env python
# coding: utf-8

# # Titanic survival prediction V1
# 
# For my first version of this classification task, I will an ensemble method, the Random Forest classifier. 

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Importing the Data
# 
# First let's import the data.

# In[ ]:


data = pd.read_csv("/kaggle/input/titanic/train.csv")
data = data.set_index("PassengerId")


# ## Manipulate the data to get the correct format
# 
# First, let's extract Y, the labels (survived or not). Then, we will reformat the data frame to only keep relevant features.

# In[ ]:


Y = data.Survived
data = data.drop(columns= ['Survived','Name', 'Ticket', 'Cabin'])


# In[ ]:


data.head()


# In[ ]:


data.Pclass = data.Pclass.astype(str)
features = [ 'Pclass','Sex', 'Embarked']

dums = pd.get_dummies(data[features])

data = data.drop(columns=features)


# ## Merge the dummies to the dataframe 
# 
# Now that we have created the categorical dummies for thos necessary, we can now merge back the two dataframe.

# In[ ]:


data = pd.concat([data, dums], axis = 1)
data.head()


# ## Address the NaN values 
# 
# We need to address the missing values of our dataset. For this part here, as any passenger should have a value in any of the categories, I decided to replace the Nan by the mean.

# In[ ]:


data.fillna(data.mean(), inplace=True)
data.isnull().values.any() #verify there's no null values


# ## Create a model
# 
# Now that our data is preprocessed, we are ready to build our first model.
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

model_features=['Age','SibSp','Parch','Fare','Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']

X = data[model_features]

X_train, X_dev, y_train, y_dev = train_test_split(X,Y, test_size = 0.15)
 
train_accs=[]
dev_accs = []
for nb_est in range(1,300):
    model = RandomForestClassifier(n_estimators=nb_est, max_depth=5, random_state=1)
    model.fit(X_train,y_train)
    
    pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, pred_train)
    
    pred_dev = model.predict(X_dev)
    acc_dev = accuracy_score(y_dev, pred_dev)
    
    train_accs.append(acc_train)
    dev_accs.append(acc_dev)


# ## Analyse the accuracy of the model
# 
# We can see that the model accuracy stops to increase at around 35 trees. Thus we will set our model to that value for the final predictions.

# In[ ]:



plt.plot(train_accs)
plt.plot(dev_accs)

plt.legend(['Training accuracy', 'Dev-set accuracy'], loc='upper left')
plt.show()


# ## Final model and predictions generation
# 
# First, we need to preprocess the test data the same way we preprocessed the training data. Thus, we will reuse the code we previouly wrote.

# In[ ]:


model_final = RandomForestClassifier(n_estimators=145, max_depth=5, random_state=1)
model_final.fit(X,Y)


# In[ ]:


test = pd.read_csv("/kaggle/input/titanic/test.csv")
test = test.set_index("PassengerId")
test = test.drop(columns= ['Name', 'Ticket', 'Cabin'])

test.Pclass = test.Pclass.astype(str)
dums2 = pd.get_dummies(test[features])
test = test.drop(columns=features)

test = pd.concat([test, dums2], axis = 1)

test.fillna(test.mean(), inplace=True)

test.head()


# In[ ]:


test = test[model_features]
pred = model_final.predict(test)

output = pd.DataFrame({'PassengerId':test.index, 'Survived': pred})
output.to_csv('my_submission.csv', index=False)

