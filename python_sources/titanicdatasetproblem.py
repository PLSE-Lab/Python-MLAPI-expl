#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


train['Embarked'].value_counts()


# As we can see that **Cabin** has 687 missing values out of a total of 891 values. That is around 77% of the data missing. So, we can not conclude anything about the missing data. Hence, it'd be better to avoid the Cabin column. 
# 
# The **Embarked** column has 2 missing values. From its analysis, we can see that 'S' has the highest frequency. We also find that if we make the missing values equal to either of 'S','C' or 'Q', it won't make a huge difference to our analysis and model. So,I decided to replace the null values with the mode, which is 'S'.
# 
# Next, we have missing values in the **Age** column. From the 5-Number Summary of the Age column, we get to know that Age has a normal distribution because the mean = 29.699 is very close to the median = 28.0. We have 177 missing values in Age. Age can gave a vital role in determining whether a person survived or not. We have two options to replace the missing values. One, we can replace them with the mean of the column or two, we can replace them with the median of the column. As it seems, we can replace them with either mean or median and it won't make a huge difference. I decided to replace the null values with the median, as median is much more robust to outliers than mean.

# In[ ]:


train['Embarked'].fillna('S',inplace = True)
test['Embarked'].fillna('S',inplace = True)
train['Age'].fillna(train['Age'].mean(),inplace = True)
test['Age'].fillna(test['Age'].mean(),inplace = True)
print(train.head())
print(train.info())
print(train['Embarked'].value_counts)
print(train.isnull().sum())


# In[ ]:


a = {'male' : 0 , 'female' : 1}
e = {'S' : 0 , 'Q' : 1 , 'C' : 1 }
train.replace({'Sex' : a , 'Embarked' : e},inplace=True)
#test.replace({'Sex' : a , 'Embarked' : e},inplace=True)
print(test.head())
train.head()


# Now we will move forward with some visualization and data exploration. We have to visualize the data to find the correlation of the different columns with each other and specifically the 'Survive' column.

# In[ ]:


import seaborn as sns
t = train.drop(['PassengerId','Ticket','Cabin','Name'],axis = 1)
c = abs(t.corr())
sns.heatmap(c,annot = True)


# In[ ]:


import seaborn as sns
t = train.drop(['PassengerId','Ticket','Cabin','Name'],axis = 1)
c = (t.corr())
y = train['Survived']
sns.heatmap(c,annot = True)


# In[ ]:


sns.countplot(x = 'Survived', data = train)


# From the above plot of the column 'Survived', we can see tha most people(>400) did not survive. Now, let us have an even cleaner visualization on the basis of the 'Sex' column

# In[ ]:


sns.countplot(x = 'Survived', hue = 'Sex', data = train)


# From the above plot we can infer that of the people who died,most were 'male'. For those who survived, 'female' were more than 'male' class. 

# In[ ]:


sns.countplot(x = 'Survived', hue = 'Pclass', data = train)


# The Pclass column denotes the class of people aboard the Titanic from a monetary point of view. We can see that of the people who died, most were from class 3. Similarly, of those who survived, most were from class 1. Hence, there seems to be some correlation of Pclass with Survived. This justifies the value of correlation(=-0.34) which we got from the heatmap. 

# In[ ]:


train['Fare'].hist(bins = 50)


# In[ ]:


sns.countplot(x = 'Survived', hue = 'SibSp', data = train)


# In[ ]:


sns.countplot(x = 'Survived', hue = 'Parch', data = train)


# From the two plots above, we can conclude that there is no clear correlation between 'SibSp' and 'Survived' and between 'Parch' and 'Survived'. Now, both 'SibSp' and 'Parch' are blurred indicators of the Age of a certain person. Hence, we can conclude the following - 
# * Either of 'SibSp' and 'Parch' or 'Age' ,must be present to make accurate predictions.
# * However, if we have all the three columns, we might get a slightly greater accuracy as we have more features then. The increase in accuracy, intuitvely, won't be huge just because there is no strong correlation between either of the three columns and 'Survived'
# * Moreover, if we use all the features, there remains a possibility of getting an overfitted model. 

# In[ ]:


#Import models from scikit learn module: 
from sklearn.linear_model import LogisticRegression   
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics


# In[ ]:


def classification_model(model, data, predictors, outcome):  
    #Fit the model:  
    model.fit(data[predictors],data[outcome])    
    #Make predictions on training set:  
    predictions = model.predict(data[predictors])    
    #Print accuracy  
    accuracy = metrics.accuracy_score(predictions,data[outcome])  
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    #Perform k-fold cross-validation with 5 folds  
    kf = KFold(5,shuffle=True)  
    error = []  
    for train, test in kf.split(data):
        # Filter training data    
        train_predictors = (data[predictors].iloc[train,:])        
        # The target we're using to train the algorithm.    
        train_target = data[outcome].iloc[train]        
        # Training the algorithm using the predictors and target.    
        model.fit(train_predictors, train_target)
        #Record error from each cross-validation run    
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
     
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))) 
    # %s is placeholder for data from format, next % is used to conert it into percentage
    #.3% is no. of decimals
    return model


# output = 'Survived'
# model = RandomForestClassifier()
# predict = ['Pclass','Sex', 'Age','SibSp','Parch','Fare']
# m = classification_model(model,train,predict,output)
# m.predict(t[predict])
# #len(m.predict(t[predict]))

# output = 'Survived'
# model = RandomForestClassifier()
# predict = ['Pclass','Sex', 'Age','Fare']
# classification_model(model,train,predict,output)
# m = classification_model(model,train,predict,output)
# m.predict(t[predict])

# output = 'Survived'
# model = RandomForestClassifier()
# predict = ['Pclass','Sex', 'Fare']
# classification_model(model,train,predict,output)
# m = classification_model(model,train,predict,output)
# m.predict(t[predict])

# In[ ]:


a = {'male' : 0 , 'female' : 1}
e = {'S' : 0 , 'Q' : 1 , 'C' : 1 }
test.replace({'Sex' : a , 'Embarked' : e},inplace=True)
test.head()


# In[ ]:



test['Age'].fillna(test['Age'].median(),inplace = True)
test['Fare'].fillna(test['Fare'].median(),inplace = True)
t = test.drop(['PassengerId','Ticket','Cabin','Name'],axis = 1)
t


# In[ ]:


output = 'Survived'
model = RandomForestClassifier()
predict = ['Sex','Parch','SibSp','Fare','Age','Embarked']
classification_model(model,train,predict,output)
m = classification_model(model,train,predict,output)
a = m.predict(t[predict])
a
#'Age','Parch','SibSp',


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': a})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:


test.info()


# In[ ]:


test.describe()


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[ ]:



output = 'Survived'
model = RandomForestClassifier()
predict = ['Sex','Parch','SibSp','Fare','Age','Embarked']
param =  {  'n_estimators': [200, 500],    'max_features': ['auto', 'sqrt', 'log2'],    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
mod  = GridSearchCV(estimator=model, param_grid=param, cv= 5)
mod.fit(train[predict], train[output])


# In[ ]:


mod.best_params_


# In[ ]:


output = 'Survived'
model = RandomForestClassifier(criterion= 'entropy', max_depth= 5,max_features= 'auto',n_estimators= 500)
predict = ['Sex','Parch','SibSp','Fare','Age','Embarked','Pclass']
classification_model(model,train,predict,output)
m = classification_model(model,train,predict,output)
a = m.predict(t[predict])
a


# In[ ]:




