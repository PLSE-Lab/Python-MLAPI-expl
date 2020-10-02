#!/usr/bin/env python
# coding: utf-8

# 
# ## INTRODUCTION
# 
# * Hi,I recently started learning Data science and this is my first time in Kaggle and I have tried my best explain what I have done with my notebook, If you find it help full I will be very happy, and if you have any suggestions I will be happy to implement that..... ENJOY THE NOTEBOOK....  

# In[ ]:


## Importing the libraries...

import pandas as pd, numpy as np 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## Read train data
train = pd.read_csv('/kaggle/input/titanic/train.csv')

## Read test data
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


## Preview of train data
train.head()


# In[ ]:


## Preview of text data
test.head()


# ## CHECKING FOR THE NULL VALUES

# In[ ]:


## for train data
train.isnull().sum()


# In[ ]:


##looking at the % of NaN values in train data

print('% of nan values in age in train is',(train['Age'].isnull().sum()/train.shape[0])*100)
print('--------------------------------------------------------------------------------------')
print('% of nan values in Cabin in train is',(train['Cabin'].isnull().sum()/train.shape[0])*100)
print('--------------------------------------------------------------------------------------')
print('% of nan values in Embarked in train is',(train['Embarked'].isnull().sum()/train.shape[0])*100)


# In[ ]:


## for test data
test.isnull().sum()


# In[ ]:


##looking at the % of NaN values in test data

print('% of nan values in age in test is',(test['Age'].isnull().sum()/test.shape[0])*100)
print('--------------------------------------------------------------------------------------')
print('% of nan values in Fare in test is',(test['Fare'].isnull().sum()/test.shape[0])*100)
print('--------------------------------------------------------------------------------------')
print('% of nan values in Cabin in test is',(test['Cabin'].isnull().sum()/test.shape[0])*100)


# ## FILLING THE NAN VALUES

# In[ ]:


## for train dataset..
## to fill the NaN values in the age column in train data first lets see the distribution of the age, 
## that will give us the idea on what value is to be used to fill the NaN values
import seaborn as sn
sn.distplot(train['Age'],bins=16)


# In[ ]:


## as we can see the data is right skeweed so we can use median value of the age to fill teh NaN values...
train['Age'].fillna(train['Age'].median(skipna=True),inplace=True)


# In[ ]:


## checking the count between the embarkments
sn.countplot(train['Embarked'])


# In[ ]:


## for the NaN values in the Embarked column as its a very small value being 0.224% 
## we can full those NaN values using the most repeated value
train['Embarked'].fillna(train['Embarked'].value_counts().idxmax(),inplace=True)


# In[ ]:


## as the NaN values in the column Cabin is more than 40% its better to drop that column..
train.drop('Cabin',axis=1,inplace=True)


# In[ ]:


##for test data
## to fill the NaN values in the age column in train data first lets see the distribution of the age, 
## that will give us the idea on what value is to be used to fill the NaN values
sn.distplot(test['Age'],bins=16)


# In[ ]:


## as we can see the data is right skeweed so we can use median value of the age to fill teh NaN values...
test['Age'].fillna(test['Age'].median(skipna=True),inplace=True)


# In[ ]:


## looking for the distribution for Fare column..
sn.distplot(test['Fare'],bins=100)


# In[ ]:


## as we can see the data is right skeweed so we can use median value of the Fare to fill teh NaN values...
test['Fare'].fillna(test['Fare'].median(skipna=True),inplace=True)


# In[ ]:


## as the NaN values in the column Cabin is more than 40% its better to drop that column..
test.drop('Cabin',axis=1,inplace=True)


# ## REMOVING THE OUTLIERS

# In[ ]:


## looking at the stats..
train.describe()


# In[ ]:


## we will make use of Z scores to remove the outliers..
from scipy import stats
numtrain = train._get_numeric_data()


# In[ ]:


z = np.abs(stats.zscore(numtrain))
print(z)


# In[ ]:


## threshold = 3
print(np.where(z > 3))


# In[ ]:


train = train[(z < 3).all(axis=1)]


# ## TIME FOR SOME EDA

# In[ ]:


## looking at all the columns in the train data
train.columns


# In[ ]:


## using pie chart for better picture of survival rate..
train['Survived'].value_counts().plot(kind='pie')


# In[ ]:


## cross tab is a powerfull tool which helps us to understand the combo of columns deeper...
## we are seeing the survival rate based on gender..
pd.crosstab(train['Sex'],train['Survived'],margins=True).style.background_gradient(cmap='PuBu')


# In[ ]:


## looking at the countplot to better understanding..
sn.countplot('Sex',hue='Survived',data=train)
plt.show()


# In[ ]:


## we can see that the female has high survival rate that male..


# In[ ]:


## now lets see the survival rate based on Pclass..
sn.countplot('Pclass',hue='Survived',data=train)
plt.show()


# In[ ]:


## using crosstab beased on gender, Pclass and Survived
pd.crosstab([train['Sex'],train['Survived']],train['Pclass'],margins=True).style.background_gradient(cmap='PuBu')


# In[ ]:


## we can see that the Pclass 3 has lot of deaths and least in Pclass 1....


# In[ ]:


## now lets see the survival rate based on Embarked..
sn.countplot('Embarked',hue='Survived',data=train)
plt.show()


# In[ ]:


## factor plot is also one of the coolset plots to use..
sn.factorplot('Pclass','Survived',hue='Sex',data=train)
plt.show()


# In[ ]:


## we can see that female in Pclass 1 has the high chance of being alive than the women in Pclass 3..


# In[ ]:


## when we go through the data discription we can see that column SibSp is # of siblings / spouses aboard the Titanic and
## column Parch is # of parents / children aboard the Titanic so we can make use of them and make a column called 'family'....
train['TravelAlone']=np.where((train["SibSp"]+train["Parch"])>0, 0, 1)
train.drop('SibSp', axis=1, inplace=True)
train.drop('Parch', axis=1, inplace=True)


# In[ ]:


## doing the same thing on test data..
test['TravelAlone']=np.where((test["SibSp"]+test["Parch"])>0, 0, 1)
test.drop('SibSp', axis=1, inplace=True)
test.drop('Parch', axis=1, inplace=True)


# ## GETTING THE DATA TO FIT THE MODEL 

# In[ ]:


## dropping the unnecessary columns in train data...
train.drop('PassengerId',axis=1,inplace=True)
train.drop('Name',axis=1,inplace=True)
train.drop('Ticket',axis=1,inplace=True)


# In[ ]:


## dropping the unnecessary columns in test data....
test.drop('Name',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)
## we might get a question here that why is the PassengerId column is not dropped from test data..
## that is because we will need that for the submission of the result, so we are kiing that for now...


# In[ ]:


## checking the categorical columns..
train.info()


# In[ ]:


## so we must convert those categorical columns into numbers so.. its time for....


# ## ONE HOT ENCODING

# In[ ]:


## just a line of code will do one hot encoding for us..
final_train = pd.get_dummies(train, columns=["Pclass","Embarked","Sex"])
final_train


# In[ ]:


## one hot encoding for test data..
final_test = pd.get_dummies(test, columns=["Pclass","Embarked","Sex"])
final_test


# ## TIME TO FIT THE MODEL (LOGISTIC REGRESSION)

# In[ ]:


## setting the X and Y...
X= final_train.drop('Survived',axis=1)
y= final_train['Survived']


# In[ ]:


## importing the train_test_split from sklearn to split the data into train and test..
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=999)


# In[ ]:


## importing the model from sklearn..
from sklearn.linear_model import LogisticRegression


# In[ ]:


## initialization and fitting the model..
logmodel = LogisticRegression(solver = 'lbfgs')
logmodel.fit(X_train,y_train)


# In[ ]:


## checking the scores..
logmodel.score(X_train,y_train)


# In[ ]:


logmodel.score(X_test,y_test)


# In[ ]:


## we dont have the problem of overfitting thats a good sign :)


# In[ ]:


##predicting...
predictions = logmodel.predict(X_test)
predictions


# In[ ]:


## importing the classification_report from sklearn metrics
from sklearn.metrics import classification_report
## classification report..
print(classification_report(y_test,predictions))


# In[ ]:


## our model has f1-score 75% which is not bad, but now with thw help of confusion matric lets see where our model is going wrong..


# In[ ]:


## confusion matrix is the best things to see how my model is predicting 
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[ ]:


## confused from confusion_matrix..? Dont worry we will see one by one..
## it says 89 members are alive and our model predicted they are alive its correct, this is TRUE POSITIVE.
## 17 members are dead and our model predicted as they are alive, this is  FALSE POSITIVE, this is "TYPE ONE" error.
## 13 members are alive and our model predictrd as dead, thats sad, this is FALSE NEGATIVE, this is "TYPR TWO" error this is more dangerous than type one error.
## 45 members are dead and our model predicted as dead, this is TRUE NEGATIVE
## classification report will make use of all these TRUE POSITIVE, FALSE POSITIVE etc...


# In[ ]:


## calculating the mean squared error
from sklearn.metrics import mean_squared_error as mse
mse(y_test,predictions)


# In[ ]:


## closer the mse value to 0 better the model


# ## PREDICTING FOR THE TEST DATA

# In[ ]:


## clearing a duplicate of test data without the PassengerId, because we did not keep that in the train model for prediction..
## if we add this column or feature here and not adding there will throw an error so lets drop it..
final_test1 = final_test.drop('PassengerId',axis=1)


# In[ ]:


## predicting the values for test data
final_test1['Survived'] = logmodel.predict(final_test1)


# ## CREATING THE SUBMISSION FILE

# In[ ]:


## taking the PassengerId from final_test..
PassengerId = final_test['PassengerId']


# In[ ]:


## taking the Survived from final_test1..
Survived = final_test1['Survived'] 


# In[ ]:


## making it as a DataFrame using pandas
Submission = pd.DataFrame([PassengerId,Survived]).T


# In[ ]:


## exporting it as csv file for the submission..
Submission.to_csv('Submission.csv',index=False)


# ## THANK YOU SO MUCH..

# In[ ]:




