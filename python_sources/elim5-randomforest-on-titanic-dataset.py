#!/usr/bin/env python
# coding: utf-8

# ![](https://images.immediate.co.uk/production/volatile/sites/7/2018/01/TIT011DJ_0-345b632.jpg?quality=90&resize=620,413)

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv("../input/test.csv")


# ##### Since RandomForest cannot work with text data , we'll take a look at all the columns, and see which ones are text and which ones** are numeric

# In[ ]:


print(trainData.dtypes.sort_values())
print(testData.dtypes.sort_values())


# ##### we also have to take care of null values also called NaN values so that R.F algorith can process the data properly****

# In[ ]:


trainData.isnull().sum()[trainData.isnull().sum()>0]


# ##### the first half "trainData.isnull().sum()" calculates all total occurance of NaN data type in each of the colomns of trainData.the secong half "[trainData.isnull().sum()>0]" acts as like a condition on firt hlaf and only produces sum of NaNs where no of Nan values is greater than 0

# In[ ]:


testData.isnull().sum()[testData.isnull().sum()>0]


# ##### now to work with the dataset we'll either have to drop the colomns with null values or sustitute suitable values in their positions to determine wether a colomn is important enough to not drop or use in R.F algo is called feature importance .Here i'll fill the nan values with mean value of that particular column .

# In[ ]:


trainData.Age=trainData.Age.fillna(trainData.Age.mean())
testData.Age=testData.Age.fillna(trainData.Age.mean())

trainData.Fare=trainData.Fare.fillna(trainData.Fare.mean())
testData.Fare=testData.Fare.fillna(trainData.Fare.mean())


trainData.Embarked=trainData.Embarked.fillna(trainData.Embarked.mode()[0])
testData.Embarked=testData.Embarked.fillna(trainData.Embarked.mode()[0])


# ##### Taking a look at our tabel 

# In[ ]:


trainData.head()
testData.head()


# ##### droppig coloums we wont be using

# In[ ]:


trainData.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
testData.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)


# In[ ]:


testData.head()


# ## Converting categorical variables into numerical

# ##### a categorical variable is a variable that can take on one of a limited, and usually fixed number of possible values, e.g sex has only male and female , In ML we make use of various techniques like one-hot encoding etc to change catagorical columns into numerical columns . i've used the get_dummies function of pandas to change all the catagorical variables at once . you can read more about catagorical variables and how to process them here , the given link only talks about one-hot encoding but you'll get the gist of the idea.

# ###### Note ot self: below colomn added after error in fitting Randomforest due to mismatch of newly added colomns from caterorical colomns . Prefered way to do "pd.concat"

# In[ ]:


combined=pd.concat([trainData, testData], sort=False)
print(combined.dtypes.sort_values())


# reason for creating a combined tabel : due to error at time of fitting the Model , solution of which was found [Here](https://stackoverflow.com/questions/44026832/valueerror-number-of-features-of-the-model-must-match-the-input)
# 

# In[ ]:


length = trainData.shape[0]
combined=pd.concat([trainData, testData], sort=False)
combined=pd.get_dummies(combined)
trainData=combined[:length]
testData=combined[length:]

trainData.Survived=trainData.Survived.astype('int')


# ##### spliting the dataframe into dependant variables (y) and independant variabeles (x)

# In[ ]:


x=trainData.drop("Survived",axis=1)
y=trainData['Survived']
xtest=testData.drop("Survived",axis=1)


# ## Creating a RandomForest Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


# In[ ]:


RF = RandomForestClassifier(random_state=1)
results = cross_val_score(RF,x,y,scoring='accuracy',cv=5)
print(results)
np.mean(results)


# In[ ]:


RF.fit(x, y)


# In[ ]:


print(RF)


# ##### Storing the result in a csv file

# In[ ]:


predictions=RF.predict(xtest)
column_name = pd.read_csv('../input/test.csv')
output=pd.DataFrame({'PassengerId':column_name['PassengerId'],'Survived':predictions})
output.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




