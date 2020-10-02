#!/usr/bin/env python
# coding: utf-8

# ## Classification Algorithm
# In this section we are going to learn about classification algorithm.
# 
# Problem Statement: The problem statement is to predict the list of survivors in the disaster situation that happened more than 100 years ago when Titanic sank to the bottom of the ocean.
# 
# 

# In[314]:


#Libraries
import numpy as np
import pandas as pd


# In[315]:


df_training = pd.read_csv('../input/train.csv')

df_training.shape


# So data set has total 891 rows. that is it has information about 891 people wether they survived or not. It has 12 rows, lets see what they are and what is there data type.

# In[316]:


df_training.dtypes


# ## Feature Description
# We have following details about the passengers of Titanic. Lets apply basic analysis on them. we will do an EDA before accepting or rejecting a column as predictor for survival value.
# - Passenger Id - uniquely identifying each passenger. This cannot help us in identifying the survival of a passenger.
# - Survived - Thie is the value that tells wether a passenger has survived or not with values 0 and 1. this is a categorical value and data type needs to be changed in the dataframe.
# - Pclass - This tells the class of passenger. When the ship was sinking, most of the survivors were chosen from high class. hence the Pclass will help us identify the survivors.
# - Name - Just like passenger Id, Name will be different for each row and so not useful in predicting the survival of passengers.
# - Sex - Female passengers were give preference over male passengers to go in life boats. and so Sex is a good predictor for the survival.
# - Age - Childrens were preferred during rescue operation on Titanic. Hence we will have to create category of age for deciding the survivors.
# - SibSp - It gives total number of siblings and spouse for that particular passenger.
# - Parch - Number of parents or children aboard the ship.
# - Ticket - Ticket number.
# - Fare - Passenger Fare.
# - Cabin - Cabin Number. A passenger can have a cabin or may not have the cabin. we can create a categorical variable which stores if a passenger has cabin or not.
# - Embarked - Port at which passenger embarked their journey.

# In[317]:


# Lets remove passenger id out of the training set and store it in another variable
training_passengerId = df_training.PassengerId

df_training.drop(columns=['PassengerId'],inplace=True)

#dropping Name and Ticket and fare as well out of the data
df_training.drop(columns=['Name','Ticket','Fare'],inplace=True)
df_training.head()


# In[318]:


#Lets annalyze the values of remaining data

print('Survived value counts: ')
print(df_training.Survived.value_counts())

print('Count by class: ')
print(df_training.Pclass.value_counts())

print('count by sex: ')
print(df_training.Sex.value_counts())

print('Cabin or without cabin count')
print('Without cabin', df_training.Cabin.isnull().sum())
print('With cabin', df_training.shape[0] - df_training.Cabin.isnull().sum())

print('Count by Journey Embarking point:')
print(df_training.Embarked.value_counts())


# Lets change these values to category type
# 

# In[319]:


#creating category types
df_training.Survived=df_training.Survived.astype('category')
df_training.Pclass=df_training.Pclass.astype('category')
df_training.Sex=df_training.Sex.astype('category')
df_training.Embarked = df_training.Embarked.astype('category')

# lets do feature engineering using cabin. if a passenger has cabin and if a passenger doesnot have a cabin.
df_training['cabinAllocated'] = df_training.Cabin.apply(lambda x: 0 if type(x)==float else 1)
df_training['cabinAllocated'] = df_training['cabinAllocated'].astype('category')


# In[320]:


df_training.dtypes


# In[321]:


# Lets drop Cabin first
df_training.drop(columns=['Cabin'],inplace=True)


# Now lets draw some garphs to understand age column's behaviour againsth the count.

# In[322]:


print("Min Age : {}, Max age : {}".format(df_training.Age.min(),df_training.Age.max()))


# In[323]:


df_training.Age.isnull().sum()


# As there are 177 records without age, we can either ignore them or randomly put some values. Age played an important role in deciding the survivals. Lets put some random numbers in place of null values.

# In[324]:


random_list = np.random.randint(df_training.Age.mean() - df_training.Age.std(), 
                                         df_training.Age.mean() + df_training.Age.std(), 
                                         size=df_training.Age.isnull().sum())
df_training['Age'][np.isnan(df_training['Age'])] = random_list
df_training['Age'] = df_training['Age'].astype(int)


# In[325]:


# Lets divide age in 5 bins

df_training['AgeGroup'] = pd.cut(df_training.Age,5,labels=[1,2,3,4,5])


# In[326]:


#As we have categorized age into AgeGroup, lets remove Age
df_training.drop(columns=['Age'],inplace=True)


# Lets get complete family size from Parch and SibSp columns by adding them.

# In[327]:


#Adding 1 to indicate the person in that row
df_training['family'] = df_training.Parch+df_training.SibSp+1


# In[328]:


df_training.drop(columns=['SibSp','Parch'],inplace=True)
df_training.head()


# In[329]:


df_training['Sex'].value_counts()


# In[330]:


df_training['category_sex'] = df_training['Sex'].apply(lambda x: 1 if x=='male'  else 0)


# In[331]:


df_training.drop(columns=['Sex'],inplace=True)


# In[332]:


df_training.Embarked.value_counts()


# In[333]:


df_training.Embarked = df_training.Embarked.fillna('S')
df_training.Embarked = df_training.Embarked.map({'S':1,'C':2,'Q':3}).astype('int')


# In[334]:


df_training.Embarked.value_counts()


# In[ ]:





# In[335]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(df_training.iloc[:,1:],df_training.iloc[:,0],test_size=0.2,random_state=0)


# In[336]:


from sklearn.ensemble import RandomForestClassifier

randomForest = RandomForestClassifier(n_estimators=100)

randomForest.fit(train_x,train_y)


# In[337]:


y_hat = randomForest.predict(test_x)


# In[338]:


from sklearn.metrics import accuracy_score
accuracy_score(test_y,y_hat)


# Lets use complete set to create the model

# In[339]:


randomForest.fit(df_training.iloc[:,1:],df_training.iloc[:,0])


# Now lets import the test file and create the output based on the test file. Before that, we will have to make all the manipulations on test file that we did on training file.

# In[340]:


df_testing = pd.read_csv('../input/test.csv')

# Lets remove passenger id out of the training set and store it in another variable
testing_passengerId = df_testing.PassengerId

df_testing.drop(columns=['PassengerId'],inplace=True)

#dropping Name and Ticket and fare as well out of the data
df_testing.drop(columns=['Name','Ticket','Fare'],inplace=True)
df_testing.head()

#creating category types
df_testing.Pclass=df_testing.Pclass.astype('category')
df_testing.Sex=df_testing.Sex.astype('category')
df_testing.Embarked = df_testing.Embarked.astype('category')

# lets do feature engineering using cabin. if a passenger has cabin and if a passenger doesnot have a cabin.
df_testing['cabinAllocated'] = df_testing.Cabin.apply(lambda x: 0 if type(x)==float else 1)
df_testing['cabinAllocated'] = df_testing['cabinAllocated'].astype('category')

# Lets drop Cabin first
df_testing.drop(columns=['Cabin'],inplace=True)

random_list_test = np.random.randint(df_testing.Age.mean() - df_testing.Age.std(), 
                                         df_testing.Age.mean() + df_testing.Age.std(), 
                                         size=df_testing.Age.isnull().sum())
df_testing['Age'][np.isnan(df_testing['Age'])] = random_list_test
df_testing['Age'] = df_testing['Age'].astype(int)

# Lets divide age in 5 bins

df_testing['AgeGroup'] = pd.cut(df_testing.Age,5,labels=[1,2,3,4,5])


#As we have categorized age into AgeGroup, lets remove Age
df_testing.drop(columns=['Age'],inplace=True)

#Adding 1 to indicate the person in that row
df_testing['family'] = df_testing.Parch+df_testing.SibSp+1

df_testing.drop(columns=['SibSp','Parch'],inplace=True)

df_testing['category_sex'] = df_testing['Sex'].apply(lambda x: 1 if x=='male'  else 0)
df_testing.drop(columns=['Sex'],inplace=True)

df_testing.Embarked = df_testing.Embarked.fillna('S')
df_testing.Embarked = df_testing.Embarked.map({'S':1,'C':2,'Q':3}).astype('int')


# In[341]:


submission_data = pd.DataFrame({'PassengerId':testing_passengerId, 'Survived':randomForest.predict(df_testing)})

submission_data.to_csv("Submission_Data.csv",index=False)


# In[ ]:




