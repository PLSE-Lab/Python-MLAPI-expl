#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Unzipping and reading from data files

import zipfile

zf = zipfile.ZipFile('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip') 
train_users = pd.read_csv(zf.open('train_users_2.csv'))
zf = zipfile.ZipFile('/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv.zip') 
test_users = pd.read_csv(zf.open('test_users.csv'))
test_users.head()


# # Data Exploration
# In the next few cells I am going to explore the data, both train_users and test_useres, to get some insights from it and discover the outliers and possible errors.

# In[ ]:


print("We have", train_users.shape[0], "users in the training set and", 
      test_users.shape[0], "in the test set.")
print("In total we have", train_users.shape[0] + test_users.shape[0], "users.")


# In[ ]:


# Merge train and test users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True, sort=True)


users.head()


# In[ ]:


users.describe()


# The maximum value of age is 2014 which means there is some errors in the age column, so let's see the number and the nature of the records where this value exists to determine whether or not dropping these records would significantly affect the dataset.
# We can also notice that the count of values in age column is much less than number of records which means there is a lot of missing data in this field (will be investigated later).

# In[ ]:



train_users.loc[train_users['age'] == 2014]


# Looking at the country_destination we can be sure that there are different values associated with this age, which makes dropping these 710 records out of over 213000 records a quite suitable solution that probably wouldn't ommit valuable information.
# However, noticing that the only field that seems to be consistent with records of age=2014 is that signup_app=web, I'll investigate more.
# 

# In[ ]:


web_2014 = users.loc[users['age'] == 2014, 'signup_app'].value_counts() 
print (web_2014)


# In[ ]:


print(users.signup_app.value_counts())


# As you can see, the web value has nothing to do with the age=2014, it just happens to be the most common value of signup_app.
# Let's see now other inconsistent values of age, considering that the longest confirmed human lifespan record=122.
# 

# In[ ]:


np.unique(users[users.age > 122]['age'])


# In[ ]:


print(sum(users.age > 122))
print(sum(users.age < 18))


#  For now, we can set an acceptance range and replace those out of it with NaN.

# In[ ]:


users.loc[users.age > 95, 'age'].count()


# In[ ]:


users.loc[users.age < 13, 'age'].count()


# In[ ]:


users.loc[users.age > 95, 'age'] = np.nan
users.loc[users.age < 13, 'age'] = np.nan


# In[ ]:


users.age.value_counts()


# Let's see linear correlation between the input features

# In[ ]:


users.corr()


# Now let's see non-linear correlation

# In[ ]:


import seaborn as sns
sns.pairplot(train_users)


# We see a very low correlation which makes sense as there is no dependency between these three features.

# # Data Cleaning

# 1) Handeling missing data

# In[ ]:


users["gender"]=users["gender"].fillna("-unknown-" )
users.head()


# In[ ]:


#How much data is missing from the dataset (apart from destination country)
users_nan = (users.isnull().sum() / users.shape[0]) * 100
users_nan[users_nan > 0].drop('country_destination')


# We have quite a lot of *NaN* in the `age` and `gender` wich will yield in lesser performance of the classifiers we will build. The feature `date_first_booking` has a 67.7% of NaN values probably because this feature is not present at the tests users, and therefore, we won't need it at the *modeling* part. Let's see the NaN values of `date_first_booking` in train_data:

# In[ ]:


train_users.date_first_booking.isnull().sum() / train_users.shape[0] * 100


# In[ ]:


train_users.country_destination.isnull().sum() / train_users.shape[0] * 100


# Since we will not need this feature in the modeling we don't need to handle its missing data.
# The other feature with a high rate of *NaN* was `gender`. Let's see:

# In[ ]:


users.gender.value_counts(dropna=False)


# In[ ]:


users.gender.describe()


# The third feature with a high rate of NaN was age. Let's see:
# 
# 

# In[ ]:


users.age.describe()


# In[ ]:


#For now, let's fill the missing values of age with the median since the mean is highly affectd by extreme values
users["age"]=users["age"].fillna( users["age"].median())

users.head()


# 2) Data Types
# 
# Let's treat each feature as what they are. This means we need to transform into categorical those features that we treat as categories and the same with the dates:
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

for categorical_feature in categorical_features:
    print (categorical_feature,users[categorical_feature].unique())


# As we can see there are some features with large number of categories, whereas there are others with relatively small number of categories.Hence, in the data encoding step we will use one hot encoding with features of small number of categories.

# In[ ]:


users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')


# In[ ]:


users.head()


# # Data Visualization

# First, let's see how the gender porpotion is visualized

# In[ ]:


users.gender.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)
plt.xlabel('Gender')
sns.despine()


# Let's see now if there is any gender preferences when travelling.

# In[ ]:


women = sum(users['gender'] == 'FEMALE')
men = sum(users['gender'] == 'MALE')

female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100
male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100

# Bar width
width = 0.4

male_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Male', rot=0)
female_destinations.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Female', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()


# As we can see, There are no big differences between the 2 main genders.
# 
# Let's see now the relative destination frecuency of the countries.

# In[ ]:


destination_percentage = users.country_destination.value_counts() / users.shape[0] * 100
destination_percentage.plot(kind='bar',color='#FD5C64', rot=0)
# Using seaborn can also be plotted
# sns.countplot(x="country_destination", data=users, order=list(users.country_destination.value_counts().keys()))
plt.xlabel('Destination Country')
plt.ylabel('Percentage')
sns.despine()


# The first thing we can see that if there is a reservation, it's likely to be inside the US. But there is around 45% of people that never did a reservation.
# 
# Let's now see the age relations with the destination.

# In[ ]:


age = 45

younger = sum(users.loc[users['age'] < age, 'country_destination'].value_counts())
older = sum(users.loc[users['age'] > age, 'country_destination'].value_counts())

younger_destinations = users.loc[users['age'] < age, 'country_destination'].value_counts() / younger * 100
older_destinations = users.loc[users['age'] > age, 'country_destination'].value_counts() / older * 100

younger_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='Youngers', rot=0)
older_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='Olders', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()


# We can see that the young people tends to stay in the US, and the older people choose to travel outside the country. Of vourse, there are no big differences between them and we must remember that we do not have the 42% of the ages.
# 
# Now, Let's see the language of users.

# In[ ]:


print((sum(users.language == 'en') / users.shape[0])*100)

En = sum(users.loc[users['language']=="en", 'country_destination'].value_counts());
No_En=sum(users.loc[users['language']!="en", 'country_destination'].value_counts());
En_destinations = users.loc[users['language']=="en" , 'country_destination'].value_counts() / En * 100
No_En_destinations = users.loc[users['language'] !="en", 'country_destination'].value_counts() / No_En * 100

younger_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='English', rot=0)
older_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='Non English', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage')

sns.despine()
plt.show()


# English people seem to be more determined to book unlike non-english whom percentage is quite large in NDF!

# # Feature Encoding

# #One hot encoder only takes numerical categorical values, hence any value of string type should be label encoded before one hot encoded.
# 
# from sklearn.preprocessing import LabelEncoder 
#   
# le = LabelEncoder() 
#   
# users['gender']= le.fit_transform(users['gender']) 
# users.head()
# 
# #creating one hot encoder object with categorical feature 0 
# #indicating the 9th column which is gender
# from sklearn.preprocessing import OneHotEncoder 
# onehotencoder = OneHotEncoder(categorical_features = [9]) 
# data = onehotencoder.fit_transform(users).toarray() 

# # Random Forest Model

# 1) Check that there is no null values:
# 

# In[ ]:


import math
users["age"].isna().any()


# In[ ]:


users["gender"].isnull().values.any()


# In[ ]:


users["age"].isnull().values.any()


# 2) Split the data again to train set and test set

# In[ ]:


train_users = users.iloc[:213451 , :]
train_users


# In[ ]:


test_users = users.iloc[213451: , :]
test_users.drop(['country_destination'], axis=1)


# 3) Build the model and calculate the cross validation accuracy
# 
# Due to time limitation, I only considered two features to use in the training. This is of course not an efficient way to train the model.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

#Convert categorical variable into dummy/indicator variables.

y = train_users["country_destination"]
features = ["gender","age"]
X = pd.get_dummies(train_users[features])
X_test = pd.get_dummies(test_users[features])
X.head()


# In[ ]:


from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)
model_random_cv=cross_val_score(model, X, y, cv=5) 
print (model_random_cv.mean())


# This is a relatively low accuracy but I will accept it for now, again due to time limitation.
# 4) Fit the model to the train-set and then predicting y for the test-set 

# In[ ]:



model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'id': test_users.id, 'country': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

