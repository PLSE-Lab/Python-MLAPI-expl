#!/usr/bin/env python
# coding: utf-8

# # The RMS Titanic
# The Titanic dataset is often the first step of a journey into Kaggle's machine learning world - it was mine. It's not without irony that one's first ML project involved analyzing the journey of a ship whose maiden voyage ended in disaster :-)
# 
# The Titanic competition asks us to predict if someone on that ill-fated journey survived or not. To help do this it give us about 10 attributes of data for each passenger - things like their age, gender, name, where they embarked etc. It takes the data of everyone on the Titanic and breaks it up into two portions:
# * a Training Portion where we are given these attributes and, in addition, the outcome ie whether the person survived or not and
# * a Test Portion where we are given just attributes and asked to predict whether the person survived or not
# 
# The data is stored as a CSV ie a Comma Separated Value. 
# 
# So let's get to it!
# 

# **First, we are going to take a look at what is in the datasets!**

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


# The output of the cell above gives us three files - the test file called test.csv, the training file called train.csv and a gender_submission.csv which actually does not have a gender but rather is a sample of the format of the file required for submission to the competition. This csv has two columns - one is the PassengerID which is from the test.csv file and a Survived column which is what we will predict.
# 
# Now that we have the names of the files of the data sets, let's load them up into Panda dataframes (DFs). Panda is a python library that helps us manipulate data. Conceptually, a Panda DF can best be approximated by a database table. It has both rows and columns. As in a DB you can access one or may rows, one or many columns or exactly one particular cell. There is a universe of methods and functions available for inspecting, querying and updating the data in a DF

# In[ ]:


from datetime import datetime
started_at = datetime.now().strftime("%H:%M:%S")

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


# Now that we have a data set, let's take a quick peek at what's inside. This can be done several ways. There is the head() command that shows us the first 5 rows - literally a quick peek data. There is also a describe() command that does some nice analytics on the contents - things such as min value, max value, mean value, how many values are not null etc. All very useful to know. 
# 
# Finally, we also want to check if there data has any nulls. Usually we will want to do something with the nulls - either drop them altogether or fill them in with some logical value (such as the average if it is numerical or the most common if it is a list of labels and so on). Ok let's get to it:
# 

# In[ ]:


print(train_data.head())
print(train_data.describe(include="all"))


# In[ ]:


# Survived values of 0 indicate they did not survive while 1 indicated that they did survive
# The crosstab function just shows counts but it is easier to absorb the info as percentages so we can calculate that as well with the apply
# Let's see how the passenger's Gender, Class of Travel and Embarkation point are distributed among the survivors
print(pd.crosstab(train_data["Sex"],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))
print("-"*50)
print(pd.crosstab(train_data["Pclass"],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))
print("-"*50)
print(pd.crosstab(train_data["Embarked"],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))


# A much larger fraction of females survived than did males. Just over 25% of the women died but over 81% of the men died. Similarly 75% of First class passengers survived while only 37% of 3rd class passengers did. Finally, 66% of people that got on at Southampton made it while only about 45% of the Cherbourg people did. What can we conclude? For example it was the age of chivalry (women over men), the English didn't care much for the French (over 60% of English port embarkers survived vs less than 50% of the French ones!) and that it was good to be rich: with almost 63% of first class passengers surviving while just 25% of 3rd class passengers did! It's quite entertaining to make some of these sweeping over-generalizations :-)
# 
# This suggests that these attribute will be particularly valuable to the prediction model. If the split had been close to 50-50 then it would be less valuable.
# 
# So far we have only looked at the data in a manner of peeking. Now let's actually ***LOOK*** at it via colorful graphs and visualizations! Onto our next Python library called seaborn that lets us plot all sorts of data visualizations and graphs and look at relationships etc etc.  We'll use two other Python libraries: seaborn and matplotlib for the graphs and visualizations.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x = 'Sex', hue = 'Survived', data = train_data)


# In[ ]:


sns.countplot(x = 'Embarked', hue = 'Survived', data = train_data)
# The graph below shows us that people who got on at Southampton had the best chances of survival. Another useful attribute for our model


# In[ ]:


sns.countplot(x = 'Pclass', hue = 'Survived', data = train_data)
# The graph below shows us that people who got on at Southampton had the best chances of survival. Another useful attribute for our model


# In[ ]:


# Let's see if any of the data values are null
print(train_data.shape)
print(train_data.isnull().sum())


# There are 891 rows of data in the train set i.e. information about 891 passengers:
# 
# There are **three** colums that contain nulls
# * **Age**: 177 of them are null.
# * **Cabin**: 687 (a large majority) are null
# * **Embarked**: Only 2 are null here
# 
# We will ingore Cabin for now and clean up Age and Embarked. Let's look at the data in Embarked. Age is self-explanatory. The dropna=False below counts the # of nulls as well.

# In[ ]:


print(train_data.Embarked.value_counts(dropna=False))
print(train_data.groupby(train_data['Pclass']).Age.median())
print(train_data.groupby(train_data['Embarked']).Age.median())


# The S, C and Q stand for Southampton, Cherbourg and Queenstown which are the three ports of embarkation. The mode() is defined as the most common value in the set and it makes logical sense for us to substitute the blanks with S (Southampton from where the most people got on)
# 
# For Age, which is a number we just fill in the **median()** age which is the value for which separates the group into two - a half that is older and a half that is younger. This is different from average or **mean()** which is often a good idea so we can avoid outliers affecting the value. 
# 
# Now we write a function that gets a DF passed in and for the DF sets all null values of Embarked (that's what the fillna means) with the most common valye or mode while for the Age it sets it to the median age. Could we get a bit more sophisticated than updating **all** missing values of Age to the median of the entire column? Yes, we could. As per the distribution above the median age differs quite a but by Passenger Class. Those in First have a median age of 37, in second of 29 and in third of 24. So rather than set it to the overall data set median we could set each PClass' Age nulls with that PClass' median age and be a bit more accurate.
# 
# The function has an inplace=True which means that the data frame itself gets updated vs just a copy of getting changed without the original getting over-written. inplace=True overwrites the dataset that we passed in. Once these nulls have been filled it we pass the data set back.
# 
# Next we call the function twice, once for the train_data and once for the test_data.

# In[ ]:


def fill_nulls(data):
    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)
    data['Age'].fillna(data['Age'].median(),inplace=True)
    data['Fare'].fillna(data['Fare'].mean(),inplace=True) # In the TEST data we have one null Fare

    return data
    
train_data = fill_nulls(train_data)
test_data = fill_nulls(test_data)


# So let's see what we got accomplished by checking for what columns are still null and what the Embarked column contains post this little massage we gave the data

# In[ ]:


print(train_data.isnull().sum())
print(train_data.Embarked.value_counts(dropna=False))


# Great! There are no columns with Null values other than the Cabin which we deliberately ignored. Also, the two null Embarked values got set to S which has gone from 644 rows to 646 (compare the prior output). Mission accomplished as far as nulls are concerned!

# Next, we want to look at the column called Name. Rather than a simple **John Smith**, the DF has values that are written in the fashion of **Smith, Mr. John**. So, we need to do some string manipulation to get to the title. We do this because we might guess that a person's title had some bearing on whether they survived. We first split the name into an array by the separator "," which creates an array with two variables such [Smith,Mr. John]. For each element of the array we then split the 2nd element further by the separator "." which creates an array with two variables such as [Mr, John]. If we grab the first element of this sub-array, we'll get 'Mr' (or 'Mrs' or 'Dr') which we'll strore in a new column of the DF called Honorific. Let's do that and then check if there is a skew based on this Honorific.

# In[ ]:


def extract_honorific_from_name(data):
    data['Honorific'] = data['Name'] # initialize this new column
    titles = data["Name"].str.split(",") # we now have an array
    for indx,title in enumerate(titles): # for each element of the array
        data["Honorific"][indx] = title[1].split(".")[0] # Get the Mr, Mrs, Dr, etc by parsing it out
    return data

train_data = extract_honorific_from_name(train_data)
test_data = extract_honorific_from_name(test_data)

#print(train_data.Honorific.value_counts(dropna=False))
#print("=======================")
#print(test_data.Honorific.value_counts(dropna=False))
test_data['Honorific'] = test_data['Honorific'].str.replace('Dona','Mrs')
#print(test_data.Honorific.value_counts(dropna=False))

print(pd.crosstab(train_data['Honorific'],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))
print(train_data['Honorific'].value_counts())

So we see two interesting things now. The first is that there are 17 Honorifics in the data set, the first four account for almost 97% of the set. We can update all the other Honorifics to 'Other' to keep things simple and not lose much 'signal'.
# In[ ]:


#blank_array = ['Z' for n in range(len(train_data))]
#train_data['foobar'] = blank_array
#print(train_data['foobar'].value_counts())
#train_data.loc[train_data['Honorific'] == 'Mr', train_data['Honorific']]


# The next step is to convert all the non numeric attributes to numbers so that our model can more easily work with them. The way we are going to do this is by label encoding wherein all values such as A, B and C get 'mapped' or 'encoded' to 1, 2 and 3. Python's sklearn has a library to do this and we import it and the thing to note here is that we want to encode the test and train data the same way ie if an A in the train gets mapped to a 1 then we want an A in the test also to get mapped to a 1. Once we do that we can look at the value distribution again via value_counts() for these three columns/attributes.

# In[ ]:


# Now to encode. We can't pass data twice as the same encoding for train has to be applied for test also
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
train_data['Embarked'] = enc.fit_transform(train_data['Embarked'])
test_data['Embarked'] = enc.transform(test_data['Embarked'])

train_data['Sex'] = enc.fit_transform(train_data['Sex'])
test_data['Sex'] = enc.transform(test_data['Sex'])

train_data['Honorific'] = enc.fit_transform(train_data['Honorific'])
test_data['Honorific'] = enc.transform(test_data['Honorific'])



#print(train_data.isnull().sum())
print(train_data.Embarked.value_counts(dropna=False))
print(train_data.Sex.value_counts(dropna=False))
print(train_data.Honorific.value_counts(dropna=False))


# Looks good! Moving on now to get at some 'derived' attributes. 
# 
# There are two attributes called SibSp and Parch which stand for Sibling & Spouse and Parents & Children that indicate if a passenger was traveling with such relatives. Let's add them up to come up with a CoPassengers attribute.
# 
# Second, if the CoPassengers value is 0 vs not set a new attribute called Solitary
# 
# Third, if the age of the passenger was 16 or under create a new attribute called Minor
# 
# Fourth, the mean Fare is 32 for the entire train set. If the Fare is 32 or below vs above 32 create an Attribute called FareCategory
# 

# In[ ]:



def massage_data(data):
    data["CoPassengers"] = data["SibSp"] + data["Parch"]
    data['Solitary'] = np.where(data['CoPassengers'] > 0, 1, 0)
    data['Minor'] = np.where(data['Age']<=16, 1, 0)
    data['FareCategory'] = np.where(data['Fare'] <= 32, 'X', 'Y')    
    return(data)


# Now, let's massage the data for both sets and the see the distribution or skew of Survived/Not based on these new attributes.

# In[ ]:


train_data = massage_data(train_data)
test_data = massage_data(test_data)

print(pd.crosstab(train_data['CoPassengers'],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))
print(pd.crosstab(train_data['Solitary'],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))
print(pd.crosstab(train_data['Minor'],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))
print(pd.crosstab(train_data['FareCategory'],train_data.Survived).apply(lambda r: r/r.sum(), axis=1))


# None of this is a boring 50:50 so they are worth considering in our model.

# In[ ]:


# A choice of columns for training the model. It's a good idea to play with this list
columns_for_fitting = ['Age','Minor','Sex','Honorific','Embarked','Pclass','Fare','Solitary'] 


# In[ ]:


X = train_data[columns_for_fitting]
y = train_data['Survived']
X1 = test_data[columns_for_fitting]


# Now we split the training data into 80%/20% cohort as a train and validation mode. That is we want to 'hold back' 20% of the rows for which we know the outcome and train the model on the 80% balance. Once the model has trained ie come up with weights then we validate it on the 20% cohort to see how many outcomes we got right. We can't do this on the test data as that does not have the 'Survived' column. sklearn has some helpful libraries to do this.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, y, test_size=0.2,random_state=21)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)


# Now that the train/validatin cohorts are available we can let multiple models (sometimes with multiple parameters for each model) train on the 80%, then validate on the 20% and then check the accuracy and pick the model/parameters with the best accuracy score.
# 
# There is no guarantee that this accuracy will hold on the unseen data in test but it's better than flying blind.
# 
# We try 4 models here - Kneightbors, RandomForest, XGB and SVC and see what happens in each case below

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


from sklearn.metrics import accuracy_score
from sklearn import preprocessing

svc_model = SVC()
svc_model.fit(X_train,Y_train)
svc_predictions = svc_model.predict(X_test)
print("SVC = {}".format(accuracy_score(Y_test,svc_predictions)))

x_model = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, objective= 'binary:logistic',gamma=0,
 subsample=0.8,colsample_bytree=0.8,scale_pos_weight=1)
x_model.fit(X_train,Y_train,verbose=False)
x_predictions = x_model.predict(X_test)
print("XGB = {}".format(accuracy_score(Y_test,x_predictions)))

est = [20,50,100,150,250,300,500,750,1000,2500]
est=[550]
for n in est:
    rf_model = RandomForestClassifier(n_estimators=n)
    rf_model.fit(X_train,Y_train)
    rf_predictions = rf_model.predict(X_test)
    #print("RandomForest = {}".format(accuracy_score(Y_test,rf_predictions)))
    print ('For RF: n and Accuracy are', n,accuracy_score(Y_test,rf_predictions))

k = 6
while k < 7:
    k = k + 1
    X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
    knn_model = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)
    X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
    knn_predictions = knn_model.predict(X_test)
    #print("KNN = {}".format(accuracy_score(Y_test,knn_predictions)))
    print ('K and Accuracy are', k,accuracy_score(Y_test,knn_predictions))


# Based on these results the RF model seems best ie with the highest accuracy score so we we'll predict the test data in X1 using that model.

# In[ ]:


yhat = rf_model.predict(X1)
yhat[0:5]


# In[ ]:


pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':yhat}).set_index('PassengerId').to_csv('titanic_submission.csv')
print("Okay , We started at " + started_at + " please check for output data now!")


# That run when submitted got a 76% score which isn't great but isn't too shabby either for some basic work. Hopefully this was helpful for other beginners.
