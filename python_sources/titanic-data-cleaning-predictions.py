#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)


# In[ ]:


# reading csv train data into dataframe
train_df = pd.read_csv("../input/train.csv")
# preview train data
train_df.head()


# In[ ]:


train_df.tail()


# In[ ]:


#reading csv test data into dataframe
test_df = pd.read_csv("../input/test.csv")
# preview test data
test_df.head()


# In[ ]:


test_df.tail()


# **One of the most elementary steps to do this is by getting a basic description of your data. A basic description of your data is indeed a very broad term: you can interpret it as a quick and dirty way to get some information on your data, as a way of getting some simple, easy-to-understand information on your data, to get a basic feel for your data. We can use the describe() function to get various summary statistics that exclude NaN values.**

# In[ ]:


# printing the total sample in train data
print("#of sample in the train data is {}".format(train_df.shape[0]))


# In[ ]:


# printing the total sample in the test data
print("#of sample in the test data is {}".format(test_df.shape[0]))


# In[ ]:


# checking missing values in the train data
train_df.isnull().sum()


# **To get better clarity lets visualize null values**

# In[ ]:


import missingno as msno


# In[ ]:


msno.matrix(train_df)


# In[ ]:


msno.matrix(test_df)


# In[ ]:


msno.heatmap(train_df)


# In[ ]:


msno.heatmap(test_df)


# In[ ]:


msno.dendrogram(train_df)


# In[ ]:


##percent of missing values in Train data
# ~20% of the Age entries are missing for passengers
# look at what age variable in general


# In[ ]:


ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='blue', alpha=0.6)
train_df["Age"].plot(kind='density', color='brown')
ax.set(xlabel='Age')
plt.xlim(-10,90)
plt.show()


# # Since age is right skewed using mean might get biased results by filling in ages that are older than desired. 
# # To deal with this, we'll use the median to impute the missing values.
# 
# 

# In[ ]:


#mean age
print("the mean of age is %.2f" %(train_df['Age'].mean(skipna = True)))


# In[ ]:


# median of age
print('the median of age is %.2f' %(train_df["Age"].median(skipna = True)))


#    # Cabin
#     # percentage of missing cabin variable

# In[ ]:


print('percent of missing "Cabin" record is %.2f%%' %((train_df["Cabin"].isnull().sum()/train_df.shape[0])*100))


# # almost 77% of the cabin records are missing we cannot use this cabin data
# 

#                   #Embarked
# # percentage of missing Embarked table

# In[ ]:


print("percent of missing 'Embarked' record is %.2f%%" %((train_df["Embarked"].isnull().sum()/train_df.shape[0])*100))


# In[ ]:


#Dividing the class of passengers in Train Data
print("Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):")
print(train_df['Embarked'].value_counts())
sns.countplot(x='Embarked', data =train_df, palette= 'Set2')
plt.show()


# In[ ]:


#Finding the common boarding among the three from train data
print('The most common boarding port of embarkation is %s.' %train_df['Embarked'].value_counts().idxmax())


# In[ ]:


# In a given row if Age is the missing value. Then i will impute median of Age
# If Embarked is the missing value in the row then i will impute as S
# i will ignore Cabnit bec. there is too much missing values in cabnit
train_data = train_df.copy()
train_data['Age'].fillna(train_df["Age"].median(skipna=True), inplace= True)
train_data["Embarked"].fillna(train_df["Embarked"].value_counts().idxmax(), inplace=True)
train_data.drop("Cabin", axis=1, inplace=True)


# In[ ]:


# Checking the missing values in Adjusted data
train_data.isnull().sum()


# In[ ]:


# preview adj train data
train_data.head()


# In[ ]:


plt.figure(figsize=(15,5))
ax = train_df['Age'].hist(bins=15, density=True, stacked = True, color = "green", alpha=0.6)
train_df["Age"].plot(kind='density', color = "green")
ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='yellow', alpha=0.5)
train_data['Age'].plot(kind='density', color = 'yellow')
ax.legend(['Raw Age', 'Adjusted Age'])
ax.set(xlabel='Age')
plt.xlim(-10,90)
plt.show()


# # from the kaggle directory both "SibSp= #of siblings"and "Parth #of parents" relate to traveling with families
# # ('multicollinearity: multiple ind. variables which have highly corrilated') for #simplicity's sake we'll comine the effect of
# # these variables into one categorical predictor
# # finding out whether or not that individual was traveling alone
# 

# In[ ]:


## Create categorical variable for traveling alone
train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)
train_data.drop('SibSp', axis=1, inplace=True)
train_data.drop('Parch', axis=1, inplace=True)


# In[ ]:


#we will also create a categorical variable for passenger-("pclass") Gender-"sex" and port Embarked = "Embarked"

# creating categorical variable and dropping some variables
training=pd.get_dummies(train_data, columns=["Pclass", "Embarked", "Sex"])
training.drop("Sex_female", axis=1, inplace=True)
training.drop('Name', axis=1, inplace=True)
training.drop("Ticket", axis=1, inplace=True)
training.drop('PassengerId', axis=1, inplace=True)

final_train = training
final_train.head()


# # Till now we did for train data now we will the same changes for test data
# # I'm apply to same imputation for "Age" in the Test data as I did for my Training data(In place of mising , 
# # Age(median value)=28) we'll also remove "Cabin" variable 
# # There were no missing values in the "Embarked" port variable.
# #lets add Dummy variable to finalize the test data set.
# # at Final, we'll impute the missing value (with median = 14.45) for "Fare"
# 

# In[ ]:


test_df.isnull().sum()


# In[ ]:


test_data = test_df.copy()
test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)
test_data['Fare'].fillna(train_df["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)


# In[ ]:


test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)

test_data.drop("SibSp", axis=1, inplace=True)
test_data.drop("Parch", axis=1, inplace=True)


# In[ ]:


testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)


# In[ ]:


final_test = testing
final_test.head()


# In[ ]:


# Explanatering & analysing The Data


# In[ ]:


#Exploaring the Age Data
plt.figure(figsize=(16,9))
ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="lightgreen", shade=True)
sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="orange", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
ax.set(xlabel='Age')
plt.xlim(-10,90)
plt.show()


# #from the above graph we can look at among all the passengers majority of the survived are children
# #which means Among all the passengers they gave first priority to save children.
# 

# In[ ]:


# so The passengers evidently made an attempt to save children by giving them a place on the life rafts.
plt.figure(figsize=(23,12))
avg_survival_byage = final_train[["Age", "Survived"]].groupby(["Age"], as_index=False).mean()
g = sns.barplot(x="Age", y= "Survived", data=avg_survival_byage, color = "Black")
plt.show()


# In[ ]:


# From the Bar plot i'll considering the survival rate of passengers under 17, 
# I'll also include another categorical variable in my dataset: Minor
final_train["IsMinor"]=np.where(final_train["Age"]<=17, 1, 0)


# In[ ]:


#we'll look at test data (it,s my assumption for considering miner = under 17)
final_test["IsMinor"]=np.where(final_test["Age"]<=17, 1, 0)


#    **#Exploration for Fare****
# 

# In[ ]:


plt.figure(figsize=(15,8))
ax = sns.kdeplot(final_train["Fare"][final_train.Survived ==1], color= "darkgreen", shade = True)
sns.kdeplot(final_train["Fare"][final_train.Survived ==0], color = "lightcoral", shade =True)
plt.legend(["Survived", "Died"])
plt.title('Density Plot of Fare for Surviving Population And Deceased Population')
ax.set(xlabel="Fare")
plt.xlim(-25,250)
plt.show()


# # From the above graphical reprentation it is clearly different for the fares of survivers v/s unservived
# # passengers who paid low fare appear to have been less likely to survive.
# # This is Strongly co-related with passengers Class, which will look next
# 

# In[ ]:


# Exploration of Passengers Class
sns.barplot('Pclass', 'Survived', data= train_df, color="yellow")
plt.show()


# #From The above Barplot 1st class people are mostly survived****

# In[ ]:


##Exploration of Embarked part

sns.barplot('Embarked', 'Survived', data=train_df, color="green")
plt.show()


# Passengers who boarded in Cherbourg, France, appear to have the highest survival rate. Passengers who boarded in Southhampton were marginally less likely to survive than those who boarded in Queenstown. This is probably related to passenger class, or maybe even the order of room assignments for instance it may be earlier passengers were more likely to have rooms closer to deck("placeses were given in the internet") It's also worth noting the size of the whiskers in these plots. Because the number of passengers who boarded at Southhampton was highest, the confidence around the survival rate is the highest. 
# 

# In[ ]:





# In[ ]:


##Exploration with alone and withFamily
sns.barplot('TravelAlone', 'Survived', data=final_train, color="yellow")
plt.show()


# # The passengers who are travelling without family were more likely to die in the disaster than those were with Family
# # It's likely that individal passengers travelling alone were likely male

# In[ ]:


##Explonaration of Gender Variable
sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")
plt.show()


#  Clerly from the above bar plot being Female increased the chances of survival

# **Conclusion :At first we looked at missing values from the both Train & test Data then we find there are missing values in 3 different columns At the Age factor when we  look we found that approx. 20% are there missing values Then in the Cabin column there are mostly 77% of them are missing So we drop that column later we looked at the percentage of missing columns in Embarked we got only 0.22%. 
#    After that we did preprocessing & adjusting the data . Then we looked at one more time whether there're any miising values in the Adjusted Data . later we looked at the Passsengers class (i.e.. there're 3 classes S, C, R)
#    Later we looked at relatives of the passengers & the the people came up with their families. and individuls
#    Among the individuls majority of the people are male. Later we got to the conclusion About the which class people are survived mostly 
#     Among All the passengers people prefered first save childern & women in the graph it left skewness mostly minor people were survived and the people who paid higher fare were also survived(majority of them) and the majority of the unsurvived are male (individuals).**

# In[ ]:




