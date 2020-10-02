#!/usr/bin/env python
# coding: utf-8

# The **Titanic** was a luxury British steamship that sank in the early hours of April 15, 1912 after striking an iceberg, leading to the deaths of more than 1,500 passengers and crew. [Link](https://www.history.com/topics/early-20th-century-us/titanic)

# ![](https://drive.google.com/uc?id=1zsW-gDKsJCuroZIqp5nzK26tXn_FGXiN)

# We may all heard what happened to Titanic. But one day when I tried to analysis the incident, I found a significant difference in the number of survivals between **male** and **female**. Reading the following Kernel, you will understand my concern and why did I choose the title of "*Where are Feminists?*"

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# # Intro

# Let me explain what you will get after reading this document:
# - "**Data Analysis**" of the Titanic Passengers (*using data analysis techniques in Python*)
# - Using "**Machine Learning**" technique to predict if a Passenger survived from the incident.
# So, this Kernel could be useful for the people who loves data and exploring around it. Yes I didn't use complicated statistic formulas in this Kernel but in some point, I'm pretty sure you will love reading it.
# 
# #### ABOUT THE TITLE
# First thing first, I do respect women and also Feminists. However, when I worked with this data set, I've found out that the number of passengers survival was hilariously against **men** in Titanic!
# 
# - Did you know that only **37% of men** from 1st class cabin survived, while this number was around **97% for woman** with the same class?
# - And it gets worst for the 2nd class tickets. **92% of woman** were survived at this class but only **16% of men** could get on life boats.
# The reason that I chose the title of "*Where are feminists*" for this Kernel was, I strongly believe if the numbers were opposite, it could be still a great issue against men!  **:)**
# 
# Once again, I do respect Feminists and all the women around the world, but next time when you were on a sinking ship, expect the same equality for all the genders. 1 men and 1 women, we should survive side-by-side!
# 
# #### DATA SET
# Okay then, it's time to get serious and back to work. In this Kernel, I followed these steps to finally have a good insight to predict our Target feature (*Survived*):
# - 1- Import all the reuquired libraries
# - 2- Read in Data
# - 3- Exploratory Data Analysis (EDA)
# - 3-1- Outliers
# - 3-2- Missing Values
# - 4- Featured Engineering
# - 5- Machine Learning Methods

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# # Import

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')


# # Read in Data

# In[ ]:


test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')

print('Test Shape: ', test.shape)
print('Train Shape: ', train.shape)

df = train.append(test, ignore_index=True)
print('Total (test & train append) Shape: ', df.shape)


# A few information about features we have in our data set:
# - *embarked* (Port of Embarkation):
# C = Cherbourg, Q = Queenstown, S = Southampton
# - *cabin*: Cabin number
# - *pclass*: Ticket class
# - *sibsp*: # of siblings / spouses aboard the Titanic
# - *parch*: # of parents / children aboard the Titanic

# Take a look at our data frame:

# In[ ]:


train.sample(5)


# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# # EXPLORATORY DATA ANALYSIS (EDA)

# ## **Male** & **Female** Passengers

# In[ ]:


print('Total number of Male passengers: %d Person (%d%%)'
      % (df.Sex.value_counts()[0],
      round(df.Sex.value_counts()[0]/len(df.Sex),2)*100))
print('Total number of Female passengers: %d Person (%d%%)'
      % (df.Sex.value_counts()[1],
      round(df.Sex.value_counts()[1]/len(df.Sex),2)*100), '\n')

print('Average Cost of Ticket Fare for Male Passengers: %d USD'
      % round(df.loc[df['Sex']== 'male', 'Fare'].mean(), 2))
print('Average Cost of Ticket Fare for Female Passengers: %d USD'
      % round(df.loc[df['Sex']== 'female', 'Fare'].mean(), 2))


# It was predictable that the ticket fare would be much higher for female than male!
# _____
# Those travelling in **first class**, most of them the wealthiest passengers on board, included prominent members of the upper class, businessmen, politicians, high-ranking military personnel, industrialists, bankers, entertainers, socialites, and professional athletes. **Second-class** passengers were middle-class travellers and included professors, authors, clergymen, and tourists. **Third-class** or steerage passengers were primarily emigrants moving to the United States and Canada. [Link](https://en.wikipedia.org/wiki/Passengers_of_the_RMS_Titanic#cite_note-Hall-4)
# _____

# In[ ]:


# Let's plot the Results:
fig, (ax1,ax2) = plt.subplots(2, figsize=(9,6))

# Ax1
p = sns.countplot(data=df, x=df['Sex'], ax=ax1)
p.set(ylabel='Count')

# Ax2
i = df[['Fare','Sex']].groupby('Sex').mean().plot(kind='bar', ax=ax2, colormap='PiYG')
i.set_ylabel('Ticket Fare')

plt.xticks(rotation=0)
plt.tight_layout()


# ### Who **Survived** most? Male or Female?

# In[ ]:


train[['Pclass', 'Survived', 'Sex']].groupby(['Pclass', 'Sex']).mean()


# As it seems from above table, most of the female (more than 90%) with Class 1 and Class 2 tickets were survived, while this number decreased dramatically for male passengers.

# ![](https://drive.google.com/uc?id=1qWo67nrlUdvZNqfhZtZAZ_xvNNZ_M7ZX)

# Let's say good bye to Jack and other %80 of other **male** passengers...

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## Outliers

# When I tried to work with NaN values, I found some outliers in 'Fare' column and that is why I added this section into my ML process in order to get rid of them.
# 
# Check out the 'Fare' feature:

# In[ ]:


df['Fare'].describe()


# Although **50%** of passengers paid around **15 USD**, but the max Fare price is 513 USD

# In[ ]:


df['Fare'].plot()


# Looking at the plot, it shows that we have 4 values with unreasonable fare price:

# In[ ]:


df.loc[df['Fare'] > 300]


# As it seems from data frame, all of these 4 passengers bought **first class** ticket from **C port** with same **Ticket** number. However, the Fare is **NOT** rational and we can change it into:

# In[ ]:


df.loc[(df['Fare'] < 300)]['Fare'].sort_values(ascending=False)[:1]


# According to above code, after 512 USD tickets, the highest Fare price is 263 USD. So, we will replace 512 with 263:

# In[ ]:


df['Fare'].replace({512.3292:263}, inplace=True)


# In[ ]:


# Let's plot Fare column again to check whether 512 USD tickets replaced successfully:
df['Fare'].plot()


# Well, it seems more reasonable now!

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## Examine Missing Values

# ![](https://drive.google.com/uc?id=1n0M4tmDxkj3yJVz_jHZuz8rfNehS5ExB)

# In[ ]:


print(df.isnull().sum().sort_values())


# So, there are 3 main features with NaN values that we need to fix.
# But first, let's get rid of 4 rows. When I checked null values in Name column, I saw that Embarked is the only column with a value. So we will drop all these 4 rows as these are pretty useless.
# ___

#    ### 1.  **Fare**

# In[ ]:


df[df['Fare'].isnull()]


# It's a 3rd class ticket, so we can replace the NaN value with the fare mean of 3rd class tickets which is 13.3 USD:

# In[ ]:


df.loc[df['Pclass']==3.0]['Fare'].mean()


# In[ ]:


# Fill that NaN value with 13.30 USD
df['Fare'] = df['Fare'].fillna('13.30').astype('O')


# ___

# ### 2.  **Embarked**

# In[ ]:


print(df['Embarked'].value_counts())
df[df['Embarked'].isnull()]


# In[ ]:


# Let's fill these 2 NaN 'Embarked' values with Southampton port because most of the passengers get on the ship at this port
df['Embarked'] = df['Embarked'].fillna('S')


# ___

# ### 3.  **Age**

# To decide which feature should we use to fill NaN values from Age column, let's check the correlation between 'Age' and other features:

# In[ ]:


df[df.columns].corr()['Age'][1:]


# It seems that the most correlated feature with Age is **Pclass** with %40.
# 
# Moreover, from code below I found out there was a little difference of Age between male and female in different classes:

# In[ ]:


# Age difference in different classes  
df.groupby(['Pclass','Sex'])['Age'].mean()


# So, it might be a good way to groupby our data frame on **Pclass** and **Sex** and then fill the NaN values with the Age mean of that group:

# In[ ]:


df['Age']=df.groupby(['Pclass','Sex']).Age.transform(lambda x: x.fillna(x.mean()))


# In[ ]:


# Check NaNs on Age column
df.Age.isnull().sum()


# ___

# ### 4.  **Cabin**

# In[ ]:


value = round((df.Cabin.isnull().sum()/df.shape[0])*100,2)
print('Around %d%% of the values in Cabin feature is null.' %value)


# 77% of missing data is pretty high. Plus, the feature itself doesn't seem to have a significant impact over our prediction. So I'll drop this feature from our dataframe:

# In[ ]:


df.drop(['Cabin'],axis=1,inplace=True)


# Great, we got rid of missing data. Now it's time to go deeper

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## Encoding Categorical Variables

# ![](https://drive.google.com/uc?id=1Q5pPTqAOx0B8y81mMeJY148NtT_uQdW8)

# First, we need to find all the categorical features in our dataframe:
# 
# {PS: * if you want to learn more about Categorical Variables, you can find this very useful Kernel from Will Koehrsen*: [Link](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction#Feature-Engineering)}

# In[ ]:


# All the Columns
cols = df.columns

# Find the Numeric Columns
num_cols = df._get_numeric_data().columns

# Find the Categorical Columns
list(set(cols)-set(num_cols))


# Oops! It seems the type of 'Fare' isn't numeric in our dataframe. So, first we change the type of 'Fare' feature from object to numeric. And then, we will encode two other categorical columns: **Embarked** and **Sex**. (*Because the ticket & name columns are not useful for this kernel*)

# **(1)** Change 'Fare' type to numeric

# In[ ]:


df['Fare'] = pd.to_numeric(df['Fare'])


# **(2)** Encode 'Sex' column
# 
# Because there's only 2 unique values in 'Sex' column, we'll use **'LabelEncoding'**
# 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Sex'])

df['Sex'] = le.transform(df['Sex'])


# **(3)** Encode 'Embarked' column
# 
# Because 'Embarked' has 3 values of 'S', 'C' and 'Q', we'll use **'OneHotEncoding'**

# In[ ]:


df = pd.get_dummies(df, columns=['Embarked'])


# Now, all the needed columns are numerics. We won't need 'Name' and 'Ticket' columns so for the final step, let's drop these two:

# In[ ]:


df.drop(['Name','Ticket', 'PassengerId'], axis=1, inplace=True)


# In[ ]:


df.head(2)


# Greate! No categorical columns are left for ML process.
# ![](https://lh3.googleusercontent.com/oO60bJ7GZZYZa_81ckzEXNaLM_Ok6fO-dE3JVjERkAXs3mmgX0bGnUIeKc56EAnCFmHHbLcWOy4hYh_UtEHK=w1440-h789-rw)

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# # Featured Engineering

# Freature Engineering is about make new feaures and learn about the interactions between them and its affects on our Target column (here: **'Survived'**).

# ### Polynomial Features

# In[ ]:


#1: make a new data frame for polynomial features
poly_features = df[['Sex', 'Age', 'Pclass', 'Survived']]

poly_target = poly_features['Survived']
poly_features = poly_features.drop(columns=['Survived'])

#2: create polynomial object with specified degree
from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree=3)

#3: train polynomial features
poly_transformer.fit(poly_features)

#4: transform the features
poly_features = poly_transformer.transform(poly_features)
print('Polynomial Features shape: ', poly_features.shape)


# In[ ]:


#5: check to see whether any of these new features are correlated with the Target (here: 'Survivde')
poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names(['Sex', 'Age', 'Pclass']))

# Add the Target column
poly_features['Survived'] = poly_target

# Find the correlations with the Target
poly_corr = poly_features.corr()['Survived'].sort_values()

poly_corr


# Nope, it seems Sex column has higher correlation than the 'Sex' & 'Pclass' columns itself, so we will leave the data frame without any changes in its features.

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# # Machine Learning Models
# ![](https://drive.google.com/uc?id=1ybkxA0PO7gcwDdUgo-sEUUGSN0F-h6R6)

# Now it's time to predict 'Survived' values from Test dataframe using ML models. But our data frame is not ready yet, we need to do a small preprocessing.

# ## Preprocessing

# Let's take a look at our dataframe again:

# In[ ]:


df.sample(4)


# All the features are likely being in a similar range, except *Age* and *Fare*. In Machine Learning, this huge difference could affect our prediction value. So, we'd better to *normalize* the range of our features.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Create a copy of our df and then drop the 'Target' column as part of it contains null values (from test data set)
df_scaled = df.copy()
df_scaled.drop('Survived', axis=1)

# Feature Names
features = list(df_scaled.columns)

# Scale Features 
scaler = MinMaxScaler(feature_range=(0,1))

# Fit on our data
scaler.fit(df_scaled)
df_scaled = scaler.transform(df_scaled)


# Let's take a look at our df_scaled dataframe:

# In[ ]:


i=pd.DataFrame(df_scaled, columns=features)
i.head(5)


# Good! Now we have a normalized data frame ready to run our ML models over it. And now, to do our ML job, we need to split this normalized dataframe into two part based on Survived feature: *Train* and *Test*

# In[ ]:


# Split our data frame into train and test
train_scaled = i.loc[i['Survived'].notnull()]
test_scaled = i.loc[i['Survived'].isnull()]

print('Train Data Frame (scaled) Shape: ', train_scaled.shape)
print('Test Data Frame (scaled) Shape: ', test_scaled.shape)


# Well, our **Train** and **Test** data frame are now normalized.

# In[ ]:


test_scaled.reset_index(drop=True, inplace=True)
train_scaled.reset_index(drop=True, inplace=True)


# In[ ]:


print('Test Shape: ',test.shape)
print('Test Scaled Shape', test_scaled.shape)


# In[ ]:


test_scaled.index.difference(test.index)


# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## 1- Logistic Regression

# **IMPORTANT NOTE**: As the '**Survived**' column in *test* data set is null, there's no way to check the accuracy of our model by predicting values of this feature. So, we need to split *Survived* values from **Train** data set cause we already have the results.
# 
# - After testing our model on *Train data set*, we can **generalize** it on the *Test data set* as well.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = train_scaled.drop(columns='Survived', axis=1)
y = train_scaled['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)


# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[ ]:


prediction = lr.predict(X_test)


# Using Logistic Regression, we could predict our y_test values and put them all inside prediction variable. Let's check the accuracy of our model using Confusion Matrix, Classification Report and Accuracy Score:

# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
print('* Confusion Matrix: \n', confusion_matrix(y_test, prediction), '\n')
print('* Classification Report: \n', classification_report(y_test, prediction))


# In[ ]:


from sklearn.metrics import accuracy_score
lr_asc = round(accuracy_score(y_test, prediction),3)*100
print('Logistic Regression Model Accuracy Score is: %', lr_asc)


# ___

# ## 2- Decision Tree

# First, we need to do the same train_test_split step in order to split our test and train data from each other:

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# Create a model called dtree
dtree = DecisionTreeClassifier()


# In[ ]:


# like other sikit learn models, we will fit this model to the training data
dtree.fit(X_train, y_train)


# In[ ]:


predictions = dtree.predict(X_test)


# In[ ]:


print('* Confusion Matrix: \n', confusion_matrix(y_test, prediction), '\n')
print('* Classification Report: \n', classification_report(y_test, prediction))


# In[ ]:


dt_asc = round(accuracy_score(y_test, predictions),3)*100
print('Decision Tree Model Accuracy Score is: %', dt_asc)


# ___

# ## 3- Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)

# Train on training data
rfc.fit(X_train, y_train)


# In[ ]:


# Make predictions on the test data
rfc_predict = rfc.predict(X_test)


# In[ ]:


print('* Confusion Matrix: \n', confusion_matrix(y_test, rfc_predict), '\n')
print('* Classification Report: \n', classification_report(y_test, rfc_predict))


# In[ ]:


rf_asc = round(accuracy_score(y_test, rfc_predict),3)*100
print('Random Forrest Model Accuracy Score is: %', rf_asc)


# ___

# ## 4- Supported Vector Machines

# In[ ]:


# We have our data split, now let's go ahead and train the support vector classifier to grab the support vector classifier model

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


print('* Confusion Matrix: \n', confusion_matrix(y_test, predictions), '\n')
print('* Classification Report: \n', classification_report(y_test, predictions))


# In[ ]:


svm_asc = round(accuracy_score(y_test, predictions),3)*100
print('SVM Model Accuracy Score is: %', svm_asc)


# **NOTE**: If you remember, we already normalized our data and it makes our model works fine. But if our data wasn't already adjusted and normalized, we can use **grid search**.
# 
# Grid search allows us to find the right parameters such as gamma values.
# 
# ``` from sklearn.grid_search import GridSearchCV ```
# 
# GridSearchCV takes in a dictionary that describes the parameters that should be tried in a model to train. The grid of parameters is defined as a dictionary where the keys are the parameters and the values is basically a list of settings to be tested.

# ___

# **HINT**: Microsoft AzureML Team has created a cheat sheet to choose proper Machine Learning algorithm for predictive analytics. It might be useful for you as well (*open it in new tab for higher resolution*): 

# ![](https://drive.google.com/uc?id=1kXlLOWN0RXUIT4bCsUgr-DCIz7cPDbRc)

# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# ## Which model has the highest accuracy score?

# In[ ]:


plt.bar(height=[lr_asc, dt_asc, rf_asc, svm_asc], x=['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM'],
       color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue')
plt.grid(axis='y')
plt.title(label='Machine Learning Models Accuray Score')


# ![](https://drive.google.com/uc?id=1PuQ33oL0QErS0P9knYqVS9634NVD-Y88)

# **Submission:**

# In[ ]:


X_train = train_scaled.drop(columns='Survived', axis=1)
X_test = test_scaled.drop(columns='Survived', axis=1)
y_train = train_scaled['Survived']
y_test = test_scaled['Survived']

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)

# Train on training data
rfc.fit(X_train, y_train)
# Make predictions on the test data
rfc_predict = rfc.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": rfc_predict
    })


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": rfc_predict
    })


# In[ ]:


filename = 'Titanic_Submission.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

