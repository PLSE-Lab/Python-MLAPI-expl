#!/usr/bin/env python
# coding: utf-8

# ## Predicting Survival on the Titanic
# 
# ### History
# Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time.
# 
# 

# In[ ]:


import re

# to handle datasets
import pandas as pd
import numpy as np

# for visualization
import matplotlib.pyplot as plt

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# ## Prepare the data set

# In[ ]:


# load the data - it is available open source and online

data = pd.read_csv('../input/phpMYEkMl.csv')

# display data
data.head()


# In[ ]:


#getting to know all the columns
data.columns


# In[ ]:


#shape of the data
data.shape


# In[ ]:


#analysing cabin columns
data['cabin'].values[:5]


# In[ ]:


type(data['cabin'].values[0])


# In[ ]:


# replace interrogation marks by NaN values

data = data.replace('?', np.nan)


# In[ ]:


# retain only the first cabin if more than
# 1 are available per passenger

def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
    
data['cabin'] = data['cabin'].apply(get_first_cabin)


# In[ ]:


# extracts the title (Mr, Ms, etc) from the name variable

def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
    
data['title'] = data['name'].apply(get_title)


# In[ ]:


# cast numerical variables as floats

data['fare'] = data['fare'].astype('float')
data['age'] = data['age'].astype('float')


# In[ ]:


# drop unnecessary variables

data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)

# display data
data.head()


# In[ ]:


# save the data set

data.to_csv('titanic.csv', index=False)


# ## Data Exploration
# 
# ### Find numerical and categorical variables

# In[ ]:


data = pd.read_csv("titanic.csv")
data.head()


# In[ ]:


data.info()


# In[ ]:


vars_num =['survived','age','sibsp','parch','fare'] # fill your code here
vars_cat =['pclass','sex','cabin','embarked','title'] # fill your code here


# In[ ]:


# for column in data.columns:
#     if(data[column].dtype == object):
#         vars_cat.append(column)
#     else:
#         vars_num.append(column)


# In[ ]:


target = 'survived'


# In[ ]:


data.columns


# In[ ]:


print('Number of numerical variables: {}'.format(len(vars_num)))
print('Number of categorical variables: {}'.format(len(vars_cat)))


# In[ ]:


vars_num


# In[ ]:


vars_cat


# ### Find missing values in variables

# In[ ]:


# first in numerical variables
data_num = data[vars_num]
data_num.head()


# In[ ]:


len(data)


# In[ ]:


data_num.isna().mean()


# In[ ]:


# now in categorical variables
data_cat = data[vars_cat]
data_cat.head()


# In[ ]:


data_cat.isna().mean()


# ### Determine cardinality of categorical variables

# In[ ]:


def analyse_cat(df, var):
    df = df.copy()
    df[var].value_counts().plot.bar()
    plt.title(var)
    plt.xlabel('attributes')
    plt.ylabel('No of passengers')
    plt.show()


# In[ ]:


for col in vars_cat:
    print(25*'-'+col+'-'*25)
    print(data[col].value_counts())
    print(len(data[col].value_counts()))
    analyse_cat(data, col)


# ### Determine the distribution of numerical variables

# In[ ]:


def analyse_continuous(df, var):
    df = df.copy()
    df[var].hist(bins=30)
    plt.ylabel('Number of passengers')
    plt.xlabel(var)
    plt.title(var)
    plt.show()


# In[ ]:


for col in vars_num:
    analyse_continuous(data, col)


# ## Separate data into train and test
# 
# Use the code below for reproducibility. Don't change it.

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape


# In[ ]:


X_train.head()


# In[ ]:


Y_train.value_counts()


# In[ ]:


Y_train[:5]


# In[ ]:


X_train[vars_cat].head()


# In[ ]:


vars_num =['age','sibsp','parch','fare']


# In[ ]:


X_train[vars_num].head()


# ### Fill in Missing data in numerical variables:
# 
# - Add a binary missing indicator
# - Fill NA in original variable with the median
# ### Replace Missing data in categorical variables with the string **Missing**

# In[ ]:


#for numerical values
X_train[vars_num].isna().sum()


# In[ ]:


for var in vars_num:
    X_train[var].fillna(data[var].median(),inplace=True)
    X_test[var].fillna(data[var].median(),inplace=True)


# In[ ]:


X_train[vars_num].isna().sum()


# In[ ]:


#for categorical values
X_train[vars_cat].isna().sum()


# In[ ]:


X_train[vars_cat] = X_train[vars_cat].fillna('missing')
X_test[vars_cat] = X_test[vars_cat].fillna('missing')


# In[ ]:


X_train[vars_cat].isna().sum()


# In[ ]:


X_train.isna().sum()


# In[ ]:


X_test.isna().sum()


# ## Feature Engineering
# 
# ### Extract only the letter (and drop the number) from the variable Cabin

# In[ ]:


X_train.cabin.value_counts()


# In[ ]:


X_train['cabin'] = X_train['cabin'].apply(lambda x: 'missing' if x == 'missing' else x[0])
X_test['cabin'] = X_test['cabin'].apply(lambda x: 'missing' if x == 'missing' else x[0])


# In[ ]:


X_train.cabin.value_counts()


# In[ ]:


X_test.cabin.value_counts()


# In[ ]:





# ### Remove rare labels in categorical variables
# 
# - remove labels present in less than 5 % of the passengers

# In[ ]:


vars_cat


# In[ ]:


for col in vars_cat:
    print(25*'-'+col+'-'*25)
    print(X_train[col].value_counts())
    print(25*'.')
    print(X_test[col].value_counts())
    print(len(X_train[col].value_counts()))


# In[ ]:


temp = X_train.sex.value_counts() 
list(temp[temp > 200].index)


# In[ ]:


def find_frequent_labels(df, var, rare_perc):
    
    # function finds the labels that are shared by more than
    # a certain % of the houses in the dataset

    df = df.copy()
    temp = df[var].value_counts() / len(df)
    return list(temp[temp > rare_perc].index)



for var in vars_cat:
    
    # find the frequent categories
    frequent_ls = find_frequent_labels(X_train, var, 0.05)
    
    # replace rare categories by the string "Rare"
    X_train[var] = np.where(X_train[var].isin(frequent_ls), X_train[var], 'Rare')
    
    X_test[var] = np.where(X_test[var].isin(frequent_ls), X_test[var], 'Rare')


# In[ ]:


for col in vars_cat:
    print(25*'-'+col+'-'*25)
    print(X_train[col].value_counts())
    print(25*'.')
    print(X_test[col].value_counts())
    print(len(X_train[col].value_counts()))


# In[ ]:





# ### Perform one hot encoding of categorical variables into k-1 binary variables
# 
# - k-1, means that if the variable contains 9 different categories, we create 8 different binary variables
# - Remember to drop the original categorical variable (the one with the strings) after the encoding

# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


X_train = pd.get_dummies(X_train, columns=vars_cat, drop_first=True)
X_test = pd.get_dummies(X_test, columns=vars_cat, drop_first=True)


# In[ ]:


X_train.columns


# In[ ]:


X_test.columns


# In[ ]:


X_train.head()


# In[ ]:


X_train.drop(columns=['embarked_Rare'],inplace=True)


# In[ ]:


len(X_train.columns)


# In[ ]:


len(X_test.columns)


# ### Scale the variables
# 
# - Use the standard scaler from Scikit-learn

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


X_train.head()


# In[ ]:


X_train = pd.DataFrame(scaler.fit_transform(X_train),columns = X_train.columns)
X_train.head()


# In[ ]:


X_test = pd.DataFrame(scaler.fit_transform(X_test),columns = X_test.columns)
X_test.head()


# ## Train the Logistic Regression model
# 
# - Set the regularization parameter to 0.0005
# - Set the seed to 0

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score


# In[ ]:


logreg = LogisticRegression(C=0.05,n_jobs=-1 ,random_state=0)
logreg.fit(X_train, Y_train)


# ## Make predictions and evaluate model performance
# 
# Determine:
# - roc-auc
# - accuracy
# 
# **Important, remember that to determine the accuracy, you need the outcome 0, 1, referring to survived or not. But to determine the roc-auc you need the probability of survival.**

# In[ ]:


Y_pred = logreg.predict(X_test)


# In[ ]:


print(confusion_matrix(Y_test,Y_pred))


# In[ ]:


print(accuracy_score(Y_test,Y_pred))


# In[ ]:


print(roc_auc_score(Y_test,logreg.predict_proba(X_test)[:,1]))


# That's it! Well done
# 
# **Do Upvote if you really liked the kernel!!**

# In[ ]:




