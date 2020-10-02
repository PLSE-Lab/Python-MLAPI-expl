#!/usr/bin/env python
# coding: utf-8

# # 1. Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Read in the data, and combine data for cleaning

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# Combine the data for cleaning and estimation of missing values

# In[ ]:


combined = pd.concat([ train, test ])


# In[ ]:


combined.describe()


# # 3. Exploratory Data Analysis
# 
# ## Missing Data
# 
# We use seaborn to create a simple heatmap to see where we are missing data

# In[ ]:


sns.heatmap(combined.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Roughly 20 percent of the Age data is missing. We can impute missing values based on the average.
# 
# 

# Explore the data to understand relationships between variables and survival

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', data=combined, palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=combined, palette='RdBu_r')


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=combined, palette='rainbow')


# In[ ]:


sns.distplot(combined['Age'].dropna(), kde=False, color='darkred', bins=30)


# In[ ]:


sns.countplot(x='SibSp',data=combined)


# In[ ]:


combined['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[ ]:


sns.countplot(x='Embarked',data=combined)


# ___
# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class. For example:
# 

# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=combined,palette='winter')


# We can see that wealthier passengers in the higher classes tend to be older. Use these average age values to impute based on Pclass for Age.

# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]   
          
    if pd.isnull(Age): 
        
        if Pclass == 1:            
            return np.mean(combined[combined['Pclass'] == 1 ]['Age'])

        elif Pclass == 2:
            return np.mean(combined[combined['Pclass'] == 2 ]['Age'])

        else:
            return np.mean(combined[combined['Pclass'] == 3 ]['Age'])

    else:
        return Age


# In[ ]:


combined['Age'] = combined[['Age','Pclass']].apply(impute_age, axis=1)


# Drop the Cabin column

# In[ ]:


combined.drop('Cabin', axis=1, inplace=True)


# Fill the row in Embarked that is NaN with 'S' (most common port), and the row in Fare with mean

# In[ ]:


combined.fillna(value={'Embarked': 'S', 'Fare': np.mean(combined['Fare'])}, inplace=True)


# ## Converting Categorical Features 
# 
# Convert categorical features to dummy variables using pandas. Otherwise the machine learning algorithm won't be able to directly take those features as inputs.

# In[ ]:


combined.info()


# In[ ]:


sex = pd.get_dummies(combined['Sex'], drop_first=True)
embark = pd.get_dummies(combined['Embarked'], drop_first=True)


# In[ ]:


combined.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)


# In[ ]:


combined = pd.concat([combined,sex,embark], axis=1)


# In[ ]:


combined.head()


# In[ ]:


sns.heatmap(combined.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# # 4. Build Logistic Regression model
# 
# Split combined data back into the training set and test set.
# Then do train/test split on the training set in order to assess model performance.
# 
# ## Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train = combined[combined['Survived'].notnull()]
test = combined[combined['Survived'].isnull()]
test = test.drop('Survived', axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'],axis=1), 
                                                    train['Survived'], test_size=0.30)


# ## Training and Predicting

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test).astype(int)


# # 5. Evaluation

# Check the precision, recall, f1-score using classification report

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# ___
# # 6. Creating Submission File

# In[ ]:


# Check for any remaining missing values
print("Remaining NaN?", np.any(np.isnan(test)) )
#np.all(np.isfinite(test))


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = logmodel.predict(test.drop('PassengerId', axis=1)).astype(int)

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

