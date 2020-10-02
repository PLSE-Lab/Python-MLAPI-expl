#!/usr/bin/env python
# coding: utf-8

# The Goal of this notebook is to create a simple Handout for those who are getting started with Data Science
# 
# Target audience: Non-computer science background. 
# 
# Preferred reader: With Descriptive Statistics background
# 
# Needed Background:
# - have done some programming
# - some experience with python and pandas
# 
# 
# This is a work-in-progress

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


# # Gather the Dataset
# - fortunately, kaggle provides it in golden platter

# In[ ]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv').set_index('PassengerId')
test_df = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')


# Let's try to explore first what the dataset contains...

# In[ ]:


train_df.head()


# ## Our Goal 
# It is to predict who survived or not. So how can we do it? Here are some possible options:
# 1. explore which features are linearly correlated with survived or not survived.
# 2. In corolarry, explore which features have pairwise correlation
# 
# Indeed, we want to analyze immediately. But, Don't forget to clean the data!

# # Data Cleaning 
# 
# In this step, we look for the ff problems:
# - Does any variables contain missing values?
# - In text, does it contain typographical errors
# 
# Aferwards, we correct if possible. 

# In[ ]:


train_df.isna().mean().plot(kind='barh')


# We see that the Age and Cabin variables have a lot of missing values. But the difference is that age has a `missing_rate` of <20% while cabin is around 77%. 
# 
# My rule of thumb is 20%. If it is greater than 20%, then discard it. If it is less than 20%, then we can do some interpolation

# In[ ]:


fill_mean = train_df[['Age','Fare']].mean()

train_df[['Age','Fare']] = train_df[['Age','Fare']].fillna(fill_mean)
test_df[['Age','Fare']] = test_df[['Age','Fare']].fillna(fill_mean)


# In[ ]:


test_df.isna().mean()


# ## Sex
# Some questions we want to answer:
# - Do we have typos?
# - Do we other genders?

# In[ ]:


train_df['Sex'].unique()


# Another Question: How's survival on the male vs female?

# In[ ]:


train_df.reset_index().groupby(['Sex','Survived']).count()['PassengerId']


# Interestingly, there is strong odds for female survivors compared to male. Let's try to get in percentage

# In[ ]:


train_df.reset_index().groupby(['Sex']).mean()['Survived']


# As we can see, a female had 75% survival rate compared to 19% survival rate of male.
# 
# Thus, without question, **Sex has strong correlation with survival**. It sounds reasonable because, as a male, we want to prioritize female and children.Moreover, I can remember in the Titanic moview, they prioritized women and children first.

# ## Age
# - Now that we've learned that genders have strong correlation. Let's try to look at other columns/variables
# - In this case, we check the age.

# In[ ]:


train_df['Age'].plot(kind='hist')


# In[ ]:


_ ,outbins = pd.cut(train_df['Age'], 10, retbins=True)
outbins[0] = 0 #Fix the edge case
train_df['AgeGroup'] = pd.cut(train_df['Age'],outbins)
test_df['AgeGroup'] = pd.cut(test_df['Age'],outbins)


# check if there are edge cases

# In[ ]:


test_df.loc[test_df['AgeGroup'].isna()]


# This plot shows the distribution of Survived over different age group

# In[ ]:


pd.crosstab(train_df['AgeGroup'], train_df['Survived'])


# In[ ]:


train_df.groupby('AgeGroup')['Survived'].mean().plot(kind='bar')


# Now, by looking at the graph, it seems that younger people have survived more than old ones, It's not as clear as Sex. But it still seems a good indicator.

# ## Passenger Class
# In this section, we want to explore the Passenger class
# 
# Our initial Hypothesis: Higher class must mean higher chance of survival

# The table below shows how many passengers we have per Pclass

# In[ ]:


train_df[['Pclass','Survived']].groupby('Pclass').count()


# Number of passengers who survived

# In[ ]:


train_df[['Pclass','Survived']].groupby('Pclass').sum()


# We now want to get ratio of number of survival compared to total passengers per class
# 
# It is shown below

# In[ ]:


train_df[['Pclass','Survived']].groupby('Pclass').sum() / train_df[['Pclass','Survived']].groupby('Pclass').count()


# Just by comparing the rate of survival across Pclass, We can now see that it is greatly biased towards Pclass 1 in descending order

# ## Multi-factor analysis (multi-variate)
# 
# Let's now check some combination, maybe some combination of a categories helps us understand more whether a passenger would significantly survive or not

# Let's now look how our data is distributed across our three factors and target outcome

# In[ ]:


pd.crosstab([train_df['Pclass'], train_df['Sex'],train_df['Survived']],train_df['AgeGroup']) # train_df.groupby(['Pclass','Sex','Survived']).count()


# Insights:
# 1. One thing we notice now is that there a strong bias with (Female and Survived) and (Male and Not Survived) across different age group.
# 2. If we look at the count on children and near senior citizen age. It seems that their count is too low. This means a good model should try to remove them first so that we can focus on the middle age group. (TODO)
# 3. If we observe now the middle age group, it seemingly showcases that many of our passengers are of middle age group. Although it might be true, but we need to recall that this might be amplified using filling missing values with mean. We need to recall we pump up the middle with 20% by filling it in with mean value. Is this good or not? Well, in term of distributions, it skews it. But the benefit of using mean is that it has minimal biasing effect on training the Logistic Regression Model.
# 
# It would be great if we can summarize the table above by using survival rate as shown below.

# In[ ]:


train_df.pivot_table(index=['Pclass','Sex'], columns='AgeGroup', values='Survived', aggfunc='mean')


# Some Insights:
# 1. We see now that clearly Pclass and Male has a stronger effect compared to their individual contribution. It would be great if we can help our ML model by making it as a separate variable rather than hoping that the model can see this pattern(TODO)
# 2. Given our previous insight on the count table, we  clearly see the effect of not isolating the children and senior citizen as a separate thing. They create an illusion of that the model is great on this age while in fact it is inconclusive due to their low count
# 

# ## Pick Columns now that are Cleaned Data

# In[ ]:


col_interest = ['Sex', 'AgeGroup', 'Pclass']
target = 'Survived'
x = train_df[col_interest]
y = train_df[target]

final_test_df = test_df[col_interest]


# # Model Training
# - at this point, we learn that Sex has strong correlation. Let's try to train a model.

# In[ ]:


# We import first the modules we would need in this section
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Again, we learn that the best way to know if you're doing great is not to predict the test dataset directly.
# 
# Why?
# - #1: You don't have the actual label of the test dataset.
# - #2 Even if you have one, if you keep on testing your code on test data, as a human being, you will be able to capture the pattern the more experiements you try. Thus invalidating the point of using test data
# 
# 
# Thus, in our train set, we want to split it into train / val set.

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.10,random_state=42)


# Now the challenge in pluggin the x data directly is that it is a categorical variable. Unfortunately, categorical variables cannot be read by logistic regression. It can only read numerical variables. Thus, what we need to do now is: Can we convert categorical variables into numerical variables?
# 
# Okay so, what are the options?
# 
# 
# You have two. Intuitively, the easiest solution is shown below
# 
# | Sex | Numerical Equivalent |
# |----------| -------------------- |
# | Male     | 0                    |
# | Female   | 1                    |
# | Gay      | 2                    |
# | Lesbian  | 3                    |
# | Queer    | 4                    |
# | Trans    | 5                    |
# 
# This is called **label encoding.**
# 
# Although it was able to convert it into numerical number, the problem with this approach is that you intrinsically create an "order". Think about it this way. If we imagine that 0 is the highest rank, does it mean the Female is a lower rank than male?
# So let's put it the other way around, suppose higher number means higher rank, does it mean that trans is higher than male?
# 
# Rationale: You might say that these are simply numbers. The challenge here is that recall that an ML model is simply an optimization game. Recall that the whole point of ML model is to get the minima of an error function. Thus, it means that at the neighbouring value of a point is critical in getting the right optimization. Equivalently, the neighbours of male category has a strong effect in determining if it's male or not.
# 
# So another method on converting it into numerical value is shown below
# 
# | Sex      | Sex_Male | Sex_Female | Sex_Gay | Sex_Lesb | Sex_Quee | Sex_Trans |
# |----------| -------- | ---------- | ------- | -------- | -------- | --------- |
# | Male     | 1        | 0          | 0       | 0        | 0        | 0         |
# | Female   | 0        | 1          | 0       | 0        | 0        | 0         |
# | Gay      | 0        | 0          | 1       | 0        | 0        | 0         |
# | Lesbian  | 0        | 0          | 0       | 1        | 0        | 0         |
# | Queer    | 0        | 0          | 0       | 0        | 1        | 0         |
# | Trans    | 0        | 0          | 0       | 0        | 0        | 1         |
# 
# This method is **One-Hot Encoding**
# 
# What we've done here is that we create more 6 columns as we have six categories. Each category has its own dedicated column so that we can remove the problem of "instrinsic order".
# 
# Although it has its own trade-off. In our case, we already good as it is. Let's use it now.

# In[ ]:


#In OOP language, We need to instantiate an object of OneHotEncoder first
#If you are not familiar with OOP, think of it as a way to create a type of OneHotEncoder with certain parameters.  
encoder = OneHotEncoder()

#Understand which of these are categories and how it would be transformed into multiple columns
encoder.fit(x_train)

x_train_enc = encoder.transform(x_train)
x_val_enc = encoder.transform(x_val)
x_test_enc = encoder.transform(final_test_df)


# In[ ]:


# Similar idea with OneHotEncoder, we want to create a Logistic Regression of our own flavor
estimator = LogisticRegression(C=1.0,class_weight='balanced', solver='lbfgs')

# The meaning comes from the idea of "best fit" line. So, we're actually training the model to get the best parameters that fits best to our data points
estimator.fit(x_train_enc,y_train)


# In[ ]:


estimator.score(x_val_enc,y_val)


# # Kaggle Submission
# This time let's try to submit it on kaggle

# In[ ]:


test_df['Survived'] =estimator.predict(x_test_enc)
test_df.head()


# In[ ]:


test_df.reset_index()[['PassengerId','Survived']].to_csv('../working/submit.csv', index=False)

