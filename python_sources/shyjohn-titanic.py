#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns


# # Step 1: Import the Data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')  # train set
test_df  = pd.read_csv('../input/test.csv')   # test  set
combine  = train_df + test_df


# # Step 2: Describe the Dataframes

# In the following, we find the features of data from the dataframes. The data dictionary is stored in https://www.kaggle.com/c/titanic/data. 

# In[ ]:


train_df.describe(include = 'all')


# In[ ]:


train_df.head()


# In[ ]:


test_df.describe(include = 'all')


# In[ ]:


test_df.head()


# ## Visualising the Variables ##
# Next we would like to visualise the results from the tables above. 

# In[ ]:


plt.title('Age distribution on Titanic',fontsize = 16)
plt.hist(train_df['Age'], bins=np.arange(train_df['Age'].min(), train_df['Age'].max()+1))
plt.show()


# In[ ]:


# code from https://www.kaggle.com/startupsci/titanic-data-science-solutions
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


# code from https://stackoverflow.com/questions/31029560/plotting-categorical-data-with-pandas-and-matplotlib
train_df['Sex'].value_counts().plot(kind='bar')


# In[ ]:


# plt.hist(train_df['Fare'], bins=np.arange(train_df['Fare'].min(), train_df['Fare'].max()+1))

plt.boxplot(train_df['Fare'])
plt.show()


# In[ ]:


plt.title('Fare and survival',fontsize = 16)
plt.scatter(train_df['Fare'],train_df['Survived'])
plt.show()


# In[ ]:


# Following code from https://stackoverflow.com/questions/8202605/matplotlib-scatterplot-colour-as-a-function-of-a-third-variable
# Function to map the colors as a list from the input list of x variables
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l == 1:
            cols.append('red')
        elif l == 2:
            cols.append('blue')
        else:
            cols.append('green')
    return cols
# Create the colors list using the function above
color_cols = pltcolor(train_df['Pclass'])

plt.scatter(x=train_df['Fare'],y=train_df['Survived'],s=20,c=color_cols) #Pass on the list created by the function here
plt.title('Fare and survival sorted by class',fontsize = 16)
plt.show()


# We could separate the variables of fare and survival into a grid plot. From below we can see that: 
# * There are more Class 1 passengers are more likely to be survived, while the converse for Class 3 passengers. 
# * Children and infants are more likely survive if they are in Class 1 and 2. 

# In[ ]:


# Following code from https://www.kaggle.com/startupsci/titanic-data-science-solutions
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.5, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.75, bins=20)
grid.add_legend();


# The 'Fare and Survival' plot poses an interesting feature, where there are people who have paid $0 on board. We need to see how many are they at below: 

# In[ ]:


# How many people paid $0 fare
(train_df.Fare == 0).sum()


# From above we could infer the following information: 
# * There are 1132 passengers that is used for this exercise, which is 50% of the passengers on board the Titanic. 
# * There are few elderly (age 65 to 80) on board. 
# * The mode of ages are around 15 to 35 and below 4. 
# * There are 15 people paid $0 on board, they could be treated as missing data or because they are children so paid no fare. 

# **Sidenote:** Is there any states labels that are aligned as a coordination? For example, age with the predicted survival rate or actually something that a person who have 2 values of and they are linked. 
# 
# *I don't think so as this dataset does not capture interactions*

# ## Missing Values
# Let us find out the missing values and see if we need to omit the data rows or substitute by average. 

# In[ ]:


total = train_df.isnull().sum().sort_values(ascending = False)
percent = round(train_df.isnull().sum().sort_values(ascending = False)/len(train_df)*100, 2)
pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])

# from 2b in https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic/notebook


# There are 687 missing cabin location data, which might be a problem for predicting lifes with their class. While there are 19.87% of age data is missing, so we can either: 
# * omit the data points (not really as it is about 20% of the data), or
# * use the average age, or
# * use another field to predict the age. In here we use **decision tree** to help us. 

# However, there are few missing data that are represented by '0' instead of Nan. For example, the fare is an important example. We can either: 
# * use the average fare, or
# * use another field to predict the age. In here we use **decision tree** to help us. 

# In[ ]:


# train_df[train_df.Age.isnull()]


# In[ ]:


total = test_df.isnull().sum().sort_values(ascending = False)
percent = round(test_df.isnull().sum().sort_values(ascending = False)/len(test_df)*100, 2)
pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])


# It is the same as above so as the conclusion. 

# ## Data Types ##
# Knowing the data type will help us when machine learning. For example, string data is not a favourite in machine learning and we may want to omit them. We could also put categorical data with dummy numbers. 

# Specificially by looking at the output below: 
# * The name contains their title and it could (possibly) validate their gender. 
# * Sex is a categorical data and we can convert them into dummy variables. 
# * Embarked is a categorical data and we can convert them into dummy variables. Since the causation is unintuitive, we could consider this later. 

# In[ ]:


# code from https://www.kaggle.com/startupsci/titanic-data-science-solutions
train_df.info()
print('_'*40)
test_df.info()


# ## Assumptions for Analysis
# The following are the assumptions to the situation so that any following analysis would work. 
# * Age is normally distributed due to the large sample pool (follows CLT). 
# * Missing values of age could be substituted. 

# # Step 3: Convert Data

# In this step we convert the sex data as a dummy variable for machine learning. First, we combine the training and validation data sets. 

# In[ ]:


combine = [train_df, test_df]


# In[ ]:


# code from https://www.kaggle.com/startupsci/titanic-data-science-solutions
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# However, there are missing values in Age. As what we have decided before, we are going to let decision tree to fill in the data. 

# In[ ]:


# code from https://www.kaggle.com/startupsci/titanic-data-science-solutions
# guess_ages is an external array which we will fill in the guessed age
guess_ages = np.zeros((2,3))
guess_ages


# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# Now we can see there are no missing values in age. 

# In[ ]:


total = train_df.isnull().sum().sort_values(ascending = False)
percent = round(train_df.isnull().sum().sort_values(ascending = False)/len(train_df)*100, 2)
pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])


# Now we can drop any redundant features for analysis. 

# In[ ]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin', 'Embarked'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Embarked'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

