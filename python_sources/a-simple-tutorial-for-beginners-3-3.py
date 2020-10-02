#!/usr/bin/env python
# coding: utf-8

# # A simple tutorial for Beginners
# 
# This notebook is a version that adds feature enginnereing part to the previous version.  
# Please refer to the [previous notebook](https://www.kaggle.com/hs1214lee/a-simple-tutorial-for-beginners-2-3).  
# 

# # Read data
# Same as previous version.

# In[ ]:


import os
import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
print('Shape of train dataset: {}'.format(train.shape))
print('Shape of test dataset: {}'.format(test.shape))


# # Merge data

# In[ ]:


train['Type'] = 'train'
test['Type'] = 'test'
all = pd.concat([train, test], sort=False).reset_index(drop=True)
print('Shape of all dataset: {}'.format(all.shape))


# # Check for missing values
# Same as previous version.

# In[ ]:


print(all.isnull().values.any())
train.isnull().sum()


# There are missing values in the 'Age', 'Cabin' and 'Embarked' features.  
# We have to fill these values with reference to other values.

# In[ ]:


# Fill missing values
all_corr = all.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
all_corr


# In[ ]:


all.info()


# ## Fill missing values - Age
# Age is float64 date.  
# Let's look at the age-related values.  

# In[ ]:


all_corr[all_corr['level_0'] == 'Age']


# Highly associated with Pclass and SibSp.  
# The Age's missing value is filled with the average value of other values having the same value.(Pclass and SibSp)

# In[ ]:


all['Age'] = all.groupby(['Pclass', 'SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))


# ## Fill missing values - Fare
# Fare is float64 data too.  
# The same method as 'Age' is applied.

# In[ ]:


print(all_corr[all_corr['level_0'] == 'Fare'])
all['Fare'] = all.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].apply(lambda x: x.fillna(x.median()))


# ## Fill missing values - Cabin
# Cabin is object(String) data.  
# Let's see what values it has.

# In[ ]:


all['Cabin'].value_counts()


# We'll see it later, we'll use only the letters in front.  
# Cabin has too many missing values.  
# So fill missing values with unused characters(I used 'N').

# In[ ]:


all['Cabin'] = all['Cabin'].fillna('N')


# ## Fill missing values - Embarked
# Embarked is object(String) data.  
# Embarked has only two missing values.  
# As with the previous version, let's fill it with the mode value.

# In[ ]:


print(all['Embarked'].value_counts())
all['Embarked'] = all['Embarked'].fillna('S')


# In[ ]:


# Check missing values again
all.isnull().values.any()


# # Create new feature
# We can create a new feature using different values.  
#   
# ## Create new feature - Name -> Title, IsMarried, Family
# Name has a lot of data, so it's hard to use.  
# Let's extract only Title(Mr, Miss, ...) and Family.  
# And through Title we can find a married woman.

# In[ ]:


import numpy as np
import string

def extract_surname(data):  
    families = []
    
    for i in range(len(data)):        
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]
        
        for c in string.punctuation:
            family = family.replace(c, '').strip()
            
        families.append(family)
            
    return families

print(all['Name'].value_counts())
all['Title'] = all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
all['Family'] = extract_surname(all['Name'])
all['IsMarried'] = np.where(all['Title'] == 'Mrs', 1, 0)
all.drop(['Name'], inplace=True, axis=1)


# In[ ]:


all['Title']


# In[ ]:


all['Family']


# ## Create new feature - SibSp, Parch -> FamilySize

# In[ ]:


all['FamilySize'] = all['SibSp'] + all['Parch'] + 1


# ## Create new feature - FamilySize -> FamilySizeGrouped

# In[ ]:


family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
all['FamilySizeGrouped'] = all['FamilySize'].map(family_map)


# ## Create new feature - Ticket -> Ticket_count

# In[ ]:


all['Ticket_count'] = all.Ticket.apply(lambda x: all[all['Ticket']==x].shape[0])


# ## Categorical data to numerical data
# Categorical data cannot be used for training.  
# So we use it by changing to a numeric type.

# In[ ]:


all.info()


# Name, Sex, Ticket, Cabin, Embarked, FamilySizeGroup are object(categorical data).  
# 'Type' is a feature that will be used later to separate the data. 

# ## Categorical data to numerical data - Sex
# Sex can only be used by male and female.
# But let's apply one assumption.  
# Children and female are more likely to live. (It appears in the movie. :D)  
# Considering this, the boy can be separated from male.

# In[ ]:


all.loc[(all['Sex'] == 'male') & (all['Age'] > 18.0), 'Sex'] = 0  # male
all.loc[(all['Sex'] == 'male') & (all['Age'] <= 18.0), 'Sex'] = 1 # boy
all.loc[all['Sex'] == 'female', 'Sex'] = 2                        # female


# ## Categorical data to numerical data - Ticket
# Tickets have a lot of data, so it's hard to use them.  
# Let's apply the assumption here that children and female are more likely to live too.  
# Pick a list of people who Children or female but dead, dead male,  
# and find their tickets.  

# In[ ]:


list1 = all[(all['Title'] != 'Mr') & (all['Survived'] == 0) ]['Ticket'].tolist() # Female and child no survide.
list2 = all[(all['Title'] == 'Mr') & (all['Survived'] == 1) ]['Ticket'].tolist() # Man survive.
all['Ticket_with_FC_dead'] = 0
all['Ticket_with_M_alive'] = 0
all.loc[all['Ticket'].isin(list1), 'Ticket_with_FC_dead'] = 1
all.loc[all['Ticket'].isin(list2), 'Ticket_with_M_alive'] = 1
all.drop(['Ticket'], inplace=True, axis=1)


# ## Categorical data to numerical data - Cabin
# 
# 'Cabin' indicating the crew of a room.
# As mentioned earlier, use only the first letter.

# In[ ]:


all['Cabin'].value_counts()


# In[ ]:


all['Deck'] = all['Cabin'].apply(lambda x: x[0])
all['Deck'] = all['Deck'].replace(['T'], 'A')
all.drop(['Cabin'], inplace=True, axis=1)


# In[ ]:


all.loc[all['Deck'] == 'A', 'Deck'] = 0
all.loc[all['Deck'] == 'B', 'Deck'] = 1
all.loc[all['Deck'] == 'C', 'Deck'] = 2
all.loc[all['Deck'] == 'D', 'Deck'] = 3
all.loc[all['Deck'] == 'E', 'Deck'] = 4
all.loc[all['Deck'] == 'F', 'Deck'] = 5
all.loc[all['Deck'] == 'G', 'Deck'] = 6
all.loc[all['Deck'] == 'N', 'Deck'] = 7


# ## Categorical data to numerical data - Embarked
# Embarked is the same as the previous version.

# In[ ]:


all.loc[all['Embarked'] == 'S', 'Embarked'] = 0
all.loc[all['Embarked'] == 'Q', 'Embarked'] = 1
all.loc[all['Embarked'] == 'C', 'Embarked'] = 2


# ## Categorical data to numerical data - Title
# There are too many types so we simplify it a bit.

# In[ ]:


all['Title'].value_counts()


# In[ ]:


all['Title'] = all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
all['Title'] = all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')
all['Title'].value_counts()


# In[ ]:


all.loc[all['Title'] == 'Mr', 'Title'] = 0
all.loc[all['Title'] == 'Miss/Mrs/Ms', 'Title'] = 1
all.loc[all['Title'] == 'Master', 'Title'] = 2
all.loc[all['Title'] == 'Dr/Military/Noble/Clergy', 'Title'] = 3


# ## Categorical data to numerical data - Family
# The family will have a similar pattern because they gather together.
# The same idea applies to tickets.

# In[ ]:


list1 = all[(all['Title'] != 0) & (all['Survived'] == 0) ]['Family'].tolist() # Female and child no survide.
list2 = all[(all['Title'] == 0) & (all['Survived'] == 1) ]['Family'].tolist() # Man survive.
all['Family_with_FC_dead'] = 0
all['Family_with_M_alive'] = 0
all.loc[all['Family'].isin(list1), 'Family_with_FC_dead'] = 1
all.loc[all['Family'].isin(list2), 'Family_with_M_alive'] = 1
all.drop(['Family'], inplace=True, axis=1)


# ## Categorical data to numerical data - FamilySizeGrouped
# Since it is simple data, it just mapping to numbers.

# In[ ]:


all.loc[all['FamilySizeGrouped'] == 'Alone', 'FamilySizeGrouped'] = 0
all.loc[all['FamilySizeGrouped'] == 'Small', 'FamilySizeGrouped'] = 1
all.loc[all['FamilySizeGrouped'] == 'Medium', 'FamilySizeGrouped'] = 2
all.loc[all['FamilySizeGrouped'] == 'Large', 'FamilySizeGrouped'] = 3


# # Separate the combined data again

# In[ ]:


train = all.loc[all['Type'] == 'train']
train.drop(['Type'], inplace=True, axis=1)
test = all.loc[all['Type'] == 'test']
test.drop(['Type', 'Survived'], inplace=True, axis=1)
print('Info of train dataset: {}'.format(train.info()))
print('Info of test dataset: {}'.format(test.info()))


# # Training
# Same as previous version.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

train_y = train['Survived']
train_x = train.drop(['Survived', 'PassengerId'], axis=1)
model = LogisticRegression()
model.fit(train_x, train_y)
pred = model.predict(train_x)
metrics.accuracy_score(pred, train_y)


# The resulting score is 0.9753.  
# Performance improved by 21% compared to the previous results(0.8002).  
# Now let's predict the test data and save it as a csv file.

# In[ ]:


import time

test_x = test.drop('PassengerId', axis=1)
timestamp = int(round(time.time() * 1000))
pred = model.predict(test_x)
output = pd.DataFrame({"PassengerId":test.PassengerId , "Survived" : pred})
output = output.astype(int)
output.to_csv("submission_" + str(timestamp) + ".csv",index = False)


# Submit the results.  
# The public score(for test dataset) is 0.81818.  
# The test data results also improved by 8% compared to the previous one(0.7536).  
# You can improve your results with this task.  
#   
# There are still many factors that can be improved.  
# Improve your model by referring to other notebooks.
