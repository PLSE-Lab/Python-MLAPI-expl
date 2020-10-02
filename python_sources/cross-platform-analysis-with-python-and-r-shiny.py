#!/usr/bin/env python
# coding: utf-8

# # Introduction to problem solving for data
# This kernel hopes to outline techniques to creatively fill missing values, transform data for statistical analysis, blend technologies, and deliver a model for estimating results.
# 
# ## Document Outline
# 1. Import Python packages needed and data
# 2. Review the data and identify missing data
# 3. Clean key data by filling estimates for missing data
# 4. Extract modified training data and visualize in shinyapps.io
# 5. Build on insights from visuals to train a basic machine learning model
# 7. Test data and submit results

# ## Import Python packages needed and data

# In[ ]:


# import packages needed as a separate cell. It makes it easy to add and rerun in the future as you progress through the project.

import pandas as pd
import numpy as np
from sklearn import tree
import graphviz


# In[ ]:


# importing data sets from kernel 

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## Review the data and identify missing data

# In[ ]:


# displaying shape of data and column names
# as you can see the test data is missing the survival rate, which is what we need to eventually estimate on the test data

print("Train Shape:   " + str(train.shape))
print("Test Shape:    " + str(test.shape))
print("Train Columns: " + str(train.columns))
print("Test Columns:  " + str(test.columns))


# In[ ]:


# displaying the first five rows of the training data to see what is in each column

train.head()


# In[ ]:


# Pandas loc operator allows you to build a boolean script to see a filtered list from the data. 
# Here I am asking for a view of the training data where the Age column contains a null value (missing value). 

# In addition, I am using the sort_values modifier to sort by the family relationship attributes SibSp and Parch to see if these are useful in possibly filling the values 

train.loc[train['Age'].isna()].sort_values(['SibSp', 'Parch'],ascending=[False, False]).head()


# This is important, since we have all heard of the women and children first approach to catastrophes. By this nature, we should expect our data to reflect a bias in survival rates towards the young and female.
# We will verify this later, but for now we just want to see how impactful these missing values could be.Clearly there are a lot of missing ages which could be helpful in determining survival rate. 
# 
# The family relationship statistics do not appear to be all that useful to me. Instead we can use the name field for evaluating average age by name prefix (Miss, Mrs, Mr, Master). There are some others like Dr, and ms. and miss seem to be interchangable. 

# In[ ]:


ages = train.loc[~train['Age'].isna()].append(test.loc[~test['Age'].isna()], sort=False)
ages = ages[['Name','Age', 'Sex']]
ages.head()


# ## Clean key data by filling estimates for missing data

# With the name, age and sex parsed out, we will use a custom function of nested if statements to group the passengers into four distinct groups: adult male, adult female, boy, and girl. We can then evaluate each subgroups average age to get closer average for each missing value.
# 
# Using an apply method like this is not ideal for large data sets, but here it is a small enough list of values the performance is not a concern and it allows for a clean view of the logic.

# In[ ]:


def groupby_age_sex(row):
    if 'Mr.' in row['Name']:
        return 0
    elif 'Master.' in row['Name']:
        return 1
    elif 'Mrs.' in row['Name']:
        return 2
    elif 'Miss.' in row['Name']:
        return 3
    elif 'Ms.' in row['Name']:
        return 3
    elif 'Dr.' in row['Name']:
        if row['Sex'] == 'Male':
            return 0
        else:
            return 2
    

ages['group'] = ages.apply(lambda x: groupby_age_sex(x), axis=1)
ages.head()


# In[ ]:


ages = ages.groupby('group').agg({'Age':'mean'})
ages.rename(columns = {'Age': 'Avg Age'}, inplace=True)
ages.head()


# Applying averages by age group to test and train data

# In[ ]:


#add group to test and train
test['group'] = test.apply(lambda x: groupby_age_sex(x), axis=1)
train['group'] = train.apply(lambda x: groupby_age_sex(x), axis=1)

#set index for easy join
test.set_index('group', inplace=True)
train.set_index('group', inplace=True)

#join to averages to add column Avg Age
test = test.join(ages, how='left')
train = train.join(ages, how='left')

#fill in avg age where age is NaN
test.loc[test['Age'].isna(), 'Age'] = test.loc[test['Age'].isna()]['Avg Age']
test.drop('Avg Age', axis=1, inplace=True)
test.head()


# In[ ]:


train.loc[train['Age'].isna(), 'Age'] = train.loc[train['Age'].isna()]['Avg Age']
train.drop('Avg Age', axis=1, inplace=True)
train.head()


# ## Extract modified training data and visualize in shinyapps.io

# After this point the modified training data was extracted and imported into R for use in an R Shiny dashboard. Please see the published R Shinny app here for visual investigations of the data:
# 
# [https://ian-stone30.shinyapps.io/Titanic_Project/]( https://ian-stone30.shinyapps.io/Titanic_Project/)
# 
# By moving the visuals to an interactive graph, we can quickly compare many different items at once without extra coding. As you will see from the home page, all data and code is available for download directly from the application itself. 
# 
# Please review and come back to this kernel for the next steps.

# ## Build on insights from visuals to train a basic machine learning model
# 
# ### Decision Trees
# 
# Decision Trees are easy to explain what is going on. Before investigating many other types of classification models, starting with one that almost emulates how people solve problems is a good start. The decision tree will investigate the attributes in our data and set decision points to get us to a logical classification of if our passenger survived or died in the Titanic crash based on the attributes we have available.

# In[ ]:


# removing our group values to avoid introducing new variables that are summaries of current variables. Also dropping Name, PassengerId, SibSp, Parch and Ticket 
# as they no longer hold value worth based on analysis in the Shiny app.

train = train.fillna(0)
test = test.fillna(0)

train.reset_index(inplace=True)
train_target = train['Survived'].copy()
train_results = train[['PassengerId', 'Survived']].copy() # preparing to check accuracy
train.drop(['group', 'Name', 'Ticket', 'PassengerId', 'SibSp', 'Parch', 'Survived'], axis=1, inplace=True)

train.head()


# In[ ]:


test.reset_index(inplace=True)
test_results = test[['PassengerId']].copy() #preparing for output
test.drop(['group', 'Name', 'Ticket', 'PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)

test.head()


# ### Cleaning Data
# 
# Next we need to make everything numeric. Here are a few other ways to handle this that are different than the grouping done above.

# In[ ]:


def build_mapping(arr):
    arr = sorted(set(arr)) #get unique values
    i=0
    
    for ea in arr:
        i += 1
        try:
            item_map.update({ea:i})
        except:
            item_map = {ea:i}
    
    return item_map

#setting mapping fields all to strings for matching purposes - ensuring everything is mapped correctly

train['Sex'] = train['Sex'].astype(str)
train['Embarked'] = train['Embarked'].astype(str)
train['Cabin'] = train['Cabin'].astype(str)

test['Sex'] = test['Sex'].astype(str)
test['Embarked'] = test['Embarked'].astype(str)
test['Cabin'] = test['Cabin'].astype(str)

map_sex = build_mapping(train['Sex']) #no need to combine, there are only two choices and both represented in train data
map_sex


# In[ ]:


map_embarked = build_mapping(train['Embarked'].astype(str)) #no need to combine, there are only three choices and both represented in train data
map_embarked


# In[ ]:


#need to combine and organize test and train data since unique values are in each. The null is probably the highest value item (now 0 to avoid it being dropped from results)
#but alphabatizing cabin should also help map it to ship location if they were logically created on the ship


cabin_list = train['Cabin'].append(test['Cabin'])
cabin_list.sort_values(inplace=True)
cabin_list


# In[ ]:


map_cabin = build_mapping(cabin_list)
map_cabin


# In[ ]:


train['Sex'] = train['Sex'].map(map_sex)
train['Cabin'] = train['Cabin'].map(map_cabin)
train['Embarked'] = train['Embarked'].map(map_embarked)

train.head(25)


# In[ ]:


test['Sex'] = test['Sex'].map(map_sex)
test['Cabin'] = test['Cabin'].map(map_cabin)
test['Embarked'] = test['Embarked'].map(map_embarked)

test.head(25)


# In[ ]:


dt = tree.DecisionTreeClassifier(min_samples_split=30)
dt


# In[ ]:


dt = dt.fit(train, train_target)
dt


# In[ ]:


dot_data = tree.export_graphviz(dt, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Titanic") 


# In[ ]:


dot_data = tree.export_graphviz(dt, out_file=None, 
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


train_results['Survived Estimate'] = dt.predict(train)
train_results


# In[ ]:


tot = len(train_results.index)
matches = len(train_results.loc[train_results['Survived'] == train_results['Survived Estimate']].index)
print("Successfully guessed " + str(matches) + " out of " + str(tot) + " total records.   " + str(round((matches/tot) * 100, 2)) + "%")


# ## Test Data and Submit Results

# In[ ]:


test_results['Survived'] = dt.predict(test)
test_results.set_index('PassengerId', inplace=True)
test_results.to_csv("results.csv")
test_results


# # Conclusion
# 
# For a reasonable effort in data cleansing and organizing, the training model accuracy of over 86% is very good for a decision tree with a data set this size. Based on this, we can confidently say we succeeded in a first run Kaggle competition. Hopefully some of the items here can help other new-Kagglers!
