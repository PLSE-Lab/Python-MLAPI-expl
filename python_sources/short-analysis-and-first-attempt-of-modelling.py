#!/usr/bin/env python
# coding: utf-8

# # Costa Rican Household Poverty Level Prediction 
# 

# This is an initial attempt to analyse the Costa Rican Household data. Please feel free to comment and give feedback.

# ## Loading and looking into the data

# In[ ]:


# import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sklearn


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.describe()


# In[ ]:


train.shape


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.describe()


# In[ ]:


test.shape


# ## Handling anomalous data

# First, let's look at the data having mixed types.

# In[ ]:


train.columns[train.dtypes=='object']


# Obviously, the 'Id' and 'idhogar' are identifiers of the data. So, let's move to the 'dependency' field and have a look at the unique values:

# In[ ]:


train['dependency'].unique()


# There are mainly numerical values, except for the 'yes' and 'no' String values. To deal with it, we can see what values has the 'SQBdependency' field which is the squared dependency.
# First, when dependency equals to 'yes' and then to 'no':

# In[ ]:


train.loc[(train['dependency']=='yes')]['SQBdependency'].unique()


# In[ ]:


train.loc[(train['dependency']=='no')]['SQBdependency'].unique()


# It is consistent, when it is 'yes', it equals to 1 and when it is 'no', it equals to 0. It is safe then to assign the respecting values to dependency:

# In[ ]:


train.loc[train['dependency']=='yes', 'dependency'] = 1
train.loc[train['dependency']=='no', 'dependency'] = 0
train['dependency'].unique()


# Converting the field to float is also useful at this stage.

# In[ ]:


train['dependency'] = train['dependency'].astype(float)


# Then comes the 'edjefe' field. Same here, we have 'yes' and 'no' among the numbers.

# In[ ]:


train['edjefe'].unique()


# We are told though in the data fields descriptions that 'yes' means 1 and 'no' 0 so we assign respectively.

# In[ ]:


train.loc[train['edjefe']=='yes', 'edjefe'] = 1
train.loc[train['edjefe']=='no', 'edjefe'] = 0


# Exactly the same happens for the 'edjefa' so we apply the same assignment and we finally convert to int type.

# In[ ]:


train['edjefa'].unique()


# In[ ]:


train.loc[train['edjefa']=='yes', 'edjefa'] = 1
train.loc[train['edjefa']=='no', 'edjefa'] = 0
train['edjefa'].unique()


# In[ ]:


train['edjefe'] = train['edjefe'].astype(int)
train['edjefa'] = train['edjefa'].astype(int)


# ## Missing data
# 
# At this point, we should check if there are missing values in our data.

# In[ ]:


train.isnull().sum().sort_values(ascending=False)[0:10]


# Five columns appear to have missing values, three of which have a high number.
# For convenience, this is the meaning of each of the variables:
# * rez_esc, Years behind in school
# * v18q1, number of tablets household owns
# * v2a1, Monthly rent payment
# * meaneduc,average years of education for adults (18+)
# * SQBmeaned, square of the mean years of education of adults (>=18) in the household

# ### rez_esc
# 
# Starting from the yeard behind in schooling, let's check the values:

# In[ ]:


train['rez_esc'].unique()


# As expected we have int values on the rest of the cases. 
# 
# Let's separate the people that have 'rez_esc' field missing and not missing and check their age distributions.

# In[ ]:


train_null = train.loc[train['rez_esc'].isnull()]
train_non_null = train.loc[train['rez_esc'].notnull()]


# In[ ]:


sns.distplot(train_null['age'], color='blue')
sns.distplot(train_non_null['age'], color='red')


# Yes, this is very interesting. It seems that people with years behind in school field missing are mainly pupils, people less than 19 years old.
# 
# It is safe then to simply fill those values with zero.

# In[ ]:


train['rez_esc'] = train['rez_esc'].fillna(0)


# ### v18q1

# The number of tablets the household owns seems to have nan instead of zero, as there are only positive values in the data. We fill nan with zero.

# In[ ]:


train['v18q1'].unique()


# In[ ]:


train['v18q1'] = train['v18q1'].fillna(0)


# ### v2a1
# 
# When the rent payment is missing, we assume that people own the house. We fill with 0 the missing values.

# In[ ]:


train['v2a1'] = train['v2a1'].fillna(0)


# ### meaneduc
# Just five rows appear to have missing years of education.

# In[ ]:


meaneduc_null = train.loc[train['meaneduc'].isnull()]
meaneduc_null


# In order to fill those values, we can just use the 'escolari' field, which gives us the years of schooling.

# In[ ]:


meaneduc_null[['Id', 'idhogar', 'escolari']]


# However, we are also given the a number of fields denoted as instlevel**x** that show the level of the education in a categorical way.

# In[ ]:


meaneduc_null[['Id', 'idhogar', 'escolari', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']]


# As we can see, there is information here about the level of education. 
# 
# As an alternative, we can simply get all the records from our dataset that have the respecting instlevel and get the mean of their meaneduc values. Then we can compare this value with the 'escolari' years, in order to make sure our assumption to use the 'escolari' years is correct.
# Starting from the case for the household with id='1b31fd159', we get all the rows with instlevel=4 (this is the level of education of this person) whose meaneduc is not null and we look at the distribution and the mean value below.

# In[ ]:


# for the household with id=1b31fd159
instlevel4_one = train.loc[(train['instlevel4']==1) & (train['meaneduc'].notnull())]
sns.distplot(instlevel4_one['meaneduc'])


# In[ ]:


# find mean
instlevel4_one['meaneduc'].mean()


# We cross check that mean value of meaneduc is almost equal to the escolari value (9.33303845601356 ~= 10). 
# 
# We apply the same logic to the rest two households, by getting the mean of the two instlevels in case we have different levels between the persons.

# In[ ]:


# for the household with id=a874b7ce7
instlevel2_one = train.loc[(train['instlevel2']==1) & (train['meaneduc'].notnull())]
instlevel3_one = train.loc[(train['instlevel3']==1) & (train['meaneduc'].notnull())]


# In[ ]:


(instlevel2_one['meaneduc'].mean() + instlevel3_one['meaneduc'].mean())/2


# In this case we have either 4 and 6 on escolari values, which is not far from what we found. Let's check the last household.

# In[ ]:


# for the household with id=faaebf71a
instlevel7_one = train.loc[(train['instlevel7']==1) & (train['meaneduc'].notnull())]


# In[ ]:


instlevel7_one['meaneduc'].mean()


# Again, 11.947~=12, so in general is safe to replace nan values with the escolari values. We also fill the SQBmeaned field with the squared value of escolari. 

# In[ ]:


# replace
train.loc[train['idhogar']=='faaebf71a', 'meaneduc'] = instlevel7_one['meaneduc'].mean()
train.loc[train['idhogar']=='faaebf71a', 'SQBmeaned'] = instlevel7_one['meaneduc'].mean()**2


# In[ ]:


# replace
train.loc[train['idhogar']=='1b31fd159', 'meaneduc'] = train.loc[train['idhogar']=='1b31fd159', 'escolari']
train.loc[train['idhogar']=='1b31fd159', 'SQBmeaned'] = train.loc[train['idhogar']=='1b31fd159', 'escolari']**2
train.loc[train['idhogar']=='faaebf71a', 'meaneduc'] = train.loc[train['idhogar']=='faaebf71a', 'escolari']
train.loc[train['idhogar']=='faaebf71a', 'SQBmeaned'] = train.loc[train['idhogar']=='faaebf71a', 'escolari']**2
train.loc[train['idhogar']=='a874b7ce7', 'meaneduc'] = train.loc[train['idhogar']=='a874b7ce7', 'escolari']
train.loc[train['idhogar']=='a874b7ce7', 'SQBmeaned'] = train.loc[train['idhogar']=='a874b7ce7', 'escolari']**2


# ## Cleaning test data

# Before modelling, we should apply the same cleaning and filling missing values techniques with the train data.

# In[ ]:


# test = test.drop(['Id', 'idhogar'], axis=1)
test.isnull().sum().sort_values(ascending=False)[0:10]


# In[ ]:


test['rez_esc'] = test['rez_esc'].fillna(0)
test['v18q1'] = test['v18q1'].fillna(0)
test['v2a1'] = test['v2a1'].fillna(0)

test_meaneduc_null = test.loc[test['meaneduc'].isnull()]
test_meaneduc_null_ids = test_meaneduc_null['idhogar'].tolist()
for idhogar in test_meaneduc_null_ids:
    test.loc[test['idhogar']==idhogar, 'meaneduc'] = test.loc[test['idhogar']==idhogar, 'escolari']
    test.loc[test['idhogar']==idhogar, 'SQBmeaned'] = test.loc[test['idhogar']==idhogar, 'escolari']**2
    # print(test.loc[test['idhogar']==idhogar][['escolari', 'meaneduc', 'SQBmeaned']])


# In[ ]:


test.columns[test.dtypes=='object']
test.loc[test['dependency']=='yes', 'dependency'] = 1
test.loc[test['dependency']=='no', 'dependency'] = 0
test.loc[test['edjefe']=='yes', 'edjefe'] = 1
test.loc[test['edjefe']=='no', 'edjefe'] = 0
test.loc[test['edjefa']=='yes', 'edjefa'] = 1
test.loc[test['edjefa']=='no', 'edjefa'] = 0


# ## Modelling (still processing...)
# 

# As a first attempt, we test different usual classifiers for this kind of data, using the default parameters.

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
]
# remove the ids from the dataset
train = train.drop(['idhogar', 'Id'], axis=1)
# preprocess dataset, split into training and test part
y = train['Target']
X = train.drop(columns=['Target'])
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=40)

for clf in classifiers:
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)


# SVC seems to have slightly better performance from the rest of the classifiers.
# 
# Still lots of work to be done....

# In[ ]:


# # create the predictions
# clf = SVC(kernel="linear", C=0.025)
# clf.fit(X_train, y_train)

# test_ids = test['Id']
# test = test.drop(['idhogar', 'Id'], axis=1)
# y_pred = clf.predict(test)

# results = pd.DataFrame(columns=['Id', 'Target'])
# results['Id'] = test_ids
# results['Target'] = y_pred

# results.to_csv('submission.csv', index=False)


# In[ ]:




