#!/usr/bin/env python
# coding: utf-8

# 
# ## Please upvote this Kernel, if you find it useful.## 
# 

# This tutorial is for you, **if you are begginer** in machine learning.
# In this kernel we will be learning **simple random forest** model to use and **learn the Visualizations** using **matplotlib**.

# #### In this tutorial, we will learn:
# 1. Reading input file
# 2. Exploring input data
# 3. Finding missing data
# 4. Filling missing data
# 5. Scale up the input data
# 6. Predict the input data using Random Forest Classifier model
# 7. Submit the file to competetion

# In[ ]:


print('Loading packages')
import os              # Package to use directory command to list files
import numpy as np     # linear algebra
import pandas as pd    # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats # for Statistics
import seaborn as sns  # Used for plotting the graph
from statistics import mean
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
le = preprocessing.LabelEncoder()
import re
print('These are the files to use: ',os.listdir("../input"))   # Listing Files


# In[ ]:


print('reading input files..')
data = pd.read_csv('../input/train.csv')
sampl = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


test  = pd.read_csv('../input/test.csv')


# **Basics of Python** : https://www.w3schools.com/python/python_syntax.asp
# 
# **Use link** https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html#min to have a basic understanding of **powerful pandas library**. It makes easy to read, process, write the data.
# 
# **Reading and writing using pandas:** https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html#getting-data-in-out
# 
# To make array operation easy, we need array handling library/utility. **Numpy** can do it all for you.
# 
# **use link** https://docs.scipy.org/doc/numpy/user/quickstart.html to learn **Numpy**

# Let's append test data with train data to make relation between train and test data. It makes all operation easy.

# In[ ]:


# Appending test data with train data, since both dataset can have related values like family name and ticket
df = data.append(test, sort = False)


# In[ ]:


df.head()


# Columns with value NaN are null values. We need to fill missing/null values after exploring the data. 
# 
# Let's first check which all columns have missing values:

# In[ ]:


totalt = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([totalt, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)


# So we got null values in **Cabin, Age, Embarked and Fare**. Survived we need to find out.

# Lets first check missing fare

# In[ ]:


df.loc[df['Fare'].isnull()]


# Let's find simlar data, and fill that for missing fare

# In[ ]:


df.loc[(df['Age'] >= 60) & (df['Pclass'] ==3) & (df['Sex'] == 'male') & (df['Embarked'] =='S')]


# Average Fare look like 7.00, let's move with this value

# In[ ]:


df.loc[df['Fare'].isnull(), 'Fare'] = 7            #First fill missing fare by least value


# Let's **Visualize** the data using plots.
# 
# First, Lets check which Passenger class Survived most. Using sns countplot.
# Additional Reference for Countplot https://seaborn.pydata.org/generated/seaborn.countplot.html?highlight=countplot#seaborn.countplot

# In[ ]:


train1 = df[0:891].copy()
sns.set(style="whitegrid")
plt.figure(figsize=(8,3.5))
ax = sns.countplot(x='Pclass',hue="Survived", data=train1)


# We can see that Passenger **class 1 Survived most** and **3rd class survived least**.

# Let's explore more for **Passenger class vs Sex**. We will be plotting **sns barplot** for this.
# 
# Additional reference for barplot: https://seaborn.pydata.org/generated/seaborn.barplot.html?highlight=barplot#seaborn.barplot

# In[ ]:


train1 = df[0:891].copy()
sns.set(style="whitegrid")
plt.figure(figsize=(10,3))
ax = sns.barplot(x="Pclass", y="Survived",hue='Sex', data=train1)


# Now we will be using **label encoding** since for most of the operations we need label encoding **to convert Text data to Numeric data**.
# 
# Additional reference for Label Encoding: 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

# In[ ]:


df.Cabin = df.Cabin.fillna('0')   # Filling missing values with zero

le.fit(df['Cabin'].astype(str))   # Using label encoder imported at beggining le = preprocessing.LabelEncoder()
df['Cabin'] = le.transform(df['Cabin'].astype(str))


# ### Now we will be **filling missing Age**.
# 
# **In beggining** you can fill **missing value by mean, meadian or mode** value, to make it easy just for beggining. e.g. 
# 
# data['Cabin'] = data['Cabin'].fillna(data['Cabin'].mode())
# 
# Later come back to this point when you start understanding prediction method. I have used ExtraTreesRegressor to predict age based on all columns.

# In[ ]:


# Fill missing Age
## Lets predict the age of a person and fill the missing Age
features = ['Pclass','SibSp','Parch','Fare']
from sklearn.ensemble import ExtraTreesRegressor as ETRg
def AgeFunc(df):
    Etr = ETRg(n_estimators = 200, random_state = 2)
    AgeX_Train = df[features][df.Age.notnull()]
    AgeY_Train = df['Age'][df.Age.notnull()]
    AgeX_Test = df[features][df.Age.isnull()]
    
    Etr.fit(AgeX_Train,np.ravel(AgeY_Train))
    AgePred = Etr.predict(AgeX_Test)
    df.loc[df.Age.isnull(), 'Age'] = AgePred
    
AgeFunc(df)


# Now let's check **missing values of Embarked** column.

# In[ ]:


df.loc[df['Embarked'].isnull()]


# ### First we will be checking **from where most of the 1st class passaengers came from**, based on that we will fill missing value.

# In[ ]:


#Lets Check first from where most 1st Class passesnger Came
sns.set(style="whitegrid")
plt.figure(figsize=(12,2))
ax = sns.barplot(x="Embarked", y="Survived",hue='Pclass', data=df)


# ### From 'C' high number of 1st Pclass people Survived, lets fill 'C' in missing value ###

# In[ ]:


def FillEmbk(data):
    var = 'Embarked'
    data.loc[(data.Embarked.isnull()),'Embarked']= 'C'
FillEmbk(df)


# ## We will now label encode Embarked and Sex column

# In[ ]:


# Label Encode Embarked
def LablFunc(data):
    lst = {'Embarked','Sex'}
    for i in lst:
        le.fit(data[i].astype(str))
        data[i] = le.transform(data[i].astype(str))
LablFunc(df)


# In[ ]:


df.columns


# ### Scale the data now

# In[ ]:


# Lets Scale the data now
from sklearn.preprocessing import StandardScaler
target = data['Survived'].values
select_features = ['Pclass', 'Age','SibSp', 'Parch', 'Fare', 'Embarked','Cabin','Sex']
scaler = StandardScaler()
dfScaled = scaler.fit_transform(df[select_features])
train = dfScaled[0:891].copy()
test = dfScaled[891:].copy()


# In[ ]:


# Checking best features
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, len(select_features))
selector.fit(train, target)
scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]

print('Features importance:')
for i in range(len(scores)):
    print('%.2f %s' % (scores[indices[i]], select_features[indices[i]]))


# ### Import the Model program to train and predict

# In[ ]:


from sklearn.ensemble import RandomForestClassifier  # importing model to use for our prediction


# In[ ]:


SrchRFC = RandomForestClassifier(max_depth = 5, min_samples_split = 4, n_estimators = 500,
                                 random_state = 20, n_jobs = -1)
SrchRFC.fit(train, target)   # Training model with training data


# #### Let's check first accuracy score after predicting values for train data.

# In[ ]:


prc = SrchRFC.predict(train)
accuracy_score(target,prc)


# ### Now we will predict values for test data

# In[ ]:


prdt2 = SrchRFC.predict(test)   #using Random Forest Classifier
print('Predicted result: ', prdt2)


# ## Submitting the file

# In[ ]:


sampl['Survived'] = pd.DataFrame(prdt2)
sampl.to_csv('submission110.csv', index=False)


# In[ ]:




