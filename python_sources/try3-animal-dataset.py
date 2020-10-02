#!/usr/bin/env python
# coding: utf-8

# #Shelter Animal Outcomes
# 
# by Karan Sharma
# ***
# 
# ## Table of contents ##
# 
# 1. Business Understanding (5 min)
#     - Objective
#     - Description
# 2. Data Understanding (15 min)
#     - Import Libraries
#     - Load data
#     - Statistical summaries and visualizations
#     - Exercises
# 3. Data Preparation (5 min)
#     - Missing values imputation
#     - Feature Engineering
# 4. Modeling (5 min)
#     - Build the model
# 5. Evaluation (25 min)
#     - Model performance
#     - Feature importance
#     - Who gets the best performing model?
# 6. Deployment (5 min)
#      - Submit result to Kaggle leaderboard
# 

# # 1. Business Understanding
# 
# ## 1.1 Objective
# Predict condition of animal at foster home.
# 
# ## 1.2 Description
# Every year, approximately 7.6 million companion animals end up in US shelters. Many animals are given up as unwanted by their owners, while others are picked up after getting lost or taken out of cruelty situations. Many of these animals find forever families to take them home, but just as many are not so lucky. 2.7 million dogs and cats are euthanized in the US every year.
# 
# Using a dataset of intake information including breed, color, sex, and age from the Austin Animal Center, we're asking Kagglers to predict the outcome for each animal.
# [Link](https://www.kaggle.com/c/shelter-animal-outcomes)
# 
# 
# # 2. Data Understanding
# 
# ## 2.1 Import Libraries

# In[1]:


# Ignore warnings
import warnings
warnings.simplefilter("ignore", category=PendingDeprecationWarning)

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import re

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from bokeh.charts import HeatMap, bins, output_file, show
from bokeh.palettes import RdYlGn6, RdYlGn9


# Configure visualisations
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# ## 2.3 Load data

# In[2]:


train = pd.read_csv("../input/train.csv")
test    = pd.read_csv("../input/test.csv")
display(train.tail())
display(test.tail())
train = train.append( test , ignore_index = True )
full = train[:]
animal = full[ :26728 ]
display(full.tail())
del train , test


# ## 2.4 Statistical summaries and visualisations

# In[3]:


animal.head()


# ### variable description 
# - AgeuponOutcome: Age when outcome happened
# - AnimalID: Id of the Animal
# - AnimalType: Type of animal, like dog and cat
# - Breed: Breed of the animal
# - Color: Color of the animal
# - DateTime: Date time when outcome happened.
# - ID: id for testing
# - Name: Name given to animal
# - OutcomeSubtype: more details about outcome
# - OutcomeType: What happened to the animal (Return to owner, adoption, etc)
# - SexUponOutcome: Sex of the animal when it was adopted.
# 
# ### 2.4.1 Next have a look at some key information about the variables

# In[4]:


display(animal.info())
animal.describe(include="all")


# Note that all features are non-numeric , we may choose to later convert 

# ### 2.4.2 A heat map of correlation may give us a understanding of which variables are important

# In[5]:


display(sns.countplot(x="AnimalType", data=animal))


# Try to plot other fields, commenting them because the graphs don't give much and add clutter, but do try to plot these at least once.

# In[6]:


#sns.countplot(x="SexuponOutcome", hue="OutcomeType" ,data=animal)
#sns.countplot(x="AgeuponOutcome" ,data=animal) #clearly filed x needs to be cleaned
#sns.countplot(x="Breed" ,data=animal)
#sns.countplot(x="Color" ,data=animal)


# # 3. Data Preparation
# 
# ## 3.1 Convert data in usable form
# 
# First we see "AgeuponOutcome" is not in it's useable form. So we split it in two parts, numeric and unit part then convert it to days.

# In[7]:


multiFactor = {'day': 1,
               'days': 1,
               'week': 7,
               'weeks': 7,
               'month': 30,
               'months': 30,
               'year': 365,
              'years': 365}

def AgeToYear(row):
    m = re.search('([0-9]+)\s([a-zA-Z]+)', str(row))
    return str( int(m.group(1))*int(multiFactor[m.group(2)]) )

full.AgeuponOutcome.fillna("-1 day",inplace=True)
full.AgeuponOutcome = full.AgeuponOutcome.apply(AgeToYear) #ignore warning
full.AgeuponOutcome = pd.to_numeric(full.AgeuponOutcome, errors='coerce') #convert to 


# We would like to know if name was given or not, maybe animals with name are adopted more often?(check hypothesis)

# In[8]:


def CheckName(row):
    if(row == "N"):
        return row
    else:
        return "Y"
    

full.Name = full.Name.fillna("N")
full.Name = full.Name.apply(CheckName)
full = full.rename(columns={'Name': 'hasName'})


# In[9]:


full.DateTime =  pd.to_datetime(full.DateTime, format='%Y-%m-%d %H:%M:%S')


# In[10]:


#seprate sex to sex and intactness
def get_sex(x):
    x = str(x)
    if x.find('Male') >= 0: return 'male'
    if x.find('Female') >= 0: return 'female'
    return 'unknown'
def get_neutered(x):
    x = str(x)
    if x.find('Spayed') >= 0: return 'neutered'
    if x.find('Neutered') >= 0: return 'neutered'
    if x.find('Intact') >= 0: return 'intact'
    return 'unknown'

full['Sex'] = full.SexuponOutcome.apply(get_sex)
full['Neutered'] = full.SexuponOutcome.apply(get_neutered)
full.drop(["SexuponOutcome"],inplace=True,axis=1)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
sns.countplot(full.Sex, palette='Set3', ax=ax1)
sns.countplot(full.Neutered, palette='Set3', ax=ax2)


# In[11]:


#checking if the breed is mix or not
def get_mix(x):
    x = str(x)
    if x.find('Mix') >= 0: return 'mix'
    return 'not'
full['Mix'] = full.Breed.apply(get_mix)
full.drop(["Breed"],inplace=True,axis=1)

#also droping colors, too many values
full.drop(["Color"],inplace=True,axis=1)

#may need to drop time as well, just getting lazy:P, will convert it to catagorial value later
full.drop(["DateTime"],inplace=True,axis=1)


# In[12]:


cols_to_transform = ["hasName", "AnimalType", "Sex", "Neutered", "Mix"]
full_X = full.loc[:,cols_to_transform]
full_X_dummies = pd.get_dummies( full_X, columns = cols_to_transform )
train_valid_X = full_X_dummies[ 0:26728 ]
train_valid_y = animal.OutcomeType
test_X = full_X_dummies[ 26728: ]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )
print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)

model = RandomForestClassifier(n_estimators=100)
model.fit( train_X , train_y )
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))


# In[13]:


test_Y = model.predict( test_X )
ID = full[26729:].ID.astype(int)
test = pd.get_dummies(test_Y)
test = test.assign(ID=[i for i in range(len(test))])
test['Died']=0
test['Euthanasia']=0
#test = pd.concat([test, pd.get_dummies(test_Y)], axis=0)
print(test)
test.to_csv( 'outcome_pred.csv' , index = False )


# In[13]:




