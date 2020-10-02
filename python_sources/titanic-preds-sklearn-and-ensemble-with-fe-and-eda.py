#!/usr/bin/env python
# coding: utf-8

# This is my first on Kaggle and hence, in this kernel, I use Kaggle's Getting Started Competition, Titanic: Machine Learning from Disaster, to use my learning of different Aspects of DataScience and ML to beat the odds in coming up with a better Accuracy and low RMSE score.
# 
# <font color=brown size=5>
# Data Science Frame Work: 
# <font color=black size=3>
# 1. Define the Problem: Besides Algorithm, Model and Technology is defined, we need to get the Business Problem defined. Typically this happens with various Stakeholders getting together to articulate it for the Technology to provide sutle requirements like  trade-off between False Negatives and True Negative. These are not available in thie competition. However, we will stick to the Kaggle's evaluation criteria that is to predict the Survival Classifier (1 for Survived and 0 for Not-Survived). Will use Python to Build the Model. 
# 2. Data Collection and Gathering: This would in reality be the task that require lots of efforts and resources. In this, case Kaggle has provided the for downloading (https://www.kaggle.com/c/titanic/data). But, Data Analysis and Correction w.r.t its integrity, meaning, abberations (Outliers and Missing data) is still to be done and that will be done during Data Clearning activity and thereafter Data Tranformation to make it ready for Machine's consumption. This is usually referred to as Data Wrangling. 
# 3. Exploratory Data Analysis: To understand data in Satistical terms that is Correlations and Linearity between and among Features. Identifying Univariate and Multivariate variables. This can be done either in Pivot and/or Graphical representation. This is where the sutle requirement of Hypothesis and rejection of it with Significance and Confidence will have to be done. 
# 4. Build Model with Data: Preparing the Model to arrive at the rules based on the Data and the Outcome. Data and Expected Outcome will determine the Algorithm to be used. Its not that selecting an Algorithm will be produce the desired output and thruput as it requires the Techiniques and Tricks that are at the Craftman's (call him/her DataScient) disposal. Typically, this forms the Activity of Building and Training the Model. 
# 5. Validate the Model: Validation is the critical step and again Craftmanship comes into play in selecting the Data for Validation(s). This step is significant as it eludicidates if the Model is fit to Predict for Known Data (in ML terminology Overfit) or can work equally good with unseen Data. The opposite of Overfitting is Underfitting and that tells us that the Model is not designed for it to grasp the completeness of the Dataset to understand various possibilities. This is also called Generalized Model. In either of the cases, will have to go back to Previous steps to inculcate the required changes for the Model to have Best fit (Training). 
# 6. Optimize and Strategize: This is task where certain Technical Or Repetative tasks can be given to the Data Engineer and to concerntrate on Optimizing the Model Performance. This is an ongoing tasks as it is expected in real-world that new data keeps coming and required Model to be retrained to maintain the Performance and Prediction Accuracy. <br>
# <font color=brown size=4>
# This kernal starts with Point 2 <font color=black size=3> as first one is already taken care by Kaggle and First part of point 2, of having Raw Data making available for this Competition is also done by Kaggle as well. <br>
# <font size=3>
# We can make use of popular Libraries Python3.x for Data Wrangling. <br>
# 2.1 Importing Libraries

# In[ ]:


import pandas as pd # for data processing and analysis modeled
import matplotlib   # for scientific and visualization
import numpy as np  # for scientific computing
import scipy as sp  # for scientific computing and mathematics Functions

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import IPython 
from IPython import display #  printing of dataframes in Jupyter notebook
import sklearn      # for machine learning algorithms

import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

#misc libraries
import random
import time

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# <font color=brown size=4>
#     2.2 Get to know the Data and go a step further to look at the Individual Charactertics and few more steps further towards gaining knowledge of Dependencies among data parts (Features) of the Data Point (Row)

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

get_ipython().system('pwd')
get_ipython().system('ls ')


# In[ ]:


data_raw = pd.read_csv('../input/titanic/train.csv') # this is the data for training and our Evaluation

data_val  = pd.read_csv('../input/titanic/test.csv') # this is provided by Kaggle and to be used to Submit the final Predictions

# make a copy for future usage to check on data.
data_train = data_raw.copy(deep = True)
data_test = data_val.copy(deep = True)

print (data_train.info())
print ("#"*50)
print (data_test.info())
print ("#"*50)

"""
Combine both Test and Train Datasets for doing analysis on Categorical values (Classes) that may be present 
only in Test but not in Training Dataset
"""
data_combine = [data_train, data_test]


# <font color=brown size=5>
# 2.2 Data Analysis <nr>
# <font color=brown size=4>
# 2.2.1 DataTypes: <br>
# <font color=black size=3>
# 1. There are two continuous quantitative variable namely, Age and Fare. This is in case of both Train and Test datasets.
# 2. There are 5 variable with Object Datatype, meaning, these could be free-flowing Nominal Datatype or Categorical
# 3. There are 5 and 4 Numerical values in Train and Test Datasets respectively. These again could be Ordinal or Nominal. However, the difference between Train and Test is that Survived Variable is not in Test Dataset. This is the Dependent varilable and the rest are potential Independent variables that could be included in the Model for it to come up with the Predictions. 
#     
# <font color=brown size=4>
# 2.2.2 MissingData: <br>
# <font color=black size=3>
# 1. In Training Dataset, Age and Cabin Features have missing values. 20% of Age values are missing, where as 80% of Cabin are missing. Will have to retain Age as missing values are less than the Standard prescription that is 40%, moreover, we will have to check Relevance of Age on Survial Chances. Eventhough Cabin has more missing values, relevance (or inference) is to be extracted so as to make a decision. Same with Test Dataset However in Test Dataset, additionally one Fare value missing. This value is to be imputed and probably with relevant Mean value.

# In[ ]:


data_train.head(15)


# <font color=brown size=4>
# 2.2.3 Individual Feature Analysis <br>
# <font size=4>
# 2.2.3 Categorical values: Going by the above
# <font color=black size=3>
#     > Survived Feature is already numeric and its Categorical, 1 Denotes Survived and 0 denotes otherwise.  <br>
#     > Pclass -- is Ordinal and denotes 1=Upper, 2=Middle 3=Lower and already in numeric format  <br>
#     > Name is a Nominal type and as it is is of no use in the Predictor Model.<br>
#     > Sex is categorical and in nominal format. Hence, this should be convered to numeric type.  <br>
#     > SibSp (Sibling & Spouse) and Parch (Parent & Children) are numeric fields and are Ordinal in nature. <br>
#     > Ticket value is Alpha-numerica and a unique value like PassengerId. These can be dropped.<br>
#     > Overall Training Sample Numeric Data Distribution after high level analysis: <br>
#             >> Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).<br>
#             >> Survived is a categorical feature with 0 or 1 values. <br>
#             >> Around 38% samples survived representative of the actual survival rate at 32%. <br>
#             >> Most passengers (> 75%) did not travel with parents or children. <br>
#             >> Nearly 30% of the passengers had siblings and/or spouse aboard. <br>
#             >> Fares varied significantly with few passengers (<1%) paying as high as $512. <br>
#             >> Few elderly passengers (<1%) within age range 65-80. <br>

# In[ ]:


data_train.describe(include=['O'])


# <font color=brown size=4>
# 2.3.3 Overall Training Sample set Categorical Data Distribution is: <br>
#     <font color=black size=3>
# > Names are unique across the dataset (count=unique=891). <br>
# > Sex variable as two possible values with 65% male (top=male, freq=577/count=891). <br>
# > Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin. <br>
# > Embarked takes three possible values. S port used by most passengers (top=S). <br>
# > Ticket feature has high ratio (22%) of duplicate values (unique=681). <br>

# <font color=brown size = 4>
# 2.2.4 Assumption based on the Data Analysis done so far <br> 
# <font color=brown size = 3> Correlation:<br> <font color=black> We will have know how each Feature (or the features that would be included in the model) with Survied Feature. Importantly, these are to be done so as to compare with the Modelled Correlation later. <br> 
# <font color=brown size = 3> Completing: <font color=black> <br>
#     Age Feature data is to be completed as it seem to have strong correlation with Survived Feature. <br>
#     Embarked feature also have correlation with the Survived Feature and needs to be completed (imputing)
# <font color=brown size = 3> Correcting: <font color=black> <br>
#     Ticket Feature has 22% duplicate values and does not contribute much to the Prediction. This field will be dropped. <br><br> 
#     Cabin Feature as it is incomplete and has high number of missing values in both Training and Validation Datasets. <br> <br>
#     PassengerId will be dropped as this is unique value and certainly have no impact on Survived Feature. <br> <br>
#     Name Feature also may not have much contribution and will be dropped. 
# <font color=brown size = 3> Creating: <font color=black> <br>
#     SibSp and Parch are two Features that more or less convey the Familysize. A new feature FamilySize will be created by combinig these two. This new feature gives out number of Family members onboarded. <br> <br>
#     Name Feature has Title in it which can be used alogn with Sex to establish some kind of correlation. <br><br>
#     Age and Fare Features are a continous numeric value and will have to create a new Ordinal Categorical Field to bucket them in different ranges. <br> <br>
# <font color=brown size = 3> Classifying: <font color=black> <br>
#     Sex Feature to be classified. Women more likely to have survived <br> <br>
#     Age Feature: Children were more likely to have survived <br> <br>
#     Pclass Feature: Upper Class (Pclass=1) were more likely to have survived. 

# <font color=brown size=4>
#     Data Exploration using Pivots and/or Visualization

# In[ ]:


data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Pclass We observe significant correlation (>0.6) among Pclass=1 and Survived (classifying #3). Retain this feature in our model.

# In[ ]:


data_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Assumption is confirmed that Sex=female had very high survival rate at 74% (classifying #1).

# In[ ]:


data_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


data_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


data_train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# SibSp and Parch features have low or zero correlation as the counts go up. It may be best to derive a feature or a set of features from these individual features (creating #1).

# In[ ]:


plt.hist(x = [data_train[data_train['Survived']==1]['Age'], data_train[data_train['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Age Histogram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()


# <font color=brown size=4>
# Observations <br>
# <font color=black size=3>
# Infants (Age <=4) had high survival rate.<br>
# Oldest passengers (Age = 80) survived.<br>
# Large number of 15-25 year olds did not survive.<br>
# Most passengers are in 15-35 age range. <br>
# <font color=brown size=4>
# Decisions<br>
# <font color=black size=3>
# This simple analysis confirms our assumptions<br>
# We should consider Age (confirms our assumption) in our model training.<br>
# Complete the Age feature for null values. <br>
# We should create band age groups as New Feature. <br>

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(data_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# <font color=brown size=4> 
# Observations. <br>
# <font color=black size=3> 
# Pclass=3 had most passengers, however most did not survive. Confirms assumption.<br>
# Infant passengers in Pclass=2 and Pclass=3 mostly survived. Further qualifies assumption. <br>
# Most passengers in Pclass=1 survived. Confirms assumption. <br>
# Pclass varies in terms of Age distribution of passengers. <br>
# 
# <font color=brown size=4> 
# Decisions. <br>
# <font color=black size=3> 
# Consider Pclass for model training.

# <font color=brown size=4>
# 3.1 Load Data Exploratory Libraries

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(data_train, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# <font color=brown size=4>
# Observations <br>
# <font color=black size=3>
# Female passengers had much better survival rate than males. Confirms classifying assumption.  <br>
# Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived. <br>
# Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports.  <br>
# Ports of embarkation have varying survival rates for Pclass=3 and among male passengers.  <br>
# 
# <font color=brown size=4>
# Decisions <br>
# <font color=black size=3>
# Add Sex feature to model training.  <br>
# Complete and add Embarked feature to model training.  <br>

# In[ ]:


grid = sns.FacetGrid(data_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# <font color=brown size=4> 
# Observations <br>
# <font color=black size=3>
# Higher fare paying passengers had better survival. Confirms our assumption for creating fare ranges.<br>
# Port of embarkation correlates with survival rates. Confirms correlating and completing assumptions. <br>
# 
# <font color=brown size=4> 
# Decisions <br>
# <font color=black size=3>
# Consider banding Fare feature.

# <font color=brown size=4> 
# Wrangle data <br>
# <font color=black size=3> 
# 
# With confirmed Assumptions and taken decisions, now time to work on Data to Create New Features, Dropping unncessisary Features and Converting Types of features and lastly imputing. 
# 
# <font color=brown size=4> 
# Correcting by dropping features
# <font color=black size=4> 
# 
# This is a good starting goal to execute. By dropping features we are dealing with fewer data points. Speeds up our notebook and eases the analysis.
# 
# Based on our assumptions and decisions we want to drop the Cabin (correcting #2) and Ticket (correcting #1) features.
# 
# Note that where applicable we perform operations on both training and testing datasets together to stay consistent.

# In[ ]:


print("Before", data_train.shape, data_test.shape, data_combine[0].shape, data_combine[1].shape)

data_train = data_train.drop(['Ticket', 'Cabin'], axis=1)
data_test = data_test.drop(['Ticket', 'Cabin'], axis=1)
data_combine = [data_train, data_test]

print("After", data_train.shape, data_test.shape, data_combine[0].shape, data_combine[1].shape)


# <font color=brown size=4> 
# Creating new feature extracting from existing <br>
#     <font color=black size=3> 
# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# In the following code we extract Title feature using regular expressions. The RegEx pattern (\w+\.) matches the first word which ends with a dot character within Name feature. The expand=False flag returns a DataFrame.<br>
# <font color=brown size=4> 
# Observations<br>
# <font color=black size=3> 
# When we plot Title, Age, and Survived, we note the following observations.
# 
# Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
# Survival among Title Age bands varies slightly.
# Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).<br>
# 
# <font color=brown size=4> 
# Decision <br>
# <font color=black size=3> 
# We decide to retain the new Title feature for model training.

# In[ ]:


for dataset in data_combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(data_train['Title'], data_train['Sex'])


# We can replace many titles with a more common name or classify them as Rare.

# In[ ]:


for dataset in data_combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
data_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# We can convert the categorical titles to ordinal.

# Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in data_combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

data_train.head()


# In[ ]:


data_train = data_train.drop(['Name', 'PassengerId'], axis=1)
data_test = data_test.drop(['Name'], axis=1)
data_combine = [data_train, data_test]
data_train.shape, data_test.shape


# <font color=brown size=4>
# Converting a categorical feature <br>
# <font color=black size=3>
# Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
# 
# Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

# In[ ]:


for dataset in data_combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

data_train.head()


# <font color=brown size=4>
# Completing a numerical continuous feature <br>
#     <font color=black size=3>
# Now we should start estimating and completing features with missing or null values. We will first do this for the Age feature.
# 
# We can consider three methods to complete a numerical continuous feature.
# 
# 1. A simple way is to generate random numbers between mean and standard deviation.
# 
# 2. More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using median values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
# 
# 3. Combine methods 1 and 2. So instead of guessing age values based on median, use random numbers between mean and standard deviation, based on sets of Pclass and Gender combinations.
# 
# Method 1 and 3 will introduce random noise into our models. The results from multiple executions might vary. We will prefer method 2.

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(data_train, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations

# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

# In[ ]:


for dataset in data_combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

data_train.head()


# Let us create Age bands and determine correlations with Survived.

# In[ ]:


data_train['AgeBand'] = pd.cut(data_train['Age'], 5)
data_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# Let us replace Age with ordinals based on these bands.

# In[ ]:


for dataset in data_combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
data_train.head()


# In[ ]:


#AgeBand feature can be removed
data_train = data_train.drop(['AgeBand'], axis=1)
data_combine = [data_train, data_test]
data_train.head()


# <font color=brown size=4>
#     Creating a new Feature: <br>
#     <font color=black size=3>
#     New FamilySize by addining up Sibsp and Parch

# In[ ]:


for dataset in data_combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

data_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# another new feature called IsAlone. 

for dataset in data_combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

data_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:


# with IsAlone field with good correlation with Survived Feature, can drop Parch, Sibsp and Familysize Features
data_train = data_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
data_test = data_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
data_combine = [data_train, data_test]

data_train.head()


# In[ ]:


# Another New feature combining Pclass and Age.
for dataset in data_combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

data_train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# <font color=brown size=4>
#     Completing a categorical feature <br>
#     <font color=black size=3>
# Embarked feature takes S, Q, C values based on port of embarcation. Our training dataset has two missing values. We simply fill these with the most common occurance

# In[ ]:


freq_port = data_train.Embarked.dropna().mode()[0]

for dataset in data_combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
data_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# <font color=brown size=4>
#     Converting categorical feature to numeric <br>
# <font color=black size=3>
# We can now convert the Embarked feature by creating a new numeric Port feature

# In[ ]:


for dataset in data_combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data_train.head()


# <font color=brown size=4> 
# Quick completing and converting a numeric feature <br>
# <font color=black size=3>
# We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently for this feature. We do this in a single line of code.

# In[ ]:


data_test['Fare'].fillna(data_test['Fare'].dropna().median(), inplace=True)
data_test.head()


# In[ ]:


# Fare has continous numeric data and hence to be converted to category range to make it categorical
data_train['FareBand'] = pd.qcut(data_train['Fare'], 4)
data_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


# Convert the Fare feature to ordinal values based on the FareBand
for dataset in data_combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

data_train = data_train.drop(['FareBand'], axis=1)
data_combine = [data_train, data_test]


# In[ ]:


print (data_train.head(10))
print ("#"*75)
print (data_test.head(10))


# In[ ]:


import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(8, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        data_train.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.6 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=12)

correlation_heatmap(data_train)


# In[ ]:


#set correlation above 0.75 and see true/false values
abs(data_train.corr())> 0.50


# In[ ]:


sns.heatmap(data_train.corr(), center=0);


# <font color=brown size=4>
# 4.1 Load Data Modelling Libraries <br>
# <font color=black size=3>
# There are many Predictive Modelling Algorithms. However, below are narrowed down give the given problem of Supervised Learning (as dataset is being used for Training the Model) and the Classification Prediction (if a passenger is survived or not).<br>
# 1. Logistic Regression
# 2. KNN or k-Nearest Neighbors
# 3. Support Vector Machines
# 4. Naive Bayes classifier
# 5. Decision Tree
# 6. Random Forrest and Gradient Descents
# 7. Perceptron
# 8. XGB
# 9. CatBoost
# 10. Voting Classifier

# In[ ]:


X_train = data_train.drop("Survived", axis=1)
Y_train = data_train["Survived"]
X_test  = data_test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


print (X_train.columns)
print ("#"*50)
print (X_test.columns)
print ("#"*50)
X_train.drop("Embarked", axis=1)
X_test.drop("Embarked", axis=1)


# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
print ('Test ACC Logistic Regression -- > ', acc_log)

# generating ROC and RMSC for training just to get hang of if RMSE is going down or up with each model
X_pred = logreg.predict(X_train)
X_predprob = logreg.predict_proba(X_train)[:,1]
t_lr_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_lr_score)
t_lr_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_lr_roc)
t_lr_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_lr_rmse)


# With Logistic Regression we can validate Assumption and Decisions made for Creating and Completing Feature Goals. Internally, Algorithm calculates the coefficients of the features in decision function. 
# Positve Coefficients increase the Odds of probability of right Prediction and Negative Coefs decrease the Odds. 

# In[ ]:


coeff_df = pd.DataFrame(data_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In this model, 
# > Sex has highest Positive Correlation/Coefficient implying as the value of Sex increases (from 0=Male to 1=Female) the probability of Survived=1 increases. So, is the Title has the second highest Postive Correlation. 
# 
# > Same with Pclass but it has inverse relationship, that is, Pclass=1 to 3 increases, Surival=1 decreases. This way Age*Class Artificial feature is the second best negative correlation with Survived. 

# <font color=brown size=5> 
#     SVM (Support Vector Machines) <br>
# <font color=black size=3> 
# SVM is a non-probabilistic Binary Classifier. This  is a Supervised Learning model with associated Learning Algorithms that analyze data for Classification and Regression. Given training samples, each sample will be marks/assigns to one of the two categories and thus makes it Non-Probabilistic. 

# In[ ]:


# Support Vector Machines

svc = SVC(probability=True) 
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
print ('Test ACC SVM -- > ', acc_svc)
# generating ROC and RMSC for training just to get hang of if RMSE is going down or up with each model
X_pred = svc.predict(X_train)
X_predprob = svc.predict_proba(X_train)[:,1]
t_svm_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_svm_score)
t_svm_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_svm_roc)
t_svm_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_svm_rmse)


# In[ ]:


from sklearn.model_selection import GridSearchCV 
  
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear', 'rbf']}  
  
#grid = GridSearchCV(LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr'), param_grid, refit = True, verbose = 2) 
grid = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr')
grid.fit(X_train, Y_train) 
Y_pred = grid.predict(X_test)
acc_svc1 = round(grid.score(X_train, Y_train) * 100, 2)
print ('Test ACC LinearSVC -- > ', acc_svc)


# <font color=brown size=5> 
#     KNN (K-Nearest Neighbours) <br>
# <font color=black size=3> 
# KNN is a non-parametric method used for Classification (and Regression). Classification happens with the majority of votes it gets from its neighbours, more votes and the Prediction is assigned to that Classifier Class. K is a positive integer, typically small (K=1). When K is 1, then the object is assigned to the class that of that single nearest neighbour. 

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
print ('Test ACC KNN ', acc_knn)

X_pred = knn.predict(X_train)
X_predprob = knn.predict_proba(X_train)[:,1]
t_knn_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_knn_score)
t_knn_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_knn_roc)
t_knn_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_knn_rmse)


# <font color=brown size=5> 
#     Naive Bayes <br>
# <font color=black size=3> 
#     Naive Bayes classifiers are the family of simple probabilistic classifiers based on applying Bayes Theorm with strong assumption of Independence assumption between the Features. This Algorithm is highly scalable, requiring a number of parameters linear in the number of variables (features) in a learning problem. 
#     Drawback of this is that its sheldom is the case in real-time to have such Independent Features. With such correlation between the feature in our case, it most probably will have the low confidence levels. 

# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
print ('Test ACC Naive ', acc_gaussian)

X_pred = gaussian.predict(X_train)
X_predprob = gaussian.predict_proba(X_train)[:,1]
t_nb_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_nb_score)
t_nb_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_nb_roc)
t_nb_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_nb_rmse)


# <font color=brown size=5> 
#     Perceptron <br>
# <font color=black size=3> 
#     Perceptron is a supervised learning of Binary classifiers with functions that decide whether an input, represented by a vector of numbers,belongs to a specific class. This is typically Linear classifier, that is prediction happens based on a linear predictor function with addition of having weights assigned to feature vector. 

# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# <font color=brown size=5>
#     Gradient Descent <br>
# <font color=black size=3>
# Every Machine Learning Engineer is looking to improve their model performance. Gradient Descent one of the most popular optimization algorithm that helps machine learning models converge at a minimum value through repeated steps. Essentially, gradient descent is used to minimize a function by finding the value that gives the lowest output of that function. Often times, this function is usually a loss function. Loss functions measure how bad our model performs compared to actual occurrences. Hence, it only makes sense that we should reduce this loss. One way to do this is via Gradient Descent. This works on the same principle of Linear Relationships between Independent and Dependent variables. 

# In[ ]:


# Gradient Descent

gd = GradientBoostingClassifier()
gd.fit(X_train, Y_train)
Y_pred = gd.predict(X_test)
acc_gd = round(gd.score(X_train, Y_train) * 100, 2)
acc_gd
print ('Test ACC Gradient Descent', acc_gd)

X_pred = gd.predict(X_train)
X_predprob = gd.predict_proba(X_train)[:,1]
t_gd_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_gd_score)
t_gd_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_gd_roc)
t_gd_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_gd_rmse)


# <font color=brown size=5>
#     Stochastic Gradient Descent <br>
# <font color=black size=3>  
# In Gradient Descent Algorithm, gradients on each observation is done one by one. It becomes resource intensive when the Dataset is Large. To overcome this, Observations are Randomly picked up. This random probabilistic selection of Observations makes this Stochastic. This Algorithm offers wide variety of parameters to minimize the lost, increase the scope and pace of descent (learning). We will explore these options later to improve on the performance. Let us for now use the basic algorithm with defaults. 

# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier(loss='log')
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
print ('Test ACC Stochastic Gradient Descent', acc_sgd)
X_pred = sgd.predict(X_train)
X_predprob = sgd.predict_proba(X_train)[:,1]
t_sgd_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_sgd_score)
t_sgd_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_sgd_roc)
t_sgd_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_sgd_rmse)


# <font color=brown size=5>
#     Decision Trees
# <font color=black size=3>
#     Decision Tree is a predictive model that maps features (Tree Branches) to conclusions about the target value (Tree Leaves). In this model, target variable takes a finite set of values called Classification trees; in these tree structures, leaves represent class labels and braches represent conjuctions of features that lead to those class labels. Decision Trees where target variable can take continous values, typically real numbers, are called Regression Trees. 
#     
# Drawback of this Learning Model is that they tend to Overfit and are rigid due to the defined Tree Structured formed during Training. 

# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
print ('Test ACC Decision Tree', acc_decision_tree)
X_pred = decision_tree.predict(X_train)
X_predprob = decision_tree.predict_proba(X_train)[:,1]
t_dt_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_dt_score)
t_dt_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_dt_roc)
t_dt_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_dt_rmse)


# In[ ]:


from sklearn.ensemble import BaggingClassifier

bag_cla = BaggingClassifier()
bag_cla.fit(X_train, Y_train)

y_pred=bag_cla.predict(X_test)
acc_bag_cla = round(bag_cla.score(X_train, Y_train) * 100, 2)

print ('Test ACC Bagging Classifier', acc_bag_cla)
X_pred = bag_cla.predict(X_train)
X_predprob = bag_cla.predict_proba(X_train)[:,1]
t_bc_score = round(metrics.accuracy_score(Y_train, X_pred) *100,2)
print ('Score -- >', t_bc_score)
t_bc_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_bc_roc)
t_bc_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_bc_rmse)
# Summary of the predictions made by the classifier
#print(classification_report(test1_y_dummy,y_pred))
#print(confusion_matrix(y_pred,test1_y_dummy))

#Accuracy Score
#print('accuracy is ',accuracy_score(y_pred,test1_y_dummy))

#BCC = accuracy_score(y_pred,test1_y_dummy)


# <font color=brown size=5>
#     Random Forest
# <font color=black size=3>
#     Random Forest are Ensemble learning method for classification and regression. This model Operates by constructing multitude of Decision Trees at training time and the output is the mode of classes (Classification) or mean prediction (regression) of individual Trees. 

# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
print ('Test ACC Random Forest', acc_random_forest)
X_pred = random_forest.predict(X_train)
X_predprob = random_forest.predict_proba(X_train)[:,1]
t_rm_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_rm_score)
t_rm_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_rm_roc)
t_rm_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_rm_rmse)


# <font color=brown size=5>
#    XGBoost (Extreme Gradient Boosting)
# <font color=black size=3>
#     XGBoost is a decision tree-based Ensemble machine learning algorithm that uses Gradient Boosting framework. In prediction problems involving unstructured data (images, text etc) artificial neural networks tend to outperform other algorithm frameworks. However, when it comes to small-to-medium structure/tabular data, decision tree based algorithms are considered best-in class now. 
#     
# Evaluation of Ensemble models started with Decision trees, a graphical representation of possible solutions to a decision based on certain conditions.<br> 
#     > <font color=brown size=4> Bootstraping aggregating or Bagging <font color=black size=3> ensemble meta-algorithms combining predictions from multiple decision trees through a voting mechaism, gave rise to Boosting Algorithms. <br>
#     > Baggin-based algorithms where only <font color=brown size=4> sub-set of features are selected at random to build a forest <font color=black size=3> or collection of decision trees, gave rise to Random Forest Algorithm <br>
#     > Models are <font color=brown size=4> build sequentially by minimizing errors <font color=black size=3> from previous models while increasing the influence of high-performing models, gave rise to Boosting Algorithms. <br>
#     > On these Boosting Algorithms, when additionally <font color=brown size=4> employed Gradient Descent algorithm <font color=black size=3> to minimize errors, gave rise to Gradient Descent Algorithms. <br>
#     > <font color=brown size=4> Optimized Gradient Boosting Algorithm <font color=black size=3> by employing parallel processing , tree purning, handling missing values and regularization to avoid overfitting (or Bias), gave rise to this new Queen of Machine Learning Algorithms. 
# 

# In[ ]:


from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100)
xgb.fit(X_train, Y_train)
Y_pred_xgb=xgb.predict(X_test)
xgb.score(X_train, Y_train)
acc_xgb = round(xgb.score(X_train, Y_train) * 100, 2)
acc_xgb
print ('Test ACC XGBoost', acc_xgb)
X_pred = xgb.predict(X_train)
X_predprob = xgb.predict_proba(X_train)[:,1]
t_xgb_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_xgb_score)
t_xgb_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_xgb_roc)
t_xgb_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_xgb_rmse)

#Parameters list can be found here as well: https://xgboost.readthedocs.io/en/latest/parameter.html


# In[ ]:


## have tunned parameters using GridSearch and randomly to come up with better score
## refer to this Notebook 
xgb_tuned = XGBClassifier(
 learning_rate =0.1,
 n_estimators=143,
 max_depth=5,
 min_child_weight=1,
 gamma=0.0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27
)
xgb_tuned.fit(X_train, Y_train)
Y_pred_tuned=xgb_tuned.predict(X_test)
xgb_tuned.score(X_train, Y_train)
acc_xgb_tuned = round(xgb_tuned.score(X_train, Y_train) * 100, 2)
acc_xgb_tuned
print ('Test ACC XGBoost Tunned', acc_xgb_tuned)
X_pred = xgb_tuned.predict(X_train)
X_predprob = xgb_tuned.predict_proba(X_train)[:,1]
t_xgbt_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_xgbt_score)
t_xgbt_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_xgbt_roc)
t_xgbt_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_xgbt_rmse)


# In[ ]:


#https://catboost.ai/docs/concepts/python-reference_parameters-list.html

from catboost import CatBoostClassifier

catb=CatBoostClassifier(iterations=2500, depth=5, learning_rate=0.3, verbose=0, 
                        allow_writing_files=False, loss_function='CrossEntropy', random_strength=0.1, leaf_estimation_method='Gradient') 
catb.fit(X_train, Y_train)

y_pred=catb.predict(X_test)
acc_catb = round(catb.score(X_train, Y_train) * 100, 2)

print ('Test ACC CatBoost', acc_catb)
X_pred = catb.predict(X_train)
X_predprob = catb.predict_proba(X_train)[:,1]
t_catb_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_catb_score)
t_catb_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_catb_roc)
t_catb_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_catb_rmse)


# In[ ]:


from sklearn.ensemble import VotingClassifier

clr = LogisticRegression()
csvc = SVC(probability=True) 
cknn = KNeighborsClassifier(n_neighbors = 3)
cgau = GaussianNB()
cgb = GradientBoostingClassifier()
csgb = SGDClassifier (loss='log')
crf = RandomForestClassifier(n_estimators=100)
cxgbt = XGBClassifier(learning_rate =0.1, n_estimators=143, max_depth=5, min_child_weight=1, gamma=0.0, subsample=0.8,
                      colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
cdt = DecisionTreeClassifier()
cbc = BaggingClassifier()
ccb = CatBoostClassifier(iterations=2500, depth=5, learning_rate=0.3, verbose=0, 
                        allow_writing_files=False #, train_dir='/gdrive/My Drive/MyLearning/MLDLAIPython/Data/TextData/'
                        , loss_function='CrossEntropy', random_strength=0.1, leaf_estimation_method='Gradient') 
"""
eclf1 = VotingClassifier(estimators=[('LogReg', clr), ('SVC', csvc), ('KNN', cknn), ('GradientBoost', cgb), 
                                    ('StochaisticGB', csgb), ('RandomForest', crf), ('XGB', cxgb), 
                                    ('DecisionTree', cdt), ('BaggClassifier', cbc), ('CatBoost', ccb)], voting='soft')
"""
eclf1 = VotingClassifier(estimators=[('RandomForest', crf), ('DecisionTree', cdt), ('BaggClassifier', cbc), ('CatBoost', ccb)], voting='soft')

eclf1.fit(X_train, Y_train)
y_pred=eclf1.predict(X_test)
acc_eclf1 = round(eclf1.score(X_train.astype(float), Y_train.astype(float)).astype(float) * 100, 2)
#acc_eclf1 = eclf1.score(X_train.astype('float64'), Y_train.astype('float64'))

print ('Test ACC Voting Classifier ', acc_eclf1)
X_pred = eclf1.predict(X_train.astype(float))
X_predprob = eclf1.predict_proba(X_train)[:,1]
t_eclf1_score = metrics.accuracy_score(Y_train, X_pred)
print ('Score -- >', t_eclf1_score)
t_eclf1_roc = metrics.roc_auc_score(Y_train, X_predprob)
print ('ROC -- > ',  t_eclf1_roc)
t_eclf1_rmse = metrics.mean_squared_error(Y_train, X_predprob)
print ('RMSC -- > ',  t_eclf1_rmse)
eclf1.fit(X_train, Y_train)
y_pred=catb.predict(X_test)
acc_eclf1 = round(eclf1.score(X_train.astype(float), Y_train.astype(float)).astype(float) * 100, 2)


# In[ ]:


from sklearn.model_selection import cross_val_score
"""
for clf, label in zip([clr, csvc, cknn, cgb, csgb, crf, cxgb, cdt, cbc, ccb, eclf1], ['Logistic Regression', 
'KNN', 'GradientBoosting', 'StochasticGB', 'RandomForest', 'XGB', 'Decision Tree', 'Bagging Classifier', 
'CatBoost', 'Voting Classifier']) :
"""
for clf, label in zip([crf, cdt, cbc, ccb, eclf1], ['RandomForest',  'Decision Tree', 'Bagging Classifier', 
                                                    'CatBoost', 'Voting Classifier']):  
  scores = cross_val_score(clf, X_train, Y_train, cv=5, scoring='accuracy')
  print ("Accuracy: %0.2f (+/- %0.2f) [%s]" % (round(scores.mean()*100,2), scores.std(), label))


# <font color=brown size=6>
# Model Evaluation
# <font color=black size=3>
# Lets rank the evaluation of all the models to choose the best. 

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Gradient Descent',
              'Stochastic Gradient Descent', 'Linear SVC', 
              'Decision Tree', 'XgBoost', 'XgBoost_Tuned', 'Bagging Classifer', 'Cat Boost', 'Voting Classifier'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_gd,acc_sgd, acc_linear_svc, acc_decision_tree, acc_xgb,acc_xgb_tuned, acc_bag_cla, acc_catb, 
              acc_eclf1],
    'ROC': [t_svm_roc, t_knn_roc, t_lr_roc, 
              t_rm_roc, t_nb_roc, -100, 
              t_gd_roc,t_sgd_roc, -100, t_dt_roc, t_xgb_roc,t_xgbt_roc, t_bc_roc, t_catb_roc, t_eclf1_roc],
    'RMSE': [t_svm_rmse, t_knn_rmse, t_lr_rmse, 
              t_rm_rmse, t_nb_rmse, 100, 
              t_gd_rmse,t_sgd_rmse, 100, t_dt_rmse, t_xgb_rmse,t_xgbt_rmse, t_bc_rmse, t_catb_rmse, t_eclf1_rmse]
    
})
models.sort_values(by='RMSE', ascending=True)


# <font color=brown size=4>
# As can see, Random Forest and Decision Trees have the same scores but Decision Tree Beats Random Forest in RMSE. The evaluation is based on lowest RMSE and hence, will be using it for submission on 02Oct19. 
# Earlier Submissions:
#     -- A week ago, Random forest but now, when i checked RMSE score, choosing DT. 

# In[ ]:


X_pred = decision_tree.predict(X_train)

submission = pd.DataFrame({
        "PassengerId": data_test["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv("../working/submission_TitanicSurvived_pred_26Feb20_1757hr.csv", index=False)

print('Validation Data Distribution: \n', submission['Survived'].value_counts(normalize = True))
submission.sample(10)

print ("Not Normalized Counts")
submission.Survived.value_counts()


# In[ ]:


X_train.drop['Embarked']
X_train.columns


# In[ ]:


X_test.columns


# In[ ]:




