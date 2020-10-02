#!/usr/bin/env python
# coding: utf-8

# # EDA and Model Building on Titanic dataset

# ## Content

# 1. Importing the required libraries
# 2. Loading and understanding the data
# 3. Analysis of the data and missing value treatment:Used imputation technique such as iterative imputer
# 4. Visualizing the data
# 5. Feature Engineering (Creating new features)
# 6. Outlier Analysis
# 7. Train-Test split
# 8. Scaling
# 9. Model Building and Evaluation
#     - Logistic Regression
#     - Random Forest
# 10. Submission

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## 1. Importing libraries

# In[ ]:


#Data Analysis
import numpy as np
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


#For missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Warnings
import warnings
warnings.filterwarnings('ignore')

#Preprocessing
from sklearn import preprocessing

#Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#Machine learning 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.api as sm

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

#Random forest
from sklearn.ensemble import RandomForestClassifier


#Vif
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ## 2. Loading and Data Understanding

# In[ ]:


# Lets import the train and test data and look into it

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_valid = pd.read_csv("/kaggle/input/titanic/test.csv")

merged = [train_df, test_df]

train_df.head()


# In[ ]:


# Looking into the datatype and count 

train_df.info()
print('-'*50)
test_df.info()


# #### Let's Gain some understanding
# 
# Since this is a past event and many of us know about titanic<br>
# Accordingly the main criteria for survival would be 
# 1. Age
# 2. Passenger's Class 
# 3. Sex

# In[ ]:


#Statistical Summary for numeric columns

train_df.describe()


# In[ ]:


#Statistical Summary for object columns columns

train_df.describe(include=['O'])


# In[ ]:


# Shape of dataset

print("Size of training data:{0}".format(train_df.shape))

print("Size of test data:{0}".format(test_df.shape))


# ## 3. Analysis of the data and missing value treatment

# In[ ]:


# Let's look into the missing value percentage 

print(round(100*(train_df.isnull().sum()/len(train_df)),2))
print('-'*40)
print(round(100*(test_df.isnull().sum()/len(test_df)),2))


# In[ ]:


plt.figure(figsize = (10,8))

sns.heatmap(train_df.corr(), annot= True, cmap = 'YlGnBu')


# Pclass have negative correlation with Fare i.e. as the fare price increases Pclass is lower

# In[ ]:


# we will drop Cabin Column from the data has it contain lot of missing values


train_df.drop('Cabin', inplace = True, axis = 1)
test_df.drop('Cabin', inplace = True, axis =1)


# In[ ]:


# Since we assume that the sex and Pclass is important let's look into the same

# sex
100*pd.crosstab(train_df.Survived, train_df.Sex, margins = True, margins_name = 'Total', normalize = True).round(4)


# Total survival rate is 38.4% out of which 26.2 % is of female

# In[ ]:


# Cross tabulation to see the M and F distribution across different PClass

100*pd.crosstab(train_df.Sex, train_df.Pclass, margins = True, margins_name = "Total", normalize = True).round(3)


# Here we can see that major of the population belongs to Passenger Class 3 around 55%<br>
# And the ratio of male to female is 13:7

# In[ ]:


100*pd.crosstab(train_df.Pclass,train_df.Survived, normalize = 'index').round(3)


# Passengers belonging to the upper class have the highest rate of survival

# In[ ]:


# Survival rate of Siblings/Spouses

100*train_df[["SibSp", 'Survived']].groupby(["SibSp"]).mean().sort_values(by = 'Survived', ascending = False).round(4)


# In[ ]:


# Survival rate of Parents/Children

100*train_df[["Parch", 'Survived']].groupby(["Parch"]).mean().sort_values(by = 'Survived', ascending = False).round(4)


# ## 4.Visualizing the data

# Since our main concern is regarding the Survival we will focusing on it more

# Now bucket the age variable into 5 groups defined as: 
# - "Age" <= 16: 0
# - 16  & <= 32 :1
# - 32 & <= 48 :2
# - 48 & <= 64 :3
# - "Age" > 64 :4

# In[ ]:


# Visualizing for age group

train_df['Age_Group'] = pd.cut(train_df.Age, bins = [0,16,32,48,64,100], labels = [0,1,2,3,4,])

plt.figure(figsize = (8,6))
sns.countplot('Age_Group', hue = 'Survived', data= train_df, palette="Set1")

plt.title("Survival distribution according to Age Group", fontsize = 20)
plt.ylabel('Frequency',fontsize = 15)
plt.xlabel('Age Groups', fontsize = 15)
plt.show()


# We can see that age group of 16-32 has the higest survival rate

# In[ ]:


# We will drop the age group column has we dont need it

train_df.drop('Age_Group', inplace = True, axis = 1)


# In[ ]:


# Visulizing for gender

plt.figure(figsize = (8,6))

sns.countplot('Sex', hue = 'Survived', data= train_df)

plt.title("Survival distribution according to Gender", fontsize = 20)
plt.ylabel('Frequency',fontsize = 15)
plt.xlabel('Gender', fontsize = 15)
plt.show()


# Clearly we can see that large number of females survived

# In[ ]:


# Visualising for different Pclass

g = sns.FacetGrid(train_df, row = 'Pclass', col = 'Survived', height=2.5, aspect=1.5)
g.map(plt.hist, 'Age', alpha = 0.5, bins = 20,edgecolor="black", color = 'g')


# - Most of the Passenger for Pclass 3 didnt survive
# - Infants from Pclass 3 and Pclass 2 survived
# - Passenegers for Pclass have mostly survived

# In[ ]:


# we will fill the missing values in embarked with mode

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace = True)


# In[ ]:


# Visualization for Embrakation

plt.figure(figsize = (8,6))

sns.countplot('Embarked', hue = 'Survived', data = train_df)

plt.title("Survival distribution according to Embrakation", fontsize = 20)
plt.ylabel('Frequency',fontsize = 15)
plt.xlabel('Port of Embarkation', fontsize = 15)
plt.show()


# C = Cherbourg, Q = Queenstown, S = Southampton


# - Major of the people are from Southampton, but highest people who didnt surive are also from Southampton

# In[ ]:


100*pd.crosstab(train_df.Embarked, train_df.Survived, normalize = 'index').round(3)


# - Looking at this we can say that major of the survived people are from Cherbourg 

# In[ ]:


# Let's look into different fare prices for different embarkation port

plt.figure(figsize = (8,6))
sns.barplot(y = 'Embarked', x = 'Fare', data = train_df, hue = 'Pclass', palette = 'Set1', ci = None)

plt.title("Fair prices for various Pclass from different Embrakation port", fontsize = 20)
plt.ylabel('Embrakation Port',fontsize = 15)
plt.yticks([0,1,2], ['Southampton', 'Cherbourg','Queenstown'])
plt.xlabel('Fare Price', fontsize = 15)
plt.show()


# - The fair prices are different for different embarkation port
# - Cherbourg has the highest number of Passenegers and that to from upper class segment

# In[ ]:


# Looking into average fair price according to gender and port embarked`

pd.pivot_table(train_df, index = ['Sex','Embarked'], columns = 'Pclass', values = 'Fare', aggfunc = np.mean).round(2)


# - Prices at Cherbourg are maximum with average for Male 93.54 and Female 115.64 dollors

# In[ ]:


#Checking weather duplicate tickets were issued

duplicate = train_df['Ticket'].duplicated().sum()

print("Number of duplicate tickets issued are {0} which contributes around {1}%".format(duplicate, 100*round(duplicate/len(train_df),2) ))


# In[ ]:


# Dropping Ticket and Passenger ID from data frame as it doesnt contribute for analysis

train_df.drop(['PassengerId', 'Ticket'], inplace = True, axis = 1)

test_df.drop(['PassengerId', 'Ticket'], inplace = True, axis = 1)


# In[ ]:


# We will replace male to 1 and female to 0 & make sex column numeric

train_df['Sex'].replace(['female', 'male'], [0,1], inplace = True)
test_df['Sex'].replace(['female','male'], [0,1], inplace = True)


# In[ ]:


# Label encoding embracked column

le = preprocessing.LabelEncoder()

train_df['Embarked'] = le.fit_transform(train_df['Embarked'])
test_df['Embarked'] = le.fit_transform(test_df['Embarked'])


# ## 5. Feature Engineering

# In[ ]:


# We will extract a new feature called Title from name

for df in merged:
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand = False)
    
pd.crosstab(train_df['Title'], train_df['Sex'])


# In[ ]:


#Replacing the least repeated keywords with others

for df in merged:
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')
    
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
100*pd.crosstab(train_df.Title, train_df.Survived, normalize = 'index').round(3)


# In[ ]:


# Encoding Title column

for df in merged:
    df['Title'] = le.fit_transform(df['Title'])


# In[ ]:


# Dropping the Name column from the dataset

for df in merged:
    df.drop('Name', axis = 1, inplace = True)


# In[ ]:


# Filling the misssing value for fare in test dataset with median

test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)


# In[ ]:


# Stroting the column names

train_columns = train_df.columns
test_columns = test_df.columns


# In[ ]:


# Filling the missing values for age with Iterative Imputer for train

ii = IterativeImputer(initial_strategy='median', min_value = 0, max_value = 80, random_state = 42)

train_df_clean = pd.DataFrame(ii.fit_transform(train_df))
train_df_clean.columns = train_columns


# In[ ]:


# Similiarly for test

test_df_clean = pd.DataFrame(ii.fit_transform(test_df))
test_df_clean.columns = test_columns


# In[ ]:


# Restoring the datatype to there original format

main = [train_df_clean, test_df_clean]

for df in main:

    for i in ['Pclass', 'Sex', 'SibSp', 'Parch','Embarked','Title']:
        df[i] = pd.to_numeric(df[i])
        df[i] = df[i].astype(int)


# In[ ]:


# Changing the datatype of survived in training dataset

train_df_clean['Survived'] = pd.to_numeric(train_df_clean['Survived'])
train_df_clean['Survived'] = train_df_clean['Survived'].astype(int)


# In[ ]:


train_df_clean.head()


# In[ ]:


# Creating a new feature called 'Familysize'

for df in main:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1


# In[ ]:


# Family size and surival chances

100 * pd.pivot_table(data = train_df_clean, index = 'FamilySize', values = 'Survived', aggfunc = np.mean).sort_values(by = 'Survived', ascending = False).round(3)


# In[ ]:


#plotting graph to see surival rate and family size

plt.figure(figsize = (8,6))

sns.lineplot(data = train_df_clean, x = 'FamilySize', y = 'Survived',ci = None, marker="o")

plt.title("Family Size vs Survival Rate", fontsize = 20)
plt.ylabel('Survival Rate',fontsize = 15)
plt.xlabel('Family Size', fontsize = 15)
plt.show()


# - We note that family size of 4 has highest survival rate whereas family greater than 8 has 0% survival rate

# In[ ]:


# Creating another attribute called Is_alone

for df in main:
    df['Is_Alone'] = 0
    df.loc[df['FamilySize']==1, 'Is_Alone'] = 1
    
# 1 = alone & 0 = Not_alone


# In[ ]:


# Let's look at survival rate of alone passaneger

100 * pd.crosstab(train_df_clean['Is_Alone'], train_df_clean['Survived'], normalize = 'index').round(3)


# - Survival rate of solo passeneger if 30.4%
# - Survival rate of family is 50.6%

# In[ ]:


plt.figure(figsize = (8,6))

sns.barplot(data = train_df_clean, x = 'Is_Alone', y = 'Survived', ci = None)

plt.title("Chances of Solo Passeneger Surviving", fontsize = 20)
plt.ylabel('Survival Rate',fontsize = 15)
plt.xlabel('Type of Passeneger', fontsize = 15)
plt.xticks([0,1], ['Family', 'Solo Passenger'])
plt.show()


# - Here we can clearly see that the chances of solo passeneger survival is very low comapred to a family

# In[ ]:


plt.figure(figsize = (15,10))

sns.heatmap(train_df_clean.corr(), annot = True)


# In[ ]:


# based on the above correaltion we will drop SibSp, Parch, Family_size

for df in main:
    df.drop(['SibSp','Parch','FamilySize'], inplace = True, axis =1)


# ## 6. Outlier Analysis

# In[ ]:


# Plotting box plot for all the variables and checking for outliers


plt.figure

for i, col in enumerate(train_df_clean.columns):
    plt.figure(i)
    sns.boxplot(train_df_clean[col])


# In[ ]:


# Doing small outlier treatment for fare attribute

train_df_clean.drop(train_df_clean.index[train_df_clean['Fare'] > 300], inplace = True)


# **This are the final data frame**

# In[ ]:


# This are the final data frame

train_df_clean.head()


# In[ ]:


test_df_clean.head()


# In[ ]:


# Let's look into class imbalance of our target varaible i.e. survived

pd.crosstab(train_df_clean['Survived'], train_df_clean['Survived'], normalize = True).round(4)*100


# - To look at this we have slight class imbalance, we will handle this later using ``weight of class`` method while building model

# ## 7. Train-Test split

# In[ ]:


X_train = train_df_clean.drop('Survived', axis =1)

y_train = train_df_clean['Survived']

X_test = test_df_clean


# In[ ]:


# Storing the column names  for train and test
X_train_col = X_train.columns

X_test_col = X_test.columns


# In[ ]:


# We will convert the data into array as it will optimize more

X_train, y_train = np.array(X_train), np.array(y_train)


# ## 8. Scaling

# In[ ]:


scaler = MinMaxScaler()

#for train data set
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns = X_train_col)


# In[ ]:


#Scaling test dataset

X_test = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test, columns = X_test_col)


# In[ ]:


#To use later for random forest

rf_X_train = X_train.copy()
rf_X_test = X_test.copy()


# **We will build 2 models:**
# 1. Logestic Regression
# 2. Random Forest

# ## Modeling Builidng and Evaluation

# ## 1. Logistic Regression

# In[ ]:


# Finding the optimum hyper paramters

## Different parameters to check
max_iter=[100,110,120,130,140]
C_param_range = [0.001,0.01,0.1,1,10,100]
folds = KFold(n_splits = 5, shuffle = True, random_state = 42)

## Setting the paramters
param_grid = dict(max_iter = max_iter, C = C_param_range)

## Setting model
log = LogisticRegression(penalty = 'l2')

## Set up GridSearch for score metric

grid_search = GridSearchCV(estimator = log, param_grid = param_grid, cv = folds, n_jobs = -1, 
                           return_train_score = True, scoring = 'accuracy')

## Fitting
grid_search.fit(X_train, y_train)


# In[ ]:


# Looking at the best parameters

print("The best accuracy score is {0:2.3} at {1}".format(grid_search.best_score_, grid_search.best_params_))


# In[ ]:


# Setting model with optimum parameters

log = LogisticRegression(penalty = 'l2', C = 10, max_iter =100, class_weight = 'balanced')


# In[ ]:


# Fitting the model

log_fit = log.fit(X_train, y_train)


# In[ ]:


# Predicting on test data set

y_test_pred = log_fit.predict(X_test)


# In[ ]:


#Accuracy score for training data

print("Accuracy score for training data is: {0}".format(round(log_fit.score(X_train, y_train) * 100, 2)))


# #### Looking into p-values and vif and selecting the appropriate features for our model
# **NOTE**: This method will reduce the features in our model, I am just trying it out to see if this results in better accuracy

# In[ ]:


X_train_sm = sm.add_constant(X_train)
log_sm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = log_sm.fit()
print(res.summary())


# - We can that we have high p-values for Fare, Title and Is_Alone

# In[ ]:


X_train.shape[1]


# In[ ]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range (X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif.reset_index(drop = True, inplace = True)
vif


# - Title and Age have high VIF

# In[ ]:


# Based on the above values we will drop Title

X_train.drop('Title', axis = 1, inplace = True)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
log_sm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = log_sm.fit()
print(res.summary())


# - We have high p-values for age and and Is_alone

# In[ ]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range (X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif.reset_index(drop = True, inplace = True)
vif


# - All the VIF values are less than 5 and are in the moderate range

# In[ ]:


# Based on the observation we will drop Is_alone

X_train.drop('Is_Alone', axis =1, inplace = True)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
log_sm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = log_sm.fit()
print(res.summary())


# In[ ]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range (X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif.reset_index(drop = True, inplace = True)
vif


# - We eill drop Fare has it had p-value greater than 0.05

# In[ ]:


#Dropping Fare

X_train.drop('Fare', inplace = True, axis =1)


# In[ ]:


X_train_sm = sm.add_constant(X_train)
log_sm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = log_sm.fit()
print(res.summary())


# In[ ]:


# Make a VIF dataframe for all the variables present

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range (X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif.reset_index(drop = True, inplace = True)
vif


# - All the p-values and VIF are less and we will procceed with this now for our predictions

# In[ ]:


#Prediciting the values of X train

y_train_pred = res.predict(sm.add_constant(X_train))


# In[ ]:


y_train_pred_final = pd.DataFrame({'Survived': y_train, 'Survived_Proab':y_train_pred})
y_train_pred_final['Survived_Proab'] = round(y_train_pred_final['Survived_Proab'],2)
y_train_pred_final.head(2)


# In[ ]:


# ROC function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


#Storing the values for FPR, TPR and thersolds

fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final['Survived'], y_train_pred_final['Survived_Proab'], drop_intermediate = False )


# In[ ]:


# Call the ROC function

draw_roc(y_train_pred_final['Survived'], y_train_pred_final['Survived_Proab'])


# In[ ]:


# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final['Survived_Proab'].map(lambda x: 1 if x > i else 0)
y_train_pred_final.head(2)


# In[ ]:


# Let's create a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final['Survived'], y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[ ]:


# Plotting sensitivity, accuracy and specificity

sns.set()
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# - Looking at this we can say 0.4 is the optimum threshold point

# In[ ]:


#Creating final predicated column

y_train_pred_final['final_predicted'] = y_train_pred_final['Survived_Proab'].map( lambda x: 1 if x > 0.4 else 0)

y_train_pred_final.head(2)


# In[ ]:


#Looking into the accuray of training data set

print("Accuracy : {:2.2}".format(metrics.accuracy_score(y_train_pred_final['Survived'], y_train_pred_final['final_predicted'])))


# #### Making predictions on Test data

# In[ ]:


# Drop the required columns from X_test as well

X_test.drop(['Fare', 'Is_Alone', 'Title'], axis =1, inplace = True)


# In[ ]:


# Making predictions on test data set

y_test_pred1 = res.predict(sm.add_constant(X_test))


# In[ ]:


# Converting y_pred to a dataframe

y_pred1 = pd.DataFrame(y_test_pred1, columns = ['Survived_Proab'])
y_pred1.reset_index(drop = True, inplace = True)
y_pred1


# In[ ]:


# Make predictions on the test set using 0.4 as the cutoff

y_pred1['final_predicted'] = y_pred1['Survived_Proab'].map(lambda x: 1 if x > 0.4 else 0)
y_pred1.head()


# In[ ]:


#Top features and there importance

IP = pd.DataFrame(res.params , columns = ['Importance'])
IP.reset_index(inplace = True)
IP.columns = ['Features', 'Importance']
IP.drop(IP.index[0], inplace = True)
IP = IP.sort_values(by = 'Importance')
IP.reset_index(drop = True, inplace =True)
IP['Importance'] =  round(IP['Importance'], 2)
IP.head(10)


# ## Random Forest

# In[ ]:


# Instantiate

rf = RandomForestClassifier()

#Fitting 
rf.fit(rf_X_train, y_train)


# In[ ]:


# Setting up folds
folds = KFold(n_splits = 5, shuffle = True, random_state = 42)

#Setting up parameters to check
param_grid = {
    'max_depth': [4,8,10],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100,200, 300], 
    'max_features': [4,5,6,7]
}

# Create a based model
rf = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = folds, n_jobs = -1,verbose = 1)


# In[ ]:


# Fit the grid search to the data

grid_search.fit(rf_X_train, y_train)


# In[ ]:


# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',round(grid_search.best_score_,2),'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters

rf = RandomForestClassifier(bootstrap=True,class_weight = "balanced", criterion = 'gini',
                             max_depth=4,
                             min_samples_leaf=100, 
                             min_samples_split=200,
                             max_features=4,
                             n_estimators=200)
rf_fit = rf.fit(rf_X_train, y_train)


# In[ ]:


#Predicitng on test

rf_y_pred_test = rf_fit.predict(rf_X_test)


# ## 10. Submission

# In[ ]:


#File submission

submission = pd.DataFrame({"PassengerId": test_valid["PassengerId"], "Survived": rf_y_pred_test})

submission.to_csv('submission_.csv', index=False)


# If you found this notebook helpful or you just liked it , some upvotes would be very much appreciated - That will keep me motivated :)
