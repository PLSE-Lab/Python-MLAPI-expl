#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# 
# **Title: TITANIC SURVIVAL PREDICTION**
# 
# Mikel Kengni / December 2018
# 
# Plan:
# 
# **First things First: Snacks and Chips checked, Coffee checked**
# 
# * **Introduction**
# 
#  1. Import the libraries
#  
#  2. Explore the dataset
#  
#  3. Feature exploration and Cleaning
#  
#  
# * **Correlation with predictor**
# 
# 
# * **Feature Engineering**
# 
#  1. Encoding the categorical data
# 
#  2. Feature scaling and Normalization
# 
#  3. Checking for skewness in the numerical data
# 
# 
# * **Prediction**
# 
#  1. Split the data into train and Validation
# 
#  2. Build the model
# 
#  3. Feature importance
# 
#  4. Predictions
# 
#  5. Stacking
#     

# # Introduction:

# The Survived column is the target variable and is the feature we are going to predict. If Suvival = 1 the passenger survived, otherwise he's dead.
# 
# **The other variables describe the passengers are:**
# 
# * PassengerId: id given to each traveler on the boat
# * Pclass: The passenger class. It has three possible values: 1,2,3 (first, second and third class)
# * Name of the passeger:
# * Sex: Either  Male or Female
# * Age:
# * SibSp: number of siblings and spouses traveling with the passenger
# * Parch: number of parents and children traveling with the passenge
# * Ticket number: 
# * Ticket Fare
# * Cabin number
# * Embarkation Gate. This describe three possible gates on the Titanic from which the people embarked(gates: S,C,Q)

# # 1- Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# # 2- Loading and Reading data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print('The size of the training set: ', train.shape)
print('The size of the test set is: ' ,test.shape)


# # 3- Feature exploration and Cleaning

# In[ ]:


test.head()


# In[ ]:


train.describe()


# Observation:
# We can see from the describe method obove that:
# * there were about 891 passengers on board(count = 891 for most of the columns)
# * There may be about 177(891 - 714) missing data in Age column 

# In[ ]:


# To get more information about the data you are working with, it is good to use the info() method

train.info()


# Here we can observe that there are some missing data in the Cabin column( about 687 missing data) and the enbarked column has 2 missing data. 

# In[ ]:





# Since we will need the passenger ID in the submission file, we save it for easy access later

# In[ ]:


Passenger_Id = test['PassengerId']


# In[ ]:


#we keep this for when we will be separating DATA back into train and test 

train_rowsize = train.shape[0]
test_rowsize = test.shape[0]


# In[ ]:


#The different data types 
train.dtypes.value_counts()


# In[ ]:


test.dtypes.value_counts()


# In[ ]:





# We observe that the train set has 5 categorical features(the object features) and 7 numerical features while the test data has 4 categorical features and 7 numerical features. We will have to separate the categorical feature from the numerical features and then use and encoding method to convert the categorical features to numerical features. 

# 

# In general, to do feature enginnering easily, i like to merge the train and the test data. This facilitates working on both datasets once instead of having to do one thing on the train data and the same thing on the test data. But it is totally up to you to merge the 2 datasets or not, either ways work.

# In[ ]:


data = pd.concat((train, test))

# we drop the preditor from the data 
data.drop('Survived', axis = 1, inplace = True)
data.drop('PassengerId', axis = 1, inplace = True)


# In[ ]:


data.hist(figsize=(10,10));


# We can see that features like fare, sibsp and parch are very much skewed to the left. We will have to normalise them at some point. It is always better to have them symmetrcal or centered rather than have them tilting ton one side as this may affect your final results.

# In[ ]:


#we find out the percentage of survivers by gender
data.Sex.value_counts(normalize = True)


# We can observe that 64 percent of all the titenic passengers were male. and 36 percent were females. Of the 64 percent of mlae, lets find out how what percentage survived.

# In[ ]:


embarked_counts = data.Embarked.value_counts(normalize = True)
embarked_counts


# In[ ]:


embarked_counts.plot(kind='bar')
plt.title("Passengers per boarding gates");


# # Visualisation:

# Let's visualize survival based on gender:

# In[ ]:


train['Died'] = 1 - train['Survived']
train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind = 'bar', figsize = (10, 5), stacked = True);


# Looks like males were more likely to Succumb than female. :|

# Is there any relation ship between the fare tickets and the Passenger Vlass?

# In[ ]:


ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(10, 5), ax = ax);


# Actually the fare price is correlated with the Passenger class. Those who paid more where in Pclass 1 and as we can see in the chart below, passengers with lower fare were more likely to succumb.

# In[ ]:


plt.figure(figsize=(10, 5))
plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']], 
         stacked=True,bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend();


# # Imputing missing values

# Lets take a look at the number of missing data in both the train and the test sets

# In[ ]:


#Here is a list of all the features with Nans and the number of null for each features
null_values = data.columns[data.isnull().any()]
null_features = data[null_values].isnull().sum().sort_values(ascending = False)
missing_data = pd.DataFrame({'No of Nulls' :null_features})
missing_data


# For the newcomers: This is basically how to find the missing data. since i merged both train and test data. The number of missing data above is the sum of the number of missing data in the train and in the test sets.

# In[ ]:


test.isnull().sum()


# In[ ]:


train.isnull().sum()


# In[ ]:


#Fare, embarked, Cabin and Age all have missing values. Lets plot the missing values
import warnings
warnings.filterwarnings('ignore')            #to silence warnings

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('dark')

plt.figure(figsize= (10, 5))
plt.xticks(rotation='90')
ax = plt.axes()
sns.barplot(null_features.index, null_features)
ax.set(xlabel = 'Features', ylabel = 'Number of missing values', title = 'Features with Missing values');


# The chart above may not shown any values for the fare and embarked but they do have some missing values.
# 
# Having missing values or NaNs in your dataset can couse error with come machine learning algorithms. We could try to replace the missing values with
# 
# * A constant value that has meaning within the domain like a 0, or
# * We could replace them with the mean, median or mode of values
# * With another values selected from a random record or another predicted model
# 
# We will start imputing the missing values in our dataset
# 

# **EMBARKED:**
# 
# We observed that there was only missing data for the embarked feature  and the most used 
# gate for embarkement is was 'S' so i will fill the missing embarked with the median of all boarding gates i.e 'S'

# In[ ]:


data["Embarked"] = data["Embarked"].fillna('S')


# **FARE:**
# 
# lets take a look at the person with the missing fare    

# In[ ]:


data[data['Fare'].isnull()]


# we can see that the passenger is a male in his 60s who embarked at gate 'S', with a pclass = 3. Lets fill his fare with the median of all passengers of pclass == 3 who embarked at gate 'S'. 

# In[ ]:


def fill_missing_fare(df):
    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()
#'S'
       #print(median_fare)
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df

data=fill_missing_fare(data)


# ** Age: **
#     
# lets plot the age distribution to get an idea of what we are working with

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_context('notebook')
sns.set_style('ticks')
sns.set_palette('dark')

plt.figure(figsize= (10, 5))

ax = sns.distplot(data["Age"].dropna(),   #plot only the numerical data
                  color = 'green',
                 kde = False)    
ax.grid(True)

ax.set(xlabel = 'Age', ylabel = 'Number of people', title = 'Age range of Passengers');


# observation:
# * Age groups range from 0 to 80
# * The largest number of age group was 20 ~ 22
# * About 100 children( ages from 0  - 18), we shall find the exact number later
# * less that 100 elderly people(ages from 60 -80)

# In[ ]:


# we will genrate a set of random values from 0 to 80 and missing ages with any one of these values

sizeof_null = data["Age"].isnull().sum()
rand_age = np.random.randint(0, 80, size = sizeof_null)


# In[ ]:


# fill NaN values in Age column with random values generated
   
age_slice = data["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
data["Age"] = age_slice
data["Age"] = data["Age"].astype(int)


# Lets creat age groups. we will divide the passemgers into children(0 - 18), young adults (19 - 35), late adult (36 - 60) and the elderly (61 - 80)

# In[ ]:


data['Age'] = data['Age'].astype(int)
data.loc[ data['Age'] <= 18, 'Age'] = 0
data.loc[(data['Age'] > 18) & (data['Age'] <= 35), 'Age'] = 1
data.loc[(data['Age'] > 35) & (data['Age'] <= 60), 'Age'] = 2
data.loc[(data['Age'] > 60) & (data['Age'] <= 80), 'Age'] = 3

data['Age'].value_counts()


# In[ ]:


data.sample(10)


# **Cabin:**

# In[ ]:


data['Cabin'].dropna().sample(10)


# If we can extract the letters in front of every cabin, which refers to the **Deck** on the ship, we can have the position of a passenger on the titanic. 

# In[ ]:


data["Deck"]=data['Cabin'].str[0]

data['Deck'].unique()


# In[ ]:


data['Deck'] = data['Deck'].fillna('H')   # replacing the nan with H


# In[ ]:


data['Deck'].unique()


# In[ ]:


# we include a new feauture, the Familysize including the passengers
data["FamilySize"] = data["SibSp"] + data["Parch"]+ 1
data['FamilySize'].value_counts()


# In[ ]:





# In[ ]:


data.loc[ data['FamilySize'] == 1, 'FSize'] = 'Single family'
data.loc[(data['FamilySize'] > 1) & (data['FamilySize'] <= 5), 'FSize'] = 'Small Family'
data.loc[(data['FamilySize'] > 5), 'FSize'] = ' Extended Family'


# In[ ]:


data.head()


# # Converting Categoricals to numericals

# In[ ]:


le = LabelEncoder()

data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])
data['Deck'] = le.fit_transform(data['Deck'])
data['FSize'] = le.fit_transform(data['FSize'])


# In[ ]:


data['Sex'].unique()


# In[ ]:


data.dtypes.value_counts()


# In[ ]:


data = data.drop(['Name', 'Ticket','Cabin',], axis = 1)


# # How Skewed is our dataset?

# Lets take alook at how skewed our dataset is. Just a little background:
# 
# * In statistics, skewness is a measure of the asymmetry of the probability distribution of a random variable about its mean. In other words, skewness tells you the amount and direction of skew (departure from horizontal symmetry). The skewness value can be positive or negative, or even undefined. If skewness is 0, the data are perfectly symmetrical, although it is quite unlikely for real-world data. As a general rule of thumb:
# 
# * If skewness is less than -1 or greater than 1, the distribution is highly skewed.
# * If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.
# * If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
# 
# references:https://help.gooddata.com/display/doc/Normality+Testing+-+Skewness+and+Kurtosis

# In[ ]:


#we check for skewness in  data

skew_limit = 0.75
skew_vals = data.skew()

skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skewness'})
            .query('abs(Skewness) > {0}'.format(skew_limit)))

skew_cols


# I will be using  a nplog1p transformation to standardise the skew columns.

# In[ ]:


print("There are {} skewed numerical features to  transform".format(skew_cols.shape[0]))


# I will be using a numpy  log1p to transform  the very skewed data. Just to give a sense of what numpy log1p tranformation does in a skewed dataset, We are going to design a before and after log1p .

# In[ ]:


tester = 'Deck'
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(16,5))
#before normalisation
data[tester].hist(ax = ax_before)
ax_before.set(title = 'Before nplog1p', ylabel = 'Frequency', xlabel = 'Value')

#After normalisation
data[tester].apply(np.log1p).hist(ax = ax_after)
ax_after.set(title = 'After nplog1p', ylabel = 'Frequency', xlabel = 'Value')

fig.suptitle('Field "{}"'.format(tester));


# The before normalisation shows a right skewed distribution or positive skewed feature. After nplog1p, the distribution is more symmetrical.
# 
# **BTW: You can change the tester and see the different features before and after nplog1p normalization**
# 

# In[ ]:


skewed = skew_cols.index.tolist()
data[skewed] = data[skewed].apply(np.log1p)


# Why do we normalise datasets:
#     
# The skewed data have to be normalise because many of the algorithms in data assume that the data science is normal and calculate various stats assuming this. So the more the data is close to normal the more it fits the assumption.
# if you want to read more:
# https://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning

# # Correlations:

# In[ ]:


# Correlation between the features and the predictor- Survived
predictor = train['Survived']
correlations = data.corrwith(predictor)
correlations = correlations.sort_values(ascending = False)
# correlations
corrs = (correlations
            .to_frame()
            .reset_index()
            .rename(columns={'level_0':'feature1',
                                0:'Correlations'}))
corrs


# In[ ]:


plt.figure(figsize= (10, 5))
ax = correlations.plot(kind = 'bar')
ax.set(ylabel = 'Pearson Correlation', ylim = [-0.4, 0.4]);


# # Modelling 

# In[ ]:


#importing libraries
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier


# In[ ]:


# First i need to break up my data into train and test
train_new = data[:train_rowsize]
test_new = data[train_rowsize:]
test_new.shape


# In[ ]:


train_new.dtypes.value_counts()


# In[ ]:


train_new.head()


# In[ ]:


test_new.dtypes.value_counts()


# In[ ]:


test_new.head()


# In[ ]:





# * **Defining a 5 fold cross validation split Strategy for the train_new set **

# In[ ]:


n_folds = 5

kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train_new)


#  * ** Define the Error method: **
#         
# I will be using the Root mean square error method from sklearn 

# In[ ]:


y_train = train.Survived
n_folds = 5
    
def f1_score (model): 
    kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train_new)
    rmse = np.sqrt(cross_val_score(model, train_new, y_train, scoring = 'f1', cv = kf))
    # f1 because it is the sweet spot between recall and precision
    return (rmse)


# # Ensembling- Stacking

# Stacking is a mechanism that uses the 'Unity is Strength' principle. Basically, it uses multiple machine learning algorithms at the same time, to obtain a better performance and thus a better prediction
# 
# Stacking Technique involves the following Steps:-
# 
# * Split the training data into 2 disjoint sets
# * Train several Base Learners on the first part
# * Test the Base Learners on the second part and make predictions
# * Using the predictions from (3) as inputs,the correct responses from the output,train the higher level learner or meta level Learner
# 
# To learn more about stackin, You can read this: https://medium.com/@gurucharan_33981/stacking-a-super-learning-technique-dbed06b1156d 

# * ** Defining My Base models **

# In[ ]:


logreg = LogisticRegression()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()
xgb = XGBClassifier()
lgbm = LGBMClassifier()


# * ** How does our Base models score **

# In[ ]:


# from sklearn.metrics import SCORERS
# print(SCORERS.keys())


# In[ ]:


score = f1_score(logreg)
print("\nLogistic regression score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))


# In[ ]:


score = f1_score(rf)
print("\nRandom Forest score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))


# In[ ]:


score = f1_score(gboost)
print("\nGradient Boosting Classifier score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))


# In[ ]:


score = f1_score(xgb)
print("\neXtreme Gradient BOOSTing score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))


# In[ ]:


score = f1_score(lgbm)
print("\nLight Gradient Boosting score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))


# * ** Voting Classifier Ensembling **

# We can put all our classifiers togeter into one classifier and use it to predict our test set, provided it score better than all the other individual models. As shown below, the voting classifier scores far better than the other classifiers. :)

# In[ ]:


all_classifier = VotingClassifier(estimators=[('logreg', logreg), ('rf', rf), 
                                              ('gboost', gboost), ('xgb', xgb),
                                             ('lgbm', lgbm)], voting='soft')

VC = all_classifier.fit(train_new, y_train)


# In[ ]:





# In[ ]:


score = f1_score(VC)
print("\nVoting Classifier score: mean = {:.4f}   std = ({:.4f}) \n".format(score.mean(), score.std()))


# In[ ]:


prediction = VC.predict(test_new)


# * ** Submission **

# In[ ]:


titanic_submission = pd.DataFrame ({"PassengerId": test["PassengerId"],
                             "Survived": prediction})
titanic_submission.to_csv('Titanic_Submission.csv', index = False)

titanic_submission.sample(10)


# kernels used/ References:
#     
# * https://www.kaggle.com/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline?scriptVersionId=1124380
# * https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
# * https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# * https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
#     

# In[ ]:




