#!/usr/bin/env python
# coding: utf-8

# # Preparation
# Import necessary library and load the train and test data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

#import train and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
dataset = pd.concat([train, test], ignore_index = True)
#Retrieve Passenger ID from test set, used for submission
PassengerId = test['PassengerId']


# Check null values and missing values

# In[ ]:


dataset = dataset.fillna(np.nan)
dataset.isnull().sum()


# From the result, Age and Cabin have a lot of missing values

# In[ ]:


#Check missing values in train set
train.info()
train.isnull().sum()


# In[ ]:


# check the first five information of the train set
train.head()


# In[ ]:


# Check the data types of every column
train.dtypes


# In[ ]:


# Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset's distribution
train.describe()


# # Data visualization
# ## Sex Feature
# female has higher survival rate than male.

# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train, palette='Set3')
print("Percentage of females that could survive: %.2f" %(train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True)[1]*100))
print("Percentage of females that could survive: %.2f" %(train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True)[1]*100))


# ## Pclass feature
# The higher the class is, the high probability survive

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=train, palette='Set3')
print("Percentage of Pclass = 1, survived probability: %.2f" %(train['Survived'][train['Pclass']==1].value_counts(normalize = True)[1]*100))
print("Percentage of Pclass = 2, survived probability: %.2f" %(train['Survived'][train['Pclass']==2].value_counts(normalize = True)[1]*100))
print("Percentage of Pclass = 3, survived probability: %.2f" %(train['Survived'][train['Pclass']==3].value_counts(normalize = True)[1]*100))


# ## SibSp Feature
# With a suitable number of siblings and spouse, he/she will have a high survival rate.

# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=train, palette='Set3')


# ## Parch Feature 
# With a suitable number of parents and children, he/she will have a high survival rate.

# In[ ]:


sns.barplot(x="Parch", y="Survived", data=train, palette='Set3')


# ## Age Feature
# Child and adolecent will have a higher survival rate

# In[ ]:


age = sns.FacetGrid(train, hue="Survived",aspect=2)
age.map(sns.kdeplot,'Age',shade= True)
age.set(xlim=(0, train['Age'].max()))
age.add_legend()


# ## Fare Feature
# Passengers who paid higher fare had higher survival rate.

# In[ ]:


fare = sns.FacetGrid(train, hue="Survived",aspect=2)
fare.map(sns.kdeplot,'Fare',shade= True)
fare.set(xlim=(0, 200))
fare.add_legend()


# Add a new feature Title
# ## Title Feature
# Retrieve the title from passengers name, classify them into six kinds, which are officer, royalty, Mrs, Miss, Mr, and Master.
# Mrs, Miss and Royalty have higher survival rate than other titles.

# In[ ]:


dataset['Title'] = dataset['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
dataset['Title'] = dataset['Title'].map(Title_Dict)
sns.barplot(x="Title", y="Survived", data=dataset, palette='Set3')


# ## FamilyLabel
# Add new feature FamilyLabel
# Calculate the family size = Sibsp+Parch+1
# We can see that middle size family has higher survival rate.
# Classify the family size into three kinds, which are middle size(2-4), small or large size(1 or 5-7) and larger size( >7 ).

# In[ ]:


dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
sns.barplot(x="FamilySize", y="Survived", data=dataset, palette='Set3')


# In[ ]:


# Based on the family size, classify them into three groups
def Family_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
dataset['FamilyLabel']=dataset['FamilySize'].apply(Family_label)
sns.barplot(x="FamilyLabel", y="Survived", data=dataset, palette='Set3')


# ## Deck Feature
# Add Deck feature:
# Fill the missing cabin as Unknown
# Retrieve the capital words as Deck number

# In[ ]:


dataset['Cabin'] = dataset['Cabin'].fillna('Unknown')
dataset['Deck']= dataset['Cabin'].str.get(0)
sns.barplot(x="Deck", y="Survived", data=dataset, palette='Set3')


# ## TicketGroup Feature
# Calculate the number of passengers who has the same ticket number
# We can see that 2-4 passengers own the same ticket number have higher survival rate.
# Classify them into three kinds.

# In[ ]:


Ticket_Count = dict(dataset['Ticket'].value_counts())
dataset['TicketGroup'] = dataset['Ticket'].apply(lambda x:Ticket_Count[x])
sns.barplot(x='TicketGroup', y='Survived', data=dataset, palette='Set3')


# In[ ]:


# Classify the TicketGroup into three kinds
def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

dataset['TicketGroup'] = dataset['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=dataset, palette='Set3')


# # Fill the missing values
# ## Age feature
# From the previous process, there are 263 missing values in the whole data set. Using Sex, Title and Pclass feature construct the Random Forest model to fill the missing values of age.

# In[ ]:


# Fill the missing age value, use feature Pclass, Sex and Title and random forest regressor model to predict 
age = dataset[['Age','Pclass','Sex','Title']]
age = pd.get_dummies(age)
# print(age)
known_age = age[age.Age.notnull()].as_matrix()
null_age = age[age.Age.isnull()].as_matrix()
x = known_age[:, 1:]
y = known_age[:, 0]
rf = RandomForestRegressor(n_jobs=-1)
rf.fit(x, y)
predictedAge = rf.predict(null_age[:, 1:])
dataset.loc[(dataset.Age.isnull()),'Age'] = predictedAge


# ## Embarked feature 
# There are only 2 missing values in the Embarked feature. The passengers who misses Embarked information all have Pclass == 1, and Fare == 80. Calculate the median of fare that Pclass == 1 and Embarked == S, C, Q, respectively.
# Since the median that Embarked == C has the closest fare. Fill the missing value of Embarked as C.

# In[ ]:


dataset[dataset['Embarked'].isnull()]


# In[ ]:


C = dataset[(dataset['Embarked']=='C') & (dataset['Pclass'] == 1)]['Fare'].median()
print(C)
S = dataset[(dataset['Embarked']=='S') & (dataset['Pclass'] == 1)]['Fare'].median()
print(S)
Q = dataset[(dataset['Embarked']=='S') & (dataset['Pclass'] == 1)]['Fare'].median()
print(Q)
dataset['Embarked'] = dataset['Embarked'].fillna('C')


# ## Fare feature
# There is only one missing values in the data set. Use the median of fare that Embarked == S and Pclass == 3.

# In[ ]:


dataset[dataset['Fare'].isnull()]


# In[ ]:


fare=dataset[(dataset['Embarked'] == "S") & (dataset['Pclass'] == 3)].Fare.median()
dataset['Fare']=dataset['Fare'].fillna(fare)


# Retrieve the surname of the passenger, classify the passengers into the same group if they have the same surname. Retrieve the number of female, male and child from the groups that have more than one person.

# In[ ]:


dataset['Surname']=dataset['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(dataset['Surname'].value_counts())
dataset['FamilyGroup'] = dataset['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=dataset.loc[(dataset['FamilyGroup']>=2) & ((dataset['Age']<=12) | (dataset['Sex']=='female'))]
Male_Adult_Group=dataset.loc[(dataset['FamilyGroup']>=2) & (dataset['Age']>12) & (dataset['Sex']=='male')]


# Most of the average survival rate of female and chile group  are 1 or 0, which means that the women or children in the same group all survive, or all die.

# In[ ]:


Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']
Female_Child


# Most of the man groups' average survival rate are 0 or 1.

# In[ ]:


Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']
Male_Adult


# From previous process, a conclusion can be drawn that women and children have high survival rate, while men have lower survival rate. We retrieve the outliers to process. We set a dead group that women and children group have 0 survival rate. And set survived group for the men's group whose survival rate is 1. We can guess that in dead group, women and children have lower survival rate. In survival group, men have higher survival rate.

# In[ ]:


Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)


# In order to classify the samples in these two outliers groups, modigy the Age, Title and Sex of the samples in these two groups.

# In[ ]:


train=dataset.loc[dataset['Survived'].notnull()]
test=dataset.loc[dataset['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'


# Select Survived, Pclass, Sex, Age, Fare, Embarked, Title, FamilyLabel, Deck and TicketGroup as features and transfer the features into numerial values.
# Get trainset and testset based on whether the value of Survived is null or not.

# In[ ]:


dataset = pd.concat([train, test])
dataset=dataset[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
dataset=pd.get_dummies(dataset)
trainset=dataset[dataset['Survived'].notnull()]
testset=dataset[dataset['Survived'].isnull()].drop('Survived',axis=1)
X = trainset.as_matrix()[:,1:]
Y = trainset.as_matrix()[:,0]


# # Model training and prediction
# Use grid search to find the best parameter of random forest classifier.

# In[ ]:


pipe=Pipeline([('select',SelectKBest(k=20)), 
               ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'classify__n_estimators':list(range(20,50,2)), 
              'classify__max_depth':list(range(3,60,3))}
gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='accuracy', cv=10)
gsearch.fit(X,Y)
print(gsearch.best_params_, gsearch.best_score_)


# The best parameters I got from grid search is n_estimators = 22, max_depth = 6. But I tried n_estimators = 24, 26, 28, respectively. I got higher cross validation score and kaggle score when n_estimators = 26, max_depth = 6.
# Train the random forest classifier model using the parameters.

# In[ ]:


select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, Y)


# Cross Validation

# In[ ]:


cv_score = model_selection.cross_val_score(pipeline, X, Y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))


# Make prediction and output the result as submission.csv

# In[ ]:


predictions = pipeline.predict(testset)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("submission.csv", index=False)


# In[ ]:




