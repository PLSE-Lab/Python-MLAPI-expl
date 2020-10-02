#!/usr/bin/env python
# coding: utf-8

# # Python Titanic
# ## I Introduction
# 
# First, I'm going to make one possible explanation about the ones who get 100% accuracy in the Leaderboard. They have obtained the whole dataset with train and test labels. Then they use the prediction model to fit and make an prediction. Since the rest of us only get the part of the information, there is no possibility to depict the picture in whole. 
# 
# Now let's go back my notebook. 
# 
# ----------
# ### 1.Features Analysis
# Here is the reference from Chris Deotte.
# 
# https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818 
# 
# In his simple model using R language, he scores 82% only using Names. He found that the male except boy is dead and the female except surname family dead are survived. This is a decision tree model which is simple and effective. So according to this logical method, other methods based on the trees should be tried. But before that, we will try to analyse the features and data cleaning. Then we create the feature mentioned by Chris Deotte and modify the test features accordingly.
# 
# ### 2.Supervised methods
# Here is the reference from Chris Deotte.
# 
# https://www.kaggle.com/cdeotte/titanic-wcg-xgboost-0-84688
# 
# In his notebook, he scored 0.85167. The notebook has tried XGBoost with the previous features and other new added features. It seems that some features should be dropped since it is a kind of noise. So in my notebook, the random forest and XGBoost will be chosen. These two methods are better than other methods because they are based on the Cart tree model, which is a good chioce for this dataset.
# 

# ## II Feature Analysis and Process
# 
# ### 1.Load data

# In[ ]:


import numpy as np
import pandas as pd

# Load train data as train_data
in_file = '/kaggle/input/titanic/train.csv'
train_data = pd.read_csv(in_file)
print("There are {} samples in train_data".format(len(train_data)))

# Load test data as test_data
in_file = '/kaggle/input/titanic/test.csv'
test_data = pd.read_csv(in_file)
print('There are {} samples in test_data'.format(len(test_data)))
PassengerId=test_data['PassengerId']

# Set the labels of test_data as na
test_data['Survived'] = np.nan

# Concat the features of train dataset and test dataset
all_data = pd.concat([train_data, test_data], ignore_index=True)

# Show the whole dataset informations
all_data.info()
display(all_data.head())


# In[ ]:


# Check how many null values in each feature.
all_data.isnull().sum()


# **For the null values above, we dicide to deal as following.**
# - The age null values can be filled by the regression method.
# - The 'Embarked' features can be filled according to the fare and pclass.
# - There are many null values in 'Cabin' features. For the better predictions, it is believed to delete the feature of 'Cabin'. But here we need this feature because when the ship sinks, certain parts of the ship have different probability drown in water. So we deal to make a new feature to substitute the feature.
# - The fare can be filled up with median value or mean value. 
# - The survived is the target to predict.

# ### 2.Missing values process

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# #### 2.1 Embarked
# For the correctness, do not fill the null values only by the original feature values. We need to see other features too. So we can see about the passangers relatives number, the pclass, the fare, and the ticket.

# In[ ]:


# Show the samples with embarked null
display(all_data[all_data['Embarked'].isnull()])


# In[ ]:


#  Only reserve the numbers of the tickets.  &(all_data['Pclass']==1)
#  We assume that the ticket number is probably same if it is bought from the same place
all_data['Ticket_num'] = all_data['Ticket'].map(lambda x: re.sub("\D", "", x))
all_data['Ticket_num'] = pd.to_numeric(all_data['Ticket_num'])
all_data.head()


# In[ ]:


all_data.isnull().sum()


# In[ ]:


#  Select the passangers relatives number equal to 0, and the pclass equal to 1, the ticket number between 113000 and 114000
select_data = all_data[(all_data['Parch']==0)&(all_data['SibSp']==0)                       &(all_data['Ticket_num']<114000)&(all_data['Ticket_num']>110000)]
select_data.sort_values('Ticket_num').head()


# In[ ]:


select_data.Embarked.value_counts()


# **The embarked place is obviously connected with the fare and the tickets. From the value counts, the probability of S is more possible than other two options.**

# In[ ]:


all_data['Embarked'] = all_data['Embarked'].fillna('S') 


# #### 2.2 Age
# 
# The age feature has 263 null values. So we use random forest regression model to simulate the value. The features we use here are sex, title, pclass.
# 

# In[ ]:


facet = sns.FacetGrid(all_data[0:890], hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, all_data.loc[0:890,'Age'].max()))
facet.add_legend()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

age_df = all_data[['Age', 'Pclass','Sex','Parch','SibSp']]
age_df=pd.get_dummies(age_df)
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y)
predictedAges = rfr.predict(unknown_age[:, 1::])
all_data.loc[(all_data.Age.isnull()), 'Age'] = predictedAges 


# In[ ]:


facet = sns.FacetGrid(all_data[0:890], hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, all_data.loc[0:890,'Age'].max()))
facet.add_legend()


# #### 2.3 Fare
# 
# The fare feature has 1 null values. So we use median value to fill up.

# In[ ]:


fare=all_data.loc[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3), 'Fare'].median()
all_data['Fare']=all_data['Fare'].fillna(fare)


# ### 3. Create new features
# 
# We all know the movie story of tatanic that women and children escapes first and their survival rate shall be higher than other category. It is a easy way to believe all women and children survived and all man died. But it is contracted with boys. So we gona to extract the titles as man, woman and boy.
# 

# #### 3.1 Title

# In[ ]:


# Extract the titles from name feature

# Create a set for all unique titles
titles = set()

# Extract from the name feature
for name in all_data['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
print(titles)


# In[ ]:


# Make a dictionary for mapping the titles into three kinds
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Master",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty",
    "Dona" : "Royalty"    
}

def get_titles(data):
    data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    data['Title'] = data['Title'].map(Title_Dictionary)
    return data

get_titles(all_data)
all_data['Title'].value_counts()  


# #### 3.2 Surname
# 
# We could make the exception list of male and female. Then change the features of test surname in the list into the dead features or survived features according to the list.

# In[ ]:


# Extract from name feature
all_data['Surname'] = all_data['Name'].map(lambda name:name.split(',')[0].strip())

# Create a new feature as surname frequency
all_data['FamilyGroup'] = all_data['Surname'].map(all_data['Surname'].value_counts()) 

sns.barplot(x='FamilyGroup', y='Survived', data=all_data, palette='Set3')

Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=16) | (all_data['Sex']=='female'))]
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)

Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>16) & (all_data['Sex']=='male')]
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)


# In[ ]:


all_data.loc[(all_data['Survived'].isnull()) & (all_data['Surname'].apply(lambda x:x in Dead_List)),             ['Sex','Age','Title']] = ['male',28.0,'Mr']
all_data.loc[(all_data['Survived'].isnull()) & (all_data['Surname'].apply(lambda x:x in Survived_List)),             ['Sex','Age','Title']] = ['female',5.0,'Miss']


# #### 3.3 FamilySize

# In[ ]:


all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1
sns.barplot(x="FamilySize", y="Survived", data=all_data, palette='Set3')


# In[ ]:


def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)
sns.barplot(x="FamilyLabel", y="Survived", data=all_data, palette='Set3')


# #### 3.4 Deck

# In[ ]:


all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck']=all_data['Cabin'].str.get(0)
sns.barplot(x="Deck", y="Survived", data=all_data, palette='Set3')


# #### 3.5 TicketGroup

# In[ ]:


Ticket_Count = dict(all_data['Ticket'].value_counts())

all_data['TicketGroup'] = all_data['Ticket'].map(Ticket_Count)
sns.barplot(x='TicketGroup', y='Survived', data=all_data, palette='Set3')


# In[ ]:


def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=all_data, palette='Set3')


# In[ ]:


all_data = all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data.isnull().sum()


# ### 4.Model Prediction

# In[ ]:


all_data = all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data = pd.get_dummies(all_data)
train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)
X = train.as_matrix()[:,1:]
y = train.as_matrix()[:,0]


# #### 4.1 Random Forest

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')

pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

cv_result = cross_validate(pipeline, X, y, cv= 10)
print("CV Test Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_result['test_score']),                                                      np.std(cv_result['test_score'])))


# In[ ]:


test_x = test.as_matrix()


# In[ ]:


predictions = pipeline.predict(test_x)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("submission_test.csv", index=False)


# #### 4.2 XGBoost

# In[ ]:


from sklearn.feature_selection import SelectKBest
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')

select = SelectKBest(k = 20)
clf = XGBClassifier(
                     max_depth=3, 
                     learning_rate=0.1, 
                     n_estimators=100, 
                     silent=True, 
                     objective='binary:logistic', 
                     booster='gbtree', 
                     nthread=None, 
                     gamma=0, 
                     subsample=0.8, 
                     colsample_bytree=1, 
                     colsample_bylevel=1, 
                     reg_alpha=0, 
                     reg_lambda=1, 
                     scale_pos_weight=1.2, 
                     base_score=0.5)

pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

cv_result = cross_validate(pipeline, X, y, cv= 10)

print(cv_result['test_score'])
print("CV Test Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_result['test_score']),                                                      np.std(cv_result['test_score'])))


# > ### 5 Conclusion
# 
# The random forest is better than XGBoost. So we submit the random forest result.

# 
