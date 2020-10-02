#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


titanic_data=pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_data.head(50)


# In[ ]:


test_data=pd.read_csv("/kaggle/input/titanic/test.csv")
test_data


# In[ ]:


test_data.Cabin.isnull().sum()


# In[ ]:


example_data=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
example_data.head(50)


# In[ ]:


training_data=titanic_data.copy()


# FALSE NEGATIVE(WHEN ACTUAL RESULT IS 0,BUT MODEL SHOWS 1)IS MORE OF A CONCERN,THAN FALSE POSITIVE(ACTUAL-1,MODEL-0).SO RECALL WILL BE MORE PRIORITIZED THAN PRECISION.

# In[ ]:


training_data.dtypes


# In[ ]:


training_data.shape


# In[ ]:


len(training_data['PassengerId'].unique())==len(training_data['PassengerId'].values)
    


# THERE ARE NO DUPLICATES,WHICH MEANS THAT EVERY CASE IS UNIQUE

# In[ ]:


training_data.isnull().sum()


# In[ ]:


missing_fraction=(training_data['Age'].isnull().sum()/len(training_data['Age']))*100
missing_fraction


# LOTS OF CASES DO NOT CONTAIN INFO ABOUT CABIN,AND ALMOST 20% OF AGE INFO IS MISSED

# In[ ]:


import seaborn as sns
sns.boxplot('Survived', 'Age', data=training_data)


# In[ ]:


sns.pairplot(training_data)


# In[ ]:


matrix=training_data.corr()
print(matrix)


# 

# In[ ]:


training_data['Cabin'].unique()


# In[ ]:


training_data.drop('PassengerId',inplace=True,axis=1)


# In[ ]:


training_data


# In[ ]:


import string
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find( substring) != -1:
            return substring
    print (big_string)
    return np.nan


# In[ ]:


title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']


# In[ ]:


training_data['Title']=training_data['Name'].map(lambda x: substrings_in_string(x, title_list))


# In[ ]:


def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
training_data['Title']=training_data.apply(replace_titles, axis=1)


# In[ ]:





# In[ ]:


training_data['Family_Size']=training_data['SibSp']+training_data['Parch']


# In[ ]:


training_data['Age*Class']=training_data['Age']*training_data['Pclass']


# In[ ]:


training_data['Fare_Per_Person']=training_data['Fare']/(training_data['Family_Size']+1)


# In[ ]:





# In[ ]:


cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
training_data['Deck']=training_data['Cabin'].map(lambda x: substrings_in_string(str(x), cabin_list))


# In[ ]:


def feature_engineering(data):
    data['Title']=data['Name'].map(lambda x: substrings_in_string(x, title_list))
    data['Title']=data.apply(replace_titles, axis=1)
    data['Family_Size']=data['SibSp']+data['Parch']
    data['Age*Class']=data['Age']*data['Pclass']
    data['Fare_Per_Person']=data['Fare']/(data['Family_Size']+1)
    data['Deck']=data['Cabin'].map(lambda x: substrings_in_string(str(x), cabin_list))
    return data


# In[ ]:


training_data


# In[ ]:


print(training_data.corr())


# In[ ]:


training_data.drop(['Name','Cabin'],inplace=True,axis=1)
training_data


# In[ ]:


error_fares=training_data['Fare'][training_data['Fare']==0.0000]
indexes=error_fares.index
indexes


# In[ ]:


for index in indexes:
    training_data.drop([index],axis=0,inplace=True)
    


# In[ ]:


checking_errors=training_data['Fare'][training_data['Fare']==0.0000]
checking_errors


# In[ ]:


identifying_missing_decks=pd.concat([training_data['Fare_Per_Person'],training_data['Deck']],axis=1)


# In[ ]:


identifying_missing_decks


# In[ ]:


for_a=identifying_missing_decks[identifying_missing_decks['Deck']=='A']
for_a


# 

# In[ ]:


average_fare_for_a=for_a['Fare_Per_Person'].mean()


# In[ ]:


for_b=identifying_missing_decks[identifying_missing_decks['Deck']=='B']
for_b


# In[ ]:


average_fare_for_b=for_b['Fare_Per_Person'].mean()


# In[ ]:


for_c=identifying_missing_decks[identifying_missing_decks['Deck']=='C']
for_c


# In[ ]:


average_fare_for_c=for_c['Fare_Per_Person'].mean()
average_fare_for_c


# In[ ]:


'D', 'E', 'F', 'T', 'G'


# In[ ]:


for_d=identifying_missing_decks[identifying_missing_decks['Deck']=='D']
average_fare_for_d=for_d['Fare_Per_Person'].mean()


# In[ ]:


average_fares=[]
for string in cabin_list:
    for_deck=identifying_missing_decks[identifying_missing_decks['Deck']==string]
    average_fare_for_deck=for_deck['Fare_Per_Person'].mean()
    average_fares.append(average_fare_for_deck)


# In[ ]:


average_fares


# A,D,E,T are almost the same.Lets check the difference in fares themselves
# 

# In[ ]:


identify_missing_decks1=pd.concat([training_data['Fare'],training_data['Deck']],axis=1)


# In[ ]:


average_fares1=[]
for string in cabin_list:
    for_deck=identify_missing_decks1[identify_missing_decks1['Deck']==string]
    average_fare_for_deck=for_deck['Fare'].mean()
    average_fares1.append(average_fare_for_deck)


# In[ ]:


average_fares1


# This approach more spreader in terms of results,than previous one.
# But the mean itself will not be enough so let me compare standard deviations as well

# In[ ]:



for string in cabin_list:
    for_deck=identify_missing_decks1[identify_missing_decks1['Deck']==string]
    print(for_deck['Fare'].describe())


# it will be good if we remove some types of decks due to the lack of data,and combine known decks as:low,medium,and high quality decks.B and C seem to us as high(combine them),A and E medium,F and G low,and D between medium and high.T should be deleted because there is only one who purchased it.

# In[ ]:


identifying_missing_decks_by_fare=training_data[['Fare','Deck']]
identifying_missing_decks_by_fare


# In[ ]:


identifying_missing_decks_by_fare['Deck'].replace(to_replace=['F','G'],value='low',inplace=True)
identifying_missing_decks_by_fare['Deck'].replace(to_replace=['A','E'],value='medium',inplace=True)
identifying_missing_decks_by_fare['Deck'].replace(to_replace='D',value='medium_high',inplace=True)
identifying_missing_decks_by_fare['Deck'].replace(to_replace=['B','C'],value='high',inplace=True)        


# In[ ]:



identifying_missing_decks_by_fare


# 

# In[ ]:


quality_list=['low','medium','medium_high','high']
for string in quality_list:
    for_deck=identifying_missing_decks_by_fare[identifying_missing_decks_by_fare['Deck']==string]
    print(for_deck['Fare'].describe())


# price <26-low.price>26 but <55-medium.>55 but <75 medium-high.>75-high.

# In[ ]:


t=identifying_missing_decks_by_fare[identifying_missing_decks_by_fare['Deck']=='T']
t.index


# In[ ]:


identifying_missing_decks_by_fare.drop(t.index,axis=0,inplace=True)


# In[ ]:


training_data.drop(t.index,axis=0,inplace=True)


# In[ ]:


identifying_missing_decks_by_fare['Deck'].unique()


# In[ ]:


identifying_missing_decks_by_fare.reset_index(inplace=True)


# In[ ]:


identifying_missing_decks_by_fare.drop(['index'],axis=1,inplace=True)


# In[ ]:





# In[ ]:


for i in range(len(identifying_missing_decks_by_fare)):
    fare=identifying_missing_decks_by_fare['Fare'][i]
    if identifying_missing_decks_by_fare['Deck'][i] is np.nan:
        if fare<=26:
            identifying_missing_decks_by_fare['Deck'][i]='low'
        elif fare>26 and fare<=55:
            identifying_missing_decks_by_fare['Deck'][i]='medium'
        elif fare>55 and fare<=75:
            identifying_missing_decks_by_fare['Deck'][i]='medium_high'
        if fare>75:
            identifying_missing_decks_by_fare['Deck'][i]='high'
    else:
        identifying_missing_decks_by_fare['Deck'][i]=identifying_missing_decks_by_fare['Deck'][i]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


identifying_missing_decks_by_fare


# In[ ]:





# In[ ]:





# **THERE IS SOMETHING UNFITTED.I FORGET SOME THINGS: FIRST THING IS MULTICORRINEARITY(IF I WANT TO ADD DECKS,I SHOULD DROP FARE FEATURE,WHICH IS NOT A GOOD OPTION).SECOND ONE IS DEPENDENCE(CHOICE OF DECKS CAN BE DEPENDENT ALSO ON THE NUMBER OF RELATIVES AND SIZE OF FAMILY,SO DOING REGRESSION,WE CAN DETERMINE DECKS,BUT WE SHOULD DROP NOT ONLY FARE,BUT ALSO FEATURES,RELATED TO PERSONAL INFORMATION).AND THE THIRD ONE IS UNCLEARITY OF INFORMATION ABOUT DECKS.WHAT I MEAN BY UNCLEARITY?SOME DECKS WERE NOT DEPENDENT ON THE SIZE OF FARE,AND DECKS CAN ALSO BE DEPENDENT ON TICKET NUMBER,SO I THINK THAT I MUST START AGAIN THIS JOURNEY OF REPLACING NAN'S,BECAUSE IT IS NOTICEABLE THAT MAJORITY OF PEOPLE IN THIS COMPETITION DID NOT USE DECK COLUMN AS A PREDICTOR VARIABLE,AND ALSO THAT DECK CAN CONTAIN SOME VALUABLE PATTERNS IN DATA. **

# In[ ]:


training_data


# In[ ]:


first_class=training_data[training_data['Pclass']==1]
first_class


# THERE IS NO CONNECTION BETWEEN TICKET AND DECKS(LOOKING THROUGH TICKET WE CAN NOT PREDICT THE CATEGORY OF DECK)

# In[ ]:


unknown_values=first_class[first_class['Deck'].isna()]
unknown_values


# In[ ]:


unknown_values['Deck'].value_counts()


# In[ ]:


unknown_values['Deck'].isnull().sum()


# In[ ]:


training_data[training_data['Deck'].isna()]


# In[ ]:


test_data


# In[ ]:


test_for_decks=test_data.copy()


# In[ ]:


test_for_decks['Deck']=test_for_decks['Cabin'].map(lambda x: substrings_in_string(str(x), cabin_list))


# In[ ]:


class_1_test=test_for_decks[test_for_decks['Pclass']==1]


# In[ ]:


class_1_test


# In[ ]:


class_1_test['Deck'].value_counts()


# In[ ]:


class_1_test['Deck'].isnull().sum()


# WHAT CAN WE DO HERE IS JUST SPLITTING DATASET INTO TWO MINI-DATASETS:NAN VALUES IN FIRST CLASS WILL BE DROPPED,AND DECK WILL BE USED AS A FEATURE,AND OTHER MINI-DATASET WILL INCLUDE 2 AND 3 CLASS PASSENGERS,BUT WHOLE DECK FEATURE WILL BE DROPPED.

# In[ ]:


first_class


# In[ ]:


first_class.isnull().sum()


# In[ ]:


first_class.dropna(subset=['Deck'],axis=0,inplace=True)


# In[ ]:


first_class['Deck'].isnull().sum()


# In[ ]:


first_class.corr()


# In[ ]:


first_class.drop(['Pclass','Age','SibSp','Parch','Ticket'],axis=1,inplace=True)


# In[ ]:


first_class


# In[ ]:





# In[ ]:


first_class.drop(first_class[first_class['Embarked'].isna()].index,axis=0,inplace=True)


# In[ ]:


first_class.isnull().sum()


# In[ ]:


train_1_class=first_class.iloc[:,1:]
test_1_class=first_class['Survived']


# In[ ]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

categorical=['Sex','Title','Deck','Embarked']

numerical=['Fare','Family_Size','Age*Class','Fare_Per_Person']
pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler()),
])

from sklearn.compose import ColumnTransformer

full_pipeline=ColumnTransformer([
    ('num',pipeline,numerical),
    ('cat',OneHotEncoder(),categorical)
])

prepared_data_for_first_class=full_pipeline.fit_transform(train_1_class)


# In[ ]:


prepared_data_for_first_class


# In[ ]:





# In[ ]:


from sklearn import svm
parameters={'gamma' : ['auto','scale'],'C' : [0.1, 1, 10],'degree' : [0, 1, 2, 3, 4, 5, 6]}
svc=svm.SVC()
from sklearn.model_selection import GridSearchCV
clf=GridSearchCV(svc,parameters,cv=2)
clf.fit(prepared_data_for_first_class,test_1_class)            


# In[ ]:


clf.best_estimator_


# In[ ]:


svc_for_1_class=svm.SVC(C=1,degree=0,gamma='auto')
svc_for_1_class.fit(prepared_data_for_first_class,test_1_class)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(test_1_class, svc_for_1_class.predict(prepared_data_for_first_class))


# In[ ]:





# In[ ]:


confusion_matrix(test_1_class, svc_for_1_class.predict(prepared_data_for_first_class))


# In[ ]:


from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(test_1_class, svc_for_1_class.predict(prepared_data_for_first_class)))
print(recall_score(test_1_class, svc_for_1_class.predict(prepared_data_for_first_class)))
print(f1_score(test_1_class, svc_for_1_class.predict(prepared_data_for_first_class)))


# precision score is great,and this is good,because it is more important to care about false positives,rather than false negatives 

# ***SO LETS TAKE A LOOK TO OTHER CLASSES***

# In[ ]:


other_classes=training_data[training_data['Pclass']!=1]
other_classes.isnull().sum()


# In[ ]:


other_classes['Deck'].value_counts()


# In[ ]:


other_classes=other_classes[other_classes['Deck'].isna()]
other_classes


# In[ ]:


other_classes.drop(['Deck'],axis=1,inplace=True)


# In[ ]:


test=other_classes['Survived']
other_classes.drop(['Survived'],axis=1,inplace=True)


# In[ ]:


other_classes.drop(['Age*Class'],axis=1,inplace=True)
imputer=SimpleImputer(strategy='median')
other_classes['Age']=imputer.fit_transform(other_classes['Age'].values.reshape(-1,1))
other_classes['Age*Class']=other_classes['Age']*other_classes['Pclass']


# In[ ]:


other_classes.drop(['Pclass','Age','Ticket','SibSp','Parch'],axis=1,inplace=True)


# In[ ]:


other_classes


# In[ ]:



numerical=['Fare','Family_Size','Age*Class','Fare_Per_Person']
categorical=['Sex','Title','Embarked']
full_pipeline=ColumnTransformer([
    ('num',pipeline,numerical),
    ('cat',OneHotEncoder(),categorical)
])
other_classes_prepared=full_pipeline.fit_transform(other_classes)


# In[ ]:





# In[ ]:





# In[ ]:


parameters={'gamma' : ['auto','scale'],'C' : [0.1, 1, 10,15,20],'degree' : [0, 1, 2, 3, 4, 5, 6]}


# In[ ]:


svc5=svm.SVC()
clf5=GridSearchCV(svc,parameters,cv=3)
clf5.fit(other_classes_prepared,test)   


# In[ ]:


clf5.best_estimator_


# In[ ]:


svc_for_other_class=svm.SVC(C=10,degree=0,gamma='auto')
svc_for_other_class.fit(other_classes_prepared,test)


# In[ ]:


accuracy_score(test, svc_for_other_class.predict(other_classes_prepared))


# In[ ]:


test_data


# In[ ]:


passengers=test_data['PassengerId']


# In[ ]:


test_data=feature_engineering(test_data)


# In[ ]:





# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_data_for_1_class=test_data[test_data['Pclass']==1]


# In[ ]:


test_data_for_1_class


# In[ ]:


test_data_for_1_class.isnull().sum()


# In[ ]:


test_data_for_1_class.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)


# In[ ]:


test_data_for_1_class['Deck'].value_counts()


# In[ ]:


print(35/sum(test_data_for_1_class['Deck'].value_counts()))
print(18/sum(test_data_for_1_class['Deck'].value_counts()))
print(11/sum(test_data_for_1_class['Deck'].value_counts()))
print(9/sum(test_data_for_1_class['Deck'].value_counts()))
print(7/sum(test_data_for_1_class['Deck'].value_counts()))


# In[ ]:


test_data_for_1_class['Deck'].replace(to_replace=[np.nan],value=np.random.choice(a=['C','B','D','E','A'],p=[0.4375,0.225,0.1375,0.1125,0.0875]),inplace=True)


# In[ ]:


test_data_for_1_class['Deck'].isnull().sum()


# In[ ]:





# In[ ]:


test_data_for_1_class.drop(['Pclass','SibSp','Parch','Age'],inplace=True,axis=1)
test_data_for_1_class


# In[ ]:


test_data_for_1_class.index


# In[ ]:


categorical=['Sex','Title','Deck','Embarked']

numerical=['Fare','Family_Size','Age*Class','Fare_Per_Person']
pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler()),
])



full_pipeline=ColumnTransformer([
    ('num',pipeline,numerical),
    ('cat',OneHotEncoder(),categorical)
])

test_1_class=full_pipeline.fit_transform(test_data_for_1_class)


# In[ ]:


predicted_for_1_class=svc_for_1_class.predict(test_1_class)


# In[ ]:


type(predicted_for_1_class)


# In[ ]:


other_classes_test=test_data[test_data['Pclass']!=1]


# In[ ]:


other_classes_test.isnull().sum()


# In[ ]:


other_classes_test.drop(['Deck'],axis=1,inplace=True)


# In[ ]:


other_classes_test.drop(['Age*Class'],axis=1,inplace=True)
imputer=SimpleImputer(strategy='median')
other_classes_test['Age']=imputer.fit_transform(other_classes_test['Age'].values.reshape(-1,1))
other_classes_test['Age*Class']=other_classes_test['Age']*other_classes_test['Pclass']


# In[ ]:


other_classes_test


# In[ ]:


other_classes_test.drop(['PassengerId','Pclass','Age','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
other_classes_test


# In[ ]:


numerical=['Fare','Family_Size','Age*Class','Fare_Per_Person']
categorical=['Sex','Title','Embarked']
full_pipeline=ColumnTransformer([
    ('num',pipeline,numerical),
    ('cat',OneHotEncoder(),categorical)
])
other_classes_prepared_test=full_pipeline.fit_transform(other_classes_test)


# In[ ]:


predicted_test_other_class=svc_for_other_class.predict(other_classes_prepared_test)


# In[ ]:


test_other=pd.Series(predicted_test_other_class)
test_1=pd.Series(predicted_for_1_class)


# In[ ]:


test_1.index=test_data_for_1_class.index


# In[ ]:


test_1


# In[ ]:


test_other.index=other_classes_test.index


# In[ ]:


test_other


# In[ ]:


predicted=pd.concat([test_other,test_1],axis=0)


predicted


# In[ ]:


predicted.sort_index(inplace=True)
predicted


# In[ ]:


predicted=predicted.to_frame()
dataframe=pd.concat([passengers,predicted],axis=1)
dataframe


# THATS ALL.DATAFRAME IS CREATED.NOW IT IS TIME FOR AS MUCH AUTOMATIZATION AS POSSIBLE

# LETS TAKE A LOOK TO DECISION TREE AND RANDOM FOREST AND HOW IT WILL PERFORM

# In[ ]:


def preprocess_the_data(data):
    


# In[ ]:


def check_accuracy_of_model(model,number_of_cv):
    


# In[ ]:


def apply_model_to_test_data(model,test_data)
    


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




