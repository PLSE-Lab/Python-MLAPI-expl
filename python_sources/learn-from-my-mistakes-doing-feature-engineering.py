#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[ ]:


# lets import data from the files 
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
submission_data = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


# lets view the data 
train_data.head()


# In[ ]:


# Objective of the problem is to classify among the passengers who survived.
# Target variable considered here is Survived
train_target = train_data.Survived
#train_data.drop('Survived',axis=1,inplace=True)
#Since passenger Id doesn't add any significant to Survival lets drop that variable as well
train_data.drop('PassengerId',axis=1,inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


train_target.head()


# ### Lets understand the data

# - Pclass : Ticket Class
# - Name : name of the passenger
# - Sex : gender of the passenger
# - Age : Age of passenger
# - SibSp : Numb of siblings or spouse
# - Parch : parents/children on board
# - Ticket : Ticket number
# - Fare : price of the ticket
# - Cabin : Location of the stay in ship
# - Embarked : point of boarding the ship

# - Now lets understand how this can be related to Survival of passenger

# - Pclass tell us passengers seat class and related to cabin. May be the passenger with higher class cabin
# - could survive as the cabin may be in safer place or safety boats are attached to this cabins. Lets check it pictorially

# - Name may not be significant but we could derive a variable from the name stating the title of the passenger by which 
# - we can infer the status of passenger like mother single such wise
# 
# - SibSp/Parch : This variable states numb of siblings or spouse so we can derive the variable familysize
# - Fare : this effects the cabin and pclass variables and embarked which indirectly relates survival
# 

# In[ ]:


# lets see the basic statistics of data
train_data.describe()


# In[ ]:


train_data.describe(include=['O'])


# In[ ]:


# Lets check the correlation matrix
train_data.corr()


# In[ ]:


# lets Clean the data first
train_data.info() # check the null in data


# In[ ]:


# from the above its clear Age,Cabin and Embarked are having null values
train_data[train_data.Embarked.isnull()]


# In[ ]:


# these are the two instance where Embarked in null.
# lets fill them with most frequent value.
train_data[((train_data.Pclass==1)&(train_data.Fare <81)&(train_data.Fare >79)&(train_data.Sex =='female'))].Embarked.mode()


# In[ ]:


# Lets fill the embarked null values with S
train_data.Embarked.fillna('S',inplace=True)


# In[ ]:


combine = [train_data,test_data]


# In[ ]:


# lets convert Sex to binary categorical
for df in combine:
    df['Sex'] = np.where(df.Sex=='male',1,0)


# In[ ]:


# lets now take care of Age and cabin
for dataset in combine:
    # Filling all Nan with unknown cabin name.
    dataset.Cabin.fillna('Unknown',inplace=True)
    dataset['Cabin_lt'] = dataset.Cabin.apply(lambda x: str(x)[0])


# In[ ]:


sns.countplot(train_data.Cabin_lt)
plt.show()


# In[ ]:


sns.countplot(test_data.Cabin_lt)
plt.show()


# In[ ]:


sns.barplot(x='Cabin_lt',y='Survived',data=train_data)
plt.show()


# - Seems T is a outlier in the Cabins lets replace it with most frequent one.
# - Passengers with cabins survival rate is more than the passengers without cabins ie., U
# - imputation of the cabins should be done more carfully based on the survival rate.

# In[ ]:


# lets check how the class and Cabin_lt are related and replace it accordingly
train_data.groupby(['Pclass','Survived','Sex']).Cabin_lt.value_counts()


# - now lets decide on what u should be.
# - if Pclass 1 Survival 0 Sex 1 then lets replace U and T with C(frequently Occurred) --- it defines people most of them in C cabin didn't survive
# - if Pclass 1 Survival 1 sex 0 then lets replace U with B(frequent and survived most of them in B)
# - if Pclass 1 Survival 1 sex 1 then lets replace U with C
# - if Pclass 2 Survival 0 sex 0 then lets replace U with E
# - if Pclass 2 Survival 0 sex 1 then lets replace U with D
# - if Pclass 2 Survival 1 sex 0 then lets replace U with F
# - if Pclass 2 Survival 1 sex 1 then lets replace U with F
# - if Pclass 3 Survival 0 sex 0 then lets replace U with G
# - if Pclass 3 Survival 0 sex 1 then lets replace U with F
# - if Pclass 3 Survival 1 sex 0 then lets replace U with G
# - if Pclass 3 Survival 1 sex 1 then lets replace U with E

# In[ ]:


# lets implement the above in order impute the cabin
train_data['Cabin']= np.where(((train_data.Pclass == 1)&(train_data.Survived == 0)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'C',
                             np.where(((train_data.Pclass == 1)&(train_data.Survived == 1)&(train_data.Sex == 0)&(train_data.Cabin_lt == 'U')),'B',
                             np.where(((train_data.Pclass == 1)&(train_data.Survived == 1)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'C',
                             np.where(((train_data.Pclass == 2)&(train_data.Survived == 0)&(train_data.Sex == 0)&(train_data.Cabin_lt == 'U')),'E',
                             np.where(((train_data.Pclass == 2)&(train_data.Survived == 0)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'D',
                             np.where(((train_data.Pclass == 2)&(train_data.Survived == 1)&(train_data.Sex == 0)&(train_data.Cabin_lt == 'U')),'F',
                             np.where(((train_data.Pclass == 2)&(train_data.Survived == 1)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'F',
                             np.where(((train_data.Pclass == 3)&(train_data.Survived == 0)&(train_data.Sex == 0)&(train_data.Cabin_lt == 'U')),'G',
                             np.where(((train_data.Pclass == 3)&(train_data.Survived == 0)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'F',
                             np.where(((train_data.Pclass == 3)&(train_data.Survived == 1)&(train_data.Sex == 0)&(train_data.Cabin_lt == 'U')),'G',
                             np.where(((train_data.Pclass == 3)&(train_data.Survived == 1)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'E',
                             np.where(((train_data.Pclass == 1)&(train_data.Survived == 0)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'T')),'C',
                             train_data.Cabin_lt))))))))))))


# In[ ]:


#Cross checking the update
train_data.groupby(['Pclass','Survived','Sex']).Cabin.value_counts()


# In[ ]:


# lets check how the class and Cabin_lt are related and replace it accordingly
test_data.groupby(['Pclass','Sex']).Cabin_lt.value_counts()


# - Since as above we dont have Survived variable here lets impute the cabin using Pcalss and sex in test data
# - Pclass 1 U = C
# - Pclass 2 Sex 0 U = F
# - PClass 2 sex 1 U = D
# - Pclass 3 sex 0 U = G
# - Pcass 3 sex 1 U = F

# In[ ]:


test_data['Cabin'] = np.where(((test_data.Pclass == 1)&(test_data.Cabin_lt == 'U')),'C',
                             np.where(((test_data.Pclass == 2)&(test_data.Sex == 0)&(test_data.Cabin_lt == 'U')),'F',
                                     np.where(((test_data.Pclass == 2 )&(test_data.Sex == 1)&(test_data.Cabin_lt == 'U')),'D',
                                             np.where(((test_data.Pclass == 3)&(test_data.Sex == 0)&(test_data.Cabin_lt =='U')),'G',
                                                     np.where(((test_data.Pclass == 3)&(test_data.Sex == 1)&(test_data.Cabin_lt =='U')),'F',
                                                             test_data.Cabin_lt)))))


# In[ ]:


test_data.groupby(['Pclass','Sex']).Cabin.value_counts()


# In[ ]:


# lets drop the Cabin_lt from train and test
for dataset in combine:
    del dataset['Cabin_lt']


# In[ ]:


for dataset in combine:
    dataset['title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
    


# In[ ]:


train_data.title.value_counts()


# - in the above Mr Miss Mrs Master are more frequent
# - lets consider all other titles in to rare group

# In[ ]:


train_data[((train_data.Age.isnull())&(train_data.title == 'Dr'))] # since there is only 1 Dr with age NaN so lets put Dr into others


# In[ ]:


for dataset in combine:# converting the titles into numericals
    dataset['title'] = np.where((dataset.title == 'Mr'),1,
                               np.where(dataset.title == 'Miss',2,
                                       np.where(dataset.title == 'Mrs',3,
                                               np.where(dataset.title == 'Master',4,5))))


# In[ ]:


train_data.title.value_counts()


# In[ ]:


# there is one more missing value that tobe imputed that is Age. so lets derive as many as varibles and use them to predict Age
# let us get one more variable Family size which can be derived from SibSp and Parch
for dataset in combine:
    dataset['Family'] = dataset.SibSp+dataset.Parch+1
    dataset['Family'] = np.where(dataset.Family == 1,'Single',
                                np.where(dataset.Family == 2,'Couple',
                                        np.where(dataset.Family <=4 ,'Nuclear','LargeF')))


# In[ ]:


sns.barplot(x=train_data.Family,y=train_data.Survived,data=train_data)
plt.show()# it shows Nuclear and Couples have survived in larger number


# In[ ]:


sns.boxplot(y=train_data.Fare)


# In[ ]:


train_data['Fare'] = train_data.Fare.astype('int32')


# In[ ]:


test_data[test_data.Fare.isnull()]


# In[ ]:


test_data.Fare.fillna(test_data[((test_data.Pclass == 3)&(test_data.Sex ==1)&(test_data.Embarked =='S')&(test_data.Family=='Single'))].Fare.mean(),inplace=True)


# In[ ]:


#lets check if the ticket is having any relation with survival
train_data.groupby(['Pclass']).Fare.min()   #[train_data.Fare >= 500].count()#['Ticket','Fare','Pclass','Family','Parch','SibSp']]


# In[ ]:


test_data.groupby(['Pclass']).Fare.min()


# In[ ]:


sns.distplot(train_data.Fare)
plt.show()


# In[ ]:


sns.distplot(test_data.Fare)
plt.show()


# In[ ]:


for dataset in combine:
    dataset['Ticket_str'] = dataset.Ticket.apply(lambda x : str(x)[0])


# In[ ]:


train_data.groupby(['Pclass']).Ticket_str.value_counts()


# In[ ]:


train_data.head()


# - From  the above it is inferred that the starting number represent the pclass and other than 1,2,3, few letters represents
# - lower price and other represents random 

# In[ ]:


for dataset in combine:
    dataset['Ticket_str'] = dataset['Ticket_str'].apply(lambda x: str(x))
    dataset['Ticket_str'] = np.where((dataset['Ticket_str']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), dataset['Ticket_str'],
                                np.where((dataset['Ticket_str']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                        'Low_ticket', 'Other_ticket'))
    dataset['Ticket_len'] = dataset['Ticket'].apply(lambda x: len(x))


# In[ ]:


# lets create dummies for all the variables avaiable
def dummy(data):
    dataset = pd.concat([data,pd.get_dummies(data.Pclass,prefix='Pclass'),
                     pd.get_dummies(data.Sex,prefix='Sex'),
                     pd.get_dummies(data.Cabin,prefix='Cabin'),
                    pd.get_dummies(data.Embarked,prefix='Embark'),
                    pd.get_dummies(data.title,prefix='title'),
                    pd.get_dummies(data.Family,prefix='Family'),
                    pd.get_dummies(data.Ticket_str,prefix='ticket_str'),
                    pd.get_dummies(data.Ticket_len,prefix='ticket_len')],axis=1)
    dataset.drop(['Pclass','Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked','title','Family','Ticket_str',
             'Ticket_len'],axis=1,inplace=True)
    return dataset


# In[ ]:


train = dummy(train_data)
test = dummy(test_data)


# In[ ]:


train.drop(['Survived'],axis=1,inplace=True)
train.columns


# In[ ]:


test.drop(['PassengerId'],axis=1,inplace=True)
test.columns


# In[ ]:


for dataset in [train_data,test_data]:
    for i in list(dataset["Age"][dataset["Age"].isnull()].index):
        age_mean = dataset["Age"][dataset['Family'] == dataset.iloc[i]['Family']].mean()
        Age_mean = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"])&(dataset['title'] == dataset.iloc[i]["title"]))].mean()
        if not np.isnan(Age_mean) :
            dataset['Age'].iloc[i] = Age_mean
        else :
            dataset['Age'].iloc[i] = age_mean


# In[ ]:


train['Age'] = train_data['Age']
test['Age'] = test_data['Age']


# In[ ]:


#trainx,testx,trainy,testy = train_test_split(train,train_target,test_size=0.2)


# In[ ]:


#param = { "criterion" : ["gini", "entropy"],
#         "max_features": [1,3,8,10,12],
#         "min_samples_leaf" : [1, 5, 10,15],
#         "min_samples_split" : [2, 4, 10, 12, 16,18,20],
#         "n_estimators": [50, 100, 400, 700,800,900, 1000]
#        }
#rf = GridSearchCV(RandomForestClassifier(oob_score=True),param_grid=param,scoring='accuracy',cv=5,n_jobs=-1,verbose=True)


# In[ ]:


#rf.fit(trainx,trainy)


# In[ ]:


#rf.best_estimator_


# In[ ]:


rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features=8, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=4,
            min_weight_fraction_leaf=0.0, n_estimators=900, n_jobs=None,
            oob_score=True, random_state=None, verbose=0, warm_start=False)

rf.fit(train, train_target)
#print("%.4f" % rf.oob_score_)


# In[ ]:


rf_train_pred = rf.predict(train)


# In[ ]:


print(metrics.accuracy_score(train_target,rf_train_pred))


# In[ ]:


test_rf = rf.predict(test)


# In[ ]:


rf_submission_testdata = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':test_rf})


# In[ ]:


#submission_testdata.to_csv('submit_learner.csv',index=False)


# ### Running with Xgboost Classifier 

# In[ ]:


from xgboost.sklearn import XGBClassifier


# In[ ]:


#param_xg = {'max_depth':[2,3,4,5,6,8,10],
#        'learning_rate':[0.1,0.01,0.001,0.02],
#        'n_estimators':[100,200,300,400,500,600,800,900,1000],
#        'subsample':[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
#        'reg_alpha':[0.0001,0.0002,0.00003,0.00004,0.00005,0.00006,0.00007]}


# In[ ]:


#xggs =  GridSearchCV(XGBClassifier(),param_grid=param_xg,cv=5,scoring='accuracy',n_jobs=-1,verbose=True)


# In[ ]:


#xggs.best_estimator_tor_


# In[ ]:


#xggs.best_score_


# In[ ]:


xggc = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=2, min_child_weight=1, missing=None, n_estimators=800,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0.0001, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=0.9)


# In[ ]:


xggc.fit(train,train_target)


# In[ ]:


xg_train_pred = xggc.predict(train)


# In[ ]:


metrics.accuracy_score(train_target,xg_train_pred)


# In[ ]:


xg_test_pred = xggc.predict(test)


# In[ ]:


print(metrics.classification_report(train_target,xg_train_pred))


# In[ ]:


print(metrics.confusion_matrix(train_target,xg_train_pred))


# In[ ]:


sns.heatmap(metrics.confusion_matrix(train_target,xg_train_pred))


# In[ ]:


xg_submission_testdata = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':xg_test_pred})


# In[ ]:


xg_submission_testdata.to_csv('submit.csv')


# In[ ]:


xg_submission_testdata


# Upvotes are most welcome

# 
