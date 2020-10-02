#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import mode
import matplotlib.pyplot as plt 

Load the test and Train datasets into Pandas Dataframe. There are missing values in both Test and Train sets. Once we train the model using the train set, we should be able to use the model to predict the test set. For that we need to make sure the structure of the train and test datasets are the same and all the features that we are creating is available in both the datasets. It's better to combine the test and train datasets and then split it before training the model.
# In[ ]:


train = pd.read_csv(r'...\train.csv')
test = pd.read_csv(r'....\test.csv')
full = pd.concat([train, test], keys=['train','test'])
#full = pd.concat([train, test])
full.head()

Split the Name column and create 2 new columns - LastName and Title. The Title column will come in handy later to guess the missing age values of the passengers
# In[ ]:


full['LastName'] = full.Name.str.split(',').apply(lambda x: x[0]).str.strip()
full['Title'] = full.Name.str.split("[\,\.]").apply(lambda x: x[1]).str.strip()


# In[ ]:


print(full.Title.value_counts())

Mr 757; Miss 260; Mrs 197; Master 61; Dr 8; Rev 8; Col 4; Ms 2; Mlle 2; Major 2; Dona 1; Lady 1; Sir 1; Mme 1; the Countess 1; Don 1; Jonkheer 1; Capt 1; the Countess 1; Jonkheer 1

As you can see, there are too many titles. We'll try and consolidate the titles to the commonly used ones. We'll consolidate the titles into the main 4 categories - Mr, Mrs, Miss and Master
# In[ ]:


##if the title is Dr and the sex is female, we'll update the Title as Miss
full.loc[(full.Title == 'Dr') & (full.Sex == 'female'), 'Title'] = 'Mrs'

##if the title is in any of the following, we'll update the Title as Miss
full.loc[full.Title.isin(['Lady','Mme','the Countess','Dona']), 'Title'] = 'Mrs'

##if the title is in any of the following, we'll update the Title as Miss
full.loc[full.Title.isin(['Ms','Mlle']), 'Title'] = 'Miss'

##if the title is Dr and the sex is female, we'll update the Title as Mr
full.loc[(full.Title == 'Dr') & (full.Sex == 'male'), 'Title'] = 'Mr'

##if the title is Rev and the sex is male, we'll update the Title as Mr
full.loc[(full.Title == 'Rev') & (full.Sex == 'male'), 'Title'] = 'Mr'

## Setting all the Rev, Col, Major, Capt, Sir --> Mr
full.loc[full.Title.isin(['Rev','Col','Major','Capt','Sir','Don','Jonkheer']) & (full.Sex == 'male'), 'Title'] = 'Mr'

Now we will define a new column - PassengerType to categorize the passengers into Adults and Children
# In[ ]:


def passenger_type (row):
   if row['Age'] < 2 :
      return 'Infant'
   elif (row['Age'] >= 2 and row['Age'] < 12):
      return 'Child'
   elif (row['Age'] >= 12 and row['Age'] < 18):
      return 'Youth'
   elif (row['Age'] >= 18 and row['Age'] < 65):
      return 'Adult'
   elif row['Age'] >= 65:
      return 'Senior'
   elif row['Title'] == 'Master':
      return 'Child'
   elif row['Title'] == 'Miss':
      return 'Child'
   elif row['Title'] == 'Mr' or row['Title'] == 'Mrs':
      return 'Adult'
   else:
      return 'Unknown'
full['PassengerType'] = full.apply(lambda row: passenger_type(row),axis=1)


# In[ ]:


full


# In[ ]:


#Now to see the distribution
full['PassengerType'].value_counts()


# In[ ]:


#factorize the PassengerType to make it numeric values
full['PassengerType'] = pd.factorize(full['PassengerType'])[0]
full['PassengerType'].value_counts()
#full = pd.get_dummies(full, columns=['PassengerType'])


# In[ ]:


#factorize the PassengerType to make it numeric values
full['Title'] = pd.factorize(full['Title'])[0]
full['Title'].value_counts()

There is one Fare that is null. We'll update that to the median fare for the Class and Embarked Combination
# In[ ]:


full['Fare'].isnull().sum()


# In[ ]:


full.loc[full.Fare.isnull()]
full.loc[full.Fare.isnull(), 'Fare'] = full.loc[(full.Embarked == 'S') & (full.Pclass == 3),'Fare'].median()


# In[ ]:


full.head()


# In[ ]:


# Now let's check for nulls in the Embarked column.
full.Embarked.isnull().sum()


# In[ ]:


print(full.groupby(['Pclass', 'Embarked'])['Fare'].median())

The median fare for passengers embarked from "C" and have a First class ticket is $77. Close to the $80 that the two paasengers paid. Based on the median fare, we'll assume that they both Embarked from Port C
# In[ ]:


full.loc[full.Embarked.isnull(), 'Embarked'] = 'C'


# In[ ]:


# We'll now create a bin for the Fare ranges. splitting into 6 groups seems to be a reasonable split.


# In[ ]:


full['Fare_bin'] = pd.qcut(full['Fare'], 6)

Create a family size variable to see if there's any reason to believe that smaller families had a better chance of survival
# In[ ]:


#Creating new family_size column
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1


# In[ ]:


#The fare for the 2 rows is 80. Let's see which class and Embarked combination gives the closest Median Fare to 80
print(full.groupby(['Pclass', 'Embarked'])['Fare'].median())

#Boxplot to show the median values for different groups. (1,c) has a median value of 80
medianprops = dict(linestyle='-', linewidth=1, color='k')
full.boxplot(column='Fare',by=['Pclass','Embarked'], medianprops=medianprops, showmeans=False, showfliers=False)


# In[ ]:


#full = pd.get_dummies(full, columns=['Embarked'])
full['Embarked'] = pd.factorize(full['Embarked'])[0]
full['Gender'] = pd.factorize(full['Sex'])[0]
full.info()


# In[ ]:


full.rename(columns={"Fare_[0, 7.75]": "Fare_1"
                                ,"Fare_(7.75, 7.896]": "Fare_2"
                                ,"Fare_(7.896, 9.844]": "Fare_3"
                                ,"Fare_(9.844, 14.454]": "Fare_4"
                                ,"Fare_(14.454, 24.15]": "Fare_5"
                                ,"Fare_(24.15, 31.275]": "Fare_6"
                                ,"Fare_(31.275, 69.55]": "Fare_7"
                                ,"Fare_(69.55, 512.329]": "Fare_8"}, inplace=True)
full.info()


# In[ ]:


cols = full.columns.tolist()
cols


# In[ ]:


feature_cols = ['Fare', 'Parch', 'SibSp', 'Pclass', 'FamilySize', 'Title','PassengerType', 'Gender']


# In[ ]:


AgeNotNull = full.loc[full.Age.notnull(),:].copy()
AgeNull = full.loc[full.Age.isnull(),:].copy()


# In[ ]:


X = AgeNotNull[feature_cols]
y = AgeNotNull.Age

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)


# In[ ]:


p = lm.predict(AgeNotNull[feature_cols])

# Now we can constuct a vector of errors
err = abs(p-y)
#print(y[:10])
#print(p[:10])
# Let's see the error on the first 10 predictions
print (err[:10])


# In[ ]:


# predict for a new observation
p1 = lm.predict(AgeNull[feature_cols])
print(p1[:10])
p1.shape
AgeNull.shape


# In[ ]:


AgeSer = full.loc[full.Age.notnull(),'Age']
plt.hist(AgeSer)
plt.ylabel("Count")
plt.xlabel("Age")
plt.show()


# In[ ]:


full.loc[full.Age.isnull(), 'Age'] = p1


# In[ ]:


train = full.loc['train']
test = full.loc['test']


# In[ ]:


y = train.loc[:,'Survived']


# In[ ]:


X = train.loc[:,['PassengerId','Age','Fare', 'Pclass','Title','PassengerType','FamilySize','Embarked','Gender']]


# In[ ]:


train_data = train.values
train_X = X.values
train_y = y.values


# In[ ]:


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV ,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc


# In[ ]:


logreg = LogisticRegression(class_weight='balanced')
param = {'C':[0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1,2,3,3,4,5,10,20]}
clf = GridSearchCV(logreg,param,scoring='roc_auc',refit=True,cv=10)
clf.fit(X,y)
print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_))


# In[ ]:


kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=20)
pred_test_full =0
cv_score =[]
i=1
for train_index,test_index in kf.split(X,y):
    print('{} of KFold {}'.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y.loc[train_index],y.loc[test_index]
    
    #model
    lr = LogisticRegression(C=2)
    lr.fit(xtr,ytr)
    score = roc_auc_score(yvl,lr.predict(xvl))
    print('ROC AUC score:',score)
    
    i+=1


# In[ ]:




