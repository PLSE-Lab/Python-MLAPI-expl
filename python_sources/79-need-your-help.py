from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.datasets import make_classification
from time import time
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import RandomizedPCA
import pandas as pd
import numpy as np
import seaborn as sns
import re
import scipy.stats as stats
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.wrappers.scikit_learn import KerasClassifier


## Import data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_data["Embarked"] = train_data["Embarked"].astype("category")
test_data["Embarked"] = train_data["Embarked"].astype("category")
IDtest = test_data["PassengerId"]




## Convert Sex and Embarked to numerical values
train_data['Sex'].replace(['male','female'],[0,1],inplace=True)
train_data["Sex"] = train_data["Sex"].astype("int")
train_data["Embarked"].replace(["S", "C", "Q"],[0,1,2],inplace=True)
train_data.loc[(train_data.Embarked.isnull()), "Embarked"]=1
train_data["Embarked"] = train_data["Embarked"].astype("int")

## Convert Sex and Embarked to numerical values
test_data['Sex'].replace(['male','female'],[0,1],inplace=True)
test_data["Sex"] = test_data["Sex"].astype("int")
test_data["Embarked"].replace(["S", "C", "Q"],[0,1,2],inplace=True)
test_data.loc[(test_data.Embarked.isnull()), "Embarked"]=0
test_data["Embarked"] = test_data["Embarked"].astype("int")
    

## Create Family Size feature. We cap at 7 since anything larger than that is rare.
train_data['family_size']=train_data['SibSp'] + train_data['Parch']
train_data['family_size']=train_data['family_size'].astype('int')
test_data['family_size']=test_data['SibSp'] + test_data['Parch']
test_data['family_size']=test_data['family_size'].astype('int')
train_data.loc[(train_data['family_size']>7), 'family_size']=7
test_data.loc[(test_data['family_size']>7), 'family_size']=7

## Break each family size into its own column. Similar to hot encoding. I'll be eliminating one column later.
train_data['family_size0']=0
train_data.loc[(train_data['family_size']==0), 'family_size0']=1
train_data['family_size1']=0
train_data.loc[(train_data['family_size']==1), 'family_size1']=1
train_data['family_size2']=0
train_data.loc[(train_data['family_size']==2), 'family_size2']=1
train_data['family_size3']=0
train_data.loc[(train_data['family_size']==3), 'family_size3']=1
train_data['family_size4']=0
train_data.loc[(train_data['family_size']==4), 'family_size4']=1
train_data['family_size5']=0
train_data.loc[(train_data['family_size']==5), 'family_size5']=1
train_data['family_size6']=0
train_data.loc[(train_data['family_size']==6), 'family_size6']=1
train_data['family_size7']=0
train_data.loc[(train_data['family_size']==7), 'family_size7']=1

## Break each family size into its own column. Similar to hot encoding. I'll be eliminating one column later.
test_data['family_size0']=0
test_data.loc[(test_data['family_size']==0), 'family_size0']=1
test_data['family_size1']=0
test_data.loc[(test_data['family_size']==1), 'family_size1']=1
test_data['family_size2']=0
test_data.loc[(test_data['family_size']==2), 'family_size2']=1
test_data['family_size3']=0
test_data.loc[(test_data['family_size']==3), 'family_size3']=1
test_data['family_size4']=0
test_data.loc[(test_data['family_size']==4), 'family_size4']=1
test_data['family_size5']=0
test_data.loc[(test_data['family_size']==5), 'family_size5']=1
test_data['family_size6']=0
test_data.loc[(test_data['family_size']==6), 'family_size6']=1
test_data['family_size7']=0
test_data.loc[(test_data['family_size']==7), 'family_size7']=1

## Create Age missing feature
train_data['Age_Missing'] = 0
train_data.loc[(train_data.Age.isnull()), 'Age_Missing']=1
test_data['Age_Missing'] = 0
test_data.loc[(train_data.Age.isnull()), 'Age_Missing']=1



##
# This section uses random sampling with a controlled seed to imputate Age.
# This technique was as accurate as using an ANN to predict missing age columns.
# We use pclass as the random controll seed.
##

observation_group0 = train_data.loc[(train_data['Pclass']==1)]
obs_group0_sample = train_data.Age.dropna().sample(observation_group0['Age'].isnull().sum(), random_state=0)
obs_group0_sample.index = observation_group0[observation_group0['Age'].isnull()].index
train_data.loc[(train_data['Pclass']==1) & (train_data.Age.isnull()), 'Age']=obs_group0_sample
#print(data.loc[(data['family_size']==1) & (data.Age.isnull())])

observation_group1 = train_data.loc[(train_data['Pclass']==2)]
obs_group1_sample = train_data.Age.dropna().sample(observation_group1['Age'].isnull().sum(), random_state=1)
obs_group1_sample.index = observation_group1[observation_group1['Age'].isnull()].index
train_data.loc[(train_data['Pclass']==2) & (train_data.Age.isnull()), 'Age']=obs_group1_sample
#print(data.loc[(data['family_size']==2) & (data.Age.isnull())])

observation_group2 = train_data.loc[(train_data['Pclass']==3)]
obs_group2_sample = train_data.Age.dropna().sample(observation_group2['Age'].isnull().sum(), random_state=2)
obs_group2_sample.index = observation_group2[observation_group2['Age'].isnull()].index
train_data.loc[(train_data['Pclass']==3) & (train_data.Age.isnull()), 'Age']=obs_group2_sample
#print(data.loc[(data['family_size']==3) & (data.Age.isnull())])


observation_group0 = test_data.loc[(test_data['Pclass']==1)]
obs_group0_sample = test_data.Age.dropna().sample(observation_group0['Age'].isnull().sum(), random_state=0)
obs_group0_sample.index = observation_group0[observation_group0['Age'].isnull()].index
test_data.loc[(test_data['Pclass']==1) & (test_data.Age.isnull()), 'Age']=obs_group0_sample
#print(data.loc[(data['family_size']==1) & (data.Age.isnull())])

observation_group1 = test_data.loc[(test_data['Pclass']==2)]
obs_group1_sample = test_data.Age.dropna().sample(observation_group1['Age'].isnull().sum(), random_state=1)
obs_group1_sample.index = observation_group1[observation_group1['Age'].isnull()].index
test_data.loc[(test_data['Pclass']==2) & (test_data.Age.isnull()), 'Age']=obs_group1_sample
#print(data.loc[(data['family_size']==2) & (data.Age.isnull())])

observation_group2 = test_data.loc[(test_data['Pclass']==3)]
obs_group2_sample = test_data.Age.dropna().sample(observation_group2['Age'].isnull().sum(), random_state=2)
obs_group2_sample.index = observation_group2[observation_group2['Age'].isnull()].index
test_data.loc[(test_data['Pclass']==3) & (test_data.Age.isnull()), 'Age']=obs_group2_sample
#print(data.loc[(data['family_size']==3) & (data.Age.isnull())])


## Bin the ages. We use retbin on the train data so that we can ensure that the train data is binned the same way
train_data['Age_Bin'], agebins=pd.qcut(train_data.Age, q=8, labels=False, retbins=True, precision=3, duplicates='raise')
test_data['Age_Bin']=pd.cut(train_data.Age, bins=agebins, labels=False)

## Broke out each Age bin into its own column like hot encoding.
train_data['Age_Bin0']=0
train_data.loc[(train_data['Age_Bin']==0), 'Age_Bin0']=1
train_data['Age_Bin1']=0
train_data.loc[(train_data['Age_Bin']==1), 'Age_Bin1']=1
train_data['Age_Bin2']=0
train_data.loc[(train_data['Age_Bin']==2), 'Age_Bin2']=1
train_data['Age_Bin3']=0
train_data.loc[(train_data['Age_Bin']==3), 'Age_Bin3']=1
train_data['Age_Bin4']=0
train_data.loc[(train_data['Age_Bin']==4), 'Age_Bin4']=1
train_data['Age_Bin5']=0
train_data.loc[(train_data['Age_Bin']==5), 'Age_Bin5']=1
train_data['Age_Bin6']=0
train_data.loc[(train_data['Age_Bin']==6), 'Age_Bin6']=1
train_data['Age_Bin7']=0
train_data.loc[(train_data['Age_Bin']==7), 'Age_Bin7']=1

test_data['Age_Bin0']=0
test_data.loc[(test_data['Age_Bin']==0), 'Age_Bin0']=1
test_data['Age_Bin1']=0
test_data.loc[(test_data['Age_Bin']==1), 'Age_Bin1']=1
test_data['Age_Bin2']=0
test_data.loc[(test_data['Age_Bin']==2), 'Age_Bin2']=1
test_data['Age_Bin3']=0
test_data.loc[(test_data['Age_Bin']==3), 'Age_Bin3']=1
test_data['Age_Bin4']=0
test_data.loc[(test_data['Age_Bin']==4), 'Age_Bin4']=1
test_data['Age_Bin5']=0
test_data.loc[(test_data['Age_Bin']==5), 'Age_Bin5']=1
test_data['Age_Bin6']=0
test_data.loc[(test_data['Age_Bin']==6), 'Age_Bin6']=1
test_data['Age_Bin7']=0
test_data.loc[(test_data['Age_Bin']==7), 'Age_Bin7']=1




## Use boxcox to make the fare data distribution more even. We then bin the data with retbin true so that we can bin test data the same way
## We use +1 to avoid 0 values.
train_data['Fare_Box'], param = stats.boxcox(train_data.Fare+1)
train_data['Fare_Bin'], farebins=pd.qcut(train_data.Fare_Box, q=8, labels=False, retbins=True, precision=3, duplicates='raise')

## I failed to document why i chose 13 for the missing value. I suppose I compared this person to others to determine the most likely bin.
test_data.loc[(test_data['Fare'].isnull()), 'Fare']=13
test_data['Fare_Box'], param = stats.boxcox(test_data.Fare+1)
test_data['Fare_Bin']=pd.cut(test_data.Fare_Box, bins=farebins, labels=False)
test_data.loc[(test_data['Fare_Bin'].isnull()), 'Fare_Bin']=0
print(test_data[test_data['Fare_Bin'].isnull()])

## Broke out the fare bins like hot encoding.
train_data['Fare_Bin0']=0
train_data.loc[(train_data['Fare_Bin']==0), 'Fare_Bin0']=1
train_data['Fare_Bin1']=0
train_data.loc[(train_data['Fare_Bin']==1), 'Fare_Bin1']=1
train_data['Fare_Bin2']=0
train_data.loc[(train_data['Fare_Bin']==2), 'Fare_Bin2']=1
train_data['Fare_Bin3']=0
train_data.loc[(train_data['Fare_Bin']==3), 'Fare_Bin3']=1
train_data['Fare_Bin4']=0
train_data.loc[(train_data['Fare_Bin']==4), 'Fare_Bin4']=1
train_data['Fare_Bin5']=0
train_data.loc[(train_data['Fare_Bin']==5), 'Fare_Bin5']=1
train_data['Fare_Bin6']=0
train_data.loc[(train_data['Fare_Bin']==6), 'Fare_Bin6']=1
train_data['Fare_Bin7']=0
train_data.loc[(train_data['Fare_Bin']==7), 'Fare_Bin7']=1
#train_data['Fare_Bin8']=0
#train_data.loc[(train_data['Fare_Bin']==8), 'Fare_Bin8']=1
#train_data['Fare_Bin9']=0
#train_data.loc[(train_data['Fare_Bin']==9), 'Fare_Bin9']=1
#train_data['Fare_Bin10']=0
#train_data.loc[(train_data['Fare_Bin']==10), 'Fare_Bin10']=1
#train_data['Fare_Bin11']=0
#train_data.loc[(train_data['Fare_Bin']==11), 'Fare_Bin11']=1
#train_data['Fare_Bin12']=0
#train_data.loc[(train_data['Fare_Bin']==12), 'Fare_Bin12']=1
#train_data['Fare_Bin13']=0
#train_data.loc[(train_data['Fare_Bin']==13), 'Fare_Bin13']=1
#train_data['Fare_Bin14']=0
#train_data.loc[(train_data['Fare_Bin']==14), 'Fare_Bin14']=1
#train_data['Fare_Bin15']=0
#train_data.loc[(train_data['Fare_Bin']==15), 'Fare_Bin15']=1

## Broke out the fare bins like hot encoding.
test_data['Fare_Bin0']=0
test_data.loc[(test_data['Fare_Bin']==0), 'Fare_Bin0']=1
test_data['Fare_Bin1']=0
test_data.loc[(test_data['Fare_Bin']==1), 'Fare_Bin1']=1
test_data['Fare_Bin2']=0
test_data.loc[(test_data['Fare_Bin']==2), 'Fare_Bin2']=1
test_data['Fare_Bin3']=0
test_data.loc[(test_data['Fare_Bin']==3), 'Fare_Bin3']=1
test_data['Fare_Bin4']=0
test_data.loc[(test_data['Fare_Bin']==4), 'Fare_Bin4']=1
test_data['Fare_Bin5']=0
test_data.loc[(test_data['Fare_Bin']==5), 'Fare_Bin5']=1
test_data['Fare_Bin6']=0
test_data.loc[(test_data['Fare_Bin']==6), 'Fare_Bin6']=1
test_data['Fare_Bin7']=0
test_data.loc[(test_data['Fare_Bin']==7), 'Fare_Bin7']=1
#test_data['Fare_Bin8']=0
#test_data.loc[(test_data['Fare_Bin']==8), 'Fare_Bin8']=1
#test_data['Fare_Bin9']=0
#test_data.loc[(test_data['Fare_Bin']==9), 'Fare_Bin9']=1
#test_data['Fare_Bin10']=0
#test_data.loc[(test_data['Fare_Bin']==10), 'Fare_Bin10']=1
#test_data['Fare_Bin11']=0
#test_data.loc[(test_data['Fare_Bin']==11), 'Fare_Bin11']=1
#test_data['Fare_Bin12']=0
#test_data.loc[(test_data['Fare_Bin']==12), 'Fare_Bin12']=1
#test_data['Fare_Bin13']=0
#test_data.loc[(test_data['Fare_Bin']==13), 'Fare_Bin13']=1
#test_data['Fare_Bin14']=0
#test_data.loc[(test_data['Fare_Bin']==14), 'Fare_Bin14']=1
#test_data['Fare_Bin15']=0
#test_data.loc[(test_data['Fare_Bin']==15), 'Fare_Bin15']=1


#train_data.loc[(train_data["Age"]>=74), 'Age']=73
#train_data['Age'] = train_data.Age**(1/1.2)
#test_data.loc[(test_data["Age"]>=74), 'Age']=73
#test_data['Age'] = train_data.Age**(1/1.2)

##Rare title work such as Mr. and Mrs.
##Names have unique titles such as Mrs or Mr. Let's extract that and place it into a new column.
## I borrowed this exact coding from a course on Udemy: Feature engineering for machine learning by Soledad Galli
def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
##Add title column to train_data with title (i.e. Mr or Mrs)  
train_data['Title'] = train_data['Name'].apply(get_title)
##Convert Title to a number representing the Title
train_data['Title0'] = 0
train_data.loc[(train_data["Title"]=="Mrs"), 'Title0']=1
train_data['Title1'] = 0
train_data.loc[(train_data["Title"]=="Mr"), 'Title1']=1
train_data['Title2'] = 0
train_data.loc[(train_data["Title"]=="Miss"), 'Title2']=1
train_data['Title3'] = 0
train_data.loc[(train_data["Title"]=="Master"), 'Title3']=1
train_data['Title4'] = 0
train_data.loc[(train_data["Title"]=="Other"), 'Title4']=1
#train_data['Title'] = train_data['Title'].astype('int')
train_data.drop(['Title'],axis=1,inplace=True)

##Add title column to train_data with title (i.e. Mr or Mrs)  
test_data['Title'] = test_data['Name'].apply(get_title)
##Convert Title to a number representing the Title
test_data['Title0']=0
test_data.loc[(test_data["Title"]=="Mrs"), 'Title0']=1
test_data['Title1']=0
test_data.loc[(test_data["Title"]=="Mr"), 'Title1']=1
test_data['Title2']=0
test_data.loc[(test_data["Title"]=="Miss"), 'Title2']=1
test_data['Title3']=0
test_data.loc[(test_data["Title"]=="Master"), 'Title3']=1
test_data['Title4']=0
test_data.loc[(test_data["Title"]=="Other"), 'Title4']=1
#test_data['Title'] = train_data['Title'].astype('int')
test_data.drop(['Title'],axis=1,inplace=True)

## Broke out pclass like hot encode
train_data['Pclass1'] = 0
train_data.loc[(train_data['Pclass']==1), 'Pclass1']=1
train_data['Pclass2'] = 0
train_data.loc[(train_data['Pclass']==2), 'Pclass2']=1

## Broke out pclass like hot encode
test_data['Pclass1'] = 0
test_data.loc[(test_data['Pclass']==1), 'Pclass1']=1
test_data['Pclass2'] = 0
test_data.loc[(test_data['Pclass']==2), 'Pclass2']=1

## Create Cabin Missing feature. We end up not using it.
train_data['Cabin_Missing'] = 0
train_data.loc[(train_data.Cabin.isnull()), 'Cabin_Missing']=1
test_data['Cabin_Missing'] = 0
test_data.loc[(test_data.Cabin.isnull()), 'Cabin_Missing']=1

## Useless code. I will cleanup later to just use Sex
train_data['Female']=0
train_data.loc[(train_data['Sex']==1), 'Female']=1
test_data['Female']=0
test_data.loc[(test_data['Sex']==1), 'Female']=1

## Useless code. I will cleanup later to just use Sex
train_data['Male']=0
train_data.loc[(train_data['Sex']==0), 'Male']=1
test_data['Male']=0
test_data.loc[(test_data['Sex']==0), 'Male']=1

## Grab first cabin letter of each observation. Nans show up as "n"
train_data['Cabin_Letter'] = train_data['Cabin'].apply(lambda x: str(x)[0])
test_data['Cabin_Letter'] = test_data['Cabin'].apply(lambda x: str(x)[0])
print(train_data['Cabin_Letter'].value_counts())

## Broke out cabins into columns
train_data['Cabin_C']=0
train_data.loc[(train_data['Cabin_Letter']=='C'), 'Cabin_C']=1
train_data['Cabin_B']=0
train_data.loc[(train_data['Cabin_Letter']=='B'), 'Cabin_B']=1
train_data['Cabin_D']=0
train_data.loc[(train_data['Cabin_Letter']=='D'), 'Cabin_D']=1
train_data['Cabin_E']=0
train_data.loc[(train_data['Cabin_Letter']=='E'), 'Cabin_E']=1
train_data['Cabin_A']=0
train_data.loc[(train_data['Cabin_Letter']=='A'), 'Cabin_A']=1
train_data['Cabin_F']=0
train_data.loc[(train_data['Cabin_Letter']=='F'), 'Cabin_F']=1
train_data['Cabin_G']=0
train_data.loc[(train_data['Cabin_Letter']=='G'), 'Cabin_G']=1
train_data['Cabin_T']=0
train_data.loc[(train_data['Cabin_Letter']=='T'), 'Cabin_T']=1

## Broke out cabins into columns
test_data['Cabin_C']=0
test_data.loc[(test_data['Cabin_Letter']=='C'), 'Cabin_C']=1
test_data['Cabin_B']=0
test_data.loc[(test_data['Cabin_Letter']=='B'), 'Cabin_B']=1
test_data['Cabin_D']=0
test_data.loc[(test_data['Cabin_Letter']=='D'), 'Cabin_D']=1
test_data['Cabin_E']=0
test_data.loc[(test_data['Cabin_Letter']=='E'), 'Cabin_E']=1
test_data['Cabin_A']=0
test_data.loc[(test_data['Cabin_Letter']=='A'), 'Cabin_A']=1
test_data['Cabin_F']=0
test_data.loc[(test_data['Cabin_Letter']=='F'), 'Cabin_F']=1
test_data['Cabin_G']=0
test_data.loc[(test_data['Cabin_Letter']=='G'), 'Cabin_G']=1
test_data['Cabin_T']=0
test_data.loc[(test_data['Cabin_Letter']=='T'), 'Cabin_T']=1



## This is where we choose what features to use. We also apply the MinMaxScaler
print(train_data.head(1))
test_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Age_Missing', 'Age', 'Embarked', 'Pclass', 'Cabin_Missing', 'Sex', 'Male', 'Cabin_Letter', 'Cabin_T', 'Cabin_F', 'Cabin_G', 'Cabin_E', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Age_Bin0','Fare_Box', 'Title4', 'family_size7'],axis=1,inplace=True)
train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Age_Missing', 'Age', 'Embarked', 'Pclass', 'Cabin_Missing', 'Sex', 'Male', 'Cabin_Letter', 'Cabin_T', 'Cabin_F', 'Cabin_G', 'Cabin_E', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Age_Bin0','Fare_Box', 'Title4', 'family_size7'],axis=1,inplace=True)
scaler = MinMaxScaler()
train_data[['Parch', 'SibSp', 'Age_Bin', 'family_size', 'Fare', 'Fare_Bin','Title0', 'Title1', 'Title2', 'Title3', 'Female', 'Pclass1', 'Age_Bin1', 'Age_Bin2', 'Age_Bin3', 'Age_Bin4', 'Age_Bin5', 'Age_Bin6', 'Age_Bin7', 'Pclass2', 'Fare_Bin0', 'Fare_Bin1', 'Fare_Bin2', 'Fare_Bin3', 'Fare_Bin4', 'Fare_Bin5', 'Fare_Bin6', 'Fare_Bin7', 'family_size0', 'family_size1', 'family_size2', 'family_size3', 'family_size4', 'family_size5', 'family_size6']] = scaler.fit_transform(train_data[['Parch', 'SibSp', 'Age_Bin', 'family_size', 'Fare', 'Fare_Bin','Title0', 'Title1', 'Title2', 'Title3', 'Female', 'Pclass1', 'Age_Bin1', 'Age_Bin2', 'Age_Bin3', 'Age_Bin4', 'Age_Bin5', 'Age_Bin6', 'Age_Bin7', 'Pclass2', 'Fare_Bin0', 'Fare_Bin1', 'Fare_Bin2', 'Fare_Bin3', 'Fare_Bin4', 'Fare_Bin5', 'Fare_Bin6', 'Fare_Bin7', 'family_size0', 'family_size1', 'family_size2', 'family_size3', 'family_size4', 'family_size5', 'family_size6']])
test_data[['Parch', 'SibSp', 'Age_Bin', 'family_size', 'Fare', 'Fare_Bin','Title0', 'Title1', 'Title2', 'Title3', 'Female', 'Pclass1', 'Age_Bin1', 'Age_Bin2', 'Age_Bin3', 'Age_Bin4', 'Age_Bin5', 'Age_Bin6', 'Age_Bin7', 'Pclass2', 'Fare_Bin0', 'Fare_Bin1', 'Fare_Bin2', 'Fare_Bin3', 'Fare_Bin4', 'Fare_Bin5', 'Fare_Bin6', 'Fare_Bin7', 'family_size0', 'family_size1', 'family_size2', 'family_size3', 'family_size4', 'family_size5', 'family_size6']] = scaler.transform(test_data[['Parch', 'SibSp', 'Age_Bin', 'family_size', 'Fare', 'Fare_Bin','Title0', 'Title1', 'Title2', 'Title3', 'Female', 'Pclass1', 'Age_Bin1', 'Age_Bin2', 'Age_Bin3', 'Age_Bin4', 'Age_Bin5', 'Age_Bin6', 'Age_Bin7', 'Pclass2', 'Fare_Bin0', 'Fare_Bin1', 'Fare_Bin2', 'Fare_Bin3', 'Fare_Bin4', 'Fare_Bin5', 'Fare_Bin6', 'Fare_Bin7', 'family_size0', 'family_size1', 'family_size2', 'family_size3', 'family_size4', 'family_size5', 'family_size6']])
print(train_data.head(1))

##Original features
#print(train_data.head(1))
#test_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Age_Missing', 'Fare', 'Age', 'Embarked', 'Fare_Square', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Cabin_Missing', 'Sex', 'Male', 'Fare_Bin', 'Cabin_Letter', 'Cabin_T', 'Cabin_F', 'Cabin_G', 'Cabin_E', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Pclass3', 'Age_Bin', 'Age_Bin0','Fare_Box'],axis=1,inplace=True)
#train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Age_Missing', 'Fare', 'Age', 'Embarked', 'Fare_Square', 'Fare', 'Parch', 'SibSp', 'Pclass', 'Cabin_Missing', 'Sex', 'Male', 'Fare_Bin', 'Cabin_Letter', 'Cabin_T', 'Cabin_F', 'Cabin_G', 'Cabin_E', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Pclass3', 'Age_Bin', 'Age_Bin0','Fare_Box'],axis=1,inplace=True)
#scaler = MinMaxScaler()
#train_data[['Title0', 'Title1', 'Title2', 'Title3', 'family_size', 'Female', 'Pclass1', 'Age_Bin1', 'Age_Bin2', 'Age_Bin3', 'Age_Bin4', 'Age_Bin5', 'Age_Bin6', 'Age_Bin7']] = scaler.fit_transform(train_data[['Title0', 'Title1', 'Title2', 'Title3', 'family_size', 'Female', 'Pclass1', 'Age_Bin1', 'Age_Bin2', 'Age_Bin3', 'Age_Bin4', 'Age_Bin5', 'Age_Bin6', 'Age_Bin7']])
#test_data[['Title0', 'Title1', 'Title2', 'Title3', 'family_size', 'Female', 'Pclass1', 'Age_Bin1', 'Age_Bin2', 'Age_Bin3', 'Age_Bin4', 'Age_Bin5', 'Age_Bin6', 'Age_Bin7']] = scaler.transform(test_data[['Title0', 'Title1', 'Title2', 'Title3', 'family_size', 'Female', 'Pclass1', 'Age_Bin1', 'Age_Bin2', 'Age_Bin3', 'Age_Bin4', 'Age_Bin5', 'Age_Bin6', 'Age_Bin7']])
#print(train_data.head(1))

#print(train_data.isnull().sum())

## Break data into train/test split
train,test=cross_validation.train_test_split(train_data,test_size=0.20,random_state=42,stratify=train_data['Survived'])
features_train=train[train.columns[1:]]
labels_train=train[train.columns[:1]]
features_test=test[test.columns[1:]]
labels_test=test[test.columns[:1]]
features=train_data[train_data.columns[1:]]
labels=train_data['Survived']

## Setup Extra Tree Classifier
etc = ExtraTreesClassifier(bootstrap = False, criterion = 'gini', max_features=12, max_depth = None,  min_samples_leaf = 5, min_samples_split = 40, n_estimators = 220)
#t0 = time()
etc.fit(features_train, labels_train.values.ravel())
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
etcpred = etc.predict(features_test)
accuracy = etc.score(features_test, labels_test)
pred_recall = recall_score(labels_test, etcpred)
pred_precision = precision_score(labels_test, etcpred)
print("ETC Recall", pred_recall)
print("ETC Precision", pred_precision)
#print pred
print("ETC Accuracy", accuracy)



## Setup Random Forrest

rf = RandomForestClassifier(min_samples_split=21, max_features=9, min_samples_leaf=5, n_estimators=14)
#t0 = time()
rf.fit(features_train, labels_train.values.ravel())
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
rfpred = rf.predict(features_test)
accuracy = rf.score(features_test, labels_test)
pred_recall = recall_score(labels_test, rfpred)
pred_precision = precision_score(labels_test, rfpred)
print("RF Recall", pred_recall)
print("RF Precision", pred_precision)
#print pred
print("RF Accuracy", accuracy)


##AdaBoost
#abc = AdaBoostClassifier()
#t0 = time()
#abc.fit(features_train, labels_train.values.ravel())
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
#abcpred = abc.predict(features_test)
#accuracy = abc.score(features_test, labels_test)
#pred_recall = recall_score(labels_test, abcpred)
#pred_precision = precision_score(labels_test, abcpred)
#print("ABC Recall", pred_recall)
#print("ABC Precision", pred_precision)
#print pred
#print("ABC Accuracy", accuracy)



##GradientBoost

gbc = GradientBoostingClassifier()
#t0 = time()
gbc.fit(features_train, labels_train.values.ravel())
#print "training time:", round(time()-t0, 3), "s"
#t0 = time()
gbcpred = gbc.predict(features_test)
accuracy = gbc.score(features_test, labels_test)
pred_recall = recall_score(labels_test, gbcpred)
pred_precision = precision_score(labels_test, gbcpred)
print("GBC Recall", pred_recall)
print("GBC Precision", pred_precision)
#print pred
print("GBC Accuracy", accuracy)



##Logistic Regression
#lgr = LogisticRegression(C=1000)
#lgr.fit(features_train, labels_train.values.ravel())
#lgrpred = lgr.predict(features_test)
#accuracy = lgr.score(features_test, labels_test)
#pred_recall = recall_score(labels_test, lgrpred)
#pred_precision = precision_score(labels_test, lgrpred)
#print("LGR Recall", pred_recall)
#print("LGR Precision", pred_precision)
#print pred
#print("LGR Accuracy", accuracy)



##XGBoost

xgb = xgb.XGBClassifier(colsample_bytree = 0.8, learning_rate = 0.5, max_depth = 1, min_child_weight = 5, n_estimators = 400, subample = 0.8)
xgb.fit(features_train, labels_train.values.ravel())
xgbpred = xgb.predict(features_test)
accuracy = xgb.score(features_test, labels_test)
pred_recall = recall_score(labels_test, xgbpred)
pred_precision = precision_score(labels_test, xgbpred)
print("XGB Recall", pred_recall)
print("XGB Precision", pred_precision)
#print pred
print("XGB Accuracy", accuracy)

## Small Vector Machine
svmc = SVC(probability=True, C = 300, gamma = 0.01, kernel = 'rbf')
svmc.fit(features_train, labels_train.values.ravel())
svmcpred = svmc.predict(features_test)
accuracy = svmc.score(features_test, labels_test)
pred_recall = recall_score(labels_test, svmcpred)
pred_precision = precision_score(labels_test, svmcpred)
print("SVM Recall", pred_recall)
print("SVM Precision", pred_precision)
#print pred
print("SVM Accuracy", accuracy)

## Setup ANN as a function so that it can be passed through a VotingC
def build_classifier():
    annc = Sequential()
    annc.add(Dense(output_dim = 17, init = 'uniform', activation = LeakyReLU(), input_dim = 35))
    #annc.add(LeakyReLU(alpha=0.01))
    #annc.add(Dropout(p = 0.1))
    annc.add(Dense(output_dim = 17, init = 'uniform', activation = LeakyReLU()))
    #annc.add(LeakyReLU(alpha=0.01))
    #annc.add(Dropout(p = 0.1))
    annc.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    annc.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    return annc
annc = KerasClassifier(build_fn = build_classifier, batch_size = 15, epochs = 50)
#accuracies = cross_val_score(estimator = annc, X = features_train, y = labels_train, cv = 3, n_jobs = 1)



## VotingC that combines the above algorythms.
votingC = VotingClassifier(estimators=[('etc', etc),('rf',rf),('gbc',gbc),('annc',annc),('svmc',svmc),('xgb', xgb)], voting='soft', weights=[1,1,1,1,1,2])
votingC = votingC.fit(features_train,labels_train.values.ravel())
pred = votingC.predict(features_test)


accuracy = votingC.score(features_test, labels_test)
pred_recall = recall_score(labels_test, pred)
pred_precision = precision_score(labels_test, pred)
print("Recall", pred_recall)
print("Precision", pred_precision)
#print pred
print("Accuracy", accuracy)
print('VotingC TEST roc-auc: {}'.format(roc_auc_score(labels_test, pred)))

## Output results to CSV for submission
test_Survived = pd.Series(votingC.predict(test_data), name="Survived")
results = pd.concat([IDtest,test_Survived],axis=1)
results.to_csv("submit_to_kaggle2.csv",index=False)
