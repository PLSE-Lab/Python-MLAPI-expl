#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
import numpy as np 
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# In[ ]:


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier


# In[ ]:


testDf=pd.read_csv("../input/test.csv")
trainDf=pd.read_csv("../input/train.csv")
genderDf=pd.read_csv('../input/gender_submission.csv')


# In[ ]:


testDf.info()


# In[ ]:


passengerID=testDf['PassengerId']


# In[ ]:


titanicDf=pd.concat([testDf,trainDf],keys=['Test','Train'],names=['Dataset','Dataset ID'])


# In[ ]:


titanicDf.head()


# In[ ]:


titanicDf.tail()


# so we have successfully concatenated the train and test data

# In[ ]:


# titanicDf.xs('Train').head()  # Another method for doing so
titanicDf.loc['Train'].head()


# In[ ]:


titanicDf.loc['Test'].head()


# In[ ]:


titanicDf.info()


# As we can see the whole df contains alot of the null values 

# In[ ]:


titanicDf.xs('Train').info()


# In[ ]:


titanicDf.xs('Test').info()


# In[ ]:


titanicDf.xs('Train').describe()


# In[ ]:


titanicDf.xs("Test").describe()


# In[ ]:


titanicDf.xs('Train').hist(bins=20,figsize=(15,10))


# In[ ]:


# As we can see there are a couple of null values that we have to resolve
titanicDf.xs('Train').isnull().sum()


# In[ ]:


# To find the most repetative data in the Embarked column
embarked_modeSeries=titanicDf.xs('Train')['Embarked'].dropna().mode()


# In[ ]:


embarked_mode=embarked_modeSeries[0]
embarked_mode


# In[ ]:


titanicDf['Embarked'].fillna(embarked_mode,inplace=True)


# In[ ]:


titanicDf['Embarked'][titanicDf['Embarked'].isnull()==True]


# In[ ]:


titanicDf.isnull().sum()


# Data preprocessing for the fare 

# In[ ]:


FareMode=titanicDf['Fare'].mode()
FareMode


# In[ ]:


titanicDf['Fare'].fillna(FareMode[0],inplace=True)


# In[ ]:


titanicDf['Fare'].isnull().sum()


# Data Preprocessing for the age column

# In[ ]:


titanicDf.xs('Train').corr()['Age'].sort_values(ascending=False)


# In[ ]:


titanicDf.xs('Train')[['Age','Sex']].groupby('Sex').mean().sort_values(by='Age',ascending=False)


# In[ ]:


titanicDf['Pclass'].unique()


# Now we are going to fill the null values of the age and fill the null values of it , according to the median value of its sex and pclass

# In[ ]:


for valAge in ['male','female']:
    for x in range(0,3):
        titanicDfMedianAge=titanicDf.xs('Train')[(titanicDf.xs('Train')['Sex']==valAge) &
                                                 (titanicDf.xs('Train')['Pclass']==x+1)]['Age'].dropna().median()
        print('the median age is ',titanicDfMedianAge)
        
        titanicDf.loc[(titanicDf["Age"].isnull()) & (titanicDf["Sex"] == valAge) & (titanicDf["Pclass"] == x+1), "Age"] = titanicDfMedianAge
        


# In[ ]:


# Display specified ages for test 
# titanicDf.loc[(titanicDf["Sex"] == valAge) & (titanicDf["Pclass"] == x+1),"Age"]


# Data preprocessing for cabin column

# In[ ]:


titanicDf.loc['Train','Cabin'].unique()


# In[ ]:


titanicDf.loc['Train','Cabin'].isnull().sum()


# In[ ]:


titanicDf.fillna('None',inplace=True)


# In[ ]:


titanicDf.loc['Train','Cabin'].isnull().sum()


# In[ ]:


titanicDf.isnull().sum()


# so we have sucessfully removed all the missing values ****

# *Now we are done with the missing values now we can work on the fetaure extraction modules.*

# In[ ]:


titanicDf['Title']=titanicDf['Name'].str.extract("([A-Za-z]+)\.",expand=False)


# In[ ]:


# Listing out the unique titles that we have created
set(titanicDf['Title'])


# In[ ]:


pd.crosstab(titanicDf['Title'],titanicDf['Sex'])


# Okay, Now we can have a rough insights from the title of the peoples and most of it makes sense 

# What we can do better is we can group the titles to make it easier for us

# In[ ]:


titanicDf['Title'].replace('Mme','Mrs',inplace=True)
titanicDf['Title'].replace('Ms','Miss',inplace=True)
titanicDf["Title"].replace(["Capt", "Col", "Countess", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major", "Rev", "Sir"], "Special", inplace=True)
titanicDf['Title'].replace('Mlle','Miss',inplace=True)


# In[ ]:


# titanicDf.xs('Train')[['Survived','Title']].groupby(['Title']).mean().sort_values(by='Survived',ascending=False)titanicDf


# In[ ]:


titanicDf.loc['Train'][['Title','Survived']].groupby('Title').sum().sort_values(by='Survived',ascending=False)


# we can see that the total number of female passengers who survived were more than that of males
# 

# So now what we can do is classify the Age group into some sort of categorical values, we will use the cut function of the pandas

# In[ ]:


pd.cut(titanicDf.loc['Train','Age'],bins=5).dtype


# In[ ]:


titanicDf.loc[titanicDf['Age']<16,'Age']=0
titanicDf.loc[(titanicDf['Age']>=16) & (titanicDf['Age']<32),'Age']=1
titanicDf.loc[(titanicDf['Age']>=32) & (titanicDf['Age']<48),'Age']=2
titanicDf.loc[(titanicDf['Age']>=48) & (titanicDf['Age']<64),'Age']=3
titanicDf.loc[(titanicDf['Age']>=64),'Age']=4


# In[ ]:


titanicDf['Age'].sort_values().unique()


# In[ ]:


# Its better to have such values in int type 
titanicDf['Age']=titanicDf['Age'].astype(int)


# In[ ]:


titanicDf['Age'].value_counts()


# In[ ]:


titanicDf.loc['Train'][['Age','Survived']].groupby('Age').sum().sort_values(by='Survived',ascending=False)


# sibsp is the number of siblings/spouses on board while parch is the number of  parents/childrens on board. We can combing both to create a family feature 

# In[ ]:


# Including the passenger on board
titanicDf['Family']=titanicDf['SibSp']+titanicDf['SibSp']+1 


# In[ ]:


titanicDf.loc['Train'][['Family','Survived']].groupby('Family').sum()


# As we can  observe that the passenger who , on board were  single , were the most to survive

# we can create another feature to demonstrate if the passenger was alone or not 

# In[ ]:


titanicDf['IsAlone']=0
titanicDf.loc[titanicDf['Family']>1,"IsAlone"]=1


# In[ ]:


titanicDf.loc['Train'][['IsAlone','Survived']].groupby('Survived').mean().sort_values(by='IsAlone',ascending=False)


# Also similar to the Age , we can group the fare into differnet categories

# In[ ]:


set(pd.qcut(titanicDf['Fare'],q=4))


# In[ ]:


titanicDf.loc[titanicDf['Fare']<=7.896,'Fare']=0
titanicDf.loc[(titanicDf['Fare']>7.896) & (titanicDf['Fare']<=14.454),'Fare']=1
titanicDf.loc[(titanicDf['Fare']>14.454) & (titanicDf['Fare']<=31.275),'Fare']=2
titanicDf.loc[(titanicDf['Fare']>31.275),'Fare']=3


# In[ ]:


titanicDf['Fare'].astype(int)


# In[ ]:


titanicDf['Fare'].unique()


# In[ ]:


titanicDf['Fare'].value_counts()


# In[ ]:


titanicDf.loc['Train'][['Fare','Survived']].groupby('Fare').sum().sort_values(by='Survived',ascending=False)


# We   can observe that the passengers who had high class tickets were the one to survive and the chances of survival of passenger who had lower ticket were comparatively low

# In[ ]:


titanicDf['Cabin'].isnull().sum()


# As we have no null values, we can continue with our feature extraction process

# In[ ]:


titanicDf['Cabin']=titanicDf['Cabin'].str.extract("([A-Za-z]+)",expand=False)


# In[ ]:


titanicDf.loc['Train'][['Cabin','Survived']].groupby(['Cabin']).sum().sort_values(by='Survived',ascending=False)


# Suprisingly , the passeners who didn't had a cabin  in the particular set of datset were the most to survive, and the cabin 'T' passengers were the least to survive

# Now it seems that our data preprocessing is almost complete but at last lets check our complete head of data frame and  drop irrelevant features 

# In[ ]:


titanicDf.info()


# In[ ]:


titanicDf.drop(['Name','PassengerId','Ticket'],axis=1,inplace=True)


# In[ ]:


titanicDf.head()


# Now further looking into the data we can see there are a lot of fetaures not having a numerical value. The machine learning model just understands the numerical value.
# So we will be using the scikit  learn libraries to change the categorical features into numerical values 

# In[ ]:


titanicDf['Survived'].value_counts()


# In[ ]:


labelEncoder=LabelEncoder()
titanicDfEncodedTrain=titanicDf.loc['Train'].apply(labelEncoder.fit_transform)
titanicDfEncodedTest=titanicDf.loc['Test'].apply(labelEncoder.fit_transform)


# In[ ]:


titanicDfEncodedTrain.head()


# In[ ]:


titanicDfEncodedTest.head()


# Now we have succesfully converted all our data into numerical data

# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(titanicDfEncodedTrain.corr(),annot=True,)


# As we can see there is a good corelation between relation between sibsp, family as the feature family is derived from these only features

# Now we will check the importance of each features and decide which of the features to keep for building the model 

# In[ ]:


X_train=titanicDfEncodedTrain.drop('Survived',axis=1)
y_train=titanicDfEncodedTrain['Survived']


# In[ ]:


randomForestClassifier=RandomForestClassifier()
randomForestClassifier.fit(X_train,y_train)


# In[ ]:


randomForestClassifier.feature_importances_


#     The above array shows all the importances of the respective features

# So what we can do is, take the necessary features and remove all  the features that do not commit much to the classification task

# In[ ]:


# Zips the feature columns to the  feature importances 
feature_importances=zip(list(X_train.columns.values),randomForestClassifier.feature_importances_)

# sort acc to the feature importances 
feature_importances=sorted(feature_importances,key=lambda feature:feature[1],reverse=True)

# print the columns names and its importances in a good fashion
for name,score in feature_importances:
    print("{:10} | {}".format(name,score))


#  As we can see that between the cabin is of low importance as compared to pclass of which it is derived from. so instead of keeping both we will remove the cabin feature

# In[ ]:


titanicDf.drop('Cabin',axis=1,inplace=True)


# So  now we have removed the fetaures that are less important to us now , get the dummies from the dataframe to remove the categorical data 

# In[ ]:


y_train=titanicDf.loc['Train']['Survived']


# In[ ]:


X_titanicdf=pd.get_dummies(titanicDf.drop('Survived',axis=1))
y_titanic=titanicDf['Survived']


# Now that we have converted all the variables into dummies, now split the data into train and  test.

# In[ ]:


X_train=X_titanicdf.loc['Train']
y_train=y_titanic.loc['Train'].astype(int)
X_test=X_titanicdf.loc['Test']


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


X_test.head()


# Now, scale the data using the Standard Scaler library of scikit learn 

# In[ ]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[ ]:


logisticClassifier=LogisticRegression()
cross_val_score(logisticClassifier,X_train,y_train,cv=10,scoring='accuracy').mean()


# In[ ]:


sgcClassifer=SGDClassifier()
cross_val_score(sgcClassifer,X_train,y_train,scoring='accuracy').mean()


# In[ ]:


svcClassifier=SVC()
cross_val_score(svcClassifier,X_train,y_train,scoring='accuracy').mean()


# In[ ]:


ldaClassifier=LinearDiscriminantAnalysis()
cross_val_score(ldaClassifier,X_train,y_train,scoring='accuracy').mean()


# Now as we have checked the different accuracies of the models, Now we will use Grid Search Cv to find the best params

# In[ ]:


grid_params=[
    {
    "C":[4,5,6],
    "kernel":["rbf"],
    "tol":[0.00001,0.00003,0.00005,0.00008],
    "gamma":["auto","scale"],
    "class_weight": ["balanced", None],
    "shrinking":[True,False],
    "probability":[True]
    },
    {
        "kernel":["linear"],
        "degree":[1,3,5],
        "gamma":['auto',"scale"],
        "probability":[True]
    }
    ]


# In[ ]:


gridsearchCV=GridSearchCV(estimator=svcClassifier,param_grid=grid_params,verbose=2,scoring="accuracy")


# In[ ]:


gridsearchCV.fit(X_train,y_train)


# In[ ]:


gridsearchCV.best_params_


# In[ ]:


gridsearchCV.best_score_


# In[ ]:


svcClassifier=gridsearchCV.best_estimator_


# In[ ]:


cross_val_score(svcClassifier,X_train,y_train,scoring='accuracy').mean()


# As we have tested many models, Now Voting classifier will be used to combine all these models as these models kind of have a similar accuracy score 

# In[ ]:


votingClassifierEstimators=[("svc",svcClassifier),
                           ("lda",ldaClassifier),
                            ("Logistic Classifier",logisticClassifier)
                           ]


# In[ ]:


votingClassifier=VotingClassifier(estimators=votingClassifierEstimators,voting="soft")


# In[ ]:


votingClassifier.fit(X_train,y_train)


# In[ ]:


cross_val_score(votingClassifier, X_train, y_train, cv=10, scoring="accuracy").mean()


# In[ ]:


predictions=votingClassifier.predict(X_test)


# In[ ]:


submissions=pd.DataFrame(
{
    'PassengerId':passengerID,
    'Survived':predictions
})


# In[ ]:


submissions.head(7)


# In[ ]:


# Writing the submissions to a csv file 
submissions.to_csv("submissions.csv",index=False)

