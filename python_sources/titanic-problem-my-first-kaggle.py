# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

train_file="../input/train.csv"

df_in=pd.read_csv(train_file)

#Gives info on columns like count, data type etc.. can be used to decide which may not be good features
# df.info()

#Checking target variable - count of yes/no
# print(df['Survived'].value_counts())

def data_imputer(df,column,strat):
    imp=Imputer(missing_values="NaN",axis=1,strategy=strat)
    imp.fit([df[column]])
    df[column]=imp.transform([df[column]])[0]

def EDA(df):
    #Let us remove Passenger Id and Name -- as they do not really add insight
    #Let us remove Cabin as it has around 650 null records; at this point I think it does not add value

    # df = df.drop(labels=['PassengerId','Name','Cabin','Ticket'],axis=1)
    df = df.drop(labels=['Cabin','Name','Ticket'],axis=1)
    #Identifying/Describing type of variable - continous or categorical

    #Survived is target
    #Pclass, Sex, SibSp,Parch, Ebarked are categorical
    #Age and Fare are continous
    
    cat_feat_list = ['Pclass','Sex','SibSp','Parch','Embarked']
    cont_feat_list=['Age','Fare']
    
    ### Converting string features to numericals as RandomForest sklearn takes in only floats/ints
    df.loc[df['Sex']=='male','Sex']=1
    df.loc[df['Sex']=='female','Sex']=0
    df.loc[df['Embarked']=='S','Embarked']=1
    df.loc[df['Embarked']=='C','Embarked']=2
    df.loc[df['Embarked']=='Q','Embarked']=3
    
    ### Ticket is alphanumeric ; with aplha being ticket type (I guess).. it may have played a part..let us explore
    
    #Univariate analysis on continous variable
    
    # print(df[cont_feat_list].describe())
    
    #Notice that Age has some missing values
    # Imputing Missing values in Age using sci-kit imputer
    
    data_imputer(df,'Age','mean')
    
    # imp=Imputer(missing_values="NaN",axis=1)
    # imp.fit([df['Age']])
    # df['Age']=imp.transform([df['Age']])[0]
    
    #Notice that Age and Fare have outliers ; (mean+2or3* Std Dev) is way lower than max value ; points to outliers
    
    #Histogram and Box plotting - Will give better view into outliers
    
    # plt.figure()
    # df['Age'].plot.hist()
    # plt.xlabel('AGE')
    # plt.show()
    # df['Age'].plot.box()
    # plt.xlabel('AGE')
    # plt.show()
    
    # plt.figure()
    # df['Fare'].plot.hist()
    # plt.xlabel('FARE')
    # plt.show()
    # df['Fare'].plot.box()
    # plt.xlabel('FARE')
    # plt.show()
    
    # Fare may not have an impact on Survival ; Bivariate analysis should throw more light.
    
    #Univariate Analysis of Categorical variables
    # plt.figure()
    
    # Pclass_CT=pd.crosstab(index=df["Pclass"],columns="count")
    # print(Pclass_CT)
    # Pclass_CT.plot.bar()
    # plt.show()
    
    # Sex_CT=pd.crosstab(index=df["Sex"],columns="count")
    # print(Sex_CT)
    # Sex_CT.plot.bar()
    # plt.show()
    
    # SibSp_CT=pd.crosstab(index=df["SibSp"],columns="count")
    # print(SibSp_CT)
    # SibSp_CT.plot.bar()
    # plt.show()
    
    # Parch_CT=pd.crosstab(index=df["Parch"],columns="count")
    # print(Parch_CT)
    # Parch_CT.plot.bar()
    # plt.show()
    
    # Embarked_CT=pd.crosstab(index=df["Embarked"],columns="count")
    # print(Embarked_CT)
    # Embarked_CT.plot.bar()
    # plt.show()
    
    ## Embarked contain some missing values
    data_imputer(df,'Embarked','most_frequent')
    data_imputer(df,'Fare','mean')
    return df


df_in=EDA(df_in)

#Fitting a Random Forest
df_target = df_in["Survived"]
df_predict=df_in.drop('Survived',axis=1)
train_x, test_x, train_y, test_y = train_test_split(df_predict, df_target,test_size=0.33, random_state=42)

clf = RandomForestClassifier()
clf.fit(train_x,train_y)
# print("Features sorted by their score:")
# print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), df_predict.columns.values), reverse=True))
predictions = clf.predict(test_x)

# print(f1_score(test_y, predictions))


test_file="../input/test.csv"

df_test=pd.read_csv(test_file)
df_test_final=EDA(df_test)
pred_out = clf.predict(df_test_final)

df_out = pd.DataFrame(pred_out,columns=['Survived'])
df_out['PassengerId']=df_test['PassengerId']
cols=['PassengerId','Survived']
df_out=df_out[cols]
df_out.head()

out_file="../output/submission.csv"
df_out.to_csv('submission.csv', index=False)