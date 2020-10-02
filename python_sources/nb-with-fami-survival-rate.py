from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import re

#data_cleaning
df_test=pd.read_csv('../input/test.csv')
df_train=pd.read_csv('../input/train.csv')
df_test['istest']=1
df_test['Survived']=0
df_train['istest']=0
df=pd.concat([df_train,df_test])
df['Sex']=df['Sex'].map({'female':1,'male':2})
df['Embarked']=df['Embarked'].map({'S':3,'C':1,'Q':2})
df['Name']=df['Name'].apply(lambda x:re.search('(.*),(.*)',x).group(1))
k=0
name_dict={}
for name in df[['Name','Sex']].groupby('Name').count().index.values.tolist():
    name_dict[name]=k
    k+=1
df['Name']=df['Name'].map(name_dict)
for [sibsp,parch] in df.groupby(['SibSp','Parch'])['Age'].mean().index.values.tolist():
    avg_age=df[(df.SibSp==sibsp)&(df.Parch==parch)]['Age'].mean()
    if avg_age!=avg_age:
        if parch>4:
            avg_age=df[df.Parch>4]['Age'].mean()
    df['Age']=df[['SibSp','Parch','Age']].apply(lambda x:avg_age if (x[0]==sibsp and x[1]==parch and x[2] != x[2]) else x[2],axis=1)
for pclass in df.groupby('Pclass')['Fare'].mean().index.values.tolist():
    avg_fare=df[df.Pclass==pclass]['Fare'].mean()
    df['Fare']=df[['Pclass','Fare']].apply(lambda x:avg_fare if (x[0]==pclass and x[1]!=x[1]) else x[1],axis=1)
avg_embarked=round(df['Embarked'].mean())
df['Embarked']=df['Embarked'].apply(lambda x:avg_embarked if (x!=x) else x)
df.loc[(df.Age>=0)&(df.Age<18),'Age']=1
df.loc[(df.Age>=18)&(df.Age<48),'Age']=2
df.loc[(df.Age>=48)&(df.Age<64),'Age']=3
df.loc[(df.Age>=64),'Age']=4
df.loc[(df.Fare>=0)&(df.Fare<10),'Fare']=1
df.loc[(df.Fare>=10)&(df.Fare<50),'Fare']=2
df.loc[df.Fare>=50,'Fare']=3
df['family_size']=df['SibSp']+df['Parch']+1
df.loc[(df.family_size>0)&(df.family_size<2),'family_size']=1
df.loc[(df.family_size>=2)&(df.family_size<5),'family_size']=2
df.loc[df.family_size>=5,'family_size']=3
df_famihood=df[['Survived','Name']].groupby('Name').mean()
df_famihood.columns=['famihood']
df_famihood.fillna(0.5,inplace=True)
df_famihood.loc[df_famihood.famihood<0.25,'famihood']=0
df_famihood.loc[(df_famihood.famihood>=0.25)&(df_famihood.famihood<=0.334),'famihood']=1
df_famihood.loc[(df_famihood.famihood>0.334)&(df_famihood.famihood<0.666),'famihood']=2
df_famihood.loc[(df_famihood.famihood>=0.666)&(df_famihood.famihood<=0.75),'famihood']=3
df_famihood.loc[(df_famihood.famihood>=0.75)&(df_famihood.famihood<=1),'famihood']=4
df_famihood.reset_index(inplace=True)
df=pd.merge(df,df_famihood,on='Name',how='left')

df_Pclass=df[['Pclass','Survived']].groupby('Pclass').mean()
df_Sex=df[['Sex','Survived']].groupby('Sex').mean()
df_Age=df[['Age','Survived']].groupby('Age').mean()
df_Fare=df[['Fare','Survived']].groupby('Fare').mean()
df_Embarked=df[['Embarked','Survived']].groupby('Embarked').mean()
df_familysize=df[['family_size','Survived']].groupby('family_size').mean()
df_famihoods=df[['famihood','Survived']].groupby('famihood').mean()

#native_bayes classifier training
def nb_clf(df):
    from sklearn.naive_bayes import GaussianNB
    clf=GaussianNB()
    df_test=df[df.istest==0]
    x_train=df_test.loc[0:799,['Pclass','Sex','Age','Fare','Embarked','family_size','famihood']]
    x_test=df_test.loc[800:890,['Pclass','Sex','Age','Fare','Embarked','family_size','famihood']]
    y_train=df_test.loc[0:799,['Survived']]
    y_test=df_test.loc[800:890,['Survived']]
    clf.fit(x_train,y_train)
    y_test['pred']=clf.predict(x_test)
    y_test['diff']=y_test[['Survived','pred']].apply(lambda x:1 if x[0]==x[1] else 0,axis=1)
    accuracy=y_test['diff'].sum()/y_test['diff'].count()
    print(nb_clf(df))
    return accuracy

#output_test_result
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
x_train=df.loc[0:890,['Pclass','Sex','Age','Fare','Embarked','family_size','famihood']]
x_test=df.loc[891:1308,['Pclass','Sex','Age','Fare','Embarked','family_size','famihood']]
y_train=df.loc[0:890,['Survived']]
y_test=df.loc[891:1308,['PassengerId']]
clf.fit(x_train,y_train)
y_test['Survived']=clf.predict(x_test)
gender_submission=y_test.groupby('PassengerId').sum()
gender_submission.to_csv('gender_submission.csv',index_label='PassengerId')


