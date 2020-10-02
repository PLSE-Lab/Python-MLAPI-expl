#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
sea.set_style('whitegrid')


# In[ ]:


from sklearn.preprocessing import LabelEncoder,PowerTransformer


# In[ ]:


df_train=pd.read_csv("../input/titanic/train.csv")
df_test=pd.read_csv("../input/titanic/test.csv")
ID_test=df_test['PassengerId']


# In[ ]:


df_train.head()


# **Removal Of Outliners**

# In[ ]:


sea.countplot(df_train['SibSp'])


# In[ ]:


df_train['SibSp'].value_counts()


# **7 entries have abnormally high SibSp value**

# In[ ]:


df_train.sort_values(by=['SibSp'],ascending=False).head(10)


# In[ ]:


outliner_SibSp=df_train.loc[df_train['SibSp']==8]
outliner_SibSp


# In[ ]:


df_train=df_train.drop(outliner_SibSp.index,axis=0)


# In[ ]:


df_train.loc[df_train['SibSp']==8]


# In[ ]:


sea.boxplot(df_train['Fare'],orient='v')


# In[ ]:


df_train.sort_values(by=['Fare','Pclass'],ascending=False).head(10)


# ** 3 entries have very high Fare **

# In[ ]:


outliner_Fare=df_train.loc[df_train['Fare']>500]
outliner_Fare


# In[ ]:


df_train=df_train.drop(outliner_Fare.index,axis=0)


# ** After removal of outliners now we can merge the train and test dataset to fill the missing values**

# In[ ]:


df_train.shape


# In[ ]:


df_test.shape


# In[ ]:


dataset=pd.concat([df_train,df_test],ignore_index=True)


# In[ ]:


dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


dataset=dataset.fillna(np.nan)
dataset.isnull().sum()


# ** 2 Entries of Emabarked Column have Null Value **

# In[ ]:


dataset.loc[dataset['Embarked'].isnull()]


# In[ ]:


sea.countplot(dataset['Embarked'])


# ** Fill the missing Embarked Value with the most frequent one i.e 'S' **

# In[ ]:


dataset['Embarked']=dataset['Embarked'].fillna('S')


# In[ ]:


dataset.loc[dataset['Fare'].isnull()]


# In[ ]:


dataset.loc[(dataset['Pclass']==3)].sort_values(by=['Fare'],ascending=False).head(15)


# ** Fare have no relation with age,sex but varies with number of passengers with same same ticket number **

# ** Get data of Passengers with Pclass=3 having 0 Parch and 0 SibSp, simliar to the requirement(fare null value) **

# In[ ]:


temp=dataset.loc[(dataset['Pclass']==3) & (dataset['Parch']==0) & (dataset['SibSp']==0) & (dataset['Fare']>0)].sort_values(by=['Fare'],ascending=False)
temp.head()


# ** Replace Fare null value with mean value of above subset **

# In[ ]:


dataset['Fare']=dataset['Fare'].fillna(temp['Fare'].mean())


# ** Now lets analyze the available "Age" data**

# In[ ]:


g= sea.FacetGrid(df_train,col='Survived')
g= g.map(sea.distplot,'Age')


# ** Age is the not very much determining factor for Survival prediction.**
# 
# ** But it seems that passengers with young age have more chance of survival **
# 
# ** Age data is also skewed so need logarithmic transformation **
# 

# In[ ]:


nullAgeSubset=dataset.loc[dataset['Age'].isnull()]
nullAgeSubset.shape


# ** Replace the missing values of Age column with entries with similar other parameters Else replace with mean age of dataset **

# In[ ]:


for index in nullAgeSubset.index:
    ageSubsetMean=dataset['Age'].loc[(dataset['Parch']==nullAgeSubset.loc[index]['Parch'])&(dataset['SibSp']==nullAgeSubset.loc[index]['SibSp'])&(dataset['Pclass']==nullAgeSubset.loc[index]['Pclass'])&(dataset['Embarked']==nullAgeSubset.loc[index]['Embarked'])&(dataset['Sex']==nullAgeSubset.loc[index]['Sex'])].mean()
    if(ageSubsetMean>0):
        dataset['Age'].loc[index]=ageSubsetMean
    else:
        ageSubsetMean=dataset['Age'].loc[(dataset['Parch']==nullAgeSubset.loc[index]['Parch'])&(dataset['SibSp']==nullAgeSubset.loc[index]['SibSp'])&(dataset['Pclass']==nullAgeSubset.loc[index]['Pclass'])&(dataset['Embarked']==nullAgeSubset.loc[index]['Embarked'])].mean()
        if(ageSubsetMean>0):
            dataset['Age'].loc[index]=ageSubsetMean
        else:
            ageSubsetMean=dataset['Age'].loc[(dataset['Parch']==nullAgeSubset.loc[index]['Parch'])&(dataset['SibSp']==nullAgeSubset.loc[index]['SibSp'])&(dataset['Pclass']==nullAgeSubset.loc[index]['Pclass'])].mean()
            if(ageSubsetMean>0):
                dataset['Age'].loc[index]=ageSubsetMean
            else:
                dataset['Age'].loc[index]=dataset['Age'].mean()
                


# ** Check if any remaining null value for age **

# In[ ]:


dataset['Age'].isnull().sum()


# In[ ]:


sea.heatmap(df_train.corr(),cmap='BrBG',annot=True)


# In[ ]:


sea.countplot(dataset['Sex'],hue=dataset['Survived'])


# In[ ]:


sea.catplot(data=dataset,x='Pclass',y='Survived',kind='bar')


# ** First class people have more count of survival **

# In[ ]:


g=sea.FacetGrid(data=dataset.loc[dataset['Survived']==1],col='Pclass')
g=g.map(sea.countplot,'Sex')


# **Above analysis shows that survival count of female is more the male irrespective of class **

# In[ ]:


dataset.head()


# Lets see distribution of Fare

# In[ ]:


sea.distplot(np.array(dataset['Fare']).reshape(-1,1),axlabel='Fare')


# Its hightly skewed so we will apply normal distribution to it.

# In[ ]:


sea.distplot(PowerTransformer().fit_transform(np.array(dataset['Fare']).reshape(-1,1)),axlabel='Fare')


# Now it looks normalized.

# Similarly for Age column

# In[ ]:


sea.distplot(np.array(dataset['Age']).reshape(-1,1),axlabel='Age')


# In[ ]:


sea.distplot(PowerTransformer().fit_transform(np.array(dataset['Age']).reshape(-1,1)),axlabel='Age')


# Similarly for SibSp column

# In[ ]:


sea.distplot(np.array(dataset['SibSp']).reshape(-1,1),axlabel='SibSp')


# In[ ]:


sea.distplot(PowerTransformer().fit_transform(np.array(dataset['SibSp']).reshape(-1,1)),axlabel='SibSp')


# Similarly for Parch column

# In[ ]:


sea.distplot(np.array(dataset['Parch']).reshape(-1,1),axlabel='Parch')


# In[ ]:


sea.distplot(PowerTransformer().fit_transform(np.array(dataset['Parch']).reshape(-1,1)),axlabel='Parch')


# In[ ]:


X=dataset.drop(['Cabin','Name','PassengerId','Survived','Ticket'],axis=1)
Y=dataset['Survived']


# In[ ]:


X.head(10)


# **Finally do all the transformation needed**
# 
# Normally ditribute the Age,Fare,Parch,SibSp
# 
# Label Encode the Sex
# 
# Create dummy columns for values for Embarked and Pclass

# In[ ]:


X['Age']=PowerTransformer().fit_transform(np.array(X['Age']).reshape(-1,1))
X['Fare']=PowerTransformer().fit_transform(np.array(X['Fare']).reshape(-1,1))
X['Parch']=PowerTransformer().fit_transform(np.array(X['Parch']).reshape(-1,1))
X['Sex']=LabelEncoder().fit_transform(X['Sex'])
X['SibSp']=PowerTransformer().fit_transform(np.array(X['SibSp']).reshape(-1,1))
dummyPclass=pd.get_dummies(X['Pclass'],prefix='Pclass')
dummyEmbarked=pd.get_dummies(X['Embarked'],prefix='Embarked')
X=pd.concat([X.drop(['Pclass','Embarked'],axis=1),dummyPclass,dummyEmbarked],axis=1)


# In[ ]:


X.head(15)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier


# In[ ]:


X_pro=PolynomialFeatures(degree=2).fit_transform(X)


# In[ ]:


trainDataX=X_pro[:df_train.shape[0]]
trainDataY=Y[:df_train.shape[0]].astype('int32')
testDataX=X_pro[df_train.shape[0]:]


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(trainDataX,trainDataY,test_size=0.1,random_state=47)


# In[ ]:


model=XGBClassifier(learning_rate=0.001,n_estimators=300,max_depth=30)
#model=SVC(kernel='poly',C=100,gamma=0.1)
model.fit(X_train,Y_train)
accuracy_score(Y_train,model.predict(X_train))


# In[ ]:


accuracy_score(Y_test,model.predict(X_test))


# In[ ]:


submission=pd.DataFrame(columns=['PassengerId','Survived'])
submission['PassengerId']=ID_test
submission['Survived']=model.predict(testDataX)


# In[ ]:


submission.head()


# In[ ]:


filename='submission.csv'
submission.to_csv(filename,index=False)
from IPython.display import FileLink
FileLink(filename)


# In[ ]:




