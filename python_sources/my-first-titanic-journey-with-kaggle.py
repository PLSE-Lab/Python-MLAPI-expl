#!/usr/bin/env python
# coding: utf-8

# Hope you are clear with problem statement and all features.So we have to predict whetaher person will survive or not using our model.We list out steps for our train and test dataset.
# Import useful libraries
# Check data,check missing values

# In[ ]:



import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()


# In[ ]:


dft=pd.read_csv('/kaggle/input/titanic/test.csv')
dft.head()


# In[ ]:


print(df.shape)
print(dft.shape)


# In[ ]:


df.describe()


# In[ ]:


dft.describe()


# In[ ]:


df.info()


# In[ ]:


dft.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


dft.isnull().sum()


# In[ ]:


#visualise missing value 
import missingno as mn
mn.matrix(df)
mn.matrix(dft)


# Since only age and cabin feature has missing values with more percentage.Due to percentage is more we can not drop missing value.We may loose some important data.We will drop Cabin row instead of 687 rows.

# In[ ]:


df = df.drop(['Cabin'], axis = 1)


# In[ ]:


dft = dft.drop(['Cabin'], axis = 1)


# In[ ]:


df['Embarked'].value_counts()


# In[ ]:


#Embarked has 2 missing value fill with S which has highest number
df['Embarked'].fillna('S',inplace=True)


# In[ ]:


dft['Embarked'].fillna('S',inplace=True)


# In[ ]:


#replace NaN value in Age with mean value
median=np.round(df['Age'].median(),1)
df['Age'].fillna(median,inplace=True)


# In[ ]:


median=np.round(dft['Age'].median(),1)
dft['Age'].fillna(median,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


dft.isnull().sum()


# In[ ]:


#replace Sex column with numeric 0,1 with male,female rep
df=df.replace({'male': 0,
            'female' : 1})
df.head()


# In[ ]:


dft=dft.replace({'male': 0,
            'female' : 1})
dft.head()


# # EDA
# Till now we were dealing with missing values.It is time for visualising data.Lets do it

# In[ ]:


#find the correlation between data
df.corr()


# We conclude that as age increased survived rate is decreasing because of negative correlation and as fare increases survive rate increases due to positive value

# In[ ]:


#visualise correlation data using heatmap
plt.figure(figsize=(14,6))
sns.heatmap(df.corr(),annot=True)


# Now visualise each data with survived rate

# In[ ]:


#plot graph between pclass and survived
sns.barplot(x='Pclass',y='Survived',data=df)


# So higher class people had higher survived rate

# In[ ]:


#Sex vs Survived
sns.barplot(x='Sex',y='Survived',data=df)


# Female had much more survived rate than male

# In[ ]:


#Embarked vs Survived
sns.barplot(x='Embarked',y='Survived',data=df)


# # Feature Engineering

# In[ ]:


#Drop unwanted columns such as Name,Ticket,Fare is decided by Pclass so drop fare also
df=df.drop(['Name','Ticket','Fare'],axis=1)


# In[ ]:


dft=dft.drop(['Name','Ticket','Fare'],axis=1)


# In[ ]:


#add SibSp and Parch in Family
df['Family']=df['SibSp']+df['Parch']+1
df=df.drop(['SibSp','Parch'],axis=1)


# In[ ]:


dft['Family']=dft['SibSp']+dft['Parch']+1
dft=dft.drop(['SibSp','Parch'],axis=1)


# We will deal with categorical output so convert numerical variable into categorical i.e Age,Family.

# In[ ]:


#Categorise Age
def AgeGroup(age):
    a=''
    if age<=10:
        a='Child'
    elif age<=30:
        a='Young'
    elif age<=50:
        a='Adult'
    else:
        a='Old'
    return a
df['AgeGroup']=df['Age'].map(AgeGroup)
df=df.drop(['Age'],axis=1)


# In[ ]:


dft['AgeGroup']=dft['Age'].map(AgeGroup)
dft=dft.drop(['Age'],axis=1)


# In[ ]:


#Categorise Family
def FamilyGroup(family):
    a=''
    if family<=1:
        a='Solo'
    elif family<=4:
        a='Small'
    else:
        a='Large'
    return a
df['FamilyGroup']=df['Family'].map(FamilyGroup)
df=df.drop(['Family'],axis=1)    


# In[ ]:


dft['FamilyGroup']=dft['Family'].map(FamilyGroup)
dft=dft.drop(['Family'],axis=1)


# In[ ]:


#get dummies variable
df=pd.get_dummies(df,columns=['Embarked','AgeGroup','FamilyGroup','Sex'])


# In[ ]:


dft=pd.get_dummies(dft,columns=['Embarked','AgeGroup','FamilyGroup','Sex'])


# In[ ]:


print(df.shape)
print(dft.shape)


# In[ ]:


df.head()


# In[ ]:


dft.head()


# # Model Creation and Evaluation

# We will use Logistic regression,SVM,KNN,and desicion tree.

# In[ ]:


X=df.drop(['Survived'],axis=1)
X.head()
y=df['Survived']
y.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)


# In[ ]:


#import all lib
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[ ]:


#KNN CLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks = 15
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[ ]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[ ]:


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X_train,y_train)
y_pred=LR.predict(X_test)
print("The best accuracy with LR is", metrics.accuracy_score(y_test,y_pred))


# In[ ]:


#SVM
from sklearn import svm
SVM=svm.SVC().fit(X_train,y_train)
y_pred=SVM.predict(X_test)
print("The best accuracy with SVM is", metrics.accuracy_score(y_test,y_pred))


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
def getaccuracy(max_leaf,X_train,y_train,X_test,y_test):
    DT=DecisionTreeClassifier().fit(X_train,y_train)
    y_pred=DT.predict(X_test)
    return(metrics.accuracy_score(y_test,y_pred))
    


# In[ ]:


for max_leaf in [5,50,500]:
    my_mae = getaccuracy(max_leaf,X_train,y_train,X_test,y_test)
    print("Max leaf : ",max_leaf,'The best accuracy with SVM is',my_mae)


# Take 50 as max leaf node.

# # Prediction
# So best classifier is SVM with high accuracy.

# In[ ]:


#define whole train as TrainX and Trainy
TrainX=df.drop(['Survived'],axis=1)
Trainy=df['Survived']


# In[ ]:


from sklearn import svm
SVM=svm.SVC().fit(TrainX,Trainy)
y_pred=SVM.predict(dft)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": dft["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('titanic.csv', index=False)
print("Submitted Successfully")


# In[ ]:




