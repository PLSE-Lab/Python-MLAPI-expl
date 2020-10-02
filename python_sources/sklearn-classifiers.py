#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Techniques to try: SVM and 

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sea

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Explore Data

# In[ ]:


train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

print(train_df.head())


sl_df=train_df[ train_df["Survived"]==1 ]
ns_df=train_df[ train_df["Survived"]==1 ]

surviving_males = sl_df[ sl_df["Sex"]=="male"]
surviving_females = sl_df[ sl_df["Sex"]=="female"]


numF= len(surviving_females)
numM=len(surviving_males)

print("Percent of survivers that are males: {}%".format( round( ( numM/(numF+numM) )*100,0) ) )
print("Percent of survivers that are females: {}%".format( round( (numF/(numF+numM))*100,0) ) )

print("Number of unique Embarked: {}".format( len(train_df["Embarked"].unique()) ))


print(list(train_df["Embarked"].unique()))


# # functions

# In[ ]:


#prepare data

def dprepv1():
    train_df=pd.read_csv("/kaggle/input/titanic/train.csv")    
    
    train_df = train_df[ ["Survived","Pclass","Sex","Age","Parch","Fare","SibSp","Embarked"] ]
    train_df=train_df.fillna(9)

    y = list(train_df["Survived"].values)

    train_df=train_df.drop( ["Survived"],axis=1 )                      

    Xn=list(train_df.values)

    X=[]
    for el in Xn:
        X.append(list(el))

        
    dep=list(train_df["Embarked"].unique())
    #change sex
    for i,el in enumerate(X):
        if el[1]=="male":
            X[i][1]=1
        if el[1]=="female":
            X[i][1]=1
        if el[6] in dep:
            X[i][6]=dep.index(el[6])

    
    X_train, X_test, y_train,y_test = X[:-150],X[-150:],y[:-150],y[-150:]

    return X_train, X_test, y_train,y_test

def dprepv1_test():
    
    train_df=pd.read_csv("/kaggle/input/titanic/test.csv")
        
    
    train_df = train_df[ ["PassengerId","Pclass","Sex","Age","Parch","Fare","SibSp","Embarked"] ]
    train_df=train_df.fillna(9)

    y = list(train_df["PassengerId"].values)

    train_df=train_df.drop( ["PassengerId"],axis=1 )                      

    Xn=list(train_df.values)

    X=[]
    for el in Xn:
        X.append(list(el))

        
    dep=list(train_df["Embarked"].unique())
    #change sex
    for i,el in enumerate(X):
        if el[1]=="male":
            X[i][1]=1
        if el[1]=="female":
            X[i][1]=1
        if el[6] in dep:
            X[i][6]=dep.index(el[6])


    return X,y



# # SVM

# In[ ]:


from sklearn import svm, datasets

X_train, X_test, y_train,y_test = dprepv1()


### linear

svc=svm.SVC(kernel="linear",C=1.0)
svc.fit(X_train,y_train)


out=svc.predict(X_test)

out
y_test

correct=0
for i,el in enumerate(out):
    if el==y_test[i]:
        correct+=1
        
print("correct: {}".format( round( (correct/float(len(out)))*100,0 ) ) )


# # Decision Tree

# In[ ]:


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train,y_test=dprepv1()

#clf = tree.DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators=20)

clf.fit(X_train,y_train)

out=clf.predict(X_test)

out
y_test

correct=0
for i,el in enumerate(out):
    if el==y_test[i]:
        correct+=1

print("correct: {}".format( round( (correct/float(len(out)))*100,0 ) ) )


# # Logistical Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train,y_test=dprepv1()

lg=LogisticRegression()

lg.fit(X_train,y_train)

out=lg.predict(X_test)

out
y_test

correct=0
for i,el in enumerate(out):
    if el==y_test[i]:
        correct+=1

print("correct: {}".format( round( (correct/float(len(out)))*100,0 ) ) )


# In[ ]:


X,y=dprepv1_test()

def create_submission(X,y):
    
    out=svc.predict(X)
    S=[]
    for i,el in enumerate(out):
        S.append([y[i],el])
    
    df=pd.DataFrame(data=S,columns=["PassengerId","Survived"])
    
    
    print( len(df) )
    
    print(df.head())
    
    df.to_csv("submission.csv",index=False)
    
    
    
create_submission( *dprepv1_test() )

    


# In[ ]:


from IPython.display import FileLink
FileLink(r'submission.csv')

