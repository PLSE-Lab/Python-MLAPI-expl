#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# ### Importing required Libraries & Dataset

# In[ ]:


import pandas as pd , numpy as np , matplotlib.pyplot as plt , seaborn as sns , warnings
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings('ignore')

df = pd.read_csv('../input/titanic/train.csv')
df_test1 = pd.read_csv('../input/titanic/test.csv')

df.head()


# ### Getting Deeper into Data

# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.index


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## Exploratory Data Analysis ( EDA )

# In[ ]:


sns.countplot(x='Survived',hue='Sex',data=df)


# In[ ]:


sns.countplot(x='Survived',hue='Pclass',data=df)


# In[ ]:


sns.countplot(df.Survived)


# In[ ]:


plt.hist(df.Age)


# In[ ]:


sns.boxplot(x='Pclass',y='Age',data=df)


# ### Dealing with Missing Values ( i.e Null Values )

# In[ ]:


df.isnull().sum()


# In[ ]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37
        elif Pclass ==2:
            return 29
        elif Pclass == 3:
            return 24
    else:
        return Age
    
df.Age=df[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


df = df.drop(columns = ['Ticket','Name','Embarked','PassengerId','Cabin'])

dfn=df._get_numeric_data()
nc=list(dfn)
dfc=df.drop(columns=nc)
cc=list(dfc)

dfc=pd.get_dummies(dfc,drop_first=True)

data=pd.concat([dfn,dfc],axis=1)
data.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


data.isnull().sum()


# ## Feature Engineering & Feature Selection

# In[ ]:


sns.heatmap(data.corr(),cmap='viridis',annot = True)


# In[ ]:


sns.pairplot(data)


# In[ ]:


c= data.corr()    # Finding correlation

i = 0

# replacing diogonal corr() which is 1 to NaN for finding
#i.e manulating and get the informative features by removing high values of correlation.

while True:    
    try:
        c.iloc[i,i] = np.nan
        i += 1
    except:
        break


# In[ ]:


data.Survived.value_counts()


# In[ ]:


# Getting high corr. values w.r.t output because it supports the output...

features = c[(c['Survived'] > 0.1) | (c['Survived'] < -0.1)].dropna(how = 'all')['Survived']     
features_col = list(features.index)
print(features.shape, len(features_col) , features_col)     # Exactly what i want....


# In[ ]:


sns.heatmap(data[features_col].corr(),cmap='viridis',annot = True)


# In[ ]:


sns.pairplot(data[features_col])


# # Comparing Models

# In[ ]:


# Preparing  pipeline for all the models
# Here DecisionTreeClassifier wins the race...

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report


X = data[features_col]
y = data.Survived
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)



models = [LogisticRegression(penalty='l2'),DecisionTreeClassifier()
          ,RandomForestClassifier(),KNeighborsClassifier(),SVC()]

for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test,y_pred),type(model).__name__)


# ## Here DecisionTreeClassifier Clearly Wins the Race

# # Building Final Model

# In[ ]:


X = data[features_col]
y = data.Survived
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)

model = DecisionTreeClassifier()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred),type(model).__name__)


# In[ ]:


print(confusion_matrix(y_test,y_pred)) 


# In[ ]:


print(classification_report(y_test,y_pred))


# # Getting Predictions Ready For Kaggle Competition

# In[ ]:


X.columns


# In[ ]:


df_test = df_test1[['Pclass', 'Fare', 'Sex']]
df_test.isnull().sum()


# In[ ]:


df_test[df_test['Fare'] != df_test['Fare']]


# In[ ]:


df_test[df_test['Pclass'] == 3]['Fare'].mean()


# In[ ]:


df_test = df_test.fillna(12.46)


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_test = pd.get_dummies(df_test,drop_first = True)


# # Final Model

# In[ ]:


X_train = data[['Pclass', 'Fare', 'Sex_male']]
X_test = df_test
y_train = data.Survived

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[ ]:


predictions = pd.concat([df_test1[['PassengerId']],pd.DataFrame(y_pred,columns = ['Survived'])],axis=1)


# In[ ]:


predictions.shape


# In[ ]:


predictions


# # ...END...
