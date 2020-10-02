#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas import Series, DataFrame 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', "inline sns.set_style('whitegrid')")
import warnings 
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


X_Test = test


# In[ ]:


y = train.iloc[:, 1]
X = train.iloc[:,:].drop(columns=['Survived'], axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[ ]:


y_train.head(3)


# In[ ]:


y_train.count()


# In[ ]:


y_train_count = y_train.value_counts() 
y_train_count


# In[ ]:


sns.countplot(x='Survived',data=train)


# In[ ]:


X_train.head()


# In[ ]:


X_train['Pclass'].value_counts()


# In[ ]:


import collections, numpy
Count_Unique_Sex = np.unique(X_train['Sex']) 
collections.Counter(Count_Unique_Sex)


# In[ ]:


X_train['Sex'].value_counts()


# In[ ]:


X_train['Parch'].value_counts()


# In[ ]:


X_train['Ticket'].value_counts().head()


# In[ ]:


X_train.isnull().sum()


# In[ ]:


sns.barplot(x=X_train.iloc[:,1], y=y_train)


# In[ ]:


sns.barplot(x=X_train.iloc[:,3], y=y_train)


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'SibSp', bins=20)


# In[ ]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Parch', bins=20)


# In[ ]:


X_train = X_train.drop(columns=['PassengerId', 'Name', 'Ticket'], axis=1) 
X_test = X_test.drop(columns=['PassengerId', 'Name', 'Ticket'], axis=1)


# In[ ]:


X_train.head()


# In[ ]:


g = sns.FacetGrid(train, col='Survived') 
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


corr = train.corr()
corr.style.background_gradient()


# In[ ]:


Cabin = X_train.iloc[:,6].fillna('unknown') 
Cabin_test = X_test.iloc[:,6].fillna('unknown')


# In[ ]:


Cabin.head()


# In[ ]:


X_train['Cabin'] = Cabin 
X_test['Cabin'] = Cabin_test


# In[ ]:


X_train.head()


# In[ ]:


X_train.iloc[:,[2]].head()


# In[ ]:


from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0) 
imp.fit(X_train.iloc[:,2].values.reshape(-1, 1))
Age = imp.transform(X_train.iloc[:,2].values.reshape(-1, 1))


# In[ ]:


imp.fit(X_test.iloc[:,2].values.reshape(-1, 1))
Age_test = imp.transform(X_test.iloc[:,2].values.reshape(-1, 1))


# In[ ]:


X_train['Age'] = Age
X_train['Age'] = X_train['Age'].astype(int)


# In[ ]:


X_test['Age'] = Age_test
X_test['Age'] = X_test['Age'].astype(int)


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
le = LabelEncoder()


# In[ ]:


Pclass_transformed = le.fit_transform(X_train.iloc[:,[0]].astype(str)) 
SibSp_transformed = le.fit_transform(X_train.iloc[:,[3]].astype(str)) 
Parch_transformed = le.fit_transform(X_train.iloc[:,[4]].astype(str))


# In[ ]:


X_train['Pclass'] = Pclass_transformed 
X_train['SibSp'] = SibSp_transformed 
X_train['Parch'] = Parch_transformed


# In[ ]:


PclassTest_transformed = le.fit_transform(X_test.iloc[:,[0]].astype(str)) 
SibSpTest_transformed = le.fit_transform(X_test.iloc[:,[3]].astype(str)) 
ParchTest_transformed = le.fit_transform(X_test.iloc[:,[4]].astype(str))


# In[ ]:


X_test['Pclass'] = PclassTest_transformed 
X_test['SibSp'] = SibSpTest_transformed 
X_test['Parch'] = ParchTest_transformed


# In[ ]:


X_train.iloc[:,6:8].head()


# In[ ]:


Cabin_transformed = le.fit_transform(X_train.iloc[:,[6]].astype(str)) 
Embarked_transformed = le.fit_transform(X_train.iloc[:,[7]].astype(str)) 
Sex_transformed = le.fit_transform(X_train.iloc[:,[1]].astype(str))


# In[ ]:


Cabin_transformedTest = le.fit_transform(X_test.iloc[:,[6]].astype(str)) 
Embarked_transformedTest = le.fit_transform(X_test.iloc[:,[7]].astype(str)) 
Sex_transformedTest = le.fit_transform(X_test.iloc[:,[1]].astype(str))


# In[ ]:


X_train['Cabin'] = Cabin_transformed 
X_train['Embarked'] = Embarked_transformed 
X_train['Sex'] = Sex_transformed


# In[ ]:


X_test['Cabin'] = Cabin_transformedTest 
X_test['Embarked'] = Embarked_transformedTest 
X_test['Sex'] = Sex_transformedTest


# In[ ]:


pd.DataFrame(X_train).head()


# In[ ]:


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()

X_train = sc.fit_transform(X_train) 
pd.DataFrame(X_train).head()


# In[ ]:


X_test = sc.fit_transform(X_test)
pd.DataFrame(X_test).head()


# In[ ]:


from sklearn.linear_model import LogisticRegression 
clf = LogisticRegression().fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test) 


# In[ ]:


clf.score(X_train, y_train)


# In[ ]:


Y_pred_proba = clf.predict_proba(X_test) 
pd.DataFrame(Y_pred_proba).head()


# In[ ]:


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia) 
classNames = ['Negatif','Positif']
plt.title('Matrice de confusion')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames)) 
plt.xticks(tick_marks, classNames, rotation=45) 
plt.yticks(tick_marks, classNames)
s = [['VN','FP'], ['FN', 'VP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j])) 
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score 
accuracy_score(y_test, y_pred)


# In[ ]:


Var_cible = y_test


# In[ ]:


Proba_estime = Y_pred_proba[:,1]


# In[ ]:


Proba_estime = pd.DataFrame(Proba_estime) 
Proba_estime.head()


# In[ ]:


y_test = pd.DataFrame(y_test) 
y_test = y_test.reset_index()


# In[ ]:


df = y_test


# In[ ]:


df['Proba'] = Proba_estime


# In[ ]:


df.head()


# In[ ]:




