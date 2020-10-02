#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


titanic_train= pd.read_csv("/kaggle/input/titanic/train.csv")
titanic_train.head(10)


# In[ ]:


titanic_test = pd.read_csv("/kaggle/input/titanic/test.csv")
titanic_test.head()


# In[ ]:


titanic_train.shape


# 891 rows/passengers and 12 columns/data points in the titanic data set

# In[ ]:


titanic_train.describe()


# oldest passenger = 80 years old, youngest passenger = 0.42 (5 months old)

# In[ ]:


titanic_train.Survived.value_counts()


# 549 dead, 342 alive passengers

# In[ ]:


survived = titanic_train.Survived
sns.countplot(survived)
plt.show()


# In[ ]:


sex_women = titanic_train.loc[titanic_train.Sex == 'female']["Survived"]
alive_women = sum(sex_women)
num_women = len(sex_women)
print(f'number of alive women: {alive_women}')
print(f'total number of women: {num_women}')


# In[ ]:


sex_men = titanic_train.loc[titanic_train.Sex == 'male']["Survived"]
alive_men = sum(sex_men)
num_men = len(sex_men)
print(f'number of alive men: {alive_men}')
print(f'total number of men: {num_men}')


# In[ ]:


rate_survived_men = alive_men / num_men
rate_survived_women = alive_women / num_women
print(f'rate of survived men: {rate_survived_men}')
print(f'rate of survived women: {rate_survived_women}')


# This means sex of the passangers plays an important rule in their survival

# In[ ]:


class_alive_men = titanic_train.loc[(titanic_train.Sex == 'male') & (titanic_train.Survived == 1)]['Pclass']
class_dead_men = titanic_train.loc[(titanic_train.Sex == 'male') & (titanic_train.Survived == 0)]['Pclass']
sns.countplot(class_alive_men, label = 'alive men')
plt.title('alive men')
plt.show()
sns.countplot(class_dead_men, label = 'dead men')
plt.title('dead men')
plt.show()


# survived men where mostly from pclass 1 and 3. dead men where mostly from class 3.

# In[ ]:


class_alive_women = titanic_train.loc[(titanic_train.Sex == 'female') & (titanic_train.Survived == 1)]['Pclass']
class_dead_women = titanic_train.loc[(titanic_train.Sex == 'female') & (titanic_train.Survived == 0)]['Pclass']
sns.countplot(class_alive_women)
plt.title('alive women')
plt.show()
sns.countplot(class_dead_women)
plt.title('dead women')
plt.show()


# dead women were mostly from class 3

# In[ ]:


age = pd.cut(titanic_train['Age'], [0, 18, 80])
titanic_train.pivot_table('Survived', ['Sex', age], 'Pclass')


# In[ ]:


titanic_train.isna().sum()


# In[ ]:


y = titanic_train["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(titanic_train[features])
X_testt = pd.get_dummies(titanic_test[features])


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=1)
model = neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='weighted')


# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score( y_test, y_pred))
print(np.mean(y_test==y_pred))


# In[ ]:


predictions = neigh.predict(X_testt)


# In[ ]:


output = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

