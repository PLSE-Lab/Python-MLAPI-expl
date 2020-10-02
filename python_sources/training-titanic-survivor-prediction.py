#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Titanic Survivor Prediction

# Load train and test data set from kaggle with pandas

# In[ ]:


import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## 1. Exploratory Data Analysis (EDA)

# In[ ]:


train.head(10)


# In[ ]:


train.describe()


# In[ ]:


train.info()
print("\n============ Missing Values ==============\n")
print(train.isnull().sum())


# ### Visualize data to gain feature insights

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[ ]:


def stacked_bar(feature_name):
    survived = train[train['Survived'] == 1][feature_name].value_counts()
    dead = train[train['Survived'] == 0][feature_name].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))


# In[ ]:


stacked_bar('Sex')


# it seems **_Female_** has higher chance of survive than **_Male_** 

# In[ ]:


stacked_bar('Pclass')


# **_Class1_** Passengers more likely survived than other class<br />
# 
# **_Class2_** Passengers has highest dead count

# In[ ]:


stacked_bar('Embarked')


# People aboarded form **S** more likely dead and from __C__ slightly has chance to survived

# In[ ]:


stacked_bar('Parch')


# People aboarded __**Alone**__ more likely dead

# In[ ]:


stacked_bar('SibSp')


# People whose **siblings or spouse** > 2 more likely to survived

# ### Feature Engineering

# Analyze features then choose some that have strong corelation with target feature or ground truth. Sometimes we might be able to extract new feature<br /> from existing features or combine several ones.<br /><br />
# Transform categorical features into numeric then create **feature vector** which is an n-dimensional vector of features before fitting it into the model.

# #### Extract title feature from names

# In[ ]:


# combining train and test dataset before extract title from name

train_test_data = [train, test] 

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train['Title'].value_counts()


# In[ ]:


test['Title'].value_counts()


# In[ ]:


#Convert categorical value of title into numeric

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


# Drop feature name & ticket from train and test dataset

train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
test.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)


# In[ ]:


#Convert categorical value of sex into numeric

sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


stacked_bar('Embarked')


# #### From **Embarked** bar chart we can see that most of passenger board from S we could fill the missing values with **S**

# In[ ]:


train['Embarked'].fillna('S', inplace = True)
test['Embarked'].fillna('S', inplace = True)


# In[ ]:


#Convert categorical value of Embarked into numeric

embark_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embark_mapping)


# #### Fill Age missing values with median of title categories then convert Age **numerical** into **categorical** feature

# In[ ]:


#fill age missing value with median value for each title

train['Age'].fillna(train.groupby("Title")["Age"].transform("median"), inplace = True)
test['Age'].fillna(test.groupby("Title")["Age"].transform("median"), inplace = True)


# In[ ]:


#Convert Age into Categorical feature using bin

for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 17, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 17) & (dataset['Age'] <= 28), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 40), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'Age'] = 3,
    dataset.loc[dataset['Age'] > 62, 'Age'] = 4,


# #### From **parent/child** and **sibling/spouse** bar chart we could examine people who aboard alone more likely to die.
# #### So we can combine this two features into Familysize feature then scale it to fit the model later

# In[ ]:


#Combine SibSp with Parch into Familysize

train["Familysize"] = train["SibSp"] + train["Parch"] + 1
test["Familysize"] = test["SibSp"] + test["Parch"] + 1


# In[ ]:


#Scale the feature values into 0 to 5 range

familysize_mapping = {0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2, 5: 2.5, 6: 3, 7: 3.5, 8: 4, 9: 4.5, 10: 5, 11: 5.5, 12: 6}
for dataset in train_test_data:
    dataset['Familysize'] = dataset['Familysize'].map(familysize_mapping)


# In[ ]:


#Drop SibSp and Parch from Dataset

train.drop(['SibSp','Parch'], axis=1, inplace=True)
test.drop(['SibSp','Parch'], axis=1, inplace=True)


# #### Fill missing values in Fare with median respected to Pclass category then convert values into categorical to normalize the outlier

# In[ ]:


#fill Fare missing value with median value for each Pclass

train['Fare'].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace = True)
test['Fare'].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace = True)


# In[ ]:


#Binning Fare numerical feature into categorical

for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 20, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 20) & (dataset['Fare'] <= 40), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 40) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 3


# In[ ]:


#Check missing values before modelling 

train.info()
print("\n=============================\n")
test.info()


# #### All features normalized and missing values already filled time to see features correlation

# In[ ]:


correlation_matrix = train.corr().round(2)
plt.figure(figsize = (16,5))
sns.heatmap(data=correlation_matrix, annot=True)


# From the heatmaps we can see that **Fare** has the highest correlation with survivality followed by **Pclass, Sex and Title**

# In[ ]:


#Slice Target feature from training data set

target = train['Survived']
train = train.drop('Survived', axis=1)


# ## 2. Modelling

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn import model_selection

import warnings  
warnings.filterwarnings('ignore')

import numpy as np


# In[ ]:


# Set models array and its parameter

rand_state = 15
models = []
models.append(("Logistic Regression", LogisticRegression(random_state=rand_state)))
models.append(("KNN", KNeighborsClassifier(n_neighbors=rand_state)))
models.append(("Decision Tree", DecisionTreeClassifier(random_state=rand_state)))
models.append(("Random Forest", RandomForestClassifier(random_state=rand_state)))
models.append(("AdaBoost", AdaBoostClassifier(random_state=rand_state)))
models.append(("Gradient Boosting", GradientBoostingClassifier(random_state=rand_state)))
models.append(("XG Boosting", xgb.XGBClassifier()))
models.append(("SVM", SVC(random_state=rand_state)))


# In[ ]:


# Train the data using k-fold cross validation

kfold = model_selection.KFold(n_splits=10)
model_name = []
model_avgscore = []


# In[ ]:


for name, model in models:
    cv_results = model_selection.cross_val_score(model,train,target,scoring="accuracy",cv=kfold)
    print("\n"+name)
    print("Avg_score : "+str(cv_results.mean()))
    model_name.append(name)
    model_avgscore.append(cv_results.mean())


# In[ ]:


# Visualize the result

cv_df = pd.DataFrame({"AverageScore":model_avgscore,"Model":model_name})
sns.barplot("AverageScore","Model",data=cv_df,)


# In[ ]:


best_model = cv_df.sort_values(by="AverageScore",ascending=False).iloc[0]


# In[ ]:


print("Best Model = "+best_model.Model)
print("Average Score = "+str(best_model.AverageScore))


# ## 3. Testing

# From modelling score above we choose **SVM Classifier** to predict the result of the test set

# In[ ]:


svm = SVC()
svm.fit(train,target)


# In[ ]:


test_id = test['PassengerId']
test_data = test.drop('PassengerId', axis=1)
prediction = svm.predict(test_data)


# In[ ]:


submission = pd.DataFrame({"PassengerId":test_id, "Survived":prediction}).set_index("PassengerId")


# In[ ]:


submission.to_csv('submission.csv')


# # References

# * [Minsuk Heo Kaggle - Titanic](https://github.com/minsuk-heo/kaggle-titanic)
# 
# * [Code A Star Titanic Survivors Data Wrangling](https://codeastar.com/data-wrangling)
