#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook is built to analyze and identify the factors that helped people to survive the horrible shipwreck.
# 
# We also would use ML techniques to predict if a person would survive or not!!
# 
# This notebook is inspired from [Titanic best working Classifier](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier/data) and [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
# 
#  

# # Table of contents
# - [Data Information](#first-bullet)
# - [Importing the packages needed](#second-bullet)
# - [Load and prepare the data](#third-bullet)
# - [Analyzing NaN/Null values](#fourth-bullet)
# - [Analyzing the columns](#fifth-bullet)
# - [Data Cleaning](#sixth-bullet)
# - [One hot Encoding](#seventh-bullet)
# - [Splitting into train and test features](#eighth-bullet)
# - [Neural Network Architecture](#ninth-bullet)
# - [Evaluating the model](#tenth-bullet)
# - [Predicting the model](#eleventh-bullet)
# - [Other Classification techniques](#twelfth-bullet)
# - [Submission](#thirteenth-bullet)

# ## Data Infromation <a class="anchor" id="first-bullet"></a>
# - Survival	0 = No, 1 = Yes
# - pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# - sex	Male/Female	
# - Age	Age in years	
# - sibsp	# of siblings / spouses aboard the Titanic	
# - parch	# of parents / children aboard the Titanic	
# - ticket	Ticket number	
# - fare	Passenger fare	
# - cabin	Cabin number	
# - embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# 

# ## Importing the packages needed <a class="anchor" id="second-bullet"></a>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
from math import pi


# ## Load and prepare the data <a class="anchor" id="third-bullet"></a>
# 

# In[ ]:


train_data_path = '../input/train.csv'
test_data_path = "../input/test.csv"

df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
full_data = [df, test_df]


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.describe()


# ## Analyzing NaN/Null values <a class="anchor" id="fourth-bullet"></a>
# 
# - Find out how many NaN/Null values are present in the data set
# - Remove them if they are a trival % 

# In[ ]:


def find_nan_percentage(df):
    total_row_count = df.shape[0]
    non_nan_row_count =  df.dropna().shape[0]
    nan_row_count = total_row_count - non_nan_row_count
    
    print("Total number of DATA rows", total_row_count)
    print("Total number of DATA rows that have NaN/Null values: ", nan_row_count)
    print("Total numver of DATA rows that doesn't have NaN/Null values", non_nan_row_count)
    
    #Draw a pie chart to represent the above values
    plt.figure(figsize=(8,6))
    labels = 'Fully filled Rows', 'NaN Rows'
    sizes = [non_nan_row_count, nan_row_count]
    colors = ['skyblue', 'yellowgreen']
    explode = (0.1, 0)

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', shadow=True, explode=explode)
    return (nan_row_count/total_row_count)*100


# In[ ]:


nan_percentage = find_nan_percentage(df)
print("{0:.2f}% of the data have NaN values".format(nan_percentage))


# ### Inference
# 
# - Around 80% of the rows aren't completely filled. So we can't omit all of them
# - We need to further dive into the data and analyze more

# ## Analyzing the columns  <a class="anchor" id="fifth-bullet"></a>

# In[ ]:


df.columns


# ### 1. Pclass

# In[ ]:


print("Number of rows with NaN is: ", df.Pclass.isna().sum())
pd.crosstab(df.Pclass, df.Survived).plot(kind='bar', figsize = (20,10))
plt.xlabel("Class")
plt.ylabel("Survival frequency")
print (df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


# ### Inference
# The higher the class, the higher the ratio of survival (1 being the highest class). It's quite clear

# ### 2.Sex

# In[ ]:


print("Number of rows with NaN is: ", df.Sex.isna().sum())
pd.crosstab(df.Sex, df.Survived).plot(kind='bar', figsize = (20,10))
print (df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())


# ### Inference
# Quite certain that females survived more than males

# ### 3. Total Family

# In[ ]:


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
pd.crosstab(df.FamilySize, df.Survived).plot(kind='bar', figsize=(10,10))


# ### Inference
# - This plot was made to check if families were saved or survived more than induviduals. It doesn't look like that
# - We could assume that families of mid sizes survived considerably well when compared to the others

# ### 4. Name

# #### - First, let us see how the names are listed

# In[ ]:


print("Number of rows with NaN is: ", df.Name.isna().sum())
df.Name.head(5)


# #### - We will get the title alone to see if they make any value

# In[ ]:


import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)


# In[ ]:


pd.crosstab(df['Title'], df['Sex'])


# #### - We coud generalize all the trivial titles to a common basket

# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# ### 5. Embarked

# In[ ]:


print("Number of rows with NaN is: ", df.Embarked.isna().sum())


# In[ ]:


df.Embarked.value_counts()


# #### We would replace the NaN values with the most common Embarked place -- S

# In[ ]:


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


pd.crosstab(df.Embarked, df.Survived).plot(kind='bar', figsize=(10,10))


# ### Inference
# 
# People who started from Cherbourg survived more than people who started from anywehre else!!!

# ### 6. Fare

# In[ ]:


print("Number of rows with NaN is: ", df.Fare.isna().sum())
df.plot(kind='scatter', x='Fare', y='Survived')


# ### Inference
# The scatter plot tells that fare doesn't make any influence on survival 

# #### - Categorize the fare and check if it makes more sense

# In[ ]:


for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(df['Fare'].median())
df['CategoricalFare'] = pd.qcut(df['Fare'], 4)
print (df[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


# In[ ]:


pd.crosstab(df.CategoricalFare,df.Survived).plot(kind='bar', figsize=(10,10))


# ### Interference
# Higher the category, higher the survival rate!!!

# ### 7. Age

# In[ ]:


print("Number of rows with NaN is: ", df.Age.isna().sum())


# - We have plenty of missing values in Age section
# - We generate random numbers between (mean - std) and (mean + std)
# - We then categorize the age in ranges

# In[ ]:


for dataset in full_data:
    age_avg        = dataset['Age'].mean()
    age_std        = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
df['CategoricalAge'] = pd.cut(df['Age'], 5)

print (df[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
pd.crosstab(df.CategoricalAge, df.Survived).plot(kind='bar', figsize=(10,10))


# ### Inference
# The survival rate was certainly high for youngsters

# In[ ]:


sns.violinplot(x=df.CategoricalAge, y=df.Survived)


# ## Data Cleaning  <a class="anchor" id="sixth-bullet"></a>

# In[ ]:


for dataset in full_data:
    # Mapping Sex
    name_map = {'female': 0, 'male': 1}
    dataset['Sex'] = dataset['Sex'].map(name_map).astype(int)
    
    # Mapping titles
    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_map)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    #dataset['Embarked'] = dataset['Embarked'].map(embarked_map).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map(embarked_map).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4


# In[ ]:


df.head()


# ### Dropping the unnecessary columns

# In[ ]:


PassengerId = test_df['PassengerId']
columns_to_be_dropped = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', ]
df = df.drop(columns_to_be_dropped, axis=1)
df = df.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
test_df = test_df.drop(columns_to_be_dropped, axis=1)


# In[ ]:


df.head()


# In[ ]:


test_df.head()


# ## One hot Encoding <a class="anchor" id="seventh-bullet"></a>

# In[ ]:



train_a = pd.get_dummies(df['Pclass'], prefix = "Pclass")
train_b = pd.get_dummies(df['Fare'], prefix = "Fare")
train_c = pd.get_dummies(df['Title'], prefix = "Title")
train_frames = [df, train_a, train_b, train_c]
df = pd.concat(train_frames, axis = 1)

test_a = pd.get_dummies(test_df['Pclass'], prefix = "Pclass")
test_b = pd.get_dummies(test_df['Fare'], prefix = "Fare")
test_c = pd.get_dummies(test_df['Title'], prefix = "Title")
test_frames = [test_df, test_a, test_b, test_c]
test_df = pd.concat(test_frames, axis = 1)

to_be_dropped = ['Pclass', 'Fare', 'Title']
df = df.drop(to_be_dropped, axis=1)
test_df = test_df.drop(to_be_dropped, axis=1)


# In[ ]:


df.head()


# ## Splitting into train and test features <a class="anchor" id="eighth-bullet"></a>

# In[ ]:


features = df.drop("Survived", axis=1)
targets = df.Survived.values


# In[ ]:


from sklearn.model_selection import train_test_split
train_features,test_features,train_targets,test_targets = train_test_split(features,targets,test_size = 0.20,random_state = 42)


# ## Neural network architecture  <a class="anchor" id="ninth-bullet"></a>
# Finally we have prepared our data. Now it's time to train it with neural nets !!!

# In[ ]:


# Imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Building the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(train_features.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss = 'mean_squared_error', optimizer='adam', metrics=['acc', 'mae'])
model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='test.model.best.hdf5', 
                               verbose=1, save_best_only=True)

history = model.fit(train_features, train_targets, validation_split=0.2, epochs=250, batch_size=32, verbose=0, callbacks=[checkpointer], shuffle=True)

#Load the Model with the Best Classification Accuracy on the Validation Set
model.load_weights('test.model.best.hdf5')


# ## Evaluating the model <a class="anchor" id="tenth-bullet"></a>

# In[ ]:


#print(vars(history))
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# ## Predicting the model <a class="anchor" id="eleventh-bullet"></a>

# In[ ]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score,mean_squared_error

y_pred = model.predict(test_df)
y_pred = y_pred.flatten()
print(y_pred.shape)
y_pred = np.where(y_pred<.5,0,1)

test_submission = "../input/gender_submission.csv"
submission_df = pd.read_csv(test_submission)
submission_targets = submission_df['Survived'].values
plt.plot(submission_targets)
plt.plot(y_pred)
plt.title('Prediction')
print("Accuracy: ", accuracy_score(submission_targets,y_pred))


# ## Other Classification techniques <a class="anchor" id="twelfth-bullet"></a>

# In[ ]:


from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn.metrics import accuracy_score,mean_squared_error


# In[ ]:


classifiers=[['Logistic Regression :',LogisticRegression()],
       ['Decision Tree Classification :',DecisionTreeClassifier()],
       ['Random Forest Classification :',RandomForestClassifier(n_estimators=30, max_features=7, max_depth=None, min_samples_split=2)],
       ['Gradient Boosting Classification :', GradientBoostingClassifier()],
       ['Ada Boosting Classification :',AdaBoostClassifier(n_estimators=100)],
       ['Extra Tree Classification :', ExtraTreesClassifier()],
       ['K-Neighbors Classification :',KNeighborsClassifier(n_neighbors=7)],
       ['Support Vector Classification :',SVC()],
       ['Gausian Naive Bayes :',GaussianNB()],
       ['XGBoost Classification :', xgb.XGBClassifier(objective="binary:logistic", random_state=42, n_estimators=20, al_metric=["auc", "error"])],]
cla_pred=[]
max_accuracy = float('-inf')
for index, (name,model) in enumerate(classifiers):
    model=model
    model.fit(train_features,train_targets)
    predictions = model.predict(test_features)
    cla_pred.append(accuracy_score(test_targets,predictions))
    accuracy_ = accuracy_score(test_targets,predictions)
    if accuracy_ > max_accuracy:
        max_index = index
    print(name, accuracy_)


# In[ ]:


y_ax=['Logistic Regression' ,
      'Decision Tree Classifier',
      'Random Forest Classifier',
      'Gradient Boosting Classifier',
      'Ada Boosting Classifier',
      'Extra Tree Classifier' ,
      'K-Neighbors Classifier',
      'Support Vector Classifier',
      'Gaussian Naive Bayes',
      'XGBoost Classification']
x_ax=cla_pred
sns.barplot(x=x_ax,y=y_ax)
plt.xlabel('Accuracy')


# In[ ]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score,mean_squared_error

y_pred = classifiers[max_index][1].predict(test_df)
y_pred = y_pred.flatten()
print(y_pred.shape)
y_pred = np.where(y_pred<.5,0,1)

test_submission = "../input/gender_submission.csv"
submission_df = pd.read_csv(test_submission)
submission_targets = submission_df['Survived'].values
plt.plot(submission_targets)
plt.plot(y_pred)
plt.title('Prediction')
print("Accuracy: ", accuracy_score(submission_targets,y_pred))


# # Submission <a class="anchor" id="thirteenth-bullet"></a>

# In[ ]:


# Generate Submission File 
Submission = pd.DataFrame({'PassengerId': PassengerId,
                            'Survived': y_pred })
Submission.to_csv("Submission.csv", index=False)

