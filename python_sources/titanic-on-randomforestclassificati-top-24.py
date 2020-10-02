#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this notebook I want to show you how I have reached up to 24%. Date of writing this notebook: 24-12-2018.
# To begin, let me introduce myself. My name is Nurtai Maksat. I am a 3rd year student at the International University of Information Technology. 
# 
# For this task, I used the RandomForestClassifier model. If you read to the end, you will find out which parameters I used. Let's get started. Below I will show the steps you need to perform.
#     1. Importing the libraries
#     2. Importing the dataset
#     3. Create function that processes data
#         a. Create new columns
#         b. Fill missing values
#         c. Delete unnecessary data
#     4. Trainig
#         a. Process data
#         b. Create model and train model with 'train' dataset
#     5. Prediction
#     6. Save result to csv

# In[ ]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[ ]:


# Importing the real dataset for prediction
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
# Temporary data set for analysis and preprocessing
temp_df = pd.concat([df_train, df_test], ignore_index=True, sort=False)

temp_df.head()


# # Feature Engineering and Data Cleaning
# For a start, look at which columns have missing data.

# In[ ]:


temp_df.isnull().sum() 


# Well, we learned in which columns there is blank data. Let me show you how the data was processed using a temporary dataset.

# In[ ]:


temp_df['NameLength'] = temp_df['Name'].apply(len)
temp_df[['Name', 'NameLength']].head() 


# In the process of studying and analyzing the data, I found a more accurate article that showed that the length of the names of passengers greatly influences the final result in the RandomForest model.

# In[ ]:


# HasCabin - means that the passenger has his own Cabin
temp_df['HasCabin'] = temp_df['Cabin'].apply(lambda i: 1 if type(i)==str else 0)
temp_df[['Cabin', 'HasCabin']].head() 


# In[ ]:


temp_df['FamilySize'] = 1 + temp_df['Parch'] + temp_df['SibSp']
# Using siblings and parents / children, we can create a new column called FamilySize.
temp_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean() 


# As we see, this has a good impact on our forecast, but let's not stop and go ahead and examine people, whether they are alone on this ship or not, and how they react to survival.

# In[ ]:


# In the beginning we will indicate that all people are lonely.
temp_df['IsAlone'] = 0
# Now check the number of people in the family. If it is 1, it means that the person is one.
temp_df.loc[temp_df['FamilySize']==1, 'IsAlone'] = 1
temp_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# Wow, good! The impact is significant.

# In[ ]:


# Using the "fillna" we fill the NA / NaN value with an "S". "S" is the most common port name.
temp_df['Embarked'] = temp_df['Embarked'].fillna('S')
# Reading a lot of kernels I found out that instead of LabelEncoder we can use the map function.
temp_df['Embarked'] = temp_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
temp_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


# In[ ]:


temp_df['Fare'] = temp_df['Fare'].fillna(temp_df['Fare'].median())
# pandas.cut - This feature is useful for moving from a continuous variable to a categorical variable.
# For example, a reduction may transform age into groups of age ranges.
CategoricalFare = pd.cut(temp_df['Fare'], 5)
# output CategoricalFare:  [(-0.512, 102.466] < (102.466, 204.932] < (204.932, 307.398] < (307.398, 409.863] < (409.863, 512.329]]
# loc function - Access a group of rows and columns by label(s) or a boolean array.
temp_df.loc[temp_df['Fare']<=102, 'Fare'] = 0
temp_df.loc[(temp_df['Fare']>102) & (temp_df['Fare']<=204), 'Fare'] = 1
temp_df.loc[(temp_df['Fare']>204) & (temp_df['Fare']<=307), 'Fare'] = 2
temp_df.loc[(temp_df['Fare']>307) & (temp_df['Fare']<=409), 'Fare'] = 3
temp_df.loc[(temp_df['Fare']>409) & (temp_df['Fare']<=512), 'Fare'] = 4
temp_df['Fare'] = temp_df['Fare'].astype(int)
temp_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean()


# In[ ]:


# To work with the Age, we need to take the mean and standard deviation.
mean = temp_df['Age'].mean()
std = temp_df['Age'].std()
# Create an array with the size of the missing data and fill the array randomly choosing one value from the following two operations.
randomAge = np.random.randint(mean-std, mean+std, size=temp_df['Age'].isnull().sum())
temp_df['Age'][np.isnan(temp_df['Age'])] = randomAge
# Then everything goes according to the previous algorithm.
CategoricalAge = pd.cut(temp_df['Age'], 5)
# output: [(0.34, 16.336] < (16.336, 32.252] < (32.252, 48.168] < (48.168, 64.084] < (64.084, 80.0]]
temp_df.loc[temp_df['Age']<=16 , 'Age'] = 0
temp_df.loc[(temp_df['Age']>16) & (temp_df['Age']<=32), 'Age'] = 1
temp_df.loc[(temp_df['Age']>32) & (temp_df['Age']<=48), 'Age'] = 2
temp_df.loc[(temp_df['Age']>48) & (temp_df['Age']<=64), 'Age'] = 3
temp_df.loc[(temp_df['Age']>64) & (temp_df['Age']<=80), 'Age'] = 4
temp_df['Age'] = temp_df['Age'].astype(int)
temp_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean() 


# In[ ]:


temp_df['Name'].head()


# In[ ]:


# Here you have to work with strings.
# That is, we need to take on behalf of the passenger polite treatment (Mr., Mrs, etc.)
temp_df['Title'] = [i.split(', ')[1].split('.')[0] for i in temp_df['Name']]
temp_df['Title'] = temp_df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
temp_df['Title'] = temp_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
temp_df['Title'] = temp_df['Title'].replace('Mme', 'Mrs')
temp_df['Title'] = temp_df['Title'].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4})


temp_df['Sex'] = temp_df['Sex'].map({'male':0, 'female':1}).astype(int)


# In[ ]:


temp_df.head()


# Well, we finished this stage. For further work and convenience, I will put all these codes into a function.

# In[ ]:


def analize(df):
    df['NameLength'] = df['Name'].apply(len)
    df['HasCabin'] = df['Cabin'].apply(lambda i: 1 if type(i)==str else 0)
    df['FamilySize'] = 1 + df['Parch'] + df['SibSp']
    
    df['IsAlone'] = 0
    df.loc[df['FamilySize']==1, 'IsAlone'] = 1

    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    CategoricalFare = pd.cut(df['Fare'], 5)
    # output: [(-0.512, 102.466] < (102.466, 204.932] < (204.932, 307.398] < (307.398, 409.863] < (409.863, 512.329]]
    df.loc[df['Fare']<=102, 'Fare'] = 0
    df.loc[(df['Fare']>102) & (df['Fare']<=204), 'Fare'] = 1
    df.loc[(df['Fare']>204) & (df['Fare']<=307), 'Fare'] = 2
    df.loc[(df['Fare']>307) & (df['Fare']<=409), 'Fare'] = 3
    df.loc[(df['Fare']>409) & (df['Fare']<=512), 'Fare'] = 4
    df['Fare'] = df['Fare'].astype(int)

    # --- Age
    mean = df['Age'].mean()
    std = df['Age'].std()
    randomAge = np.random.randint(mean-std, mean+std, size=df['Age'].isnull().sum())
    df['Age'][np.isnan(df['Age'])] = randomAge
    CategoricalAge = pd.cut(df['Age'], 5)
    # output: [(0.34, 16.336] < (16.336, 32.252] < (32.252, 48.168] < (48.168, 64.084] < (64.084, 80.0]]
    df.loc[df['Age']<=16 , 'Age'] = 0
    df.loc[(df['Age']>16) & (df['Age']<=32), 'Age'] = 1
    df.loc[(df['Age']>32) & (df['Age']<=48), 'Age'] = 2
    df.loc[(df['Age']>48) & (df['Age']<=64), 'Age'] = 3
    df.loc[(df['Age']>64) & (df['Age']<=80), 'Age'] = 4
    df['Age'] = df['Age'].astype(int)

    # --- Title
    df['Title'] = [i.split(', ')[1].split('.')[0] for i in df['Name']]
    df['Title'] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4})

    df['Sex'] = df['Sex'].map({'male':0, 'female':1}).astype(int)
    
    df = df.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])
    return df


# # Training model

# In[ ]:


train = analize(df_train)
train.head()


# In[ ]:


# Splitting dataset into X and y
y = train['Survived'].values
x = train.iloc[:, 1:].values

X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=0)


# The next step is to create a model with the following parameters:
# 1. n_estimators=100
# 2. max_depth=20
# 3. max_features='sqrt'

# In[ ]:


classifier = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test) 


# In[ ]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
import itertools
labels = ['Predicted NO', 'Predicted YES','Actual NO','Actual YES']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier\n')
ax.set_xticklabels([''] + labels[0:2])
ax.set_yticklabels([''] + labels[2:4])
fmt = '.0f'
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
        horizontalalignment="center",
        color="red", fontsize = 22)
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# # Prediction

# In[ ]:


test = analize(df_test)
test.head() 


# In[ ]:


output_train = train['Survived'].values
intput_train = train.drop(columns=['Survived']).values
input_test = test.values


# In[ ]:


output_pred = classifier.predict(input_test)


# In[ ]:


result = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': output_pred.astype(int)
})


# In[ ]:


result.head()

