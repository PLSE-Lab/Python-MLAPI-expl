#!/usr/bin/env python
# coding: utf-8

# ## Import All The Things

# In[ ]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="ticks", color_codes=True)

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2


# In[ ]:


seed = 7
np.random.seed(seed)


# <h2>Loading Data</h2>

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_copy = train.copy()
test_copy = test.copy()

all_data = pd.concat([train, test], sort = False)

train.head()


# ## Data Analysis

# ### Data Description
# 
# **PassengerId:** Unique identifier for each row. It can't have any effect on the outcome.
# 
# **Survived:** Survival of the passenger. 0 = No, 1 = Yes
# 
# **Pclas:** A proxy for socio-economic status (SES); 1 = Upper, 2 = Middle, 3 = Lower. People with higher social status may have higher survival rate.
# 
# **Name:** Name of the passenger. I don't think in its entirety have any effect on that person's survival. But name may carry additional information like title, family name, which may indicate that persons social staus and thus having impact on survival.
# 
# **Sex:** Gender may have significant role on survival. 
# 
# **Age:** Age in years. Old people have lower survival rate or childrens? 
# 
# **SibSp:**  Number of siblings / spouses aboard the Titanic. 
# 
# **Parch:**  Number of parents / children aboard the Titanic. 
# 
# **Ticket:** Ticket number
# 
# **Fare:** Passenger fare. Higher the fare, higher the social class?
# 
# **Cabin:** Cabin number
# 
# **Embarked:** Port of Embarkation; C = Cherbourg, Q = Queenstown, S = Southampton
# 

# In[ ]:


train.info()


# We have 7 numerical columns and 5 non numerical columns. And missing values in 3 columns. We have to convert non numerical columns to numerical columns and fill up the missing values before we fit the dataset into our model. But first, let's see how each of them affects the survival chance of the titanic passengers. 
# 
# 

# In[ ]:


def cat_plot(x, y = 'Survived', kind = 'bar', hue = None, data = train):
    sns.catplot(x = x, y = y, kind = kind, hue = hue, data = data)
    
def count_plot(x, col = 'Survived', data = train):
    sns.catplot(x = x, col = col, kind = 'count', data = data)
    


# ### Relationship between "Pclass" and "Survived"

# In[ ]:


train.groupby('Pclass').Survived.mean()


# In[ ]:


cat_plot('Pclass')


# As we assumed, passenger class has a significant impact on survival rate. Passengers from upper class survived most and passengers from lower class survived less.

# ### Relationship between "Sex" and "Survived"

# In[ ]:


train.groupby('Sex').Survived.mean()


# In[ ]:


cat_plot('Sex')


# Most female survived while most male didn't make it! Women and children first? We will have to see.

# ### Relationship between "Age" and "Survived"

# In[ ]:


grid = sns.FacetGrid(data = train, hue = 'Survived', height = 5, aspect = 3)
plt.xticks(range(0, 81, 5))
grid.map(sns.kdeplot, 'Age').add_legend()


# Age doesn't have any linear relationship with survival. But we can break them into few categories -
# 
# Age 30 to 57 has no effect of survival
# 
# Age under 13 has higer chance of survival
# 
# Age 14 to 30 has lower chance of survival
# 
# Age above 57 has lower chance of survival

# ### Relationship between "SibSp" and "Survived"

# In[ ]:


train.groupby(['SibSp']).Survived.mean()


# In[ ]:


cat_plot('SibSp')


# Passengers with more SipSp (larger family) has lower chance of survival. But with no SibSp (traveling alone?) lower the survival chances too. 

# ### Relationship between "Parch" and "Survived"

# In[ ]:


train.groupby(['Parch']).Survived.mean()


# In[ ]:


cat_plot('Parch')


# No obvious pattern here. We will combined above two feature into one to see if family size matters.

# ### Relationship between "Fare" and "Survived"

# In[ ]:


grid = sns.FacetGrid(data = train, hue = 'Survived', height = 5, aspect = 2)
grid.map(sns.kdeplot, 'Fare').add_legend()


# The relationship is as same as Pclss. Higher fare means higher class, thus higer survival rate.

# ### Relationship between "Embarked" and "Survived"

# In[ ]:


cat_plot('Embarked')


# C has highest chance of survival, but the other two is same.

# ## Feature Engineering and Data Preparation

# ### Title

# In[ ]:


all_data['Title'] = all_data['Name'].str.extract('([A-Za-z]+)\.', expand=True)


# In[ ]:


all_data['Title'].value_counts()


# In[ ]:


all_data['Title'].value_counts().plot(kind = 'bar')


# In[ ]:


sns.catplot(x = 'Title', y = 'Survived', kind = 'bar', hue = None, data = all_data, aspect = 3)


# We will replace the less frequently used titles with meaningful categories.

# In[ ]:


mappings = {'Dr':'Respected_Male', 'Col':'Respected_Male', 'Major':'Respected_Male', 'Capt':'Respected_Male',
            'Mme':'Noble', 'Mlle':'Noble', 'Countess': 'Noble', 'Lady': 'Noble', 'Sir':'Noble',
            'Ms' : 'Miss', 'Rev': 'Other', 'Jonkheer': 'Other', 'Dona': 'Other', 'Don': 'Other' 
           }

all_data.replace({'Title': mappings}, inplace = True)


# In[ ]:


all_data['Title'].value_counts()


# In[ ]:


sns.catplot(x = 'Title', y = 'Survived', kind = 'bar', hue = None, data = all_data, aspect = 3)


# In[ ]:


one_hot_encoding_list = []
one_hot_encoding_list.append('Title')


# ### Age

# In[ ]:


all_data['Age'].isnull().sum()


# There are quite a few missing values in Age column. We will fill these with the average of same titled passeners. Then we will divide age into five bins. Since we have seen affect of age on survival rate changes after every 15 years or so.

# In[ ]:


title_grouped = all_data.groupby(['Title'])

for title in all_data.Title.unique():
    all_data.loc[(all_data.Age.isnull()) & (all_data.Title == title), 'Age'] = title_grouped.get_group(title).Age.mean()


# In[ ]:


all_data['AgeBin'] = pd.qcut(all_data['Age'], 5)


# In[ ]:


label = preprocessing.LabelEncoder()
all_data['AgeBin'] = label.fit_transform(all_data['AgeBin'])
cat_plot('AgeBin', data = all_data)


# No linear relationship, so we will apply one hot encoding to 'AgeBin' too.

# In[ ]:


one_hot_encoding_list.append('AgeBin')


# ### Family Size

# In[ ]:


all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1


# In[ ]:


all_data['FamilySize'].value_counts(dropna = False)


# In[ ]:


cat_plot('FamilySize', data = all_data)


# ### Alone, SmallFamily, BigFamily

# In[ ]:


all_data['Alone'] = 0;
all_data.loc[all_data['FamilySize'] == 1, 'Alone'] = 1

all_data['SmallFamily'] = 0;
all_data.loc[(all_data['FamilySize'] > 1) & (all_data['FamilySize'] <= 4), 'SmallFamily'] = 1

all_data['BigFamily'] = 0;
all_data.loc[all_data['FamilySize'] > 4, 'BigFamily'] = 1


# ### HaveCabin

# In[ ]:


all_data.Cabin.describe()


# In[ ]:


all_data.Cabin.isnull().sum()


# In[ ]:


all_data['HaveCabin'] = 0;
all_data.loc[all_data['Cabin'].notnull(), 'HaveCabin'] = 1


# In[ ]:


cat_plot('HaveCabin', data = all_data)


# In[ ]:


cat_plot('HaveCabin', data = all_data, hue = 'Pclass', kind = 'point')


# The relationship is not entirely linear with Pclass.

# ### Anyone Survived From Group

# In[ ]:


all_data['AnyoneSurvivedFromGroup'] = 0.5
                
for _, grp_df in all_data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['AnyoneSurvivedFromGroup'] == 0) | (row['AnyoneSurvivedFromGroup']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    all_data.loc[all_data['PassengerId'] == passID, 'AnyoneSurvivedFromGroup'] = 1
                elif (smin == 0.0):
                    all_data.loc[all_data['PassengerId'] == passID, 'AnyoneSurvivedFromGroup'] = 0


# In[ ]:


cat_plot('AnyoneSurvivedFromGroup', data = all_data)


# In[ ]:



all_data[all_data['AnyoneSurvivedFromGroup'] != 0.5]['PassengerId'].count()


# In[ ]:


for _, grp_df in all_data.groupby('Cabin'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['AnyoneSurvivedFromGroup'] == 0) | (row['AnyoneSurvivedFromGroup']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    all_data.loc[all_data['PassengerId'] == passID, 'AnyoneSurvivedFromGroup'] = 1
                elif (smin == 0.0):
                    all_data.loc[all_data['PassengerId'] == passID, 'AnyoneSurvivedFromGroup'] = 0


# In[ ]:


all_data[all_data['AnyoneSurvivedFromGroup'] != 0.5]['PassengerId'].count()


# In[ ]:


cat_plot('AnyoneSurvivedFromGroup', data = all_data)


# ### FareBin

# We fill missing fare with the average fare of Pclass.

# In[ ]:


all_data.Fare.isnull().sum()


# In[ ]:


pclass_grouped = all_data.groupby(['Pclass'])

for pclass in all_data.Pclass.unique():
    all_data.loc[(all_data.Fare.isnull()) & (all_data.Pclass == pclass), 'Fare'] = pclass_grouped.get_group(pclass).Fare.mean()


# In[ ]:


all_data['FareBin'] = pd.qcut(all_data['Fare'], 4)
label = preprocessing.LabelEncoder()
all_data['FareBin'] = label.fit_transform(all_data['FareBin'])


# In[ ]:


cat_plot('FareBin', data = all_data)


# A nice linear relationship!

# In[ ]:


cat_plot(x= 'Pclass', y ='FareBin', kind = 'point', data = all_data)


# But the fare information is already captured by Pclass. So, this will be redundant given that we have a fairly small dataset.

# ### Filling Missing Values

# In[ ]:


all_data.isnull().sum()


# We have two missing values in Embarked. Lets fill that up! 

# In[ ]:


all_data.loc[all_data['Embarked'].isnull(), 'Embarked'] = all_data['Embarked'].mode()[0]


# Also, we will apply one hot encoding to 'Embarked'

# In[ ]:


one_hot_encoding_list.append('Embarked')


# ### Data preaparation

# Let's first remove the columns that have already proven less useful and also columns that we have turned into bins.

# In[ ]:


all_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'FareBin'], axis = 1, inplace = True)


# In[ ]:


all_data.head()


# Let's turn rest of the non numerical columns into numerical ones!

# In[ ]:


sex_mappings = {'male': 0,'female': 1}
all_data['Sex'].replace(sex_mappings, inplace = True)


# In[ ]:


one_hot_encoding_list


# We all add 'Pclass' to this list too.

# In[ ]:


one_hot_encoding_list.append('Pclass')


# In[ ]:


all_data = pd.get_dummies(data = all_data, columns = one_hot_encoding_list)


# In[ ]:


all_data.head()


# ## Build A Simple Neural Network Model

# In[ ]:


train_data = all_data[:len(train)]
test_data = all_data[len(train):]


# In[ ]:


X = train_data.iloc[: , 1:].values
y = train_data.iloc[:, 0].values

test_data = test_data.iloc[: , 1:]
X_test = test_data.values

print(str(X.shape))
print(str(X_test.shape))


# In[ ]:


def create_model():

    model = Sequential()
    model.add(Dense(16, input_dim = X.shape[1], activation = 'relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(16, activation = 'relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model


# First we will find the best training epoch for our model

# In[ ]:


epochs = 150
model = create_model()
history = model.fit(X, y, epochs=epochs, validation_split = 0.2, batch_size=10, verbose = 0)

val_acc = history.history['val_acc']
acc = history.history['acc']

max_val_acc_epoch = np.argmax(val_acc) + 1;
print("Maximum validation accuracy epoch: {}".format(max_val_acc_epoch))

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15,12))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


num_epochs = max_val_acc_epoch

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

cvscores = []

for train, test in kfold.split(X, y):
  model = create_model()
  history = model.fit(X[train], y[train], epochs=num_epochs, batch_size=10, verbose = 0)
  scores = model.evaluate(X[test], y[test], verbose=0)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


# In[ ]:


model = create_model()
history = model.fit(X, y, epochs=num_epochs, batch_size=10, verbose = 0)


# In[ ]:


prediction = model.predict(X_test)


# In[ ]:


submission = pd.DataFrame(test_copy[['PassengerId']])
submission['Survived'] = prediction
submission['Survived'] = submission['Survived'].apply(lambda x: 0 if x < 0.5 else 1)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', index = False)

