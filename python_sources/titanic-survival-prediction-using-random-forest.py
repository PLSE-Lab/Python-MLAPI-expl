#!/usr/bin/env python
# coding: utf-8

# 
# ## Please upvote this Kernel, if you find it useful.
# 

# **In this kernel we will learn machine learning step by step:**
# 1. **Find out missing data** columns in train and test data
# 2. **Merging datasets**
# 3. **Filling missing values** in datasets
# 4. **Creating new features** (columns) from existing features
# 5. **Visualizing data** using **matplotlib**
# 6. **Label Encoding** the categorical data (**A method to convert catagorical text data into numerical data**)
# 7. Checking approximate **feature importance**
# 8. **Scaling up data** using Standard Scaler
# 9. Initializing **simple random forest** model to use
# 10. **Train** the Model
# 11. **Predict the outcome** from the model
# 11. Submit the **output** file to competition

# In[3]:


print('Loading packages')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print('These are the files to use: ',os.listdir("../input"))
from sklearn import preprocessing
from statistics import mean
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import random
from matplotlib import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
le = preprocessing.LabelEncoder()
import re


# In[28]:


print('reading input files..')
data = pd.read_csv('../input/train.csv')
sampl = pd.read_csv('../input/gender_submission.csv')


# In[29]:


test  = pd.read_csv('../input/test.csv')


# In[30]:


data.head()


# Value NaN means null value. It looks like data has some **missing values**. We need to **first check which columns have missing values**, and then we will fill them accordingly

# First, **let's combine both** train and test data, to make it easy to fill values and make operation and create relation between train and test data.

# In[31]:


# Appending test data with train data, since both dataset can have related values like family name and ticket
df = data.append(test, sort = False)


# **Find out missing data**

# In[32]:


totalt = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([totalt, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)


# Oh!!!, we have **Cabin, Age, Embarked and Fare** has missing values after combining both training and test data.

# **Creating New Features**

# In[33]:


# Creating a TicketId feature, it will tell which person was part of that Family or group
ticketNum = pd.DataFrame(df.Ticket.value_counts())
ticketNum.rename(columns = {'Ticket' : 'TicketNum'}, inplace = True)
ticketNum['TicketId'] = pd.Categorical(ticketNum.index).codes
ticketNum.loc[ticketNum.TicketNum < 3, 'TicketId'] = -1
df = pd.merge(left = df, right = ticketNum, left_on = 'Ticket', 
              right_index = True, how = 'left', sort = False)
df = df.drop(['TicketNum'],axis=1)
df.head()


# In[34]:


# Separating FamilyName
df['FamilyName'] = df.Name.apply(lambda x : str.split(x, ',')[0])


# If you are a begginer, you can leave this portion of creating 'FamilySurv, and come back later when start unserstanding.

# In[35]:


# Lets create one more feature FamilySurvival
df['FamilySurv'] = 0.5
for _, grup in df.groupby(['FamilyName','Fare']):
    if len(grup) != 1:
        for index, row in grup.iterrows():
            smax = grup.drop(index).Survived.max()
            smin = grup.drop(index).Survived.min()
            pid = row.PassengerId
            
            if smax == 1:
                df.loc[df.PassengerId == pid, 'FamilySurv'] = 1.0
            elif smin == 0:
                df.loc[df.PassengerId == pid, 'FamilySurv'] = 0.0
for _, grup in df.groupby(['Ticket']):
    if len(grup) != 1:
        for index, row in grup.iterrows():
            if (row.FamilySurv == 0.0 or row.FamilySurv == 0.5):
                smax = grup.drop(index).Survived.max()
                smin = grup.drop(index).Survived.min()
                pid  = row.PassengerId

                if smax == 1:
                    df.loc[df.PassengerId == pid, 'FamilySurv'] = 1.0
                elif smin == 0:
                    df.loc[df.PassengerId == pid, 'FamilySurv'] = 0.0
df.FamilySurv.value_counts()


# In[36]:


# CabinNum (Finding, how many cabin a person has)
def CabinNum(data):
    data.Cabin = data.Cabin.fillna('0')
    regex = re.compile('\s*(\w+)\s*')
    data['CabinNum'] = data.Cabin.apply(lambda x : len(regex.findall(x)))
CabinNum(df)


# In[37]:


df.CabinNum.value_counts()


# In[38]:


# Creating Feature Title, since Higher Rank people has more survival chances, should give hit and trail
def TitleFunc(data):
    sub = {'Col.','Rev.', 'Mr.','Sir.','Jonkheer.', 'Don.','Dona.','Capt.',
           'General.','Major.'}
    sub1 = {'Miss.','Mme.','Mlle.','Ms.'}
    sub2 = {'Mrs.','Countess.','Lady.'}
    sub3 = {'Master.'}
    sub4 = {'Dr.'}
    pattern, pattern1, pattern2, pattern3 = '|'.join(sub), '|'.join(sub1), '|'.join(sub2), '|'.join(sub3)
    pattern4 = '|'.join(sub4)
    data['Title'] = ''
    data.loc[data['Name'].str.contains(pattern),'Title'] = 'Mr.'
    data.loc[data['Name'].str.contains(pattern1),'Title'] = 'Miss.'
    data.loc[data['Name'].str.contains(pattern2),'Title'] = 'Mrs.'
    data.loc[data['Name'].str.contains(pattern3),'Title'] = 'Master.'
    data.loc[(data['Name'].str.contains(pattern)) & (data['Age'] <=13),'Title'] = 'Master.'
    data.loc[(data['Name'].str.contains(pattern4)) & (data['Sex'] == 'female'),'Title'] = 'Dr.f'
    data.loc[(data['Name'].str.contains(pattern4)) & (data['Sex'] == 'male'),'Title'] = 'Dr.m'
TitleFunc(df)


# ### Visualize data

# In[39]:


#Lets see Who Survived most
train1 = df[0:891].copy()
sns.set(style="whitegrid")
plt.figure(figsize=(10,3))
ax = sns.barplot(x="Title", y="Survived", data=train1)
#ax = sns.barplot(x="Title", y="Survived",hue='Title', data=train1)


# Oh Wow... Doctor of female category survived most and after that Mrs and after that Miss and then Master and then Male Doctors

# In[51]:


# Lets first check missing fare
df.loc[df['Fare'].isnull()]


# In[41]:


#Let's find simlar data, and fill that for missing fare
df.loc[(df['Age'] >= 60) & (df['Pclass'] ==3) & (df['Sex'] == 'male') & (df['Embarked'] =='S')]


# **Fill Missing data**

# In[50]:


df.loc[df['Fare'].isnull(), 'Fare'] = 7            #First fill missing fare by least value


# **More Feature Creation**

# In[43]:


# Creating FareCat Title, since High Fare people has more survival chances
def FareFunc(data):
    data['FareCat'] = 0
    data.loc[data['Fare'] < 8, 'FareCat'] = 0
    data.loc[(data['Fare'] >= 8 ) & (data['Fare'] < 16),'FareCat' ] = 1
    data.loc[(data['Fare'] >= 16) & (data['Fare'] < 30),'FareCat' ] = 2
    data.loc[(data['Fare'] >= 30) & (data['Fare'] < 45),'FareCat' ] = 3
    data.loc[(data['Fare'] >= 45) & (data['Fare'] < 80),'FareCat' ] = 4
    data.loc[(data['Fare'] >= 80) & (data['Fare'] < 160),'FareCat' ] = 5
    data.loc[(data['Fare'] >= 160) & (data['Fare'] < 270),'FareCat' ] = 6
    data.loc[(data['Fare'] >= 270), 'FareCat'] = 7
FareFunc(df)


# ### Lets check which Fare class Survived along with their title

# In[44]:


train1 = df[0:891].copy()
sns.set(style="whitegrid")
plt.figure(figsize=(14,3.5))
ax = sns.barplot(x="FareCat", y="Survived",hue='Title', data=train1)


# Bravo!! Mrs. Misss. and female Dr. survived even in zero fare category. However Male Survived in higher Fare category only..

# In[45]:


# Creating FamlSize Feature, since Very big family dint survive as per data
def FamlSize(data):
    data['FamlSize'] = 0
    data['FamlSize'] = data['SibSp'] + data['Parch'] + 1
FamlSize(df)


# In[46]:


df.head(1)


# ### Lable Encoding Categorical Data

# In[47]:


def LablFunc(data):
    lsr = {'Title','Cabin'}
    for i in lsr:
        le.fit(data[i].astype(str))
        data[i] = le.transform(data[i].astype(str))
LablFunc(df)


# ## Fill missing Age

# In[52]:


## Lets predict the age of a person and fill the missing Age
features = ['Pclass','SibSp','Parch','TicketId','Fare','CabinNum','Title']
from sklearn.ensemble import ExtraTreesRegressor as ETRg
def AgeFunc(df):
    Etr = ETRg(n_estimators = 200, random_state = 2)
    AgeX_Train = df[features][df.Age.notnull()]
    AgeY_Train = df['Age'][df.Age.notnull()]
    AgeX_Test = df[features][df.Age.isnull()]
    
    Etr.fit(AgeX_Train,np.ravel(AgeY_Train))
    AgePred = Etr.predict(AgeX_Test)
    df.loc[df.Age.isnull(), 'Age'] = AgePred
    
AgeFunc(df)


# In[53]:


# Lets derive AgeGroup feature from age
def AgeCat(data):
    data['AgeCat'] = 0
    data.loc[(data['Age'] <= 5), 'AgeCat'] = 0
    data.loc[(data['Age'] <= 12) & (data['Age'] > 5), 'AgeCat'] = 1
    data.loc[(data['Age'] <= 18) & (data['Age'] > 12), 'AgeCat'] = 2
    data.loc[(data['Age'] <= 22) & (data['Age'] > 18), 'AgeCat'] = 3
    data.loc[(data['Age'] <= 32) & (data['Age'] > 22), 'AgeCat'] = 4
    data.loc[(data['Age'] <= 45) & (data['Age'] > 32), 'AgeCat'] = 5
    data.loc[(data['Age'] <= 60) & (data['Age'] > 45), 'AgeCat'] = 6
    data.loc[(data['Age'] <= 70) & (data['Age'] > 60), 'AgeCat'] = 7
    data.loc[(data['Age'] > 70), 'AgeCat'] = 8
AgeCat(df)


# ## Lets check which Fare class Survived along with their title

# In[54]:


train1 = df[0:891].copy()
sns.set(style="whitegrid")
plt.figure(figsize=(14,3.5))
ax = sns.barplot(x="AgeCat", y="Survived",hue='Sex', data=train1)


# ### Finding out missing values in Embarked column

# In[56]:


df.loc[df['Embarked'].isnull()]


# ### Lets Check first from where 1st Class passesnger Came

# In[58]:


sns.set(style="whitegrid")
plt.figure(figsize=(12,2))
ax = sns.barplot(x="Embarked", y="Survived",hue='Pclass', data=df)


# #### from 'C' high number of 1st Pclass people Survived, lets fill 'C' in missing value

# In[59]:


def FillEmbk(data):
    var = 'Embarked'
    data.loc[(data.Embarked.isnull()),'Embarked']= 'C'
FillEmbk(df)


# ### Lable Encoding

# In[60]:


# Label Encode Embarked
def LablFunc(data):
    lst = {'Embarked','Sex'}
    for i in lst:
        le.fit(data[i].astype(str))
        data[i] = le.transform(data[i].astype(str))
LablFunc(df)


# In[61]:


df.columns


# ### Scaling up data

# In[62]:


from sklearn.preprocessing import StandardScaler
target = data['Survived'].values
select_features = ['Pclass', 'Age','AgeCat','SibSp', 'Parch', 'Fare', 
                   'Embarked', 'TicketId', 'CabinNum', 'Title','Cabin',
                   'FareCat', 'FamlSize','FamilySurv','Sex']
scaler = StandardScaler()
dfScaled = scaler.fit_transform(df[select_features])
train = dfScaled[0:891].copy()
test = dfScaled[891:].copy()


# ### Checking best features

# In[63]:


from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, len(select_features))
selector.fit(train, target)
scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]

print('Features importance:')
for i in range(len(scores)):
    print('%.2f %s' % (scores[indices[i]], select_features[indices[i]]))


# ### Define Feture importance function

# In[64]:


def FeatFunc(t_data,model):
    names = t_data.columns.values
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), names), 
                 reverse=True))


# ### Importing RandomForestClassifier

# In[67]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


# ### Initializing Model and Training the the Model

# In[68]:


SrchRFC = RandomForestClassifier(max_depth = 5, min_samples_split = 4, n_estimators = 500,
                                 random_state = 20, n_jobs = -1)          # Initialize
SrchRFC.fit(train, target)          # Train


# ### Predict the values on Train data to check Accuracy Score

# In[69]:


prc = SrchRFC.predict(train)
accuracy_score(target,prc)


# ### Let's Predict the outcome on Test data

# In[70]:


prdt2 = SrchRFC.predict(test)   #using Random Forest Classifier
print('Predicted result: ', prdt2)


# #### Submit the file to competition

# In[71]:


sampl['Survived'] = pd.DataFrame(prdt2)
sampl.to_csv('submission.csv', index=False)


# In[ ]:




