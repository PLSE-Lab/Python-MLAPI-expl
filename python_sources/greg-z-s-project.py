#!/usr/bin/env python
# coding: utf-8

# Let us load the datasets. Since we are going to apply changes to both training and testing datasets, it also helps to create a collection, *df_all*

# In[239]:


#
# Code by Gregory Zabrodskiy & Poome 
#
#


# loading nessasary packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Common Model Algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from collections import Counter

from sklearn.naive_bayes import GaussianNB
from sklearn import svm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#from sklearn.naive_bayes import GaussianNB

#from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
#from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import LabelEncoder
#from sklearn import feature_selection
#from sklearn import model_selection
from sklearn import metrics


#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,12


# In[240]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_all = [df_train, df_test]


# **Section I. Preparing the data**
# 
# 
# Before proceeding with any predictions, we will analyze and augment the data. We will complete the following steps:
# 1. Filling the data gaps
# 2. Creating range (bin) groups for certain features and encoding features
# 3. Engineering new features
# 4. Dropping unnecessary colums
# 
# Even on this step, we need to familiarize ourselves with the data. After reviewing the data in the next section, we came back and made some corrections, such as changing the bins of Age and Family groups.
# We also decided against using some additional derived features, such as title, because they seem highly correlated with other features and don't bring any additional benefits.

# 1. Let us now look at the data gaps. We can see that most values are missing in the Age and Cabin categories. We can guess that, at least, Age is very relevant to the prediction. 
# 

# In[241]:


for df in df_all:
    print(df.isna().sum())


# a. Embarkment 
# 
# We fill the value with a mode-0 value of the field.
# 
# b. Age/TitleType
# 
# Before filling Age, we create a new feature, TitleType focusing on several special cases:
# 
#     1. Unmarried women (Miss. or Mlle.)
#     
#     2. Young men (Master.)
#     
#     3. Professionals (Col, Major, Capt, Dr, Rev)
#     
#     4. Married women (Mrs.)
#     
#     0. Everyone else 
#     
# Now, we will fill the missing Age value with a median per Sex and TitleType.
# 
# c. Fare
# 
# We replace the missing values with a median per group (Pclass)
# 
# Looking at the data, we noticed that 15 passengers have $Fare = 0$ (including 5 1st class passengers and 6 2nd class passengers), which we initially interprested as missing data. However, upon further review, we noticed that only one of those passengers survived, making this value an important predictor.

# In[242]:


def titleType(row):
    if 'Miss.' in row['Name']:
        return 1
    elif 'Mlle.' in row['Name']:
        return 1
    elif 'Master.' in row['Name']:
        return 2
    elif 'Rev.' in row['Name']:
        return 3
    elif 'Rev.' in row['Name']:
        return 3
    elif 'Col.' in row['Name']:
        return 3
    elif 'Capt.' in row['Name']:
        return 3
    elif 'Major.' in row['Name']:
        return 3
    elif 'Dr.' in row['Name']:
        return 3   
    elif 'Mrs.' in row['Name']:
        return 4
    else:
        return 0 #others
for df in df_all:
    df['TitleType'] = df.apply(titleType, axis=1)


# In[243]:




for df in df_all:   
    
    #df['Embarked'].fillna('S', inplace = True)
    df['Fare'] = df[['Fare', 'Pclass']].groupby('Pclass').transform(lambda x: x.fillna(x.median()))    
    #df['Age'] = df[['Age', 'Sex', 'Pclass', 'Parch']].groupby(['Sex', 'Pclass', 'Parch']).transform(lambda x: x.fillna(x.mean()))
    df['Age'] = df[['Age', 'Sex', 'TitleType']].groupby(['Sex', 'TitleType']).transform(lambda x: x.fillna(x.median()))
    df['Age'].fillna(df['Age'].median(), inplace = True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)


# Cabin - we replace Cabin value with the first letter and set missing values to U. As we are going to see later, except for the 'Unknown' category, Cabin does not change the chances of survival significantly (over Pclass), therefore we will exclude it from the final list of the features.

# In[244]:


for df in df_all:
    df['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'U' for i in df['Cabin'] ])
    df['Cabin'] = LabelEncoder().fit_transform(df['Cabin'])


# 2. We would like to group Fare and Age values into bins. 
# 
# We split Fare into 5 equal-sized buckets based on sample quantiles (FareGroup)
# 
# We split Age into three buckets (child, teen, adult) 
# 
# We encode these two new features as well as Sex and Embarked features

# In[245]:


def ageGroup(row):
    if np.isnan(row['Age']):
        return -1 # unknown
    elif row['Age'] < 12:
        return 0 # child
    elif row ['Age'] < 20:
        return 1 # teen
    else:
        return 2 # adult 

    
for df in df_all:
    
    df['AgeGroup'] = df.apply(ageGroup, axis=1)
    #df['AgeGroup'] = pd.qcut(df['Age'], 4)
    #df['AgeGroup'] = LabelEncoder().fit_transform(df['AgeGroup'])
    df['FareGroup'] = pd.qcut(df['Fare'], 5)
    df['FareGroup'] = LabelEncoder().fit_transform(df['FareGroup'])
    df.loc[df.Fare == 0, 'FareGroup'] = -1   
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    



# 4. In addition to AgeGroup and FareGroup, we can consider several additional features.
# 
# a. FamilyGroup/isAlone
# 
# We can theorize that the survival depends on the family size, which we define as number of siblings and parents on board. Let's create a corresponding feature. Note, that the feature does not fully represent the passenger grouping, as friends, servants, fiances, etc. are not captured by this feature.Upon review the survival dependence on this feature, we noticed that people who travelled alone were at disadvantage, so it seemed for large families. However the number of large families was relatively small, and we decided to introduce another feature, isAlone, that returned 1 if sum of Parch and SibSp was > 0 and 0 otherwise.
# 
# b. Title
# 
# Several people introduced Title feature based on the title appearing in the name. However, upon looking at the data, we did not see any particular way that the title brings any more information above Sex, Pclass, Fare features. One particilar title, Rev. yields in zero survaval rate, however it only belongs to 2 passengers in the train data and 1 passenger in the test data, therefore we decided against using it.
# 
# c. Ticket Frequency
# 
# While ticket number appears meaningless, we can look at how many people share the same ticket number, as an alternative for estimating the party size (see FamilyGroup). Perhaps, this can be added to the next version of this program.

# In[246]:


def familyGroup(row):
    if row['SibSp'] + row['Parch'] == 0:
        return 0 # alone
    elif row['SibSp'] + row['Parch']  < 3:
        return 1 # small
    else:
        return 2 # large
    #return row['SibSp'] + row['Parch'] 

def isAlone(row):
    return 1 if row['SibSp'] + row['Parch'] == 0 else 0
    
for df in df_all:
    df['FamilyGroup'] = df.apply(familyGroup, axis=1)
#    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    df['isAlone'] = df.apply(isAlone, axis=1)


# 4. Now we can drop the unneeded columns
# 
# Note that we keep passengerID column in our test dataset to identify the passenger in our submission file.

# In[247]:


for df in df_all:
    df.drop(['Name', 'Ticket'], axis=1, inplace = True)
#df_train.drop('PassengerId', axis=1, inplace = True)

df_train.describe()


# In[248]:


print(df_test.isnull().sum())
df_test.describe()


# **Section II. Analyzing the data**
# 
# We first look at the correlation among the columns

# In[249]:


sns.heatmap(df_train.corr(), cmap =  sns.diverging_palette(255, 0), annot=True, fmt='0.2f', linewidths=0.1,vmax=1.0, linecolor='white', annot_kws={'fontsize': 7 })


# We can see that, as we could have predicted, the driving features of the survival include Sex, Pclass, FareGroup (highly correlated with Pclass), and Age. Port of Embarkment also correlates with survival.
# 
# Now, let us look at the distribution of survaved passengers among different feature groups

# In[250]:


for col in df_train:
    if df_train[col].dtype == 'int64':
        if col not in ['Survived', 'PassengerId']:
            print(df_train[[col, 'Survived']].groupby(col).agg(['mean', 'count']))


# Let's look at the results graphically

# In[251]:


g = sns.catplot(x="Pclass",y="Survived",data=df_train,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[252]:


g = sns.catplot(x="Sex",y="Survived",data=df_train,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[253]:


g = sns.catplot(x="Embarked",y="Survived",data=df_train,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[254]:


g = sns.catplot(x="AgeGroup",y="Survived",data=df_train,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[255]:


g = sns.catplot(x="FamilyGroup",y="Survived",data=df_train,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[256]:


g = sns.catplot(x="FareGroup",y="Survived",data=df_train,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[257]:


g = sns.catplot(x="TitleType",y="Survived",data=df_train,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[258]:


g = sns.catplot(x="isAlone",y="Survived",data=df_train,kind="bar", height = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[259]:


g = sns.catplot(x="Cabin",y="Survived",data=df_train,kind="bar", height = 6 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# We now look at the distributions of Age and Fare to confirm that the survaval for certain ranges differ from the others. 

# In[260]:


g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 0) & (df_train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 1) & (df_train["Age"].notnull())], ax =g, color="Green", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# In[261]:


g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 0) & (df_train["Fare"].notnull())], color="Red", shade = True)
g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 1) & (df_train["Fare"].notnull())], ax =g, color="Green", shade= True)
g.set_xlabel("Fare")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# Port of Embarkment correlation with survival is a curious one. We can try to explain it by the fact that its correlation with Pclass and Sex are of the same magnitude. 
# However, looking at the distribution per Pclass, Sex, and Embarked, we can see that for the 3rd class passengers (especially for male), port of Embarkment matters.

# In[262]:


print(df_train[['Pclass', 'Sex', 'Embarked', 'Survived']].groupby(['Pclass', 'Sex', 'Embarked']).agg(['mean', 'count']))


# Dependency on belonging to a family is also a curious one. Some sample groups are too small to make conclusions, but clearly single males are at large disadvantage, and belonging to a large family is also reduces the odds. 

# In[263]:


print(df_train[['Pclass', 'Sex', 'FamilyGroup', 'Survived']].groupby(['Pclass', 'Sex', 'FamilyGroup']).agg(['mean', 'count']))


# In[264]:


print(df_train[['Pclass', 'Fare', 'Survived']].groupby(['Pclass', 'Survived']).agg(['mean', 'median', 'min', 'max', 'count']))


# Section III. Modeling 
# 
# 

# Since we don't have Survaved value in the test file, we will create the model and test it by splitting the train file.

# Considering that different factors drive the survival odds of different subgroups of the passengers, using a decision tree algorithms appears a logical choice for our prediction. We chose the RandomForest classifier that combines a power of a stochastic model with a decision tree methodology.

# In[265]:


dt = tree.DecisionTreeClassifier()
rf = RandomForestClassifier(criterion='gini', n_estimators=1000, random_state=1000)

nb = GaussianNB()
svm = svm.SVC(gamma='scale')

vc = VotingClassifier(estimators=[('nb', nb), ('rf', rf), ('svm', svm)], voting='hard')


# We use 30% of the data in the train file for testing, which is close to the relative size of the actual test file.

# In[266]:


x = df_train.drop(columns=['Survived', 'PassengerId'])
y = df_train['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1000)


# Let us first consider a model trained with only two features, Sex and Fare

# In[267]:


x_train1 = x_train[['Sex', 'Fare']]
x_test1 = x_test[['Sex', 'Fare']]
y_pred = dt.fit(x_train1, y_train).predict(x_test1)
print('Accuracy (DT): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))

y_pred = rf.fit(x_train1, y_train).predict(x_test1)
print('Accuracy (RF): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))


# The accuracy of the Random Forest Classifier is 83%, which looks very reasonable given such a small number of features.

# In[268]:


x_train1 = x_train[['Sex', 'Fare', 'Age']]
x_test1 = x_test[['Sex',  'Fare', 'Age']]
y_pred = dt.fit(x_train1, y_train).predict(x_test1)
print('Accuracy (DT): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))

y_pred = rf.fit(x_train1, y_train).predict(x_test1)
print('Accuracy (RF): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))


# Adding Age to the mix slightly reduces accuracy to under 80%

# No we run the final set of features, using an extended set of classifiers. RandonForest still performs slightly better.

# In[269]:


final_features = ['Sex', 'FareGroup', 'Pclass', 'Embarked', 'AgeGroup', 'isAlone']

x_train1 = x_train[final_features]
x_test1 = x_test[final_features]

y_pred = nb.fit(x_train1, y_train).predict(x_test1)
print('Accuracy (NB): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))

y_pred = svm.fit(x_train1, y_train).predict(x_test1)
print('Accuracy (SVM): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))


y_pred = dt.fit(x_train1, y_train).predict(x_test1)
print('Accuracy (DT): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))

y_pred = vc.fit(x_train1, y_train).predict(x_test1)
print('Accuracy (VC): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))


y_pred = rf.fit(x_train1, y_train).predict(x_test1)
print('Accuracy (RF): {}'.format(accuracy_score(y_test, y_pred, normalize=True)))


# while adding Pclass, AgeGroup, Embarked  and isAlone increases it to over 84%

# In[270]:


cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index = [i for i in ['not survived', 'survived']], columns = [i for i in ['not survived', 'survived']])
plt.figure(figsize = (3,3))
a = sns.heatmap(cm, annot=True, fmt='g')
a.set (ylabel='True label', xlabel='Predicted label')
plt.setp(a.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
plt.setp(a.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


# [](http://)

# Therefore, we will run our model with the following five features:
# - Sex
# - FareGroup
# - AgeGroup
# - Pclass
# - Embarked
# - isAlone

# Section V. Producing output

# Now let's use the full train set for training and predict the test set results

# In[271]:


x_train = df_train[final_features]
y_train = df_train['Survived']
x_test = df_test[final_features]


y_pred = rf.fit(x_train, y_train).predict(x_test)


df_feature_importance = pd.DataFrame()
df_feature_importance['feature'] = x_train.columns
df_feature_importance['importance'] = rf.feature_importances_

plt.figure(figsize=(5, 5))
sns.barplot(x='importance', y='feature', data=df_feature_importance.sort_values(by='importance', ascending=False))
plt.show()


# In[272]:


sdf = pd.DataFrame(columns=['PassengerId', 'Survived'])
sdf['PassengerId'] = df_test['PassengerId']
sdf['Survived'] = y_pred
sdf.to_csv('submissions.csv', header=True, index=False)
print(sdf.head(10))
print(sdf.groupby('Survived').count())

