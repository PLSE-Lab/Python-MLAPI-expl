#!/usr/bin/env python
# coding: utf-8

# # Titanic Data

# ## Index
# - [1. Import libraries and download data](#section1)
# - [2. EDA](#section2)
# - [3. Cleaning Data](#section3)
# - [4. Data Engineering](#section4)
# - [5. Modelling](#section5)
# 

# ## 1. Import libraries and download data<a id='section1'></a>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud


# In[ ]:


path = '/kaggle/input/titanic/'

train = pd.read_csv(path + 'train.csv', sep=",")
test = pd.read_csv(path + "test.csv", sep=",")
test_sub = test.copy()


# ## 2. EDA<a id='section2'></a>
# We are going to study two dataframes _train_ and _test_.
# 
# First, we apply info to both of dataframes, in order to see the features and missing values.

# - train set

# In[ ]:


print('Shape:', train.shape)
train.info()


# - test set

# In[ ]:


print('Shape:', test.shape)
test.info()


# Observing the data before, we can say the following:
# - the __train__ set is composed by 12 features and 891 cases. These features are different type of data, such as integer, float, string, therefore we must homogeneous them, in order to build the model. Some of these features contain missing values: _Age, Cabin_ and _Embarked_.
# - the __test__ set is formed by 11 features (not contains the target) and 418 cases. And there are also some features with missing values: _Age, Fare_ and _Cabin_. 

# Analysing the target of this data, which is _Survived_ feature, we can say that the majority of people did not survived. Talking with probabilities,
# 
# P(survived) = 0.384
# 
# P(no\_survived) =0.616
# 

# In[ ]:


train['Survived'].value_counts()


# ### 2.1 Features vs target
# Following with the analysis, we are going to try to find some relationship between the target and the others features.

# In[ ]:


class Plot_class():

    def __init__(self,feature, my_dataframe, my_table):
        self.feature = feature
        self.my_dataframe = my_dataframe
        
        
    def plot_bar(feature, my_dataframe):
        fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(12,4))
        my_dataframe.groupby([feature,'Survived'])[feature].count().unstack().plot(kind='bar',stacked=True, ax=axes)
        plt.title('Frequency of {} feature vs  survived (target)'.format(feature))

    def plot_bar_table(feature, my_dataframe, my_table):
        fig = plt.figure()
        # definitions for the axes
        left, width = 0.10, 1.5
        bottom, height = 0.1, .8
        bottom_h = left_h = left + width + 0.02

        rect_cones = [left, bottom, width, height]
        rect_box = [left_h, bottom, 0.17, height]
        
        # plot
        ax1 = plt.axes(rect_cones)
        my_dataframe.groupby([feature,'Survived'])[feature].count().unstack().plot(kind='bar',stacked=True, ax=ax1)
        plt.title('Frequency of {} feature vs  survived (target)'.format(feature))
        
        # Table
        ax2 = plt.axes(rect_box)
        my_table = ax2.table(cellText = table_data, loc ='right')
        my_table.set_fontsize(40)
        my_table.scale(4,4)
        ax2.axis('off')
        plt.show()
    
    def distri(feature, my_dataframe):
        fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(12,5))
        ax = sns.distplot(train.loc[(train['Survived'] == 0),feature].dropna(),color='orange',bins=40)
        ax = sns.distplot(train.loc[(train['Survived'] == 1),feature].dropna(),color='blue',bins=40)
        plt.legend(labels=['not survived','survived'])
        plt.title('{} Distribution Survived vs Non Survived'.format(feature))
        plt.ylabel("Frequency of Passenger Survived")
        plt.xlabel(feature)
        plt.show()
   


# - Sex

# In[ ]:


#values
Total = train.Survived.count()
Female = train[train['Sex'] == 'female'].Survived.count()
Male = train[train['Sex'] == 'male'].Survived.count()
P_Female = round(Female / Total,2)
P_Male = round(Male / Total,2)
P_Female_and_Survived = round(train[((train['Sex']=='female') & (train['Survived']==1))].Survived.count() / Total, 2)
P_Male_and_Survived = round(train[((train['Sex']=='male') & (train['Survived']==1))].Survived.count() / Total, 2)
P_Survived = round(train[train['Survived'] == 1].Survived.count()/Total, 2)
P_Survived_Female = round(P_Female_and_Survived / P_Female,2) #P(survived|female)
P_Survived_Male = round(P_Male_and_Survived / P_Male, 2) #P(survived|male)
P_Female_Survived = round(P_Female_and_Survived / P_Survived,2) #P(Female | Survived)
P_Male_Survived = round(P_Male_and_Survived / P_Survived,2) #P(Male | Survived)

table_data=[
    ["P(Female)", P_Female],
    ["P(Male)", P_Male],
    ["P(Survived | Female)", P_Survived_Female ],
    ["P(Survived | Male)", P_Survived_Male ],
    ["P(Female | Survived)", P_Female_Survived],
    ["P(Male | Survived)", P_Male_Survived ]
]


# In[ ]:


Plot_class.plot_bar_table('Sex',train,table_data)


# - Class 

# In[ ]:


#values
P_First = round(train[train['Pclass'] == 1].Survived.count() / Total, 2)
P_Middle = round(train[train['Pclass'] == 2].Survived.count() / Total, 2)
P_Third = round(train[train['Pclass'] == 3].Survived.count() / Total, 2)
P_First_and_Survived = round(train[((train['Pclass']==1) & (train['Survived']==1))].Survived.count() / Total, 2)#P(first and survived)
P_Middle_and_Survived = round(train[((train['Pclass']==2) & (train['Survived']==1))].Survived.count() / Total, 2)#P(middle and survived)
P_Third_and_Survived = round(train[((train['Pclass']==3) & (train['Survived']==1))].Survived.count() / Total, 2)#P(third and survived)
P_First_Survived = round(P_First_and_Survived / P_Survived, 2) #P(first | survived)
P_Middle_Survived = round(P_Middle_and_Survived / P_Survived, 2) #P(middle | survived)
P_Third_Survived = round(P_Third_and_Survived / P_Survived, 2) #P(first | survived)
P_Survived_First = round(P_First_and_Survived / P_First, 2) #P(Survived | First)
P_Survived_Middle = round(P_First_and_Survived / P_Middle, 2) #P(survived | middle)
P_Survived_Third = round(P_First_and_Survived / P_Third, 2) #P(survived | third)

table_data = [
    ["P(First)", P_First],
    ["P(Middle)", P_Middle],
    ["P(Third)", P_Third],
    ["P(First | Survived)", P_First_Survived],
    ["P(Middle | Survived)", P_Middle_Survived],
    ["P(Third | Survived)", P_Third_Survived],
    ["P(Survived | First)", P_Survived_First],
    ["P(Survived | Middle)", P_Survived_Middle],
    ["P(Survived | Third)", P_Survived_Third]
]


# In[ ]:


Plot_class.plot_bar_table('Pclass', train, table_data)


# - Embarked

# In[ ]:


#values
P_C = round(train[train['Embarked'] == 'C'].Survived.count() / Total, 2)
P_Q = round(train[train['Embarked'] == 'Q'].Survived.count() / Total, 2)
P_S = round(train[train['Embarked'] == 'S'].Survived.count() / Total, 2)
P_C_and_Survived = round(train[((train['Embarked']=='C') & (train['Survived']==1))].Survived.count() / Total, 2)#P(first and survived)
P_Q_and_Survived = round(train[((train['Embarked']=='Q') & (train['Survived']==1))].Survived.count() / Total, 2)#P(middle and survived)
P_S_and_Survived = round(train[((train['Embarked']=='S') & (train['Survived']==1))].Survived.count() / Total, 2)#P(third and survived)
P_C_Survived = round(P_C_and_Survived / P_Survived, 2) #P(first | survived)
P_Q_Survived = round(P_Q_and_Survived / P_Survived, 2) #P(middle | survived)
P_S_Survived = round(P_S_and_Survived / P_Survived, 2) #P(first | survived)
P_Survived_C = round(P_C_and_Survived / P_C, 2) #P(Survived | First)
P_Survived_Q = round(P_Q_and_Survived / P_Q, 2) #P(survived | middle)
P_Survived_S = round(P_S_and_Survived / P_S, 2) #P(survived | third)

table_data = [
    ["P(C)", P_C],
    ["P(Q)", P_Q],
    ["P(S)", P_S],
    ["P(C | Survived)", P_C_Survived],
    ["P(Q | Survived)", P_Q_Survived],
    ["P(S | Survived)", P_S_Survived],
    ["P(Survived | C)", P_Survived_C],
    ["P(Survived | Q)", P_Survived_Q],
    ["P(Survived | S)", P_Survived_S]
]


# In[ ]:


Plot_class.plot_bar_table('Embarked', train, table_data)


# - Cabin

# In[ ]:


train['Cabin_'] = train['Cabin'].astype(str).str[0]
train['Cabin_'] = train['Cabin_'].replace({'n':'No_value'})
Plot_class.plot_bar('Cabin_', train)


# - SibSp

# In[ ]:


Plot_class.plot_bar('SibSp', train)


# - Parch

# In[ ]:


Plot_class.plot_bar('Parch',train)


# - People's title

# In[ ]:


train['Title'] = train['Name'].str.replace('(.*, )|(\..*)',"")
Plot_class.plot_bar('Title', train)
# although test is not plotting we create the featue
test['Title'] = test['Name'].str.replace('(.*, )|(\..*)',"")


# - Fare

# In[ ]:


Plot_class.distri('Fare',train)


# - Age

# In[ ]:


Plot_class.distri('Age',train)


# - Correlations between features

# In[ ]:


#f, ax = plt.subplots(figsize=(10, 8))
#Firstly, sex feature change numerical 0 for male and 1 for female
train_corr = train.replace({'Sex':{'male': 0, 'female':1}})
train_corr = train_corr.replace({'Embarked':{'C': 0, 'Q': 1 ,'S':2}})
corr=train_corr[['Survived','Sex', 'Pclass','Embarked','Age', 'SibSp', 'Parch', 'Fare']].corr()
#train_corr.corr()
corr.style.background_gradient().set_precision(2)


# ### 2.2  Visiualising some correlations 

# In[ ]:


d = {'color': ['orange', 'b']}
g = sns.FacetGrid(train, col='Embarked')
g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
g.add_legend()


# In[ ]:


d = {'color': ['orange', 'b']}
g = sns.FacetGrid(train, row='Sex', col='Survived', hue_kws=d, hue='Survived')
g.map(plt.hist, 'Age')
g.add_legend()


# In[ ]:


d = {'color': ['orange', 'b']}
g = sns.FacetGrid(train, row='Sex', col='Survived', hue_kws=d, hue='Survived')
g.map(plt.hist, 'Fare', bins=20)
g.add_legend()


# ## 3. Cleaning Data<a id='section3'></a>
# In this section, we are going to work with the features that contain missing values, in both datasets.

# ### 3.1 Checking missing values

# - train dataset

# In[ ]:


# Filling empty and NaNs values with NaN
train = train.fillna(np.nan)
# Checking for Null values
train.isnull().sum()


# - test dataset

# In[ ]:


# Filling empty and NaNs values with NaN
test = test.fillna(np.nan)
# Checking for Null values
test.isnull().sum()


# Looking at the information above, we need to fill in three features: _Age_ , _Cabin_ and _Embarked_ .
# 
# - Embarked
# 
# We start for the easiest feature because there are only two missing values and we will fill in with the mode value, which is _S_ , how is represented by the plot of frequency of _Embarked_ feature.

# In[ ]:


train['Embarked'].fillna(train.Embarked.mode()[0], inplace = True)


# - Fare
# 
# There is only one missing value in test dataframe, so we are going to fill with the mean value of Fare.

# In[ ]:


test['Fare'].fillna(test.Fare.mean(), inplace = True)


# - Cabin 
# 
# We are goint to eleminate this feature in both sets, because the percentage of missing values is really high and the train plot also does not show difference survive or not between different cabins.
# 

# In[ ]:


train = train.drop(['Cabin','Cabin_'], axis=1)
test = test.drop(['Cabin'], axis=1)


# - Age
# 
# In order to fill the Nan, we are going to obrserve the relationship between age and three features (SibSp, Title, Pclass) becasue it can give us a hint.
# 
# Firstly, we would like to group the Title feature in Mr, Mrs, Miss, Master, Dr, and Other.

# In[ ]:


test.Title.unique()


# In[ ]:


def transform_title(dataset):
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',                        'Don', 'Major', 'Rev', 'Sir', 'Jonkheer','the Countess', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[ ]:


transform_title(train)
transform_title(test)


# Secondly, we plot different boxplot for the features SibSp, Title, Pclass.

# In[ ]:


fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(12,4))
sns.boxplot(data=train, x = 'Title', y = 'Age', ax=axes[0,0])
sns.boxplot(data=train, x = 'SibSp', y = 'Age', ax=axes[0,1])
sns.boxplot(data=train, x = 'Pclass', y = 'Age', ax=axes[0,2])
sns.boxplot(data=test, x = 'Title', y = 'Age', ax=axes[1,0])
sns.boxplot(data=test, x = 'SibSp', y = 'Age', ax=axes[1,1])
sns.boxplot(data=test, x = 'Pclass', y = 'Age', ax=axes[1,2])
fig.suptitle('Boxplot before filling missing values')
plt.show()


# In[ ]:


def filling_Age(dataset):
    dataset_aux = dataset.dropna()
    dataset_aux = dataset_aux.reset_index(drop=True)
    dataset_aux = dataset_aux.groupby(['Title','Pclass','SibSp'])['Age'].apply(lambda g: g.mean(skipna=True)).to_frame()
    
    aux = []

    for idx,row in dataset.iterrows():
        if row.isnull().sum() == 0:
            aux.append(dataset.loc[idx]['Age'])
        else:
            val_1 = dataset.loc[idx]['Title']
            val_2 = dataset.loc[idx]['Pclass']
            val_3 = dataset.loc[idx]['SibSp']
            if (val_1, val_2, val_3) in list(dataset_aux.index):
                val_sus = dataset_aux.loc[val_1, val_2, val_3][0]
                aux.append(val_sus)
            else:
                aux.append(dataset.Age.mean())
    
    dataset['Age']=aux


# In[ ]:


filling_Age(train)
filling_Age(test)


# In[ ]:


fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(12,4))
plt.subplots_adjust(hspace = 0.8)


sns.boxplot(data=train, x = 'Title', y = 'Age', ax=axes[0,0])
sns.boxplot(data=train, x = 'SibSp', y = 'Age', ax=axes[0,1])
sns.boxplot(data=train, x = 'Pclass', y = 'Age', ax=axes[0,2])
sns.boxplot(data=test, x = 'Title', y = 'Age', ax=axes[1,0])
sns.boxplot(data=test, x = 'SibSp', y = 'Age', ax=axes[1,1])
sns.boxplot(data=test, x = 'Pclass', y = 'Age', ax=axes[1,2])
fig.suptitle('Boxplot after filling missing values', fontsize=14)

plt.show()


# ## 4. Data Engineering<a id='section4'></a>

# Preparing the datasets in order to apply the models.
# 
# ### 4.1. Dropping features.
# 
# PassengerId, Name, Ticket

# In[ ]:


train = train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test = test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)


# ### 4.2 Converting features
# 
# #### 4.2.1.There are some features are string, and they have to become in numerical.
# 
# - Sex 
# 
# female = 1 and male = 0

# In[ ]:


train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# - Embarked
# 
# S = 0 , C = 1, Q = 2 

# In[ ]:


train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2 } ).astype(int)
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2 } ).astype(int)


# - Title
# 
# Mr = 1, Miss = 2, Mrs = 3, Master = 4, Dr = 5 and Other = 6

# In[ ]:


titles_numerics={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Other":6}
train['Title'] = train['Title'].map(titles_numerics).astype(int)
test['Title'] = test['Title'].map(titles_numerics).astype(int)


# #### 4.2.2. Numerical continues to categorical.
# We are going to create two functions, that they convert from continuous variable to categorical. 

# In[ ]:


def Ages_to_category(dataset):
    dataset_aux = dataset.copy()
    bins = [0, 3, 16, 25, 45, 60, dataset_aux.Age.max()]
    labels = ['Baby', 'Child', 'Young', 'Adult', 'Older Adult','Senior']
    dataset_aux['Age'] = pd.cut(dataset_aux['Age'], bins, labels = labels)
    Ages_numerics = {"Baby": 1, "Child": 2, "Young": 3, "Adult": 4, "Older Adult": 5, "Senior":6}
    dataset_aux['Age'] = dataset_aux['Age'].map(Ages_numerics).astype(int)
    return(dataset_aux)

def Fare_to_category(dataset):
    dataset_aux = dataset.copy()
    bins = [0, 50, 100, 150 , 200, 250, dataset_aux.Fare.max()]
    labels = [1, 2, 3, 4, 5, 6]
    dataset_aux['Fare'] = pd.cut(dataset_aux['Fare'], bins, labels = labels)
    dataset_aux['Fare'] = dataset_aux['Fare'].astype(int)
    return(dataset_aux)
  


# ### 4.3 Creating new features
# 
# - FamilySyze
# 
# We join two features SibSp and Parch. We are going to consider a person who travelled alone as 1.

# In[ ]:


train['FamilySize'] = train.SibSp + train.Parch + 1
test['FamilySize'] = test.SibSp + test.Parch + 1

train = train.drop(['SibSp', 'Parch'], axis=1)
test = test.drop(['SibSp', 'Parch'], axis=1)


# # 5. Modelling<a id='section5'></a>

# For modelling we are going to use two dataframes in order to see if the Age and Fare work better as categorical features or continuous features. Therefore train contains continuous and train_aux contains categorical variables.

# In[ ]:


######Prueba como funciona el modelo
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#making the dummy varaible of catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[ ]:


class Models:
    def __init__(self,dataset, Model_name, X_train, X_test, y_train, y_test):
        self.dataset = dataset
        self.Model_name = Model_name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
       
    def preparing_data(dataset):
        X = dataset.drop("Survived", axis=1)
        y = dataset["Survived"]
        X = preprocessing.StandardScaler().fit(X).transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)
        return(X_train, X_test, y_train, y_test)
    
    def one_hot_encoding(dataset):
        #OneHotEncoder
        OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
        object_cols  = ['Pclass', 'Embarked', 'Title', 'FamilySize']
        OH_cols_dataset = pd.DataFrame(OH_encoder.fit_transform(dataset[object_cols]))
        #Remove categorical columns(will replace with one hot encoding)
        dataset = dataset.drop(object_cols, axis = 1)
        dataset = pd.concat([dataset, OH_cols_dataset], axis=1)
        return(dataset)
    
    def fitting_Model(Model_name, X_train, y_train):
        if Model_name == DecisionTreeClassifier:
            decision_tree = DecisionTreeClassifier(max_depth=4)
            return(decision_tree.fit(X_train, y_train))
        elif Model_name == KNeighborsClassifier:
            knn = KNeighborsClassifier(n_neighbors = 4)
            return(knn.fit(X_train, y_train))
        elif Model_name == RandomForestClassifier:
            random_forest = RandomForestClassifier(n_estimators=100, max_depth=4)
            return(random_forest.fit(X_train, y_train))
        elif Model_name == LogisticRegression:
            lg = LogisticRegression(solver='lbfgs')
            return(lg.fit(X_train, y_train))
        else:
            model = Model_name()
            return(model.fit(X_train, y_train))
    
    def predicting_Model(Model_name, X_train, y_train, X_test):
        model = Models.fitting_Model(Model_name, X_train, y_train)
        return(model.predict(X_test))
    
    def score_model(Model_name, X_train, y_train, X_test, y_test):
        model = Models.fitting_Model(Model_name, X_train, y_train)
        y_pred = model.predict(X_test)    
        acc_model = round(model.score(X_train, y_train) * 100, 2)
        acc_test = round(metrics.accuracy_score(y_test, y_pred)*100,2)
        return(acc_model, acc_test)


# In[ ]:


X_train, X_test, y_train, y_test = Models.preparing_data(train)


# In[ ]:


acc_train, acc_test = Models.score_model(DecisionTreeClassifier, X_train, y_train, X_test, y_test)
print('- Decision_tree:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
acc_train, acc_test = Models.score_model(RandomForestClassifier, X_train, y_train, X_test, y_test)
print('- Random Forest:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
acc_train, acc_test = Models.score_model(SGDClassifier, X_train, y_train, X_test, y_test)
print('- SGD_classifier:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
acc_train, acc_test = Models.score_model(KNeighborsClassifier, X_train, y_train, X_test, y_test)
print('- KNN:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
acc_train, acc_test = Models.score_model(GaussianNB, X_train, y_train, X_test, y_test)
print('- Gaussian:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))


# In[ ]:


#converting Age and Fare features, from continuous to categorical.
train_aux = Ages_to_category(train)
train_aux = Fare_to_category(train_aux)


# In[ ]:


X_train_aux, X_test_aux, y_train_aux, y_test_aux = Models.preparing_data(train_aux)


# In[ ]:


acc_train, acc_test = Models.score_model(DecisionTreeClassifier, X_train_aux, y_train_aux, X_test_aux, y_test_aux)
print('- Decision_tree:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
acc_train, acc_test = Models.score_model(RandomForestClassifier, X_train_aux, y_train_aux, X_test_aux, y_test_aux)
print('- Random Forest:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
acc_train, acc_test = Models.score_model(SGDClassifier, X_train_aux, y_train_aux, X_test_aux, y_test_aux)
print('- SGD_classifier:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
acc_train, acc_test = Models.score_model(KNeighborsClassifier, X_train_aux, y_train_aux, X_test_aux, y_test_aux)
print('- KNN:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))
acc_train, acc_test = Models.score_model(GaussianNB, X_train_aux, y_train_aux, X_test_aux, y_test_aux)
print('- Gaussian:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))


# In[ ]:


# In order to use logistic regression, we applys hot encoding to Pclass, Embarked, Title, FamilySize features 
# We keep Age and Fare features as continuous features.
train_hot = Models.one_hot_encoding(train)
test_hot = Models.one_hot_encoding(test)


# In[ ]:


X_train_hot, X_test_hot, y_train_hot, y_test_hot = Models.preparing_data(train_hot)


# In[ ]:


acc_train, acc_test = Models.score_model(LogisticRegression, X_train_hot, y_train_hot,X_test_hot, y_test_hot)
print('- Logistic Regression:\n \t accuracy_train: {}\n \t accuracy_test:{}'.format(acc_train, acc_test))


# In[ ]:


#taking the LogisticRegression we have the follow prediction
test_hot = preprocessing.StandardScaler().fit(test_hot).transform(test_hot)


# In[ ]:


y_pred = Models.predicting_Model(LogisticRegression, X_train_hot, y_train_hot, test_hot)


# In[ ]:


gender_submission_LG = pd.DataFrame({
        "PassengerId": test_sub["PassengerId"],
        "Survived": y_pred
    })


# In[ ]:




