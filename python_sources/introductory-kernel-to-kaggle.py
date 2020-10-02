#!/usr/bin/env python
# coding: utf-8

# # Getting to know the data
# 
# Titanic data exploration and survival prediction is one of he most important step in kaggle's world. Titanic survival prediction is one of the longest running competition on kaggle and its a good one to start and get to know the kaggle competition environment and how it works. 
# The Data involves 2 data files one is training data file and another testing file. The training data file contains 13 columns(the testing data cntains 12 columns as we need to predict survival rate):-
# PassengerID-The id of the passenger
# Survived(target column)-column determining whether the person survived or not.(0 for not surviving and 1 for surviving)
# Pclass -Ticket class	(1 = 1st, 2 = 2nd, 3 = 3rd)
# Sex	-Sex of the passenegr
# Age	-Age of the passenger in years	
# sibsp	-Number of siblings / spouses aboard the Titanic	
# parch	-Number of parents / children aboard the Titanic	
# ticket	-Ticket number	
# fare	-Passenger fare	
# cabin	-Cabin number	
# embarked	-Port of Embarkation	(C = Cherbourg, Q = Queenstown, S = Southampton)

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
train = pd.read_csv("../input/train.csv")
holdout = pd.read_csv("../input/test.csv")
train.head()


# In[ ]:


holdout.head()


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


holdout.isnull().sum()


# # Cleaning and standardizing the data
# 
# Getting the data ready for analysis and extracting what informatin we can from the data visualization. Cleaning is a necessary step as empty values or NaN values in data set  can becme a outliner or result in errors respectively. Therefore, removal or adjustment of NaN or empty is important. This process can be done by the following ways:-
# By removal of all or few of the Nan values from the data
# Or by imputing the data with mean, mode or median 
# Or by random filling of the data.
# After taking care of the cleaning data, next step comes is the standardizing and aggregating the data to remove biasness towards a column in the data.
# Here we are filling the empty data with the mean and doing one hot encoding to all the columns to  convert them to binary data.

# In[ ]:


def process_missing(df):
    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df

def process_age(df):
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def process_fare(df):
    cut_points = [-1,12,50,100,1000]
    label_names = ["0-12","12-50","50-100","100+"]
    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)
    return df

def process_cabin(df):
    df["Cabin_type"] = df["Cabin"].str[0]
    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")
    df = df.drop('Cabin',axis=1)
    return df

def process_titles(df):
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }
    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df


# In[ ]:


def pre_process(df):
    df = process_missing(df)
    df = process_age(df)
    df = process_fare(df)
    df = process_titles(df)
    df = process_cabin(df)

    for col in ["Age_categories","Fare_categories",
                "Title","Cabin_type","Sex"]:
        df = create_dummies(df,col)
    
    return df

train = pre_process(train)
holdout = pre_process(holdout)
train.head(2)


# # Data Exploration and Analysis
# Using the pivot_table method of the pandas library the reduce and condense the data for better data exploration and data insights.

# In[ ]:


dff=train.groupby('Pclass')
x=train.pivot_table(index='Pclass',values=['Fare','Age','Survived'])
x


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
explore_cols = ["SibSp","Parch","Survived"]
explore = train[explore_cols].copy()
explore["familysize"] = explore[["SibSp","Parch"]].sum(axis=1)
explore.drop("Survived",axis=1).plot.hist(alpha=0.5,bins=10)
plt.xticks(range(11))
plt.show()


# Visualizing the survival rate across a family by mapping Survival column of the data against number of siblings, number of parents and the total family size.

# In[ ]:


for col in explore.columns.drop("Survived"):
    pivot = explore.pivot_table(index=col,values="Survived")
    pivot.plot.bar(ylim=(0,1),yticks=np.arange(0,1,.1))
    plt.show()


# In[ ]:


plt.scatter(train['Fare'],train['Age'],c=train['Survived'])
plt.legend()


# In[ ]:


pivot = train.pivot_table(index='Survived',values="Fare")
pivot.plot.bar(color="lightgreen")
plt.show()


# In[ ]:


pivot = train.pivot_table(index='Embarked',values="Survived")
pivot.plot.bar()
plt.show()


# In[ ]:


x=x.drop(['Survived'],axis=1)
x.plot(kind='bar')


# # Feature Selection
# Here the sklearn library method RFECV is used - Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.

# In[ ]:


def process_isalone(df):
    df["familysize"] = df[["SibSp","Parch"]].sum(axis=1)
    df["isalone"] = 0
    df.loc[(df["familysize"] == 0),"isalone"] = 1
    df = df.drop("familysize",axis=1)
    return df

train = process_isalone(train)
holdout = process_isalone(holdout)
corr_matrix=train.corr()
corr_matrix.shape


# In[ ]:


abs_matrix=corr_matrix['Survived'].abs()
sorted_matrix=abs_matrix[abs_matrix>=0.2]
indexed_matrix=sorted_matrix.index
indexed_matrix


# In[ ]:


temp_df=train[indexed_matrix]
corr_temp_df=temp_df.corr()
import seaborn as sns
sns.heatmap(corr_temp_df,annot=True)


# In[ ]:


def select_features(df):
    df = df.select_dtypes([np.number]).dropna(axis=1)
    all_X = df.drop(["Survived","PassengerId"],axis=1)
    all_y = df["Survived"]
    
    clf = LogisticRegression(random_state=4)
    selector = RFECV(clf,cv=10)
    selector.fit(all_X,all_y)
    
    best_columns = list(all_X.columns[selector.support_])
    print("Best Columns \n"+"-"*12+"\n{}\n".format(best_columns))
    
    return best_columns

cols = select_features(train)


# In[ ]:


sns.set(style='white')
mask=np.zeros_like(train[cols].corr(),dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
f,ax=plt.subplots(figsize=(11,9))
cmap=sns.diverging_palette(220,10,as_cmap=True)
sns.heatmap(train[cols].corr(),mask=mask,cmap=cmap,annot=True)


# GridSearchCV is a very efficient method to use multiple machine learning algorithms against each other and check there accuracy for the best algo.

# In[ ]:


from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import warnings
#warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
def select_model(df,features):
    
    all_X = df[features]
    all_y = df["Survived"]
    models = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(),
            "hyperparameters":
                {
                    "solver": ["newton-cg", "lbfgs", "liblinear"]
                }
        },
        {
            "name": "SupportVectorMachine",
            "estimator": SVC(),
            "hyperparameters":
            {
                "kernel":["linear", "rbf","poly","sigmoid"]
            }
        },
        {
            "name": "RandomForestClassifier",
            "estimator": RandomForestClassifier(random_state=1),
            "hyperparameters":
                {
                    "n_estimators": [5, 10, 14,15,20],
                    "criterion": ["entropy", "gini"],
                    "max_depth": [10,15,20],
                    "max_features": ["log2", "sqrt","auto"],
                    "min_samples_leaf": [1, 3,4,5]
                }
        }
    ]

    for model in models:
        print(model['name'])
        print('-'*len(model['name']))

        grid = GridSearchCV(model["estimator"],
                            param_grid=model["hyperparameters"],
                            cv=10)
        grid.fit(all_X,all_y)
        model["best_params"] = grid.best_params_
        model["best_score"] = grid.best_score_
        model["best_model"] = grid.best_estimator_

        print("Best Score: {}".format(model["best_score"]))
        print("Best Parameters: {}\n".format(model["best_params"]))

    return models
column_list=["Title_Mr","Pclass","Fare","Age","SibSp","Sex_female","Cabin_type_Unknown"]
result = select_model(train,cols)

