#!/usr/bin/env python
# coding: utf-8

# # <a id="top_section"></a>
# 
# <div align='center'><font size="6" color="#000000"><b>Titanic Survival Prediction : The beginner's way !</b></font></div>
# <hr>
# <div align='center'><font size="5" color="#000000">Introduction</font></div>
# <hr>
# 
# When I started doing this analysis my main goal was getting experience. I'm still learning and trying to improve my skills, so there is a great chance that there will be some areas for improvement.
# <br><br>
# In this notebook I will show you how a beginner (in this case ,me) approached the Titanic Survival prediction problem.I read a lot of notebooks of very experienced and helpful kagglers and watched some videos on youtube for better understanding. So without further ado , let's get this started!
# 
# ### Here are the things I will try to cover in this Notebook:
# 
# - What is the survival rate for specific groups?
# - Is there any relation between given info of passengers and their survival?
# - Was women and children first policy in order?
# - Having higher social status in helped people getting in life boats?
# - The mustering order from captain was highly dependent on the class of the passengers, can we spot this  effect between pclass and the survival rates?
# - What are the effects of family size?
# 
# - Can we predict if a passenger survived from the disaster with using machine learning techniques?
# 
# ### If you liked this kernel feel free to upvote and leave feedback, thanks!

# <a id="toc_section"></a>
# ## Table of Contents
# * [Introduction](#top_section)
# * [Importing all the Required Libraries](#section1)
# * [Exploring the Data](#section2)
#     - [Visualizing given dataset](#section3)
# * [Building the Feature Engineering Machine](#section4)
#     - [Merging Parch and SibSp](#section5)
#     - [Converting Sex column to numerical type](#section6)
#     - [Extracting Title from Name](#section7)
#     - [Filling and Encoding the Fare feature](#section8)
#     - [Encoding the Embarked feature](#section9)
#     - [Filling and Encoding the Age feature](#section10)
#     - [Do not make this mistake !](#section11)
#     - [Dropping the unnecessary features](#section12)
# * [Modelling](#section13)
# * [Submission & Some Last Words](#sectionlst)
# 

# <a id="section1"></a>
# ## Importing the libraries
# 
# ### [Back To Table of Contents](#toc_section)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Let us also import all the sklearn libraries we might need 

# In[ ]:


from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV,learning_curve,cross_val_score


# <a id="section2"></a>
# ## Exploring the data
# 
# ### [Back To Table of Contents](#toc_section)

# Let's read the data , both train and test one and to make sure that they are in the same format we will for the time being concatinate them so that both have the same operations

# In[ ]:


train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
df=pd.concat([train,test],axis=0,ignore_index=True)
print(f'Train:{train.shape}\nTest:{test.shape}\nDf:{df.shape}')


# Mind taking a sneak peak in the dataset now ?

# In[ ]:


df.head()


# The first thing we will do is see for any missing data in the dataset.

# In[ ]:


#columns with missing values
df.isnull().sum()[df.isnull().sum()>0]


# Let's see the stats of our data now using the describe function (I just transposed it because i prefer it that way)

# In[ ]:


df.describe().T


# Now let's see the stats of the categorical features

# In[ ]:


# only describing the categorical columns
df.describe(exclude='number')


# There is clearly alot of data missing in the age and cabin column , a few in embarked too , the one you see in survived is not really the missing data but the data we need to predict so don't worry about it.

# We will take care of the missing data , but first we need to see how each feature is connected to one another and you'll see how it will help us to take care of the missing data as well as understand the data more clearly.

# <a id="section3"></a>
# ## Visualizing given dataset
# 
# ### [Back To Table of Contents](#toc_section)

# In[ ]:


#Let's draw a heatmap of correlation between the features
plt.figure(figsize=(12,7))
sns.heatmap(df.drop('PassengerId',axis=1).corr(),annot=True,center=0)


# As we can see from the heatmap , that the Survival is highly correlated to Patch,SibSp,Age,Sex,Fare and Pclass.

# Now let's plot and analyze the age and survival correlation.

# In[ ]:


g=sns.FacetGrid(train,col='Survived').map(sns.distplot,'Age',hist=False,kde=True,rug=False,kde_kws={'shade':True})


# Now let's plot and analyse how the Passenger Class(Pclass in dataset) affects survival chances of a person.

# In[ ]:


sns.catplot(x='Pclass',y='Survived',data=train,kind='bar')


# What about the sex of a person , how does it affect a person's survival? Let's find out !

# In[ ]:


sns.catplot(x='Sex',y='Survived',hue='Pclass',data=train,kind='bar')


# Does the Embarked feature also affect the survival chances, there only one way to find out , let's see!

# In[ ]:


sns.catplot(x="Embarked", y="Survived", data=train, kind="bar")


# <a id="section4"></a>
# ## Building the Feature Engineering Machine
# 
# ### [Back To Table of Contents](#toc_section)
# 

# <a id="section5"></a>
# ### Merging Parch and SibSp
# 

# Let's combine the Parch and SibSp feature to form a better Family_size feature 

# In[ ]:


df['Family_Size']=df['Parch']+df['SibSp']
df.groupby('Family_Size')['Survived'].mean()


# Now let's see how the Family Size of a person affects his/her chance of survival.

# In[ ]:


sns.catplot(x='Family_Size',y='Survived',data=df,kind='bar')


# <a id="section6"></a>
# ### Converting Sex column to numerical type
# 

# Let's convert the Sex column from categorical to Numerical , will use the map function for this.

# In[ ]:


#converting the sex column into numerical column
df.Sex=df.Sex.map({'male':0,'female':1}).astype('int')


# <a id="section7"></a>
# ### Extracting Title from Name
# 

# The Name feature is not really useful but we can use the Title of a person(such as Mr,Miss,etc) as a feature , so let's do it !

# In[ ]:


df['Title'] = df['Name']

for name_string in df['Name']:
    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

df.replace({'Title': mapping}, inplace=True)


# Let's see if the Title feature really is affecting the Survival chances or not!

# In[ ]:


sns.barplot(x='Title',y='Survived',data=df.iloc[:len(train)])


# <a id="section8"></a>
# ### Filling and Encoding the Fare feature
# 

# Let's see for missing values in the Fare column.

# In[ ]:


df[df['Fare'].isnull()]


# As there is only one missing value , we can easily fill it up by either mean or median of the column , I am using mean here.

# In[ ]:


df['Fare'].fillna(df['Fare'].mean(), inplace=True)


# Now let's encode the Fare column into 5 categories.I am using LabelEncoder from sklearn here.

# In[ ]:


df['FareBin'] = pd.qcut(df['Fare'], 5)

label = LabelEncoder()
df['FareBin_Code'] = label.fit_transform(df['FareBin'])
df.drop(['Fare'], 1, inplace=True)


# Let's see how the encoded fare column looks.

# In[ ]:


df['FareBin_Code'].value_counts()


# <a id="section9"></a>
# ### Encoding the Embarked feature
# 

# Now we will use the get_dummy method of pandas to encode the Embarked column.

# In[ ]:


embarked=pd.get_dummies(df['Embarked'],drop_first=True)
df=pd.concat([df,embarked],axis=1)


# <a id="section10"></a>
# ### Filling and Encoding the Age feature
# 

# Now , the time for taking care of the Age column is here, we have two options here either use Pclass or Title to help fill the missing age values , I first used the Pclass but , using the Title increased my score so here are both for you !

# Here's how you can fill missing ages using the Pclass column as reference.

# In[ ]:


#  def fix_age(cols):
#     Age=cols[0]
#     Pclass=cols[1]
    
#     if pd.isnull(Age):
#         if Pclass==1:
#             return 37 
#         elif Pclass==2:
#             return 29
#         else:
#             return 24
#     else:
#         return Age 


# In[ ]:


# df['Age']=df[['Age','Pclass']].apply(fix_age,axis=1)


# Here's how you can fill missing ages using the Title column as reference.(this yield better results for me)

# In[ ]:


# filling missing values in 'age' column
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:
    age_to_impute = df.groupby('Title')['Age'].median()[titles.index(title)]
    df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = age_to_impute


# Let's make sure it's done !

# In[ ]:


df['Age'].isnull().sum()


# Now let us encode the age column into 4 parts. 

# In[ ]:


df['AgeBin'] = pd.qcut(df['Age'], 4)

label = LabelEncoder()
df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])


# We have done a lot of operations lately , let's see how our data looks like now.

# In[ ]:


df.sample(2)


# In[ ]:


df['AgeBin_Code'].value_counts()


# <a id="section11"></a>
# ### Do not make this mistake !

# Now we are done with all the encoding and taking care of mising values stuff , let's drop the features that are not useful.

# But wait what about the Cabin column , like you i also thought that with so many missing values , I should just drop it but i thought of something , made a column with people having cabin(entry with cabin !=Nan) and people not having a cabin (entry with cabin column = Nan) ,and it actually has a great effect on the survival chances.

# So let's do it !

# In[ ]:


df.loc[(df['Cabin'].isnull()),'Cabin_status'] = 0
df.loc[(df['Cabin'].notnull()),'Cabin_status']=1
df.Cabin_status.astype('int')


# So , let me show you how the not so useful seeming Cabin is actually useful !

# In[ ]:


sns.barplot(x='Cabin_status',y='Survived',data=df.iloc[:len(train)])


# So finally , let's drop all the unnecessary columns.

# <a id="section12"></a>
# ### Dropping the unnecessary features
# 

# In[ ]:


df.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin','Embarked',
                 'FareBin', 'AgeBin', 'Survived', 'Title', 'Age'], axis = 1, inplace = True)


# After all the dropped columns , our data must be clean and not-so-long , let's see for ourselves

# In[ ]:


df.sample(2)


# <a id="section13"></a>
# # Modelling
# 
# Since preprocessing is done we are ready for training our models. We start with loading packages and splitting our transformed data so we have 22 features and and 891 observations to train our estimators. Our test set has 418 observations to make predictions.
# 
# ### [Back To Table of Contents](#toc_section)

# Now , we have to head towards the modelling and submision part , so let's split the data into it's initial train and test sets. 

# In[ ]:


X_train = df[:len(train)]
X_test = df[len(train):]

y_train = train['Survived']


# In[ ]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


kfold = StratifiedKFold(n_splits=8)


# In[ ]:


RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],
              "max_features": [3,"sqrt", "log2"],
              "min_samples_split": [n for n in range(1, 9)],
              "min_samples_leaf": [5, 7],
              "bootstrap": [False, True],
              "n_estimators" :[200, 500],
              "criterion": ["gini", "entropy"]}

rf_param_grid_best = {"max_depth": [None],
              "max_features": [3],
              "min_samples_split": [4],
              "min_samples_leaf": [5],
              "bootstrap": [False],
              "n_estimators" :[200],
              "criterion": ["gini"]}

gs_rf = GridSearchCV(RFC, param_grid = rf_param_grid_best, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)

gs_rf.fit(X_train, y_train)

rf_best = gs_rf.best_estimator_
RFC.fit(X_train, y_train)


# In[ ]:


print(f'RandomForest GridSearch best params: {gs_rf.best_params_}\n')
print(f'RandomForest GridSearch best score: {gs_rf.best_score_}')
print(f'RandomForest score:                 {RFC.score(X_train,y_train)}')


# In[ ]:


XGB = XGBClassifier()

xgb_param_grid = {'learning_rate':[0.05, 0.1], 
                  'reg_lambda':[0.3, 0.5],
                  'gamma': [0.8, 1],
                  'subsample': [0.8, 1],
                  'max_depth': [2, 3],
                  'n_estimators': [200, 300]
              }

xgb_param_grid_best = {'learning_rate':[0.1], 
                  'reg_lambda':[0.3],
                  'gamma': [1],
                  'subsample': [0.8],
                  'max_depth': [2],
                  'n_estimators': [300]
              }

gs_xgb = GridSearchCV(XGB, param_grid = xgb_param_grid_best, cv=kfold, scoring="roc_auc", n_jobs= 4, verbose = 1)

gs_xgb.fit(X_train,y_train)
XGB.fit(X_train, y_train)

xgb_best = gs_xgb.best_estimator_


print(f'XGB GridSearch best params: {gs_xgb.best_params_}\n')
print(f'XGB GridSearch best score: {gs_xgb.best_score_}')
print(f'XGB score:                 {XGB.score(X_train,y_train)}')


# <a id="sectionlst"></a>
# #  Submission
# 
# 
# ### [Back To Table of Contents](#toc_section)

# In[ ]:


results=pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':RFC.predict(X_test)})
results.to_csv("submission.csv", header=True,index=False)

print("The submission file is ready, here's a sample of it!")
print(results.sample(2))


# # Some last words:
# 
# ### Thank you for reading! I'm still a beginner and want to improve myself in every way I can. So if you have any ideas to feedback please let me know in the comments section!
# 
# 
# <div align='center'><font size="5" color="#000000"><b>And again please vote if you liked this notebook so it can reach more people, Thanks!</b></font></div>

# <img src="https://cloud.netlifyusercontent.com/assets/344dbf88-fdf9-42bb-adb4-46f01eedd629/7543b6dd-a4db-46d7-9a46-9755c6399b0e/puppy-thanks.jpg" alt="Thank you!" width="500" height="600">
