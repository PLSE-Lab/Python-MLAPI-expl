#!/usr/bin/env python
# coding: utf-8

# <h1 style='text-align:center'>Titanic.</h1>
# <br>
# 
# ![](http://media.giphy.com/media/1Nk9bIidJVTy0/giphy.gif)
# 
# **Titanic** is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the **Titanic** sank after colliding with an iceberg, killing *1502* out of *2224* passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.<br><br>
# 
# **What particularly we need do in this challange ?**
# 
# In this challenge, we need to complete the analysis of what sorts of people were likely to survive. In particular,  we apply the tools of machine learning to predict which passengers survived the tragedy?.
# 

# ### Importing the data

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#import train and test data.
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
name=train.Name
train.head()


# <h3>What does this data set mean.</h3>
# ____

# The data has been split into two groups:
# - training set (train.csv)
# - test set(test.csv)
# <br>
# 
# The training set includes passengers survival status(also know as the ground truth from the titanic tragedy) which along with other features like gender, class, fare and pclass is used to create machine learning model.
# <br><br>
# The test set should be used to see how well my model performs on unseen data. The test set does not provide passengers survival status. We are going to use our model to predict passenger survival status.
# <br><br>
# 
# Lets describe whats the meaning of the features given the both train & test datasets.
# <h4>Variable Definition Key.</h4>
# - Survival
#  - 0= No
#  - 1= Yes
# - pclass (Ticket class)
#  - 1=1st
#  - 2=2nd
#  - 3=3rd
#  
# - sex
# <br>
# 
# - age
# 
# 
# - sibsp (# of siblings / spouses aboard the Titanic)
# <br>
# - parch (# of parents / children aboard the Titanic)
# <br>
# - tickets
# <br>
# - fare
# <br>
# - cabin
# - embarked Port of Embarkation.
#  - C = Cherbourg,
#  - Q = Queenstown,
#  - S = Southampton
# - pclass: A proxy for socio-economic status (SES)
# <br>
# <h4>This is important to remember and will come in handy for later analysis.</h4>
#  - 1st = Upper
#  - 2nd = Middle
#  - 3rd = Lower
# 
# 

# 
# 
# ## Part 1. Cleaning the data.

# In[ ]:


## Lets againgtake a quick glance of what we are dealing with.
train.head(5)


# It looks like this dataset is quite organized, however, before using this dataset for analyzing and visualizing we need to deal with ..
# - Different variables
# - Null values

# ## Different variables present in the datasets.
#  - **There are four type of variables**
#   - **Numerical Features**: Age, Fare, SibSp and Parch
#   - **Categorical Features**: Sex, Embarked, Survived and Pclass
#   - **Alphanumeric Features**: Ticket and Cabin(Contains both alphabets and the numeric value)
#   - **Text Features**: Name
# 
# ** We really need to tweak these features so we get the desired form of input data**

# In[ ]:


train.shape# Means 891 rows and 12 columns.


# In[ ]:


train.isnull().sum()


# We see Age  and Cabin have a lot of missing value.So First we need to deal with all these NaN values.
# - As in Cabin column about 1\3rd of the values are missing.So we get rid of this column. 
# <br>
# 
# ## Why missing values treatment is required?
# Missing data in the training data set can reduce the power / fit of a model or can lead to a biased model because we have not analysed the behavior and relationship with other variables correctly. It can lead to wrong prediction or classification.
# - Here the methods to deal with missing values.
# 
# ### KNN Imputation. 
# ------
# In this method of imputation, the missing values of an attribute are imputed using the given number of attributes that are most similar to the attribute whose values are missing.
# 
# For more...
# <br>
# **Method 1**
# - [KNN Imputation](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/)
# - [Blog](https://towardsdatascience.com/the-use-of-knn-for-missing-values-cf33d935c637)
# 
# **Method 2**
# - [sklearn.preprocessing.Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html)
# 

# **We use the Method 2 i.e(sklearn.preprocessing.Imputer)**
# 
# Just because it is easy to use....
# 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
Imp=Imputer(missing_values='NaN',strategy='median',axis=1)
new=Imp.fit_transform(train.Age.values.reshape(1,-1))
train['Age2']=new.T
#Lets drop the old one age Column.


# In[ ]:


train.drop('Age',axis=1,inplace=True)


# In[ ]:


train.isnull().sum()


# In[ ]:


train.set_index('PassengerId',inplace=True)
## get dummy variables for Column sex and embarked since they are categorical value.
train = pd.get_dummies(train, columns=["Sex"], drop_first=True)
train = pd.get_dummies(train, columns=["Embarked"],drop_first=True)


#Mapping the data.
train['Fare'] = train['Fare'].astype(int)
train.loc[train.Fare<=7.91,'Fare']=0
train.loc[(train.Fare>7.91) &(train.Fare<=14.454),'Fare']=1
train.loc[(train.Fare>14.454)&(train.Fare<=31),'Fare']=2
train.loc[(train.Fare>31),'Fare']=3

train['Age2']=train['Age2'].astype(int)
train.loc[ train['Age2'] <= 16, 'Age2']= 0
train.loc[(train['Age2'] > 16) & (train['Age2'] <= 32), 'Age2'] = 1
train.loc[(train['Age2'] > 32) & (train['Age2'] <= 48), 'Age2'] = 2
train.loc[(train['Age2'] > 48) & (train['Age2'] <= 64), 'Age2'] = 3
train.loc[train['Age2'] > 64, 'Age2'] = 4


# In[ ]:


# In our data the Ticket and Cabin,Name are the base less,leds to the false prediction so Drop both of them.
train.drop(['Ticket','Cabin','Name'],axis=1,inplace=True)
train.head()
print(type(train.Age2))


# <h2 >Part 2.Exploratory data analysis</h2>.
# 
# **Exploratory data analysis (EDA)** is an approach to analyzing data sets to summarize their main characteristics, often with visual methods.
# 
# ![](http://media.giphy.com/media/m3UHHYejQ4rug/giphy.gif)

# In[ ]:


train.shape
# 891 rows and 9 columns.


# In[ ]:


train.Survived.value_counts()/len(train)*100
#This signifies almost 61% people in the ship died and 38% survived.


# In[ ]:


train.describe()


# In[ ]:


train.groupby('Survived').mean()


# In[ ]:


train.groupby('Sex_male').mean()


# There are a couple of points that should be noted from the statistical overview. They are..
# - About the survival rate, only 38% passenger survived during that tragedy.
# - About the survival rate for genders, 74% female passengers survived, while only 19% male passengers survived.

# **Correlation Matrix and Heatmap**

# In[ ]:


train.corr()


# In[ ]:


plt.subplots(figsize = (15,8))
sns.heatmap(train.corr(), annot=True,cmap="PiYG")
plt.title("Correlations Among Features", fontsize = 20);


# **Positive Correlation Features:**
# - Fare and Survived: 0.26.
# 
# There is a positive correlation between Fare and Survived rated. This can be explained by saying that, the passenger who paid more money for their ticket were more likely to survive. 

# **Negative Correlation Features:**
# - Fare and Pclass: -0.55
#  - This relationship can be explained by saying that first class passenger(1) paid more for fare then second class passenger(2), similarly second class passenger paid more than the third class passenger(3). 
# - Gender and Survived: -0.54
#  - Basically is the info of whether the passenger was male or female.
# - Pclass and Survived: -0.34

# **Gender and Survived**
# 

# In[ ]:


plt.subplots(figsize = (15,8))
sns.barplot(x = "Sex_male", y = "Survived", data=train, edgecolor=(0,0,0), linewidth=2)
plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25)
labels = ['Female', 'Male']
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Gender",fontsize = 15)
plt.xticks(sorted(train.Sex_male.unique()), labels)

# 1 is for male and 0 is for female.


# This bar plot above shows the distribution of female and male survived. The x_label shows gender and the y_label shows % of passenger survived. This bar plot shows that 74% female passenger survived while only ~19% male passenger survived.

# In[ ]:


sns.set(style='darkgrid')
plt.subplots(figsize = (15,8))
ax=sns.countplot(x='Sex_male',data=train,hue='Survived',edgecolor=(0,0,0),linewidth=2)
train.shape
## Fixing title, xlabel and ylabel
plt.title('Passenger distribution of survived vs not-survived',fontsize=25)
plt.xlabel('Gender',fontsize=15)
plt.ylabel("# of Passenger Survived", fontsize = 15)
labels = ['Female', 'Male']
#Fixing xticks.
plt.xticks(sorted(train.Survived.unique()),labels)
## Fixing legends
leg = ax.get_legend()
leg.set_title('Survived')
legs=leg.texts
legs[0].set_text('No')
legs[1].set_text('Yes')


# This count plot shows the actual distribution of male and female passengers that survived and did not survive. It shows that among all the females ~ 230 survived and ~ 70 did not survive. While among male passengers ~110 survived and ~480 did not survive.

# **Summary**
# - As we suspected, female passengers have survived at a much better rate than male passengers.
# - It seems about right since females and children were the priority.

# **Pclass and Survived**

# In[ ]:


train.head(4)


# In[ ]:


plt.subplots(figsize = (10,10))
ax=sns.countplot(x='Pclass',hue='Survived',data=train)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
leg=ax.get_legend()
leg.set_title('Survival')
legs=leg.texts

legs[0].set_text('No')
legs[1].set_text("yes")


# So it clearly seems that,The survival of the people belong to 3rd class is very least.
# It looks like ...
# -  63% first class passenger survived titanic tragedy, while
# -  48% second class and
# -  only 24% third class passenger survived.

# In[ ]:


plt.subplots(figsize=(10,8))
sns.kdeplot(train.loc[(train['Survived'] == 0),'Pclass'],shade=True,color='r',label='Not Survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'],shade=True,color='b',label='Survived' )

labels = ['First', 'Second', 'Third']
plt.xticks(sorted(train.Pclass.unique()),labels)


# This kde plot is pretty self explanatory with all the labels and colors. Something I have noticed that some readers might find questionable is that in, the plot; the third class passengers have survived more than second class passnegers. It is true since there were a lot more third class passengers than first and second.
# 
# 

# **Summary**
# 
# First class passenger had the upper hand during the tragedy than second and third class passengers. You can probably agree with me more on this, when we look at the distribution of ticket fare and survived column.

# **Fare and Survived**

# In[ ]:


plt.subplots(figsize=(15,10))

ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'],color='r',shade=True,label='Not Survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'],color='b',shade=True,label='Survived' )
plt.title('Fare Distribution Survived vs Non Survived',fontsize=25)
plt.ylabel('Frequency of Passenger Survived',fontsize=20)
plt.xlabel('Fare',fontsize=20)


# **Age and Survived**

# In[ ]:


#fig,axs=plt.subplots(nrows=2)
fig,axs=plt.subplots(figsize=(10,8))
sns.set_style(style='darkgrid')
sns.kdeplot(train.loc[(train['Survived']==0),'Age2'],color='r',shade=True,label='Not Survived')
sns.kdeplot(train.loc[(train['Survived']==1),'Age2'],color='b',shade=True,label='Survived')


# There is nothing out of the ordinary of about this plot, except the very left part of the distribution. It shows that
# 
# children and infants were the priority.

# **Modeling the Data**

# I will train the data with the following models:
# - Logistic Regression
# - Gaussian Naive Bayes
# - Support Vector Machines
# - Decision Tree Classifier
# - K-Nearest Neighbors(KNN)
#  -  and many other.....

# In[ ]:


X=train.drop('Survived',axis=1)
y=train['Survived'].astype(int)


# <h2>Classifier Comparision</h2>
# 
# By Classifier Comparison we choose which model best for the given data.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis 
from xgboost import XGBClassifier

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    XGBClassifier(),
    RandomForestClassifier(n_estimators=100, max_features=3),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]
    


log_cols = ["Classifier", "Accuracy"]
log= pd.DataFrame(columns=log_cols)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
acc_dict = {}

for train_index, test_index in sss.split(X, y):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    for clf in classifiers:
        name = clf.__class__.__name__
    
        clf.fit(X_train,y_train)
        predict=clf.predict(X_test)
        acc=accuracy_score(y_test,predict)
        if name in acc_dict:
            acc_dict[name]+=acc
        else:
            acc_dict[name]=acc


# In[ ]:


log['Classifier']=acc_dict.keys()
log['Accuracy']=acc_dict.values()
log.set_index([[0,1,2,3,4,5,6,7,8,9,10]])
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_color_codes("muted")
ax=plt.subplots(figsize=(10,8))
ax=sns.barplot(y='Classifier',x='Accuracy',data=log,color='b')
ax.set_xlabel('Accuracy',fontsize=20)
plt.ylabel('Classifier',fontsize=20)
plt.grid(color='r', linestyle='-', linewidth=0.5)
plt.title('Classifier Accuracy',fontsize=20)


# From the above barplot, we can clearly see that the accuracy of the SVC classifier is best out of all other classifiers..
# 
# Lets apply this to our test data.

# <h2>Prediction</h2>
# 
# Lets use the SVC classifier to predict our data.

# In[ ]:


train.head()


# In[ ]:


classifier=SVC()
xtrain=train.iloc[:,1:]
ytrain=train.iloc[:,:1]
ytrain=ytrain.values.ravel()
classifier.fit(xtrain,ytrain)
#we need to convert the test data 


# In[ ]:


testIm=Imputer(missing_values='NaN',strategy='most_frequent',axis=1)
Age1=testIm.fit_transform(test.Age.values.reshape(1,-1))
Fare2=testIm.fit_transform(test.Fare.values.reshape(1,-1))
test.drop(['Name','Cabin','Age','Ticket','Fare'],axis=1,inplace=True)
test['Age1']=Age1.T
test['Fare2']=Fare2.T
test.set_index('PassengerId',inplace=True)
#test[test.Fare.isnull()]#this will tell us which row have null so we can drop that particular row.
#test.drop([1044],axis=0,inplace=True)#drop the row which NaN .
#test.isnull().sum()


# In[ ]:


## get dummy variables for Column sex and embarked since they are categorical value.
test = pd.get_dummies(test, columns=["Sex"], drop_first=True)
test = pd.get_dummies(test, columns=["Embarked"],drop_first=True)


#Mapping the data.
test['Fare2'] = test['Fare2'].astype(int)
test.loc[test.Fare2<=7.91,'Fare2']=0
test.loc[(test.Fare2>7.91) &(test.Fare2<=14.454),'Fare2']=1
test.loc[(test.Fare2>14.454)&(test.Fare2<=31),'Fare2']=2
test.loc[(test.Fare2>31),'Fare2']=3

test['Age1']=test['Age1'].astype(int)
test.loc[ test['Age1'] <= 16, 'Age1']= 0
test.loc[(test['Age1'] > 16) & (test['Age1'] <= 32), 'Age1'] = 1
test.loc[(test['Age1'] > 32) & (test['Age1'] <= 48), 'Age1'] = 2
test.loc[(test['Age1'] > 48) & (test['Age1'] <= 64), 'Age1'] = 3
test.loc[test['Age1'] > 64, 'Age1'] = 4


# <h2>Result</h2>
# 
# The final result is

# In[ ]:


Result=classifier.predict(test)
print(Result)
print(len(Result))


# **This kernal is still under  process for further imporvement.**
# 
# I will always incorporate new concepts of data science as I master them. This journey of learning is worth sharing as well as collaborating. Therefore any comments about further improvements would be genuinely appreciated.

# **Hope you find it useful.** 
# 
# **If this notebook helped you in anyway, please do upvote!**
# 

# In[ ]:





# In[ ]:




