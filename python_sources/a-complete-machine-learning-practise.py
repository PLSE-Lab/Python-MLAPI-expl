#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns
sns.set(style='darkgrid')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


traindf=pd.read_csv("/kaggle/input/titanic/train.csv")
testdf=pd.read_csv("/kaggle/input/titanic/test.csv")
submissiondf=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print(traindf.shape)
traindf.head()


# In[ ]:


print(testdf.shape)
testdf.head()


# <h1> Exploratory Data Analysis </h1>

# **Target Variable is "Survived"**
# 
# ##### Basically the problem is a classification problem and we need to classify if the passenger survived(1) or not survived(0)

# ### Analysing and understanding data

# In[ ]:


submissiondf.head()


# In[ ]:


traindf.info()


# In[ ]:


traindf.describe()


# In[ ]:


for col in traindf.columns:
    if traindf[col].isnull().values.any():
        print(col,traindf[col].isna().sum())


# <ul>
#     <li>Over 20% of data in Age column is missing </li>
# <li>Over 77% of data in Cabin is missing</li>
# <li>Only 2 entries are Embarked is missing</li>
# </ul>
# 
# Lets handle Embarked feature Null entires first

# In[ ]:


traindf['Embarked'].value_counts()


# In[ ]:


print(traindf.shape)
traindf=traindf.dropna(axis=0, subset=['Embarked'])
traindf.head()


# In[ ]:


traindf = pd.get_dummies( traindf, columns = ['Embarked'],prefix="EM" )
testdf=pd.get_dummies(testdf,columns = ['Embarked'],prefix="EM")
#traindf.drop('Embarked', axis=1, inplace=True)
traindf.head()


# <h4>Lets Handle Cabin feature</h4>
# with 775 of data missing in this column, Deleting the column all together
# 

# In[ ]:


traindf.drop('Cabin', axis=1, inplace=True)
testdf.drop('Cabin', axis=1, inplace=True)
traindf.head()


# <h4>Lets Handle Age Feature</h4>

# In[ ]:


plt.figure(figsize=(20,5))
g = sns.distplot(traindf['Age'], bins= 50,color='black')
g.set_xlabel("Age", fontsize=18)
g.set_ylabel("Density", fontsize=18)
plt.show()


# In[ ]:


print(traindf['Age'].mean())


# In[ ]:


traindf['Age'].fillna(traindf['Age'].mean(), inplace=True)


# In[ ]:


traindf.head()


# In[ ]:


## Plotting the bar char to identify the frequnecy of values
sns.countplot(traindf["Sex"],color='black')
##prinitng number of values for each type
print(traindf["Sex"].value_counts())


# In[ ]:


Sex_binary = {'male': 1,'female': 0}
traindf["Sex"]= [Sex_binary[item] for item in traindf["Sex"]]
testdf["Sex"]= [Sex_binary[item] for item in testdf["Sex"]]
traindf.head()


# In[ ]:


print(traindf.Ticket.value_counts().count())


# There are 680 unique values in Ticket column so may not be useful, so removing the column for now.<br/>
# Similarly deleting Name column too

# In[ ]:


traindf.drop('Ticket', axis=1, inplace=True)
traindf.drop('Name', axis=1, inplace=True)
traindf.drop('PassengerId', axis=1, inplace=True)

testdf.drop('Ticket', axis=1, inplace=True)
testdf.drop('Name', axis=1, inplace=True)
#testdf.drop('PassengerId', axis=1, inplace=True)

traindf.head()


# In[ ]:


# Map each Age value to a numerical value: 
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
mylabels = ['Baby', 'Child', 'Youth', 'Student', 'Young Adult', 'Adult', 'Senior']
traindf['AgeGroup'] = pd.cut(traindf["Age"], bins, labels = mylabels)
traindf[["AgeGroup","Survived"]].groupby("AgeGroup").mean()
age_mapping = {'Baby': 1, 'Child': 2, 'Youth': 3, 'Student': 4, 'Young Adult':5 , 'Adult': 6, 'Senior':7}
traindf['AgeGroup'] = traindf['AgeGroup'].map(age_mapping)

testdf['AgeGroup'] = pd.cut(testdf["Age"], bins, labels = mylabels)
age_mapping = {'Baby': 1, 'Child': 2, 'Youth': 3, 'Student': 4, 'Young Adult':5 , 'Adult': 6, 'Senior':7}
testdf['AgeGroup'] = testdf['AgeGroup'].map(age_mapping)


# In[ ]:


traindf.drop('Age', axis=1, inplace=True)
testdf.drop('Age', axis=1, inplace=True)
traindf.head()


# In[ ]:


plt.figure(figsize=(20,5))
g = sns.distplot(traindf['Fare'], bins= 50,color='black')
g.set_xlabel("Fare", fontsize=18)
g.set_ylabel("Density", fontsize=18)
plt.show()


# In[ ]:


sns.boxplot(traindf["Fare"], color='black')


# Fare is a continous number and varied to huge extent, instead of having it a number let's group the fare intervals into Bins using Binning

# In[ ]:


traindf['FareRange'] = pd.qcut(traindf['Fare'], 10, labels = [1, 2, 3, 4,5,6,7,8,9,10])
testdf['FareRange'] = pd.qcut(testdf['Fare'], 10, labels = [1, 2, 3, 4,5,6,7,8,9,10])
traindf.drop('Fare', axis=1, inplace=True)
testdf.drop('Fare', axis=1, inplace=True)
traindf.head()


# In[ ]:


plt.figure(figsize=(20,5))
g = sns.distplot(traindf['FareRange'], bins= 50,color='black')
g.set_xlabel("FareRange", fontsize=18)
g.set_ylabel("Density", fontsize=18)
plt.show()


# In[ ]:


## Plotting the bar char to identify the frequnecy of values
sns.countplot(traindf["Survived"], color='black')
##prinitng number of values for each type
print(traindf["Survived"].value_counts())


# In[ ]:


f,ax = plt.subplots(figsize=(20, 5))
sns.heatmap(traindf.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[ ]:


traindf.head()


# In[ ]:


testdf.head()


# <h1> Modeling, Evaluation and Model Tuning </h1>

# In[ ]:


from sklearn.model_selection import train_test_split

# Putting feature variable to X
X = traindf.drop(['Survived'],axis=1)
# Putting response variable to y
y = traindf['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[ ]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn import tree


# In[ ]:


seed = 7
# prepare models
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DecisonTree', DecisionTreeClassifier()))
models.append(('NB', BernoulliNB()))
models.append(('SVM', SVC()))
models.append(('BaggingDecisonTree', BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))))
models.append(('Adaboost',AdaBoostClassifier()))
models.append(('Logistic', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
# evaluate each model in turn
results = []
names = []
performance=[]
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train,y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	performance.append(msg)
# boxplot algorithm comparison
fig = plt.figure(figsize=(20,8))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results,widths = 0.5)
ax.set_xticklabels(names)
plt.show()

for perf in performance:
    print(perf)


# #### As per above data, **Logistic Regression and SVM models** gave better results, so considering them as challening models and tuning them indivually to get champion model

# ### 1. Logistic Regression 

# In[ ]:


from sklearn.linear_model import LogisticRegression

LR= LogisticRegression(penalty='none')
LR= LR.fit(X_train,y_train)
y_predict = LR.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics

def Metrics_func( actual, probs ):
    print(classification_report(actual,probs))
    print("accuracy", metrics.accuracy_score(actual, probs))
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    return None

Metrics_func(y_test, y_predict)


# ### 2.SVM

# In[ ]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
clf= clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
Metrics_func(y_test, y_predict)


# ### 3. Bagging Classifer

# In[ ]:


from sklearn.ensemble import BaggingClassifier
dtc = DecisionTreeClassifier(criterion="entropy")
bag_model=BaggingClassifier(base_estimator=dtc, n_estimators=100, bootstrap=True)
bag_model=bag_model.fit(X_train,y_train)
Y_pred=bag_model.predict(X_test)


# In[ ]:


Metrics_func(y_test, Y_pred)


# In[ ]:


traindf.head()


# In[ ]:


for col in testdf.columns:
    if testdf[col].isnull().values.any():
        print(col,testdf[col].isna().sum())


# In[ ]:


testdf['FareRange'].fillna(3, inplace=True)


# In[ ]:


testdf['AgeGroup'].fillna(5, inplace=True)


# In[ ]:


x=testdf.drop('PassengerId', axis=1)
submission_df=bag_model.predict(x)


# In[ ]:


PassengerId = np.array(testdf["PassengerId"])
Survived = np.array(submission_df)
submission_dataset = pd.DataFrame({'PassengerId': PassengerId, 'Survived': Survived}, columns=['PassengerId', 'Survived'])
submission_dataset.to_csv('submission.csv', header=True, index=False) 


# In[ ]:




