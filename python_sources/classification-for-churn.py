#!/usr/bin/env python
# coding: utf-8

# # Churn Classification
# by : Hesham Asem
# 
# this is a data set which contain several features , & we need to apply classification model , to be able to detect if the person exited or not
# 
# here is the data : 
# 
# https://www.kaggle.com/mrtechnical011/classification-dataset
# 
# 
# 
# so let's start importing needed libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


# _____
# 
# # Reading Data
# 
# then read the data file

# In[ ]:


data = pd.read_csv('/kaggle/input/Churn_Modelling.csv')
data.head()


# what is the dimension 

# In[ ]:


data.shape


# _____
# 
# great , what is the type of values , & is there any nulls ? 

# In[ ]:


data.info()


# ____
# 
# looks clean data , also alkl features even numerical or categorical , so we don't have a mized data which need conversion 
# 
# ok , how about the details of it ? 

# In[ ]:


data.describe()


# ____
# 
# 
# # Unique Values 
# 
# ok we can notice that row number is just a series of number , so it'll not be helpful for training , so we'll drop it later 
# 
# also we can notice a binary values (just ones & zeros) at HasCrCard , IsActiveMember , and ofcourse the output Exited
# 
# we need to look to the unique values of each feature , and this can be easily done here 

# In[ ]:


for column in data.columns : 
    print('Number of unique data for {0} is {1}'.format(column , len(data[column].unique())))
    print('unique data for {0} is {1}'.format(column , data[column].unique()))
    print('=====================================')


# ____
# 
# so it's clear that we'll not use ('RowNumber', 'CustomerId', 'Surname') , since they will help us with nothing in trainging the model , let's drop them 
# 

# In[ ]:


data.drop(['RowNumber', 'CustomerId', 'Surname' ], axis=1, inplace=True)


# now how data looks like ? 

# In[ ]:


data.head()


# we need to have another look to the unique values of features 

# In[ ]:


for column in data.columns : 
    print('Number of unique data for {0} is {1}'.format(column , len(data[column].unique())))
    print('unique data for {0} is {1}'.format(column , data[column].unique()))
    print('=====================================')


# ____
# 
# 
# # Needed Functions
# 
# so before we handle the dummies values for categrocial features , let's first build an important functions that we'll need , to know the relationship & the correlations between features & each other
# 
# _____
# 
# first a function to make pie chart depend on the the value counts & their index
# 
# 

# In[ ]:


def make_pie(feature) : 
    plt.pie(data[feature].value_counts(),labels=list(data[feature].value_counts().index),
        autopct ='%1.2f%%' , labeldistance = 1.1,explode = [0.05 for i in range(len(data[feature].value_counts()))] )
    plt.show()


# then a function for making countplot using seaborn

# In[ ]:


def make_countplot(feature) :
    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("prism", 3)) 


# another one for kdeplot also using seaborn

# In[ ]:


def make_kdeplot(feature) : 
    sns.kdeplot(data[feature], shade=True)


# also we'll need this function to divide some features into few segmentations

# In[ ]:


def divide_feature(feature,n):
    return round((data[feature]- data[feature].min())/n)


# also for making dummies for categorical features , using LabelEncoder from sklearn

# In[ ]:


def make_label_encoder(original_feature , new_feature) : 
    enc  = LabelEncoder()
    enc.fit(data[original_feature])
    data[new_feature] = enc.transform(data[original_feature])
    data.drop([original_feature],axis=1, inplace=True)


# also , we'll need to make standardization for each feature here

# In[ ]:


def make_standardization(feature) : 
    data[feature] =  (data[feature] - data[feature].mean()) / (data[feature].max() - data[feature].min())


# and at last to make a classification report using its class from sklearn

# In[ ]:


def make_report() : 
    print(classification_report(y_test,y_pred))
    print('************************************')
    CM = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix is : \n', CM)
    print('************************************')
    sns.heatmap(CM, center = True)
    plt.show()


# ____
# 
# # Data Correlations
# 
# now let's use these functions . to show everything we need in features 
# 
# let's start with countplotting the output , to know how many people are exited & how many are not

# In[ ]:


make_countplot("Exited")


# ok , about 20 % , which is enough data for training both types
# 
# now when we move to the creditscore features , how many unique values it contain ? 

# In[ ]:


len(data['CreditScore'].unique())


# it's a big a mount which will not enable us to plot it easily , since it's numerical values so we can use them in training , but we need to divide it now into segmentaion to have a look to it , let's use this function 

# In[ ]:


data['temp'] = divide_feature('CreditScore',100)


# let's see it in the data

# In[ ]:


data.head()


# now we can know the countplot for each segmentation

# In[ ]:


make_countplot('temp')


# ok , majority of them in 2nd , 3rd & 4th segmentaion , now we can drop it since we'll not use it in training

# In[ ]:


data.drop(["temp" ], axis=1, inplace=True)


# _____
# 
# how about the original country , let's have a look at it

# In[ ]:


make_countplot("Geography")


# almost 50% of people are from france & the rest are equally divided between spain & germany  , ok will pie chart graph whelp us in something ? 

# In[ ]:


make_pie('Geography')


# ok it gave us the same idea 
# 
# _____
# 
# how about the Gender distribution ? 

# In[ ]:


make_countplot("Gender")


# not exactly equally divided but they are pretty close to each other 
# 
# ____
# 
# 
# now let's have a look to the Age distribution 

# In[ ]:


make_pie("Age")


# Oh , since the age unique values are so much , so we'll need to divide them into segmentaions  

# In[ ]:


data['temp'] = divide_feature('Age',10)


# now we can make the pie again

# In[ ]:


make_pie('temp')


# which refer to us that almost 75% of people are from first 2 segmentaions , ok how about kdeplot , for the Age feature itself 

# In[ ]:


make_kdeplot('Age')


# almost same result , majority of people from 30 to 50 , ok , let's drop the temp feature

# In[ ]:


data.drop(["temp" ], axis=1, inplace=True)


# ____
# 
# now we can have a look to Tenure feature

# In[ ]:


make_countplot("Tenure")


# almost equally distributed  . . 
# 
# how about balane , let's graph it

# In[ ]:


make_kdeplot('Balance')


# majority of people either have zero balance , or between 10 & 15 thousand . 
# 
# to have an accurate look , let's divide it 

# In[ ]:


data['temp'] = divide_feature('Balance',10000)
print('Number of Sectors are {}'.format(len(data['temp'].unique())))


# 25 segmentations are fine , now let's make pie chart about it 

# In[ ]:


make_pie('temp')


# kinda more clear , ok drop it 

# In[ ]:


data.drop(["temp" ], axis=1, inplace=True)


# ow about Nomber of Products ? 

# In[ ]:


make_pie('NumOfProducts')


# majority of it either 1 or 2 , so this data will affect in a bad way finding any product number 3 or 4 , but there is nothing to do here
# 
# ___
# 
# ok, how about either he had a card or not

# In[ ]:


make_countplot('HasCrCard')


# great . now if he is an active member or now

# In[ ]:


make_pie('IsActiveMember')


# almost equl numbers 
# 
# ___
# 
# 
# how about the estimated salary , let's know its unique values

# In[ ]:


len(data['EstimatedSalary'].unique())


# since it's a big amount of unique values , we have to divide it

# In[ ]:


data['temp'] = divide_feature('EstimatedSalary',10000)
print('Number of Sectors are {}'.format(len(data['temp'].unique())))


# ok , now we can plot it easily 

# In[ ]:


make_pie('temp')


# kinda equally distributed , let's have a look to the kdeplot

# In[ ]:


make_kdeplot('temp')


# now let's drop it 

# In[ ]:


data.drop(["temp"], axis=1, inplace=True)


# ok , we now finished data processing , so we can move to get dummies step
# 
# _____
# 
# 
# # Get Dummies
# 
# first let's now what categorical features needed to convert it 

# In[ ]:


data.head()


# ____
# 
# both features Geography & Gender  , so let's convert both of them into new features & frop the original features 

# In[ ]:


make_label_encoder('Geography' , 'Geography Code')


# let's have a look

# In[ ]:


data.head()


# now the gender

# In[ ]:


make_label_encoder('Gender' , 'Gender Code')
data.head()


# 
# _____
# 
# # Data Standardization
# 
# ok , lets use the defined function above , to startdardize all features , except the output

# In[ ]:


for column in data.columns  : 
    if not column  =='Exited' :
        make_standardization(column)


# how it looks now ? 

# In[ ]:


data.head()


# ____
# 
# # Data Splitting
# 
# now we are ready to define X , y data
# 

# In[ ]:


X = data.drop(['Exited'], axis=1, inplace=False)
y = data['Exited']


# now split it using sklearn

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# _____
# 
# # Building the Model
# 
# there are several classifier models , let's start with LogisticRegression

# In[ ]:


LogisticRegressionModel = LogisticRegression(penalty='l2',solver='sag',C=1.0,random_state=33)
LogisticRegressionModel.fit(X_train, y_train)


# In[ ]:


print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train, y_train))
print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test, y_test))
print('LogisticRegressionModel Classes are : ' , LogisticRegressionModel.classes_)
print('LogisticRegressionModel No. of iteratios is : ' , LogisticRegressionModel.n_iter_)
print('----------------------------------------------------')


# not very good accuracy , let's see classification report for it

# In[ ]:


y_pred = LogisticRegressionModel.predict(X_test)
y_pred_prob = LogisticRegressionModel.predict_proba(X_test)
make_report()


# ___
# 
# how about Gaussian NB , will it helps ? 

# In[ ]:


GaussianNBModel = GaussianNB()
GaussianNBModel.fit(X_train, y_train)

print('GaussianNBModel Train Score is : ' , GaussianNBModel.score(X_train, y_train))
print('GaussianNBModel Test Score is : ' , GaussianNBModel.score(X_test, y_test))


# a little better , but we might catch something best

# In[ ]:


y_pred = GaussianNBModel.predict(X_test)
y_pred_prob = GaussianNBModel.predict_proba(X_test)
make_report()


# how about Decision Tree ? 

# In[ ]:


DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=33) #criterion can be entropy
DecisionTreeClassifierModel.fit(X_train, y_train)

print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))
print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))
print('DecisionTreeClassifierModel Classes are : ' , DecisionTreeClassifierModel.classes_)
print('DecisionTreeClassifierModel feature importances are : ' , DecisionTreeClassifierModel.feature_importances_)


# a slightly better

# In[ ]:


y_pred = DecisionTreeClassifierModel.predict(X_test)
y_pred_prob = DecisionTreeClassifierModel.predict_proba(X_test)

make_report()


# ___
# 
# now let's check SVC

# In[ ]:


SVCModel = SVC(kernel= 'sigmoid',# it can be also linear,poly,sigmoid,precomputed
               max_iter=1000,C=0.5,gamma='auto')
SVCModel.fit(X_train, y_train)

print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))
print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))


# oh , very far from calling it suitable , how about Random Forest ? 

# In[ ]:


RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=1000,max_depth=2,random_state=33) #criterion can be also : entropy 
RandomForestClassifierModel.fit(X_train, y_train)

print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))
print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))
print('RandomForestClassifierModel features importances are : ' , RandomForestClassifierModel.feature_importances_)


# 
# ____
# 
# 
# how about Gradient Boosting Classifier, will it be better ? 

# In[ ]:


GBCModel = GradientBoostingClassifier(n_estimators=100,max_depth=5,random_state=33) 
GBCModel.fit(X_train, y_train)

#Calculating Details
print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))
print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))


# Ok , better accuracy so we can focus in the model , let's check the 

# In[ ]:


y_pred = GBCModel.predict(X_test)
y_pred_prob = GBCModel.predict_proba(X_test)

print('Predicted Value for GBCModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for GBCModel is : ' , y_pred_prob[:10])


# In[ ]:


make_report()


# 
# ____
# 
# # Using GridSearch
# 
# since we see the GBClassifier is the most suitable model , let's use GridSearch tool to look for the best parameters for it

# In[ ]:


SelectedModel = GradientBoostingClassifier()
SelectedParameters = {'loss':('deviance', 'exponential'), 'max_depth':[1,2,3,4,5] , 'n_estimators':[50,75,100]}

GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters,cv = 5,return_train_score=True)
GridSearchModel.fit(X_train, y_train)
sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]


# Ok , how about the accuracy ? 

# In[ ]:


print('All Results are :\n', GridSearchResults )
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)
print('Best Estimator is :', GridSearchModel.best_estimator_)


# ____
# 
# ok better accuracy , let's use the best model to fit & predict the data

# In[ ]:


GBCModel = GridSearchModel.best_estimator_
GBCModel.fit(X_train, y_train)


# how about the score ? 

# In[ ]:


print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))
print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))
print('GBCModel features importances are : ' , GBCModel.feature_importances_)


# now let's use it to predict the test value

# In[ ]:


y_pred = GBCModel.predict(X_test)
y_pred_prob = GBCModel.predict_proba(X_test)

print('Predicted Value for GBCModel is : ' , y_pred)
print('Prediction Probabilities Value for GBCModel is : ' , y_pred_prob)


# no we can insert the predicted value to X_test

# In[ ]:


X_test.insert(10,'Predicted Valued',y_pred)


# & see the final Result

# In[ ]:


X_test.head(30)


# In[ ]:




