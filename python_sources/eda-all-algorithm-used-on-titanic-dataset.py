#!/usr/bin/env python
# coding: utf-8

# **Titanic prediction using data science lifecycle**
# 
# With this notebook, my objective is to follow a complete process of data science project and explain each step and rationale behind every decision we take during development of solution.

# ![data%20science%20lifecycle.png](attachment:data%20science%20lifecycle.png)
# 
# I found this amazing infographics in data science lifecycle. You can click on the reference link below to read more about it. 
# 
# 
# Reference: http://sudeep.co/data-science/Understanding-the-Data-Science-Lifecycle/

# **Business Understanding**
# 
# It is the most important part of the data science cycle as more than 50% fail because they do not understand the problem properly. We are not subject matter expert so this part becomes even more important and chalenging. 
# 
# **Titanic: Machine learning from disaster** 
# 
# Titanic was one of the biggest ship to be built in 1912. 
# 
# - On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. It means 31.46 survival rate.
# - The shortage of lifeboats for the passengers and crew led to such loss of life. 
# - Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# - Our aim is to predict which passengers survived the tragedy. 

# **Data Mining**
# 
# Starting from the available data, we should always keep looking features/ data that can improve our prediction. I could not find much data which can influence our predictions. Therefore, we will work with available data. We have train and test files. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #for visualisation
import seaborn as sns   #for visualisation
sns.set(style="white") #setting background of vizualisation as white 
get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#using pandas to read the train and test file \ntrain= pd.read_csv('/kaggle/input/titanic/train.csv')\ntest= pd.read_csv('/kaggle/input/titanic/test.csv')")


# Noting down train and test rows to separate it easily when required

# In[ ]:


print(train.shape)
print(test.shape)


# **Data Exploration**
# 
# Data exploration is an approach similar to initial data analysis, whereby a data analyst uses visual exploration to understand what is in a dataset and the characteristics of the data, rather than through traditional data management systems. We will now try to understand the train data by forming hypothesis such female and higher class passengers were most likely to survive the tragedy

# In[ ]:


train.shape, train.columns


# We have 11 columns and 891 rows in our train data where 1 column is our target variable.

# In[ ]:


train.head() #to understand the type of values in these columns


# In[ ]:


#Lets now understand our target variable that out 891 how many survived or died in this tragedy
f,ax=plt.subplots(1,2,figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Died vs Survived percentage')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Died vs Survived Count')
ax[1].set_xlabel('')
plt.show()


# So, 61.6% passengers in train dataset survived and 38.3% died. We know from the movie that Female were given preference over male. Lets check..

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex Percentage')
sns.countplot(x='Survived',hue='Sex',data=train,ax=ax[1])
ax[1].set_title('Survived vs Sex')
plt.legend(loc = 'top left')


# We can clearly see that almost 75% female survived the tragedy. Lets check if class had an impact on survival or not 

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Class Percentage')
sns.countplot(x='Survived',hue='Pclass',data=train,ax=ax[1])
ax[1].set_title('Survived vs Class')
plt.legend(loc = 'top left')


# Clearly buying higher class ticket had more than 60% survival chance and only 25% class 3 passengers could survive. Lets see what are the chance for female passenger of Pclass 1, probably 100%..Lets check on a factor plot..

# In[ ]:


sns.factorplot('Pclass','Survived',hue='Sex',data=train)#This clearly summarises our hypothesis


# Lets check another hypothesis, was class a factor for children survival as well..

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])#lets check using violin plot
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# survival rate for children under 10 looks irrespective of class but we not be certain as we have missing values in age column. Therefore, it requires further analysis.

# In[ ]:


#Lets plot survival vs rest of the variable 
variable = ['Embarked', 'Parch', 'SibSp']

fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(10, 8))
plt.subplots_adjust(right=1.5, top=1.25)

for i, v in enumerate(variable, 1):    
    plt.subplot(1, 3, i)
    sns.countplot(x=v, hue='Survived', data=train)
    
    plt.xlabel('{}'.format(v), size=10, labelpad=10)
    plt.ylabel('Passenger Count', size=10, labelpad=10)    
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    
    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 10})
    plt.title('Count of Survival in {} Feature'.format(v), size=10, y=1.05)

plt.show()

Lets now merge both train and test data to perform data preprocessing together. After preprocessing, we will separate the train and test file again. 
# In[ ]:


get_ipython().run_cell_magic('time', '', 'titanic_df = pd.concat([train, test])\nprint(titanic_df.shape) #lets check how concatenation worked\ntitanic_df.head()   #lets check the first 5 rows now')


# **Data Cleaning/ treatment**
# 
# We cannot expect data to be clean and readily available for modelling. Therefore, we will try to understand type of data and find if they have missing values

# In[ ]:


titanic_df.dtypes #Lets check the type of data we are dealing with...


# We can now see that 
# 
# ![image.png](attachment:image.png)

# In[ ]:


titanic_df.describe() #summarizes all numerical columns


# We can clear make out from the available data that Age column has missing values. The mean of the age is 29 but max age is upto 80 years. Similarly average fare is 33 and someone paid 512 as well. It also can be infered that the number sibling and parents are less and max is 8 and 6 repsectively.

# In[ ]:


#lets check data has how many missing values
titanic_df.isnull().sum()


# Therefore, age of people are missing along with cabin details. We will now try to handle these missing values. We can basically treat missing values in three ways. 
# 
# * 1. Remove rows or columns containing missing values
# * 2. Fill missing values with test statistics such as mean, median, mode etc.
# * 3. Predict Missing value with machine learning algorithm such as regression, KNN
# 
# Each one has its own pros and cons. Type of data is one of the keys parameters of choosing the treatment methodology. We should always try different methodology and see if our accuracy improves. 

# We can see that we have Name column and it has salutation. It will be insteresting to impute on the basis of their salutation. 

# In[ ]:


titanic_df['Title'] = titanic_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
titanic_df.head()


# In[ ]:


titanic_df['Title'].value_counts()


# We can now see that we have alot of rare titles. Lets club them

# In[ ]:


mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',
           'Don': 'Mr', 'Mme': 'Mrs', 'Jonkheer': 'Mr', 'Lady': 'Mrs',
           'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
titanic_df.replace({'Title': mapping}, inplace=True)
titanic_df['Title'].value_counts()#left with only 6 titles now


# In[ ]:


# impute missing Age values using median of Title groups
title_ages = dict(titanic_df.groupby('Title')['Age'].median())


titanic_df['age_med'] = titanic_df['Title'].apply(lambda x: title_ages[x])

# replace all missing ages with the value in this column
titanic_df['Age'].fillna(titanic_df['age_med'], inplace=True, )
del titanic_df['age_med']
sns.distplot(titanic_df['Age'])


# In[ ]:


sns.barplot(x='Title', y='Age', data=titanic_df, estimator=np.median, ci=None, palette='Blues_d')
plt.xticks(rotation=45)
plt.show()


# We can also impute mean using sklearn library

# In[ ]:


titanic_df.isnull().sum()


# In[ ]:


get_ipython().run_cell_magic('time', '', "#from sklearn.preprocessing import Imputer\n#impute = Imputer(missing_values='NaN', strategy='mean', axis=0) #SKlearn Mean imputer\n#impute.fit(titanic_df['Age'])\n#titanic_df= impute.transform(titanic_df['age'])\n\n#from fancyimpute import KNN\n#imputer = KNN(k=2)\n#titanic_df['Age'] = imputer.fit_transform(titanic_df['Age'])")


# In[ ]:


sns.heatmap(titanic_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
#titanic_df.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# impute missing Fare values using median of Pclass groups\nclass_fares = dict(titanic_df.groupby('Pclass')['Fare'].median())\n\n# create a column of the average fares\ntitanic_df['fare_med'] = titanic_df['Pclass'].apply(lambda x: class_fares[x])\n\n# replace all missing fares with the value in this column\ntitanic_df['Fare'].fillna(titanic_df['fare_med'], inplace=True, )\ndel titanic_df['fare_med']")


# In[ ]:


#lets check data has how many missing values
titanic_df.isnull().sum()


# In[ ]:


titanic_df['Embarked'].value_counts()


# In[ ]:


#As Embarked is a categorical variable only 2 missing values so we will simply use backfill    
titanic_df['Embarked'].fillna(method='backfill', inplace=True)
titanic_df['Embarked'].isnull().sum()


# **Feature Engineering**- It is one of the key process and can become a differentiator for your model. Feature engineering is the process of using domain knowledge to extract features from raw data via data mining techniques. These features can be used to improve the performance of machine learning algorithms. Feature engineering can be considered as applied machine learning itself.
# source- Wikipedia

# In[ ]:


#we have number of parents and no.of siblings so we can calculate the size of the family 
titanic_df['Family Size'] = titanic_df['Parch'] + titanic_df['SibSp']+1
titanic_df['IsAlone'] = 1 #initialize to yes/1 is alone
titanic_df['IsAlone'].loc[titanic_df['Family Size'] > 1] = 0
titanic_df.columns


# Looks great...

# In[ ]:


#We will now drop column which have mostly Null and unimportant values. 
drop = ['Cabin', 'Ticket','Name','PassengerId']
titanic_df.drop(drop, axis=1, inplace = True)
titanic_df.head()


# We have string value in 3 columns and SKlearn doesnot support strings. Therefore, we can change them categorical values as 0,1,2 etc. or use onehot encoding 

# In[ ]:


titanic_df['Sex'].replace(['male','female'],[0,1],inplace=True)
titanic_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
titanic_df['Title'].replace(['Mr','Mrs','Miss','Master','Dr','Rev'],[0,1,2,3,4,5],inplace=True)

#you can also use get dummies or label encoder


# In[ ]:


titanic_df.head()


# **Predictive Modelling**
# 
# Lets start building models to predict our target variable. We will start will KNN and gradually move towards more complex models 

# In[ ]:


#Import all neccessary SKlearn libraries for KNN classifier
from sklearn.preprocessing import MinMaxScaler#to make sure all variables are on the same scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score
from sklearn import metrics


# In[ ]:


y_train=titanic_df['Survived'].iloc[:891]#training target variable
x_train=titanic_df.drop('Survived', axis=1)#dropping target variable
#y_train.tail()
x_train.tail()


# In[ ]:


scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_train)
x_scaled = pd.DataFrame(x_scaled, columns = x_train.columns)
x_scaled.tail()#All variables are now in same scale of 0 to 1 


# In[ ]:


from sklearn.model_selection import train_test_split
train_scaled= x_scaled.iloc[:891] #train dataset
#train_scaled.shape
test_scaled=x_scaled.iloc[891:]#final dataset
#test_scaled.shape


# In[ ]:


#Creating training and test dataset from from training dataset
train_x,test_x,train_y,test_y = train_test_split(train_scaled,y_train, random_state = 56, stratify=y_train)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Creating instance of KNN\nclf = KNN(n_neighbors = 10)\n\n# Fitting the model\nclf.fit(train_x, train_y)\n\n# Predicting over the Train Set and calculating accuracy score\ntest_predict = clf.predict(test_x)\nprint('The accuracy of the KNN is',metrics.accuracy_score(test_predict,test_y))\nfinal_predict1=clf.predict(test_scaled)\nprint(final_predict1)")


# Lets now try to predict using logistic regression

# # **MAX Vote**
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Max vote takes prediction from multiple models and provide its prediction on the basis of maximum votes

# In[ ]:


#Running logistics Regression model
lr = LogisticRegression()
lr.fit(train_x,train_y)
#making predicition on test set created from train_scaled data
valid1=lr.predict(test_x)
print(lr.score(test_x, test_y))
pred1=lr.predict(test_scaled)
pred1[:10]


# In[ ]:


#Running KNN model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_x,train_y)
#making predicition on test set created from train_scaled data
valid2=knn.predict(test_x)
print(knn.score(test_x, test_y))
pred2=knn.predict(test_scaled)
pred2[:10]


# In[ ]:


#Running Decision tree model
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(train_x,train_y)
#making predicition on test set created from train_scaled data
valid3=dt.predict(test_x)
print(dt.score(test_x, test_y))
pred3=dt.predict(test_scaled)
pred3[:10]


# In[ ]:


from sklearn.metrics import accuracy_score
from statistics import mode
final_pred = np.array([])
for i in range(0,len(test)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))

final_pred=final_pred.astype(int)
final_pred.shape


# # ** Basic Stacking**

# In[ ]:


#Creating a New train dataframe
train_prediction = {
              'LR': valid1,
              'knn': valid2,
              'DT': valid3
              }
train_predictions = pd.DataFrame(train_prediction)
train_predictions.shape, test_y.shape


# In[ ]:


#Creating a New test dataframe
test_prediction = {
              'LR': pred1,
              'knn': pred2,
              'DT': pred3
              }
test_predictions = pd.DataFrame(test_prediction)
test_predictions.head()


# In[ ]:


# Stacking Model
model = LogisticRegression()
model.fit(train_predictions, test_y)
final_pred=model.predict(test_predictions)
final_pred=final_pred.astype(int)
final_pred[:10],final_pred.dtype


# # **Stacking Using Kfold**

# In[ ]:


from sklearn.model_selection import KFold
train_pred = np.empty((0,0) , int)
skfold = KFold(10, random_state = 101)
  
#For every permutation of KFold
for i,j in skfold.split(train_x, train_y):
    x_train, x_test = train_x.iloc[i], train_x.iloc[j]
    y_train, y_test = train_y.iloc[i], train_y.iloc[j]

#Everything else remains same as regular stacking

#Running logistics Regression model
lr = LogisticRegression()
lr.fit(x_train,y_train)
#making predicition on test set created from train_scaled data
valid1=lr.predict(x_test)
print(lr.score(x_test, y_test))
pred1=lr.predict(test_scaled)


#Running KNN model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)
#making predicition on test set created from train_scaled data
valid2=knn.predict(x_test)
print(knn.score(x_test, y_test))
pred2=knn.predict(test_scaled)


#Running Decision tree model
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(x_train,y_train)
#making predicition on test set created from train_scaled data
valid3=dt.predict(x_test)
print(dt.score(x_test, y_test))
pred3=dt.predict(test_scaled)


#Creating a New train dataframe
train_prediction = {
              'LR': valid1,
              'knn': valid2,
              'DT': valid3
              }
train_predictions = pd.DataFrame(train_prediction)



#Creating a New test dataframe
test_prediction = {
              'LR': pred1,
              'knn': pred2,
              'DT': pred3
              }
test_predictions = pd.DataFrame(test_prediction)



# Stacking Model
model = KNeighborsClassifier(n_neighbors=10)
model.fit(train_predictions, y_test)
final_pred=model.predict(test_predictions)
final_pred=final_pred.astype(int)
final_pred[:10],final_pred.dtype


# # **Bagging Ensemble model- Random Forest**

# In[ ]:


#Importing random forest classifier 
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


train_accuracy = []
validation_accuracy = []
for depth in range(1,10):
    rfc_model = RandomForestClassifier(max_depth=depth, random_state=10)
    rfc_model.fit(train_x, train_y)
    train_accuracy.append(rfc_model.score(train_x, train_y))
    validation_accuracy.append(rfc_model.score(test_x, test_y))


# In[ ]:


table = pd.DataFrame({'max_depth':range(1,10), 'train_acc':train_accuracy, 'test_acc':validation_accuracy})
table


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(table['max_depth'], table['train_acc'], marker='o')
plt.plot(table['max_depth'], table['test_acc'], marker='o')
plt.xlabel('Depth of tree')
plt.ylabel('performance')
plt.legend()


# In[ ]:


#creating a random forest instance
RFC = RandomForestClassifier(random_state=10,max_depth=7, max_leaf_nodes=25,n_estimators=12)
RFC.fit(train_x,train_y)
pred4=dt.predict(test_scaled)
final_pred=pred4.astype(int)
final_pred[:10],final_pred.dtype


# # **Boosting Technique- GBC**

# In[ ]:


#Importing GBDT Classifier 
from sklearn.ensemble import GradientBoostingClassifier
gbc= GradientBoostingClassifier(random_state=101)


# In[ ]:


parameter_grid = {
    'max_depth' : [4,5,6,7,8],
    'n_estimators': [100,150,200, 250],
    'min_samples_split': [50,100,150,200]
    }


# In[ ]:


from sklearn.model_selection import GridSearchCV
gridsearch = GridSearchCV(estimator=gbc, param_grid=parameter_grid, scoring='neg_mean_squared_error', cv=5)


# In[ ]:


gridsearch.fit(train_x, train_y)


# In[ ]:


gridsearch.best_params_


# In[ ]:


#creating an Gradient boosting instance
gbc= GradientBoostingClassifier(random_state=101, n_estimators=150,min_samples_split=100, max_depth=6)
gbc.fit(train_x,train_y)
print(gbc.score(test_x, test_y))
pred5=gbc.predict(test_scaled)
final_pred=pred5.astype(int)


# In[ ]:


#Output the predictions into a csv
output= pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':final_pred})
output.to_csv('gender_submission.csv', index=False)

