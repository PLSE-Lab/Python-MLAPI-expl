#!/usr/bin/env python
# coding: utf-8

# Project Overview
# 
# We are going to apply a predictive algorithm to a political dataset where you classify the party affiliation of United States congressmen based on their voting records.We'll be working with a dataset obtained from the UCI Machine Learning Repository consisting of votes made by US House of Representatives Congressmen. Our goal will be to predict their party affiliation ('Democrat' or 'Republican') based on how they voted on certain key issues. 

# Citation
# 
# I have added notes for my own learning as I worked through this dataset. These notes are from the learning videos of Data School and the instructor is Kevin Markham (@justmarkham)

# In[ ]:


#Getting Setup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


#Load the Data
df = pd.read_csv('../input/housingvotes/house-votes-84.csv',index_col=None,header=None,na_values = '?')
df.columns = ['Party',
   'Infants', 
   'Water-project',
   'Budget',
   'Physician',
   'El-salvador',
   'Religion',
   'Satellite',
   'Nicaragua',
   'Missile',
   'Immigration',
   'Synfuels',
   'Education',
   'Superfund',
   'Crime',
   'Duty-free',
   'Export']


# In[ ]:


#Describe the data
df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


#View the count of Nans in each factor

df.isnull().sum().sort_values(ascending=False)


# In[ ]:


#Examining the datatypes
df.dtypes


# In[ ]:


#Make a copy of the original dataset and work on the new copy
df1=df.copy(deep=True)


# In[ ]:


#Leaving this code in even though it is not required. This was to replace the missing values (?) with np.nan but giving the parameter 'na_values = ?' while loading the dataset will do #the job

#df1.replace('?',np.nan,inplace=True)


# In[ ]:


#Replace 'y' by 1 and'n' by 0 (By running the replace command, the data type of all the factors got converted from object to flaoatint64. 

df1.replace({'n': 0,'y': 1},inplace=True)
print(df1.head())
df1.dtypes


# In[ ]:


#As Export column has missing values for a large percentage of records this column will be dropped
df1.drop(['Export'],axis=1,inplace=True)


# In[ ]:


# Visualizing the highest predictors for each party

df1.head()
df2=df1.melt(id_vars = 'Party',
         var_name = 'Predictors',
         value_name = 'Vote'
          )
df2.head()

table=df2.groupby(['Predictors','Party'])['Vote'].mean().unstack()
table


ax=table.plot(kind='bar',width=0.60,figsize=(15,10))
plt.ylabel('Average Vote')
plt.title('Probability of Voting')
plt.grid(axis='x')

#Got this from Stackoverflow, #Try this later )
#sns.set()
#df.set_index('App').T.plot(kind='bar', stacked=True)


# In[ ]:


#For the remaining factors with missing values I created a function that will display a countplot for each party. This will help with choosing what value to fill missing values in for each factor.For eg, here's a sample plot for the factor Duty-free. The Republican party is marked by Red and Democratic by blue. As you can see there is a resounding difference in the way each party voted for Duty-free. Similarly I've created count plots for all other factors
plt.figure()
sns.countplot(x='Duty-free', hue='Party', data=df1, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()


# In[ ]:


# function to display countplot for a factor i.e Democrate Vs Republicans on each issue/factor

def countplot(column, dataframe):
    plt.figure()
    sns.countplot(x=dataframe[column], hue='Party', data=dataframe, palette='RdBu')
    plt.title(column)
    plt.xticks([0,1], ['No', 'Yes'],rotation=45)
    plt.show()


# In[ ]:


# I've taken out 'Party' from the list of columns to be plotted as that is our response vector
factor_cols = df1.columns[1:]
factor_cols


# In[ ]:


# Iterate the countplot for every factor in the dataset
for col in factor_cols:
      countplot(col,df1)


# In[ ]:


#The countplot for Water Project shows that equal number of democrats and Republicans voted against and in favour and therefore is inconclusive. Therefore I'm taking this factor out of the dataframe for the predictive model and and reset factor_cols to get the list of column names from the latest modified dataset. 
df1.drop(['Water-project'],axis=1,inplace=True)
factor_cols = df1.columns[1:]


# In[ ]:


#Check to see if Water-project was removed
print(df1.head())


# In[ ]:


#For all other factors I've updated the missing values with the mode or the value that most people voted for. The function below 
#1. Reads every row of the dataset df1
#2. Gets the corresponding 'Party' of that row
#3. Calculates the mode of that column for rows pertaining to the party from step #2
#4. Updates the value in that row with the mode obtained through step 3 if there is a missing value. Ideally, iterrows meathod does not allow to update the original dataframe, so I used the command 'dataframe.at' to do an inplace update

#This function is then iterated over every column of the dataset


# In[ ]:


# function to retrive and update the mode ( most frequent value) of the party for a factor

def fillpartymode(column,dataframe) :
    for index,row in dataframe.iterrows():
 #      print(row)
        party = row['Party']
 #      print ("Party is ", party)
        df1_subset = df1[df1['Party'] == party].reset_index() 
        value = df1_subset.loc[:,column].mode()
        if pd.isna(row[column]):
 #          print("Filled Nan with ", value) 
            dataframe.at[index, column] = value
            
           


# In[ ]:


# Iterate the fillna module for every factor in the dataset
for col in factor_cols:
      print('Column:',col)  
      fillpartymode(col,df1)
      print (df1[col].value_counts(dropna=False)) 
      


# In[ ]:


#Check to see if there are still missing values. The function din't seem to work for the first column Infants ( odd !). So I had to update the missing value in Infants with the median value
df1.isnull().sum()
#df1.loc[:, df1.isnull().any()] (lists the column with null values)


# In[ ]:


#Now that we've cleaned the dtaaset and filled the missing values, It is time to build the classifier.I've used a k-Nearest Neighbors classifier to the voting dataset, 
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


#First let's train and test the entire dataset

#1. Train the model on the entire dataset.
#2. Test the model on the same dataset, and evaluate how well we did by comparing the predicted response values with the true response values.


# In[ ]:


# Create arrays for the features and the response variable
y = df1['Party'].values
X = df1.drop('Party', axis=1).values


# In[ ]:


#Create an instance of the estimator. 
knn = KNeighborsClassifier(n_neighbors=6)


# In[ ]:


#Fit the model with the data
knn.fit(X,y)


# In[ ]:


# predict the response value for the observations in X
y_pred = knn.predict(X)


# In[ ]:


# check how many predictions were generated
len(y_pred)


# In[ ]:


# compute classification accuracy for the knn classifier model (Known as training accuracy when you train and test the model on the same data)
from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))


# #The Accuracy score is 96%
# #Problems with training and testing on the same data
# #1. Goal is to estimate likely performance of a model on out-of-sample data
# #2. But, maximizing training accuracy rewards overly complex models that won't necessarily generalize
# #3. Unnecessarily complex models overfit the training data
# 
# #Evaluation procedure #2: Train/test split
# #1. Split the dataset into two pieces: a training set and a testing set.
# #2. Train the model on the training set.
# #3. Test the model on the testing set, and evaluate how well we did.

# In[ ]:


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)


# In[ ]:


# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 6)


# In[ ]:


# Fit the classifier to the training data
knn.fit(X_train,y_train)


# In[ ]:


#Make predictions on the training set
y_pred = knn.predict(X_test)


# In[ ]:


# compute classification accuracy for the knn classifier model
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:


# Print the accuracy (Another way to get the Accuracy score)
print(knn.score(X_test, y_test))


# In[ ]:


#Repeat for knn = 7


# In[ ]:


# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

#Make predictions on the training set
y_pred = knn.predict(X_test)

# compute classification accuracy for the knn classifier model
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:


# try K=1 through K=25 and record testing accuracy
k_range = list(range(1, 20))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# #1. Training accuracy rises as model complexity increases
# #2. Testing accuracy penalizes models that are too complex or not complex enough
# #3. For KNN models, complexity is determined by the value of K (lower value = more complex). In this case 12 neighbours or more creates a simple model that neither underfits nor overfits #the data. I've tried once again with knn=12

# In[ ]:


# Create a k-NN classifier with 12 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 12)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

#Make predictions on the training set
y_pred = knn.predict(X_test)

# compute classification accuracy for the knn classifier model
print(metrics.accuracy_score(y_test, y_pred))


# In[ ]:


#Evaluting the model using the Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


#Building a Logistic Regression Model
from sklearn.linear_model import LogisticRegression

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# compute classification accuracy for the knn classifier model
print(metrics.accuracy_score(y_test, y_pred))

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


# I tested with an out-of-sample observation. My sample is modeled on Republican votes.
X_new = [[0,0,1,1,1,0,0,0,1,0,1,1,1,0]]
y_pred = knn.predict(X_new)
y_pred


# In[ ]:


y_test.shape

