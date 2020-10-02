#!/usr/bin/env python
# coding: utf-8

# # IBM HR Analytics

# ## Dataset: [IBM HR Analytics Attrition Dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

# ### Import all the necessary header files as follows:

# **pandas** : An open source library used for data manipulation, cleaning, analysis and visualization. <br>
# **numpy** : A library used to manipulate multi-dimensional data in the form of numpy arrays with useful in-built functions. <br>
# **matplotlib** : A library used for plotting and visualization of data. <br>
# **seaborn** : A library based on matplotlib which is used for plotting of data. <br>
# **sklearn.metrics** : A library used to calculate the accuracy, precision and recall. <br>
# **sklearn.preprocessing** : A library used to encode and onehotencode categorical variables. <br>

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics


# ### Read the data from the dataset using the read_csv() function from the pandas library.

# In[ ]:


# Importing the dataset
data = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")


# ### Inspecting and cleaning the data

# In[ ]:


# Printing the 1st 5 columns
data.head()


# In[ ]:


# Printing the dimenions of data
data.shape


# In[ ]:


# Viewing the column heading
data.columns


# In[ ]:


# Inspecting the target variable
data.Attrition.value_counts()


# In[ ]:


data.dtypes


# In[ ]:


# Identifying the unique number of values in the dataset
data.nunique()


# In[ ]:


# Checking if any NULL values are present in the dataset
data.isnull().sum()


# In[ ]:


# See rows with missing values
data[data.isnull().any(axis=1)]


# In[ ]:


# Viewing the data statistics
data.describe()


# In[ ]:


# Here the value for columns, Over18, StandardHours and EmployeeCount are same for all rows, we can eliminate these columns
data.drop(['EmployeeCount','StandardHours','Over18','EmployeeNumber'],axis=1, inplace=True)


# ### Data Visualization

# In[ ]:


# Plotting a boxplot to study the distribution of features
fig,ax = plt.subplots(1,3, figsize=(20,5))               
plt.suptitle("Distribution of various factors", fontsize=20)
sns.boxplot(data['DailyRate'], ax = ax[0]) 
sns.boxplot(data['MonthlyIncome'], ax = ax[1]) 
sns.boxplot(data['DistanceFromHome'], ax = ax[2])  
plt.show()


# In[ ]:


# Finding out the correlation between the features
corr = data.corr()
corr.shape


# In[ ]:


# Plotting the heatmap of correlation between features
plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')


# In[ ]:


# Check for multicollinearity using correlation plot
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data[['DailyRate','HourlyRate','MonthlyIncome','MonthlyRate']].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


# Plotting countplots for the categorical variables
fig,ax = plt.subplots(2,3, figsize=(20,20))            
plt.suptitle("Distribution of various factors", fontsize=20)
sns.countplot(data['Attrition'], ax = ax[0,0]) 
sns.countplot(data['BusinessTravel'], ax = ax[0,1]) 
sns.countplot(data['Department'], ax = ax[0,2]) 
sns.countplot(data['EducationField'], ax = ax[1,0])
sns.countplot(data['Gender'], ax = ax[1,1])  
sns.countplot(data['OverTime'], ax = ax[1,2]) 
plt.xticks(rotation=20)
plt.subplots_adjust(bottom=0.4)
plt.show()


# In[ ]:


# Combine levels in a categorical variable by seeing their distribution
JobRoleCrossTab = pd.crosstab(data['JobRole'], data['Attrition'], margins=True)
JobRoleCrossTab


# In[ ]:


JobRoleCrossTab.div(JobRoleCrossTab["All"], axis=0)


# In[ ]:


# Combining job roles with high similarities together
data['JobRole'].replace(['Human Resources','Laboratory Technician'],value= 'HR-LT',inplace = True)
data['JobRole'].replace(['Research Scientist','Sales Executive'],value= 'RS-SE',inplace = True)
data['JobRole'].replace(['Healthcare Representative','Manufacturing Director'],value= 'HE-MD',inplace = True)


# In[ ]:


# Encoding Yes / No values in Attrition column to 1 / 0
data.Attrition.replace(["Yes","No"],[1,0],inplace=True)
data.head()


# In[ ]:


# One hot encoding for categorical variables
final_data = pd.get_dummies(data)
final_data.head().T


# In[ ]:


final_data.shape


# #### Once the data is cleaned, we split the data into training set and test set to prepare it for our machine learning model in a suitable proportion.

# In[ ]:


# Spliting target variable and independent variables
X = final_data.drop(['Attrition'], axis = 1)
y = final_data['Attrition']


# In[ ]:


# Splitting the data into training set and testset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0, stratify=y)


# In[ ]:


y_train.value_counts()


# In[ ]:


# Checking distribtution of Target varaible in training set
y_train.value_counts()[1]/(y_train.value_counts()[0]+y_train.value_counts()[1])*100


# In[ ]:


y_test.value_counts()


# In[ ]:


# Checking distribtution of Target varaible in test set
y_test.value_counts()[1]/(y_test.value_counts()[0]+y_test.value_counts()[1])*100


# ### Logistic Regression

# In[ ]:


# Logistic Regression

# Import library for LogisticRegression
from sklearn.linear_model import LogisticRegression

# Create a Logistic regression classifier
logreg = LogisticRegression()

# Train the model using the training sets 
logreg.fit(X_train, y_train)


# In[ ]:


# Prediction on test data
y_pred = logreg.predict(X_test)


# In[ ]:


# Calculating the accuracy, precision and the recall
acc_logreg = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Total Accuracy : ', acc_logreg )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# In[ ]:


# Create confusion matrix function to find out sensitivity and specificity
from sklearn.metrics import auc,confusion_matrix
def draw_cm(actual, predicted):
    cm = confusion_matrix( actual, predicted, [1,0]).T
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Yes","No"] , yticklabels = ["Yes","No"] )
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.show()


# In[ ]:


# Confusion matrix 
draw_cm(y_test, y_pred)


# ### Gaussian Naive Bayes

# In[ ]:


# Gaussian Naive Bayes

# Import library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(X_train,y_train)


# In[ ]:


# Prediction on test set
y_pred = model.predict(X_test)


# In[ ]:


# Calculating the accuracy, precision and the recall
acc_nb = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Total Accuracy : ', acc_nb )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# In[ ]:


# Confusion matrix 
draw_cm(y_test, y_pred)


# ### Decision Tree Classifier

# In[ ]:


# Decision Tree Classifier

# Import Decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# Create a Decision tree classifier model
clf = DecisionTreeClassifier(criterion = "gini" , min_samples_split = 100, min_samples_leaf = 10, max_depth = 50)

# Train the model using the training sets 
clf.fit(X_train, y_train)


# In[ ]:


# Model prediction on train data
y_pred = clf.predict(X_train)


# In[ ]:


# Finding the variable with more importance
feature_importance = pd.DataFrame([X_train.columns, clf.tree_.compute_feature_importances()])
feature_importance = feature_importance.T.sort_values(by = 1, ascending=False)[1:10]


# In[ ]:


sns.barplot(x=feature_importance[1], y=feature_importance[0])
# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[ ]:


# Prediction on test set
y_pred = clf.predict(X_test)


# In[ ]:


# Confusion matrix
draw_cm(y_test, y_pred)


# In[ ]:


# Calculating the accuracy, precision and the recall
acc_dt = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Total Accuracy : ', acc_dt )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# ### Random Forest Classifier

# In[ ]:


# Random Forest Classifier

# Import library of RandomForestClassifier model
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
rf = RandomForestClassifier()

# Train the model using the training sets 
rf.fit(X_train,y_train)


# In[ ]:


# Finding the variable with more importance
feature_imp = pd.Series(rf.feature_importances_,index= X_train.columns).sort_values(ascending=False)
# Creating a bar plot
feature_imp=feature_imp[0:10,]
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[ ]:


# Prediction on test data
y_pred = rf.predict(X_test)


# In[ ]:


# Confusion metrix
draw_cm(y_test, y_pred)


# In[ ]:


# Calculating the accuracy, precision and the recall
acc_rf = round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 )
print( 'Total Accuracy : ', acc_rf )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100 , 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# ### Support Vector Machine Classifier

# In[ ]:


# SVM Classifier

# Creating scaled set to be used in model to improve the results
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# Import Library of Support Vector Machine model
from sklearn import svm

# Create a Support Vector Classifier
svc = svm.SVC()

# Train the model using the training sets 
svc.fit(X_train,y_train)


# In[ ]:


# Prediction on test data
y_pred = svc.predict(X_test)


# In[ ]:


# Confusion Matrix
draw_cm(y_test, y_pred)


# In[ ]:


# Calculating the accuracy, precision and the recall
acc_svm = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Total Accuracy : ', acc_svm )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# ## Evaluation and comparision of all the models

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Support Vector Machines'],
    'Score': [acc_logreg, acc_nb, acc_dt, acc_rf, acc_svm]})
models.sort_values(by='Score', ascending=False)


# ## Hence we can see that the Logistic Regression works the best for this dataset. 

# ### Please upvote if you found this kernel useful! :) <br>
# ### Any sort of feedback is appreciated!
