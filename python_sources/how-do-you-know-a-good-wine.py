#!/usr/bin/env python
# coding: utf-8

# # Predicting Good Wine With ML Algorithms

# Welcome to this notebook of predicting the best wine quality using provided dataset. In this notebook, we will have the following parts:
# 
# * Introduction
# * Exploratory Data Analysis
# * Data Pre-processing
# * Creating a Models
# * Improving Model Performance
# * Conclusion
# * Credits & Aspirations
# 
# Let us get started.

# # Introduction
# 
# If you thought to work on exploring more on Red Wine Quality, it is probably that you don't have any problem having a wine(Same for me, I may not be right on this saying). Having that in mind, let us start to see what makes good wine. Each wine is rated as good or bad on scale of 0 to 10. Good wine start from 7 to 10 whereas bad wine is less than 7 based on label of quality. 
# 
# Let us start by importing relevant libraries. I will import all libraries to use all over the notebook. 

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# Also, let us import the data that we are going to be working with. 

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


wine_data=pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")


# Let us see the variables that we are going to be working with. 

# In[ ]:


wine_data.columns


# In[ ]:


print("Rows,columns:" + str(wine_data.shape))


# The data has 1599 rows and 12 columns. Let us quickly use 3 methods to explore our data, which are info, describe, and head. 

# In[ ]:


wine_data.info()


# In[ ]:


wine_data.describe().transpose()


# In[ ]:


wine_data.head(5)


# # Exploratory Data Analysis
# 
# In this section, we will check if our data is ready for further processing, perform visualization with seaborn, and clean it (where necessary). 
# 
# Let us start by checking the missing values

# In[ ]:


wine_data.isnull().sum()
#Same as wine_data.isna().sum()


# Now that we are pretty sure our data is okay, we can start understanding the data by using the power of visualizations.

# In[ ]:


#From this plot, we see that most wine types are in 5 and 6 category, which means most are considered bad. 
plt.figure(figsize = (11,6))
sns.countplot(data=wine_data, x='quality')


# In[ ]:


# Let us see the relationship between the label "quality" and other variables.
plt.figure(figsize = (11,6))
sns.heatmap(wine_data.corr(), 
            xticklabels=wine_data.corr().columns, 
            yticklabels=wine_data.corr().columns, 
            annot=True, 
            cmap=sns.diverging_palette(220,20,
            as_cmap=True))


# Now we see that the most 3 features which are very linked to the quality of the wine are alcohol, citric acid, and sulphates.

# In[ ]:


#PH has a correlation of -0.058. Now plotting it is obvious that wine quality doesn't depend on pH. 
#PH is Power of Hydrogen which is a scale used to specify how acidic or basic a water-based solution is
plt.figure(figsize = (11,6))
sns.barplot(data=wine_data, x='quality',y='pH')


# In[ ]:


#Alcohol is the most correlated feature with the quality of the wine, hence a reason why here good wine has high alcohol
plt.figure(figsize = (11,6))
sns.barplot(data=wine_data, x='quality',y='alcohol')


# In[ ]:


#Citric acid also correlates with the quality of the wine, hence a reason why here good wine has high citric acid
plt.figure(figsize = (11,6))
sns.barplot(data=wine_data, x='quality',y='citric acid')


# In[ ]:


#Another feature to explore is sulphates. 
plt.figure(figsize = (11,6))
sns.barplot(data=wine_data, x='quality',y='sulphates')


# In[ ]:


#Also, we can see that the quality of the wine does not depend on the residual sugar
#Alcohol is the most correlated feature with the quality of the wine, hence a reason why here good wine has high alcohol
plt.figure(figsize = (11,6))
sns.barplot(data=wine_data, x='quality',y='residual sugar')


# # Data Preprocessing

# We've see that the wine quality is rated from 0 to 10. I will make a column of good quality, where wine from 0 to 6 or less than 7 will be labelled 0, and else 1. [1]

# In[ ]:


wine_data['good quality']=[1 if x>=7 else 0 for x in wine_data['quality']]


# In[ ]:


#plt.figure(figsize = (11,6))
sns.countplot(data=wine_data, x='good quality')


# In[ ]:


#The exact values of wine quality. 0 stands for bad wine, whereas 1 is for good wine.
wine_data['good quality'].value_counts()


# In[ ]:


#Let us drop quality off the dataset

wine_data=wine_data.drop('quality',axis=1)


# In[ ]:


#Now let us seperate the dataset as response variable and label or target variable
X = wine_data.drop('good quality', axis = 1)
y = wine_data['good quality']


# In[ ]:


#Splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[ ]:


#Applying Standard scaling to get optimized result
scaler = StandardScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# # Creating a Model

# I wiss compare three Machine learning algotithms namely SVC(Support Vector Classifier), Random Forest Classifier,and Logistic Regression. 
# 
# # With Support Vector Classifier **(SVC)

# In[ ]:


model1=SVC()


# In[ ]:


model1.fit(X_train, y_train)


# In[ ]:


pred_svc=model1.predict(X_test)


# In[ ]:


print(classification_report(y_test, pred_svc))


# The SVC Model achieved 86% accuracy. 

# # With Random Forest Classifier

# In[ ]:


model2 = RandomForestClassifier(n_estimators=200)
model2.fit(X_train, y_train)
pred_rfc = model2.predict(X_test)


# In[ ]:


print(classification_report(y_test, pred_rfc))


# The Random Forest Classifier Model achieved 88% accuracy. 

# In[ ]:


#Confusion matrix
print(confusion_matrix(y_test, pred_rfc))


# # With Logistic Regression

# In[ ]:


lrmodel=LogisticRegression()
lrmodel.fit(X_train,y_train)
logpred=lrmodel.predict(X_test)
print(classification_report(y_test, logpred))


# Looking on the classification report, the model achieved 86% of accuracy.

# # Improving Model Performance[2]

# In this section, we will make the use two powerful options for improving model performance, which are Grid search and Cross Validation.

# In[ ]:


#Finding best parameters for our SVC model
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(model1, param_grid=param, scoring='accuracy', cv=10)


# In[ ]:


grid_svc.fit(X_train, y_train)


# In[ ]:


grid_svc.best_params_


# In[ ]:


#Let's run SVC again with the best parameters.
model_svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
model_svc2.fit(X_train, y_train)
pred_svc2 = model_svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))


# As seen on above report, the model with the best parameters achieved 90% accuracy from 86%.

# # Cross Validation Score for random forest and SVC

# Now, let us attempt to improve the model performance by using cross validation.
# 
# > Cross Validation is used to assess the predictive performance of the models and and to judge how they perform outside the sample to a new data set also known as test data. The motivation to use cross validation techniques is that when we fit a model, we are fitting it to a training dataset.  (Full definition by Research Gate)

# In[ ]:


rand_forest_val = cross_val_score(estimator = model2, X = X_train, y = y_train, cv = 10)
rand_forest_val.mean()


# As we now see, the model accuracy achieved with cross validating the model is 91% from 88%

# In[ ]:


svc_val = cross_val_score(estimator = model1, X = X_train, y = y_train, cv = 10)
svc_val.mean()


# Also, as seen above, by assessing the model to unseen data or test set (Cross validation), The SVC model accuracy increased from 86% to 89% 

# In[ ]:


log_val = cross_val_score(estimator = lrmodel, X = X_train, y = y_train, cv = 10)
log_val.mean()


# By doing the same thing on the Logistic model, the accuracy increase 2%, which is good improvement. 

# # Conclusion

# Jumping on the start of our work, the goal was to predict the good quality of the wine given the data of what makes the good wine.
# We performed analysis to have insights on various quality measures such as alcohol, citric acid, etc... The data set was orginally clean, so we didn't have to spend much time doing cleaning. 
# 
# We also compared three different Machine Learning Algorith which are Support Vector Classifier (SVC), Random Forest Classifier, and Logistic Regression. Random Forest outlined other algorithms in making good predictions. 
# There are many other algorithms that can be applied here, our focus was to use some of them.  
# 
# 
# Finally, we opted to use two techniques (which are GridSearch for finding best parameters, and Cross Validation) for improving accuracy on each of the 3 Machine learning algorithms. The model accuracy for each one was improved a number of percentages(Example is for Random Forest Classifer, accuracy went from 88% to 91%). 

# # Thank you for reading. If you made it to this point, help me know your observations(can be a comment, correcting mistake, or additional input) so that anyone reading this can get the full understanding of algorithms used in this notebook. 
# 
# # * Happy modeling & predicting!*

# # Credits & Inspirations
# 
# 
# [1] Blog, Terence Shin, TDS. Available [Here](https://towardsdatascience.com/predicting-wine-quality-with-several-classification-techniques-179038ea6434)
# 
# [2] Kaggle Notebook, Prediction of quality of Wine, Vishar Kumar. Available [Here](https://www.kaggle.com/vishalyo990/prediction-of-quality-of-wine)
