#!/usr/bin/env python
# coding: utf-8

# # 1) import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from matplotlib import pyplot


# # 2)loading data

# In[ ]:


df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# In[ ]:


df.head()


# # data description

# In[ ]:


df.describe()


# In[ ]:



df.info()


# # Finding missing value

# In[ ]:


import missingno
missingno.matrix(df,figsize = (12,6))
#df.isnull().sum()


# by using matrix graph we can say that 'train' doesnt have any Null value.

# # 4) carryout univariate and multivariate analysis using graphical and non graphical(some number represent data)
# 

# In[ ]:


#df.corr() or
plt.figure(figsize = (10,10))
sns.heatmap(df.corr(),cmap = 'viridis',annot=True,linewidth = 0.5)


# chance of admit is highly correlated with 'GRE Score' & 'TOFEL Score'.

# In[ ]:


sns.distplot(df['GRE Score'])


# between 300-340 'GRE Score' is highly distributed.

# In[ ]:


sns.pairplot(df,hue='Research');


# In[ ]:


sns.distplot(df['TOEFL Score'],kde = False)
df.head(1)


# In[ ]:


sns.jointplot(x='CGPA',y = 'GRE Score',data=df,kind = 'hex')


# 
# 'GRE Score' between 300-340 and'CGPA' 8.5-9.5 is highly correlated.

# In[ ]:


sns.jointplot(x = 'GRE Score',y = 'CGPA',data=df,kind = 'kde')


# In[ ]:


sns.scatterplot(x = 'University Rating',y = 'CGPA',data=df,color = 'blue')


# by using Scater plo we can easily said that higher 'CGPA' have higher 'Univercity Rating'.

# # Now lets set Some cut-off scores and try to analyse scores above the cut-off

# In[ ]:


df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})


# In[ ]:


co_gre = df[df['GRE Score']>=300]
co_toefel = df[df['TOEFL Score']>=100]


# In[ ]:


fig, ax = pyplot.subplots(figsize=(15,8))
sns.barplot(x='GRE Score',y='Chance of Admit',data= co_gre)
plt.show()


# In[ ]:


fig,ax = pyplot.subplots(figsize=(15,8))
sns.barplot(x='TOEFL Score',y='Chance of Admit',data=co_toefel)
plt.show()


# the above two graphs make it clear tha higher the scores better the chance of Admit.

# In[ ]:


s = df[df['Chance of Admit'] >=0.75]['University Rating'].value_counts().head(5)
plt.title('University Ratings of candidate with an 75% acceptance chance')
s.plot(kind = 'bar',figsize=(20,10))
plt.xlabel('University Rating')
plt.ylabel('Candidates')
plt.show()


# In[ ]:


print('Average GRE Score :{0:.2f}out of 340'.format(df['GRE Score'].mean()))
print('Average TOEFL Score:{0:2f} out of 120'.format(df['TOEFL Score'].mean()))
print('Average CGPA:{0:.2f} out of 10'.format(df['CGPA'].mean()))
print('Average Chance of getting admitted:{0:.2f}%'.format(df['Chance of Admit'].mean()*100))


# https://www.kaggle.com/nitindatta/graduate-admissions

# #  preprocessing and prepare data for statistical modeling

# In[ ]:


df = df.drop(['Serial No.'],axis = 1)


# # Feature Selection 

# In[ ]:


df1 = pd.read_csv('../input/Admission_Predict.csv')


# In[ ]:


serialNo = df1['Serial No.'].values
df1.head()


# In[ ]:


df1.drop(['Serial No.'],axis=1,inplace=True)


# In[ ]:


df1=df1.rename(columns = {'Chance of Admit ':'Chance of Admit'})


# # spliting data on train and test set

# In[ ]:


x = df1.drop(["Chance of Admit"],axis=1)
y = df1["Chance of Admit"].values


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test  = train_test_split(x,y,test_size = 0.20,random_state= 42)


# In[ ]:


#normalization
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0,1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])


# #now we use logistic regression

# In[ ]:


from sklearn.preprocessing import Imputer, MinMaxScaler

#machin learning model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


#create an impute object with a median fitting strategy
imputer = Imputer(strategy = 'median')

#train on training features
imputer.fit(x_train)

#transform both training and testing data
x_train = imputer.transform(x_train)
x_test = imputer.transform(x_test)


# In[ ]:


#convert y to one-dimensional array(vector)
y_train = np.array(y_train).reshape((-1,))
y_test = np.array(y_test).reshape((-1,))


# In[ ]:


#we will use 5 different model :
#1 - Linear Regression
#2 - Support Vector Machin Regression
#3 - Random Forest Regression
#4 - Gradient Boosting Regression
#5 - K-Nearest Neighbors Regression


#Function to calculate mean absolute error

def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

# takes in a model, train the model, and evaluate the model on test set
def fit_and_evaluate(model):
    
    #train the model 
    model.fit(x_train, y_train)
    
    #prediction
    model_pred = model.predict(x_test)
    model_mae = mae(y_test, model_pred)
    
    #return the performance metric
    return model_mae


# In[ ]:


#linear model
lr = LinearRegression()
lr_mae = fit_and_evaluate(lr)

print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)


# In[ ]:


# # SVM

svm = SVR(C = 1000, gamma = 0.1)
svm_mae = fit_and_evaluate(svm)

print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' % svm_mae)


# In[ ]:


# # Random Forest

random_forest = RandomForestRegressor(random_state=60)
random_forest_mae = fit_and_evaluate(random_forest)

print('Random Forest Regression Performance on the test set: MAE = %0.4f' % random_forest_mae)


# In[ ]:


# # Gradiente Boosting Regression

gradient_boosted = GradientBoostingRegressor(random_state=60)
gradient_boosted_mae = fit_and_evaluate(gradient_boosted)

print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)


# In[ ]:


# # KNN

knn = KNeighborsRegressor(n_neighbors=10)
knn_mae = fit_and_evaluate(knn)

print('K-Nearest Neighbors Regression Performance on the test set: MAE = %0.4f' % knn_mae)


# Pyrameter Tunning

# In[ ]:


# # # Model Optimization

# # Hyperparameter

# Hyperparameter Tuning with Random Search and Cross Validation

# Here we will implement random search with cross validation to select the optimal hyperparameters for the gradient boosting regressor. 
# We first define a grid then peform an iterative process of: randomly sample a set of hyperparameters from the grid, evaluate the hyperparameters using 4-fold cross-validation, 
# and then select the hyperparameters with the best performance.

# Loss function to be optimized
loss = ['ls', 'lad', 'huber']

# Number of trees used in the boosting process
n_estimators = [100, 500, 900, 1100, 1500]

# Maximum depth of each tree
max_depth = [2, 3, 5, 10, 15]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}


# In[ ]:


# In the code below, we create the Randomized Search Object passing in the following parameters:


# In[ ]:


model = GradientBoostingRegressor(random_state = 42)

# Set up the random search with 4-fold cross validation
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25, 
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True,
                               random_state=42)


# In[ ]:


random_cv.fit(x_train,y_train)


#  Scikit-learn uses the negative mean absolute error for evaluation because it wants a metrics to maximize.
#  
#  therefore a better score will be closer to 0. we can get the results of the randomized search into a dataframe, and sort the value by performance.
#  
#  

# In[ ]:


#get all of the cv results and sort by the test performance

random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score',ascending = False)

random_results.head(5)


# In[ ]:


#to find best estimator
random_cv.best_estimator_


# In[ ]:


#the best gradient boosted model the following hyperparameters:

#loss = lad
#n_estimators = 500
#max_depth = 2
#min_samples_leaf = 8
#min_sample_split = 6
#max_features = None


# i will focus on a single line, the number of trees in the forest(n_stimators).
# by varying only one hperparameter, we can directly observe how it affects performance.
# in the case of number of trees, we could expect to see a significant affect on the amount of under vs overfitting.
# 
# ###here we will use grid search with a grid that only has the n_estimators hyperparameter.
# 
# we will evaluate a range of trees that plot the training and testing performance to get idea of what increasing the number of trees does for our model.
# 
#  we will fix the other hyperparameter at the best values returned from random search to isolate the number of trees effect.

# In[ ]:



#Create a range of trees to evaluate

trees_grid = {'n_estimators' : [100,150,200,250,300,350,400,450,500,550,
                               600,650,700,750,800]}

model = GradientBoostingRegressor(loss = 'lad', max_depth = 2,min_samples_leaf = 8,
                                 min_samples_split= 6, max_features = None, 
                                 random_state = 42)

#Grid Search Object using the trees range and the random forest model

grid_search = GridSearchCV(estimator = model, param_grid = trees_grid, cv = 4,
                          scoring = 'neg_mean_absolute_error',verbose = 1,
                          n_jobs = -1,return_train_score = True )


# In[ ]:


# fit the grid search

grid_search.fit(x_train, y_train)


# In[ ]:


# get the result into DaaFrame
results = pd.DataFrame(grid_search.cv_results_)

# plot the training and testing error vs number of trees

figsize = (8,8)
plt.style.use('fivethirtyeight')
plt.plot(results['param_n_estimators'],-1 * results
        ['mean_test_score'],label = 'Testing error')
plt.plot(results['param_n_estimators'], -1 *results
        ['mean_train_score'],label = 'Training error')
plt.xlabel('Number of Trees');plt.ylabel('Mean Absolute Error');plt.legend();
plt.title('Performance vs Number of Trees')




# #there willl always be a difference between the training error and testing error (the training error is always lower ) but if there is significant difference,
# #we want to try and reduce overfiting, either by getting more trainnig data or reducing the complexity of the model through hyperparameter tuning or regularization.
# 
# #for now, we will use the model with the best performance and accept that it may be overfiting to the training set 

# In[ ]:


results.sort_values('mean_test_score', ascending = False).head()


# In[ ]:


# evaluate final model on the test set

#we will use the best model from hyperparameter tuning to make prediction on  the testing set

#for comparision, we can also look at the performance of the default model.
#the code below creates the final model, trains it(with timimng), and evaluates on the test set.

#Default model

default_model = GradientBoostingRegressor(random_state = 42)

#select the best model
final_model = grid_search.best_estimator_
final_model


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n 1 -r 5', ' default_model.fit(x_train, y_train)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '-n 1 -r 5', 'final_model.fit(x_train, y_train)')


# In[ ]:


default_pred = default_model.predict(x_test)
final_pred = final_model.predict(x_test)

print('Default Model Performance on test set:MAE = %0.4f.' %mae(y_test, default_pred))
print('Final model performance on the test set: MAE = %0.4f.' %mae(y_test,
                                                                   final_pred))


# In[ ]:


# distribution of the true value on test set and the predicted on train set

lr.fit(x_train, y_train)

model_pred = lr.predict(x_test)


figsize=(8,8)

#values

sns.kdeplot(model_pred, label = 'Predictions')
sns.kdeplot(y_test, label= ' Values')

plt.xlabel('Chance of Admission');plt.ylabel('Density');
plt.title('Test Values and Predictions');


# # residual
# #ideally we would hope that residuals are ormally distributed, meaning that the model is wrong the same amount in both directons(high and low)

# In[ ]:


figsize = (6,6)

#calculate the residuals
residuals = model_pred - y_test

#plots the residual in histogram

plt.hist(residuals, color = 'red', bins = 20,
        edgecolor = 'black')
plt.xlabel('Error');plt.ylabel('Count')
plt.title('Distribution of Residuals');


# the residuals are close to normally distributed, with a few noticable outlier on the low end.
# these indicate errors where the model estimate was far below that of the true value.


# In[ ]:


model.fit(x_train, y_train)
df1.head()


# # Feature importance

# In[ ]:


#extract in dataframe

df1_features = df1.drop(labels = 'Chance of Admit', axis = 1)
feature_results = pd.DataFrame({'feature': list(
df1_features.columns), 'importance': model.feature_importances_})


# In[ ]:


feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)

feature_results.head(10)


# NOW,we see in above table that CGPA is more important than anything for univercity Admission. 

# In[ ]:


# Use Features Importances for Feature Selection

#lets try using only the 10 most important features in the linear regression 
#to see if performance is improved

#we can also limit to these features and re-evaluate the random-forest.
# Extract the names of the most important features
most_important_features = feature_results['feature'][:10]

# Find the index that corresponds to each feature name
indices = [list(df1_features.columns).index(x) for x in most_important_features]

# Keep only the most important features
x_train_reduced = x_train[:, indices]
x_test_reduced = x_test[:, indices]

print('Most important training features shape: ', x_train_reduced.shape)
print('Most important testing  features shape: ', x_test_reduced.shape)


# In[ ]:


lr = LinearRegression()

#fit on full set Features
lr.fit(x_train, y_train)
lr_full_pred = lr.predict(x_test)

#Fit on reduced set of features
lr.fit(x_train_reduced, y_train)
lr_reduced_pred = lr.predict(x_test_reduced)

# display results

print('Linear Regression Full Results: MAE = %0.4f.' %mae(y_test, lr_full_pred))
print('Linear Regression Reduced Results: MAE = %0.4f.' %mae(y_test, lr_reduced_pred))

#well reducing the features did not improve the linear results!
# it turns that the extra information in the features with low importance do actually improve performance.


# In[ ]:


# Let's look at using the reduced set of features in the gradient boosted regressor.

# Create the model with the same hyperparamters
model_reduced = GradientBoostingRegressor(loss='lad', max_depth=2, max_features=None,
                                  min_samples_leaf=8, min_samples_split=6, 
                                  n_estimators=800, random_state=42)

# Fit and test on the reduced set of features
model_reduced.fit(x_train_reduced, y_train)
model_reduced_pred = model_reduced.predict(x_test_reduced)

print('Gradient Boosted Reduced Results: MAE = %0.4f' % mae(y_test, model_reduced_pred))

# The model results are slightly worse with the reduced set of features and we will keep all of the features for the final model


# In[ ]:


# find the residual

residuals = abs(model_reduced_pred - y_test)

plt.hist(residuals, color = 'red', bins = 20,
        edgecolor = 'black')
plt.xlabel('Error');plt.ylabel('Count')
plt.title('Distribution of Residuals');


# https://www.kaggle.com/panamby/graduate-admissions/notebook
