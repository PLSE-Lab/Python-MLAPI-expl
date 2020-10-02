#!/usr/bin/env python
# coding: utf-8

# # 1. About the Kernel

# In this Kernel we will be exploring data set provided for Forest Cover type classification in Beginner's playground competition and apply different classifier algorithm ,will evaluate each algorithm.
# 
# To get more on EDA refer my another [kernel](https://www.kaggle.com/kushbhatnagar/first-competition-kernel-house-pricing-prediction) on House Sale Price Prediction Competition 
# 
# If you like the kernel please upvote :)

# # 2. Get the Dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Importing the required libraries and data set 
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset_train=pd.read_csv("../input/learn-together/train.csv")


# Checking data set details like number of records , number of columns , column data type

# In[ ]:


#Checking first few rows
dataset_train.head()


# In[ ]:


#Total number of records
dataset_train.shape


# In[ ]:


#Column Details
dataset_train.columns


# In[ ]:


#Create feature matrix , will keep all columns except 'Id'
X=dataset_train.drop(columns=['Id'])
X.head()


# # 3. Data Analysis

# Let's examin relationship between differnet variables and cover type

# In[ ]:


sns.scatterplot(X['Elevation'],X['Aspect'],hue=X['Cover_Type'],palette='rainbow')


# Scatterplot is not give us any clear picture , let's try box plot

# In[ ]:


#Boxplot between elevation and Cover type
sns.boxplot(y=X['Elevation'],x=X['Cover_Type'],palette='rainbow')


# In[ ]:


#Boxplot between Aspect and Cover type
sns.boxplot(y=X['Aspect'],x=X['Cover_Type'],palette='rainbow')


# In[ ]:


#Boxplot between Slope and Cover type
sns.boxplot(y=X['Slope'],x=X['Cover_Type'],palette='rainbow')


# In[ ]:


#Boxplot between Horizontal_Distance_To_Hydrology and Cover type
sns.boxplot(y=X['Horizontal_Distance_To_Hydrology'],x=X['Cover_Type'],palette='rainbow')


# In[ ]:


#Boxplot between Vertical_Distance_To_Hydrology and Cover type
sns.boxplot(y=X['Vertical_Distance_To_Hydrology'],x=X['Cover_Type'],palette='rainbow')


# In[ ]:


#Boxplot between Horizontal_Distance_To_Roadways and Cover type
sns.boxplot(y=X['Horizontal_Distance_To_Roadways'],x=X['Cover_Type'],palette='rainbow')


# In[ ]:


#Boxplot between Hillshade_9am and Cover type
sns.boxplot(y=X['Hillshade_9am'],x=X['Cover_Type'],palette='rainbow')


# In[ ]:


#Boxplot between Hillshade_Noon and Cover type
sns.boxplot(y=X['Hillshade_Noon'],x=X['Cover_Type'],palette='rainbow')


# In[ ]:


#Boxplot between Hillshade_3pm and Cover type
sns.boxplot(y=X['Hillshade_3pm'],x=X['Cover_Type'],palette='rainbow')


# In[ ]:


#Boxplot between Horizontal_Distance_To_Fire_Points and Cover type
sns.boxplot(y=X['Horizontal_Distance_To_Fire_Points'],x=X['Cover_Type'],palette='rainbow')


# Above box plots gives us fair understanding that for each forest cover type there are entries in every variables so for now we will consider all these variables in our analysis . Remaining variables are binary variables , which means they are already label encoded (*this makes our work little Simpler)*
# 
# For now we will group similar varaibles and try to understand relationship across  differnt forest cover types. We can divide variables in follwoing categories 
# * **Degree Variables** : We can consider variables *'Elevation,'Aspect' and 'Slope'* under this category , as these three variable are about  measurments either in angular or numerical form
# * **Distance Variables** : We can consider following variables as they are about different distances from varoius points *'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points*
# * **Hillsahde Variables** : Three hillshade variables comes under this cateogry
# * **Wilderness Variables** : All four wilderness variables comes under this category
# 

# In[ ]:


#Creating data frame for Degree Variables 
X_deg=X[['Elevation','Aspect','Slope','Cover_Type']]


# In[ ]:


#Creating pairplot for Degree Variables
sns.pairplot(X_deg,hue='Cover_Type')


# From above pairplots we can say that for forest cover types '1' and '7' elevation value lies between '2500' and 4000' and for forest cover type '2' elevation value lies between '2000' and '3500'.
# 
# For 'Aspect' and 'Slope' each forest cover type has almost equal distribution. So, we can say 'Elevation' can play a role in classification.

# In[ ]:


#Creating data frame for Distance Variables 
X_dist=X[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Cover_Type']]


# In[ ]:


#Creating pairplot for Degree Variables
sns.pairplot(X_dist,hue='Cover_Type')


# Let's focus on pairplots between variables 'Horizaontal/Vertical_Distance_To_Hydrology' and Cover Type. 
# 
# For cover type *'3','4' and '6*'   distances are not going to upper values and for * '1','2' ,'5' and '7'*   it's going to higher range.
# 
# Let's check 'Horizontal_Distance_To_Rodways/Fire_Points'. Here also, for cover type  *'3','4' and '6' *  distances are not going to upper values and for  *'1','2' ,'5' and '7'*   it's going to higher range
# 
# It's  quiet evident that these distances are playing role in classification of forest cover type

# In[ ]:


#Creating data frame for Hillshade Variables 
X_hs=X[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Cover_Type']]


# In[ ]:


#Creating pairplot for Hillshade Variables
sns.pairplot(X_hs,hue='Cover_Type')


# From above graphs it's evident that 'Hillshade_9am' and 'Hillshade_Noon' have differnt ranges of start index for all forest cover types . Where as , 'Hillshade_3pm' gives almost same ranges for all forest cover type. We can consider them in our analysis

# In[ ]:


#Creating data frame for Hillshade Variables 
X_wild=X[['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Cover_Type']]


# In[ ]:


#Creating pairplot for Hillshade Variables
sns.pairplot(X_wild,hue='Cover_Type')


# We can see that Cover Type '2' are spread across all four wilderness area , cover types '1' , '7' are in three wilderness area while , '5' , '6' are in two and '4' , '3' are in one wilderness area
# 
# It's clear that  Wilderness areas variables are important in forest cover type classifications
# 
# For 'Soil Type' variables , we will consider them all because as per the data description it looks like they can play role in forest cover type classification.

# **Conclusion** For our model training we will be considering all variables present in data set . I do have feeling that we can skip either 'Aspect' or 'Slope' variables based on the pairplot which we have plotted in previous section but let's keep them both for now , we can see our results and then decide regarding this.

# # 4. Missing Data

# Let's check missing values in our independent and dependent variables

# In[ ]:


#Checking missing values 
total_missing_values_X=X.isnull().sum().sort_values(ascending=False)
total_missing_values_X


# There is no missing values in our independent and dependent variables

# In[ ]:


#Taking independent variable out of X and assigning to y
y=X[['Cover_Type']]
X=X.drop(columns=['Cover_Type'])


# # 5. Different Classifier Models and Predictions

# In[ ]:


#**Commenting data scaling as scores are improved without data scaling ***
# Feature Scaling training set for better predictions 
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)


# In[ ]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
classifier_lr.fit(X,y)


# In[ ]:


# Predicting the Train set results
y_pred_lr=classifier_lr.predict(X)


# In[ ]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lr=confusion_matrix(y,y_pred_lr)
cm_lr


# As confusion matrix is not giving as clear picture , let's try with scatter plot to compare between 'y' and 'y_pred_lr'

# In[ ]:


#Converting y from series to array , to generate a graph for comparision with y_pred_
y=y.values


# In[ ]:


#Converting 2 dimensional y and y_pred array into single dimension 
y=y.ravel()
y_pred_lr=y_pred_lr.ravel()


# In[ ]:


#Creating data frame for y and y_pred_ to create line plot
df_lr=pd.DataFrame({"y":y,"y_pred_lr":y_pred_lr})


# In[ ]:


#Creating scatter plot for both values to see comparision between y and y_pred
plt.figure(figsize=(25,10))
ax=sns.scatterplot(x=range(1,15121),y=df_lr['y'],color='red')
ax=sns.scatterplot(x=range(1,15121),y=df_lr['y_pred_lr'],color='blue')
ax.set_xscale('log')


# ****How to read above graph : ** Red points represents actual values i.e 'y' and blue dots represents predicted one i.e. 'y_pred_lr'. We are seeing only few red dots because most of the points are overlapped which means correct predictions(y=y_pred_lr) but the one which are not are not overlapped are seeing seprately means incorrect predictions . In other words **All visible red dots are incorrect predictions**.
# 
# If we zoom this graph *(which we can do by increasing values in **plt.figure(figsize=(25,10)**)* we can see more red dots
# 
# **Note:** Since 'x' axis range is big as compare to 'y' axis we have converted it into log values to see maximum values in small scale
# 
# So this model is giving us few incorrect predictions .Let's try other model

# In[ ]:


# Fitting KNN classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_knn=KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)
classifier_knn.fit(X, y)


# In[ ]:


# Predicting the Train set results
y_pred_knn=classifier_lr.predict(X)


# In[ ]:


#Converting 2 dimensional  y_pred array into single dimension 
y_pred_knn=y_pred_knn.ravel()


# In[ ]:


#Creating data frame for y and y_pred_ to create line plot
df_knn=pd.DataFrame({"y":y,"y_pred_knn":y_pred_knn})


# In[ ]:


#Creating scatter plot for both values to see comparision between y and y_pred
plt.figure(figsize=(25,10))
ax=sns.scatterplot(x=range(1,15121),y=df_knn['y'],color='red')
ax=sns.scatterplot(x=range(1,15121),y=df_knn['y_pred_knn'],color='blue')
ax.set_xscale('log')


# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X, y)


# In[ ]:


# Predicting the Train set results
y_pred_rf=classifier_rf.predict(X)


# In[ ]:


#Converting 2 dimensional  y_pred array into single dimension 
y_pred_rf=y_pred_rf.ravel()


# In[ ]:


#Creating data frame for y and y_pred_ to create line plot
df_rf=pd.DataFrame({"y":y,"y_pred_rf":y_pred_rf})


# In[ ]:


#Creating scatter plot for both values to see comparision between y and y_pred
plt.figure(figsize=(25,10))
ax=sns.scatterplot(x=range(1,15121),y=df_rf['y'],color='red')
ax=sns.scatterplot(x=range(1,15121),y=df_rf['y_pred_rf'],color='blue')
ax.set_xscale('log')


# We can see that above graph is with very less number of red dots , as compare to previous graphs this is very accurate predictions , so let's focus on this algorithm i.e. on ** Random Forest Classifier **.
# 
# Now , let's evaluate the accuracies of this model with the help of K-Fold Corss Validation

# # 6. K-Fold Cross Validation
# 

# In[ ]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_rf = cross_val_score(estimator = classifier_rf, X = X, y = y, cv = 10)
accuracies_rf


# In[ ]:


#Calculating mean and standard deviation for random forest model
accuracies_rf.mean()
accuracies_rf.std()


# Mean Accuracy is coming close to 75% and standard Devaition is also not that much (~5%) , still we can improve this model.
# 
# Let's try grid search to hypertune the parameters

# # 7. Grid Search

# > *Changing below two cells to markdown to prevent execution at the time of commit because this particular code is taking long time to execute . So I have executed this seprately to get best parameters and commenting this part to avoid long execution time during commit and generating output.*

# #Applying Grid Search to find the best model and the best parameters
# from sklearn.model_selection import GridSearchCV
# #Create the parameter grid based on the results of random search 
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 300, 500, 1000]
# }
# grid_search = GridSearchCV(estimator = classifier_rf, param_grid = param_grid,cv = 3, n_jobs = -1)
# grid_search = grid_search.fit(X, y)

# #Getting the best params
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# best_accuracy
# best_parameters

# #Getting the best params
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# best_accuracy
# best_parameters

# Let's create a new classifier model with above parameters . In addition to above grid search results I have refererd following kernels for fine tunning 
# *  https://www.kaggle.com/arateris/2-layer-k-fold-learning-forest-cover#Feature-removal
# *  https://www.kaggle.com/joshofg/pure-random-forest-hyperparameter-tuning

# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_rf_new = RandomForestClassifier(n_estimators = 719,
                                           bootstrap=False,
                                           max_depth=464,
                                           max_features=0.3,
                                           min_samples_leaf=1,
                                           min_samples_split=2,
                                           random_state=42)
classifier_rf_new.fit(X, y)


# In[ ]:


# Predicting the Train set results
y_pred_rf_new=classifier_rf_new.predict(X)


# In[ ]:


#Converting 2 dimensional  y_pred array into single dimension 
y_pred_rf_new=y_pred_rf_new.ravel()


# In[ ]:


#Creating data frame for y and y_pred_ to create line plot
df_rf_new=pd.DataFrame({"y":y,"y_pred_rf_new":y_pred_rf_new})


# In[ ]:


#Creating scatter plot for both values to see comparision between y and y_pred
plt.figure(figsize=(25,10))
ax=sns.scatterplot(x=range(1,15121),y=df_rf_new['y'],color='red')
ax=sns.scatterplot(x=range(1,15121),y=df_rf_new['y_pred_rf_new'],color='blue')
ax.set_xscale('log')


# With latest fine tuning of random forest classifier model red dots are negligible , let's check K-Fold Cross Validation for this model

# In[ ]:


# Applying k-Fold Cross Validation
accuracies_rf_new = cross_val_score(estimator = classifier_rf_new, X = X, y = y, cv = 10)
accuracies_rf_new


# In[ ]:


#Calculating mean and standard deviation for random forest model
accuracies_rf_new.mean()
accuracies_rf_new.std()


# Mean Accuracy is coming close to 80% and standard Devaition is also not that much (~4%) , it's improved a lot after finetuning .
# 
# > **Note**: As if now best score i.e. '0.77' is coming with this model only and with these parameters
# 
#  Let's try another algorithm called **XG-boost**

# In[ ]:


#importing required library and creating XGboost classifier model
#Refered above mentioned kernels for fine tuning XGB classifier model
from xgboost import XGBClassifier
classifier_xgb=XGBClassifier(n_estimators = 719,
                             max_depth = 10)
classifier_xgb.fit(X,y)


# In[ ]:


# Predicting the Train set results
y_pred_xgb=classifier_xgb.predict(X)


# In[ ]:


#Converting 2 dimensional  y_pred array into single dimension 
y_pred_xgb=y_pred_xgb.ravel()


# In[ ]:


#Creating data frame for y and y_pred_ to create line plot
df_xgb=pd.DataFrame({"y":y,"y_pred_xgb":y_pred_xgb})


# In[ ]:


#Creating scatter plot for both values to see comparision between y and y_pred
plt.figure(figsize=(25,10))
ax=sns.scatterplot(x=range(1,15121),y=df_xgb['y'],color='red')
ax=sns.scatterplot(x=range(1,15121),y=df_xgb['y_pred_xgb'],color='blue')
ax.set_xscale('log')


# Here also red dots are becoming negligible , let's evaluate the model with K-Cross fold validation

# In[ ]:


accuracies_xgb = cross_val_score(estimator = classifier_xgb, X = X, y = y, cv = 10)
accuracies_xgb


# In[ ]:


#Calculating mean and standard deviation for random forest model
accuracies_xgb.std()
accuracies_xgb.mean()


# Mean Accuracy is coming close to 79% and standard Devaition is also not that much (~4%).
# 
# For predictions from Test set let's take last three models which are random forest without tuned parameters, random forest with tuned parameters and XGboost

# # 8. Cover Type Prediction from Test Set

# In[ ]:


#Get test data 
dataset_test = pd.read_csv("../input/learn-together/test.csv")


# In[ ]:


#Create X_test and fetching id in different frame
X_test=dataset_test.drop(columns=['Id'])
y_test_id=dataset_test[['Id']]


# In[ ]:


#Converting Id into array
y_test_id=y_test_id.values


# In[ ]:


#Converting 2 dimensional y_test_id into single dimension 
y_test_id=y_test_id.ravel()


# In[ ]:


#Checking missing value in test data set
total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)
total_missing_values_X_test


# There is no missing value in test data

# In[ ]:


#**Commenting data scaling as scores are improved without data scaling ***
#Scaling and Transforming test set also as train set is already scaled and transformed
#X_test = sc.fit_transform(X_test)


# In[ ]:


#Creating predictions from random forest model without fine tuned parameters
y_test_pred_rf=classifier_rf.predict(X_test)


# In[ ]:


#Converting 2 dimensional y_test_pred into single dimension 
y_test_pred_rf=y_test_pred_rf.ravel()


# In[ ]:


#Creating Submission dataframe from id and predecited Sale price
submission_df_rf=pd.DataFrame({"Id":y_test_id,"Cover_Type":y_test_pred_rf})
#Setting index as Id Column
submission_df_rf.set_index("Id")


# In[ ]:


#Converting into CSV file for submission
submission_df_rf.to_csv("submission_rf.csv",index=False)


# In[ ]:


#Creating predictions from random forest model with fine tuned parameters
y_test_pred_rf_new=classifier_rf_new.predict(X_test)


# In[ ]:


#Converting 2 dimensional y_test_pred into single dimension 
y_test_pred_rf_new=y_test_pred_rf_new.ravel()


# In[ ]:


#Creating Submission dataframe from id and predecited Sale price
submission_df_rf_new=pd.DataFrame({"Id":y_test_id,"Cover_Type":y_test_pred_rf_new})
#Setting index as Id Column
submission_df_rf_new.set_index("Id")


# In[ ]:


#Converting into CSV file for submission
submission_df_rf_new.to_csv("submission_rf_new.csv",index=False)


# In[ ]:


#Creating predictions from XGB model
y_test_pred_xgb=classifier_xgb.predict(X_test)


# In[ ]:


#Converting 2 dimensional y_test_pred into single dimension 
y_test_pred_xgb=y_test_pred_xgb.ravel()


# In[ ]:


#Creating Submission dataframe from id and predecited Sale price
submission_df_xgb=pd.DataFrame({"Id":y_test_id,"Cover_Type":y_test_pred_xgb})
#Setting index as Id Column
submission_df_xgb.set_index("Id")


# In[ ]:


#Converting into CSV file for submission
submission_df_xgb.to_csv("submission_xgb.csv",index=False)

