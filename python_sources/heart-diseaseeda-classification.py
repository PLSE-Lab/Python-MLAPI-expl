#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# Accessing the dataset 

# In[ ]:


data = pd.read_csv('../input/heart-disease-dataset/datasets_heart.csv') 


# Starting with the descriptive statistics approach 

# #### Peak at the data 

# In[ ]:


data.head()


# Following dataset has Only numeric attributes and of different scales 

# #### Checking the dimension , dtypes , descriptive info and the balance of the dataset 

# In[ ]:


print('Number of datapoints : {}'.format(data.shape[0]))
print('Number of Columns : {}'.format(data.shape[1])) 

print(data.dtypes) 

print( ' Distribution of class label points ' )
data['target'].value_counts()


# Our dataset has few datapoints . Most of the attributes are Numeric and its a slightly imbalanced dataset .

# In[ ]:


data.describe().T


# #####checking the nullity and skewness with these attributes 

# In[ ]:


print( data.isnull().sum() )
print( 'skewness table : \n' , data.skew() )


# There is no missing values . Most of the skewness are nearer to zero except few .

# #### Visualization aprroach

# renaming our columns for easier interpretation 

# In[ ]:


data.columns = [ 'age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar > 120 mg/dl', 'resting EC results', 'maximum heart rate',
       'induced angina', 'ST depression', 'slope', 'num_vessels', 'thalassemia', 'target'  ] 


# In[ ]:


# look the data now 

data.head() 


# Much better :-)

# Pickking the continuous numerical attributes for univariate plots .

# In[ ]:


data['induced angina'].value_counts() 


# In[ ]:


# age , resting blood pressure , serum cholestoral , maximum heart rate , ST depression 
plt.figure( figsize= (25,25) )
data[ ['age' , 'resting blood pressure' , 'serum cholestoral' , 'maximum heart rate' , 'ST depression'] ].hist(bins=10)
plt.show()


# ST depression looks like a log distribution , age and heart rate has negative skew . where as BP and cholestrol apparently looks gaussian 

# #### Lets pick age and analyse the data

# In[ ]:


# copying the existing data for future use and trying to map some categorical values to strings for better interpretation

workout_data = data.copy()
data['sex'] = data['sex'].map( { 1 : 'MALE' , 0 : 'FEMALE'} )
data['fasting blood sugar > 120 mg/dl'] = data['fasting blood sugar > 120 mg/dl'].map( { 1 : 'HIGH SUGAR' , 0 : 'LOW SUGAR' } ) 


# In[ ]:


data['target'] = data['target'].map({ 1 : 'HEART DISEASE' , 0 : 'NO HEART DISEASE' })


# In[ ]:


plt.figure(figsize=(10,10))
px.box( data_frame=data , x='sex' , y= 'age' , color='target'  ) 


# age range of females who catch heart complaint is pretty much wide compared to males , which is not so good for female category 

# In[ ]:


sns.countplot( data = data , x = 'sex' , hue = 'target' )


# Ratio of females catching heart complaint to that of males is pretty much high . 

# In[ ]:


plt.figure( figsize = (30,10) )
sns.countplot( data = data , x = 'age' , hue = 'target' )


# from the plot , we can analyse that heart disease  chance is really favourable to happen for ages lesser than 54 , where as lesser occurance post 54 in a general sense . 
# 

# In[ ]:


sns.countplot( data = data , x = 'fasting blood sugar > 120 mg/dl' , hue = 'target' )


# People having high sugar level has equal chance of getting and not getting a heart disease , where as for those who have low sugar , chance of getting heart disease is a bit high . so keep your sugar level above a threshold .

# In[ ]:


px.box( data_frame= data , x= 'sex' , y= 'serum cholestoral' , color= 'target' )


# Females were already a leading contender in heart disease based on this dataset , but females who had higher serum cholestoral had equal say in getting and not getting heart complaint , almost a similar say in case of male . seems like serum cholestoral is not a big indicator of our target.

# In[ ]:


# ST depression 

px.box( data_frame= data , x= 'sex' , y= 'ST depression' , color= 'target' )


# In[ ]:


plt.figure( figsize = (30,10) )
sns.countplot( data= data , x= 'ST depression' , hue = 'target' )


# when the ST segement depression is increasing , chance of not getting heart disease is dominating . 

# In[ ]:


# 'resting blood pressure' , 'maximum heart rate' 


px.box( data_frame= data , x = 'sex' , y= 'maximum heart rate' , color= 'target' )


# In[ ]:


plt.figure( figsize= (30,10) )
sns.countplot(data = data , x = 'resting blood pressure' , hue = 'target')


# 1. When the resting blood pressure is lower , occurence of heart disease is quite leading and goes dominating even though resting bp increases .
# 
# 2. People having higher maximum heart rate got to have high chance of getting heart disease irrespective male or female .

# In[ ]:


sns.heatmap( workout_data.corr() , vmin = 0 , vmax= 0.5 )

corr_matrix = workout_data.corr()
corr_matrix['target'].sort_values( ascending = False )


# Seems like , there is good correlation contributed by chest pain , resting EC results , max heart rate , slope  .

# #### Preparing the data

# In[ ]:


# checking missing values 

workout_data.isnull().sum()


# In[ ]:


dummy_data = pd.get_dummies( data=workout_data , columns=['chest pain type' , 'slope' , 'thalassemia'] , 
               prefix= [ 'cp' , 'slope' , 'thal' ] ) 
print('Dimension of our dummy data is : \n')
dummy_data.shape


# There is no missing value . Only the scale of certain attributes need to be normalised , which could be done by a template . We have handled the categorical attributes by dummy technique .

# #### Creating our validation set

# In[ ]:


x = dummy_data.drop( labels= ['target'] , axis = 1)
y = dummy_data['target']

seed = 20 
test_size = 0.2 

x_train , x_test , y_train , y_test = train_test_split( x.values , y.values , test_size = test_size , random_state = seed ,
                                                       stratify = y.values ) 


# #### Template for spot check algorithm 

# In[ ]:


models = [] 
models.append( ( 'Logistic' , LogisticRegression( n_jobs = -1 )   ) )
models.append( ( 'KNN' , KNeighborsClassifier()   ) )
models.append( ( 'NB' , GaussianNB()  ) )
models.append( ( 'tree' , DecisionTreeClassifier()  ) )
models.append( ( 'SVM' , SVC()   ) )
models.append( ( 'RandForest' , RandomForestClassifier() ) )

RESULTS = []
NAME = [] 

scale = MinMaxScaler() 
scale.fit(x_train) # scaling the x_train for spot check validation stage 

x_train_scaled = scale.transform( x_train )

for name , model in models:
    
    kfold = KFold( n_splits= 10 , random_state= seed , shuffle= True )
    cv_score = cross_val_score( model , x_train_scaled , y_train , cv = kfold , scoring='accuracy' )
    RESULTS.append(cv_score)
    NAME.append(name)
    
    result = 'name of the model : {} , score returned : {} % '.format( name , (cv_score.mean()*100).round(3) )
    print(result) 


# In[ ]:


sns.boxplot( x= NAME , y= RESULTS )


# Observing the spot check stage , we could pick - SVM , KNN , Logistic Regression , and finally Random Forest , because of its robust performance upon tuning . This spot check was performed on training set to filter out best techniques .

# #### Improving our results 

# Using Hyperparameter tuning

# In[ ]:


x_test_scaled = scale.transform( x_test ) 

testmodels = [] 

testmodels.append( ( 'SVM' , SVC()   ) )
testmodels.append( ( 'Logistic' , LogisticRegression( n_jobs = -1 )   ) )
testmodels.append( ( 'KNN' , KNeighborsClassifier()   ) )
testmodels.append( ( 'RandForest' , RandomForestClassifier(n_jobs= -1) ) )


params = []

params.append( ( 'SVM' ,     { 'C' : [ 0.1,1,3,5,7,10,50,100,1000 ] , 'gamma' : [ 1,0.1,0.01,0.001 ] , 'kernel': ['rbf', 'poly', 'sigmoid']} ) )
params.append( ( 'Logistic', { 'C' : [ 0.1,1,3,5,7 ,10, 100 , 1000 ] }  ) )
params.append( ( 'KNN' ,     { 'n_neighbors' : [ 3,5,7,11,19] , 'weights' : [ 'uniform' , 'distance' ] , 'metric' : ['euclidean' , 'manhattan' ] } ) )
params.append( ( 'RandForest'  , {'n_estimators': [100, 200, 300, 500]  } ) )



def tune( models_collection , param_grid , x_train , y_train  ) :

  for name_ , model in models_collection :  
    for name , paramdict in params:
      if name_ == name :
        
        kfold = KFold( n_splits= 10 , random_state= seed , shuffle= True )
        grid = GridSearchCV( model , param_grid= paramdict , n_jobs= -1 , cv=kfold )
        grid.fit( x_train , y_train )  
   
        print('name of the model : ',name) 
        print( grid.best_estimator_  )  


      
tune( models_collection= testmodels , param_grid= params , x_train= x_train_scaled , y_train= y_train )
     


# Observing the results from grid search , we could say our algorithms could use these best estimated parameters for better results

# #### Finalising Model and presenting Results 

# 1. SVM

# In[ ]:


estimators = []

estimators.append( ('scaler' , MinMaxScaler() ) )
estimators.append( ( 'SVM' , SVC(C=5, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.01, kernel='sigmoid',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)  ) )

final_model = Pipeline(estimators)

final_model.fit(x_train , y_train) 

print('train score : {}'.format( (final_model.score(x_train , y_train)*100).round(3)) )

print('test score : {}'.format( (final_model.score(x_test , y_test)*100).round(3))) 

prediction_svm = final_model.predict( x_test )


# SVM is promising 

# 2.  lOGISTIC REGRESSION

# In[ ]:


estimators = []

estimators.append( ('scaler' , MinMaxScaler() ) )
estimators.append( ( 'Logistic' , LogisticRegression(C=5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)  ) )

final_model = Pipeline(estimators)

final_model.fit(x_train , y_train) 

print('train score : {}'.format( (final_model.score(x_train , y_train)*100).round(3)) )

print('test score : {}'.format( (final_model.score(x_test , y_test)*100).round(3)))


# Looks like our Logistic regression is overfitting . Try to avoid such algorithm

# 3. KNN

# In[ ]:


estimators = []

estimators.append( ('scaler' , MinMaxScaler() ) )
estimators.append( ( 'KNN' , KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=11, p=2,
                     weights='uniform')  ) )

final_model = Pipeline(estimators)

final_model.fit(x_train , y_train) 

print('train score : {}'.format( (final_model.score(x_train , y_train)*100).round(3)) )

print('test score : {}'.format( (final_model.score(x_test , y_test)*100).round(3)))


# Relatively low scores from KNN

# 4. Random Forest

# In[ ]:


estimators = []

estimators.append( ('scaler' , MinMaxScaler() ) )
estimators.append( ( 'Random Forest' , RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                       warm_start=False)  ) )

final_model = Pipeline(estimators)

final_model.fit(x_train , y_train) 

print('train score : {}'.format( (final_model.score(x_train , y_train)*100).round(3)) )

print('test score : {}'.format( (final_model.score(x_test , y_test)*100).round(3)))


# Seems like Random Forest is badly overfittting , lets solve that independently

# In[ ]:


# added more hyperparameters to be optimized . 

kfold = KFold( n_splits= 10 , random_state= seed , shuffle= True )
grid = GridSearchCV( RandomForestClassifier() , param_grid= { 'max_depth': [80, 90, 100, 110], 'min_samples_leaf': [2, 3, 4, 5], 'min_samples_split': [8, 10, 12], 'n_estimators': [100, 200, 300, 1000] } , 
                    n_jobs= -1 , cv=kfold )


grid.fit( x_train , y_train )  

grid.best_score_ , grid.best_estimator_


# After waiting over 20 minutes of Grid Search , we gonna use this estimations for our randomforest  

# In[ ]:


estimators = []

estimators.append( ('scaler' , MinMaxScaler() ) )
estimators.append( ( 'Random Forest' , RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                        criterion='gini', max_depth=100, max_features='auto',
                        max_leaf_nodes=None, max_samples=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=5, min_samples_split=10,
                        min_weight_fraction_leaf=0.0, n_estimators=200,
                        n_jobs=None, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)  ) ) #you could get all these values from colab

final_model = Pipeline(estimators)

final_model.fit(x_train , y_train) 

print('train score : {}'.format( (final_model.score(x_train , y_train)*100).round(3)) )

print('test score : {}'.format( (final_model.score(x_test , y_test)*100).round(3)))


# SUMMARY : Seems like Random Forest has returned some favourable score , but since it looks like a bit overfitting due to the complexity our model offers . So either try to tune more hyper parameters in RF or Go with Support vector machines since its a clean binary classification problem .
# 
#  

# MODEL CHOSEN : SVM 

# In[ ]:


print( classification_report( y_test , prediction_svm ) )


# Looks ok to me :-) 
# 
# i'm a beginner in datascience , so feel free to correct me .

# In[ ]:




