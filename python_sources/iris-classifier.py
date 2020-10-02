#!/usr/bin/env python
# coding: utf-8

# # Iris Classifier

# # Table of Contents
# -  [Problem Statement](#section1)</br>
# -  [Load the packages and data](#section2)</br>
# -  [Preprocessing](#section3)</br>
#     - [Missing Values](#section301)<br/>
#     - [Correlation between Target and Features](#section302)<br/>
#     - [Handling Categorical Features](#section303)<br/>
# -  [Model Development](#section4)</br>
#     - [Preparing X and y using pandas](#section401)<br/>
#     - [Splitting X and y into training and test datasets](#section402)<br/>
#     - [Logistic Regression](#section403)<br/>
#     - [Decision Tree Classifier](#section404)<br/>
#     - [Random Forest Classifier](#section405)<br/>
#     - [KNN Classifier](#section406)<br/>
#     - [Naive Bayes Classifier](#section407)<br/>
#     - [Support Vector Classifier](#section408)<br/>
# -  [Model Evaluation](#section5)</br>
# -  [Auto ML](#section6)</br>
# -  [Neural Network Approach](#section7)<br/>
# -  [Conclusion](#section8)<br/>

# <a id=section1></a> 
# # 1. Problem Statement

# Given the dataset containing Iris Species of flowers, different Machine Learning models to predict the species of a flower

# <a id=section2></a> 
# ## 2. Load the packages and data
# <section1>
# <br/>

# In[ ]:


from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import warnings
import pandas_profiling 
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 200)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from scipy.stats import randint as sp_randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# In[ ]:


iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
iris.head()


# In[ ]:


iris.shape


# In[ ]:


iris.info()


# In[ ]:


sns.countplot(iris.Species)


# #### Looking at the count plot,  It is an balanced dataset

# <a id=section3></a> 
# # 3. Preprocessing

# In[ ]:


iris.describe()


# ### Removing id column

# In[ ]:


iris = iris.drop(['Id'], axis=1)


# <a id=section301></a> 
# ## 3.1 Missing Values

# In[ ]:


dataset_na = (iris.isnull().sum() / len(iris)) * 100
dataset_na = dataset_na.drop(dataset_na[dataset_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :dataset_na})
missing_data.head(20)


# There is no missing value in the dataset

# <a id=section302></a> 
# ## 3.2 Find Correlation

# In[ ]:


plt.figure(figsize=(6,4))
sns.heatmap(iris.corr(),annot=True) 
plt.ylim(4,0)


# #### Looking at the coorelation heatmap
# - PetalLengnthCm and PetalWidthCm are positively correlated
# - PetalWidthCm and SepalLengthCm are positively correlated
# - SepalLengthCm and PetalLengnthCm are positively coorelated

# ### Scatter Plots

# In[ ]:


sns.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "PetalLengthCm", "PetalWidthCm")    .add_legend()


# In[ ]:


sns.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "PetalLengthCm", "SepalLengthCm")    .add_legend()


# In[ ]:


sns.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "PetalWidthCm", "SepalLengthCm")    .add_legend()


# In[ ]:


sns.pairplot(iris, hue="Species", size=3)


# <a id=section303></a> 
# ## 3.3 Handling Categorical Features 

# #### Encoding the Target Variable - Species

# In[ ]:


le = LabelEncoder()
iris.Species = le.fit_transform(iris.Species)


# In[ ]:


iris['Species'].value_counts()


# In[ ]:


iris.head()


# <a id=section4></a> 
# # 4. Model Development

# <a id=section401></a> 
# ## 4.1 Preparing X and y using pandas

# In[ ]:


X = iris.loc[:,iris.columns != 'Species']
X.head()


# In[ ]:


y = iris["Species"]
y[:5]


# In[ ]:


print(X.shape)
print(y.shape)


# <a id=section402></a> 
# ## 4.2 Splitting X and y into training and test datasets.
#  

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# ## Standardizing the variables

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# <a id=section403></a> 
# ## 4.3 Logistic Regression 

# In[ ]:


lr = LogisticRegression(random_state = 0)
lr.fit(X_train,y_train)


# In[ ]:


y_pred_train_lr = lr.predict(X_train)
y_pred_test_lr = lr.predict(X_test)


# In[ ]:


# Accuracy for Logistic Regression
print('Accuracy score for train data is:', accuracy_score(y_train,y_pred_train_lr))
print('Accuracy score for test data is:', accuracy_score(y_test,y_pred_test_lr))


# In[ ]:


# from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_test_lr))
confusion_matrix.index = ['Actual 1','Actual 2','Actual 3']
confusion_matrix.columns = ['Predicted 1','Predicted 2','Predicted 3']
print(confusion_matrix)


# In[ ]:


lr_report = classification_report(y_test, y_pred_test_lr)
print(lr_report)


# <a id=section404></a> 
# ## 4.4 Decision Tree Classifier

# In[ ]:


dt = tree.DecisionTreeClassifier(random_state = 0)
dt.fit(X_train, y_train)


# In[ ]:


y_pred_train_dt = dt.predict(X_train)  
y_pred_test_dt = dt.predict(X_test)  


# In[ ]:


# Accuracy for Decision Tree
print('Accuracy score for train data is:', accuracy_score(y_train,y_pred_train_dt))
print('Accuracy score for test data is:', accuracy_score(y_test,y_pred_test_dt))


# In[ ]:


dt_report = classification_report(y_test, y_pred_test_dt)
print(dt_report)


# <a id=section405></a> 
# ## 4.5 Random Forest Classifier 

# In[ ]:


rf = RandomForestClassifier(random_state = 0)
rf.fit(X_train, y_train)


# In[ ]:


y_pred_train_rf = rf.predict(X_train)
y_pred_test_rf = rf.predict(X_test)


# In[ ]:


# Accuracy for Random Forest
print('Accuracy score for train data:', accuracy_score(y_train,y_pred_train_rf))
print('Accuracy score for test data using the model without parameter specification:', accuracy_score(y_test,y_pred_test_rf))


# In[ ]:


rf_report = classification_report(y_test, y_pred_test_rf)
print(rf_report)


# <a id=section406></a> 
# ## 4.6 KNN Classifier 

# In[ ]:


knn=KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train,y_train) 


# In[ ]:


y_pred_train_knn = knn.predict(X_train)
y_pred_test_knn = knn.predict(X_test)


# In[ ]:


# Accuracy for KNN
print('Accuracy score for train data:', accuracy_score(y_train,y_pred_train_knn))
print('Accuracy score for test data using the model without parameter specification:', accuracy_score(y_test,y_pred_test_knn))


# In[ ]:


knn_report = classification_report(y_test, y_pred_test_knn)
print(knn_report)


# <a id=section407></a> 
# ## 4.7 Naive Bayes Classifier 

# In[ ]:


nb = GaussianNB()
nb.fit(X_train, y_train)


# In[ ]:


y_pred_train_nb = nb.predict(X_train)
y_pred_test_nb = nb.predict(X_test)


# In[ ]:


# Accuracy for Naive Bayes
print('Accuracy score for train data:', accuracy_score(y_train,y_pred_train_nb))
print('Accuracy score for test data using the model without parameter specification:', accuracy_score(y_test,y_pred_test_nb))


# In[ ]:


nb_report = classification_report(y_test, y_pred_test_nb)
print(nb_report)


# <a id=section408></a> 
# ## 4.8 Support Vector Classifier 

# In[ ]:


svc = SVC()
svc.fit(X_train,y_train)


# In[ ]:


y_pred_train_svc = svc.predict(X_train)
y_pred_test_svc = svc.predict(X_test)


# In[ ]:


# Accuracy for Support Vector
print('Accuracy score for train data:', accuracy_score(y_train,y_pred_train_svc))
print('Accuracy score for test data using the model without parameter specification:', accuracy_score(y_test,y_pred_test_svc))


# In[ ]:


svc_report = classification_report(y_test, y_pred_test_svc)
print(svc_report)


# <a id=section5></a> 
# # 5. Model Evaluation

# In[ ]:


model_names = ['--------------------Logistic Regression---------------------\n',  
               '\n------------------Decsision Classifier------------------\n', 
               '\n----------------Random Forest Classifier------------\n',
              '\n--------------------KNN Classifier------------\n',
              '\n-----------------Naive Bayes Classifier------------\n',
              '\n--------------Support Vector Classifier------------\n']
report = model_names[0] + lr_report + model_names[1] + dt_report + model_names[2] + rf_report + model_names[3] + knn_report + model_names[4] + nb_report + model_names[5] + svc_report
print(report)


# <a id=section6></a> 
# # 6. Auto ML
# 

# In[ ]:


MLA = [
    #Ensemble Methods
    ensemble.RandomForestClassifier(),

    
    #Linear Model
    linear_model.LogisticRegression(),
    
    #Tree   
    tree.DecisionTreeClassifier()  ,
    
    
    ]


# In[ ]:


#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .3, train_size = .7, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%


# In[ ]:


#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time', 'TrainTestDifference']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = iris[["Species"]]  # Y 

#index through MLA and save performance to table
row_index = 0
Feature_Importance = {}


# In[ ]:


Y = iris["Species"].values.reshape(-1, 1)


# In[ ]:


for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, X, Y, cv  = cv_split,return_train_score=True,scoring='f1_weighted')

    
    # cv_result is a dictionary -> All the results of diff models are saved 
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    #MLA_compare.loc[row_index, 'TrainTestDifference'] = cv_results['train_score'].mean() - cv_results['test_score'].mean() 

    #save MLA predictions - see section 6 for usage
    alg.fit(X, Y)

    try:
      Feature_Importance[MLA_name] = alg.feature_importances_
    except AttributeError:
      pass
      
    MLA_predict[MLA_name] = alg.predict(X)
    
    row_index+=1
    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
MLA_compare['TrainTestDifference'] = (MLA_compare['MLA Test Accuracy Mean']-MLA_compare['MLA Train Accuracy Mean'])*100
MLA_compare   


# <a id=section7></a> 
# # 7. Neural Network Approach

# In[ ]:


#class 0 [0,0,1]
#class 1 [0,1,0]
#class 2 [0,0,1]


# In[ ]:


y = to_categorical(y)


# In[ ]:


y.shape


# In[ ]:


X = X.values


# In[ ]:


X.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:





# In[ ]:


model=Sequential()
model.add(Dense(16, activation='relu', input_dim=4))
model.add(Dense(32, activation='relu', input_dim=4))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()             


# In[ ]:


model.fit(X_train, y_train,epochs=150, verbose=2)


# In[ ]:


predictions = model.predict_classes(X_test)


# In[ ]:


y_test.argmax(axis=1)


# In[ ]:


# Accuracy for Random Forest
print('Accuracy score for test data:', accuracy_score(y_test.argmax(axis=1),predictions))


# In[ ]:


nn_report = classification_report(y_test.argmax(axis=1), predictions)
print(nn_report)


# In[ ]:


# from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(y_test.argmax(axis=1), predictions))
cm.index = ['Actual 1','Actual 2','Actual 3']
cm.columns = ['Predicted 1','Predicted 2','Predicted 3']
print(cm)


# In[ ]:


model.save('iris_nn_model.h5')


# In[ ]:


from keras.models import load_model
new_model = load_model('iris_nn_model.h5')


# In[ ]:


new_model.predict_classes(X_test)


# <a id=section8></a> 
# # 8. Conclusion

# **KNN Classifier is the best model in terms of accuracy comparison to other models evaluated**
