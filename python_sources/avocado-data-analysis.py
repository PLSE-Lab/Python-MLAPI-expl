#!/usr/bin/env python
# coding: utf-8

# ## Table of Content
# 
# 1. [Problem Statement](#section1)<br>
# 2. [Data Loading and Description](#section2)<br>
# 3. [Exploratory Data Analysis](#section3)<br>
#     - 3.1 [Type of Avocado vs Average Price](#section301)<br>
#     - 3.2 [Total Volume vs Small, Large and XLarge](#section302)<br>
#     - 3.3 [Total Bags vs Small Bags, Large Bags and XLarge Bags](#section303)<br>
#     - 3.4 [Region Vs Year distribution](#section304)<br>
#     - 3.5 [Region Vs AveragePrice distribution](#section305)<br>
# 4. [Classifying Type of Avocado](#section4)<br>
#     - 4.1 [Using Logistic Regression](#section401)<br>
#     - 4.2 [Using Random forest classifier](#section402)<br>
# 5. [Predicting Average Price of Avocado](#section5)<br>
#     - 5.1 [Using Linear Regression model](#section501)<br>
#     - 5.2 [Model Evaluation for Linear Regression Model](#section502)<br>
#     - 5.3 [Evaluation of Linear Regression Model using different columns](#section503)<br>
#     - 5.4 [Using Random Forest Regressor](#section504)<br>
#     - 5.5 [Model Evaluation for Random Forest Regressor](#section505)<br>
# 6. [Conclusion](#section6)<br>

# <a id=section1></a> 
# ## 1. Problem Statement !
# 
# "The __Avocado__ dataset we are classifying __Organic & Conventional Type__ and prediting the __Average price__ using Regression model from year __2015, 2016, 2017 and 2018 data.__"
# 

# <a id=section2></a> 
# ## 2. Data Loading and Description
# The Avocado dataset includes consumption of fruit in different regions of USA from 2015 till 2018 years of data.
# * We have two types of Avocado available
# 1. Organic (Healthy)
# 2. Conventional

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

from matplotlib import pyplot

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import warnings                                                                 
warnings.filterwarnings('ignore') 

# allow plots to appear directly in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


#data = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/avocado.csv', index_col=0)
data = pd.read_csv("../input/avocado.csv")
data.drop("Unnamed: 0", axis=1,inplace=True)
names = ['Date', 'AveragePrice', 'TotalVolume', 'Small', 'Large', 'XLarge', 'TotalBags', 'SmallBags', 'LargeBags', 'XLargeBags', 'Type', 'Year' ,'Region']
data = data.rename(columns=dict(zip(data.columns, names)))
data.head()


# We can see there is no null values. We have 18249 records.<br>
# No need to add values on the provided data.

# In[ ]:


data.info()


# In[ ]:


data.describe(include='all')


# ## 3. Exploratory Data Analysis

# In[ ]:


data.Type.unique()


# In[ ]:


data.Year.unique()


# From the data we have four years of data, we can use different years for analysis.<br>
# we can divide our whole dataset into Organic and Conventional types.

# ### 3.1. Type of Avocado vs Average Price 

# In[ ]:


sns.boxplot(y="Type", x="AveragePrice", data=data, palette = 'pink')


# From the above boxplot we can say that Organic fruit price is more as compared to conventional fruit.

# In[ ]:


label = LabelEncoder()
dicts = {}

label.fit(data.Type.drop_duplicates()) 
dicts['Type'] = list(label.classes_)
data.Type = label.transform(data.Type)


# In[ ]:


cols = ['AveragePrice','Type','Year','TotalVolume','TotalBags']
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale = 1.7)
hm = sns.heatmap(cm,cbar = True, annot = True,square = True, fmt = '.2f', annot_kws = {'size':15}, yticklabels = cols, xticklabels = cols)


# We can see there is a strong relation between TotalBags and TotalVolume ie, 0.96 and also Type and AveragePrice ie, 0.62.<br> Other than that there is weak realation.

# ### 3.2. Total Volume vs Small, Large and XLarge

# In[ ]:


sns.pairplot(data, x_vars=['Small', 'Large', 'XLarge'], y_vars='TotalVolume', size=5, aspect=1, kind='reg')


# ### 3.3. Total Bags vs Small Bags, Large Bags and XLarge Bags

# In[ ]:


sns.pairplot(data, x_vars=['SmallBags', 'LargeBags', 'XLargeBags'], y_vars='TotalBags', size=5, aspect=1, kind='reg')


# There is a strong co-relation between TotalVolume Vs Small and TotalBags Vs SmallBags.<br>
# We can say weak co-relation between TotalVolume Vs XLarge and TotalBags Vs XLargeBags.<br>
# Large and LargeBags comes in the middle.

# ### 3.4. Region Vs Year distribution
# From the graph we can say that in year 2017 the HartfordSpringfield region being the maximum consumption of Avocado.

# In[ ]:


plt.figure(figsize=(12,20))
sns.set_style('whitegrid')
sns.pointplot(x='AveragePrice',y='Region',data=data, hue='Year',join=False)
plt.xticks(np.linspace(1,2,5))
plt.xlabel('Region',{'fontsize' : 'large'})
plt.ylabel('AveragePrice',{'fontsize':'large'})
plt.title("Yearly Average Price in Each Region",{'fontsize':20})


# ### 3.5. Region Vs AveragePrice distribution
# From the graph we can say that Organic Type Avocado prices are high in HartfordSpringfield and Sanfrancisco region.<br>
# For Conventional Type we have an average price < 1.50$.

# In[ ]:


plt.figure(figsize=(12,20))
sns.set_style('whitegrid')
sns.pointplot(x='AveragePrice', y='Region', data=data, hue='Type',join=False)
plt.xticks(np.linspace(1,2,5))
plt.xlabel('Region',{'fontsize' : 'large'})
plt.ylabel('AveragePrice',{'fontsize':'large'})
plt.title("Type Average Price in Each Region",{'fontsize':20})


# ## 4. Classifying Type of Avocado

# ### 4.1. Using Logistic Regression

# In[ ]:


X=data[['AveragePrice', 'Small', 'Large', 'XLarge', 'SmallBags', 'LargeBags', 'XLargeBags']] #feature columns
y=data.Type #predictor variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

print("X Train Shape ",X_train.shape)
print("Y Train Shape ",y_train.shape)

print("X Test Shape ",X_test.shape)
print("Y Test Shape ",y_test.shape)


# In[ ]:


#Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred_train = logreg.predict(X_train)  
y_pred_test = logreg.predict(X_test)  

#Acuuracy score
print('Accuracy score for Logistic Regression test data is:', accuracy_score(y_test,y_pred_test))

print('----------------------------------------------------------------------------------------')

#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_test))
confusion_matrix.index = ['organic','Conventional']
confusion_matrix.columns = ['Predicted organic','Predicted Conventional']
print("Confusion matrix for logistic regression model")
print(confusion_matrix)

print('----------------------------------------------------------------------------------------')

#AUC ROC Curve
probs = logreg.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### 4.2. Using Random forest classifier

# In[ ]:


#Randomforest classifier
rfclass = RandomForestClassifier(random_state = 0)
rfclass.fit(X_train, y_train)

y_pred_train = rfclass.predict(X_train)
y_pred_test = rfclass.predict(X_test)

#Accuracy score
print('Accuracy score for test data using Random Forest :', accuracy_score(y_test,y_pred_test))

print('----------------------------------------------------------------------------------------')

#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred_test))
confusion_matrix.index = ['organic','Conventional']
confusion_matrix.columns = ['Predicted organic','Predicted Conventional']
print("Confusion matrix for Random forest model")
print(confusion_matrix)

print('----------------------------------------------------------------------------------------')

#AUC ROC Curve
probs = rfclass.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### 5. Predicting Average Price of Avocado

# In[ ]:


data.drop(['Date', 'TotalVolume', 'TotalBags', 'Region', 'Year'], axis = 1,inplace = True)


# In[ ]:


data.columns


# We are calculting Average price of Avocado considering columns:<br> __['AveragePrice', 'Small', 'Large', 'XLarge', 'SmallBags', 'LargeBags', 'XLargeBags', 'Type']__ <br>

# In[ ]:


scaler = StandardScaler().fit(data)
data_avocado_scaler = scaler.transform(data)
data_avocado = pd.DataFrame(data_avocado_scaler)
data_avocado.columns = ['AveragePrice', 'Small', 'Large', 'XLarge', 'SmallBags', 'LargeBags', 'XLargeBags', 'Type']
data_avocado.head()


# In[ ]:


feature_cols = ['Small', 'Large', 'XLarge', 'SmallBags', 'LargeBags', 'XLargeBags', 'Type']
X = data_avocado[feature_cols]


# In[ ]:


y = data_avocado.AveragePrice


# In[ ]:


def split(X,y):
    return train_test_split(X, y, test_size=0.20, random_state=1)


# In[ ]:


X_train, X_test, y_train, y_test=split(X,y)
print('Train cases as below')
print('X_train shape: ',X_train.shape)
print('y_train shape: ',y_train.shape)
print('\nTest cases as below')
print('X_test shape: ',X_test.shape)
print('y_test shape: ',y_test.shape)


# ### 5.1. Using Linear Regression model

# In[ ]:


def linear_reg( X, y, gridsearch = False):
    
    X_train, X_test, y_train, y_test = split(X,y)
    
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    
    if not(gridsearch):
        linreg.fit(X_train, y_train) 

    else:
        from sklearn.model_selection import GridSearchCV
        parameters = {'normalize':[True,False], 'copy_X':[True, False]}
        linreg = GridSearchCV(linreg,parameters, cv = 10)
        linreg.fit(X_train, y_train)                                                           # fit the model to the training data (learn the coefficients)
        print("Mean cross-validated score of the best_estimator : ", linreg.best_score_)  
        
        y_pred_test = linreg.predict(X_test)                                                   # make predictions on the testing set

        RMSE_test = (metrics.mean_squared_error(y_test, y_pred_test))                          # compute the RMSE of our predictions
        print('RMSE for the test set is {}'.format(RMSE_test))

    return linreg


# In[ ]:


linreg = linear_reg(X,y)


# In[ ]:


linreg.score(X,y)


# In[ ]:


print('Intercept:',linreg.intercept_)                                           # print the intercept 
print('Coefficients:',linreg.coef_)


# In[ ]:


feature_cols.insert(0,'Intercept')
coef = linreg.coef_.tolist()
coef.insert(0, linreg.intercept_)


# In[ ]:


eq1 = zip(feature_cols, coef)

for c1,c2 in eq1:
    print(c1,c2)


# __Y = -0.002 - (Small `*` 0.313) + (Large `*` 0.320) - (XLarge `*` 0.123) + (SmallBags `*` 0.061) - (LargeBags `*` 0.073) + (XLargeBags `*` 0.076) + Type `*` 0.605__
# <br>
# From the above equation __XLarge__ and __LargeBags__ are being __negative__. ie. If the value of __XLarge__ and __LargeBags__ decreases, the __Y__ value will increase and vise-versa.

# In[ ]:


y_pred_train = linreg.predict(X_train)


# In[ ]:


y_pred_test = linreg.predict(X_test)


# Calculating __Mean Absolute error__, __Mean Squared error__, __Root Mean Squared error__

# In[ ]:


MAE_train = metrics.mean_absolute_error(y_train, y_pred_train)
MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)

print('MAE for training set is {}'.format(MAE_train))
print('MAE for test set is {}'.format(MAE_test))

print('----------------------------------------------------------------------------------------')

MSE_train = metrics.mean_squared_error(y_train, y_pred_train)
MSE_test = metrics.mean_squared_error(y_test, y_pred_test)

print('MSE for training set is {}'.format(MSE_train))
print('MSE for test set is {}'.format(MSE_test))

print('----------------------------------------------------------------------------------------')

RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))

print('RMSE for training set is {}'.format(RMSE_train))
print('RMSE for test set is {}'.format(RMSE_test))


# If we compare RMSE and MSE value, we can conclude that RMSE is greater than MSE. 

# ### 5.2. Model Evaluation for Linear Regression Model

# We are calculating __Linear Regression__ model with same type of data.

# In[ ]:


print("Model Evaluation for Linear Regression Model")

print('----------------------------------------------------------------------------------------')

yhat = linreg.predict(X_train)
SS_Residual = sum((y_train-yhat)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print("r_squared for train data ",r_squared, " and adjusted_r_squared for train data",adjusted_r_squared)

print('----------------------------------------------------------------------------------------')

yhat = linreg.predict(X_test)
SS_Residual = sum((y_test-yhat)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r_squared for test data ",r_squared, " and adjusted_r_squared for test data",adjusted_r_squared)


# ### 5.3. Evaluation of Linear Regression Model using different columns

# In[ ]:


feature_cols = ['Small', 'SmallBags', 'Type']
X1 = data_avocado[feature_cols]  
y1 = data_avocado.AveragePrice
linreg=linear_reg(X1,y1, gridsearch = True)


# In[ ]:


feature_cols = ['Large', 'LargeBags', 'Type']
X1 = data_avocado[feature_cols]  
y1 = data_avocado.AveragePrice
linreg=linear_reg(X1,y1, gridsearch = True)


# In[ ]:


feature_cols = ['XLarge', 'XLargeBags', 'Type']
X1 = data_avocado[feature_cols]  
y1 = data_avocado.AveragePrice
linreg=linear_reg(X1,y1, gridsearch = True)


# From the above analysis we can say that __RMSE value 0.6095__ is lower between the three. __Lesser the RMSE value better would be the model.__

# ### 5.4. Using Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor(random_state = 0)
model2.fit(X_train, y_train)
y_pred_train = model2.predict(X_train)
y_pred_test = model2.predict(X_test) 


# ### 5.5. Model Evaluation for Random Forest Regressor

# In[ ]:


print("Model Evaluation for Random Forest Regressor ")
RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))

print('RMSE for training set is {}'.format(RMSE_train),' and RMSE for test set is {}'.format(RMSE_test))

print('----------------------------------------------------------------------------------------')

yhat = model2.predict(X_train)
SS_Residual = sum((y_train-yhat)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print("r_squared for train data ",r_squared, " and adjusted_r_squared for train data",adjusted_r_squared)

print('----------------------------------------------------------------------------------------')

yhat = model2.predict(X_test)
SS_Residual = sum((y_test-yhat)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("r_squared for test data ",r_squared, " and adjusted_r_squared for test data",adjusted_r_squared)


# ### 6. Conclusion

# * Columns like Type of avocado, size and bags have impact on Average Price, __lesser the RMSE value__ accurate the model is, when we consider Small Hass in Small Bags.
# * Random forest Classifier has more accuracy than Logistic regression model for this dataset , accuracy is 0.99 it may also denote it is overfitting as it even classifies the outliers perfectly.
# * Random forest classifier model predicts the __type of Avocado__ more accurately than Logistic regression model. 
# * Random Forest Regressor model predicts the __average price__ more accurately than linear regression model.
