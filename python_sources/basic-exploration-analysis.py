#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Please give your feedback about my overall approach.


Steps Performed. 

1. Simple Data exploratory Analysis
2. Understanding data through graph 
3. modelling  
4. Udnerstanding Accuracy  
5. Trying out different models (I will update this section soon)

By looking through data initially, some stats i tried to make out. 
Who are your customers:
    
1. 80% of customers are young, wih No partner. 


Who are helping to not to churn:

1. 36% of customers having partners
2. 45% having no dependants
3. 63% have phone service
4. Online Security / Backup help to retain customers as well. 

Who are impacting Churn:
    

1. 18% Non-Senior Citizens
2. 16% of Non-Partners
3. 20% of dependants. 
4. 23% having phone service
5. 20% of customers dont have Online security 
   (Partial impact, customers still stay even if they dont have online security)
6. 20% of customers dont have Online backup 
   (Partial impact, customers still stay even if they dont have online backup)
No Impact:
1. Gender
2. Multiple Lines

"""

# Importing the libraries
import numpy as np               # Contains Math tools / functions
import matplotlib.pyplot as plt  # Help us plot charts
import seaborn as sns            # for graphs
import pandas as pd              # Helps to import / manage datasets
import warnings


# Importing the dataset
dataset = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

dataset.head()
# Handling missing values
dataset['TotalCharges'] = dataset['TotalCharges'].replace(" ", 0).astype('float32')


# Basic visuals to see how each feature react

pd.crosstab(dataset.Churn,dataset.gender).plot(kind='bar',title='Churn vs gender'
           ,rot=1)

# Customer base is almost young. 
pd.crosstab(dataset.Churn,dataset.SeniorCitizen).plot(kind='bar',title='Churn vs SeniorCitizen'
           ,rot=1)

pd.crosstab(dataset.Churn,dataset.Partner).plot(kind='bar',title='Churn vs Partner'
           ,rot=1)

pd.crosstab(dataset.Churn,dataset.Dependents).plot(kind='bar',title='Churn vs Dependents'
           ,rot=1)

# Phone service is impacting. 
pd.crosstab(dataset.Churn,dataset.PhoneService).plot(kind='bar',title='Churn vs PhoneService'
           ,rot=1)

pd.crosstab(dataset.Churn,dataset.MultipleLines).plot(kind='bar',title='Churn vs MultipleLines'
           ,rot=1)
           
#Online security is vital and impacting a lot. 
pd.crosstab(dataset.Churn,dataset.OnlineSecurity).plot(kind='bar',title='Churn vs OnlineSecurity'
           ,rot=1)
# Online backup has same issue as online security. 
pd.crosstab(dataset.Churn,dataset.OnlineBackup).plot(kind='bar',title='Churn vs OnlineBackup'
           ,rot=1)
           
# Device Protection is also impacting churn
pd.crosstab(dataset.Churn,dataset.DeviceProtection).plot(kind='bar',title='Churn vs DeviceProtection'
           ,rot=1)
#Techsupport has issues as well
pd.crosstab(dataset.Churn,dataset.TechSupport).plot(kind='bar',title='Churn vs TechSupport'
           ,rot=1)

pd.crosstab(dataset.Churn,dataset.StreamingTV).plot(kind='bar',title='Churn vs StreamingTV'
           ,rot=1)

pd.crosstab(dataset.Churn,dataset.StreamingMovies).plot(kind='bar',title='Churn vs StreamingMovies'
           ,rot=1)

pd.crosstab(dataset.Churn,dataset.PaperlessBilling).plot(kind='bar',title='Churn vs PaperlessBilling'
           ,rot=1)




X=pd.get_dummies(data=dataset,columns=['InternetService'
                                       , 'Contract'
                                       ,'PaymentMethod'
                                       ])
    
X.gender=np.where(dataset.gender == 'Male',1,0)    
X.Partner=np.where(dataset.Partner == 'Yes',1,0)    
X.Dependents=np.where(dataset.Dependents == 'Yes',1,0)    
X.PhoneService=np.where(dataset.PhoneService == 'Yes',1,0)    
X.MultipleLines=np.where(dataset.MultipleLines == 'Yes',1,0)    
X.OnlineSecurity=np.where(dataset.OnlineSecurity == 'Yes',1,0)    
X.OnlineBackup=np.where(dataset.OnlineBackup == 'Yes',1,0)    
X.DeviceProtection=np.where(dataset.DeviceProtection == 'Yes',1,0)    
X.TechSupport=np.where(dataset.TechSupport == 'Yes',1,0)    
X.StreamingTV=np.where(dataset.StreamingTV == 'Yes',1,0)    
X.StreamingMovies=np.where(dataset.StreamingMovies == 'Yes',1,0)    
X.PaperlessBilling=np.where(dataset.PaperlessBilling == 'Yes',1,0)    
X.Churn=np.where(dataset.Churn == 'Yes',1,0)    
#X = X.drop(['customerID','Churn'],axis=1)
X = X.drop(['customerID'],axis=1) 




# Corelation matrix with all other features
# Short term Contracts, Features like billing, streaming impacts positively.
# Long term contracts, suppports are negative impacts. 
plt.figure(figsize=(15,8))
X.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

# Correlation also confirms along with individual feature understanding through cross tabs that,
# 1. customers are young, Facilities arent given to them properly which impacts churn heavy.  

# Given the tenure seem to be completely negative, lets see how tenure really impacts

sns.distplot(X.tenure, hist=True, kde=False, 
             bins=int(180/10), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# Bins = This will explain how many bars you need, you have 18 here
# Most of the customers tend to leave less then 12 months. 


y = np.where(dataset.Churn == 'Yes',1,0) 
X1=np.array(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 0)


""" ------------------------------------------------------------------------- """

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import mean_absolute_error, accuracy_score
print(accuracy_score(y_test,y_pred))











