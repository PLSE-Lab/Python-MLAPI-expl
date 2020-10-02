# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# API kaggle datasets download -d blastchar/telco-customer-churn
telcoData = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

###### Cleaning up the data set######
# Need to clean up data set, turning everything into numerics
# Cleaning up individual columns

#gender column
# Converted Female to 1, and Make to 0
telcoData.gender.replace(to_replace=dict(Female=1, Male=0), inplace=True)
telcoData.gender.head()


# Converting Partner field
# Converting Yes to 1, Converting No to 0
telcoData.Partner.replace(to_replace=dict(Yes=1, No=0), inplace=True)
telcoData.Partner.head()


# Converting Dependents field
# Converting Yes to 1, Converting No to 0
telcoData.Dependents.replace(to_replace=dict(Yes=1, No=0), inplace=True)


# Converting PhoneService field
# Converting Yes to 1, Converting No to 0
telcoData.PhoneService.replace(to_replace=dict(Yes=1, No=0), inplace=True)
telcoData.PhoneService.head()


# Converting Multiple Lines field
#Removing spaces from No Phone Service
telcoData.MultipleLines = telcoData.MultipleLines.replace("No phone service", "No_phone_service")
# Converting Yes to 1, Converting No to 0, Converting No phone service to 2
telcoData.MultipleLines.replace(to_replace=dict(Yes=1, No=0, No_phone_service=2), inplace=True)


# Converting InternetService field
#Removing spaces from Fiber optic
telcoData.InternetService = telcoData.InternetService.replace("Fiber optic", "Fiber_optic")
# Converting Fiber_optic to 1, Converting No to 0, Converting DSL to 2
telcoData.InternetService.replace(to_replace=dict(Fiber_optic=1, No=0, DSL=2), inplace=True)


# Converting OnlineSecurity field
#Removing spaces from No internet Service
telcoData.OnlineSecurity = telcoData.OnlineSecurity.replace("No internet service", "No_internet_service")
# Converting Yes to 1, Converting No to 0, Converting No phone service to 2
telcoData.OnlineSecurity.replace(to_replace=dict(Yes=1, No=0, No_internet_service=2), inplace=True)


# Converting OnlineBackup field
#Removing spaces from No internet Service
telcoData.OnlineBackup = telcoData.OnlineBackup.replace("No internet service", "No_internet_service")
# Converting Yes to 1, Converting No to 0, Converting No phone service to 2
telcoData.OnlineBackup.replace(to_replace=dict(Yes=1, No=0, No_internet_service=2), inplace=True)


# Converting DeviceProtection field
#Removing spaces from No internet Service
telcoData.DeviceProtection = telcoData.DeviceProtection.replace("No internet service", "No_internet_service")
# Converting Yes to 1, Converting No to 0, Converting No phone service to 2
telcoData.DeviceProtection.replace(to_replace=dict(Yes=1, No=0, No_internet_service=2), inplace=True)



# Converting TechSupport field
#Removing spaces from No internet Service
telcoData.TechSupport = telcoData.TechSupport.replace("No internet service", "No_internet_service")
# Converting Yes to 1, Converting No to 0, Converting No phone service to 2
telcoData.TechSupport.replace(to_replace=dict(Yes=1, No=0, No_internet_service=2), inplace=True)



# Converting StreamingTV field
#Removing spaces from No internet Service
telcoData.StreamingTV = telcoData.StreamingTV.replace("No internet service", "No_internet_service")
# Converting Yes to 1, Converting No to 0, Converting No phone service to 2
telcoData.StreamingTV.replace(to_replace=dict(Yes=1, No=0, No_internet_service=2), inplace=True)



# Converting StreamingMovies field
#Removing spaces from No internet Service
telcoData.StreamingMovies = telcoData.StreamingMovies.replace("No internet service", "No_internet_service")
# Converting Yes to 1, Converting No to 0, Converting No phone service to 2
telcoData.StreamingMovies.replace(to_replace=dict(Yes=1, No=0, No_internet_service=2), inplace=True)



# Converting Contract field
#Removing spaces attributes
telcoData.Contract = telcoData.Contract.replace("One year", "One_Year")
telcoData.Contract = telcoData.Contract.replace("Two year", "Two_Year")
telcoData.Contract = telcoData.Contract.replace("Month-to-month", "Month_to_month")
# Converting Yes to 1, Converting No to 0, Converting No phone service to 2
telcoData.Contract.replace(to_replace=dict(One_Year=1, Two_Year=0, Month_to_month=2), inplace=True)


# Converting PaperlessBilling field
# Converting Yes to 1, Converting No to 0
telcoData.PaperlessBilling.replace(to_replace=dict(Yes=1, No=0, No_internet_service=2), inplace=True)


# Converting PaymentMethod field
telcoData.PaymentMethod = telcoData.PaymentMethod.replace("Electronic check", "1")
telcoData.PaymentMethod = telcoData.PaymentMethod.replace("Mailed check", "2")
telcoData.PaymentMethod = telcoData.PaymentMethod.replace("Bank transfer (automatic)", "3")
telcoData.PaymentMethod = telcoData.PaymentMethod.replace("Credit card (automatic)", "4")


# Normalizing the numerical columns
telcoData.tenure = (telcoData.tenure-min(telcoData.tenure))/(max(telcoData.tenure)-min(telcoData.tenure))
telcoData.MonthlyCharges = (telcoData.MonthlyCharges-min(telcoData.MonthlyCharges))/(max(telcoData.MonthlyCharges)-min(telcoData.MonthlyCharges))

# Rounding the numeric columns to the second decimal place
telcoData.tenure = telcoData.tenure.round(2)
telcoData.MonthlyCharges = telcoData.MonthlyCharges.round(2)

# Converting Churn field
# Converting Yes to 1, Converting No to 0
#telcoData.Churn.replace(to_replace=dict(Yes=1, No=0), inplace=True)
#telcoData.Churn.head()
###### Selecting our best attributes ######

# y includes our labels and x includes our features
y = telcoData.Churn                       # Yes or No
#Dropping irrelevant information
list = ['customerID','TotalCharges','Churn']
x = telcoData.drop(list,axis = 1 )

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

select_feature = SelectKBest(chi2, k=14).fit(x_train, y_train)
print('Score list:', select_feature.scores_)
print('Feature list:', x_train.columns)


x_train_2 = select_feature.transform(x_train)
x_test_2 = select_feature.transform(x_test)
#random forest classifier with n_estimators=10 (default)
clf_rf_2 = RandomForestClassifier()      
clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)
ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))
print('Accuracy is: ',ac_2)
cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test_2))
sns.heatmap(cm_2,annot=True,fmt="d")

telcoData.to_csv("telcoData.csv")




select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)

select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)

select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)


# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
#Finding the best 10 features
select_feature = SelectKBest(chi2, k=14).fit(x_train, y_train)
print('Score list:', select_feature.scores_)
print('Feature list:', x_train.columns)