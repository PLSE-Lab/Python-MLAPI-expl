#!/usr/bin/env python
# coding: utf-8

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


## Importing all the necessary libraries.
import pandas as pd
import numpy as np
pd.options.display.max_rows = 300
pd.options.display.max_columns = 300
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

import os

# hide warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


## making two dataframes by importing the test and train data from the csv files.
pd.options.display.max_columns = 300
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")
test_df=pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


##Viewing the contents of train.csv
train_df.head()


# In[ ]:


## Viewing the number of rows and columns in train.csv
train_df.shape


# In[ ]:


##Viewing the contents of the columns in train.csv

train_df.describe()


# In[ ]:


## Tpe of data in the columns of the dataframe
train_df.info()


# In[ ]:


## Viewing the type of data in test.csv
test_df.info()


# In[ ]:


#Checking the percentage missing values in each column for both train and test dataset
round(100*(train_df.isnull().sum()/len(train_df.index)), 2)


# In[ ]:


round(100*(test_df.isnull().sum()/len(test_df.index)), 2)


# In[ ]:


## Here we can see both test and train dataset have more than %)% missing values in the cabin column 
## hence deleting the Cabin column from both test and train dataset
train_df=train_df.drop('Cabin',axis=1)
test_df=test_df.drop('Cabin',axis=1)


# In[ ]:


train_df.info()


# In[ ]:


##Data Preparation
##
#df3['GarageYrBlt']=df3['GarageYrBlt'].convert_objects(convert_numeric=True)
df1=train_df
df1_test=test_df
## Deleting the columns which do not look relevant for our analysis from both train and test dataset
#df1=df1.drop(['Ticket','Name','PassengerId'],axis=1)
#df1_test=df1_test.drop(['Ticket','Name','PassengerId'],axis=1)


# In[ ]:


##Vewing the shape of our dataframes after deleting the columns
df1.shape
print(list(df1.columns))


# In[ ]:


df1_test.shape
print(list(df1_test.columns))


# In[ ]:


##Creating dummy variables
embark = pd.get_dummies(df1['Embarked'],prefix='Embarked',drop_first=True)
#Adding the results to the new dataframe
df1 = pd.concat([df1,embark],axis=1)

pclass = pd.get_dummies(df1['Pclass'],prefix='Pclass',drop_first=True)
#Adding the results to the new dataframe
df1 = pd.concat([df1,pclass],axis=1)

df1['Sex'] = df1['Sex'].map({'male': 1, 'female': 0})

parch = pd.get_dummies(df1['Parch'],prefix='Parch',drop_first=True)
#Adding the results to the new dataframe
df1 = pd.concat([df1,parch],axis=1)

sibsp = pd.get_dummies(df1['SibSp'],prefix='SibSp',drop_first=True)
#Adding the results to the new dataframe
df1 = pd.concat([df1,sibsp],axis=1)

### Repeating the same for the test set

embark = pd.get_dummies(df1_test['Embarked'],prefix='Embarked',drop_first=True)
#Adding the results to the new dataframe
df1_test = pd.concat([df1_test,embark],axis=1)

pclass = pd.get_dummies(df1_test['Pclass'],prefix='Pclass',drop_first=True)
#Adding the results to the new dataframe
df1_test = pd.concat([df1_test,pclass],axis=1)

df1_test['Sex'] = df1_test['Sex'].map({'male': 1, 'female': 0})

parch = pd.get_dummies(df1_test['Parch'],prefix='Parch',drop_first=True)
#Adding the results to the new dataframe
df1_test = pd.concat([df1_test,parch],axis=1)

sibsp = pd.get_dummies(df1_test['SibSp'],prefix='SibSp',drop_first=True)
#Adding the results to the new dataframe
df1_test = pd.concat([df1_test,sibsp],axis=1)


# In[ ]:


df2=df1
df2=df2.drop(['Embarked','Pclass','Parch','SibSp'],axis=1)

df2_test=df1_test
df2_test=df2_test.drop(['Embarked','Pclass','Parch','SibSp'],axis=1)


# In[ ]:


df2.shape


# In[ ]:


#Checking the datatype of all the columns of test and train dataset
df2.info()


# In[ ]:


#Viewing if there are any object datatype in our test dataframe
df2_test.info()


# In[ ]:


#Checking for the outlier in the continuous variables in test and train dataset
num_df= df2[['Age','Fare']] 


# In[ ]:


num_df.describe(percentiles=[.25,.5,.75,.90,.95,.99])


# In[ ]:


num_df.boxplot('Age',figsize=(10,8))


# In[ ]:


num_df.boxplot('Fare',figsize=(10,8))


# In[ ]:


# Viewing outlier in Age column
df2[df2['Age']>75]


# In[ ]:


##Removing the outlier in Age column
df3=(df2[df2['Age']<75]) 


# In[ ]:


# Viewing outlier in Fare column
df3[df3['Fare']>500]


# In[ ]:


##Removing the outlier in Fare column
df4=(df3[df3['Fare']<500]) 


# In[ ]:


##Checking for missing values in the daraset
round((df4.isnull().sum()/len(df4.index)), 2)


# As we can see from above there are no missing values in the dataframe

# In[ ]:


## Feature Standardisation
## We will perform feature standardisation for numberic/countinuous columns.
# Features Scaling and checking dataframe
df4.info()


# In[ ]:



final_df = df4[['Age','Fare']]


# In[ ]:


normalized_df=(final_df-final_df.mean())/final_df.std()
df4 = df4.drop(['Age','Fare'], 1)
df4 = pd.concat([df4,normalized_df],axis=1)
df4.head()


# In[ ]:


## Checking survived rate
survived = (sum(df4['Survived'])/len(df4['Survived'].index))*100
print(survived)


# ##### Hence we have a survived rate of 40%

# In[ ]:


#### Model Building
df5=df4
X = df5.drop(['Ticket','Name','PassengerId','Survived'],axis=1)
#X1=df2_test.drop(['Ticket','Name','PassengerId'],axis=1)
# Putting response variable to y
y = df5['Survived']

y.head()


# In[ ]:


X_train=X
y_train=y
#X_test=X1
#, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)
#print(list(df2_test.columns))


# In[ ]:


## using RFE for feature selection
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 8)             # running RFE with 8 variables as output
rfe = rfe.fit(X,y)
print(rfe.support_)           # Printing the boolean results
print(rfe.ranking_)           # Printing the ranking


# 

# In[ ]:


# Variables selected by RFE 
list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[ ]:


col = X_train.columns[rfe.support_]


# 

# In[ ]:


## Selecting the columns as chosen after performing RFE
X_train.columns[~rfe.support_]


# In[ ]:


## Assessing model with statsmodel
import statsmodels.api as sm 
X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[ ]:


## AS we can see the P-value for Parch_4 is very high hence removing Parch_6
col=col.drop('Parch_4',1)


# In[ ]:


## Assessing model with statsmodel
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[ ]:


## AS we can see that the p-value for SibSp_5 is very high hence dropping that column
col=col.drop('SibSp_5',1)


# In[ ]:


# Let's re-run the model using the selected variables
X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[ ]:


##Now thw p-value for all columns is below 5% checking the VIF values for all columns
### LEts check VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


## THe VIF value for all the columns is less than 5 which is acceptable hence all columns are significant
## making prediction on the train dataset
y_train_pred = res.predict(X_train_sm).values.reshape(-1)


# In[ ]:


y_train_pred[:10]


# In[ ]:


### Creating a dataframe with actual survived flag  and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survived_Prob':y_train_pred})
## the index number from original dataframe is stored as customer id
y_train_pred_final['PassengerId'] = y_train.index
y_train_pred_final.head()


# In[ ]:



##Also adding a Survived_pred column with a cutoff
## of 0.5 or 50% cutoff  i.e. all value above 50% probabilities are 1

y_train_pred_final['Survived_pred'] = y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[ ]:


from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.Survived_pred )
print(confusion)


# In[ ]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.Survived_pred))


# In[ ]:


## Plotting ROC curve to get a more clear threshold value
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.Survived_Prob, drop_intermediate = False )


# In[ ]:


draw_roc(y_train_pred_final.Survived, y_train_pred_final.Survived_Prob)


# In[ ]:


# finding optimal cutoff
### IT is that probability where we get balanced sensitivity and specificity


# In[ ]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[ ]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[ ]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[ ]:


### from above plot and the above table 0.4 is the cutoff as the three values i.e accuracy,sensitivity and specificity
### are nearly the same or coincide


# In[ ]:


y_train_pred_final['final_survived'] = y_train_pred_final.Survived_Prob.map( lambda x: 1 if x > 0.4 else 0)

y_train_pred_final.head()


# In[ ]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.final_survived)


# In[ ]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.final_survived )
confusion2


# In[ ]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[ ]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[ ]:


# Let us calculate specificity
TN / float(TN+FP)


# In[ ]:


# Calculate false postive rate - predicting converted when customer have not converted
print(FP/ float(TN+FP))


# In[ ]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[ ]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[ ]:


### Making predictions on Test-set
X_test=df2_test
X_test = X_test[col]
X_test.head()


# In[ ]:


## Feature standardisation on test set for age column_

final_df_test = X_test[['Age']]
normalized_df=(final_df_test-final_df_test.mean())/final_df_test.std()
X_test = X_test.drop(['Age'], 1)
X_test = pd.concat([X_test,normalized_df],axis=1)
X_test.head()


# In[ ]:


X_test_sm = sm.add_constant(X_test)


# In[ ]:


y_test_pred = res.predict(X_test_sm)


# In[ ]:


y_test_pred[:10]


# In[ ]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[ ]:


# Let's see the head
y_pred_1.head()


# In[ ]:


## Having the final test-dataframe with passenger id and survived column
#y_test_df = pd.DataFrame(y_test)
y_pred_1['PassengerId'] =df2_test['PassengerId']


# In[ ]:


y_pred_1.head()


# In[ ]:


y_pred_1= y_pred_1.rename(columns={ 0 : 'Survived_Prob'})
y_pred_1['Survived'] = y_pred_1.Survived_Prob.map(lambda x: 1 if x > 0.4 else 0)


# In[ ]:


y_pred_1.head()


# In[ ]:


y_pred_Final=y_pred_1


# In[ ]:


## getting the final dataframe with ust two column survived and passenger id
y_pred_Final=y_pred_Final.drop("Survived_Prob",axis=1)


# In[ ]:


## Viewing the contents of the final dataframe
y_pred_Final.head()


# In[ ]:


## Viewing the shape of the final dataframe
y_pred_Final.shape

## Hence as per requirement we have 416 rows and two columns namely PassengerID and Survived
# In[ ]:


## Checking survived rate in the test dataframe
survived = (sum(y_pred_Final['Survived'])/len(y_pred_Final['Survived'].index))*100
print(survived)


# In[ ]:


## Sorting the dataframe with respect to passengerid in ascending order
y_pred_Final=y_pred_Final.sort_values('PassengerId',ascending=True)


# In[ ]:


y_pred_Final.head()


# In[ ]:





# In[ ]:


y_pred_Final.to_csv('submission_15.csv', index=False)


# In[ ]:




