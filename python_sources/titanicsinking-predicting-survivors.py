#!/usr/bin/env python
# coding: utf-8

# In[162]:


#The following codes implement a Machine Learning classification on
#the titanic passenger dataset to predict which passenger survived.
#the dataset is already divided into a titanic_train and a titanic_test sets.
#A Random Forests Classifier model is trained using the titanic_train dataset.
#The trained model is then used to predict survival of passengers listed in the test set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#this command enables auto prediction of commands
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[163]:


#read in the train and test datasets.
titanic_train=pd.read_csv('../input/train.csv')
titanic_test=pd.read_csv('../input/test.csv')


# In[164]:


#take a peek on the features involved
titanic_train.head()


# In[165]:


#check out the size of the training dataset
len(titanic_train)


# In[166]:


#check out the size of the test dataset
len(titanic_test)


# In[167]:


#check out the total categories in Pclass
titanic_train['Pclass'].unique()


# In[168]:


#check out the total categories in Embarked (ports of embarkation)
titanic_train['Embarked'].unique()


# In[169]:


#check out the total categories in parent-child count
titanic_train['Parch'].unique()


# In[170]:


#Do an analysis of how passenger gender has an effect on survival rate
titanic_train.groupby(['Survived','Sex']).count()['PassengerId']


# In[171]:


print('The survival rate of females is',round(233*100/(81+233),2),'%')
print('The survival rate of males is',round(109*100/(468+109),2),'%')


# In[172]:


#Do an analysis of how passenger class has an effect on survival rate
titanic_train.groupby(['Survived','Pclass']).count()['PassengerId']


# In[173]:


print('The survival rate of Class1 passenger is',round(136*100/(136+80),2),'%')
print('The survival rate of Class2 passenger is',round(87*100/(87+97),2),'%')
print('The survival rate of Class3 passenger is',round(119*100/(119+372),2),'%')


# In[174]:


#take a peek into the test dataset
titanic_test.head()


# In[175]:


#Combine both titanic train and titanic test datasets, to carry out imputation operation on this total dataset
#We have certain missing values in train and missing values in test sets, so by combining the sets, we have a 
#larger dataset to do a better imputation.
combi=pd.concat([titanic_train,titanic_test],axis=0)


# In[176]:


#check out the total size of the combined test and train dataset.
len(combi)


# In[177]:


#Check the total number of missing values in each field.
#No need to impute missing Survived values because this is the independent variable we want to predict!
#For Cabin field, because it has more than 77% (1014/1309) missing values, there is little value in imputing so many fields.
#Thus we only impute Age, Embarked and Fare fields.
combi.isnull().sum()


# In[178]:


#imputation starts here
from sklearn.preprocessing import Imputer

imputer_mean=Imputer(strategy='mean')
combi[['Age']]=imputer_mean.fit_transform(combi[['Age']])

imputer_median=Imputer(strategy='median')
combi[['Fare']]=imputer_median.fit_transform(combi[['Fare']])


# In[179]:


#Take a peek into the combined dataset.
combi.head()


# In[180]:


#Check out after initial imputation, which variables still remain to be imputed.
#We see Embarked remains to be dealt with!
combi.isnull().sum()


# In[181]:


#We do a check on the proportion of passengers embarking at each of the ports.
#The code will show that most passengers embarked at port S (Southampton)
#Thus to impute Embarked, it is more likely that using 'S' would be the correct value.
combi.groupby(['Embarked']).count()['PassengerId']


# In[182]:


#Find out those passengers with null Embarked field.
combi[combi['Embarked'].isnull()]


# In[183]:


#For these passengers whose Embarked is null, we set to 'S'.
combi.Embarked[combi['PassengerId']==62]='S'
combi.Embarked[combi['PassengerId']==830]='S'


# In[184]:


#check that these passengers with null embarkation field now have 'S' as their embarkation port
combi[combi['PassengerId']==62]


# In[185]:


#check that these passengers with null embarkation field now have 'S' as their embarkation port
combi[combi['PassengerId']==830]


# In[186]:


#Now do a final confirmation that the null values for Embarked have been succcessfully imputed
combi.isnull().sum()


# In[187]:


combi.columns


# In[188]:


#With all the imputation done in the combined dataset,
#we split this "cleaned" combined dataset back into its original training and test sets
titanic_train=combi.iloc[0:890,]
titanic_test=combi.iloc[891:,]


# In[189]:


#list out all the variables in titanic dataset.
titanic_train.columns


# In[190]:


#We split the train dataset into constituent independent and dependent variables
iv=titanic_train[['Age','Embarked','Fare','Parch','Pclass','Sex','SibSp']]
dv=titanic_train[['Survived']]


# In[191]:


#perform OneHotEncoding to convert all categorical variables into binary vectors
iv=pd.get_dummies(iv,drop_first=True)


# In[192]:


#take a peek into the iv_train dataset after converion of categorical variables into binary vectors
iv.head()


# In[193]:


from sklearn.model_selection import train_test_split
iv_train,iv_test,dv_train,dv_test=train_test_split(iv,dv,test_size=0.2,random_state=0)


# In[194]:


#Perform feature scaling to normalise all variabls to comparable scales so that 
#the analysis will not be skewed by certain variables taking on large values.
#No need to scale Gender, since it is already in 1s and 0s.
#No need to scale Pclass, since it is only 1,2,3.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
iv_train[['Age','Fare','Parch','SibSp']] = sc.fit_transform(iv_train[['Age','Fare','Parch','SibSp']])
iv_test[['Age','Fare','Parch','SibSp']] = sc.transform(iv_test[['Age','Fare','Parch','SibSp']])


# In[195]:


#try Logistic Regression classification technique
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(random_state=1)
log_reg.fit(iv_train,dv_train)
log_reg.predict(iv_test)


# In[196]:


results=pd.DataFrame()
results['Actuals']=pd.Series(range(len(dv_test)))
results['Actuals']=pd.DataFrame(dv_test).reset_index(drop=True) #if you do not put drop=True, then the actual index values will show


# In[197]:


#check to see that items are in order
results.head()


# In[198]:


#display the predicted values (using Log Regression) compare with actual test set values
results['Log Predicted']=pd.DataFrame(log_reg.predict(iv_test)).reset_index(drop=True)


# In[199]:


#apply Confusion Matrix to the Log Regression model to measure the accuracy of the predictions
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(dv_test,log_reg.predict(iv_test))
cm


# In[200]:


TN=cm[0][0]
FP=cm[0][1]
FN=cm[1][0]
TP=cm[1][1]


# In[201]:


#Log Regression model accuracy
LR_accuracy = round((TP+TN)/(TP+FN+TN+FP),2)

#Log Regression model precision
LR_precision = round(TP/(TP+FP),2)

#Log Regression model sensitivity
LR_sensitivity = round(TP/(TP+FN),2)

#Log Regression model F-score
LR_Fscore= 2*LR_precision*LR_sensitivity / (LR_precision+LR_sensitivity)


# In[202]:


#now try using Random Forests model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=500) #can also specify no. of trees you want by RandomForestRegressor(n_estimators=100)
rf.fit(iv_train,dv_train)


# In[203]:


y_rf_pred=rf.predict(iv_test)


# In[204]:


y_rf_pred


# In[205]:


#display the predicted values (using Random Forests) compare with actual test set values
results['RF Predicted']=pd.DataFrame(rf.predict(iv_test)).reset_index(drop=True)


# In[206]:


#display predictions using both Log Regression and Random Forests classifiers
results.head(10)


# In[207]:


#apply Confusion Matrix to the Random Forests
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(dv_test,y_rf_pred)
cm


# In[208]:


TN=cm[0][0]
FP=cm[0][1]
FN=cm[1][0]
TP=cm[1][1]


# In[209]:


#RF model accuracy
RF_accuracy = round((TP+TN)/(TP+FN+TN+FP),2)

#RF model precision
RF_precision = round(TP/(TP+FP),2)

#RF model sensitivity
RF_sensitivity = round(TP/(TP+FN),2)

#RF model F-score
RF_Fscore= 2*RF_precision*RF_sensitivity / (RF_precision+RF_sensitivity)


# In[210]:


#create the test dataset
Actual_test_set=titanic_test[['Age','Embarked','Fare','Parch','Pclass','Sex','SibSp']]

#perform OneHotEncoding to convert all categorical variables into binary vectors
Actual_test_set=pd.get_dummies(Actual_test_set,drop_first=True)


# In[211]:


#Create a dataframe to store survivals predictions using the more accurate model, with two columns PassengerId and Survived.
if ((RF_Fscore>LR_Fscore) and (RF_accuracy>LR_accuracy)):
    submission = pd.DataFrame({
        "PassengerId": titanic_test['PassengerId'],
       "Survived": (rf.predict(Actual_test_set))
    })
else:
    submission = pd.DataFrame({
        "PassengerId": titanic_test['PassengerId'],
       "Survived": (log_reg.predict(Actual_test_set))
    })


# In[212]:


#Display the dataframe to show that everything is in order
submission.head(10)


# In[213]:


#confirm that the number of passengers in the dataframe is same size as the titanic_test dataset.
len(submission)


# In[214]:


#Submit survival prediction output file to Kaggle
submission.to_csv('submission.csv',index=False)

