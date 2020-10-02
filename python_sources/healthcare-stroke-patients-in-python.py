#!/usr/bin/env python
# coding: utf-8

# ## Prediction Stroke Patients

# ### Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,auc,roc_auc_score,precision_score,recall_score


# ### Import Files. Train Dataset and Test Dataset. Target Variable is 'Stroke'

# In[ ]:


train_data = pd.read_csv('../input/train_2v.csv')
test_data = pd.read_csv('../input/test_2v.csv')


# In[ ]:


train_data.shape


# In[ ]:


test_data.head()


# ### Datasets Shape

# In[ ]:


print ('Train Data Shape: {}'.format(train_data.shape))

print ('Test Data Shape: {}'.format(test_data.shape))


# ### Description of Train Data 

# In[ ]:


train_data.describe()


# ## Data Preprocessing

# ### Data Cleaning

# ### Missing Values for Train and Test Data

# In[ ]:


train_data.isnull().sum()/len(train_data)*100


# In[ ]:


test_data.isnull().sum()/len(test_data)*100


# In[ ]:


joined_data = pd.concat([train_data,test_data])


# In[ ]:


print ('Joined Data Shape: {}'.format(joined_data.shape))


# ### Missing Data for Joined Data

# In[ ]:


joined_data.isnull().sum()/len(joined_data)*100


# ### Joined Data has bmi 3.33% data is missing and smoking_status is 30.7% missing

# In[ ]:


train_data["bmi"]=train_data["bmi"].fillna(train_data["bmi"].mean())


# In[ ]:


train_data.head()


# ### Handling Categorical Variables

# In[ ]:


label = LabelEncoder()
train_data['gender'] = label.fit_transform(train_data['gender'])
train_data['ever_married'] = label.fit_transform(train_data['ever_married'])
train_data['work_type']= label.fit_transform(train_data['work_type'])
train_data['Residence_type']= label.fit_transform(train_data['Residence_type'])


# In[ ]:


train_data_without_smoke = train_data[train_data['smoking_status'].isnull()]
train_data_with_smoke = train_data[train_data['smoking_status'].notnull()]


# In[ ]:


train_data_without_smoke.drop(columns='smoking_status',axis=1,inplace=True)


# In[ ]:


train_data_without_smoke.head()


# In[ ]:


train_data_with_smoke.head()


# In[ ]:


train_data_with_smoke['smoking_status']= label.fit_transform(train_data_with_smoke['smoking_status'])


# In[ ]:


train_data_with_smoke.head()
train_data_with_smoke.shape


# In[ ]:


train_data_with_smoke.corr('pearson')


# ### Handling Imbalanced Data
# #### Now lets look at the number of positive and negative cases we have for stroke data

# In[ ]:


train_data_with_smoke['stroke'].value_counts()


# In[ ]:


train_data_without_smoke['stroke'].value_counts()


# #### In both cases we can see we are dealing with imbalanced data set, if we go ahead with that there is a high possibility that it ML algorithm will predict no stroke for all data. So we need to make the data more balanced
# 
# #### I am using ROSE method to deal with that and make data more balanced which generates artificial data to make the set more balanced

# In[ ]:


ros = RandomOverSampler(random_state=0)
smote = SMOTE()


# In[ ]:


X_resampled, y_resampled = ros.fit_resample(train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'], 
                                            train_data_with_smoke['stroke'])


# In[ ]:


train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns


# In[ ]:


print ('ROS Input Data Shape for Smoke Data: {}'.format(X_resampled.shape))
print ('ROS Output Data Shape for Smoke Data: {}'.format(y_resampled.shape))


# In[ ]:


X_resampled_1, y_resampled_1 = ros.fit_resample(train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'], 
                                            train_data_without_smoke['stroke'])


# In[ ]:


print ('ROS Input Data Shape for Non Smoke Data: {}'.format(X_resampled_1.shape))
print ('ROS Output Data Shape for Non Smoke Data: {}'.format(y_resampled_1.shape))


# ### Train Test Split of the balanced Data

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2)
print(X_train.shape)
print(X_test.shape)


# In[ ]:


X_train_1,X_test_1,y_train_1,y_test_1 = train_test_split(X_resampled_1,y_resampled_1,test_size=0.2)
print(X_train_1.shape)
print(X_test_1.shape)


# ## Applying Model
# 

# ### Decision Tree Classifier with Smoking Status

# In[ ]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

pred = dtree.predict(X_test)
print(classification_report(y_test,pred))
print (accuracy_score(y_test,pred))
print (confusion_matrix(y_test,pred))

precision = precision_score(y_test,pred)
recall = recall_score(y_test,pred)
print( 'precision = ', precision, '\n', 'recall = ', recall)

y_pred_proba = dtree.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

impFeatures = pd.DataFrame(dtree.feature_importances_ ,index=train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
print (impFeatures)


# ### Decision Tree Classifier without Smoking Status

# In[ ]:


dtree_nosmoke = DecisionTreeClassifier()
dtree_nosmoke.fit(X_train_1,y_train_1)

pred = dtree_nosmoke.predict(X_test_1)
print(classification_report(y_test_1,pred))
print ('Accuracy: {}'.format(accuracy_score(y_test_1,pred)))
print ('COnfusion Matrix: \n {}'.format(confusion_matrix(y_test_1,pred)))

precision = precision_score(y_test_1,pred)
recall = recall_score(y_test_1,pred)
print( 'precision = ', precision, '\n', 'recall = ', recall)

y_pred_proba = dtree_nosmoke.predict_proba(X_test_1)[::,1]
fpr, tpr, _ = roc_curve(y_test_1,  y_pred_proba)
auc = roc_auc_score(y_test_1, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

impFeatures = pd.DataFrame(dtree_nosmoke.feature_importances_ ,index=train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
print (impFeatures)


# ### Logistic Regression Classifier with Smoking Status

# In[ ]:


log = LogisticRegression(penalty='l2', C=0.1)
log.fit(X_train,y_train)

pred = log.predict(X_test)
print(classification_report(y_test,pred))
print (accuracy_score(y_test,pred))
print (confusion_matrix(y_test,pred))

precision = precision_score(y_test,pred)
recall = recall_score(y_test,pred)
print( 'precision = ', precision, '\n', 'recall = ', recall)

y_pred_proba = log.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
impFeatures = pd.DataFrame(log.coef_[0] ,index=train_data_with_smoke.loc[:,train_data_with_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
print (impFeatures)


# ### Logistic Regression Classifier without Smoking Status

# In[ ]:


logg = LogisticRegression(penalty='l2', C=0.1)
logg.fit(X_train_1,y_train_1)

pred = logg.predict(X_test_1)
print(classification_report(y_test_1,pred))
print (accuracy_score(y_test_1,pred))
print (confusion_matrix(y_test_1,pred))

precision = precision_score(y_test_1,pred)
recall = recall_score(y_test_1,pred)
print( 'precision = ', precision, '\n', 'recall = ', recall)

y_pred_proba = logg.predict_proba(X_test_1)[::,1]
fpr, tpr, _ = roc_curve(y_test_1,  y_pred_proba)
auc = roc_auc_score(y_test_1, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

impFeatures = pd.DataFrame(logg.coef_[0] ,index=train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
print (impFeatures)


# ### Random Forest Classifier with Smoking Status

# In[ ]:


ran = RandomForestClassifier(n_estimators=50,random_state=0)
ran.fit(X_train_1,y_train_1)

pred = ran.predict(X_test_1)
print(classification_report(y_test_1,pred))
print (accuracy_score(y_test_1,pred))
print (confusion_matrix(y_test_1,pred))

precision = precision_score(y_test_1,pred)
recall = recall_score(y_test_1,pred)
print( 'precision = ', precision, '\n', 'recall = ', recall)

y_pred_proba = ran.predict_proba(X_test_1)[::,1]
fpr, tpr, _ = roc_curve(y_test_1,  y_pred_proba)
auc = roc_auc_score(y_test_1, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


impFeatures = pd.DataFrame((ran.feature_importances_) ,index=train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns,columns=['Importance']).sort_values(by='Importance',ascending=False)
print (impFeatures)


# ### Feature Importance with random Forest

# #### So we can see that Age, hypertension, heart disease, Residence type, Avg Glucose level, BMI and Smoking status comes as significate variable here. A few of them are intuitive as well, but Gender, Marriage status and Work Status are some which we can ignore.

# In[ ]:


feat_importances = pd.Series(ran.feature_importances_, index=train_data_without_smoke.loc[:,train_data_without_smoke.columns!='stroke'].columns)
feat_importances.plot(kind='barh')


# In[ ]:


test_data["bmi"]=test_data["bmi"].fillna(test_data["bmi"].mean())


# In[ ]:


test_data.drop(axis=1,columns=['smoking_status'],inplace=True)


# In[ ]:


label = LabelEncoder()
test_data['gender'] = label.fit_transform(test_data['gender'])
test_data['ever_married'] = label.fit_transform(test_data['ever_married'])
test_data['work_type']= label.fit_transform(test_data['work_type'])
test_data['Residence_type']= label.fit_transform(test_data['Residence_type'])
pred = ran.predict(test_data)


# In[ ]:


prediction = pd.DataFrame(pred,columns=['Pred'])


# ### Predicted Value. Model Predicted 10 strokes for the test data

# In[ ]:


prediction['Pred'].value_counts()


# ### Conclusion
# Overall we used logistic regression to forecast weather a patient can have stroke or not. We has to deal with imbalanced data which is common in such healthcare problems. For improving the model we could try out other ways of dealing with imbalanced data like SMOTE.
# 
# Also we could have dealt with missing data of smoke status in other ways as well for e.g. Age less than 10 or 15 years patients could have been tagged as never_smoked etc.
# 
# Finally just one thought on why the 2 models were so different, one of the reasons could be the age distribution of the 2 data set. Median age of Smoke dataset was 48 while that of Non smoke dataset was 21. These are some ways Logistic model could have been improved.

# In[ ]:





# In[ ]:




