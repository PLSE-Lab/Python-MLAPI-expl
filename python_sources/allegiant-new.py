#!/usr/bin/env python
# coding: utf-8

# Batman! lets load couple of required libraries which we will require in our analysis before we jump into predicting JOKER and we will see that joker is not entirely random

# In[280]:


#loading required libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score

import os
print(os.listdir("../input"))


# **Lets load the dataset to see what information batman needs to know and prove him that joker isn't entirely random and he can predict the future. In this case JOKER!!!!!!!!**

# In the **First** part we will predict joker present using just sensor_data dataset and later in the **Second** part we will combine all the given dataset and make predictions and see which one peforms better

# **PART 1**

# In[281]:


#loading sensor_data
data = pd.read_csv('../input/sensor_data.csv')
data.head(5)


# In[282]:


# checking datatypes of all the features
data.dtypes


# It looks like we need to change some of their datatypes.

# In[283]:


data = data.astype({"x": int, "y": int, "z": int})
data['Timestamp'] = pd.to_datetime(data['Timestamp'])


# In[284]:


# checking datatypes again after correction
data.dtypes


# PERFECT!!!!! we are ready to analyze this dataset

# Lets check for the **total number of records** in the dataset

# In[285]:


len(data)


# Lets check for **null values** in the dataset. 

# In[286]:


data.isnull().sum()


# AWESOME! There are no null values in the data. Lets proceed to further analysis.

# Lets check **target** variable distribution. Here target variable is **Joker Present?**. This will give us the idea of class balance/imbalance. 

# In[287]:


print(data['joker present?'].value_counts())
sns.countplot(x='joker present?', data=data, palette="deep")
#data['joker present?'].value_counts().plot.bar()


# WOW! looks like majority of the records says that joker present?  is No. This shows that this is an **unbalanced classification problem** since only 5% records has joker present? as Yes. Lets move on keeping this in mind.

# Lets check **Univariate analysis** to get an idea on its distribution.

# In[288]:


import seaborn as sns
import matplotlib.pyplot as plot
print("**********X***************")
print(data['x'].value_counts())
print("******************************")
print("**********Y**************")
print(data['y'].value_counts())
print("******************************")
print("***********Z********")
print(data['z'].value_counts())


plot.figure(figsize=(14,14))
plot.subplot(2,2,1)
ax = sns.countplot(x='y', data=data, palette="vlag")
ax.set_xlabel("x")

plot.subplot(2,2,2)
ax = sns.countplot(x='y', data=data, palette="pastel")
ax.set_xlabel("y")

plot.subplot(2,2,3)
ax = sns.countplot(x='z', data=data, palette="Set3")
ax.set_xlabel("z")

plot.subplot(2,2,4)
ax = sns.countplot(x='number of thugs',data=data,order=pd.value_counts(data['number of thugs']).iloc[:20].index)
ax.set_xlabel("number of thugs")


# Interesting!!! all setup sensor coordinates (x,y and z) distribution are equal. This proves that sensor are scattered in a very organized fashion. Cool.

# Lets do some **Bivariate Analysis**.

# In[289]:


df1 = data['number of thugs'].value_counts()[:20]
df1


# In[290]:


plot.figure(figsize=(14,14))
plot.subplot(2,2,1)
ax = sns.countplot(x=data['joker present?'],hue=data['x'],data=data, palette="Set3")
ax.set_xlabel("x")

plot.subplot(2,2,2)
ax = sns.countplot(x=data['joker present?'],hue=data['y'],data=data, palette="Set3")
ax.set_xlabel("y")

plot.subplot(2,2,3)
ax = sns.countplot(x=data['joker present?'],hue=data['z'],data=data, palette="Set3")
ax.set_xlabel("z")


# Let's check whether we have noise (outliers) in our dataset which might hinder our analyis.... sounds good?

# In[291]:


plot.figure(figsize=(14,14))
plot.subplot(2,2,1)
ax = sns.boxplot(x="number of citizens", data=data, palette="Set3")
ax.set_xlabel("Number of citizens")

plot.subplot(2,2,2)
ax = sns.boxplot(x="number of thugs", data=data, palette="Set3")
ax.set_xlabel("Number of thugs")


# Did you see **number of thugs** feature looks like having lot of outliers? Okay we will keep this in mind.

# Now, lets **encode** our target variable (Joker Present?), so that later modeling part makes more sense. 

# In[292]:


data['joker present?'].replace({'no':0, 'yes':1}, inplace=True)


# Do you want to see datatypes again of this dataset to see whether the above change worked?

# In[293]:


data.dtypes


# COOL!!!

# Lets see how these features are **correalted** to each other. 

# In[294]:


# checking correlation
corr = data.corr()
abs(corr['joker present?']).sort_values(ascending=False)


# It looks that the largest correlation value is 0.07 which is very less. Since, the dependent/target variable (joker present?) is binary and independent variables are continuous, in such scenario, correlation is not a good matrix to judge. 

# Lets drop Timestamp feature, since it seems not useful here in our problem.

# In[295]:


#data1= data

data1 = data.copy(deep=True)


# In[296]:


data1.drop(['Timestamp'],axis=1, inplace=True)


# Again! checking dataset to see change effect

# In[297]:


data1.head(5)


# In[298]:


data.head(5)


# AWESOME!! we are ready to move on model building part. 

# Lets first create base model to see **Feature Importance**. 

# In[299]:


#separating target variable.
y = data1['joker present?']
data1.drop(['joker present?'],axis=1, inplace=True)


# In[300]:


# splitting our dataset into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data1, y, test_size = 0.3, random_state = 100)


# In[301]:


len(y_test)


# Alright! lets build our base model using **Lightgbm** algorithm and see feature importance. sounds good?

# In[302]:


import lightgbm as lgb
model_lgb = lgb.LGBMClassifier(n_estimator=2000, 
                         learning_rate =0.08
                         )
model_lgb.fit(X_train, y_train)


# In[303]:


eli5.explain_weights(model_lgb)


# Cool! so it looks like number of citizens is more important feature in our dataset followed by number of thugs. Hmmm. 

# Lets predict the future and see how accurate it is with lightgbm algorithm on test data

# In[304]:


y_pred_lgbm = model_lgb.predict(X_test)


# **Confusion Matrix?** lets check it.

# In[305]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_lgbm)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# OK! so the above confusion matrix shows that out of 70956 records test dataset, our model with LGBMClassifier shows (68306+0) = 68306 correct prediction and (2650+0) = 2650 incorrect prediction

# Lets check other score matrix....

# In[306]:


print(classification_report(y_test,y_pred_lgbm))


# In[307]:


print("Accuracy score with baseline lightgbm:", metrics.accuracy_score(y_test, y_pred_lgbm))
print("roc_auc score with decision tree:", roc_auc_score(y_test, y_pred_lgbm))


# Allright! so **accuracy** of the model with lightgbm is 96%

# In[308]:


cv_scores = cross_val_score(model_lgb, X_train, y_train, cv=10)
cv_scores


# In[309]:


print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))


# Lets move on using another classification algorithm. **Decision Tree** and lets use class_weight as balanced because our dataset looks unbalanced. class_weight as balanced uses the values of target variable to automatically adjust weights inversely proportional to class frequencies in the input data.

# In[310]:


from sklearn.tree import DecisionTreeClassifier
#model_decision = DecisionTreeClassifier(min_samples_split=20, random_state=100)
model_decision = DecisionTreeClassifier(class_weight="balanced", random_state=100, max_depth=1)
model_decision.fit(X_train, y_train)


# In[311]:


y_pred_decision = model_decision.predict(X_test)


# In[312]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_decision)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# So the above confusion matrix shows that out of 70956 records test dataset, our model with decision tree shows (65964+298) = 66262 correct predictions and (2352+2342) = 4694 incorrect predictions

# Lets check other score metrics for our model with decision tree.

# In[313]:


print(classification_report(y_test,y_pred_decision))


# In[314]:


print("Accuracy score with decision tree:", metrics.accuracy_score(y_test, y_pred_decision))
print("roc_auc score with decision tree:", roc_auc_score(y_test, y_pred_decision))


# OK! so the **accuracy** of the model with decision tree is 93%

# In[315]:


cv_scores = cross_val_score(model_decision, X_train, y_train, cv=10)
cv_scores


# In[316]:


print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))


# Now lets try simplest **Logistic Regression** to your dataset.

# In[317]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred_log=logreg.predict(X_test)


# In[318]:



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_log)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# So the above confusion matrix shows that out of 70956 records test dataset, our model with decision tree shows (68306+0) = 68306 correct predictions and (2650+0) =2650  incorrect predictions

# Allright! we will check other score metrics now...

# In[319]:


print(classification_report(y_test,y_pred_log))


# In[320]:


print("Accuracy score with logistic regression:", metrics.accuracy_score(y_test, y_pred_log))
print("roc_auc score with logistic regression:", roc_auc_score(y_test, y_pred_log))


# Lets check whether we can improve sensitivity/recall score with different threshold. That means when joker is actually present, how often does our model predict that the joker is present?

# In[321]:


from sklearn.preprocessing import binarize
for i in range(1,5):
    cm2=0
    y_pred_prob_yes=logreg.predict_proba(X_test)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')


# Looks like maximimum sensitivity achieved is 0.06. 

# Now let see ROC-AUC curve through visualization. 

# In[322]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# In[323]:


cv_scores = cross_val_score(logreg, X_train, y_train, cv=10)
print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))


# Want to do **Random Forest?** Lets check that as well. 

# In[324]:


from sklearn.ensemble import RandomForestClassifier
random_classifier= RandomForestClassifier(class_weight="balanced", random_state=100)
random_classifier.fit(X_train,y_train)


# In[325]:


y_pred_random= random_classifier.predict(X_test)


# In[326]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_random)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# So the above confusion matrix shows that out of 70956 records test dataset, our model with decision tree shows (65486+241) = 65727 correct predictions and (2820+2409) = 5229 incorrect predictions

# In[327]:


print(classification_report(y_test,y_pred_random))


# In[328]:


print("Accuracy score with random forest:", metrics.accuracy_score(y_test, y_pred_random))
print("precision score with random forest:", metrics.precision_score(y_test, y_pred_random))


# In[329]:


cv_scores = cross_val_score(random_classifier, X_train, y_train, cv=10)
print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))


# POLL Time!!! Selecting the best model uptil now.... So, since lightgbm and logistic regression has same and hightest prediction accuracy is 96%. Random Forest also performed very well. Considering the accuracy of the models, we would make **lightgbm** winner because it is more robust to outlier compared to logistic regression in comparison 

# **PART 2**

# In this part we will combine all the given datasets. We will make analysis and predictions on this combined datasets and see whethere it makes any difference or not. 

# In[330]:


#loading all the datasets
data2= data
df1 = pd.read_csv('../input/bat_signal_data.csv')
df2 = pd.read_csv('../input/moon_phases_data.csv')
df3 = pd.read_csv('../input/weather_data.csv')
data2.head(5)


# COOL!! now lets combine all the above datasets

# In[331]:


data3= pd.concat([data2, df1, df2, df3], ignore_index=True)
data3.head(5)


# Lets check the datatypes of all the combined features.......

# In[332]:


# checking the data types
data3.dtypes


# Hmmm!!! Looks like we want to change datatypes of some of the features..Lets do it!!

# In[333]:


data3['x'] = data3['x'].fillna(0).astype(int)
data3['y'] = data3['y'].fillna(0).astype(int)
data3['z'] = data3['z'].fillna(0).astype(int)
data3['joker present?'] = data3['joker present?'].fillna(0).astype(int)
data3['number of citizens'] = data3['number of citizens'].fillna(0).astype(int)
data3['number of thugs'] = data3['number of thugs'].fillna(0).astype(int)
data3['temperature'] = data3['temperature'].fillna(00.00000)


# Allright! lets check datatypes again..

# In[334]:


data3.dtypes


# In[335]:


data3.head(5)


# Looks like we can still see NaNs' in our dataset. Lets manipulate them..

# In[336]:


data3['phase'] = data3['phase'].fillna('no info phase')
data3['precip'] = data3['precip'].fillna('no info precip')
data3['sky conditions'] = data3['sky conditions'].fillna('no info sky')
data3['status'] = data3['status'].fillna('no info status')


# Want to check our dataset again?

# In[337]:


data3.head(5)


# In[338]:


import seaborn as sns
import matplotlib.pyplot as plot
print("**********Phase***************")
print(data3['phase'].value_counts())
print("******************************")
print("**********precip**************")
print(data3['precip'].value_counts())
print("******************************")
print("***********sky conditions********")
print(data3['sky conditions'].value_counts())
print("******************************")
print("*********status*************")
print(data3['status'].value_counts())

plot.figure(figsize=(25,35))
plot.subplot(2,2,1)
ax = data3['phase'].value_counts().plot.bar()
ax.set_xlabel("Phase")

plot.subplot(2,2,2)
ax = data3['precip'].value_counts().plot.bar()
ax.set_xlabel("precip")

plot.subplot(2,2,3)
ax = data3['sky conditions'].value_counts().plot.bar()
ax.set_xlabel("sky conditions")

plot.subplot(2,2,4)
ax = data3['status'].value_counts().plot.bar()
ax.set_xlabel("status")


# Since four features (sky conditions, status, Phase, Precip) in the combined datasets are categorical, we need to encode them. 

# In[339]:


#encoding categorical variables
precip=pd.get_dummies(data3['precip'])
sky_condition = pd.get_dummies(data3['sky conditions'])
status = pd.get_dummies(data3['status'])
phase = pd.get_dummies(data3['phase'])


# In[340]:


#adding all the above dataframes into our main dataframe
data3 = pd.concat([data3, phase, status, sky_condition, precip], axis=1)


# want to check our final dataframe? Lets do it. 

# In[341]:


data3.head(5)


# Looks it the categorical features are still present, but now we have their dummies as well. So lets get rid of the main categrical features. 

# In[342]:


data3.drop(['phase','precip','sky conditions','status'],axis=1, inplace=True)


# In[343]:


data3.drop(['Timestamp'], axis=1, inplace=True)


# In[344]:


data3.head(5)


# OK! we are set to analyze this above dataset. 

# First, lets check correlation of all the features with our target feature 'joker present?'

# In[345]:


corr = data3.corr()
abs(corr['joker present?']).sort_values(ascending=False)


# Hmmm! Correlation is very less. Might be because of the reason described in the Part 1. Lets move on. 

# In[346]:


#splitting target column
y_new = data3['joker present?']


# In[347]:


#dropping target feature from the main dataframe
data3.drop(['joker present?'],axis=1, inplace=True)


# Lets see our dataframe again to see whether things are as expected. 

# In[348]:


data3.head(5)


# In[349]:


# splitting our dataset into train and test 
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(data3, y_new, test_size = 0.3, random_state = 100)


# In[350]:


#check the split of train and test shape
print('Train:',X_train_new.shape)
print('Test:',X_test_new.shape)


# Allright! Time to start model building and see how well we can predict the JOKER.....

# Lets start again with **LightGBM** and see feature importance. 

# In[351]:


model = lgb.LGBMClassifier(n_estimator=2000,
                         learning_rate =0.05
                         )
model.fit(X_train_new, y_train_new)
eli5.explain_weights(model, top=30)


# Cool! so it looks like number of thugs is more important feature in our dataset followed by number of citizens. Hmmm. 

# In[352]:


y_pred_lgbm_new = model.predict(X_test_new)


# **Confusion Matrix**? Lets check it...

# In[353]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test_new,y_pred_lgbm_new)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# OK! so the above confusion matrix shows that out of 71583 records test dataset, our model with LGBMClassifier shows (68992+0) = 68992 correct prediction and (2591+0) = 2591 incorrect prediction.

# Lets check some other score metrics

# In[354]:


print(classification_report(y_test_new,y_pred_lgbm_new))


# In[355]:


print("Accuracy score with baseline lightgbm:", metrics.accuracy_score(y_test_new, y_pred_lgbm_new))
print("roc_auc score with baseline lightgbm:", roc_auc_score(y_test_new, y_pred_lgbm_new))


# COOL!! we are getting 96% accuracy of our model with lightgbm.

# Lets do some feature scaling using **standard scaler** and see whether we are getting any difference in our model using same lightgbm. 

# In[356]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_new1 = sc.fit_transform(X_train_new)
X_test_new1 = sc.fit_transform(X_test_new)


# In[357]:


X_train_new1


# In[358]:


model = lgb.LGBMClassifier(n_estimator=2000,
                         learning_rate =0.05
                         )
model.fit(X_train_new1, y_train_new)
y_pred_lgbm_new1 = model.predict(X_test_new1)
print("Accuracy score:", metrics.accuracy_score(y_test_new, y_pred_lgbm_new1))


# In[359]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test_new,y_pred_lgbm_new)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# OK!! looks like even after feature scaling we are getting the same results. So we will move on with using our dataframe without scaling it. 

# In[360]:


cv_scores = cross_val_score(model, X_train_new, y_train_new, cv=10)
print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))


# Lets move on using another classification algorithm. **Decision Tree** just like Part 1

# In[361]:


model_decision = DecisionTreeClassifier(class_weight="balanced", random_state=100, max_depth=1)
model_decision.fit(X_train_new, y_train_new)


# In[362]:


y_pred_decision = model_decision.predict(X_test_new)


# In[363]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test_new,y_pred_decision)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# So the above confusion matrix shows that out of 70956 records test dataset, our model with decision tree shows (66+298) = 66262 correct predictions and (2352+2342) = 4694 incorrect predictions

# Lets check other score metrics for our model with decision tree.

# In[364]:


print(classification_report(y_test_new,y_pred_decision))


# In[365]:


print("Accuracy score with decision tree:", metrics.accuracy_score(y_test_new, y_pred_decision))
print("roc_auc score with decision tree:", roc_auc_score(y_test_new, y_pred_decision))


# OK! so the **accuracy** of the model with decision tree is 93.6%

# In[366]:


cv_scores = cross_val_score(model_decision, X_train_new, y_train_new, cv=10)
print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))


# Now lets try simplest **Logistic Regression** to your this combined dataset.

# In[367]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train_new,y_train_new)
y_pred_log=logreg.predict(X_test_new)


# In[368]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test_new,y_pred_log)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# So the above confusion matrix shows that out of 71583 records test dataset, our model with decision tree shows (68992+0) = 68992 correct predictions and (2591+0) =2591  incorrect predictions.

# OK!! so now lets see some score metrics for logistic regression

# In[369]:


print(classification_report(y_test_new,y_pred_log))


# In[370]:


print("Accuracy score with logistic regression:", metrics.accuracy_score(y_test_new, y_pred_log))
print("roc_auc score with logistic regression:", roc_auc_score(y_test_new, y_pred_log))


# So, the **accuracy** of the model with logistic regression is **96%**.

# In[371]:


cv_scores = cross_val_score(logreg, X_train_new, y_train_new, cv=10)
print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))


# Now, let even check **Random Forest** and see what prediction says with this. 

# In[372]:


from sklearn.ensemble import RandomForestClassifier
random_classifier= RandomForestClassifier(class_weight="balanced", random_state=100)
random_classifier.fit(X_train_new,y_train_new)


# In[373]:


y_pred_random= random_classifier.predict(X_test_new)


# In[374]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test_new,y_pred_random)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# So the above confusion matrix shows that out of 71583 records test dataset, our model with decision tree shows (66060+270) = 66330 correct predictions and (2321+2932) = 5253 incorrect predictions

# In[375]:


print(classification_report(y_test_new,y_pred_random))


# In[376]:


print("Accuracy score with Random Forest:", metrics.accuracy_score(y_test_new, y_pred_random))
print("roc_auc score with Random Forest:", roc_auc_score(y_test_new, y_pred_random))


# Allright!! Lets evalute all the models till now for PART2. Looks like lightgbm and logistic regression again have the same highest accuracy of 96.3% for the combined dataset as well. We will go for lightgbm since ofcourse our dataset have outliers and lightgbm algorithm is comparatively robust to outliers compared to logistic regression algorithm. 

# So BATMAN! you might want to watch out for number of thugs and number of citizens as compared to the coordiantes (x,y,z), because those two features are more focused in the model and prediction overall. 
