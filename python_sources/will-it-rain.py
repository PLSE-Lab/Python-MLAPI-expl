#!/usr/bin/env python
# coding: utf-8

# # **Will It Rain Tomorrow?**
# 
# We will predict if it will rain tomorrow in Australia. This is a classification problem as we are predicting whether it will Rain or Not Rain.
# ### **Table of Contents**
# 
# 1. Data Preprocessing and Exploratory Data Analysis
# 2. Random Forest
# 3. Logistic Regression
# 4. Comparison Between Random Forest and Logistic Regression
# 5. Conclusion
# 
# ### **1. Data Preprocessing and Exploratory Data Analysis**
# #### **Import Libraries and Dataset**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load data
rain_data = pd.read_csv('../input/weatherAUS.csv')
rain_data.head()


# #### **Missing Data:**

# In[ ]:


# Visualising missing data:
sns.heatmap(rain_data.isnull(),yticklabels=False,cbar=False,cmap='Reds_r')


# In[ ]:


# High percentage of missing data for Evaporation, Sunshine, Cloud9am and Cloud3pm features.
# Date, Location and RISK_MM will be removed.
# Lastly, remove any observations/rows with missing data
rain_data.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm','RISK_MM','Date','Location'],axis=1,inplace=True)
rain_data.dropna(inplace=True)
rain_data[['RainTomorrow','RainToday']] = rain_data[['RainTomorrow','RainToday']].replace({'No':0,'Yes':1})


# #### **Rain vs. No Rain**

# In[ ]:


# Frequency of Rainy and No Rain:
mpl.style.use('ggplot')
plt.figure(figsize=(6,4))
plt.hist(rain_data['RainTomorrow'],bins=2,rwidth=0.8)
plt.xticks([0.25,0.75],['No Rain','Rain'])
plt.title('Frequency of No Rain and Rainy days\n')
print(rain_data['RainTomorrow'].value_counts())


# #### **Histogram of the numerical features:**

# In[ ]:


# Segregating our numerical features from the categorical
rain_data_num = rain_data[['MinTemp','MaxTemp','Rainfall','WindSpeed9am','WindSpeed3pm',
                           'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm',
                           'Temp9am','Temp3pm','RainToday','RainTomorrow']]

# Histogram of each numerical feature
mpl.rcParams['patch.force_edgecolor'] = True
ax_list = rain_data_num.drop(['RainTomorrow'],axis=1).hist(figsize=(20,15),bins=20)
ax_list[2,1].set_xlim((0,100))


# #### **Correlation matrix between the numerical features:**

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(rain_data_num.corr(),annot=True,cmap='bone',linewidths=0.25)


# #### **Categorical Features and Dummy Variables**

# In[ ]:


# Creating dummy variables for the categorical features:
WindGustDir_data = pd.get_dummies(rain_data['WindGustDir'])
WindDir9am_data = pd.get_dummies(rain_data['WindDir9am'])
WindDir3pm_data = pd.get_dummies(rain_data['WindDir3pm'])

# Dataframe of the categorical features
rain_data_cat = pd.concat([WindGustDir_data,WindDir9am_data,WindDir3pm_data],
                          axis=1,keys=['WindGustDir','WindDir9am','WindDir3pm'])

# Combining the Numerical and Categorical/Dummy Variables
rain_data = pd.concat([rain_data_num,rain_data_cat],axis=1)


# #### **Splitting Data by Train and Test Data**

# In[ ]:


from sklearn.model_selection import train_test_split

X = rain_data.drop(['RainTomorrow'],axis=1)
y = rain_data['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=88)


# ## **2. Random  Forest**
# #### **Instantiate the Random Forest classifier, fit then predict**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# Out of Bag (oob) set to True. We will compare the oob_score with accuracy to see if they differ by much
# n_estimators, or number of decision trees set to 100
rf = RandomForestClassifier(n_estimators=100,oob_score=True,random_state=88)
rf.fit(X_train,y_train)
y_rf_pred = rf.predict(X_test)


# #### **Null Accuracy**

# In[ ]:


# No Rain and Rain frequency in test set
print(y_test.value_counts())
null_accuracy = float(y_test.value_counts().head(1) / len(y_test))
print('Null Accuracy Score: {:.2%}'.format(null_accuracy))


# If we built a model that guesses 'No Rain' every time, then we would obtain the 'Null Accuracy' or 'Baseline Accuracy' of 77.6%. 

# #### **Random Forest result:**

# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
print('Accuracy Score: {:.2%}'.format(accuracy_score(y_test,y_rf_pred),'\n'))
print('Out of Bag Accuracy Score: {:.2%}'.format(rf.oob_score_),'\n')
print('Confusion Matrix:\n',confusion_matrix(y_test,y_rf_pred))


# Accuracy Rate is 84.98%, however the Baseline Accuracy is 77.6%. Still some improvement though. The False Positives (3,933) are greater than the True Positives (3,653). Let's try and make our model simplier through feature selection as there are 60 features. Then we will try and improve the True Positive Rate.

# #### **Which Features Add Value?**
# Feature selection (reducing the number of variables) will make our model simplier and reduce the chances of overfitting. We don't want to use variables that don't any or much value when it comes training our model to predict the out-of-sample data. 

# In[ ]:


# Using feature_importance_ for feature selection
feature_importance_rf = pd.DataFrame(rf.feature_importances_,index=X_train.columns,columns=['Importance']).sort_values(['Importance'],ascending=False)
feature_importance_rf.head(5)


# In[ ]:


# Plot feature_importance
feature_importance_rf.plot(kind='bar',legend=False,figsize=(15,8))


# Remember our Categorical Variables: WindGustDir,WindDir9am,WindDir3pm, were transformed into dummy variables to make the Random Forest algorithm perform better. Because they were transformed into dummy variables, then they would appear to rank low under the important features as plotted above.
# 
# To see whether the Categorical Variables: WindGustDir,WindDir9am,WindDir3pm, add much value to our model, we will compute the accuracy rate for the Random Forest using only the top 5 features, and compare it with the accuracy rate for the Random Forest using the top 5 features as well as the Categorical Variables.

# #### **Train Random Forest with the subset of features. Then, predict.**
# #### **Random Forest Top 5 Features Result:**

# In[ ]:


# Our Top 5 Features
features_top_5 = list(feature_importance_rf.index[0:6])

# X dataframe - with only the top 5 features
subset_1 = [X.columns.get_loc(x) for x in features_top_5]

# Split, Train, Predict
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,subset_1], y, test_size=0.30, random_state=88)
rf.fit(X_train,y_train)
y_rf_pred = rf.predict(X_test)

print('Accuracy Score: {:.2%}'.format(accuracy_score(y_test,y_rf_pred)))
print('Out of Bag Score {:.2%}:'.format(rf.oob_score_),'\n')
print('Confusion Matrix:\n',confusion_matrix(y_test,y_rf_pred))


# #### **Random Forest Top 5 Features with Categorical (Dummy) Variables Result:**

# In[ ]:


# X dataframe - with top 5 features and the categorical variables
subset_2 = subset_1 + list(range(12,len(X.columns)))

# Split, Train, Predict
X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,subset_2], y, test_size=0.30, random_state=88)
rf.fit(X_train,y_train)
y_rf_pred = rf.predict(X_test)

print('Accuracy Score: {:.2%}'.format(accuracy_score(y_test,y_rf_pred)))
print('Out of Bag Score {:.2%}:'.format(rf.oob_score_),'\n')
print('Confusion Matrix:\n',confusion_matrix(y_test,y_rf_pred))


# From here we can conclude that the categorical features do not add significant amount of value to the accuracy rate. Therefore we can remove the categorical features.

# #### **How many features should we use?**
# There's no one right answer to this. To help us choose what number of features to use, we will visualise the relationship between number of features used vs. accuracy rate.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Up to what number of features to plot\nindex = np.array(list(range(2,9)) + [15, 30, 60])\n\n# creating list of index location\nfeatures = list(feature_importance_rf.index)\nfeatures = [X.columns.get_loc(x) for x in features]\n\n# instantiate classifier\nrf = RandomForestClassifier(n_estimators=100,random_state=88)\n\naccuracy_rate = []\n\n# append the accuracy rate\nfor i in index:\n    X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,features[0:i]], y, test_size=0.30, random_state=88)\n    rf.fit(X_train,y_train)\n    y_rf_pred = rf.predict(X_test)    \n    accuracy_rate.append(accuracy_score(y_test,y_rf_pred))')


# In[ ]:


# Plot accuracy vs. number of features
plt.figure(figsize=(7,5))
plt.scatter(x=index-1,y=accuracy_rate)
plt.ylabel('Accuracy Rate',fontsize=12)
plt.xlabel('Number of Features',fontsize=12)
plt.xlim(-0.2,60)
plt.title('Random Forest \nAccuracy Rate vs. Number of Features', fontsize = 14)


# As you can see, there is not much improvement to the accuracy rate when the number of features are at 5 or more. To keep our model simple, we will use 7 features instead of 59. The accuracy rate from 7 to 59 features differs by only 0.68%. 

# In[ ]:


# Split, Train, Predict on The 7 Features
X_train, X_test, y_train, y_test = train_test_split(X[feature_importance_rf.head(7).index], y, test_size=0.30, random_state=88)
rf.fit(X_train,y_train)
y_rf_pred = rf.predict(X_test)
cm = pd.DataFrame(confusion_matrix(y_test,y_rf_pred), index=['NO RAIN','RAIN'],columns=['NO RAIN','RAIN'])


# #### **Confusion Matrix with simplier Random Forest model (Top 7 Features)**

# In[ ]:


print('Accuracy Score (Top 7 Features): {:.2%}'.format(accuracy_score(y_test,y_rf_pred)),'\n')

# plot confusion matrix
fig = plt.figure(figsize=(8,6))
ax = sns.heatmap(cm,annot=True,cbar=False, cmap='CMRmap_r',linewidths=0.5,fmt='.0f')
ax.set_title('Random Forest Confusion Matrix',fontsize=16,y=1.25)
ax.set_ylabel('ACTUAL',fontsize=14)
ax.set_xlabel('PREDICTED',fontsize=14)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=12)


# In[ ]:


TP = cm.iloc[1,1] # True Positive - Predicted Rain Correctly
TN = cm.iloc[0,0] # True Negative - Predicted No Rain Incorrectly
FP = cm.iloc[0,1] # False Positive - Predicted Rain when it didn't rain
FN = cm.iloc[1,0] # False Negative - Predicted No Rain when it did rain


# #### **Sensitivty vs. Specificity**
# **Sensitivity:** When it rains, how often are our predictions correct?
# **Specificity:** When it does not rain, how often are our predictions correct? 

# In[ ]:


print('Sensitivity: {:.2%}'.format(TP/(FN+TP)))
print('Specificity: {:.2%}'.format(TN/(FP+TN)))


# Of the time it rained, we have correctly predicted Rain 47.76% of the time. Of the time it did not rain, we have correctly predicted No Rain 94.78% of the time. Considering it doesn't rain 77.6% of the time, you would expect the specificity rate to be higher.

# #### **Probability Score of Rain and No Rain from the Random Forest**
# The Random Forest algorithm produces a probability score (the proportion of votes of the trees in the ensemble) in order to make classification. Let's visualise the probability scores for Rain vs. No Rain.

# In[ ]:


# proves np.array of the probability scores
y_prob_rain = rf.predict_proba(X_test)

# To convert x-axis to a percentage
from matplotlib.ticker import PercentFormatter

# Plot histogram of predicted probabilities
fig,ax = plt.subplots(figsize=(10,6))
plt.hist(y_prob_rain[:,1],bins=50,alpha=0.5,color='teal',label='Rain')
plt.hist(y_prob_rain[:,0],bins=50,alpha=0.5,color='orange',label='No Rain')
plt.xlim(0,1)
plt.title('Histogram of Predicted Probabilities')
plt.xlabel('Predicted Probability (%)')
plt.ylabel('Frequency')

ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.text(0.025,0.83,'n = 33,878',transform=ax.transAxes)

plt.legend()


# The Random Forest classication model uses 50% as the threshold for classification. Meaning that if probability of Rain is > 50%, it would predict Rain or assign a value of 1. Else if, the probability of Rain is < 50%, it would predict No Rain or assign a value of 0. 
# 
# Because of the binary relationship between Rain and No Rain, the histogram has the highest frequency at 0% for Rain and 100% for No Rain.
# 
# But what if, the cost of a False Negative is greater than a False Positive? That is, the cost of it actually raining and predicting that it won't rain, is greater than the cost of when it does not rain when we have predicted that it will rain.
# 
# The 50% threshold can be reduced in order to increase the sensitivity rate. However this will reduce the specificity rate because there is an inverse relationship between sensitivity and specificity.

# #### **ROC Curve (True Positive vs. False Positive Rate) and Threshold Curve for the Random Forest Model**

# In[ ]:


#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test,y_prob_rain[:,1])

fig,ax1 = plt.subplots(figsize=(9,6))
ax1.plot(fpr, tpr,color='orange')
ax1.legend(['ROC Curve'],loc=1)
ax1.set_xlim([-0.005, 1.0])
ax1.set_ylim([0,1])
ax1.set_ylabel('True Positive Rate (Sensitivity)')
ax1.set_xlabel('False Positive Rate \n(1 - Specificity)\n FP / (TN + FP)')
ax1.set_title('ROC Curve for RainTomorrow Random Forest Classifier\n')

plt.plot([0,1],[0,1],linestyle='--',color='teal')
plt.plot([0,1],[0.5,0.5],linestyle='--',color='red',linewidth=0.25)

#Threshold Curve
ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='black')
ax2.legend(['Threshold'],loc=4)
ax2.set_ylabel('Threshold',color='black')
ax2.set_ylim([0,1])
ax2.grid(False)


# The default threshold is 50%, which had resulted in a low sensitivity rate of 47.76% and specificity of 94.78%. Now will change the threshold so that it provides a higher sensitivity rate at the cost of a lower specificity rate.

# #### **Selecting The Threshold Rate**

# In[ ]:


# Function to calc sensitivity and specificity rate for a given threshold
def evaluate_threshold(threshold):
    print('Sensitivity: {:.2%}'.format(tpr[thresholds > threshold][-1]))
    print('Specificity: {:.2%}'.format(1 - fpr[thresholds > threshold][-1]))
    
evaluate_threshold(0.25)


# #### **We will reduce the threshold from 50% to 25%.**
# 
# If the probability of Rain is > 25%, the model will predict Rain (for tomorrow). If the probability of Rain is < 25%, the model will predict No Rain (for tomorrow).

# #### **Confusion Matrix with simplier Random Forest model (top 7 Features) and threshold rate set to 25%**

# In[ ]:


from sklearn.preprocessing import binarize
# change the predicted class with 25% threshold
y_pred_class = binarize(y_prob_rain,0.25)[:,1]

cm = pd.DataFrame(confusion_matrix(y_test,y_pred_class), index=['NO RAIN','RAIN'],columns=['NO RAIN','RAIN'])

print('Accuracy Score (Top 7 Features with 25% Threshold): {:.2%}'.format(accuracy_score(y_test,y_pred_class)),'\n')

# Plot Confusion Matrix
fig = plt.figure(figsize=(8,6))
ax = sns.heatmap(cm,annot=True,cbar=False, cmap='CMRmap_r',linewidths=0.5,fmt='.0f')
ax.set_title('Random Forest Confusion Matrix',fontsize=16,y=1.25)
ax.set_ylabel('ACTUAL',fontsize=14)
ax.set_xlabel('PREDICTED',fontsize=14)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=12)


# In[ ]:


TP = cm.iloc[1,1] # True Positive - Predicted Rain Correctly
TN = cm.iloc[0,0] # True Negative - Predicted No Rain Incorrectly
FP = cm.iloc[0,1] # False Positive - Predicted Rain when it didn't rain
FN = cm.iloc[1,0] # False Negative - Predicted No Rain when it did rain

sens_rf = TP/(FN+TP)
spec_rf = TN/(FP+TN)

print('Sensitivity: {:.2%}'.format(sens_rf))
print('Specificity: {:.2%}'.format(spec_rf))


# Of the time it rained, we have correctly predicted Rain 74.26% of the time. Of the time it did not rain, we have correctly predicted No Rain 81.16% of the time.

# #### **AUC for Random Forest**
# The AUC is the Area Under the ROC Curve. If the model produces a high sensitivity and specificity rate (which is what you would want to achieve), then the ROC curve will be stretched towards the top left of the x-y axis.
# The AUC provides in indication on how well the model had performed in comparison to another model.

# In[ ]:


rf_auc = roc_auc_score(y_test,y_prob_rain[:,1])
print('AUC Score: {:.2%}'.format(rf_auc))


# ### **3. Logistic Regression**
# #### **Train and Predict using the entire number of features**

# In[ ]:


# Libraries for the Logistic Regression 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = rain_data.drop(['RainTomorrow'],axis=1)
y = rain_data['RainTomorrow']
# Remove the Categorical (Dummy) Variables, as we have identified earlier that they do not add much value
X = X.iloc[:,0:12] 


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=88)

# Logistic Regression train
lr = LogisticRegression(random_state=88, solver='liblinear')
lr.fit(X_train,y_train)

# predict
y_lr_pred = lr.predict(X_test)


# #### **Logistic Regression result using all the features:**

# In[ ]:


# The 10-Fold Cross Validation method is used to calculate the accuracy score of the Logistic Regression model.
print('Accuracy Score with 10-KFolds: {:.2%}'.format(cross_val_score(lr,X,y,cv=10,scoring='accuracy').mean()),'\n')
print('Confusion Matrix:\n',confusion_matrix(y_test,y_lr_pred))


# #### **Feature Selection - using Recursive Feature Elimination (RFE)**
# 
# *"Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached."*
# 
# The Final Random Forest model used 7 features. We will have the same number of features for the Logistic Regression Model. 

# In[ ]:


get_ipython().run_cell_magic('time', '', '#Feature Selection Method: Recursive Feature Elimination \nrfe = RFE(estimator=lr, n_features_to_select=7)\nrfe = rfe.fit(X_train,y_train)\n\nprint("Number of Features: {}".format(rfe.n_features_)) \nprint("Selected Features: {}".format(rfe.support_))\nprint("Feature Ranking: {}".format(rfe.ranking_))')


# #### **The Selected 7 Features:**

# In[ ]:


pd.DataFrame(X.iloc[:,rfe.support_].columns,columns=['Importance'])


# #### **Train and Predict using the 7 chosen features**

# In[ ]:


X_rfe = X.iloc[:,rfe.support_]
# Train Test split with subset of X features
X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.30, random_state=88)
# Train and Predict
lr.fit(X_train,y_train)
y_lr_pred = lr.predict(X_test)


# #### **Logistic Regression result using the selected 7 features:**

# In[ ]:


#accuracy rate using 10-Fold CV
accuracy_kfold = cross_val_score(lr,X_rfe,y,cv=10,scoring='accuracy').mean()
print('Accuracy Score with 7 Features and 10-KFolds: {:.2%}'.format(accuracy_kfold),'\n')
print('Confusion Matrix:\n',confusion_matrix(y_test,y_lr_pred))


# #### **The coefficients and features in the Logistic Regression**

# In[ ]:


pd.concat([pd.DataFrame(lr.coef_,index=['coefficient'],columns=X_train.columns).T, 
                         X_train.aggregate([np.mean,np.std,np.min,np.max]).T],axis=1)


# #### **Confusion Matrix with simplier Logistic Regression model (7 Features)**

# In[ ]:


cm = pd.DataFrame(confusion_matrix(y_test,y_lr_pred), index=['NO RAIN','RAIN'],columns=['NO RAIN','RAIN'])

print('Accuracy Score with 7 Features and 10-KFolds: {:.2%}'.format(accuracy_kfold),'\n')

# Plot CM
fig = plt.figure(figsize=(8,6))
ax = sns.heatmap(cm,annot=True,cbar=False, cmap='CMRmap_r',linewidths=0.5,fmt='.0f')
ax.set_title('Logistic Regression Confusion Matrix',fontsize=16,y=1.25)
ax.set_ylabel('ACTUAL',fontsize=14)
ax.set_xlabel('PREDICTED',fontsize=14)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=12)


# In[ ]:


TP = cm.iloc[1,1] # True Positive - Predicted Rain Correctly
TN = cm.iloc[0,0] # True Negative - Predicted No Rain Incorrectly
FP = cm.iloc[0,1] # False Positive - Predicted Rain when it didn't rain
FN = cm.iloc[1,0] # False Negative - Predicted No Rain when it did rain

print('Sensitivity: {:.2%}'.format(TP/(FN+TP)))
print('Specificity: {:.2%}'.format(TN/(FP+TN)))


# Of the time it rained, we have correctly predicted Rain 42.35% of the time. Of the time it did not rain, we have correctly predicted No Rain 95.20% of the time. Like what we did before, assuming the cost of False Positive is greater than False Negative, we will adjust the threshold level to 25%.

# #### **ROC Curve and Threshold Curve for the Logistic Regression Model**

# In[ ]:


# Probability of Rain for X_test
y_prob_rain = lr.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test,y_prob_rain[:,1])

#ROC Curve
fig,ax1 = plt.subplots(figsize=(9,6))
ax1.plot(fpr, tpr,color='orange')
ax1.legend(['ROC Curve'],loc=1)
ax1.set_xlim([-0.005, 1.0])
ax1.set_ylim([0,1])
ax1.set_ylabel('True Positive Rate (Sensitivity)')
ax1.set_xlabel('False Positive Rate \n(1 - Specificity)\n FP / (TN + FP)')
ax1.set_title('ROC Curve for RainTomorrow Logistic Regression Classifier\n')

plt.plot([0,1],[0,1],linestyle='--',color='teal')
plt.plot([0,1],[0.5,0.5],linestyle='--',color='red',linewidth=0.25)

#Threshold Curve
ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='black')
ax2.legend(['Threshold'],loc=4)
ax2.set_ylabel('Threshold',color='black')
ax2.set_ylim([0,1])
ax2.grid(False)


# #### **Confusion Matrix with simplier Logistic Regression model (7 Features) and 25% threshold**

# In[ ]:


# Changing predictions using threshold of 25%
y_pred_class = binarize(y_prob_rain,0.25)[:,1]

cm = pd.DataFrame(confusion_matrix(y_test,y_pred_class), index=['NO RAIN','RAIN'],columns=['NO RAIN','RAIN'])

print('Accuracy Score (Top 7 Features with 25% Threshold): {:.2%}'.format(accuracy_score(y_test,y_pred_class)),'\n')

fig = plt.figure(figsize=(8,6))
ax = sns.heatmap(cm,annot=True,cbar=False, cmap='CMRmap_r',linewidths=0.5,fmt='.0f')
ax.set_title('Logistic Regression Confusion Matrix',fontsize=16,y=1.25)
ax.set_ylabel('ACTUAL',fontsize=14)
ax.set_xlabel('PREDICTED',fontsize=14)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=12)


# In[ ]:


TP = cm.iloc[1,1] # True Positive - Predicted Rain Correctly
TN = cm.iloc[0,0] # True Negative - Predicted No Rain Incorrectly
FP = cm.iloc[0,1] # False Positive - Predicted Rain when it didn't rain
FN = cm.iloc[1,0] # False Negative - Predicted No Rain when it did rain

sens_lr = TP/(FN+TP)
spec_lr = TN/(FP+TN)

print('Sensitivity: {:.2%}'.format(sens_lr))
print('Specificity: {:.2%}'.format(spec_lr))


# Of the time it rained, we have correctly predicted Rain 70.05% of the time. Of the time it did not rain, we have correctly predicted No Rain 79.53% of the time.

# ## 4. **Comparison Between Random Forest and Logistic Regression Model**
# We will compare the results. The Area Under the ROC Curve is an indicator on which classifier model has the stronger performance.

# In[ ]:


lr_auc = cross_val_score(lr,X,y,cv=10,scoring='roc_auc').mean()

print('Null Accuracy Score: {:.2%}\n'.format(null_accuracy))
print('{:>30} {:>26}'.format('Random Forest','Logistic Regression'))
print('{} {:>17.2%} {:>22.2%}'.format('AUC Score',rf_auc,lr_auc))
print('{} {:>14.2%} {:>22.2%}'.format('Sensitivity*',sens_rf,sens_lr))
print('{} {:>14.2%} {:>22.2%}'.format('Specificity*',spec_rf,spec_lr))
print('\n*25% Threshold')


# ## **5. Conclusion**
# #### **The Random Forest model is the better performer as the AUC is 85.83% vs. 83.41% for the Logistic Regression model.**
# 
# From comparing accuracy rates, the Categorical Features: WindGustDir,WindDir9am,WindDir3pm, offered little value. We saw the increase in the accuracy rates from having 1 feature to 59 and chose 7 features as that was the approximate point at which the increase in accuracy rate was very small. Having less features would simplify the model, reduce chances of overfitting, and provide better interpretability.
# 
# Both the Random Forest and Logistic Regression had a low sensitivity rate which often incorrectly predicted that it won't rain the next day when it actually did rain. 
# 
# The ROC and Threshold Curve demonstrates the relationshp between sensitivity and specificity at each threshold. Assuming that the cost of a False Positive was greater than a False Negative, we have chosen to reduce the threshold from 50% to 25%. This resulted in the sensitivity rate for the Random Forest model increasing from 47.76% to 74.26% at the trade-off of decreasing the specificity rate from 94.78% to 81.16%. 
# 
# The features used in the Random Forest and Logistic Regression differ. Features used in the final model were:
# 
#     Random Forest                 Logistic Regression
#     1. Humidity3pm                1. Humidity3pm
#     2. Pressure3pm                2. Pressure3pm
#     3. Humidity9am                3. RainToday
#     4. Pressure9am                4. Pressure9am
#     5. Temp3pm                    5. Temp3pm
#     6. Rainfall                   6. MaxTemp
#     7. MinTemp                    7. MinTemp
