#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This is an interesting problem in flagging credit risk when all variables are dimensionally reduced (likely PCA as hinted in the problem). It becomes a pure data science exercise to see what can be done with the data and how to handle highly imbalanced classes. 
# 
# In this kernel we'll explore how to 
# * look at imbalanced classes
# * test performance of the permutation importance in selecting important features
# * and finally look at model explanation using shap and eli5.
# 
# # Data Load
# We'll use simple logistic regression model and compare it's performance against a tree based classifier like random forest.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler
import shap
from pdpbox import pdp, get_dataset, info_plots


# In[ ]:


raw_data = pd.read_csv('../input/creditcard.csv')
raw_data.head()


# ## Features
# **Amount** and **Time** were not scaled in the dataset. Since time is increasing, it may be possible that there are certain times when we may get fraudulent cases to spike up. So we'll find the interval between each consecutive transactions and use that as the proxy for time. In cases of 0, we'll use the mean of the difference. We'll use the **RobustScalar** from **_sklearn_** to scale the Amount column.

# In[ ]:


rob_scaler = RobustScaler()
raw_data['Time_Int'] = raw_data['Time'].diff() + raw_data['Time'].diff().mean()
raw_data['Time_Int'][0] =  raw_data['Time'].diff().mean()
raw_data['Scl_Amount'] = rob_scaler.fit_transform(raw_data['Amount'].values.reshape(-1,1))
raw_data.drop(['Time', 'Amount'], axis=1, inplace=True)
raw_data.head()


# In[ ]:


raw_data.describe().transpose()


# # EDA
# There is a very high class imbalance between fraudulent and non-fraudulent cases. We'll explore a few cases of balancing cases before building a classification model.
# 
# ## Class distribution

# In[ ]:


print(raw_data.groupby(['Class']).Class.count())
sns.set_style('dark')
plt.figure(figsize = (10,5))
sns.countplot(raw_data['Class'], 
              alpha =.60, 
              palette= ['lightgreen','red'])
plt.title('Fraud vs Non Fraud')
plt.ylabel('# Cases')
plt.show()


# Let's look at the predictor variables and see how the variables are distributed.

# In[ ]:


X = raw_data.drop('Class', axis = 1)
y = raw_data['Class']
cols = X.columns.tolist()


# ## Boxplots by class
# There are certain outliers in the data that ideally we'd like to eliminate. Removing too much of them would result in over/under fitting the model. Since quite a bit of business context is missing, we'll skip removing that for now.
# 
# 1. Higher values - more likely to be fraud
#   * V2, V4, V11, V19   
# 2. Lower values - more likely to be fraud
#   * V3, V9, V10, V12, V14, V17, V18
#   
# The above can be seen from the below box plots that might have a correlating impact on the fraudulent cases. We'll deep dive into features driving fraudulent classes in a bit.

# In[ ]:


sns.set_style('dark')
fig = plt.figure(figsize= (20,40))
fig.subplots_adjust(hspace = 0.30, wspace = 0.30)
k=0
for i in range(1,len(raw_data.columns)+1):
    ax = fig.add_subplot(11,3,i)
    sns.boxplot(x = 'Class', 
                y = X.columns[k], 
                data = raw_data, 
                palette = 'Blues')
    k = k + 1
    if k == len(X.columns): break
plt.show()


# ## Distributions of predictor variables
# Most of the variables seem to be normal/tending to be normal. Some of them have huge tails, and we can consider those to be extreme outliers. We'll look at removing those post some modeling phase to see if removing the outliers improve our model accuracy.

# In[ ]:


sns.set_style('dark')
fig = plt.figure(figsize = (20,40))
fig.subplots_adjust(hspace = 0.30, 
                    wspace = 0.30)
k=0
for i in range(1, len(X.columns) + 1):
    ax = fig.add_subplot(11, 3, i)
    sns.distplot(X[X.columns[k]], 
                 color = 'teal')
    k = k + 1
    if k == len(X.columns): break
plt.show()


# ## Correlation
# The below plot looks fishy. We're trying to understand the correlation of variables of a heavily imbalanced dataset. Ideally we should have an under sample of the dataset and look at the correlation then. Let's use the over sample method to have equal data points and then look at the correlation between the variables.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
sns.heatmap(corr, 
            mask = np.zeros_like(corr, 
                                 dtype=np.bool), 
            cmap = sns.diverging_palette(275, 
                                         150, 
                                         as_cmap=True), 
            square = True, 
            ax = ax)
plt.title('Correlation matrix of the imbalanced data')


# ### Balanced using SMOTE
# We'll use SMOTE method to generate some data points in the minority class and see if the correlation matrix improves to give some insights.

# In[ ]:


sm = SMOTE(random_state=101)
X_sm, y_sm = sm.fit_sample(X, y.ravel())
bal_data = pd.DataFrame(X_sm)
bal_data.columns = cols
bal_data['Class'] = y_sm
print(bal_data.groupby(['Class']).Class.count())
sns.set_style('dark')
plt.figure(figsize = (10,5))
sns.countplot(bal_data['Class'], 
              alpha =.60, 
              palette= ['lightgreen','red'])
plt.title('Fraud vs Non Fraud')
plt.ylabel('# Cases')
plt.show()


# In[ ]:


bal_data = bal_data.drop('Class', axis = 1)
fig, ax = plt.subplots(figsize=(10, 8))
corr = bal_data.corr()
sns.heatmap(corr, 
            mask = np.zeros_like(corr, 
                                 dtype=np.bool), 
            cmap = sns.diverging_palette(275, 
                                         150, 
                                         as_cmap=True), 
            square = True, 
            ax = ax)
plt.title('Correlation matrix of the balanced data (SMOTE)')


# ### Balanced using under sampling
# We'll under sample data points from the majority class and see if the correlation matrix improves to give some insights.

# In[ ]:


un_sam = RandomUnderSampler(random_state=101)
X_un_sam, y_un_sam = un_sam.fit_sample(X, y.ravel())
bal_data = pd.DataFrame(X_un_sam)
bal_data.columns = cols
bal_data['Class'] = y_un_sam
print(bal_data.groupby(['Class']).Class.count())
sns.set_style('dark')
plt.figure(figsize = (10,5))
sns.countplot(bal_data['Class'], 
              alpha =.60, 
              palette= ['lightgreen','red'])
plt.title('Fraud vs Non Fraud')
plt.ylabel('# Cases')
plt.show()


# In[ ]:


bal_data = bal_data.drop('Class', axis = 1)
fig, ax = plt.subplots(figsize=(10, 8))
corr = bal_data.corr()
sns.heatmap(corr, 
            mask = np.zeros_like(corr, 
                                 dtype=np.bool), 
            cmap = sns.diverging_palette(275, 
                                         150, 
                                         as_cmap=True), 
            square = True, 
            ax = ax)
plt.title('Correlation matrix of the balanced data (under sampling)')


# We see there is no difference between under sampling and SMOTE visually from a correlation perspective, but might have impact on model accuracy. 
# 
# # Feature selection
# We see a few highly correlated features from the above correlation matrix. We'll revisit removing some features at a later stage.
# 
# ## Feature importance using random forest
# We'll use a random forest model to estimate the important features. Based on the importance, we'll use a cut of to choose the top features for training our model. Here while estimating the importance of a variable, each time a feature is dropped and the loss of accuracy is used to estimate the importance of that particular feature.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = .30, 
                                                    random_state = 101)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
rf = RandomForestClassifier(n_jobs = -1, 
                            random_state = 101)
rf.fit(X_train_res, y_train_res)
feat_imp = rf.feature_importances_
sns.set(style="dark")
fig, ax = plt.subplots(figsize=(10, 8))
var_imp = pd.DataFrame({'feature':cols, 
                        'importance':feat_imp})
var_imp = var_imp.sort_values(ascending=False, 
                              by='importance')
ax = sns.barplot(x='importance', 
                 y='feature', 
                 data=var_imp)


# In[ ]:


var_imp['feature_imp_cumsum'] = var_imp['importance'].cumsum()
var_imp


# In[ ]:


top_features = SelectFromModel(rf, threshold=0.01)
top_features.fit(X_train_res, y_train_res)
rf_features= X_train.columns[(top_features.get_support())]
rf_features = rf_features.tolist()


# ## Permutation importance
# Let's look at the permutation importance method of variable importance that changes the order of a column and measures the loss of accuracy of the model to estimate the importance of the feature set. Instead of dripping a feature like in a random forest method, the feature column in randomized. Intuitively, we should see same/similar set of features from both the techniques, ordered differently, and the performance shouldn't differ drastically. We'll use [eli5](https://eli5.readthedocs.io/en/latest/overview.html) to compute importance.

# In[ ]:


perm = PermutationImportance(rf.fit(X_train_res, 
                                    y_train_res), 
                             random_state=1).fit(X_train_res,
                                                 y_train_res)
eli5.show_weights(perm, 
                  feature_names = X_train.columns.tolist(), 
                  top=(30))


# In[ ]:


pi_features = eli5.explain_weights_df(perm, feature_names = X_train.columns.tolist())
pi_features = pi_features.loc[pi_features['weight'] >= 0.01]['feature'].tolist()


# In[ ]:


print("\nFeatures from random forest", rf_features)
print("\nFeatures from permutation importance", pi_features)


# # SMOTE and classification
# ## Logistic regression - baseline (all features)

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=101)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
lreg = LogisticRegression()
lreg_model_all = lreg.fit(X_train_res, y_train_res)
y_pred_lreg_all = lreg_model_all.predict(X_test)

print('Confusion Matrix')
print('__'*10)
print(confusion_matrix(y_test, 
                       y_pred_lreg_all))
print('__'*30)
print('\nClassification Metrics')
print('__'*30)
print(classification_report(y_test, 
                            y_pred_lreg_all))
print('__'*30)
logreg_accuracy = round(accuracy_score(y_test, 
                                       y_pred_lreg_all) * 100,2)
print('Accuracy', logreg_accuracy,'%')


# In[ ]:


fpr_lreg_all, tpr_lreg_all, thresholds = roc_curve(y_test, 
                                                   y_pred_lreg_all)
roc_auc_lreg_all = auc(fpr_lreg_all,
                       tpr_lreg_all)
plt.title('Receiver Operating Characteristic - LReg (All features)')
plt.plot(fpr_lreg_all, 
         tpr_lreg_all, 
         'b',
         label='AUC = %0.3f'% roc_auc_lreg_all)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Logistic regression - RF feature selection

# In[ ]:


X_rf = X[rf_features]
X_train, X_test, y_train, y_test = train_test_split(X_rf,
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=101)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
lreg = LogisticRegression()
lreg_model_rf = lreg.fit(X_train_res, y_train_res)
y_pred_lreg_rf = lreg_model_rf.predict(X_test)

print('Confusion Matrix')
print('__'*10)
print(confusion_matrix(y_test, 
                       y_pred_lreg_rf))
print('__'*30)
print('\nClassification Metrics')
print('__'*30)
print(classification_report(y_test, 
                            y_pred_lreg_rf))
print('__'*30)
logreg_accuracy = round(accuracy_score(y_test, 
                                       y_pred_lreg_rf) * 100,2)
print('Accuracy', logreg_accuracy,'%')


# In[ ]:


fpr_lreg_rf, tpr_lreg_rf, thresholds_rf = roc_curve(y_test, 
                                                   y_pred_lreg_rf)
roc_auc_lreg_rf = auc(fpr_lreg_rf,
                       tpr_lreg_rf)
plt.title('Receiver Operating Characteristic - LReg (All features + RF Features)')
plt.plot(fpr_lreg_rf, 
         tpr_lreg_rf, 
         'g',
         label='AUC = %0.3f'% roc_auc_lreg_rf)
plt.plot(fpr_lreg_all, 
         tpr_lreg_all, 
         'b',
         label='AUC = %0.3f'% roc_auc_lreg_all)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Logistic regression - PI feature selection

# In[ ]:


X_pf = X[pi_features]
X_train, X_test, y_train, y_test = train_test_split(X_pf, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=101)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
lreg_model_pi = lreg.fit(X_train_res, y_train_res)
y_pred_lreg_pi = lreg_model_pi.predict(X_test)

print('Confusion Matrix')
print('__'*10)
print(confusion_matrix(y_test, 
                       y_pred_lreg_pi))
print('__'*30)
print('\nClassification Metrics')
print('__'*30)
print(classification_report(y_test, 
                            y_pred_lreg_pi))
print('__'*30)
logreg_accuracy = round(accuracy_score(y_test, 
                                       y_pred_lreg_pi) * 100,2)
print('Accuracy', logreg_accuracy,'%')


# In[ ]:


fpr_lreg_pi, tpr_lreg_pi, thresholds_pi = roc_curve(y_test, 
                                                   y_pred_lreg_pi)
roc_auc_lreg_pi = auc(fpr_lreg_pi,
                       tpr_lreg_pi)
plt.title('Receiver Operating Characteristic - LReg (All features + RF Features + PI features)')
plt.plot(fpr_lreg_rf, 
         tpr_lreg_rf, 
         'g',
         label='AUC = %0.3f'% roc_auc_lreg_rf)
plt.plot(fpr_lreg_all, 
         tpr_lreg_all, 
         'b',
         label='AUC = %0.3f'% roc_auc_lreg_all)
plt.plot(fpr_lreg_pi, 
         tpr_lreg_pi, 
         'y',
         label='AUC = %0.3f'% roc_auc_lreg_pi)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# The permutation importance features seem to have marginally improved the accuracy and the AUC. Let's see how it does on a tree based classifier.
# 
# ## Random forest - all features

# In[ ]:


rf_model_all = rf.fit(X_train_res, y_train_res)
y_pred_rf_all = rf_model_all.predict(X_test.as_matrix())

print('Confusion Matrix')
print('__'*10)
print(confusion_matrix(y_test, 
                       y_pred_rf_all))
print('__'*30)
print('\nClassification Metrics')
print('__'*30)
print(classification_report(y_test, 
                            y_pred_rf_all))
print('__'*30)
logreg_accuracy = round(accuracy_score(y_test, 
                                       y_pred_rf_all) * 100,2)
print('Accuracy', logreg_accuracy,'%')


# In[ ]:


fpr_rf_all, tpr_rf_all, thresholds = roc_curve(y_test, 
                                                   y_pred_rf_all)
roc_auc_rf_all = auc(fpr_rf_all,
                       tpr_rf_all)
plt.title('Receiver Operating Characteristic - RF (All features)')
plt.plot(fpr_rf_all, 
         tpr_rf_all, 
         'b',
         label='AUC = %0.3f'% roc_auc_rf_all)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Random forest - RF feature selection

# In[ ]:


X_rf = X[rf_features]
X_train, X_test, y_train, y_test = train_test_split(X_rf, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=101)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
rf_model_rf = rf.fit(X_train_res, y_train_res)
y_pred_rf_rf = rf_model_rf.predict(X_test.as_matrix())

print('Confusion Matrix')
print('__'*10)
print(confusion_matrix(y_test, 
                       y_pred_rf_rf))
print('__'*30)
print('\nClassification Metrics')
print('__'*30)
print(classification_report(y_test, 
                            y_pred_rf_rf))
print('__'*30)
logreg_accuracy = round(accuracy_score(y_test, 
                                       y_pred_rf_rf) * 100,2)
print('Accuracy', logreg_accuracy,'%')


# In[ ]:


fpr_rf_rf, tpr_rf_rf, thresholds_rf = roc_curve(y_test, 
                                                   y_pred_rf_rf)
roc_auc_rf_rf = auc(fpr_rf_rf,
                       tpr_rf_rf)
plt.title('Receiver Operating Characteristic - RF (All features + RF features)')
plt.plot(fpr_rf_rf, 
         tpr_rf_rf, 
         'g',
         label='AUC = %0.3f'% roc_auc_rf_rf)
plt.plot(fpr_rf_all, 
         tpr_rf_all, 
         'b',
         label='AUC = %0.3f'% roc_auc_rf_all)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ## Random forest - PI feature selection

# In[ ]:


X_pf = X[pi_features]
X_train, X_test, y_train, y_test = train_test_split(X_pf, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=101)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
rf_model_pi = rf.fit(X_train_res, y_train_res)
y_pred_rf_pi = rf_model_pi.predict(X_test.as_matrix())

print('Confusion Matrix')
print('__'*10)
print(confusion_matrix(y_test, 
                       y_pred_rf_pi))
print('__'*30)
print('\nClassification Metrics')
print('__'*30)
print(classification_report(y_test, 
                            y_pred_rf_pi))
print('__'*30)
logreg_accuracy = round(accuracy_score(y_test, 
                                       y_pred_rf_pi) * 100,2)
print('Accuracy', logreg_accuracy,'%')


# In[ ]:


fpr_rf_pi, tpr_rf_pi, thresholds_pi = roc_curve(y_test, 
                                                   y_pred_rf_pi)
roc_auc_rf_pi = auc(fpr_rf_pi,
                       tpr_rf_pi)
plt.title('Receiver Operating Characteristic - RF (All features + RF features + PI features)')
plt.plot(fpr_rf_rf, 
         tpr_rf_rf, 
         'g',
         label='AUC = %0.3f'% roc_auc_rf_rf)
plt.plot(fpr_rf_all, 
         tpr_rf_all, 
         'b',
         label='AUC = %0.3f'% roc_auc_rf_all)
plt.plot(fpr_rf_pi, 
         tpr_rf_pi, 
         'y',
         label='AUC = %0.3f'% roc_auc_rf_pi)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Looking at the performance of random forest feature importance and permutation importance, in both cases permutation importance had a marginal improvement in model accuracy (not so much in the random forest model). 
# 
# Using a random forest method the number of false positives decreased significantly. We'll have to evaluate the cost of model performance vs the cost of missing out on wrong predictions - the trade-off between FPR and TPR as you can see in ROC plots. In a banking use case like this where I'm interested to flag fraudulent cases, it's actually better to have more false positives.

# # Model explaining
# ## Partial dependency of features
# We've seen how randomizing the values of a column and estimating the loss to calculate the importance of features. We can also look at seeing how changing the values of a feature impacts the predictions.
# 
# ### V4 feature
# If the feature V4 has higher values, it has a higher change of being a case of fraud. You can see an increase in the y-axis as compared to a flat baseline in red.

# In[ ]:


pdp_V4 = pdp.pdp_isolate(model = rf_model_pi,
                         dataset = X_test, 
                         model_features = pi_features, 
                         feature='V4', 
                         num_grid_points = 20)

pdp.pdp_plot(pdp_V4, 
             'V4', 
             plot_pts_dist=True)
plt.show()


# ### V14 feature
# If you look back at the correlation matrix, V14 was negatively correlated to V4 feature. So, if we did the above analysis for the V14 feature, we should see a reverse of the above plot. This also says that having more negative values of this feature V14, has a higher change of predicting fraudulent cases. A really high number shows a tendency to bring the value close to the baseline.

# In[ ]:


pdp_V14 = pdp.pdp_isolate(model = rf_model_pi,
                         dataset = X_test, 
                         model_features = pi_features, 
                         feature='V14', 
                         num_grid_points = 20)

pdp.pdp_plot(pdp_V14, 
             'V14', 
             plot_pts_dist=True)
plt.show()


# ## SHAP
# ### Explaining predictions
# Let's take a look at rows 443 (fraudulent) and 442 (non-fraudulent) from our test data and see what is the predicted probability for each of the case. The SHAP plot shows how away is the predicted probability from the base value. The pink arrows indicate features that show increased predictions for the fraudulent class and the blue ones from the non-fraudulent class.
# 
# You can also see the baseline value is the difference between the pink and blue bands - variables driving the prediction in their respective direction.

# In[ ]:


explainer = shap.TreeExplainer(rf_model_pi)


# In[ ]:


data_row = 443
data_row = X_test.iloc[data_row]
print("Predicted probability\nNon-fraud - Fraud\n", rf_model_pi.predict_proba(data_row.values.reshape(1, -1)))
shap_values = explainer.shap_values(data_row)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_row)


# One can interpret the features values of V14, V10, and V12 had a higher impact in classifying this record as fraudulent. This becomes very important when financial institutes want to know the reason behind fraudulent instances and explain the decisions to stakeholders.
# 
# In the below, the same three variables have the same impact in classifying the record as non-fraudulent (not in the same order and magnitude). Comparing the values of the features from the case above, all of them have a higher value when non-fraudulent. This reinforces the assumptions we saw in the box plots by class.

# In[ ]:


data_row = 442
data_row = X_test.iloc[data_row]
print("Predicted probability\nNon-fraud - Fraud\n", rf_model_pi.predict_proba(data_row.values.reshape(1, -1)))
shap_values = explainer.shap_values(data_row)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_row)


# ### SHAP summary plot
# The summary plot tells us the importance of features but with additional information like the spread of each feature, the impact of a variable, and the value of the respective feature.
# 
# Look at the plot below and notice the points on the extremes - pink dots from V14, V16 have an impact of shifting the prediction direction for those records by 40% and having higher values in magnitude.

# In[ ]:


shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)


# ### SHAP dependence plot
# The SHAP dependence plot shows the interaction of a feature against another, and how it's impact on model prediction varies with another. It's different from the above plot and showcases a few more extra details - 
# * We observe the obvious that a higher value of the feature V16 it could be fraudulent
# * The downward slope indicates the same
# * The plot below also sheds some light that there are interactions among the features that drive the prediction. Since the spread is pretty broad, the same value of the feature V16 can have different impact on predicting fraud
# * If you observe the outlier on the right, you can see a higher value of V4 feature and a higher value of V16 existed together for a record
# * The correlation between the two features also can be seen by the abundance of the pink records

# In[ ]:


shap.dependence_plot('V16', shap_values[1], X_test, interaction_index = 'V4')

