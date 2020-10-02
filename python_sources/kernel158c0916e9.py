#!/usr/bin/env python
# coding: utf-8

# # Device failure Data Set Description

# <h1> Table of Contents </h1>
# 
# 1. [Dataset Description](#columns)
# 2. [Importing Packages and have a quick glance at the data](#packages)
# 3. [Exploratory Data Analysis](#EDA)
# 4. [Normalization](#Scaling)
# 5. [FeatureEngineering](#feature) 
# 6. [Train/Test split Before Sampling](#traintest)
# 7. [Sampling the data using SMOTE](#SMOTE)
#      - 7.1 [Training a Random Forest](#SMOTE_RF)
#      - 7.2 [Logistic Regression ](#SMOTE_log )
# 8. [Up-Down sample using Resampling](#Resample)
#       - 8.1 [Training a Random Forest](#Resample_RF)
#       - 8.2 [Logistic Regression](#Resample_log)
#       - 8.3 [Adaboost](#adb)
#       - 8.4 [MLPClassifier(NN)](#MLP)
#       - 8.5 [Stacking](#stacking)
# 
#      
# 
# 

# ## 1. Dataset Description <a id='columns'>
# 
# 
# This data set contains around 125000 records containing device information with attribute combination and Whether there was a failure or not
# 
# 
# Below are the columns information
# 
# 
# -**date**
# 
# -**device id**
# 
# -**attribute1 to attribute9**
# 
# -**Label 'failure' indicating  if the device is failed (1) or not (0)**

# ## 2. Importing Packages and have a quick glance at the data <a id='packages'>

# In[118]:


#Importing the neccessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[119]:


# Load the input file
my_local_path='../input/'
device_data=pd.read_csv(my_local_path+'device_failure.csv')


# In[120]:


device_data.info()


# *Observation* :So we dont have any null values in the given data set

# ## 3. Exploratory Data Analysis <a id='EDA'>

# In[121]:


device_data.head()


# Scale is different for the attributes and there is a huge difference in the ranges. It needs a normalization in this case

# Let us see the basic statistics of the distribution

# In[122]:


device_data.describe()


# We see the min/max values are too far and the standard deviation is also more for almost all the attributes

# Now, let us check the distribution of the output ,label

# In[123]:


device_data['failure'].value_counts()


# *Observation*: Here we see a huge class imbalance,  disproportionate ratio of observations . Out of the 124494 records in the data set, only 106 are failure cases. This needs to be handled

# # 4. Normalization <a id='Scaling'>

# Before we handle the data imbalance, lets first normalise the data set and prepare the train and test data. We will use the MinMaxScaler in this case given the outliers in the data
# 

# In[124]:


from sklearn.preprocessing import MinMaxScaler


# In[125]:


scale=MinMaxScaler().fit(device_data[['attribute1','attribute2','attribute3','attribute4','attribute5','attribute6',
                                            'attribute7','attribute8','attribute9']])
device_data_scaled=scale.fit_transform(device_data[['attribute1','attribute2','attribute3','attribute4','attribute5','attribute6',
                                            'attribute7','attribute8','attribute9']])


# In[126]:


device_df_scaled=pd.DataFrame(device_data_scaled,columns=['attribute1','attribute2','attribute3','attribute4','attribute5','attribute6',
                                            'attribute7','attribute8','attribute9'])


# In[127]:


device_df_scaled.head()


# In[128]:


device_df_scaled['failure']=device_data['failure']


# In[129]:


device_df_scaled.head()


# Lets see the if there is any colinearity among the attributes

# In[130]:


corr=device_df_scaled.corr()


# In[131]:


corr


# In[132]:


sns.heatmap(corr,annot=True,fmt=".1f")


# *Obervation* Attribute 7 & 8 are strongly correlated. We can drop the attribute 8.
#              Attribute 9 is 50% correlated with attribute3

# In[133]:


device_df_scaled.drop('attribute8',axis=1,inplace=True)


# In[134]:


device_df_scaled.head()


# We will observe the relation of output with these attributes using the box plots

# In[135]:


sns.boxplot(x='failure',y='attribute1',data=device_df_scaled)


# In[136]:


sns.boxplot(x='failure',y='attribute2',data=device_df_scaled)


# In[137]:


sns.boxplot(x='failure',y='attribute3',data=device_df_scaled)


# In[138]:


sns.boxplot(x='failure',y='attribute5',data=device_df_scaled)


# In[139]:


sns.boxplot(x='failure',y='attribute6',data=device_df_scaled)


# In[140]:


sns.boxplot(x='failure',y='attribute7',data=device_df_scaled)


# In[141]:


sns.boxplot(x='failure',y='attribute9',data=device_df_scaled)


# *Observation* : attribute 9,7,5,3,2 have more outliers .No precise segregation of classes for the given set of attributes.
# 
# Considering the use case as the critical one , we will not be removing the outliers

# # 5. Feature Engineering  <a id='feature'>

# In[142]:


from datetime import datetime
device_df_scaled['month']=pd.to_datetime(device_data['date']).dt.month


# In[143]:


month_dummies=pd.get_dummies(device_df_scaled.month,prefix='month',drop_first=False)
device_df_scaled=pd.concat([device_df_scaled,month_dummies],axis=1)


# In[144]:


device_df_scaled.pivot_table(index='month',columns='failure',aggfunc='size')


# **Note**:
# 
# Please note that the feature engineering has been done,howwever after executing the models, it didnt have any added value and hence has been removed. So the below models will not use this feature

# # 6. Train/Test Split Before Resampling <a id='traintest'>
# 

# Lets split the sample to train and test before we do this. This will segregate the train data and then we can apply the sampling techniques with the train data

# In[145]:


features=['attribute1','attribute2','attribute3','attribute4','attribute5','attribute6','attribute7','attribute9']


# In[146]:


#Spliting the features and labels
X=device_df_scaled[features]
Y=device_df_scaled['failure']


# In[147]:


X.head()


# In[148]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=333)


# In[149]:


x_train.shape


# In[150]:


x_test.shape


# In[151]:


np.bincount(y_train)


# There are multiple ways of handling this imbalance data. 
# 
# In this case we will try generate Synthetic Samples using the SMOTE algorithm, since we have a huge data imbalance as we can see above.(87070 / 75)
# 
# # 7. Sampling the data using SMOTE <a id='SMOTE'>
# 

# In[152]:


from imblearn.over_sampling import SMOTE


# In[153]:


sm=SMOTE(random_state=555)
x_res,y_res=sm.fit_resample(x_train,y_train)


# In[154]:


print("resample data set class distrbibution :", np.bincount(y_res))


# In[155]:


x_res.shape


# Now, we have an equal distribution of the date class. 

# ## 7.1  Random Forest <a id='SMOTE_RF'>
# We will use the RandromForest algorithm since this is best suited for class imbalance/non linear data.

# In[156]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(criterion='entropy',max_depth =4, n_estimators = 150,max_leaf_nodes=10, random_state = 1,min_samples_leaf=5,
                            min_samples_split=10)


# In[157]:


# Train the model with the resampled data 
my_forest=model.fit(x_res,y_res)


# In[158]:


# Training accuracy
my_forest.score(x_res,y_res)


# In[159]:


y_train_pred=my_forest.predict(x_res)


# In[160]:


y_pred=my_forest.predict(x_test)


# In[161]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score


# In[162]:


# Test accuracy
accuracy_score(y_pred,y_test)


# In[163]:


# Testing Confusion Matrix
print(confusion_matrix(y_test,y_pred))


# In[164]:


#Training CF
print(confusion_matrix(y_res,y_train_pred))


# In[165]:


cr=metrics.classification_report(y_test,y_pred)
print(cr)


# *Observation*: Above Metrics shows that the TruePositive(TP/Sensitivity/Recall) is not so impressive for the test data
# 
# Please note that the parameters for the tree are obtained from various combinations & as well with the Grid Search below

# ## Hyperparameter tuning using Grid Search

# In[166]:


dt_parameters={"criterion":['gini','entropy'],"max_depth":[3,7],"max_leaf_nodes": [20,30],"n_estimators":[100,200,300]}


# In[167]:


from sklearn.model_selection  import GridSearchCV
grid_rf=GridSearchCV(RandomForestClassifier(),dt_parameters)


# In[168]:


grid_rf_model=grid_rf.fit(x_res,y_res)


# In[169]:


grid_rf_model.best_params_


# In[170]:


grid_predictor=grid_rf_model.predict(x_test)


# In[171]:


print(confusion_matrix(y_test,grid_predictor))


# # 7.2  Logistic Regression <a id='SMOTE_log'>

# In[172]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=0.2)


# In[173]:


logreg.fit(x_res,y_res)


# In[174]:


y_logi_pred=logreg.predict(x_test)


# In[175]:


print(confusion_matrix(y_test,y_logi_pred))


# *Observation*: So far LogisticRegression and Random Forest didnt give a good TP/TN values

# 
# ## 8. Upsample the minorty and downsample the majority classes   
# Now, lets see a different sampling technique  <a id='Resample'>

# In[176]:


from sklearn.utils import resample


# In[177]:


train_data=pd.concat([x_train,y_train],axis=1)


# In[178]:


train_data.shape


# In[179]:


df_majority = train_data[train_data.failure==0]
df_minority = train_data[train_data.failure==1]


# In[180]:


df_majority.shape


# In[181]:


df_minority.shape


# Lets do the resampling now for both the classes

# In[182]:


df_minorty_upsample=resample(df_minority, 
                             replace=True,     # sample with replacement
                             n_samples=30000,    # to some good number
                             random_state=123) # reproducible results
 


# In[183]:


df_majoirty_downsample=resample(df_majority, 
                             replace=False,     # sample without replacement
                             n_samples=50000,    # to some good number
                             random_state=321) # reproducible results
 


# In[184]:


df_majoirty_downsample.shape


# In[185]:


df_minorty_upsample.shape


# In[186]:


final_sample_merged=pd.concat([df_minorty_upsample,df_majoirty_downsample],axis=0)


# In[187]:


final_sample_merged.head()


# In[188]:


final_sample_merged['failure'].value_counts()


# In[189]:


x_resample_train=final_sample_merged[features]
y_resample_train=final_sample_merged['failure']


# In[190]:


y_resample_train.shape


# #  8.1 RandomForest  <a id='Resample_RF'>

# In[191]:


model_rf=RandomForestClassifier(criterion='entropy', max_depth = 6, n_estimators = 200, max_leaf_nodes=10,
                                    min_samples_leaf=10,min_samples_split=40,random_state = 1)


# *Note*: Above parameters have been obtained through trails and grid search

# In[192]:


model_rf.fit(x_resample_train,y_resample_train)


# In[193]:


y_pred_train=model_rf.predict(x_resample_train)


# In[194]:


y_pred=model_rf.predict(x_test)


# In[195]:


confusion_matrix(y_test,y_pred)


# In[196]:


print(metrics.classification_report(y_test,y_pred))


# # K Fold Cross Validation 

# In[197]:


from sklearn.model_selection import cross_val_score


# In[198]:


model_kfold_rf=RandomForestClassifier(criterion='entropy', max_depth = 6, n_estimators = 200, max_leaf_nodes=10,
                                    min_samples_leaf=10,min_samples_split=40,random_state = 1,)


# In[199]:


scores = cross_val_score(model_kfold_rf, x_resample_train, y_resample_train, scoring='recall', cv=10)


# In[200]:


scores


# In[201]:


model_kfold_rf.fit(x_resample_train,y_resample_train)


# In[202]:


y_pred_prob=model_kfold_rf.predict_proba(x_test)


# In[203]:


y_cv_pred=model_kfold_rf.predict(x_test)


# In[204]:


confusion_matrix(y_test,y_cv_pred)


# In[205]:


print(metrics.classification_report(y_test,y_cv_pred))


# Here we have used the 10 Fold cross validation to cover the entire data set so that the model will learn from all the folds and validate with 1 fold each time

# **Observation**: This Sensitivity(TP) rate of 0.68 is relatively better than the intial Random forest 0.58, keeping the Specificity(TN) intact

# Now , we will see the ROC curves to find out the correct threshold probability to segregate the classes to  0 and 1

# #  ROC/AUC Curve
# Let us draw ROC AUC curve to see the correct threshold B/W TP and FPs

# In[206]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob[:,1])


# In[207]:


plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Device Failure classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# *Observation*: From the Curve , we can obseve that the FP,TP combination seems to be optimal at point approximately close to  (0.2,0.8). Now let us calcualte the corresponding threshold

# In[208]:


roc_df=pd.DataFrame(columns={'fpr','tpr','threshold'})


# In[209]:


roc_df['fpr']=fpr
roc_df['tpr']=tpr
roc_df['threshold']=thresholds


# In[210]:


roc_df.loc[(roc_df['fpr']<0.16) & (roc_df['tpr']>0.77) ]


# *Observation*: From the above, we can see the right threshold is 0.218607

# In[211]:


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    """
    return [1 if y >= t else 0 for y in y_scores]


# In[212]:


y_pred=adjusted_classes(y_pred_prob[:,1],0.22)


# In[213]:


confusion_matrix(y_test,y_pred)


# In[214]:


print(metrics.classification_report(y_test,y_pred))


# So, accordingly we need to adjust the threshold probability to have a trade off between the TruePositive/TN rates

# In[215]:


# We will also see the Area under curve
print(metrics.roc_auc_score(y_test, y_pred_prob[:,1]))


# # 8.2 Logistic Regression   <a id='Resample_log'>

# In[216]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=0.6)


# In[217]:


logreg.fit(x_resample_train,y_resample_train)


# In[218]:


y_log_pred=logreg.predict(x_test)


# In[219]:


print(confusion_matrix(y_test,y_log_pred))


# In[220]:


y_pred_log_prob=logreg.predict_proba(x_test)


# In[221]:


scores_logreg = cross_val_score(logreg, x_resample_train, y_resample_train, scoring='recall', cv=5)


# In[222]:


scores_logreg


# **Observation** : So far, Randorm forest with the up/down sampling of Minority/Majoiry data sets seems to be perfoming better , relatively

# In[223]:


# ROC Curve for the logistic regression
fpr_log, tpr_log, thresholds_log = metrics.roc_curve(y_test, y_pred_log_prob[:,1])

plt.plot(fpr_log, tpr_log)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Device Failure classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[224]:


# We will also see the Area under curve
print(metrics.roc_auc_score(y_test, y_pred_log_prob[:,1]))


# **Observation**: So far , this value is less than the RandomForest Area under curve

#  

# 
# Now, Lets us apply the Advanced ML boosting techinque
# # 8.3 Adaboost <a id='adb'>

# In[225]:


from sklearn.ensemble import AdaBoostClassifier


# In[226]:


adb_model=AdaBoostClassifier(n_estimators=300,
                             learning_rate=0.2)


# In[227]:


adb_model.fit(x_resample_train,y_resample_train)


# In[228]:


y_adb_predict=adb_model.predict(x_test)


# In[229]:


print(confusion_matrix(y_test,y_adb_predict))


# In[230]:


# GridSearch on Adaboost
from sklearn.model_selection import  GridSearchCV
dt_parameters={"n_estimators":[100,200],"learning_rate":[0.1,0.2,0.4]}

grid_adaboost=GridSearchCV(AdaBoostClassifier(),dt_parameters)
grid_adaboost.fit(x_resample_train,y_resample_train)


# In[231]:


grid_adaboost.best_params_


# In[232]:


y_pred_adb_prob=adb_model.predict_proba(x_test)


# In[233]:


# ROC AUC Curve
fpr_adb, tpr_adb, thresholds_adb = metrics.roc_curve(y_test, y_pred_adb_prob[:,1])

plt.plot(fpr_adb, tpr_adb)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Device Failure classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[234]:


print(metrics.roc_auc_score(y_test, y_pred_adb_prob[:,1]))


# So far, we have tried diversified models like RandomForest, LogisticRegression & Adaboost ( Ensemble) algorithms.
# Random Forest is outperforming the rest 2.
# 
# Now we will also use stacking technique using all the three models and build a meta model to see if we the score is improved
# 

# # 8.4 MLPClassifier(NN) <a id='MLP'>

# In[270]:


from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(learning_rate_init=0.00008)
mlp_clf.fit(x_resample_train,y_resample_train)


# In[271]:


y_mlp_pred=mlp_clf.predict(x_test)
confusion_matrix(y_test,y_mlp_pred)


# In[272]:


y_pred_mlp_prob=mlp_clf.predict_proba(x_test)


# In[273]:


print(metrics.classification_report(y_test,y_mlp_pred))


# In[274]:


fpr_mlp, tpr_mlp, thresholds_mlp = metrics.roc_curve(y_test, y_pred_mlp_prob[:,1])

plt.plot(fpr_mlp, tpr_mlp)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Device Failure classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[275]:


print(metrics.roc_auc_score(y_test, y_pred_mlp_prob[:,1]))


# **Observation**: TN is increased here compared to the Random Forest, although the TP is same. Bu the Area under Curve is better in case of RF after adjusting the probabilities
# 

# # 8.5  Stacking  <a id='stacking'>
#     
#     We will use all the 4 models as base models and build a stacking model to see if there is any improvement

# In[276]:


#Build Stacking model
from sklearn.model_selection import KFold

base1_clf = model_rf
base2_clf = logreg
base3_clf = adb_model
base4_clf = mlp_clf
final_clf =logreg

# Defining the K Fold
n_folds = 5
n_class = 2
kf = KFold(n_splits= n_folds, shuffle=True, random_state=42)


# stacking uses predictions of base classifiers as input for training to a second-level model

# In[277]:


def get_oof(clf, x_train, y_train, x_test):
    ntest = x_test.shape[0]
    oof_train = np.zeros((x_train.shape[0],n_class))
    oof_test  = np.zeros((x_test.shape[0],n_class))
    oof_test_temp = np.empty((n_folds, ntest))
   
    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
  
        
        clf.fit(x_tr, y_tr)

        pred_te = clf.predict_proba(x_te)
        oof_train[test_index,:] = pred_te
        
        pred_test = clf.predict_proba(x_test)
        oof_test += pred_test

    return oof_train, oof_test/n_folds


# In[278]:


base1_oof_train, base1_oof_test = get_oof(base1_clf, x_resample_train.values,y_resample_train.values, x_test.values)
base2_oof_train, base2_oof_test = get_oof(base2_clf, x_resample_train.values,y_resample_train.values, x_test.values)
base3_oof_train, base3_oof_test = get_oof(base3_clf, x_resample_train.values,y_resample_train.values, x_test.values)
base4_oof_train, base4_oof_test = get_oof(base4_clf, x_resample_train.values,y_resample_train.values, x_test.values)


# In[279]:


base1_oof_train


# In[280]:


base1_oof_test


# In[281]:


x_train_stack = np.concatenate((base1_oof_train, 
                          base2_oof_train,
                          base3_oof_train,
                          base4_oof_train), axis=1)
x_test_stack = np.concatenate((base1_oof_test,
                         base2_oof_test,
                         base3_oof_test,
                         base4_oof_test),axis=1)


# In[282]:


x_train_stack.shape


# In[248]:


y_resample_train.shape


# In[249]:


x_test_stack.shape


# In[283]:


final_clf.fit(x_train_stack,y_resample_train)


# In[284]:


y_stacked_predict=final_clf.predict(x_test_stack)


# In[285]:


confusion_matrix(y_test,y_stacked_predict)


# In[286]:


print(metrics.classification_report(y_test,y_stacked_predict))


# **Conclusion**
# 
# When it comes to the TN, stacking is performing good. But overall, the RandormForest and the MLPClassifiers are doing better as a trade of between TP and TN for the given training data.
# 
# 
