#!/usr/bin/env python
# coding: utf-8

# # Predicting breast cancer using PCA and Multiple classification models.

# # Attribute Information:
# 1) ID number                                   
# 2) Diagnosis (M = malignant, B = benign)       
# 
# All attributes are real values. Some of the attributes are as follows:
# 
#     a) radius (mean of distances from center to points on the perimeter) 
#     b) texture (standard deviation of gray-scale values)
#     c) perimeter
#     d) area
#     e) smoothness (local variation in radius lengths)
#     f) compactness (perimeter^2 / area - 1.0)
#     g) concavity (severity of concave portions of the contour)
#     h) concave points (number of concave portions of the contour)
#     i) symmetry
#     j) fractal dimension ("coastline approximation" - 1)
# 
# Perform modeling using six algorithms (click to read its documentation in sklearn):
# 
#            i) Decision Trees,
#            ii) Random Forest,
#            iii) ExtraTreesClassifier,
#            iv) Gradient Boosting Machine
#            v) XGBoost, and
#            vi) KNeighborsClassifier 
# In this assignment, we will be doing the following activities.
# 
#         i)  Read dataset. Check if any column has any missing variable.
#         ii) Drop any column not needed (ID column, for example)
#         iii)Segregate dataset into predictors (X) and target (y)
#         iv) Map values in ' y ' (target) from 'M' and 'B' to 1 and 0
#         v)  Scale all numerical features in X  using sklearn's StandardScaler class
#         vi) Perform PCA on numeric features, X. Only retain as many PCs, to explain 95% of the variance   
#         vii)Use PCs from (vi) as your explanatory variables. This is our new X.
#         viii)Split X,y into train and test datasets in the ratio of 80:20.
#         ix) Perform modeling on (X_train,y_train) using above listed algorithms (six).
#         x) Make predictions on test (X_test) for each one of the models. Compare the output of predictions 
#         in each case with actual (y_test)
#         xi) Compare the performance of each of these models by calculating metrics::
#              a) accuracy,
#              b) Precision & Recall,
#              c) F1 score,
#              d) AUC
#         xii) Also draw ROC curve for each

# ### Import the required libraries and classifiers from the sklearn. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.datasets import make_hastie_10_2
import os


# #### Import StandardScaler function to scale the numerical features.

# In[ ]:


from sklearn.preprocessing import StandardScaler as ss


# #### Import the metrics from sklearn

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split


# #### Import PCA from skelarn

# In[ ]:


from sklearn.decomposition import PCA


# #### Get the file and read it

# In[ ]:


os.chdir("../input")
os.listdir()


# In[ ]:


data = pd.read_csv("../input/data.csv")


# #### Simple Exploration of data

# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.dtypes


# In[ ]:


data.head()


# In[ ]:


data.shape


# ##### Droping the column id and Unnamed: 32 (null column) as these won't play any role while predicting the breast camcer.

# In[ ]:


df=data.drop(['id','Unnamed: 32'],axis=1)


# In[ ]:


df.shape


# In[ ]:


df.head()


# #### Separate the predictors and target. Here our aim to predict whether a breast cancer cell is benign or malignant. So the column 'diagnosis' is the target and rest other 30 columns are act as predictors(features).

# In[ ]:


X = df.loc[: , 'radius_mean':'fractal_dimension_worst']
y = df.loc[:, 'diagnosis']


# #### Map the values in target column with 1 and 0, here we are mapping M as 1 and B as 0. 

# In[ ]:


df['diagnosis'].replace('M',1,inplace=True)
df['diagnosis'].replace('B',0,inplace=True)


# #### Scale all numerical features in X  using sklearn's StandardScaler class

# In[ ]:


scale = ss()
X = scale.fit_transform(X)


# #### Performing PCA on numeric columns. 
# ###### PCA is helpfull to reduce the dimentionality of the feature columns. As of now we have 30 features in the data, so we need to reduce the number of feature columns, at the same time we need to take care about the variance in data also. In this case we are considering the variance as .95. PCA transforms the the existing set of features into new set of features(reduced in number) and these are called Principal Componants. 
# 

# In[ ]:


pca = PCA()
out = pca.fit_transform(X)
out.shape 


# #### Calculate the variance of each columns in the predictors.

# In[ ]:


pca.explained_variance_ratio_


# #### Calculate the cumulative sum of each column, by this we can decide that how many PCs (Principal components) we need to consider to get desired variance. As I told before in this case we are considering the .95 variance, so we can take first 10 columns as PCs.

# In[ ]:


pca.explained_variance_ratio_.cumsum()


# #### Assign the first 10 columns of the 'out' to final_data. 'out' has the fit and transformed values after we performed the PCA of feature columns.

# In[ ]:


final_data = out[:, :10]


# In[ ]:


final_data.shape


# In[ ]:


final_data[:5,:]


# ##### We need to create the new Dataframe with 10 PCs as predictors and with target column. This will be our new data file. As showed below we are creating the dataframe with values in the final_data and with column names as PC1 through PC10. And 'target' as target column.

# In[ ]:


pcdf = pd.DataFrame( data =  final_data,
                    columns = ['pc1', 'pc2','pc3', 'pc4','pc5','pc6','pc7','pc8','pc9','pc10'])


# In[ ]:


pcdf['target'] = data['diagnosis'].map({'M': 1, "B" : 0 })


# In[ ]:


pcdf.head()


# #### As this is our new file, we need to separate the X(Predictors) and y(target) . These 10 PCs will be our new predictors.

# In[ ]:


X = pcdf.loc[: , 'pc1':'pc10']
y = pcdf.loc[:,'target']


# In[ ]:


X.head()


# In[ ]:


y.head()


# #### Once we get the final predictors and target, we need to split the data into train and test data. For this we can use sklearn's train_test_split function. Here I am using test_size = 0.2, that means I am using 20% of the data as test data and other 80% data for training the model. shuffle=True make sure the data are shuffled before the split, so that random data will go into train and test splits.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 0.2,shuffle=True)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# #### As I told in the begining, here we are using multiple classifier models for prediction. We are creating the default classifier. 

# In[ ]:


dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
xg = XGBClassifier(learning_rate=0.5,
                   reg_alpha= 5,
                   reg_lambda= 0.1
                   )
gbm = GradientBoostingClassifier()
et = ExtraTreesClassifier()
knn = KNeighborsClassifier()


# #### Train the data using data in the X_train and y_train.

# In[ ]:


dt1 = dt.fit(X_train,y_train)
rf1 = rf.fit(X_train,y_train)
xg1 = xg.fit(X_train,y_train)
gbm1 = gbm.fit(X_train,y_train)
et1 = et.fit(X_train,y_train)
knn1 = knn.fit(X_train,y_train)


# #### Once the training the model is done, Make the prediction on the test data. 

# In[ ]:


y_pred_dt = dt1.predict(X_test)
y_pred_rf = rf1.predict(X_test)
y_pred_xg= xg1.predict(X_test)
y_pred_gbm= gbm1.predict(X_test)
y_pred_et= et1.predict(X_test)
y_pred_knn= knn1.predict(X_test)
y_pred_dt


# #### Calculate the prediction probability of the models.

# In[ ]:


y_pred_dt_prob = dt1.predict_proba(X_test)
y_pred_rf_prob = rf1.predict_proba(X_test)
y_pred_xg_prob = xg1.predict_proba(X_test)
y_pred_gbm_prob= gbm1.predict_proba(X_test)
y_pred_et_prob= et1.predict_proba(X_test)
y_pred_knn_prob= knn1.predict_proba(X_test)


# #### Calculate the accuracy score using the sklearn's accuracy_score function. accuracy score is calcuated by y_test,y_pred_dt. y_test is having actual values and y_pred_dt is having predicted values. This is nothingbut how accurate the model is predicting.

# In[ ]:


print (accuracy_score(y_test,y_pred_dt))
print (accuracy_score(y_test,y_pred_rf))
print (accuracy_score(y_test,y_pred_xg))
print (accuracy_score(y_test,y_pred_gbm))
print (accuracy_score(y_test,y_pred_et))
print (accuracy_score(y_test,y_pred_knn))


# #### Prepare the confusion matrix. This uses the y_test and y_pred_dt. y_test is having actual values and y_pred_dt is having predicted values. Once confusion matrix prepared, we will come to know TP,TN,FP,FN values.

# In[ ]:


confusion_matrix(y_test,y_pred_dt)
confusion_matrix(y_test,y_pred_rf)
confusion_matrix(y_test,y_pred_xg)
confusion_matrix(y_test,y_pred_gbm)
confusion_matrix(y_test,y_pred_et)
confusion_matrix(y_test,y_pred_knn)
tn,fp,fn,tp= confusion_matrix(y_test,y_pred_dt).flatten()


# #### ROC Graph

# In[ ]:


fpr_dt, tpr_dt, thresholds = roc_curve(y_test,
                                 y_pred_dt_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


fpr_rf, tpr_rf, thresholds = roc_curve(y_test,
                                 y_pred_rf_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


fpr_xg, tpr_xg, thresholds = roc_curve(y_test,
                                 y_pred_xg_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test,
                                 y_pred_gbm_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


fpr_et, tpr_et,thresholds = roc_curve(y_test,
                                 y_pred_et_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


fpr_knn, tpr_knn,thresholds = roc_curve(y_test,
                                 y_pred_knn_prob[: , 1],
                                 pos_label= 1
                                 )


# #### Calculate the Precision, Recall and F1 Score

# In[ ]:


p_dt,r_dt,f_dt,_ = precision_recall_fscore_support(y_test,y_pred_dt)
p_rf,r_rf,f_rf,_ = precision_recall_fscore_support(y_test,y_pred_rf)
p_gbm,r_gbm,f_gbm,_ = precision_recall_fscore_support(y_test,y_pred_gbm)
p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,y_pred_xg)
p_et,r_et,f_et,_ = precision_recall_fscore_support(y_test,y_pred_et)
p_knn,r_knn,f_knn,_ = precision_recall_fscore_support(y_test,y_pred_knn)
p_dt,r_dt,f_dt


# #### Calculate the AUC(Area Under the ROC Curve). More the AUC, more the better model.

# In[ ]:


print (auc(fpr_dt,tpr_dt))
print (auc(fpr_rf,tpr_rf))
print (auc(fpr_gbm,tpr_gbm))
print (auc(fpr_xg,tpr_xg))
print (auc(fpr_et,tpr_et))
print (auc(fpr_knn,tpr_knn))


# #### Below is the plotting of ROC curve for all the models, we can see that KNN is having more AUC and we can say that KNN is better in this case.

# In[ ]:


fig = plt.figure(figsize=(12,10))          # Create window frame
ax = fig.add_subplot(111)   # Create axes
# 9.2 Also connect diagonals
ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line
# 9.3 Labels etc
ax.set_xlabel('False Positive Rate')  # Final plot decorations
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for multiple models')
# 9.4 Set graph limits
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])

# 9.5 Plot each graph now
ax.plot(fpr_dt, tpr_dt, label = "dt")
ax.plot(fpr_rf, tpr_rf, label = "rf")
ax.plot(fpr_xg, tpr_xg, label = "xg")
ax.plot(fpr_gbm, tpr_gbm, label = "gbm")
ax.plot(fpr_et, tpr_et, label = "et")
ax.plot(fpr_knn, tpr_knn, label = "knn")
# 9.6 Set legend and show plot
ax.legend(loc="lower right")
plt.show()

