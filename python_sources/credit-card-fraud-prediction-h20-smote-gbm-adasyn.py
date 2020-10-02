#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Call libraries
get_ipython().run_line_magic('reset', '-f')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[ ]:


# For measuring time elapsed
from time import time

#  Working with imbalanced data
from imblearn.over_sampling import SMOTE, ADASYN

# Processing data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss

#  Model building
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from sklearn.ensemble import GradientBoostingClassifier

# for ROC graphs & metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics

# Model evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


# In[ ]:


# Change ipython options to display all data columns
pd.options.display.max_columns = 300


# In[ ]:


cc = pd.read_csv("../input/creditcard.csv")


# In[ ]:


cc.head(3)


# In[ ]:


cc.info()


# In[ ]:


cc.describe()


# In[ ]:


# Visualization
cc.boxplot(column = ['Amount'], by = ['Class'])


# In[ ]:


cc.boxplot(column = ['Time'], by = ['Class'])


# In[ ]:


cc.shape 


# In[ ]:


cc.columns.values 


# In[ ]:


cc.dtypes.value_counts()


# In[ ]:


cc['Class'].value_counts()


# In[ ]:


(cc.isnull()).apply(sum, axis = 0)


# In[ ]:


cc['Class'].value_counts()[1]/cc.shape[0]


# In[ ]:


# Separation into target/predictors
y = cc.iloc[:,30]
X = cc.iloc[:,0:30]
X.shape


# In[ ]:


y.shape


# In[ ]:


X.head(3)


# In[ ]:


y.head(3)


# In[ ]:


# Scale and Transform data
scale = ss()
X_trans = scale.fit_transform(X)
X_trans.shape 


# In[ ]:


# Split data into train/test
#     train-test split. startify on 'y' variable, Default is None.
X_train, X_test, y_train, y_test =   train_test_split(X_trans,
                                                      y,
                                                      test_size = 0.3,
                                                      stratify = y
                                                      )

X_train.shape


# In[ ]:


# -------------ADASYN and Gradient Boosting Classifier-----------------
#  Oversample X_train data with ADASYN()
ad = ADASYN(random_state=42)
X_ads, y_ads = ad.fit_sample(X_train, y_train)
X_ads.shape 


# In[ ]:


np.unique(y_ads, return_counts = True)


# In[ ]:


X_ads.shape


# In[ ]:


y_ads.shape


# In[ ]:


#Creating default classifier
gbm = GradientBoostingClassifier()


# In[ ]:


#Training data
gbm1 = gbm.fit(X_ads,y_ads)


# In[ ]:


#Making predictions
y_pred_gbm = gbm1.predict(X_test)


# In[ ]:


#Getting probability values
y_pred_gbm_prob = gbm1.predict_proba(X_test)


# In[ ]:


#Calculating accuracy
print("Accuracy score of Gradient Boost Classifier is :" ,accuracy_score(y_test,y_pred_gbm))


# In[ ]:


# Calculating Precision/Recall/F-score
p_gbm,r_gbm,f_gbm,_ = precision_recall_fscore_support(y_test,y_pred_gbm)
print("The precision, recall and fscore of Gradient Boost Classifier are :", p_gbm,r_gbm,f_gbm)


# In[ ]:


#Drawing Confusion matrix
print("Confusion matrix of Gradient Boost Classifier is :" ,confusion_matrix(y_test,y_pred_gbm))


# In[ ]:


#FPR and TPR Values
fpr_gbm, tpr_gbm, thresholds = roc_curve(y_test,
                                 y_pred_gbm_prob[: , 1],
                                 pos_label= 1
                                 )


# In[ ]:


#Calculating AUC values
print("The AUC Value of Gradient Boost Classifier is :",auc(fpr_gbm,tpr_gbm))


# In[ ]:


# Plotting Graph
fig = plt.figure(figsize=(12,10))          # Create window frame
ax = fig.add_subplot(111)   # Create axes
# connecting diagonals
ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line
# Creating Labels for Graph
ax.set_xlabel('False Positive Rate')  # Final plot decorations
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curve for models')
# Setting graph limits
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])

# Plotting each graph
ax.plot(fpr_gbm, tpr_gbm, label = "Gradient Boost")

# Setting legend and show plot
ax.legend(loc="lower right")
plt.show()


# In[ ]:


# -------------SMOTE and H20-----------------
#  Oversample X_train data with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_train, y_train)


# In[ ]:


type(X_res)


# In[ ]:


X_res.shape


# In[ ]:


np.unique(y_res, return_counts = True)


# In[ ]:


X_res.shape


# In[ ]:


y_res.shape 


# In[ ]:


y_res = y_res.reshape(y_res.size, 1)
y_res.shape


# In[ ]:


X = np.hstack((X_res,y_res))
X.shape 


# In[ ]:


#  Start h2o
h2o.init()


# In[ ]:


# Transform X to h2o dataframe
df = h2o.H2OFrame(X)
len(df.columns)


# In[ ]:


df.shape 


# In[ ]:


df.columns


# In[ ]:


# Get list of predictor column names and target column names
#     Column names are given by H2O when we converted array to H2o dataframe
X_columns = df.columns[0:30]        # Only column names. No data
X_columns   


# In[ ]:


y_columns = df.columns[30]
y_columns


# In[ ]:


df['C31'].head()


# In[ ]:


# For classification, target column must be factor
#      Required by h2o
df['C31'] = df['C31'].asfactor()


# In[ ]:


#  Build a deeplearning model on balanced data
dl_model = H2ODeepLearningEstimator(epochs=1000,
                                    distribution = 'bernoulli',                 
                                    missing_values_handling = "MeanImputation", 
                                    variable_importances=True,
                                    nfolds = 2,                           
                                    fold_assignment = "Stratified",      
                                    keep_cross_validation_predictions = True,  
                                    balance_classes=False,          
                                    standardize = True,                 
                                    activation = 'RectifierWithDropout',
                                    hidden = [100,100],                
                                    stopping_metric = 'logloss',
                                    loss = 'CrossEntropy')


# In[ ]:


# Train our model
start = time()
dl_model.train(X_columns,
               y_columns,
               training_frame = df)
end = time()
(end - start)/60


# In[ ]:


# Get model summary
type(dl_model)


# In[ ]:


print(dl_model)


# In[ ]:


dl_model.cross_validation_holdout_predictions()


# In[ ]:


dl_model.varimp()


# In[ ]:


##### Making predictions on X_test ######
##  We need to transfrom (X_test, y_test) to H2O DataFrame
#      note that (X_test,y_test) is unbalanced and both are numpy arrays
#      Check:
np.unique(y_test, return_counts = True)  


# In[ ]:


# Time to make predictions on actual unbalanced 'test' data
# create a composite X_test data before transformation to H2o dataframe.
# Check the shape of X_test
X_test.shape    


# In[ ]:


#  Test the shape of y_test
y_test.shape   


# In[ ]:


# we want a vertical y_test array to stack with X_test
#      y_test should have as many rows as X_test has, and each
#      element of y_test will be stacked against corresponding
#      row. That is y_test should be 2D
y_test = y_test.ravel()
y_test = y_test.reshape(len(y_test), 1)
y_test.shape  


# In[ ]:


# Column-wise stack X-test and y_test
X_test = np.hstack((X_test,y_test))
X_test.shape    


# In[ ]:


# Transform X_test to h2o dataframe
X_test = h2o.H2OFrame(X_test)
X_test['C31'] = X_test['C31'].asfactor()


# In[ ]:


#  Make prediction on X_test
result = dl_model.predict(X_test[: , 0:30])
result.shape    


# In[ ]:


result.as_data_frame().head() 


# In[ ]:


# Ground truth
# Convert H2O frame back to pandas dataframe
xe = X_test['C31'].as_data_frame()
xe['result'] = result[0].as_data_frame()
xe.head()


# In[ ]:


xe.columns


# In[ ]:


# compare ground truth with predicted
out = (xe['result'] == xe['C31'])
np.sum(out)/out.size


# In[ ]:


# create confusion matrix
f  = confusion_matrix( xe['C31'], xe['result'] )
f 


# In[ ]:


#  Flatten 'f' now
tn,fp,fn,tp = f.ravel()
tn,fp,fn,tp


# In[ ]:


#  Evaluate precision/recall
precision = tp/(tp+fp)
precision


# In[ ]:


recall = tp/(tp + fn)
recall 


# In[ ]:


#  fpr and tpr for all thresholds of the classification
pred_probability = result["p1"].as_data_frame()


# In[ ]:


# fpr, tpr for various thresholds
fpr, tpr, threshold = metrics.roc_curve(xe['C31'],
                                        pred_probability,
                                        pos_label = 1
                                        )


# In[ ]:


# Plot AUC curve now
plt.plot(fpr,tpr)
plt.show()


# In[ ]:


#  Calculate AUC
auc = np.trapz(tpr,fpr)
auc


# In[ ]:


#  Which columns are important
var_df = pd.DataFrame(dl_model.varimp(),
             columns=["Variable", "Relative Importance", "Scaled Importance", "Percentage"])
var_df.head(10)


# In[ ]:




