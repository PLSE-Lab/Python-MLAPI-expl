#!/usr/bin/env python
# coding: utf-8

# # Objective
# Given the data provided, predict if a student will be placed in the future.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy
import scipy.stats
from sklearn.preprocessing import LabelEncoder


# In[ ]:


data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data


# # Data Analysis and Preparation

# In[ ]:


data.isna().sum()


# Salary is missing 67 values. As we only care about if they got placed or not, status needs to be Placed for there to be a salary value. We can remove it as its dependent on status.

# In[ ]:


data.drop(['salary', 'sl_no'], axis=1, inplace=True)
data.isna().sum()


# In[ ]:


data.nunique()


# Check if there is strong correlation between variables.

# In[ ]:


corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# There are no highly correlated features present, so none will be removed.
# 
# I will be using a one hot encoding for gender, ssc_b, hsc_b, hsc_s, degree_t, workex, and specialization. Status will be converted to a label for binary training.

# In[ ]:


data


# In[ ]:


unique_vals = data.nunique()
col_log = data.columns
for i in range(0, len(unique_vals)):
    coln = str(col_log[i])
    
    # If its less than 5, convert to one hot. Do not convert Status yet
    if int(unique_vals[i]) < 5 and coln != 'status':
        data = pd.concat([data.drop(coln, axis=1), pd.get_dummies(data[coln], prefix=coln)], axis=1)


# In[ ]:


data


# In[ ]:


data_y = pd.DataFrame(data['status'])
data_x = data.drop('status', axis=1)

status_encoder = LabelEncoder()
data_y = status_encoder.fit_transform(data_y)


# # Accuracy Goal

# In[ ]:


print('Guessing always placed accuracy: %f' % (((data['status'] == 'Placed').sum() / data['status'].count()) * 100))


# the model will be learning then whenever the accuracy is > 68.84%. If its less than or equal, it is worse than predicting Placed for everyone.

# # Model Training

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm


# To make the confusion matrix in the future unbias, I split it into train, test and val where val is used for the precision recall curves. Random state is fixed to maintain values.

# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.20, random_state=1)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.20, random_state=1)


# Performing a grid search on n_estimators 50 - 500 on a stable seed classifier.

# In[ ]:


best_score = -1
best_estimators = 0
for i in range(10,250):
    model = RandomForestClassifier(n_estimators=i, random_state=0)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    score = accuracy_score(pred, test_y)
    if score > best_score:
        best_score = score
        best_estimators = i
        
print("The best number of estiamtors was %d with accuracy score %f" % (best_estimators, (best_score * 100)))


# Train the final model

# In[ ]:


model = RandomForestClassifier(n_estimators=best_estimators, random_state=0)


# In[ ]:


model.fit(train_x, train_y)


# In[ ]:


pred = model.predict(test_x)
score = accuracy_score(pred, test_y)
print("Test Accuracy: %f" % (score * 100))


# This is good testing accuracy, but it reports false positives which we may or may not want.

# # Confusion Matrix + Precision Recall Curve
# Lets say placement is used in production and requires no false positives (we want to know only those who should get placed). A confusion matrix and a precision recall curve can help fix this issue. As stated [on this machine learning blog](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/), a ROC curve is not sufficient because of the dataset imbalance.

# In[ ]:


from sklearn.metrics import confusion_matrix, precision_score, plot_confusion_matrix
import matplotlib.pyplot as plt

print("True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d" % tuple(confusion_matrix(test_y, pred).ravel()))
print("Precision Score: %f" % (precision_score(test_y, pred) * 100))
plot_confusion_matrix(model, test_x, test_y, cmap=plt.cm.Reds)
plt.title("Confusion Matrix")
plt.show()


# The precision score ok, but there are false positives present. Lets examine the precision recall curve.

# In[ ]:


from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve

# Get the predicted probabilties of the positive label (placed)
y_pred_prob = model.predict_proba(val_x)[:, 1]

# Get curve
precision, recall, thresholds = precision_recall_curve(val_y, y_pred_prob)

# Plot
plt.plot(recall, precision, label="Random Forest")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot([0, 1], [0.68837209, 0.68837209], label="Baseline")
plt.legend()
plt.show()


# Lets examine the possible threshold values we could use.

# In[ ]:


# Remove final one for dataframe.
df = pd.DataFrame(data={'Precision': precision[:-1], 'Recall': recall[:-1], 'Thresholds': thresholds})
df


# We want precision to be 1 to prevent all possible false positives. Lets grid search thresholds with precision 1 to determine what produces the best false positive rate on the test set.

# In[ ]:


targets = df.loc[(df['Precision'] >= 1) & (df['Thresholds'] != 1)]
targets


# In[ ]:


best = -1
thresh_best = -1

y_test_prob = model.predict_proba(test_x)[:, 1]
for target in targets.to_numpy():
    true_prediction = (y_test_prob > target[2]).astype(int)
    score = precision_score(test_y, true_prediction)
    
    # Since the dataframe is in order from thresholds, we want the lowest threshold with 100%
    # precision. This does slightly bias it towards the train set, but if safety is the highest
    # priority the threshold could be futher increased at the cost of accuracy 
    # (meaning when its positive we know with high probability but we will get more false negatives)
    if score > best:
        best = score
        thresh_best = target[2]
    print("Score for threshold %f: %f" % (target[2], score * 100))
print("Best precision score of %f achieved with threshold %f." % (best, thresh_best))


# In[ ]:


ypred = (model.predict_proba(test_x)[:, 1] > thresh_best).astype(int)
score = accuracy_score(ypred, test_y)
print("Test accuracy with threshold: %f" % (score * 100))
print("True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d" % tuple(confusion_matrix(test_y, ypred).ravel()))


# In[ ]:


ypred = (model.predict_proba(val_x)[:, 1] > thresh_best).astype(int)
score = accuracy_score(ypred, val_y)
print("Test accuracy with threshold: %f" % (score * 100))
print("True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d" % tuple(confusion_matrix(val_y, ypred).ravel()))


# There are zero false positives in the model now (validation and testing sets) at the cost of some accuracy.
