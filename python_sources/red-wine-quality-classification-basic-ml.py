#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
from scipy import stats
from itertools import cycle


# # Reading data

# In[ ]:


data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.head()


# # EDA: Exploratory Data Analysis

# Note: We'll consider all possible qualities. In a further step we can transform it into a binary classification problem

# In[ ]:


print(f'Data length: {len(data)}')


# In[ ]:


#check missing data
data.isnull().sum()


# In[ ]:


#correlation matrix
plt.figure(figsize=(10,5))
heatmap = sns.heatmap(data.corr(), annot=True, fmt=".1f")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.show()


# In[ ]:


#Plot pairwise relationships in data for few features  (plot size constraint)
cols_sns = ['residual sugar', 'chlorides', 'density', 'pH', 'alcohol', 'quality']
sns.set(style="ticks")
sns.pairplot(data[cols_sns], hue='quality')


# In[ ]:


sns.countplot(x='quality', data=data)


# In[ ]:


# Features distribution over quality's possible values
fig, ax = plt.subplots(4, 3, figsize=(15, 15))
for var, subplot in zip(data.columns, ax.flatten()):
    if var == "quality":
        continue
    else:
        sns.boxplot(x=data['quality'], y=data[var], data=data, ax=subplot)


# # Preparing Data

# In[ ]:


features, labels = data.loc[:,data.columns !='quality'], data['quality']


# In[ ]:


sns.distplot(labels, kde=True, hist=False)
plt.title('KDE: Kernel Density Estimation')
plt.show()


# We can see that data is skewed and some qualities are more probable (5 and 6) to occur than the others.

# In[ ]:


#scaling data
scaler = MinMaxScaler()
X = scaler.fit_transform(features)


# In[ ]:


#classic split the data into train and test sets for unbalanced data isn't a good idea
xtr, xts, ytr, yts = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle = True)
#We'll opt for stratified shuffled split for better class proportions
sss = StratifiedShuffleSplit(test_size=0.3)
for train_index, test_index in sss.split(X, labels):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


# In[ ]:


#Checking if train and test lables have the same set of possible values
y_train.unique(), y_test.unique()


# # Logistic Regression

# In[ ]:


logreg = LogisticRegression(multi_class='ovr', class_weight='balanced', random_state=42)


# In[ ]:


logreg.fit(X_train, y_train)


# In[ ]:


y_pred = logreg.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


classes_q = sorted(data.quality.unique())
classes_q


# In[ ]:


#For ROC curves we have to binarize lables
y_test_bin = label_binarize(y_test, classes=classes_q)
y_pred_bin = label_binarize(y_pred, classes=classes_q)
#Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes_q)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[ ]:


#ROC for a specific class
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


#ROC for multiclass #sklearn doc
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes_q))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes_q)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(classes_q)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(figsize=(10,5))
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'magenta'])
for i, color in zip(range(len(classes_q)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of quality {0} (area = {1:0.2f})'
             ''.format(classes_q[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# Bigger area in the top left corner means Better results. Considering the task as a binary classification one may give better results

# # Binary Classification

# In[ ]:


#Initial dataset
data.head()


# In[ ]:


#Transform labels into binary ones: 'bad' and 'good'
bins = (2, 6.5, 8)
quality_names = ['bad', 'good']
data['quality'] = pd.cut(data.quality, bins=bins, labels=quality_names)
data.head()


# ### EDA: Binary Classification Task

# In[ ]:


sns.countplot(data=data, x='quality')
plt.title('Quality Count Plot')
plt.show()


# In[ ]:


#Plot pairwise relationships in data for few features  (plot size constraint)
cols_sns = ['residual sugar', 'chlorides', 'density', 'pH', 'alcohol', 'quality']
sns.set(style="ticks")
sns.pairplot(data[cols_sns], hue='quality')


# In[ ]:


# Features distribution over quality's possible values
fig, ax = plt.subplots(4, 3, figsize=(15, 15))
for var, subplot in zip(data.columns, ax.flatten()):
    if var == "quality":
        continue
    else:
        sns.boxplot(x=data['quality'], y=data[var], data=data, ax=subplot)


# In[ ]:


#Encoding labels
label_encoder = LabelEncoder()
data['quality'] = label_encoder.fit_transform(data['quality'])
data.head()


# In[ ]:


#Prepare data
X, y = data.loc[:, data.columns != 'quality'], data['quality']
new_scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#We'll opt for stratified shuffled split again for better class proportions
new_sss = StratifiedShuffleSplit(test_size=0.3)
for train_index, test_index in new_sss.split(X, labels):
    new_X_train, new_X_test = X[train_index], X[test_index]
    new_y_train, new_y_test = y[train_index], y[test_index]


# In this part we'll try multiple algorithms.

# ## Logistic Regression

# In[ ]:


log_reg = LogisticRegression(random_state=42)


# In[ ]:


log_reg.fit(new_X_train, new_y_train)


# In[ ]:


logreg_y_pred = log_reg.predict(new_X_test)


# In[ ]:


print(f'Precision Score: {precision_score(new_y_test, logreg_y_pred)}')
print(f'Recall Score: {recall_score(new_y_test, logreg_y_pred)}')
print(f'F1-Score: {f1_score(new_y_test, logreg_y_pred)}') 


# In[ ]:


#Cross Validation for Logistic Regression
lg = LogisticRegression()
cv_lg_result = cross_val_score(lg, X, y, cv=5, scoring='f1_macro')


# In[ ]:


print(f'Mean F1-Score of Cross Validation {np.mean(cv_lg_result)}')


# In[ ]:


#Grid Search 
grid ={"C": [0.001,0.01,0.1,1,10,100]}
lg_ = LogisticRegression()
lg_cv = GridSearchCV(lg_, grid, cv=3)
lg_cv.fit(new_X_train, new_y_train)

#hyperparameters
print(f"Tuned hyperparameters: {lg_cv.best_params_}")
print(f"Best score: {lg_cv.best_score_}")


# In[ ]:


best_logreg = lg_cv.best_estimator_
best_logreg


# In[ ]:


y_gs_pred = best_logreg.predict(new_X_test)
print("With Grid Search...")
print(f'Precision Score: {precision_score(new_y_test, y_gs_pred)}')
print(f'Recall Score: {recall_score(new_y_test, y_gs_pred)}')
print(f'F1-Score: {f1_score(new_y_test, y_gs_pred)}') 


# In[ ]:


#ROC
fpr, tpr, thresholds = roc_curve(new_y_test, y_gs_pred)
plt.plot([0,1], [0,1], '--k')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()


# In[ ]:


#Confusion Matrix
cm = confusion_matrix(new_y_test, y_gs_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()


# ## Random Forest

# In[ ]:


rf = RandomForestClassifier(random_state=42)


# In[ ]:


rf.fit(new_X_train, new_y_train)


# In[ ]:


y_rf_pred = rf.predict(new_X_test)


# In[ ]:


print('Random Forest Performance...')
print(f'Precision Score: {precision_score(new_y_test, y_rf_pred)}')
print(f'Recall Score: {recall_score(new_y_test, y_rf_pred)}')
print(f'F1-Score: {f1_score(new_y_test, y_rf_pred)}') 


# In[ ]:


#ROC
fpr_rf, tpr_rf, thresholds_rf = roc_curve(new_y_test, y_rf_pred)
plt.plot([0,1], [0,1], '--k')
plt.plot(fpr_rf, tpr_rf)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()


# We can easily notice that the results improved comparing to Logistic Regression

# In[ ]:


rf_param_grid = {"n_estimators": np.arange(2,50),
                "max_depth": np.arange(2,50),
                "min_samples_split": np.arange(2,50),
                "min_samples_leaf":np.arange(2,50),
                "max_leaf_nodes": np.arange(2,50)}

i=1
for param, range_param in rf_param_grid.items():
        rf_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid={param:range_param},
                              scoring='f1_macro')
        rf_grid.fit(new_X_train, new_y_train)
        df = pd.DataFrame(rf_grid.cv_results_)
        plt.figure(figsize=(20,5))
        plt.subplot(2,3,i)
        plt.plot(range_param, df.mean_test_score.values)
        plt.title(param)
        i += 1


# In[ ]:


#Based on the plots above we can test the following model
rf_test = RandomForestClassifier(n_estimators=28,
                                max_depth=14,
                                random_state=42)
rf_test.fit(new_X_train, new_y_train)
yy = rf_test.predict(new_X_test)
f1_score(new_y_test, yy)


# In[ ]:




