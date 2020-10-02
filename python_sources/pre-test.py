#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Change directory to VSCode workspace root so that relative path loads work correctly. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../../../introML2019_TA/midterm_project/pre-midterm'))
	print(os.getcwd())
except:
	pass


# In[ ]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools
import csv


# In[ ]:


X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")


# In[ ]:


X_train.isna().any()[lambda x: x]


# In[ ]:


NaN_cols = ['D_floorTAGround', 'D_Demand', 'D_sds', 'D_sd1', 'D_td0', 'D_sms', 'D_sm1', 'D_tm0', 'D_MaxCl', 'D_NeutralDepth', 'D_Ra_Capacity','D_Ra_CDR']
NaN_cols


# In[ ]:


sel_cols = set(X_train.columns) - set(NaN_cols)
sel_cols


# In[ ]:


X_data = X_train[sel_cols].values
y_data = y_train["Category"].values.reshape(-1,1)

print(X_data.shape)
print(y_data.shape)


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X_data)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(x_scaled, y_data, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In[ ]:


sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)


# In[ ]:


sel.get_support()


# In[ ]:


X_train_sel = X_train[:, sel.get_support()]
len(X_train_sel[0])


# In[ ]:


X_colname = list(sel_cols)
print("selected features:")
[ X_colname[i] for i in np.where(sel.get_support())[0]]


# In[ ]:


[ X_colname[i] for i in np.where(~sel.get_support())[0]]


# In[ ]:


X_train_sel.shape


# In[ ]:


from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


def scoring(y_test, y_test_pred):
#   print(metrics.f1_score(y_test,y_test_pred))
#   print(metrics.accuracy_score(y_test,y_test_pred))
#   print(metrics.recall_score(y_test,y_test_pred))
    
    print(metrics.classification_report(y_test,y_test_pred))
    plot_confusion_matrix(metrics.confusion_matrix(y_test,y_test_pred), classes=range(2))


# In[ ]:


model = SVC(class_weight='balanced',)


# In[ ]:


model.fit(X_train_sel, y_train)


# In[ ]:


y_test_pred = model.predict(X_test[:, sel.get_support()])


# In[ ]:


plot_confusion_matrix(metrics.confusion_matrix(y_test,y_test_pred), classes=range(2))


# In[ ]:


scoring(y_test, y_test_pred)


# In[ ]:


svm = SVC(kernel='rbf', gamma=0.001, C=10000, class_weight='balanced', degree=8)
svm.fit(X_train_sel, y_train)


# In[ ]:


y_test_pred = svm.predict(X_test[:, sel.get_support()])
scoring(y_test, y_test_pred)
plot_confusion_matrix(metrics.confusion_matrix(y_test,y_test_pred), classes=range(2))


# In[ ]:


rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train_sel, y_train)

y_test_pred = rfc.predict(X_test[:, sel.get_support()])
scoring(y_test, y_test_pred)


# In[ ]:


from xgboost import XGBClassifier
xgbc = XGBClassifier(learning_rate=0.1, n_estimators=1000)
xgbc.fit(X_train_sel, y_train)

y_test_pred = xgbc.predict(X_test[:, sel.get_support()])
scoring(y_test, y_test_pred)


# In[ ]:


xgbc = XGBClassifier(learning_rate=0.1, n_estimators=600, max_depth=4, scale_pos_weight=0.9)
xgbc.fit(X_train_sel, y_train)

y_test_pred = xgbc.predict(X_test[:, sel.get_support()])
scoring(y_test, y_test_pred)


# In[ ]:


xgbc = XGBClassifier(learning_rate=0.1, n_estimators=600, max_depth=100)
xgbc.fit(X_train_sel, y_train)

y_train_pred = xgbc.predict(X_train[:, sel.get_support()])
y_test_pred = xgbc.predict(X_test[:, sel.get_support()])
# print("Training:")
# scoring(y_train, y_train_pred)
print("Testing:")
scoring(y_test, y_test_pred)


# In[ ]:


X_test_test = pd.read_csv("X_test.csv")
X_test_test = X_test_test[sel_cols].values


# In[ ]:


def pred_and_output(model, X_test, outputcsv='pred.csv'):
    y_pred = model.predict(X_test) 
    with open(outputcsv, 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['Id','Category'])
        for i in range(X_test.shape[0]):
            content = [str(i), y_pred[i]]
            w.writerow(content)  


# In[ ]:


pred_and_output(rfc, X_test_test[:, sel.get_support()], 'rfc_pred.csv')
pred_and_output(svm, X_test_test[:, sel.get_support()], 'svm_pred.csv')
pred_and_output(xgbc, X_test_test[:, sel.get_support()], 'xgbc_pred.csv')

