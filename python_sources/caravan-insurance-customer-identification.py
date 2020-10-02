#!/usr/bin/env python
# coding: utf-8

# ### Pre-Processing

# ###### Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE                   # For Oversampling
#from outliers import smirnov_grubbs as grubbs
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC


import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns


# ###### Read and Partition Data

# In[ ]:


dataset = read_csv('../input/caravan-insurance-challenge.csv')


var=16 

print(dataset.describe())
print('Variables selected :  ', list(dataset.columns.values[[3,10,16,25,29,31,33,40,41,42,43,44,47,59,61,68]]))

selected = dataset.columns.values[[3,10,16,25,29,31,33,40,41,42,43,44,47,59,61,68]]

X = (dataset[dataset.columns[[3,10,16,25,29,31,33,40,41,42,43,44,47,59,61,68]]].values)



# Normalization - Using MinMax Scaler
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

y = np.vstack(dataset['CARAVAN'].values)

print('\n')
print('X and y Input Data:   ', X.shape, y.shape)


X_train_original, X_test2, y_train_original, y_test2 = train_test_split(X, y, test_size=0.3,
                                                                        random_state=42)

print('Training Set Shape:   ', X_train_original.shape, y_train_original.shape)

X_val, X_test, y_val, y_test = train_test_split(X_test2, y_test2, test_size=0.33,random_state=42)
# Used Seed in Partitioning so that Test Set remains same for every Run

print('Validation Set Shape: ', X_val.shape,y_val.shape)
print('Test Set Shape:       ', X_test.shape, y_test.shape)


# ###### Outlier Detection

# In[ ]:


#for i in range(var):
#    print((grubbs.test(X_train[:,i], alpha=0.025).reshape(-1)).shape)


# ###### Oversampling of underrepresented class

# In[ ]:


doOversampling = True

if doOversampling:
# Apply regular SMOTE
    sm = SMOTE(kind='regular')
    X_train, y_train = sm.fit_sample(X_train_original, y_train_original)
    print('Training Set Shape after oversampling:   ', X_train.shape, y_train.shape)
    print(pd.crosstab(y_train,y_train))
else:
    X_train = X_train_original
    y_train = y_train_original


# ###### Scatterplot for Variable Selection

# In[ ]:


# Plot the feature importances of the forest
'''
plt.figure(figsize=(6 * 2, 2.4 * int(var/2+.5)))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
plt.title("Variable Co-relation with Outcome",size=20)
for i in range(var):
    plt.subplot(8, 2, i+1)
    plt.title(selected[i], size=9,color='darkslateblue',fontweight='bold')
    plt.scatter(range(len(X)),X[:,i], s=40, marker= 'o',c=((y[:,0:1])+20).reshape(-1), alpha=0.5)
    plt.yticks()
    plt.xticks()
plt.show()
'''


# ###### Feature Reduction thru PCA - Not used in final phase

# In[ ]:


doPCA = False

if doPCA:
    pca = PCA(svd_solver='randomized',n_components=10,random_state=42).fit(X_train)

    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    #print(pca.components_)
    #print(pca.explained_variance_)
    #print(pca.explained_variance_ratio_) 
    #print(pca.mean_)
    print(pca.n_components_)
    print(pca.noise_variance_)
    plt.figure(1, figsize=(8, 4.5))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')
    plt.show()
else:
    X_train = X_train
    X_val = X_val  


# ###### Flag for Final Run

# In[ ]:


Final_Run = True          # Will Not Process Test Set if value is False


# ### Build Models

# ###### Decision Tree Classifier

# In[ ]:


clf_DT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10, 
                                min_samples_split=2, min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, max_features=None, 
                                max_leaf_nodes=None, min_impurity_split=1e-07)
clf_DT.fit(X_train, y_train)
y_pred_DT = clf_DT.predict(X_val)


# ###### Naive Bayes Classifier

# In[ ]:


clf_NB = BernoulliNB()
clf_NB.fit(X_train, y_train)
y_pred_NB = clf_NB.predict(X_val)
#print(clf_NB.predict_proba(X_val))


# ###### Neural Network Classifier

# In[ ]:


MLPClassifier(activation='relu', alpha=1e-05,
       batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(64), learning_rate='constant',
       learning_rate_init=0.001, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
       tol=0.001, validation_fraction=0.1, verbose=True,
       warm_start=False)
clf_MLP = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(64))

clf_MLP.fit(X_train, y_train)
y_pred_MLP = clf_MLP.predict(X_val)


# ###### Logistic Regression Classifier

# In[ ]:


#clf_Log = LogisticRegression(solver='sag', max_iter=1000, random_state=42,verbose=2)
clf_Log = LogisticRegression(solver='liblinear', max_iter=1000, 
                             random_state=42,verbose=2,class_weight='balanced')

clf_Log.fit(X_train, y_train)
y_pred_Log = clf_Log.predict(X_val)
print(clf_Log.coef_)
print(clf_Log.intercept_)


# ###### Random Forest Classifier

# In[ ]:


clf_RF = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=15,
                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, 
                                bootstrap=True, oob_score=False, n_jobs=1, 
                                random_state=42, verbose=1, warm_start=False, class_weight=None)
clf_RF.fit(X_train, y_train)
y_pred_RF = clf_RF.predict(X_val)


# ###### AdaBoost Classifier

# In[ ]:


clf_AdaB = AdaBoostClassifier(n_estimators=100)
clf_AdaB.fit(X_train, y_train)
y_pred_AdaB = clf_AdaB.predict(X_val)


# ###### Gradient Boost Classifier

# In[ ]:


clf_GB = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
clf_GB.fit(X_train, y_train)
y_pred_GB = clf_GB.predict(X_val)


# ###### Extra Tree Classifier

# In[ ]:


clf_ET = ExtraTreesClassifier(n_estimators=250, random_state=42)
clf_ET.fit(X_train, y_train)
y_pred_ET = clf_ET.predict(X_val)


# ###### SVM Classifier

# In[ ]:


clf_SVM = SVC(C=10, class_weight='balanced', gamma='auto', kernel='rbf',
              max_iter=-1, probability=True, random_state=42, verbose=True)
clf_SVM.fit(X_train, y_train)
y_pred_SVM = clf_SVM.predict(X_val)


# In[ ]:


y_val = y_val.reshape(-1)


# ### Model Performance Comparison

# ###### Compare Accuracy of Models on Validation Set

# In[ ]:


print('       Accuracy of Models       ')
print('--------------------------------')
print('Decision Tree           '+"{:.2f}".format(accuracy_score(y_val, y_pred_DT)*100)+'%')
print('Naive Bayes             '+"{:.2f}".format(accuracy_score(y_val, y_pred_NB)*100)+'%')
print('Neural Network          '+"{:.2f}".format(accuracy_score(y_val, y_pred_MLP)*100)+'%')
print('Logistic Regression     '+"{:.2f}".format(accuracy_score(y_val, y_pred_Log)*100)+'%')
print('Random Forest           '+"{:.2f}".format(accuracy_score(y_val, y_pred_RF)*100)+'%')
print('AdaBoost                '+"{:.2f}".format(accuracy_score(y_val, y_pred_AdaB)*100)+'%')
print('GradientBoost           '+"{:.2f}".format(accuracy_score(y_val, y_pred_GB)*100)+'%')
print('Extra Tree              '+"{:.2f}".format(accuracy_score(y_val, y_pred_ET)*100)+'%')
print('Support Vector Machine  '+"{:.2f}".format(accuracy_score(y_val, y_pred_SVM)*100)+'%')


# ###### Print Confusion Matrix for all Models

# In[ ]:


print('Decision Tree  ')
cm_DT = confusion_matrix(y_val,y_pred_DT)
print(cm_DT)
print('\n')

print('Naive Bayes  ')
cm_NB = confusion_matrix(y_val,y_pred_NB)
print(cm_NB)
print('\n')

print('Neural Network  ')
cm_MLP = confusion_matrix(y_val,y_pred_MLP)
print(cm_MLP)
print('\n')

print('Logistic Regression  ')
cm_Log = confusion_matrix(y_val,y_pred_Log)
print(cm_Log)
print('\n')

print('Random Forest  ')
cm_RF = confusion_matrix(y_val,y_pred_RF)
print(cm_RF)
print('\n')

print('AdaBoost  ')
cm_AdaB = confusion_matrix(y_val,y_pred_AdaB)
print(cm_AdaB)
print('\n')

print('GradientBoost  ')
cm_GB = confusion_matrix(y_val,y_pred_GB)
print(cm_GB)
print('\n')

print('Extra Tree  ')
cm_ET = confusion_matrix(y_val,y_pred_ET)
print(cm_ET)
print('\n')

print('SVM  ')
cm_SVM = confusion_matrix(y_val,y_pred_SVM)
print(cm_SVM)


# ### Test Set Results

# ###### Compare Models on Training, Validation and Test Set Results

# ###### Execute only on Final Run

# In[ ]:


if Final_Run:
    if doPCA:
        X_test = pca.transform(X_test)
        X_train_original = pca.transform(X_train_original)
    y_test = y_test.reshape(-1)
    y_train_original = y_train_original.reshape(-1)
    
    y_pred_train_DT = clf_DT.predict(X_train_original)
    y_pred_train_NB = clf_NB.predict(X_train_original)
    y_pred_train_MLP = clf_MLP.predict(X_train_original)
    y_pred_train_Log = clf_Log.predict(X_train_original)
    y_pred_test_DT = clf_DT.predict(X_test)
    y_pred_test_NB = clf_NB.predict(X_test)
    y_pred_test_MLP = clf_MLP.predict(X_test)
    y_pred_test_Log = clf_Log.predict(X_test)
    cm_DT_train = confusion_matrix(y_train_original,y_pred_train_DT)
    cm_NB_train = confusion_matrix(y_train_original,y_pred_train_NB)
    cm_MLP_train = confusion_matrix(y_train_original,y_pred_train_MLP)
    cm_Log_train = confusion_matrix(y_train_original,y_pred_train_Log)
    cm_DT_test = confusion_matrix(y_test,y_pred_test_DT)
    cm_NB_test = confusion_matrix(y_test,y_pred_test_NB)
    cm_MLP_test = confusion_matrix(y_test,y_pred_test_MLP)
    cm_Log_test = confusion_matrix(y_test,y_pred_test_Log)
    
    print('Decision Tree Classification Matrix  ')
    print('Training')
    print(cm_DT_train)
    print('Validation')
    print(cm_DT)
    print('Test')
    print(cm_DT_test)
    print('\n')

    print('Naive Bayes Classification Matrix ')
    print('Training')
    print(cm_NB_train)
    print('Validation')
    print(cm_NB)
    print('Test')
    print(cm_NB_test)
    print('\n')

    print('Neural Network Classification Matrix ')
    print('Training')
    print(cm_MLP_train)
    print('Validation')
    print(cm_MLP)
    print('Test')
    print(cm_MLP_test)
    print('\n')

    print('Logistic Regression Classification Matrix ')
    print('Training')
    print(cm_Log_train)
    print('Validation')
    print(cm_Log)
    print('Test')
    print(cm_Log_test)
    print('\n')


# ###### Choose Final Model

# Choosing Decision Tree Model as Final Model due to Accuracy + Simplicity

# In[ ]:


clf = clf_NB


# ###### Execute only on Final Run

# In[ ]:


importances_RF = clf_RF.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_RF.estimators_],
             axis=0)
indices1 = np.argsort(importances_RF[0:var])[::-1]

indices = indices1[0:var]
# Print the feature ranking
print("Feature ranking:")

for f in range(var):
    print("%d. %s (%f)" % (f + 1, (dataset.columns.values[[3,10,16,25,29,31,33,40,41,42,43,44,47,59,61,68]]).reshape(-1)[indices[f]], importances_RF[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(14, 3))
plt.title("Most Important Features - Random Forest",size=20)
plt.bar(range(var), importances_RF[indices],
       color="#aa6d0a", yerr=std[indices], align="center")
plt.yticks(size=14)
plt.xticks(range(var), (dataset.columns.values[[3,10,16,25,29,31,33,40,41,42,43,44,47,59,61,68]]).reshape(-1)[indices],rotation='vertical',size=12,color="#201506")
plt.xlim([-1, var])
plt.show()


# In[ ]:


FN_amount = -9950
TP_amount = 9950
TN_amount = 0
FP_amount = -50

if Final_Run:
    print('Compare Profit from Models - Test Set')
    print('-------------------------------------')

    Profit_DT     = (cm_DT_test[0][0]*TN_amount + cm_DT_test[1][0]*FN_amount + cm_DT_test[0][1]*FP_amount +
                   cm_DT_test[1][1]*TP_amount)
    print('Decision Tree Profit(Rs):        ' + str(Profit_DT))

    Profit_NB     = (cm_NB_test[0][0]*TN_amount + cm_NB_test[1][0]*FN_amount + cm_NB_test[0][1]*FP_amount + 
                  cm_NB_test[1][1]*TP_amount)
    print('Naive Bayes Profit(Rs):          ' + str(Profit_NB))

    Profit_MLP    = (cm_MLP_test[0][0]*TN_amount + cm_MLP_test[1][0]*FN_amount + cm_MLP_test[0][1]*FP_amount + 
                  cm_MLP_test[1][1]*TP_amount)
    print('Neural Network Profit(Rs):       ' + str(Profit_MLP))

    Profit_Log    = (cm_Log_test[0][0]*TN_amount + cm_Log_test[1][0]*FN_amount + cm_Log_test[0][1]*FP_amount + 
                  cm_Log_test[1][1]*TP_amount)
    print('Logistic Regression Profit(Rs):  ' + str(Profit_Log))
    
    
    y_pred_test = clf.predict(X_test)
    print('\n\nBest Model Accuracy on Test Set: '+"{:.2f}".format(accuracy_score(y_test, y_pred_test)*100)+'%')
    print('\nConfusion Matrix on Test Set  ')
    cm = confusion_matrix(y_test,y_pred_test)
    print(cm)
    print('\n')
    Profit = cm[0][0]*TN_amount + cm[1][0]*FN_amount + cm[0][1]*FP_amount + cm[1][1]*TP_amount
    print('Profit(Rs) for Test Set:        ' + str(Profit))
    
    # Checked Actual Positive and Negative Class in Test Set
    Max_Profit = 146*TN_amount + 0*FN_amount + 0*FP_amount + 4*TP_amount
    print('Max_Profit = ' + str(Max_Profit))
    
    print('\n')
    print('Test Set Profit % w.r.t Maximum Profit: ' + "{:.2f}".format(float(Profit)/Max_Profit*100)+'%')
    print('\n')
    print('Final Model')
    print('-----------')
    print(str(clf))


# In[ ]:


print(str(clf_DT));print('\n')
print(str(clf_NB));print('\n')
print(str(clf_MLP));print('\n')
print(str(clf_Log));print('\n')
print(str(clf_RF));print('\n')
print(str(clf_AdaB));print('\n')
print(str(clf_GB));print('\n')
print(str(clf_ET));print('\n')
print(str(clf_SVM));print('\n')


# In[ ]:




