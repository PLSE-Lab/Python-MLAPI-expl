#!/usr/bin/env python
# coding: utf-8

# ### Import packages

# In[39]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import data

# In[40]:


data = pd.read_csv("../input/winequality-red.csv")
data.head()


# In[41]:


data.info()


# In[42]:


data.describe()


# In[43]:


from sklearn.preprocessing import LabelEncoder

bins = (2, 6, 8)
group_names = ['bad', 'good']

data['quality'] = pd.cut(data["quality"], bins = bins, labels = group_names)

label_quality = LabelEncoder()

data['quality'] = label_quality.fit_transform(data['quality'].astype(str))
data['quality'].value_counts()


# In[44]:


sns.countplot(data['quality'])
plt.show()


# In[45]:


X = data.drop("quality", axis=1)
y = data["quality"]


# In[46]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25, random_state=10)


# ### Compare Models:

# In[47]:


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[48]:


import warnings
warnings.filterwarnings("ignore")


# In[49]:


# prepare configuration for cross validation test harness
seed = 7

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(("RFC",RandomForestClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[50]:


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ### Fine tuning the models:

# **1. Logistic Regression**

# In[51]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

params_dict={'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty':['l1','l2']}
clf_lr=GridSearchCV(estimator=LogisticRegression(),param_grid=params_dict,scoring='accuracy',cv=10)
clf_lr.fit(X_train,y_train)


# In[52]:


clf_lr.best_params_


# In[53]:


clf_lr.best_score_ 


# In[54]:


pred=clf_lr.predict(X_test)
accuracy_score(pred,y_test)


# **Evaluate a score by cross-validation**

# In[55]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf_lr, X_train, y_train,n_jobs=-1, cv=10)
scores


# In[56]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(scores)


# In[57]:


y_test.value_counts()


# In[58]:


from sklearn.metrics import confusion_matrix,  roc_auc_score
confusion_matrix(pred, y_test)


# In[59]:


roc_auc_score(y_test, pred)


# In[60]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve LR')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[61]:


auc = np.trapz(tpr,fpr)
print('AUC:', auc)


# **Evaluate threshold**

# ![](http://cancerdiscovery.aacrjournals.org/content/candisc/3/2/148/F1.large.jpg?width=800&height=600&carousel=1)

# In[62]:


def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


# In[63]:


evaluate_threshold(0.5)


# In[64]:


evaluate_threshold(.3)


# It seems like changing the threshold values does not implies any changes in sensitivity and specificity.

# **Lets see cross_validation in AUC scores:**

# In[65]:


# calculate cross-validated AUC
cross_val_score(clf_lr, X_train, y_train, cv=10, scoring='roc_auc').mean()


# Better than previous auc score(.65)

# **2. SVM **

# In[66]:


params_dict={'C':[0.01,0.1,1,10],'gamma':[0.01,0.1,1,10],'kernel':['linear','rbf']}
clf=GridSearchCV(estimator=SVC(),param_grid=params_dict,scoring='accuracy',cv=5)
clf.fit(X_train,y_train)


# In[67]:


clf.best_params_


# In[68]:


clf.best_score_ 


# In[72]:


pred_svm=clf.predict(X_test)
accuracy_score(pred_svm,y_test)


# In[76]:


confusion_matrix(pred_svm, y_test)


# In[77]:


roc_auc_score(y_test, pred_svm)


# In[80]:


# calculate cross-validated AUC
cross_val_score(clf, X_train, y_train, cv=4, scoring='roc_auc').mean()


# In[81]:


scores = cross_val_score(clf, X_train, y_train, cv=5)
display_scores(scores)


# **3. Random Forest**

# In[82]:


params_dict={'n_estimators':[500],'max_features':['auto','sqrt','log2']}
clf_rf=GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),param_grid=params_dict,scoring='accuracy',cv=5)
clf_rf.fit(X_train,y_train)


# In[83]:


clf_rf.best_params_


# In[84]:


clf_rf.best_score_ 


# In[85]:


pred_rf=clf_rf.predict(X_test)
accuracy_score(pred_rf,y_test)


# In[86]:


confusion_matrix(pred_rf, y_test)


# In[87]:


roc_auc_score(y_test, pred_rf)


# In[88]:


cross_val_score(clf_rf, X_train, y_train, cv=4, scoring='roc_auc').mean()


# In[89]:


scores = cross_val_score(clf_rf, X_train, y_train, cv=5)
display_scores(scores)


# In[ ]:




