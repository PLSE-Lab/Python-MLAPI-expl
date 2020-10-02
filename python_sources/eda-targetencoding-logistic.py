#!/usr/bin/env python
# coding: utf-8

# > Hi guys, if you like my work then please upvote it.
# * *If you have any suggestions then please let me know.*

# **Importing necessary libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# **Importing train**

# In[ ]:


train = pd.read_csv("../input/cat-in-the-dat/train.csv")


# > Below code is for visualising maximum rows and columns

# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[ ]:


print("Total number of train data is:", train.shape)


# In[ ]:


train.head(6)


# **Dropping id**

# In[ ]:


train.drop(["id"], axis=1, inplace=True)


# **Checking For NaN values**

# In[ ]:


train.isna().sum()


# > There is no nan values in the dataset

# **Describing train**

# In[ ]:


train.describe()


# **Checking for duplicates**

# In[ ]:


#train.duplicated()


# > there is no duplicate rows in the train, 
# > you can check it by uncommenting above code

# **Target visualisation(Balanced or not)**

# In[ ]:


tar = train['target'].value_counts()
print("Number of cat ", tar[1], ", (", (tar[1]/(tar[1]+tar[0]))*100,"%)")
print("Number of non_cat ", tar[0], ", (", (tar[0]/(tar[1]+tar[0]))*100,"%)")


# **Bar plot for all features **

# In[ ]:


def bar_plot(feature):
    sns.set(style="darkgrid")
    ax = sns.countplot(x=feature , data=train)
    

    


# **Target**

# In[ ]:


bar_plot("target")


# **bin_0**

# In[ ]:


bar_plot("bin_0")


# **bin_1**

# In[ ]:


bar_plot("bin_1")


# **bin_2**

# In[ ]:


bar_plot("bin_2")


# **bin_3**

# In[ ]:


bar_plot("bin_3")


# **bin_4**

# In[ ]:


bar_plot("bin_4")


# **nom_0**

# In[ ]:


bar_plot("nom_0")


# **nom_1**

# In[ ]:


bar_plot("nom_1")


# **nom_2**

# In[ ]:


bar_plot("nom_2")


# **nom_3**

# In[ ]:


bar_plot("nom_3")


# **nom_4**

# In[ ]:


bar_plot("nom_4")


# **nom_5**

# In[ ]:


bar_plot("nom_5")


# **nom_6**

# In[ ]:


bar_plot("nom_6")


# **nom_7**

# In[ ]:


bar_plot("nom_7")


# **nom_8**

# In[ ]:


bar_plot("nom_8")


# **nom_9**

# In[ ]:


bar_plot("nom_9")


# > From nom_5 to nom_9 barplot is not clear that means number of category for these features is more

# **Let's check for counts for nom_5 to nom_9 **

# In[ ]:


print("Total number of different category for nom_5 is:", train["nom_5"].value_counts().shape[0])
print("Total number of different category for nom_6 is:", train["nom_6"].value_counts().shape[0])
print("Total number of different category for nom_7 is:", train["nom_7"].value_counts().shape[0])
print("Total number of different category for nom_8 is:", train["nom_8"].value_counts().shape[0])
print("Total number of different category for nom_9 is:", train["nom_9"].value_counts().shape[0])


# **ord_0**

# In[ ]:


bar_plot("ord_0")


# **ord_1**

# In[ ]:


bar_plot("ord_1")


# **ord_2**

# In[ ]:


bar_plot("ord_2")


# **ord_3**

# In[ ]:


bar_plot("ord_3")


# **ord_4**

# In[ ]:


bar_plot("ord_4")


# **ord_5**

# In[ ]:


bar_plot("ord_5")


# **day**

# In[ ]:


bar_plot("day")


# **month**

# In[ ]:


bar_plot("month")


# **Test**

# In[ ]:


test = pd.read_csv("../input/cat-in-the-dat/test.csv")


# In[ ]:


test.shape


# In[ ]:


test.head(3)


# In[ ]:


test.drop(["id"], axis=1, inplace=True)


# **Checking if categories of test is present in train or not**

# In[ ]:


test["bin_0"].isin(train["bin_0"]).value_counts()


# In[ ]:


test["bin_1"].isin(train["bin_1"]).value_counts()


# In[ ]:


test["bin_2"].isin(train["bin_2"]).value_counts()


# In[ ]:


test["bin_3"].isin(train["bin_3"]).value_counts()


# In[ ]:


test["bin_4"].isin(train["bin_4"]).value_counts()


# In[ ]:


test["nom_0"].isin(train["nom_0"]).value_counts()


# In[ ]:


test["nom_1"].isin(train["nom_1"]).value_counts()


# In[ ]:


test["nom_2"].isin(train["nom_2"]).value_counts()


# In[ ]:


test["nom_3"].isin(train["nom_3"]).value_counts()


# In[ ]:


test["nom_4"].isin(train["nom_4"]).value_counts()


# In[ ]:


test["nom_5"].isin(train["nom_5"]).value_counts()


# In[ ]:


test["nom_6"].isin(train["nom_6"]).value_counts()


# In[ ]:


test["nom_7"].isin(train["nom_7"]).value_counts()


# In[ ]:


test["nom_8"].isin(train["nom_8"]).value_counts()


# In[ ]:


test["nom_9"].isin(train["nom_9"]).value_counts()


# In[ ]:


test["ord_0"].isin(train["ord_0"]).value_counts()


# In[ ]:


test["ord_1"].isin(train["ord_1"]).value_counts()


# In[ ]:


test["ord_2"].isin(train["ord_2"]).value_counts()


# In[ ]:


test["ord_3"].isin(train["ord_3"]).value_counts()


# In[ ]:


test["ord_4"].isin(train["ord_4"]).value_counts()


# In[ ]:


test["ord_5"].isin(train["ord_5"]).value_counts()


# In[ ]:


test["day"].isin(train["day"]).value_counts()


# In[ ]:


test["month"].isin(train["month"]).value_counts()


# **Conclusion**
# > From the above we can see that nom_8 and nom_9 is the only feature which is present is test but not in train.

# In[ ]:


y = train["target"]
train.drop(["target"], inplace=True, axis=1)


# In[ ]:


X =train
T =test 


# In[ ]:


train = X
test = T


# **OneHotEncoding**

# In[ ]:


df = pd.concat([train, test])
dummies = pd.get_dummies(df, columns=df.columns, drop_first=True, sparse=True)
train = dummies.iloc[:train.shape[0], :]
test = dummies.iloc[train.shape[0]:, :]


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


# **Logistic Regression**

# In[ ]:


def log_alpha(al):
    alpha=[]
    for i in al:
        a=np.log(i)
        alpha.append(a)
    return alpha    


# In[ ]:


# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

svm = SGDClassifier(loss='log', class_weight='balanced')
alpha=alpha = [0.000001,0.000002, 0.000005, 0.00001, 0.00003, 0.00005, 0.00007]
parameters = {'alpha':alpha}
clf = RandomizedSearchCV(svm, parameters, cv=5, scoring='roc_auc', n_jobs=-1, return_train_score=True,)
clf.fit(train, y)

print("Model with best parameters :\n",clf.best_estimator_)

alpha = log_alpha(alpha)


best_alpha = clf.best_estimator_.alpha
best_penalty = clf.best_estimator_.penalty
#best_split = clf.best_estimator_.min_samples_split

print(best_alpha)
print(best_penalty)
#print(best_split)

train_auc= clf.cv_results_['mean_train_score']
train_auc_std= clf.cv_results_['std_train_score']
cv_auc = clf.cv_results_['mean_test_score'] 
cv_auc_std= clf.cv_results_['std_test_score']

plt.plot(alpha, train_auc, label='Train AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(alpha, cv_auc, label='CV AUC')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
plt.gca().fill_between(alpha,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(alpha, train_auc, label='Train AUC points')
plt.scatter(alpha, cv_auc, label='CV AUC points')


plt.legend()
plt.xlabel("alpha and l1")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# In[ ]:


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV

svm = SGDClassifier(loss='log', alpha=best_alpha, penalty=best_penalty, class_weight="balanced")
#svm.fit(train_1, project_data_y_train)
# roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# not the predicted outputs

#sig_clf = CalibratedClassifierCV(svm, method="isotonic")
svm = svm.fit(train, y)


y_train_pred1 = svm.predict(train) 
y_test_pred1 = svm.predict(test)

train_fpr, train_tpr, tr_thresholds = roc_curve(y, y_train_pred1)
#test_fpr, test_tpr, te_thresholds = roc_curve(project_data_y_test, y_test_pred)

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
#plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel(" hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()


# **Submission**

# In[ ]:


sub = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")


# In[ ]:


sub.head(2)


# In[ ]:


submission = pd.DataFrame({'id': sub["id"], 'target': y_test_pred1})
submission.to_csv('submission_log.csv', index=False)


# **Please support my work by upvoting **

# **To be continued....**
