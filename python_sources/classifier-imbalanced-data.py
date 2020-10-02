#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Credit Card Fraud Detection
# 
# We will make a ML model that will predict which credit card transactions are fraudulent.

# In[ ]:


train = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")


# **Fast check data.**

# In[ ]:


train.info()


# We have 31 columns: Time, Amount, V1-V28 which are the anonymized data, and the predicting variable Class. 

# In[ ]:


train.head()


# The anonymized data is already scaled and normalized, which means we may scale and normalize other variables that are not scaled

# In[ ]:


train.describe().T


# At first glance, we can sea count are all the same means there may be no missing values. lets double check. 

# In[ ]:


train.isnull().values.any()


# In[ ]:


train["Class"].value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


labels = ["Normal", "Fraud"]
sns.countplot("Class",data=train)
plt.title("Class Distribution")
plt.xticks(range(2), labels)
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[ ]:


plt.figure()
plt.scatter(train['Amount'], train['Class'])
plt.show()


# In[ ]:


plt.subplots(figsize =(14,14))
corr = train.corr()
sns.heatmap(corr)


# Because of extreme unbalanced data.. We cant get much from correlation matrix. So we will to do under-sampling method and later implement over-sampling method to see which method will give good results.

# **Under-sampling Method**
# 
# We have 492 fraudulent cases, so to balance the dataset we need to get 492 non-fraudulent data. .
# 
# 
# 

# In[ ]:


from sklearn.utils import shuffle
#get the fraud data and concatenate with ewual size non-fraud data
under_fraud = train[train["Class"] ==1]
under_non_fraud = shuffle(train[train["Class"] == 0], n_samples =492,random_state =42)


# In[ ]:


under_non_fraud


# In[ ]:


under_sample = shuffle(pd.concat([under_fraud,under_non_fraud]),random_state =42)


# In[ ]:


under_sample.shape


# In[ ]:


under_sample.head()


# In[ ]:


plt.subplots(figsize =(14,14))
corr = under_sample.corr()
sns.heatmap(corr)


# With the same number of fraud and non-fraud data we can see the negative and positive correlation of data. but will this help us give the right prediction?

# In[ ]:


from sklearn.preprocessing import StandardScaler, RobustScaler

std_scaler = StandardScaler()
rob_scaler = RobustScaler()
under_sample["Amount"] =  rob_scaler.fit_transform(under_sample['Amount'].values.reshape(-1,1))
under_sample['Time'] = std_scaler.fit_transform(under_sample['Time'].values.reshape(-1,1))


# In[ ]:


from sklearn.model_selection import train_test_split as tts
under_y = under_sample['Class']
under_X = under_sample.drop('Class', axis = 1)


# In[ ]:


X_train, X_test, y_train, y_test = tts(under_X, under_y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier()
}


# In[ ]:


from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    y_pred = cross_val_predict(classifier, X_train, y_train, cv=5)
    cf = confusion_matrix(y_train, y_pred)
    precision, recall, threshold = precision_recall_curve(y_train, y_pred)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
    print("Confusion matrix:\n",cf)
    print('Area under ROC curve score: ', roc_auc_score(y_train, y_pred))
    print('Overfitting: \n')
    print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))
    print('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))
    print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))
    print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))
    print('---' * 45)


# In[ ]:


log_reg = LogisticRegression()
knears_neighbors = KNeighborsClassifier()
svc = SVC()
tree_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()


# In[ ]:


#https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    # First Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    
    # Second Estimator 
    train_sizes, train_scores, test_scores = learning_curve(
        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")
    
    # Third Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)
    ax3.set_xlabel('Training size (m)')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend(loc="best")
    
    # Fourth Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)
    ax4.set_xlabel('Training size (m)')
    ax4.set_ylabel('Score')
    ax4.grid(True)
    ax4.legend(loc="best")
    return plt


# In[ ]:


cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)


# **Lets compare with Over-sampling method(RandomOverSampler)**

# In[ ]:


from imblearn.over_sampling import RandomOverSampler


# In[ ]:


from collections import Counter
over_y = train['Class']
over_X = train.drop('Class', axis = 1)
ros = RandomOverSampler(random_state=42)
X_over, y_over = ros.fit_resample(over_X,over_y)


# In[ ]:


from collections import Counter
print(sorted(Counter(y_over).items())) 


# In[ ]:


plt.subplots(figsize =(14,14))
corr = pd.DataFrame(X_over).corr()
sns.heatmap(corr)


# In[ ]:


X_train, X_test, y_train, y_test = tts(X_over, y_over, test_size=0.2, random_state=42)


# In[ ]:


classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
}


# In[ ]:


for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    y_pred = cross_val_predict(classifier, X_train, y_train, cv=5)
    cf = confusion_matrix(y_train, y_pred)
    precision, recall, threshold = precision_recall_curve(y_train, y_pred)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
    print("Confusion matrix:\n",cf)
    print('Area under ROC curve score: ', roc_auc_score(y_train, y_pred))
    print('Overfitting: \n')
    print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))
    print('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))
    print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))
    print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))
    print('---' * 45)


# **Another Over- Sampling Method (SMOTE)**

# In[ ]:


from imblearn.over_sampling import SMOTE

X_smote, y_smote = SMOTE().fit_resample(over_X,over_y)
print(sorted(Counter(y_smote).items()))


# In[ ]:


X_train, X_test, y_train, y_test = tts(X_smote, y_smote, test_size=0.2, random_state=42)


# In[ ]:


for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    y_pred = cross_val_predict(classifier, X_train, y_train, cv=5)
    cf = confusion_matrix(y_train, y_pred)
    precision, recall, threshold = precision_recall_curve(y_train, y_pred)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
    print("Confusion matrix:\n",cf)
    print('Area under ROC curve score: ', roc_auc_score(y_train, y_pred))
    print('Overfitting: \n')
    print('Recall Score: {:.2f}'.format(recall_score(y_train, y_pred)))
    print('Precision Score: {:.2f}'.format(precision_score(y_train, y_pred)))
    print('F1 Score: {:.2f}'.format(f1_score(y_train, y_pred)))
    print('Accuracy Score: {:.2f}'.format(accuracy_score(y_train, y_pred)))
    print('---' * 45)

