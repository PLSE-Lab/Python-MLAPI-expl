#!/usr/bin/env python
# coding: utf-8

# This kernel basin on the materials from:
# * https://scikit-learn.org/stable/modules/calibration.html
# * https://www.kaggle.com/martin1234567890/logistic-regression

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


# In[ ]:


submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")
train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

labels = train.pop('target')
train_id = train.pop("id")
test_id = test.pop("id")


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


labels = labels.values


# In[ ]:


train.head(5)


# In[ ]:


#Thanks to https://www.kaggle.com/martin1234567890/logistic-regression
data = pd.concat([train, test])
data["ord_5a"] = data["ord_5"].str[0]
data["ord_5b"] = data["ord_5"].str[1]
data.drop(["bin_0", "ord_5"], axis=1, inplace=True)


# In[ ]:


# Determination categorical features
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = data.columns.values.tolist()
for col in features:
    if data[col].dtype in numerics: continue
    categorical_columns.append(col)
    
# Encoding categorical features
for col in categorical_columns:
    if col in data.columns:
        le = LabelEncoder()
        le.fit(list(data[col].astype(str).values))
        data[col] = le.transform(list(data[col].astype(str).values))


# In[ ]:


train = data.iloc[:train.shape[0], :]
test = data.iloc[train.shape[0]:, :]


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.info()


# In[ ]:


train = train.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.99,
                                                    random_state=42)


# In[ ]:


def model_pred(clf, X_test):
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos =         (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    return prob_pos


# In[ ]:


#Thanks to https://scikit-learn.org/stable/modules/calibration.html
def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        prob_pos = model_pred(clf, X_test) 
        clf_score = brier_score_loss(y_test, prob_pos, pos_label=1)
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value =             calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)
        if name == "Naive Bayes + Sigmoid":
            test_pred = model_pred(clf, test)
            submission["id"] = test_id
            submission["target"] = clf.predict_proba(test)[:, 1]
            submission.to_csv("submission.csv", index=False)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


# In[ ]:


# Plot calibration curve for Gaussian Naive Bayes
plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)


# In[ ]:


# Plot calibration curve for Linear SVC
plot_calibration_curve(LinearSVC(max_iter=10000), "SVC", 2)

plt.show()

