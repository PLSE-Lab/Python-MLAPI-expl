#!/usr/bin/env python
# coding: utf-8

# **Fraud Detection with SVM**
# 
# 

# # Fraud detection principal components using SVM and undersampling for correcting unbalance
# 
# This Kernel is based on work by [Davide Vegliante](https://www.kaggle.com/davidevegliante/nn-for-fraud-detection#).
# Instead of a Neural Network, we will train an SVM, and later use a technique described by [Aneesha Bakharia](https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d) to inspect which of the unnamed features V1..V28 contributed most to the decision boundary / margin of the trained SVM. To increase our confidence that our results are sound, we use [Louis Headley's work](https://www.kaggle.com/louish10/anomaly-detection-for-fraud-detection/notebook) to visualize the probability distributions of various features.
# 
# todo: compare to [Variance Threshold](https://scikit-learn.org/stable/modules/feature_selection.html)
# 
# ## Dataset
# 
# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style
get_ipython().run_line_magic('matplotlib', 'inline')
import os

import warnings  
warnings.filterwarnings('ignore')


matplotlib.style.use('ggplot')
from cycler import cycler
color_palette = sns.color_palette()
color_palette[0], color_palette[1] = color_palette[1], color_palette[0]
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=color_palette)

# read the dataset and print five rows
original_dataset = pd.read_csv('../input/creditcard.csv')

dataset = original_dataset.copy()
print(dataset.head(5))


# Let's see how many examples and features our dataset contains. 

# In[ ]:


# count how many entry there are for every class
classes_count = pd.value_counts(dataset['Class'])

print("{} Bonafide examples\n{} Fraud examples".format(classes_count[0], classes_count[1]))

# classes_count is a Series. 
classes_count.plot(kind = 'bar')
plt.xlabel('Classes')
plt.ylabel('Frequencies')
plt.title('Fraud Class Hist')


# The Features `V1`..`V28` seem to be normalized. The creators of the dataset could not disclose what they represent, but did note that this features are already selected from a larger set using [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis). The (transaction) `Amount` and `Time`  are not normalized, however. SVM algorithms are not scale invariant, so it is highly recommended to scale these 2 remaing features.
# 
# The `Time` feature represents the number of seconds since data recording started at the moment the transaction was performed. The total dataset covers a timespan of two days. It might be interesting to replace this feature by two new features `Day` (having value 0 or 1) and `TimeOfDay` (having values between 0 and 1, where 0 is 00:00  and 1 is 23:59)
# 
# Todo: rescale the Time dimension. Removed for now.

# In[ ]:


# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
from sklearn.preprocessing import RobustScaler

# RobustScaler is less prone to outliers.
rob_scaler = RobustScaler()
dataset['Amount'] = rob_scaler.fit_transform(dataset['Amount'].values.reshape(-1,1))

# remove the Time Feature
dataset.drop(['Time'], axis = 1, inplace = True)

dataset.head(5)


# ###  Undersampling with ratio 1

# In[ ]:


X = dataset.loc[:, dataset.columns != 'Class' ]
y = dataset.loc[:, dataset.columns == 'Class' ]

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = 0, sampling_strategy = 1.0)

X_resampled, y_resampled = rus.fit_resample(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.20, stratify = y_resampled)

assert len(y_train[y_train == 1]) + len(y_test[y_test == 1]) == len(dataset[dataset.Class == 1])
print("train_set size: {} - Class0: {}, Class1: {}".format( len(y_train), len(y_train[y_train == 0]), len(y_train[y_train == 1]) ))
print("test_set size: {} - Class0: {}, Class1: {}".format( len(y_test), len(y_test[y_test == 0]), len(y_test[y_test == 1]) ))


# ### Train SVM Structure
# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

classifier = LinearSVC(dual=False)
classifier.fit(X_train, y_train.ravel())


# ### Inspect the coefficients

# In[ ]:


def plot_coefficients(classifier, feature_names, top_features=-1):
 if top_features == -1:
    top_features = len(feature_names)
    
 coef = classifier.coef_.ravel()
 abs_coef = np.abs(coef)
 top_coefficients = np.argsort(-abs_coef)[-top_features:]

 # create plot
 plt.clf()
 plt.figure(figsize=(15, 3))
 colors = [color_palette[c > 0] for c in coef[top_coefficients]]
 plt.bar(np.arange(top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(0, top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.title("Feature coefficients")
 plt.ylabel("Coefficient")

feature_names = list(X.columns.values)
plot_coefficients(classifier, feature_names)
plt.show()


# In[ ]:


dataset.groupby("Class")['V14', 'V15', 'V19'].describe(percentiles=[])


# In[ ]:


def remove_outliers(series):
    return series[np.abs(series-series.mean()) <= (1*series.std())]

for feature in ['V14', 'V15', 'V19']:
    ax = plt.subplot()
    positive = dataset[feature][dataset.Class == 1]
    negative = dataset[feature][dataset.Class == 0]

    sns.distplot(positive, bins=50, label='Fraudulent')
    sns.distplot(negative, bins=50, label='Bonafide')
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(feature))
    plt.legend(loc='best')
    plt.show()


# On first inspection, it seems that coefficient strength is indeed a good indicator for the importance of a feature. When the coefficient is positive, the feature is positively correlated with the output class. In other words: Examples with higher `V14` values are more likely to be fraudulent than lower values. Conversely, high values for `V19` are indicative for a bonafide transaction. However, the coefficient for `V19` is much weaker than the coefficient for `V14`, and thus we expect `V19` to have much less of an influence on our prediction. We also expect the probability distributions for Fraudulent and Bonafide transaction to overlap more for `V19` than for `V14`. The coefficient of `V15` is even weaker than that of `V19`. And indeed,, we see that there is larger overlap in the probabilities of fraudulent and bonafide transations for `V15` than for `V19`. All of these predictions were made based on the trained SVM coefficients, and all of them are in line with the probability distributions shown above.

# ## Report the classifier performance
# 
# Note that **preciosion and recall are unreliable in an unbalanced dataset**, but we used undersampling to account for this.

# In[ ]:


from sklearn.metrics import classification_report
y_test_pred = classifier.predict(X_test) > 0.5
target_names = ["Bonafide", "Fraudulent"]
print(classification_report(y_test, y_test_pred, target_names=target_names))


# ### Confusion Matrix

# In[ ]:



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)

plt.clf()
plt.grid('off')
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
classNames = target_names
plt.title('Fraud or Not Fraud Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.show()

