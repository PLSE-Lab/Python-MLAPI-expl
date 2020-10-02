#!/usr/bin/env python
# coding: utf-8

# Authors: [Renan Costalonga](https://www.kaggle.com/rcmonteiro) and [Guilherme Rinaldo](https://www.kaggle.com/grinaldo)
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing 
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc,recall_score,precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

pd.set_option("display.max_columns", 111)
sns.set_palette("Set2")


# # Checking Dataset

# In[ ]:


df = pd.read_excel("/kaggle/input/covid19/dataset.xlsx")
df.head()


# In[ ]:


df.shape


# Counting number of NaNs in each column.

# In[ ]:


df.isna().sum()


# Checking colum names

# In[ ]:


df.columns.tolist()


# Checking duplicates

# In[ ]:


df['Patient ID'].nunique()


# Since the patient ID is irrelevant in this context, we will drop it.

# In[ ]:


df.drop(columns='Patient ID', inplace=True)


# # Dataset Balance
# 
# A first glance shows that the dataset is quite imbalanced.

# In[ ]:


print('Negative: {} ({}%)'.format(df['SARS-Cov-2 exam result'].value_counts()[0], round(df['SARS-Cov-2 exam result'].value_counts()[0]/len(df)*100, 2)))
print('Positive: {} ({}%)'.format(df['SARS-Cov-2 exam result'].value_counts()[1], round(df['SARS-Cov-2 exam result'].value_counts()[1]/len(df)*100, 2)))
sns.countplot('SARS-Cov-2 exam result',data=df)


# Checking missing data regard patient admission.

# In[ ]:


df[['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)']].isna().sum()


# Cheking metrics about patients age.

# In[ ]:


df['Patient age quantile'].describe()


# Just mapping the target variable to 1s and 0s.

# In[ ]:


df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].map({'positive': 1, 'negative': 0})


# # Treating NaN values
# 
# In this section we treat some NaN values. First, we remove columns with only NaNs since they don't have any value for us. We end up with 105 columns, but there are still a lot of NaNs on these columns. 
# 
# We avoid to use common strategies for filling NaNs in this case, because any data inputation can easily lead to unrealistic values that do not correspond to those of a living human being. Instead, we choose to drop columns with more than 80% of NaNs. To get rid of the remaining NaNs, we drop rows that contains missing data. We end up with a dataset of shape (1352, 22).

# Dropping columns with only NaN values

# In[ ]:


drop_index = []
for i in range(df.shape[1]):
    if df.iloc[:,i].isna().sum() == len(df):
        drop_index.append(df.iloc[:,i].name)
        
for j in drop_index:
    df = df.drop([j],axis=1)


# Drop columns with more than 20% NaN values 

# In[ ]:


df = df.dropna(thresh=0.20*len(df), axis=1)


# Drop rows with any Nan value

# In[ ]:


df = df.dropna(axis=0)


# Dataset shape after dropout

# In[ ]:


df.shape


# # Feature analysis

# Analisying type of data in each column

# In[ ]:


df.dtypes.value_counts()


# Analysing unique categories in each column

# In[ ]:


df.select_dtypes(['float64','object','int64']).apply(pd.Series.nunique, axis = 0)


# Checking categorical variables

# In[ ]:


categorical_variables = df.select_dtypes(['object'])
categorical_variables = categorical_variables.columns
categorical_variables.tolist()


# Encoding labels of categorical data.
# * detected = 1 
# * not_detected = 0

# In[ ]:


for i in categorical_variables:
    le = preprocessing.LabelEncoder()
    le.fit(df[i].values)
    df[i] = le.transform(df[i].values)


# # Correlation

# In this section we study the correlation between each features.

# In[ ]:


#Correlation between features
corr = df.corr('pearson')

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# # Feature Importance
# 
# In this section we study the importance of each feature to the target variable. Using the chi-squared statistical test we try to identify the top 10 most important features among the remaining ones. We see that the patient age quartile and patient admittance to the ward points out as the most important features. This may be an indicative of the expertise of doctors identifying more dangerous COVID-19 cases early on, so that a high incidence of patients admited to the ward tests positive for the virus.\
# 
# Reference: https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
# 

# In[ ]:


X = df.loc[:, df.columns != 'SARS-Cov-2 exam result']
y = df['SARS-Cov-2 exam result']

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  
featureScores = featureScores.nlargest(10, 'Score')

sns.barplot(x="Score", y="Specs", data=featureScores)
featureScores


# # Balancing the Dataset
# 
# We still have a imbalanced dataset, with abou 8% of positive cases for the virus against almost 92% of negative cases.
# 
# Again, in order to avoid unrealistic entries to the dataset, we choose a strategie of random undersampling. We end up with a 50/50 balanced dataset with 224 entries.

# In[ ]:


print('Negative: {} ({}%)'.format(df['SARS-Cov-2 exam result'].value_counts()[0], round(df['SARS-Cov-2 exam result'].value_counts()[0]/len(df)*100, 2)))
print('Positive: {} ({}%)'.format(df['SARS-Cov-2 exam result'].value_counts()[1], round(df['SARS-Cov-2 exam result'].value_counts()[1]/len(df)*100, 2)))
sns.countplot('SARS-Cov-2 exam result',data=df)


# In[ ]:


rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))


# Comparing dataset after undersampling

# In[ ]:


print('Negative: {} ({}%)'.format(y_res.value_counts()[0], round(y_res.value_counts()[0]/len(y_res)*100, 2)))
print('Positive: {} ({}%)'.format(y_res.value_counts()[1], round(y_res.value_counts()[1]/len(y_res)*100, 2)))
sns.countplot(y_res)


# # Training Strategy and Modelling
# 
# We chose to undersample our dataset in order to garantee only realistc data for each patient. The downside for this approach is the small number of entires left, which can lead a model to overfit.
# 
# To prevent this behaviour, we use a cross-validation with a high number of folds strategy. Since our dataset is now smaller than the original, this approach becames computationally feseable and makes quite harder for our model to overfit.
# 
# For the model, we choose the GradientBoosting Classifier for its simplicity and robustness. Since we are dealing with medical data we try to maximize the recall metric so that every patient that is positive for the virus can be identified, even at the cost of some false positives.
# 
# Others machine learning algorithms are also tested, with similar or inferior behaviour.
# 
# Reference: https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.1, random_state=42,)


# In[ ]:


lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 1, 1.25, 1.5, 1.75, 2]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))


# In[ ]:


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[ ]:


gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=10, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_test)

plot_confusion_matrix(confusion_matrix(y_test, predictions), target_names=['0', '1'], normalize=False)

print(classification_report(y_test, predictions))

scores = cross_validate(gb_clf2, X_train, y_train, cv=45, scoring=['precision','recall','roc_auc'])

print("Cross Validation Scores: ")
print('Precision: ', scores.get('test_precision').mean())
print('Recall: ', scores.get('test_recall').mean())
print('ROC_ACU: ', scores.get('test_roc_auc').mean())


# # Conclusions

# Our strategy led to a Precision of 74%, Recall of 88% and Roc_AuC of approximatelly 83%. Since our goal was to maximize recall, without loosing performance in Precision, we believe we have achieved a simple androbust classifier. 
# 
# The high number of folds used for cross-validation helps the model to not overfit and possibly generalize well for other datasets. We hope this contribution, as small as it is, can be useful to our community in the search for means to stop COVID-19.
# 
# 
# # #StaySafe
