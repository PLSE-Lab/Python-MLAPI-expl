#!/usr/bin/env python
# coding: utf-8

# # Methods for Dealing with Imbalanced Data
# Imbalanced classes are a common problem in machine learning classification where there are a disproportionate ratio of observations in each class.  Class imbalance can be found in many different areas including medical diagnosis, spam filtering, and fraud detection.
# 
# In this guide, we'll look at five possible ways to handle an imbalanced class problem using credit card data.  Our objective will be to correctly classify the minority class of fraudulent transactions.
# 
# Important Note:
# This guide will focus soley on addressing imbalanced classes and will not addressing other important machine learning steps including, but not limited to, feature selection or hyperparameter tuning.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score


# In[ ]:


# setting up default plotting parameters
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = [20.0, 7.0]
plt.rcParams.update({'font.size': 22,})

sns.set_palette('viridis')
sns.set_style('white')
sns.set_context('talk', font_scale=0.8)


# In[ ]:


# read in data
df = pd.read_csv('../input/creditcard.csv')

print(df.shape)
df.head()


# In[ ]:


print(df.Class.value_counts())


# In[ ]:


# using seaborns countplot to show distribution of questions in dataset
fig, ax = plt.subplots()
g = sns.countplot(df.Class, palette='viridis')
g.set_xticklabels(['Not Fraud', 'Fraud'])
g.set_yticklabels([])

# function to show values on bars
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
show_values_on_bars(ax)

sns.despine(left=True, bottom=True)
plt.xlabel('')
plt.ylabel('')
plt.title('Distribution of Transactions', fontsize=30)
plt.tick_params(axis='x', which='major', labelsize=15)
plt.show()


# In[ ]:


# print percentage of questions where target == 1
(len(df.loc[df.Class==1])) / (len(df.loc[df.Class == 0])) * 100


# From the plot above, we can see we have a very imbalanced class -  just 0.17% of our dataset belong to the target class!
# 
# This is a problem because many machine learning models are designed to maximize overall accuracy, which especially with imbalanced classes may not be the best metric to use. Classification accuracy is defined as the number of correct predictions divided by total predictions times 100. For example, if we simply predicted all transactions are not fraud, we would get a classification acuracy score of over 99%!
# 
# ### Create Train and Test Sets
# 
# The training set is used to build and validate the model, while the test set is reserved for testing the model on unseen data.

# In[ ]:


# Prepare data for modeling
# Separate input features and target
y = df.Class
X = df.drop('Class', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)


# ## Baseline Models

# In[ ]:


# DummyClassifier to predict only target 0
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

# checking unique labels
print('Unique predicted labels: ', (np.unique(dummy_pred)))

# checking accuracy
print('Test score: ', accuracy_score(y_test, dummy_pred))


# As predicted our accuracy score for classifying all transactions as not fraud is 99.8%!  
# 
# As the Dummy Classifier predicts only Class 0, it is clearly not a good option for our objective of correctly classifying fraudulent transactions.
# 
# Let's see how logistic regression performs on this dataset.

# In[ ]:


# Modeling the data as is
# Train model
lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)
 
# Predict on training set
lr_pred = lr.predict(X_test)


# In[ ]:


# Checking accuracy
accuracy_score(y_test, lr_pred)


# In[ ]:


# Checking unique values
predictions = pd.DataFrame(lr_pred)
predictions[0].value_counts()


# Logistic Regression outperformed the Dummy Classifier!  We can see that it predicted 94 instances of class 1, so this is definitely an improvement.  But can we do better?
# 
# Let's see if we can apply some techniques for dealing with class imbalance to improve these results.
# 
# ## 1.  Change the performance metric
# Accuracy is not the best metric to use when evaluating imbalanced datasets as it can be misleading.  Metrics that can provide better insight include:
#  - **Confusion Matrix:**  a talbe showing correct predictions and types of incorrect predictions.
#  - **Precision: **  the number of true positives divided by all positive predictions. Precision is also called Positive Predictive Value. It is a measure of a classifier's exactness. Low precision indicates a high number of false positives.
#  - **Recall:**  the number of true positives divided by the number of positive values in the test data. Recall is also called Sensitivity or the True Positive Rate. It is a measure of a classifier's completeness. Low recall indicates a high number of false negatives.
#  - **F1: Score:**  the weighted average of precision and recall.
#  
# Since our main objective with the dataset is to prioritize accuraltely classifying fraud cases the recall score can be considered our main metric to use for evaluating outcomes.
# 

# In[ ]:


# f1 score
f1_score(y_test, lr_pred)


# In[ ]:


# confusion matrix
pd.DataFrame(confusion_matrix(y_test, lr_pred))


# In[ ]:


recall_score(y_test, lr_pred)


# We have a very high accuracy score of 0.999 but a F1 score of only 0.752.  And from the confusion matrix, we can see we are misclassifying several observations leading to a recall score of only 0.64.
# 
# ## 2. Change the algorithm
# While in every machine learning problem, its a good rule of thumb to try a variety of algorithms, it can be especially beneficial with imbalanced datasets.  Decision trees frequently perform well on imbalanced data.  They work by learning a hierachy of if/else questions.  This can force both classes to be addressed.
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# train model
rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

# predict on test set
rfc_pred = rfc.predict(X_test)

accuracy_score(y_test, rfc_pred)


# In[ ]:


# f1 score
f1_score(y_test, rfc_pred)


# In[ ]:


# confusion matrix
pd.DataFrame(confusion_matrix(y_test, rfc_pred))


# In[ ]:


# recall score
recall_score(y_test, rfc_pred)


# # Resampling Techniques
# 
# ## 3. Oversampling Minority Class
# Oversampling can be defined as adding more copies of the minority class.  Oversampling can be a good choice when you don't have a ton of data to work with.  A con to consider when undersampling is that it can cause overfitting and poor generalization to your test set.
# 
# We will use the resampling module from Scikit-Learn to randomly replicate samples from the minority class.
# 
# ### **Important Note**
# Always split into test and train sets BEFORE trying any resampling techniques!  Oversampling before splitting the data can allow the exact same observations to be present in both the test and train sets!  This can allow our model to simply memorize specific data points and cause overfitting.

# In[ ]:


from sklearn.utils import resample


# In[ ]:


# Separate input features and target
y = df.Class
X = df.drop('Class', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)


# In[ ]:


# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)
X.head()


# In[ ]:


# separate minority and majority classes
not_fraud = X[X.Class==0]
fraud = X[X.Class==1]

# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
upsampled.Class.value_counts()


# In[ ]:


# trying logistic regression again with the balanced dataset
y_train = upsampled.Class
X_train = upsampled.drop('Class', axis=1)

upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

upsampled_pred = upsampled.predict(X_test)


# In[ ]:


# Checking accuracy
accuracy_score(y_test, upsampled_pred)


# In[ ]:


# f1 score
f1_score(y_test, upsampled_pred)


# In[ ]:


# confusion matrix
pd.DataFrame(confusion_matrix(y_test, upsampled_pred))


# In[ ]:


recall_score(y_test, upsampled_pred)


# Our accuracy score decreased after upsampling, but the model is now predicting both classes more equally, making it an improvement over our plain logistic regression above.
# 
# ## 4. Undersampling Majority Class
# Undersampling can be defined as removing some observations of the majority class.  Undersampling can be a good choice when you have a ton of data -think millions of rows.  But a drawback to undersampling is that we are removing information that may be valuable.
# 
# We will again use the resampling module from Scikit-Learn to randomly remove samples from the majority class.

# In[ ]:


# still using our separated classes fraud and not_fraud from above

# downsample majority
not_fraud_downsampled = resample(not_fraud,
                                replace = False, # sample without replacement
                                n_samples = len(fraud), # match minority n
                                random_state = 27) # reproducible results

# combine minority and downsampled majority
downsampled = pd.concat([not_fraud_downsampled, fraud])

# checking counts
downsampled.Class.value_counts()


# In[ ]:


# trying logistic regression again with the undersampled dataset

y_train = downsampled.Class
X_train = downsampled.drop('Class', axis=1)

undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)

undersampled_pred = undersampled.predict(X_test)


# In[ ]:


# Checking accuracy
accuracy_score(y_test, undersampled_pred)


# In[ ]:


# f1 score
f1_score(y_test, undersampled_pred)


# In[ ]:


# confusion matrix
pd.DataFrame(confusion_matrix(y_test, undersampled_pred))


# In[ ]:


recall_score(y_test, undersampled_pred)


# Downsampling produced a higher recall score than upsampling!  My concern here is the small number of total samples we used to train the model.
# 

# ## 5. Generate Synthetic Samples
# SMOTE or Synthetic Minority Oversampling Technique is a popular algorithm to creates sythetic observations of the minority class.

# In[ ]:


from imblearn.over_sampling import SMOTE

# Separate input features and target
y = df.Class
X = df.drop('Class', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)


# In[ ]:


smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)

smote_pred = smote.predict(X_test)

# Checking accuracy
accuracy_score(y_test, smote_pred)


# In[ ]:


# f1 score
f1_score(y_test, smote_pred)


# In[ ]:


# confustion matrix
pd.DataFrame(confusion_matrix(y_test, smote_pred))


# In[ ]:


recall_score(y_test, smote_pred)


# ## Conclusion
# 
# We covered 5 different methods for dealing with imbalanced datasets:
# 1.  Change the performance metric
# 2.  Oversampling minority class
# 3.  Undersampling majority class
# 4.  Change the algorithm
# 5.  Generate synthetic samples
# 
# These are just some of the many possible methods to try when dealing with imbalanced datasets, and not an exhaustive list.  Some others methods to consider are collecting more data or choosing different resampling ratios - you don't have to have exactly a 1:1 ratio!  You should always try several approaches and then decide which is best for your problem.
