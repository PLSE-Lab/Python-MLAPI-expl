#!/usr/bin/env python
# coding: utf-8

# ## Detecting Credit Card Fraud 
# 
# This dataset contains a signifcant class imbalance where only 492 of the almost 285,000 entries are labeled as having had credit card fraud. This huge imbalance brings up challenges when trying to accurately predict whether or not a transaction is fradulent. This notebook will take a quick look at using a sampling method called SMOTE, or Synthetic Minority Oversampling Technique, as a way to generate new observations from the minority class in hopes of helping the performance of the model. SMOTE constructs synthetic observations using k-Nearest Neighbors. 

# In[17]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


fraud_df = pd.read_csv("../input/creditcard.csv")
fraud_df.head()


# Quick data checks...

# In[3]:


fraud_df.info()


# In[4]:


fraud_df.describe()


# Talk about class imbalance!!!

# In[5]:


print(fraud_df['Class'].value_counts())
sns.countplot(x = 'Class', data = fraud_df)
plt.show()


# This dataset contains two days worth of data and no time analysis will be done here so I'm going to drop the time column...

# In[6]:


fraud_df.drop(columns = 'Time', inplace = True)


# Cross validation will be used to find optimal parameters for the regularized logistic model so the dataset is split into train and a hold off test set where the train set will be used for further cross validation. With imbalanced datasets, we can guarantee that our train-test splits have equivalent proportions of the minority class, which we do by setting the stratify parameter. 

# In[7]:


y = fraud_df.pop('Class').values
X = fraud_df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)


# The model that will be tested is a regularized logistic regression algorithm. It is important that we scale the features because regularization penalizes the magnitude of the beta coefficients. If we don't standardize, features on smaller scales in turn will more likely by nature have larger coefficient values than features on larger scales and we don't want our regularization to have that bias. 

# In[14]:


scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# Quick look at minority proportions in train and test sets...

# In[8]:


y_train.sum() / len(y_train)


# In[9]:


y_test.sum() / len(y_test)


# Below is a function that will perform gridsearch for a set of hyperparameters. In this case, since I'm using regularization, the hyperparameters will be L1 (Lasso) or L2 (Ridge) regularization, and the regularization strength parameter, C, which affects the complexity of the model and the penalization on the beta coefficients. The function will print the average cross validation score and the best parameters chosen. The model with the best parameters fit on the ENTIRE train set will be returned so we can perform some additional model evaluation after. 

# In[10]:


def get_opt_model(clf, param_grid, X_train, y_train, scoring = 'accuracy'):
    '''
    PARAMETERS:
    clf(classifier): classifier to run GridSearch on
    param_grid(dict): dictionary of parameters to test in GridSearch
    X_train(array): array of predictor variables
    y_train(array): array of response variable
    scoring: score to maximize during GridSearch, default to accuracy
    PRINTS: best training score and best parameter values determined by GridSearch
    RETURNS: fitted model (based on entire train set) with best parameter values
    '''

    best_clf = GridSearchCV(clf, param_grid, scoring = scoring, cv = 5, n_jobs = -1)
    best_clf.fit(X_train, y_train)
    print(best_clf.best_score_)
    print(best_clf.best_params_)
    return best_clf.best_estimator_


# Accuracy is an inappropriate metric to evaluate when we're dealing with a class imbalance, especially like the one that we've seen here.  The credit card company, within reason, would rather classify a non-fraudulent transaction as fraudulent (and investigate it later to find out it was a false alarm) than miss an actual fraudulent case. When we take this into consideration, recall, or the true positive rate, is the score we should be evaluating our model with. 
# 
# Again, the SMOTE algorithm randomly creates new minority classes by generating features that are a weighted combination of the features of a minority sample and the features of its kth nearest neighbors. With cross validation, it is important that that train-validation sets are created prior to implementing SMOTE so that the synthetic points are created using only the training portion and no information from the validation set is leaking in. I was having issues getting this to work, but luckily I ran into this [link](https://stackoverflow.com/questions/48370150/how-to-implement-smote-in-cross-validation-and-gridsearchcv) on stackoverflow to add SMOTE correctly into the pipeline. Again, my function returns the optimal model fitted on the entire training set, with SMOTE applied too I believe. The optimal model is one that uses ridge regularization with 0.01 as the value for C. 
# 

# In[15]:


smt = SMOTE()
log_model = LogisticRegression()

steps = [('smote', smt), ('mod', log_model)]
pipeline = Pipeline(steps)

c_param_grid = {'mod__C': [0.01, 0.1, 1, 10, 100], 'mod__penalty': ['l1', 'l2']}
best_mod = get_opt_model(pipeline, c_param_grid, X_train_sc, y_train, scoring = 'recall')


# Here, I evaluate the chosen model on the hold out test set. I also calculate the recall of the training set to take a look at any overfitting, which there is, but having some overfitting is typical as the model is being trained to optimize the training set. The recall score on the test set is still pretty high i.e. not significantly lower than recall on the training set. 

# In[16]:


fraud_pred_train = best_mod.predict(X_train_sc)
fraud_pred_test = best_mod.predict(X_test_sc)
recall_train = recall_score(y_train, fraud_pred_train)
recall_test = recall_score(y_test, fraud_pred_test)
print('Recall - Training set: ', recall_train)
print('Recall - Test set: ', recall_test)


# Confusion matrix with/scikit-learn's implementation. The count of true negatives is C_{0,0}, false negatives is C_{1,0}, true positives is C_{1,1} and false positives is C_{0,1}. Of the 123 entries in the test dataset that are labeled fraud, the model was able to correctly identify 110 of them. However, the precision is very low (0.05869797), which might not be acceptable in the real world. We'll take a look at adjusting the thresholds to which we label a prediction as fraud or not to get the precision score up. 

# In[20]:


print(confusion_matrix(y_test, fraud_pred_test))


# By default, the algorithm will label a prediction as fraud if its predicted probability is greater than 0.5. We can adjust this probability threshold and label a prediction as fraud at a different probability level. From there, we can see how the recall and precision scores are affected. 
# 
# From the output below, we can see that as the threshold increases, the recall starts to drop and the precision rises. What's interesting to note is that the recall doesn't drop drastically. Thinking about it, that must mean that the probabilities for the fraudulent transactions predicted by the model are relatively high. That is, the probabilities for predicted fraud are high enough that increasing the probability threshold does not cause a prediction to go from being one that's correctly predicted as fraud to one that's incorrectly predicted as not fraud. So our model must have pretty good certainty that a transaction is fraud when it is indeed fraud. So depending on what levels of recall and precision we're okay at can help us determine what probability threshold to use. I'm also still not sure if our highest 0.23 precision score at our highest threshold is acceptable...

# In[44]:


pos_probs = best_mod.predict_proba(X_test_sc)[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for t in thresholds: 
    fraud_pred = pos_probs > t
    confusion = confusion_matrix(y_test, fraud_pred)
    print('Recall for threshold, ', t, ' is:', confusion[1,1]/(confusion[1,0] + confusion[1,1]))
    print('Precision for threshold, ', t, ' is:', confusion[1,1]/(confusion[0,1] + confusion[1,1]))
    print('-----------------------')


# Here I mostly focused on using the sampling method SMOTE to help create a model that accurately predicts credit card fraud when we have a significantly imbalanced dataset. There are other sampling methods such as undersampling and oversampling. Another way to combat imbalance is by using a profit curve, which maybe someday I'll revisit. 
# 
# As always, constructive comments are welcome!! Please let me know if you see anything incorect in my methodology, etc. 

# In[ ]:




