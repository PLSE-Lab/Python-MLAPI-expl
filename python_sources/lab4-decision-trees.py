#!/usr/bin/env python
# coding: utf-8

# # Morgan Dally - 1313361

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt


# In[ ]:


# CONSTS USED
# pinned random state
RANDOM_STATE = 1313361

RFC = 'RandomForestClassifier'
ETC = 'ExtraTreesClassifier'


# In[ ]:



train = pd.read_csv('../input/fashion-mnist_train.csv', dtype=int)
X_train = train.drop('label', axis=1)
y_train = train['label']
test = pd.read_csv('../input/fashion-mnist_test.csv', dtype=int)
X_test = test.drop('label', axis=1)
y_test = test['label']


# In[ ]:


def plot_matrix(matrix, title='', use_sns=True):
    '''
    Creates a heatmap of the input matrix.

    Arguments:
        matrix: The matrix to plot. Must have a width and
            a height defined when matrix.shape is called.
        title: If provided, heatmap title will be set to
            this value.
        use_sns: If True, heatmap will be created
            using seaborn. If False heatmap will be created
            using matplotlib. Default True.
    '''
    fig = plt.figure(figsize=matrix.shape)

    if title:
        plt.suptitle(title)

    # use matplotlib instead
    if not use_sns:
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        return

    sns.heatmap(matrix, annot=True)

X_matrix = X_train[0:1].values.reshape(28,28)
plot_matrix(X_matrix, title='GaussianNB Matrix Heatmap')


# In[ ]:


# get an initial benchmark on how "hard" the dataset is
# fit and redict using GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def test_classifier(CLF, X_train, X_test, y_train):
    '''Fits and predicts using input params'''
    classifier = CLF()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    return classifier, predictions

naive_bayes, y_pred = test_classifier(GaussianNB, X_train, X_test, y_train)
model_accuracy = accuracy_score(y_test, y_pred)
print('Benchmark model accuracy: %.2f%%' % (model_accuracy * 100))


# In[ ]:


import itertools

def get_ft_depth_pairs():
    '''
    Creates a list with all of the pairings of feature_values
    and depth_values
    '''
    feature_values = [1, 4, 16, 64,'auto']
    depth_values = [1, 4, 16, 64, None]
    pairing = [
        feature_values,
        depth_values
    ]
    return list(itertools.product(*pairing))

print(get_ft_depth_pairs())


# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import time

def grid_search(CLF, X_train, y_train, n_estimators=30):
    '''
    Uses all possible pairs from get_ft_depth_pairs
    to find the best hyper params for the input
    classifier.
    '''
    possible_pairs = get_ft_depth_pairs()

    for pairing in possible_pairs:
        features_paramter = pairing[0]
        max_depth_parameter = pairing[1]

        # bootstrap=true to get oob_score_ for ExtraTrees
        # oob_score=true to get oob_score_ for RandomForest
        classifier = CLF(
            max_features=features_paramter,
            max_depth=max_depth_parameter,
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            oob_score=True,
            bootstrap=True
        )

        start = time.time()
        classifier.fit(X_train, y_train)
        end = time.time()
        oob_score = classifier.oob_score_

        print('max_features=%s max_depth=%s oob_score=%f execution_time=%.2fs' % (
            features_paramter, max_depth_parameter, oob_score, end-start)
        )

classifiers = [RandomForestClassifier, ExtraTreesClassifier]
for classifier in classifiers:
    print('running %s for best hyper_params' % str(classifier))
    grid_search(classifier, X_train, y_train)
    
        


# # Best model parameters:
# ## RandomForestClassifier
# * max_features=auto max_depth=16 86.1683% accuracy at 33.59s
# * max_features=4 max_depth=None 84.2833% accuracy at 9.84s   <- I beleive this is the optimal model as due to it's effeciency
# * max_features=64 max_depth=16 86.53% accuracy at 69.92s
# * max_features=64 max_depth=None 86.6183% accuracy at 91.30s  <- Best overall accuracy, however slowest to train
# 
# ## ExtraTreesClassifier
# * max_features=64 max_depth=64 85.9050% at 36.63s
# * max_features=64 max_depth=16 85.8150% at 30.61s
# * max_features=16 max_depth=None 84.5417% at 13.38s <- Optimal model
# 
# 

# In[ ]:



from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# best models, trained
optimal_rfc = RandomForestClassifier(
    max_features=4, max_depth=None, random_state=RANDOM_STATE, n_estimators=30
)
optimal_rfc.fit(X_train, y_train)

optimal_etc = ExtraTreesClassifier(
    max_features=16, max_depth=64, random_state=RANDOM_STATE, n_estimators=100
)
optimal_etc.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import confusion_matrix

def get_results(trained_clf, X_test, y_test):
    '''Predicts using X_test, prints accuracy and confusion matrix.'''
    model_pred = trained_clf.predict(X_test)
    model_acc = accuracy_score(y_test, model_pred)
    cm = confusion_matrix(y_test, model_pred)

    print('model acheived %.2f%% accuracy\ncr: %s' % (
        (model_acc * 100), cm)
    )
    return model_pred, cm

print('getting accuracy and cm for ' + RFC)
rfc_pred, rfc_cm = get_results(optimal_rfc, X_test, y_test)

print()

print('getting accuracy and cm for ' + ETC)
etc_pred, etc_cm = get_results(optimal_etc, X_test, y_test)


# In[ ]:


FIG_SIZE = (10, 7)
ANNOT = True
matricies = {
    RFC: rfc_cm,
    ETC: etc_cm,
}

for model_name in matricies:
    cm = matricies[model_name]
    plt.figure(figsize=FIG_SIZE)
    plt.suptitle(model_name)
    sns.heatmap(cm, annot=ANNOT)


# In[ ]:


column_names = list(X_train)

rfc_ft_impt = optimal_rfc.feature_importances_.reshape(28, 28)
etc_ft_impt = optimal_etc.feature_importances_.reshape(28, 28)

plot_matrix(rfc_ft_impt, title='RandomForestClassifier Feature Importance')


# In[ ]:


plot_matrix(etc_ft_impt, title='ExtraTreesClassifier Feature Importance')


# # Questions
# ## 1. Regarding OOB scores, is the RandomForestClassifier better than the ExtraTreesClassifier?
# ### scores:
# #### RandomForestClassifier (RFC)
# * max_features=auto max_depth=16 86.1683% accuracy at 33.59s
# * max_features=4 max_depth=None 84.2833% accuracy at 9.84s   <- I beleive this is the optimal model as due to it's effeciency // choosen model
# * max_features=64 max_depth=16 86.53% accuracy at 69.92s
# * max_features=64 max_depth=None 86.6183% accuracy at 91.30s  <- Best overall accuracy, however slowest to train
# 
# #### ExtraTreesClassifier (ETC)
# * max_features=64 max_depth=64 85.9050% at 36.63s   <- Best Accuracy
# * max_features=64 max_depth=16 85.8150% at 30.61s
# * max_features=16 max_depth=None 84.5417% at 13.38s <- Optimal model
# 
# ### Answer
# When comparing effeciency, RFC when used with max_features=4 max_depth=None acheives 84.2833% oob score taking 9.84s compared to ETC with a similar model at 84.5417% oob score at 13.38s when using max_features=16 max_depth=None. I beleive both models both models are viable, with ETC having a slightly higher oob score but taking 35% longer. For overall effeciency i'd go with RFC.
# 
# When comparing top accuracy RFC when using max_features=64, max_depth=None acheives 86.6183% oob score, however it also took 91.30s to run. ETC on the other hand when using max_features=64, max_depth=64 achieves 85.9050% but only taking 36.63s. I beleive ETC to be the best model for overall oon score + time effeciency.
# 
# ## 2. Is the OOB score for the RandomForestClassifier lower than its test-set accuracy?
# ### Score + Actual Comparison
# #### Comparison
# Out of bag score: 84.2833%
# Acutal prediction: 86.15%
# 
# ### Answer
# No the out of bag score was slighlty lower. This is generally the case with out of bag scores, this is due to the oob set being slighty "harder" as it has more variation. This arrises from the how the set is created, as its created as a subset of the train set that was not used during bagging.
# 
# ## 3. Is the OOB score for the ExtraTreesClassifier lower than its test-set accuracy?
# ### Score + Actual Comparison
# #### Comparison
# Out of bag score: 84.5417%
# Actual prediction: 87.89%
# 
# ### Answer
# Actual prediction is more accuracte than the out of bag score.
# 
# ## 4. Do the feature_importances matrix plots make sense (explain)?
# ### Answer
# Looking at the GitHub[https://github.com/zalandoresearch/fashion-mnist] for what the example images look like, the feature important heatmap does look like it makes sense. The line down the middle of the graphs would be important as a tshirt will have direct colouration through the center, pants will have most of it left blank and shoes will be a mixture of both. Towards the top of the heatmap the importance looks to be higher. I beleive this would be due to the images such as the shoes and the handbags. Alot of the handbags take up around 2/3 of the image, with the top 1/3 having less features. This could be a determining factor for whether something is a tshirt or a handbag for example. Towards the top-left of the heatmaps there is more importance, when classifying between a dress and a tshirt this area could play into it as a dress would genr
# 
