#!/usr/bin/env python
# coding: utf-8

# **Flavors of Physics**
# 
# This is a problem that is not easy to understand for someone who is not into Quantum Physics. Though a bit intriguing, the problem seems to be very interesting. 
# 
# In simple terms, as I could undertand, the scientists are trying to observe the decay of tau to three muons - a phenomenon that is very rare.  The null hypothesis is that this decay does not happen and the idea here is to see that the decay happens more than what is currently known.
# 
# The dataset consists of real and simulated events. Real events (or Background events or those with Signal value = 0) and Simulated Events (or Signal Events or those with Signal value = 1) are part of the dataset. The analysis is to be done on the entire dataset to classify the data based on various attributes. 
# 
# Before the algorithm is evaluated (based on AUC metric), the algorithm needs to pass through Agreement Test and Correlation Test. The code for these tests is provided along with the dataset.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
import featuretools as ft


# In[ ]:


def __rolling_window(data, window_size):
    """
    Rolling window: take window with definite size through the array

    :param data: array-like
    :param window_size: size
    :return: the sequence of windows

    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4
        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending

    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    target_distribution = numpy.arange(1, total_events + 1, dtype='float') / total_events
    subarray_distribution = numpy.cumsum(numpy.bincount(subindices, minlength=total_events), dtype='float')
    subarray_distribution /= 1.0 * subarray_distribution[-1]
    return numpy.mean((target_distribution - subarray_distribution) ** 2)


def compute_cvm(predictions, masses, n_neighbours=200, step=50):
    """
    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.
    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.

    :param predictions: array-like, predictions
    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass
    :param n_neighbours: count of neighbours for event to define mass bin
    :param step: step through sorted mass-array to define next center of bin
    :return: average cvm value
    """
    predictions = numpy.array(predictions)
    masses = numpy.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[numpy.argsort(masses)]

    # Second, replace probabilities with order of probability among other events
    predictions = numpy.argsort(numpy.argsort(predictions, kind='mergesort'), kind='mergesort')

    # Now, each window forms a group, and we can compute contribution of each group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return numpy.mean(cvms)


def __roc_curve_splitted(data_zero, data_one, sample_weights_zero, sample_weights_one):
    """
    Compute roc curve

    :param data_zero: 0-labeled data
    :param data_one:  1-labeled data
    :param sample_weights_zero: weights for 0-labeled data
    :param sample_weights_one:  weights for 1-labeled data
    :return: roc curve
    """
    labels = [0] * len(data_zero) + [1] * len(data_one)
    weights = numpy.concatenate([sample_weights_zero, sample_weights_one])
    data_all = numpy.concatenate([data_zero, data_one])
    fpr, tpr, _ = roc_curve(labels, data_all, sample_weight=weights)
    return fpr, tpr


def compute_ks(data_prediction, mc_prediction, weights_data, weights_mc):
    """
    Compute Kolmogorov-Smirnov (ks) distance between real data predictions cdf and Monte Carlo one.

    :param data_prediction: array-like, real data predictions
    :param mc_prediction: array-like, Monte Carlo data predictions
    :param weights_data: array-like, real data weights
    :param weights_mc: array-like, Monte Carlo weights
    :return: ks value
    """
    assert len(data_prediction) == len(weights_data), 'Data length and weight one must be the same'
    assert len(mc_prediction) == len(weights_mc), 'Data length and weight one must be the same'

    data_prediction, mc_prediction = numpy.array(data_prediction), numpy.array(mc_prediction)
    weights_data, weights_mc = numpy.array(weights_data), numpy.array(weights_mc)

    assert numpy.all(data_prediction >= 0.) and numpy.all(data_prediction <= 1.), 'Data predictions are out of range [0, 1]'
    assert numpy.all(mc_prediction >= 0.) and numpy.all(mc_prediction <= 1.), 'MC predictions are out of range [0, 1]'

    weights_data /= numpy.sum(weights_data)
    weights_mc /= numpy.sum(weights_mc)

    fpr, tpr = __roc_curve_splitted(data_prediction, mc_prediction, weights_data, weights_mc)

    Dnm = numpy.max(numpy.abs(fpr - tpr))
    return Dnm


def roc_auc_truncated(labels, predictions, tpr_thresholds=(0.2, 0.4, 0.6, 0.8),
                      roc_weights=(4, 3, 2, 1, 0)):
    """
    Compute weighted area under ROC curve.

    :param labels: array-like, true labels
    :param predictions: array-like, predictions
    :param tpr_thresholds: array-like, true positive rate thresholds delimiting the ROC segments
    :param roc_weights: array-like, weights for true positive rate segments
    :return: weighted AUC
    """
    assert numpy.all(predictions >= 0.) and numpy.all(predictions <= 1.), 'Data predictions are out of range [0, 1]'
    assert len(tpr_thresholds) + 1 == len(roc_weights), 'Incompatible lengths of thresholds and weights'
    fpr, tpr, _ = roc_curve(labels, predictions)
    area = 0.
    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
    for index in range(1, len(tpr_thresholds)):
        tpr_cut = numpy.minimum(tpr, tpr_thresholds[index])
        tpr_previous = numpy.minimum(tpr, tpr_thresholds[index - 1])
        area += roc_weights[index - 1] * (auc(fpr, tpr_cut, reorder=True) - auc(fpr, tpr_previous, reorder=True))
    tpr_thresholds = numpy.array(tpr_thresholds)
    # roc auc normalization to be 1 for an ideal classifier
    area /= numpy.sum((tpr_thresholds[1:] - tpr_thresholds[:-1]) * numpy.array(roc_weights))
    return area


# In[ ]:


folder = "../input/"
train = pd.read_csv(folder+'training.csv', index_col='id')


# In[ ]:


train.head()


# In[ ]:


test = pd.read_csv(folder + 'test.csv', index_col='id')
test.shape


# Let's try to visualise the data - it always helps to see if there is some apparent relationship or anomaly in the dataset.

# In[ ]:


# visualize the relationship between the features and the response using scatterplots
sns.pairplot(train, x_vars=['LifeTime','dira','FlightDistance','FlightDistanceError'], y_vars='signal', size=7, aspect=0.7)


# In[ ]:


sns.pairplot(train, x_vars=['LifeTime','dira','FlightDistance'], y_vars='FlightDistanceError', size=7, aspect=0.7)


# Since the test dataset does not have 4 features, it is prudent to remove these from the training dataset for the analysis.

# In[ ]:


removeFeatures = ('signal','production','mass','min_ANNmuon')
trTarget = train.loc[:,'signal']
trTarget.head()


# In[ ]:


trainNew = train.drop(columns=['signal','production','mass','min_ANNmuon'])
trainNew.head()


# In[ ]:


trainNew.shape


# In[ ]:


trainNew.tail(1)


# In[ ]:


test.head(1)


# Let's try to add some features to the dataset. 
# 
# We can add features by introducing some categorical variable (if binary, we do not need OneHotEncoding). 
# We can also use FeatureTools library to automatically generate features.

# In[ ]:


sns.boxplot( y=trainNew["pt"] )


# In[ ]:


trainNew['pt'].describe()


# In[ ]:


combi = pd.concat([trainNew, test])
combi.shape


# In[ ]:


def convertNum2Bin(x):
    if 0 < x <= 5028:
        return 0
    return 1

combi['ptBin'] = combi['pt'].apply(convertNum2Bin)


# In[ ]:


trainNew['LifeTime'].describe()


# In[ ]:


def convertLT2Bin(x):
    if 0 < x <= 0.001255:
        return 0
    return 1

combi['LifeTimeBin'] = combi['LifeTime'].apply(convertLT2Bin)


# In[ ]:


trainNew['FlightDistance'].describe()


# In[ ]:


def convertFD2Bin(x):
    if 0 < x <= 15.154:
        return 0
    return 1

combi['FDBin'] = combi['FlightDistance'].apply(convertFD2Bin)


# In[ ]:


trainNew['FlightDistanceError'].describe()


# In[ ]:


def convertFDE2Bin(x):
    if 0 < x <= 0.5018:
        return 0
    return 1

combi['FDEBin'] = combi['FlightDistanceError'].apply(convertFDE2Bin)


# In[ ]:


combi.shape


# In[ ]:



# Make an entityset and add the entity
#es = ft.EntitySet(id = 'particleData')
#es.entity_from_dataframe(entity_id = 'TrainAndTest', dataframe = combi, 
#                         make_index = True, index = 'index')

# Run deep feature synthesis with transformation primitives
#feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'TrainAndTest', max_depth = 2, verbose = 1,
                                      agg_primitives = ['count','mean'])
#                                      trans_primitives = ['add', 'multiply'])

#feature_matrix.head()


# In[ ]:


#feature_matrix.columns
#feature_matrix.shape


# In[ ]:


#combi.shape


# In[ ]:


#combi['id']=combi.reset_index().index
#combi.iloc[67552:67554,:]
trainNew = combi.iloc[:67553,:]    # or should we use feature_matrix instead of combi?
test= combi.iloc[67553:,:]
trainNew.shape


# In[ ]:


test.shape


# Before we proceed, let's Scale the data so that all features contribute equally to the prediction. We do not want some features to have more weight compared to others just becasue of a difference in the units of measurement.

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
trainScaled = scaler.fit_transform(trainNew)
trainScaled[0:2,]


# In[ ]:


testScaled = scaler.transform(test)
testScaled[0:2,]


# It might be a good idea to do a Principal Component Analysis of the data to avoid features that are highly correlated. Also, we will go for features that represent 95% of the variance to avoid overfitting of the data.

# In[ ]:


from sklearn.decomposition import PCA

pca = PCA().fit(trainScaled)
train_pca = pca.transform(trainScaled)
test_pca = pca.transform(testScaled)


# In[ ]:


def pca_summary(pca, standardized_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardized_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)
    
    if out:
        print("Importance of components:")
        display(summary)
    return summary


# In[ ]:


summary = pca_summary(pca, train_pca)


# In[ ]:


import matplotlib.pyplot as plt

def screeplot(pca, standardized_values):
    y = np.std(pca.transform(standardized_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()

screeplot(pca, trainScaled)


# First 31 PCA attributes give more than 95% variance. Let's go ahead with 31 PCA Attributes for our analysis.

# In[ ]:


pcatrain = pd.DataFrame(train_pca[:,0:31])
pcatrain.head()


# In[ ]:


pcatest = pd.DataFrame(test_pca[:,0:31])
pcatest.head()


# We will try various models - Gradient Boosting Classifier, Random Forest Classifier and Extreme Gradient Boosting (XGB) Classifier. 
# 
# We can try Parameter Tuning using Grid Cross Validation to fine tune the model. 

# **Baseline Training (Gradient Boosting Classifier)**

# In[ ]:


baseline = GradientBoostingClassifier(n_estimators=40, learning_rate=0.01, subsample=0.7,
                                      min_samples_leaf=10, max_depth=7, random_state=11)
baseline.fit(pcatrain, trTarget)
#baseline.fit(train[variables], train['signal'])


# **Random Forest Classifier (with Parameter Tuning)**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#pcatrain, pcatest, trTarget - already available

# Create the parameter grid 
param_grid = {
    'bootstrap': [False],
#    'max_depth': [100, 200],       ------ The RF Classifier failed the Agreement Test based on these parameters. So, leaving this step out for faster re-run of code.
    'max_features': ['sqrt'],
#    'min_samples_leaf': [1, 3],
#    'n_estimators': [200, 400]
}
# Create a RF Classifier model
rf = RandomForestClassifier(random_state = 42)
# Instantiate the grid search model
#grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 2, n_jobs = -1, verbose = 2) --- CV works but takes time so doing it without CV
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(pcatrain, trTarget)
grid_search.best_params_


# In[ ]:


best_gridRFCModel = grid_search.best_estimator_


#  **XGB Classifier (First set of parameters)**

# In[ ]:


import xgboost as xgb


clfXGB = xgb.XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5,random_state = 42)
clfXGB.fit(pcatrain, trTarget)


# **XGB Classifier (Second set of parameters)**

# In[ ]:


clfXGB2 = xgb.XGBClassifier(n_estimators=40, learning_rate=0.01, subsample=0.7,min_samples_leaf=10, max_depth=7, random_state=11)
clfXGB2.fit(pcatrain, trTarget)


#  **XGB Classifier (Third set of parameters)**

# In[ ]:


clfXGB3 = xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11)
clfXGB3.fit(pcatrain, trTarget)


# # Check agreement test

# In[ ]:


check_agreement = pd.read_csv(folder + 'check_agreement.csv', index_col='id')
check_agreement.shape


# In[ ]:


check_agreement.head(2)


# Check Agreement Dataframe has weight and signal in addition to the 50 attributes given in Train dataset.

# In[ ]:


check_agreement_sig_wt = check_agreement.loc[:,['signal','weight']]
check_agreement_sig_wt.head(2)

check_agreement_var = check_agreement.drop(columns=['signal','weight']) 
check_agreement_var.head(2)


# In[ ]:


check_agreement_var['ptBin'] = check_agreement_var['pt'].apply(convertNum2Bin)
check_agreement_var['LifeTimeBin'] = check_agreement_var['LifeTime'].apply(convertLT2Bin)
check_agreement_var['FDBin'] = check_agreement_var['FlightDistance'].apply(convertFD2Bin)
check_agreement_var['FDEBin'] = check_agreement_var['FlightDistanceError'].apply(convertFDE2Bin)

agreementScaled = scaler.transform(check_agreement_var)
agreement_pca = pca.transform(agreementScaled)
pcaAgreement = pd.DataFrame(agreement_pca[:,0:31])

pcaAgreement.head(2)


# In[ ]:


#agreement_probs = baseline.predict_proba(check_agreement[variables])[:, 1]
agreement_probs = baseline.predict_proba(pcaAgreement)[:, 1]

ks = compute_ks(
    agreement_probs[check_agreement_sig_wt['signal'].values == 0],
    agreement_probs[check_agreement_sig_wt['signal'].values == 1],
    check_agreement[check_agreement_sig_wt['signal'] == 0]['weight'].values,
    check_agreement[check_agreement_sig_wt['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)


# Let's do the Agreement Test on Random Forest Classifier too.

# In[ ]:


agreement_probs = best_gridRFCModel.predict_proba(pcaAgreement)[:, 1]

ks = compute_ks(
    agreement_probs[check_agreement_sig_wt['signal'].values == 0],
    agreement_probs[check_agreement_sig_wt['signal'].values == 1],
    check_agreement[check_agreement_sig_wt['signal'] == 0]['weight'].values,
    check_agreement[check_agreement_sig_wt['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)


# Random Forest Classifier test passed the test. Let's do it on XGB Classifier for 3 different sets of parameters.

# In[ ]:


agreement_probs = clfXGB.predict_proba(pcaAgreement)[:, 1]

ks = compute_ks(
    agreement_probs[check_agreement_sig_wt['signal'].values == 0],
    agreement_probs[check_agreement_sig_wt['signal'].values == 1],
    check_agreement[check_agreement_sig_wt['signal'] == 0]['weight'].values,
    check_agreement[check_agreement_sig_wt['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)


# In[ ]:


agreement_probs = clfXGB2.predict_proba(pcaAgreement)[:, 1]

ks = compute_ks(
    agreement_probs[check_agreement_sig_wt['signal'].values == 0],
    agreement_probs[check_agreement_sig_wt['signal'].values == 1],
    check_agreement[check_agreement_sig_wt['signal'] == 0]['weight'].values,
    check_agreement[check_agreement_sig_wt['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)


# In[ ]:


agreement_probs = clfXGB3.predict_proba(pcaAgreement)[:, 1]

ks = compute_ks(
    agreement_probs[check_agreement_sig_wt['signal'].values == 0],
    agreement_probs[check_agreement_sig_wt['signal'].values == 1],
    check_agreement[check_agreement_sig_wt['signal'] == 0]['weight'].values,
    check_agreement[check_agreement_sig_wt['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)


# Out of the above 5 models, Baseline model (Gradient Boosting), Random Forest Classifier, XGB (with second set of parameters) and XGB (with third set of parameters) passed the Agreement Test. We will proceed with these 4 models for the next steps.

# # Check correlation test
# 

# In[ ]:


check_correlation = pd.read_csv(folder + 'check_correlation.csv', index_col='id')
check_correlation.shape
check_correlation.head(1)


# In[ ]:


check_correlation_mass = check_correlation.loc[:,'mass']
check_correlation_mass.head(2)

check_correlation_var = check_correlation.drop('mass',axis = 1) 
check_correlation_var.head(2)


# In[ ]:


check_correlation_var.shape


# In[ ]:


check_correlation_var['ptBin'] = check_correlation_var['pt'].apply(convertNum2Bin)
check_correlation_var['LifeTimeBin'] = check_correlation_var['LifeTime'].apply(convertLT2Bin)
check_correlation_var['FDBin'] = check_correlation_var['FlightDistance'].apply(convertFD2Bin)
check_correlation_var['FDEBin'] = check_correlation_var['FlightDistanceError'].apply(convertFDE2Bin)

correlationScaled = scaler.transform(check_correlation_var)
correlation_pca = pca.transform(correlationScaled)
pcaCorrelation = pd.DataFrame(correlation_pca[:,0:31])

pcaCorrelation.head(2)


# In[ ]:


correlation_probs = baseline.predict_proba(pcaCorrelation)[:, 1]
cvm = compute_cvm(correlation_probs, check_correlation_mass)
print ('CvM metric', cvm, cvm < 0.002)


# In[ ]:


correlation_probs = best_gridRFCModel.predict_proba(pcaCorrelation)[:, 1]
cvm = compute_cvm(correlation_probs, check_correlation_mass)
print ('CvM metric', cvm, cvm < 0.002)


# In[ ]:


correlation_probs = clfXGB2.predict_proba(pcaCorrelation)[:, 1]
cvm = compute_cvm(correlation_probs, check_correlation_mass)
print ('CvM metric', cvm, cvm < 0.002)


# In[ ]:


correlation_probs = clfXGB3.predict_proba(pcaCorrelation)[:, 1]
cvm = compute_cvm(correlation_probs, check_correlation_mass)
print ('CvM metric', cvm, cvm < 0.002)


# So, only 3 of the models - passed the Correlation Test.

# # Compute weighted AUC on the training data with min_ANNmuon > 0.4
# 

# In[ ]:


train_eval = train[train['min_ANNmuon'] > 0.4]
trainEvalSignal = train_eval['signal']
trainEvalTruncated = train_eval.drop(columns=['signal','production','mass','min_ANNmuon'])

trainEvalTruncated['ptBin'] = trainEvalTruncated['pt'].apply(convertNum2Bin)
trainEvalTruncated['LifeTimeBin'] = trainEvalTruncated['LifeTime'].apply(convertLT2Bin)
trainEvalTruncated['FDBin'] = trainEvalTruncated['FlightDistance'].apply(convertFD2Bin)
trainEvalTruncated['FDEBin'] = trainEvalTruncated['FlightDistanceError'].apply(convertFDE2Bin)

trainEvalScaled = scaler.transform(trainEvalTruncated)
trainEval_pca = pca.transform(trainEvalScaled)
pcaTrainEval = pd.DataFrame(trainEval_pca[:,0:31])

pcaTrainEval.head(2)


# In[ ]:


train_probs = baseline.predict_proba(pcaTrainEval)[:, 1]
AUC = roc_auc_truncated(trainEvalSignal, train_probs)
print ('AUC', AUC)


# In[ ]:


train_probs = clfXGB2.predict_proba(pcaTrainEval)[:, 1]
AUC = roc_auc_truncated(trainEvalSignal, train_probs)
print ('AUC', AUC)


# In[ ]:


train_probs = clfXGB3.predict_proba(pcaTrainEval)[:, 1]
AUC = roc_auc_truncated(trainEvalSignal, train_probs)
print ('AUC', AUC)


# AUC Metric is best for XGB with the third set of parameters. Let's create file for Kaggle.

# # Predict test, create file for kaggle

# In[ ]:


result = pd.DataFrame({'id': test.index})
result['prediction'] = clfXGB3.predict_proba(pcatest)[:, 1]


# In[ ]:


sub = result.to_csv('Achal_baseline_5.csv', index=False, sep=',')


# 
