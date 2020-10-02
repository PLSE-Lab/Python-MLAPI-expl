#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


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
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def __cvm(subindices, total_events):
    """
    Compute Cramer-von Mises metric.
    Compared two distributions, where first is subset of second one.
    Assuming that second is ordered by ascending
    :param subindices: indices of events which will be associated with the first distribution
    :param total_events: count of events in the second distribution
    :return: cvm metric
    """
    target_distribution = np.arange(1, total_events + 1, dtype='float') / total_events
    subarray_distribution = np.cumsum(np.bincount(subindices, minlength=total_events), dtype='float')
    subarray_distribution /= 1.0 * subarray_distribution[-1]
    return np.mean((target_distribution - subarray_distribution) ** 2)


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
    predictions = np.array(predictions)
    masses = np.array(masses)
    assert len(predictions) == len(masses)

    # First, reorder by masses
    predictions = predictions[np.argsort(masses)]

    # Second, replace probabilities with order of probability among other events
    predictions = np.argsort(np.argsort(predictions, kind='mergesort'), kind='mergesort')

    # Now, each window forms a group, and we can compute contribution of each group to CvM
    cvms = []
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return np.mean(cvms)


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
    weights = np.concatenate([sample_weights_zero, sample_weights_one])
    data_all = np.concatenate([data_zero, data_one])
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

    data_prediction, mc_prediction = np.array(data_prediction), np.array(mc_prediction)
    weights_data, weights_mc = np.array(weights_data), np.array(weights_mc)

    assert np.all(data_prediction >= 0.) and np.all(data_prediction <= 1.), 'Data predictions are out of range [0, 1]'
    assert np.all(mc_prediction >= 0.) and np.all(mc_prediction <= 1.), 'MC predictions are out of range [0, 1]'

    weights_data /= np.sum(weights_data)
    weights_mc /= np.sum(weights_mc)

    fpr, tpr = __roc_curve_splitted(data_prediction, mc_prediction, weights_data, weights_mc)

    Dnm = np.max(np.abs(fpr - tpr))
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
    assert np.all(predictions >= 0.) and np.all(predictions <= 1.), 'Data predictions are out of range [0, 1]'
    assert len(tpr_thresholds) + 1 == len(roc_weights), 'Incompatible lengths of thresholds and weights'
    fpr, tpr, _ = roc_curve(labels, predictions)
    area = 0.
    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
    for index in range(1, len(tpr_thresholds)):
        tpr_cut = np.minimum(tpr, tpr_thresholds[index])
        tpr_previous = np.minimum(tpr, tpr_thresholds[index - 1])
        area += roc_weights[index - 1] * (auc(fpr, tpr_cut) - auc(fpr, tpr_previous))
    tpr_thresholds = np.array(tpr_thresholds)
    # roc auc normalization to be 1 for an ideal classifier
    area /= np.sum((tpr_thresholds[1:] - tpr_thresholds[:-1]) * np.array(roc_weights))
    return area


# In[ ]:


ls ../input/flavours-of-physics-kernels-only


# In[ ]:


print("Load the train/test/eval data using pandas")
train = pd.read_csv("../input/flavours-of-physics-kernels-only/training.csv.zip")
test  = pd.read_csv("../input/flavours-of-physics-kernels-only/test.csv.zip")


# In[ ]:


# -------------- loading data files -------------- #



#--------------- feature engineering -------------- #
def add_features(df):
    # features used by the others on Kaggle
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    #df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError'] # modified to:
    df['flight_dist_sig2'] = (df['FlightDistance']/df['FlightDistanceError'])**2
    # features from phunter
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['NEW_IP_dira'] = df['IP']*df['dira']
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)
    # My:
    # new combined features just to minimize their number;
    # their physical sense doesn't matter
    df['NEW_iso_abc'] = df['isolationa']*df['isolationb']*df['isolationc']
    df['NEW_iso_def'] = df['isolationd']*df['isolatione']*df['isolationf']
    df['NEW_pN_IP'] = df['p0_IP']+df['p1_IP']+df['p2_IP']
    df['NEW_pN_p']  = df['p0_p']+df['p1_p']+df['p2_p']
    df['NEW_IP_pNpN'] = df['IP_p0p2']*df['IP_p1p2']
    df['NEW_pN_IPSig'] = df['p0_IPSig']+df['p1_IPSig']+df['p2_IPSig']
    #My:
    # "super" feature changing the result from 0.988641 to 0.991099
    df['NEW_FD_LT']=df['FlightDistance']/df['LifeTime']
    return df

print("Add features")
train = add_features(train)
test = add_features(test)


print("Eliminate features")
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',
              'SPDhits','CDF1', 'CDF2', 'CDF3',
              'isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt',
              'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',
              'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf',
              'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT',
              'p0_IP', 'p1_IP', 'p2_IP',
              'IP_p0p2', 'IP_p1p2',
              'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof',
              'p0_IPSig', 'p1_IPSig', 'p2_IPSig',
              'DOCAone', 'DOCAtwo', 'DOCAthree']

features = list(f for f in train.columns if f not in filter_out)


# In[ ]:


mungedtrain = train[features].astype(complex)
mungedtest = test[features].astype(complex)


# In[ ]:


def Output(p):
    return 1./(1.+np.exp(-(0.476446 +p)))
def GPReal(data):
     return (0.659360*np.tanh(np.real(((np.sin((np.sin(((((((data["p_track_Chi2Dof_MAX"]) * (data["p_track_Chi2Dof_MAX"]))) + (data["IPSig"]))/2.0)))))) - ((((((data["p_track_Chi2Dof_MAX"]) * (np.power((data["iso"]),2.)))) + ((((((data["IPSig"]) + (data["VertexChi2"]))/2.0)) - (((data["flight_dist_sig"]) / (data["IPSig"]))))))/2.0))))) +
             0.800000*np.tanh(np.real(((np.cos((((np.power((((((data["LifeTime"]) / (((np.tanh((complex(1.86195778846740723)))) * (np.conjugate(data["NEW_FD_SUMP"]))*(complex(0,1)))))) - (((((data["dira"]) + (data["LifeTime"]))) + (data["LifeTime"]))))),2.)) / 2.0)))) - (((data["VertexChi2"]) + (np.power((((np.tanh((data["LifeTime"]))) / (data["NEW_FD_SUMP"]))),3.))))))))
def GPComplex(data):
    return (0.659360*np.tanh(np.imag(((np.sin((np.sin(((((((data["p_track_Chi2Dof_MAX"]) * (data["p_track_Chi2Dof_MAX"]))) + (data["IPSig"]))/2.0)))))) - ((((((data["p_track_Chi2Dof_MAX"]) * (np.power((data["iso"]),2.)))) + ((((((data["IPSig"]) + (data["VertexChi2"]))/2.0)) - (((data["flight_dist_sig"]) / (data["IPSig"]))))))/2.0))))) +
             0.800000*np.tanh(np.imag(((np.cos((((np.power((((((data["LifeTime"]) / (((np.tanh((complex(1.86195778846740723)))) * (np.conjugate(data["NEW_FD_SUMP"]))*(complex(0,1)))))) - (((((data["dira"]) + (data["LifeTime"]))) + (data["LifeTime"]))))),2.)) / 2.0)))) - (((data["VertexChi2"]) + (np.power((((np.tanh((data["LifeTime"]))) / (data["NEW_FD_SUMP"]))),3.))))))))


# In[ ]:


plt.figure(figsize=(15,15))
colors=['r','g']
plt.scatter(GPReal(mungedtrain),GPComplex(mungedtrain),s=1,color=[colors[int(c)] for c in train.signal.values])
plt.show()


# In[ ]:


plt.figure(figsize=(15,15))
plt.scatter(GPReal(mungedtest[::12]),GPComplex(mungedtest[::12]),s=1)


# In[ ]:


check_agreement = pd.read_csv('../input/flavours-of-physics-kernels-only/check_agreement.csv.zip')
check_correlation = pd.read_csv('../input/flavours-of-physics-kernels-only/check_correlation.csv.zip')
check_agreement.head()
check_agreement = add_features(check_agreement)
check_agreement = check_agreement[features+['weight','signal']]
check_agreement.head()
check_correlation = add_features(check_correlation)
check_correlation = check_correlation[features+['mass']]
check_agreement[check_agreement.columns[:-2]] = check_agreement[check_agreement.columns[:-2]].astype(complex)
agreement_probs = Output(GPReal(check_agreement))
print('Checking agreement...')
ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric', ks, ks < 0.09)
check_correlation[check_correlation.columns[:-1]] = check_correlation[check_correlation.columns[:-1]].astype(complex)
correlation_probs = Output(GPReal(check_correlation))
print ('Checking correlation...')
cvm = compute_cvm(correlation_probs, check_correlation['mass'])
print ('CvM metric', cvm, cvm < 0.002)

train_eval_probs = Output(GPReal(mungedtrain))

print ('Calculating AUC...')
AUC = roc_auc_truncated(train['signal'], train_eval_probs)
print ('AUC', AUC)


# In[ ]:


test.head()


# In[ ]:


pd.DataFrame({'id':test.id.values,'prediction':Output(GPReal(mungedtest))}).to_csv('submission.csv',index=False)

