#!/usr/bin/env python
# coding: utf-8

# So, during the [Freesound General-Purpose Audio Tagging Challenge](Freesound General-Purpose Audio Tagging Challenge) I got into top 11% by using LightGBM and CatBoost, and running those on a large amount of manually extracted features. So, how well will this approach work this time? (Old code can be found [here](https://github.com/knstmrd/kagglefreesound))
# Not as well, unfortunately, due to a few reasons:
# 1. The main reason is that previously I used VGGish, a pre-trained Tensorflow network which uses a VGG-type CNN to produce a 128-dimensional feature vector for an audiofile. Since it was pre-trained, it's not possible to use it here; and those features turned out to be one of the most important ones
# 2. I used YAAFE and Essentia, two fast toolboxes that are unavailable in Kaggle kernels; so not only I'm limited by the speed of the feature extraction process, I'm also more limited in the features I can extract
# 3. 2-hour kernel runtime limit; but it's still possible to get a somewhat passable result in this timeframe, and do all the feature extraction and training and prediction in under two hours (without any CV though); obviously, the pre-processing step and training steps can be separated from the classification step (pre-process, save dataset, train and save classifiers), but where's the fun in that?
# 
# So what I'm doing is
# 1. Extract a lot of features (both from the raw wav file and from a spectrogram)
# 2. Most of these are time-series; take various percentiles of these
# 3. For time series also take time derivative and take various percentiles of that
# 4. Remove some of the highly-correlated features
# 
# To account the multiple-labels, I did the following:
# 1. Extract features for the train and test sets
# 2. For each train example with N labels, add N-1 copies (each with a new label) to the train set
# 3. During train, set sample weights like 1 / (1 + w * label_secondary), where w is some parameter, and label_secondary is either 1 or 0 (depending on whether it's the main or additional label); this basically weights the examples with more than 1 label somewhat lower if their non-main label is used
# 
# And then just run LightGBM and XGB and average.
# 
# **Possible improvements**
# 1. Of course, one can fine-tune more classifier parameters; find better blending weights, etc., etc.
# 2. Train a neural net and use some intermediate layer output as a feature set
# 3. Forget about this whole approach altogether :)
# 4. Find a way to utilize the noisy training dataset (some preliminary runs I did with it were pretty terrible)

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
import librosa as lr
import glob
import lightgbm as lgbm
import xgboost as xgb
import time
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import LabelEncoder
from scipy.signal import filtfilt, argrelmax, find_peaks, correlate
from scipy.stats import skew, kurtosis


# In[26]:


srate = 44100  # loading without re-sampling is faster
use_noisy = True
use_noisy = False

do_cv = True
# do_cv = False

n_folds = 10
actual_folds = 4

n_mfcc = 20  # number of MFCC coefficients to use

expand_multi_label = True
# expand_multi_label = False
multi_label_weight_multiplier = 0.5 # (w = 1 / (1+w.mult))


# In[3]:


# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class


# Wrapper for fast.ai library
def lwlrap(scores, truth, **kwargs):
    score, weight = calculate_per_class_lwlrap(truth, scores)
    return (score * weight).sum()


# In[4]:


t1 = time.time()

df_train_noisy = pd.read_csv('../input/train_noisy.csv')
df_train_curated = pd.read_csv('../input/train_curated.csv')
df_test = pd.read_csv('../input/sample_submission.csv')
print(len(df_test.columns) - 1, 'categories')
df_test = df_test[['fname']]

print(df_train_noisy.shape, df_train_curated.shape)
time_to_load_df = time.time() - t1
print('Time to load dataframes: ', time_to_load_df)


# In[6]:


t1 = time.time()

labels = []

for row in df_train_noisy['labels'].values:
    for sublabel in row.split(','):
        labels.append(sublabel)

df_train_noisy['labels_list'] = df_train_noisy['labels'].apply(lambda x: x.split(','))
df_train_noisy['label0'] = df_train_noisy['labels_list'].apply(lambda x: x[0])
df_train_noisy['labels_count'] = df_train_noisy['labels_list'].apply(lambda x: len(x))
df_train_noisy['label_secondary'] = 0

for row in df_train_curated['labels'].values:
    for sublabel in row.split(','):
        labels.append(sublabel)

df_train_curated['labels_list'] = df_train_curated['labels'].apply(lambda x: x.split(','))
df_train_curated['label0'] = df_train_curated['labels_list'].apply(lambda x: x[0])
df_train_curated['labels_count'] = df_train_curated['labels_list'].apply(lambda x: len(x))
df_train_curated['label_secondary'] = 0

n_labels_max_curated = df_train_curated['labels_count'].max()
n_labels_max_noisy = df_train_noisy['labels_count'].max()
print('Max labels, curated, noisy', n_labels_max_curated, n_labels_max_noisy)

labels = set(labels)
print(len(labels))
labels = list(labels)

lenc = LabelEncoder()
lenc.fit(labels)

df_train_curated['int_label0'] = lenc.transform(df_train_curated['label0'])
df_train_noisy['int_label0'] = lenc.transform(df_train_noisy['label0'])


df_train_curated['labels_list_int'] = df_train_curated['labels_list'].apply(lambda x: lenc.transform(x))
df_train_noisy['labels_list_int'] = df_train_noisy['labels_list'].apply(lambda x: lenc.transform(x))



df_train_curated['full_fname'] = df_train_curated['fname'].apply(lambda x: '../input/train_curated/' + x)
df_train_noisy['full_fname'] = df_train_noisy['fname'].apply(lambda x: '../input/train_noisy/' + x)

print(df_train_curated['int_label0'].nunique(), df_train_noisy['int_label0'].nunique())

time_to_process_labels = time.time() - t1
print('Time to process labels: ', time_to_process_labels)


# In[7]:


def load_file_librosa(fname, sr=44100):
#     print(lr.core.load(fname, sr=sr)[0].shape)
    return lr.core.load(fname, sr=sr)[0]


# In[8]:


def length(array, sample_rate=44100):
    return array.shape[0] / sample_rate


# In[9]:


def wav_features(array):
    if len(array) == 0:
        return [0.] * 16
    else:
        argmin = np.argmin(array)
        argmax = np.argmax(array)
        std = np.std(array, ddof=1)
        if array[argmin] == 0.0:
            min_arr_corr = array[argmin] + 1e-10
        else:
            min_arr_corr = array[argmin]
        return [argmin / array.shape[0], argmax / array.shape[0], # 2
                array[argmin], array[argmax], # 4
                np.mean(array), # 5
                np.percentile(array, 10), np.percentile(array, 25), # 7
                np.percentile(array, 50),  # 8
                np.percentile(array, 75), np.percentile(array, 90),  # 10
                skew(array), kurtosis(array),  # 12
                std,  # 13
                array[argmax] / min_arr_corr]  # 15


# In[10]:


def get_zcr(array, sample_rate=44100):
    return np.sum(lr.core.zero_crossings(array)) / array.shape[0]


# In[11]:


def wav_autocorrelation(array):
    try:
        if array.shape[0] > 3 * srate:
            tmp_arr = array[:3 * srate]
        else:
            tmp_arr = array
        autocorr = correlate(tmp_arr, tmp_arr)
        autocorr = autocorr[autocorr.shape[0]//2:]
        peaks = find_peaks(autocorr[:800])
        if len(peaks[0]) == 0:
            peakpos = 1000
            peakval = 1
        else:
            peakpos = peaks[0][0]
            peakval = autocorr[peaks[0][0]] / autocorr[0]
        return {'wav_autocorr_peak_position': peakpos / srate,
                'wav_autocorr_peak_value_normalized': peakval,
                'wav_autocorr_ZCR': np.sum(lr.core.zero_crossings(autocorr[:800])) / 800}
    except ValueError as e:
        return {'wav_autocorr_peak_position': 1000 / srate,
                'wav_autocorr_peak_value_normalized': 1,
                'wav_autocorr_ZCR': 0}


# In[12]:


def spec_features(spec, srate):
    spectroid = lr.feature.spectral_centroid(sr=srate, S=spec)[0, :]
    rolloff = lr.feature.spectral_rolloff(sr=srate, S=spec, roll_percent=0.85)[0, :]
    rolloff50 = lr.feature.spectral_rolloff(sr=srate, S=spec, roll_percent=0.5)[0, :]
    spec_max = np.max(spectroid)
    spec_min = np.min(spectroid)
    
    roll_max = np.max(rolloff)
    roll_min = np.min(rolloff)
    
    roll_max50 = np.max(rolloff50)
    roll_min50 = np.min(rolloff50)
    
    if spectroid.shape[0] > 1:
        grad_spec = spectroid[1:] - spectroid[:-1]
    else:
        grad_spec = [0.]
    
    grad_spec_max = np.max(grad_spec)
    grad_spec_min = np.min(grad_spec)
    
    if rolloff.shape[0] > 1:
        grad_rolloff = rolloff[1:] - rolloff[:-1]
        grad_rolloff50 = rolloff50[1:] - rolloff50[:-1]
    else:
        grad_rolloff = [0.]
        grad_rolloff50 = [0.]
        
    grad_rolloff_max = np.max(grad_rolloff)
    grad_rolloff_min = np.min(grad_rolloff)
    
    grad_rolloff_max50 = np.max(grad_rolloff50)
    grad_rolloff_min50 = np.min(grad_rolloff50)
    
    rms = lr.feature.rms(S=spec)
    
    rms = rms[0, :]
    
    if rms.shape[0] > 2:
        grad_rms = rms[1:] - rms[:-1]
    else:
        rms = [0., 0.]
        grad_rms = [0., 0.]
    
    if spec.shape[1] > 2:
        sflux_a = spec[:, 1:]
        sflux_b = spec[:, :-1]
        
        sflux = sflux_a / np.max(sflux_a, axis=0) - sflux_b / np.max(sflux_b, axis=0)
        sflux = np.sum(sflux**2, axis=0)
        
    else:
        sflux = [0., 0.]
    
    sflux = np.nan_to_num(sflux)
    
    return {'rms_kurtosis': kurtosis(rms),
            'rms_skew': skew(rms),
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms, ddof=1),
            'rms_median': np.median(rms),
            
            'd_rms_kurtosis': kurtosis(grad_rms),
            'd_rms_skew': skew(grad_rms),
            'd_rms_mean': np.mean(grad_rms),
            'd_rms_std': np.std(grad_rms, ddof=1),
            'd_rms_median': np.median(grad_rms),
            
            'sflux_perc10': np.percentile(sflux, 10),
            'sflux_perc25': np.percentile(sflux, 25),
            'sflux_perc75': np.percentile(sflux, 75),
            'sflux_mean': np.mean(sflux),
            'sflux_median': np.median(sflux),
            'sflux_skew': skew(sflux),
            'sflux_kurtosis': kurtosis(sflux),
            
            'spectroid_mean': np.mean(spectroid), 'spectroid_std': np.std(spectroid, ddof=1),
            'spectroid_max_div_min': spec_max / (spec_min + 1e-20),
            'spectroid_max': spec_max, 'spectroid_min': spec_min,
            'spectroid_median': np.median(spectroid),
            'spectroid_perc10': np.percentile(spectroid, 10),
            'spectroid_perc25': np.percentile(spectroid, 25),
            'spectroid_perc75': np.percentile(spectroid, 75),
            'rolloff_mean': np.mean(rolloff), 'rolloff_std': np.std(rolloff, ddof=1),
            'rolloff_max_div_min': roll_max / (roll_min + 1e-20),
            'rolloff_max': roll_max, 'rolloff_min': roll_min,
            'rolloff_median': np.median(rolloff),
            'rolloff_perc10': np.percentile(rolloff, 10),
            'rolloff_perc25': np.percentile(rolloff, 25),
            'rolloff_perc75': np.percentile(rolloff, 75),
            'd_spectroid_mean': np.mean(grad_spec), 'd_spectroid_std': np.std(grad_spec, ddof=1),
            'd_spectroid_max_div_min': grad_spec_max / (grad_spec_min + 1e-20),
            'd_spectroid_max': grad_spec_max, 'd_spectroid_min': grad_spec_min,
            'd_spectroid_median': np.median(grad_spec),
            'd_spectroid_perc10': np.percentile(grad_spec, 10),
            'd_spectroid_perc25': np.percentile(grad_spec, 25),
            'd_spectroid_perc75': np.percentile(grad_spec, 75),
            'd_rolloff_mean': np.mean(grad_rolloff), 'd_rolloff_std': np.std(grad_rolloff, ddof=1),
            'd_rolloff_max_div_min': grad_rolloff_max / (grad_rolloff_min + 1e-20),
            'd_rolloff_max': grad_rolloff_max, 'd_rolloff_min': grad_rolloff_min,
            'd_rolloff_median': np.median(grad_rolloff),
            'd_rolloff_perc10': np.percentile(grad_rolloff, 10),
            'd_rolloff_perc25': np.percentile(grad_rolloff, 25),
            'd_rolloff_perc75': np.percentile(grad_rolloff, 75),
        
            'rolloff_mean50': np.mean(rolloff50),
            'rolloff_std50': np.std(rolloff50, ddof=1),
            'rolloff_max50': roll_max50,
            'rolloff_min50': roll_min50,
            'rolloff_median50': np.median(rolloff50),
            'rolloff_perc1050': np.percentile(rolloff50, 10),
            'rolloff_perc2550': np.percentile(rolloff50, 25),
            'rolloff_perc7550': np.percentile(rolloff50, 75),
            'd_rolloff_mean50': np.mean(grad_rolloff50),
            'd_rolloff_std50': np.std(grad_rolloff50, ddof=1),
            'd_rolloff_max50': grad_rolloff_max50,
            'd_rolloff_min50': grad_rolloff_min50,
            'd_rolloff_median50': np.median(grad_rolloff50),
            'd_rolloff_perc1050': np.percentile(grad_rolloff50, 10),
            'd_rolloff_perc2550': np.percentile(grad_rolloff50, 25),
            'd_rolloff_perc7550': np.percentile(grad_rolloff50, 75),
           }


# In[13]:


def mfcc_features(spec, srate):
    
    mfcc = lr.feature.mfcc(sr=srate, S=spec, n_mfcc=n_mfcc)
    output = []
    
    for i in range(n_mfcc):
        output.append(np.mean(mfcc[i, :]))
        output.append(np.std(mfcc[i, :], ddof=1))
        output.append(skew(mfcc[i, :]))
        output.append(kurtosis(mfcc[i, :]))
    
    return output


# In[14]:


def extract_features_from_file_v1(fname):
    audio = load_file_librosa(fname, sr=srate)
    output = {'length, s': length(audio)}
    output['ZCR'] = get_zcr(audio, sample_rate=srate)
        
    output['wav features'] = wav_features(audio)
    output['wav autocorr'] = wav_autocorrelation(audio)
    
    if len(audio) < 3:
        d_audio = np.zeros(2)
    else:
        d_audio = audio[1:] - audio[:-1]
        
    
    output['d_wav features'] = wav_features(d_audio)
    output['d_wav autocorr'] = wav_autocorrelation(d_audio)
    
    spec = np.abs(lr.core.stft(audio))
    spec_mel = lr.feature.melspectrogram(sr=srate, S=spec**2)
    output['spec features'] = spec_features(spec, srate)
    output['mfcc features'] = mfcc_features(spec_mel, srate)
    return output


# In[15]:


def process_names_v1(df):
    all_names = list(df['feats'][0].keys())
    
    for featname in ['length, s', 'ZCR']:
        df[featname] = df['feats'].apply(lambda x: x[featname])
    
    for featname in ['wav_autocorr_peak_position', 'wav_autocorr_peak_value_normalized',
                     'wav_autocorr_ZCR']:
        df[featname] = df['feats'].apply(lambda x: x['wav autocorr'][featname])
    
    
    
    for i, funcname in enumerate(['argmin_rel', 'argmax_rel', 'min', 'max',
                                  'mean', 'perc10', 'perc25', 'perc50',
                                  'perc75', 'perc90', 'skew', 'kurtosis', 'std',
                                  'max_div_min']):
        df[' '.join(('wav', funcname))] = df['feats'].apply(lambda x: x['wav features'][i])
        
        
    # dw/dt
    for featname in ['wav_autocorr_peak_position', 'wav_autocorr_peak_value_normalized',
                     'wav_autocorr_ZCR']:
        df[' '.join(('d_wav', funcname))] = df['feats'].apply(lambda x: x['d_wav autocorr'][featname])
    
    
    
    for i, funcname in enumerate(['argmin_rel', 'argmax_rel', 'min', 'max',
                                  'mean', 'perc10', 'perc25', 'perc50',
                                  'perc75', 'perc90', 'skew', 'kurtosis', 'std',
                                  'max_div_min']):
        df[' '.join(('d_wav', funcname))] = df['feats'].apply(lambda x: x['d_wav features'][i])
        
    for i, funcname in enumerate(['rms_kurtosis', 'rms_skew',
                                  'rms_mean', 'rms_median', 'rms_std',
                                  'd_rms_kurtosis', 'd_rms_skew',
                                  'd_rms_mean', 'd_rms_median', 'd_rms_std',
                                  'sflux_perc10',
                                  'sflux_perc25',
                                  'sflux_perc75',
                                  'sflux_mean',
                                  'sflux_median',
                                  'sflux_skew',
                                  'sflux_kurtosis',
                                  'spectroid_mean', 'spectroid_std',
                                  'spectroid_max_div_min',
                                  'spectroid_max', 'spectroid_min',
                                  'rolloff_mean', 'rolloff_std',
                                  'rolloff_max_div_min',
                                  'rolloff_max', 'rolloff_min',
                                  'd_spectroid_mean', 'd_spectroid_std',
                                  'd_spectroid_max_div_min',
                                  'd_spectroid_max', 'd_spectroid_min',
                                  'd_rolloff_mean', 'd_rolloff_std',
                                  'd_rolloff_max_div_min',
                                  'd_rolloff_max', 'd_rolloff_min',
                                  'spectroid_median', 'rolloff_median',
                                  'd_spectroid_median', 'd_rolloff_median',
                                  'spectroid_perc10', 'spectroid_perc25', 'spectroid_perc75',
                                  'rolloff_perc10', 'rolloff_perc25', 'rolloff_perc75',
                                  'd_spectroid_perc10', 'd_spectroid_perc25', 'd_spectroid_perc75',
                                  'd_rolloff_perc10', 'd_rolloff_perc25', 'd_rolloff_perc75',
                
                                  'rolloff_mean50', 'rolloff_std50',
                                  'rolloff_max50', 'rolloff_min50',
                                  'rolloff_median50', 'd_rolloff_median50',
                                  'd_rolloff_mean50', 'd_rolloff_std50',
                                  'd_rolloff_max50', 'd_rolloff_min50',
                                  'rolloff_perc1050', 'rolloff_perc2550', 'rolloff_perc7550',
                                  'd_rolloff_perc1050', 'd_rolloff_perc2550', 'd_rolloff_perc7550'
                                 ]):
        df[' '.join(('spec', funcname))] = df['feats'].apply(lambda x: x['spec features'][funcname])
    
    counter = 0
    for i in range(n_mfcc):
        for k in ['mean', 'std', 'skew', 'curtosis']:
            df[' '.join(('mfcc', str(i), k))] = df['feats'].apply(lambda x: x['mfcc features'][counter])
            counter += 1
    
    df.drop(['feats'], axis=1, inplace=True)


# In[16]:


if use_noisy:
    df_train = pd.concat([df_train_curated, df_train_noisy], axis=0)
else:
    df_train = df_train_curated
print(len(df_train))


# In[17]:


t1 = time.time()

df_train['feats'] = df_train['full_fname'].apply(lambda x: extract_features_from_file_v1(x))
process_names_v1(df_train)

time_to_extract_v1_train = time.time() - t1
print('Time to extract v1 features from train set:', time_to_extract_v1_train)


# In[18]:


t1 = time.time()

df_test['feats'] = df_test['fname'].apply(lambda x: extract_features_from_file_v1('../input/test/' + x))
process_names_v1(df_test)

time_to_extract_v1_test = time.time() - t1
print('Time to extract v1 features from test set:', time_to_extract_v1_test)


# In[19]:


feat_names = [col for col in df_train.columns if not (col.startswith('label')
                                                      or col.endswith('fname') or col.startswith('int_label'))]
print('Total features, ', len(feat_names))


# In[20]:


if expand_multi_label:
    df_train_multilabel = df_train[df_train['labels_count'] > 1]

    startpos = len(df_train)
    print(startpos, len(df_train_multilabel))
    for row in df_train_multilabel.iterrows():
        for i in range(1,row[1]['labels_count']):
            df_train = df_train.append(row[1])
            df_train.iloc[startpos, df_train.columns.get_loc('label0')] = row[1]['labels_list'][i]
            df_train.iloc[startpos, df_train.columns.get_loc('int_label0')] = row[1]['labels_list_int'][i]
            df_train.iloc[startpos, df_train.columns.get_loc('label_secondary')] = 1
            startpos += 1


# In[21]:


real_classes = df_train['int_label0'].unique()
real_classes.sort()


# In[22]:


corr = df_train[feat_names].corr()
corr = corr.abs()

correlation_threshold = 0.95

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
columns_to_remove = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
print(len(feat_names), columns_to_remove, len(columns_to_remove))

feat_names = [x for x in feat_names if x not in columns_to_remove]
print(len(feat_names), 'features left after removing highly correlated features')


# In[28]:


clf1 = lgbm.LGBMClassifier(n_estimators=1100, num_leaves=25, learning_rate=0.005,
                           colsample_bytree=0.75, objective='multiclass', random_state=47,
                           reg_alpha=0.09, reg_lambda=0.05)

clf2 = xgb.XGBClassifier(max_depth=5, learning_rate=0.005, n_estimators=850, colsample_bytree=0.8,
                         colsample_bylevel=0.95, object='multi:softmax')

clf1.fit(df_train[feat_names], df_train['int_label0'],
         sample_weight=1.0/(1.0 + multi_label_weight_multiplier * df_train['label_secondary']))
output = 0.3 * clf1.predict_proba(df_test[feat_names])

clf2.fit(df_train[feat_names], df_train['int_label0'],
         sample_weight=1.0/(1.0 + multi_label_weight_multiplier * df_train['label_secondary']))
output += 0.7 * clf2.predict_proba(df_test[feat_names])

print(output.shape, len(labels))
    


# In[ ]:


output_df = pd.DataFrame({'fname': df_test['fname']}) 
        
for i in range(output.shape[1]):
    real_col_name = lenc.inverse_transform([real_classes[i]])[0]
    output_df[real_col_name] = output[:, i]

if output.shape[1] < len(labels):
    for i in range(len(labels)):
        if i not in real_classes:
            real_col_name = lenc.inverse_transform([i])[0]
            output_df[real_col_name] = np.zeros(output.shape[0])
            
output_df.to_csv('submission.csv', index=False)

