#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import *
import gc
from sklearn.feature_selection import f_classif
import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, uniform, norm
from scipy.stats import randint, poisson
from sklearn.metrics import confusion_matrix, make_scorer

sns.set(style="darkgrid", context="notebook")
rand_seed = 135
np.random.seed(rand_seed)
xsize = 12.0
ysize = 8.0

import os
print(os.listdir("../input"))


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print("Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_meta_df = pd.read_csv("../input/metadata_train.csv")\ntrain_df = pq.read_pandas("../input/train.parquet").to_pandas()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_meta_df = reduce_mem_usage(train_meta_df)\ntrain_df = reduce_mem_usage(train_df)\ngc.collect()')


# In[ ]:


train_meta_df.shape


# In[ ]:


train_meta_df.head(6)


# In[ ]:


train_df.head()


# In[ ]:


fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(xsize, 2.0*ysize)

sns.countplot(x="phase", data=train_meta_df, ax=axes[0])

sns.countplot(x="target", data=train_meta_df, ax=axes[1])

plt.show()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(xsize, ysize)

sns.countplot(x="phase", hue="target", data=train_meta_df, ax=ax)

plt.show()


# So the phase counts are all equal, so this will not be a useful variable on its own for detecting a fault. Furthermore, it's interesting to not that the target much more likely to be 0, or the line has no fault, by default. This might might make models difficult to calibrate later on, but that's a later issue.
# 
# Now let's take a look at some of these signals.

# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2)
fig.set_size_inches(2.0*xsize, 2.0*ysize)
axes = axes.flatten()

axes[0].plot(train_df["0"].values, marker="o", linestyle="none")
axes[0].set_title("Signal ID: 0")

axes[1].plot(train_df["2"].values, marker="o", linestyle="none")
axes[1].set_title("Signal ID: 1")

axes[2].plot(train_df["3"].values, marker="o", linestyle="none")
axes[2].set_title("Signal ID: 2")

axes[3].plot(train_df["4"].values, marker="o", linestyle="none")
axes[3].set_title("Signal ID: 3")

axes[4].plot(train_df["5"].values, marker="o", linestyle="none")
axes[4].set_title("Signal ID: 4")

axes[5].plot(train_df["6"].values, marker="o", linestyle="none")
axes[5].set_title("Signal ID: 5")

plt.show()


# Note signals 0, 1, and 2 are not faulty and signals 3, 4, and 5 are faulty. They're messy, noisy, and not obviously periodic, oh boy. However, there are quite a few signal processing techniques that can be used anyways. Speaking of which, it's time for some feature engineering. Starting with some basic aggregations.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_meta_df["signal_mean"] = train_df.agg(np.mean).values\ntrain_meta_df["signal_sum"] = train_df.agg(np.sum).values\ntrain_meta_df["signal_std"] = train_df.agg(np.std).values')


# In[ ]:


train_meta_df.head()


# Now to look into some power spectrums since this is a signal processing challenge after all.

# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(2.0*xsize, 2.0*ysize)
axes = axes.flatten()

f, Pxx = welch(train_df["0"].values)
axes[0].plot(f, Pxx, marker="o", linestyle="none")
axes[0].set_title("Signal ID: 0")
axes[0].axhline(y=2.5, color="k", linestyle="--")

f, Pxx = welch(train_df["1"].values)
axes[1].plot(f, Pxx, marker="o", linestyle="none")
axes[1].set_title("Signal ID: 1")
axes[1].axhline(y=2.5, color="k", linestyle="--")

f, Pxx = welch(train_df["2"].values)
axes[2].plot(f, Pxx, marker="o", linestyle="none")
axes[2].set_title("Signal ID: 2")
axes[2].axhline(y=2.5, color="k", linestyle="--")

f, Pxx = welch(train_df["3"].values)
axes[3].plot(f, Pxx, marker="o", linestyle="none")
axes[3].set_title("Signal ID: 3")
axes[3].axhline(y=2.5, color="k", linestyle="--")

plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef welch_max_power_and_frequency(signal):\n    f, Pxx = welch(signal)\n    ix = np.argmax(Pxx)\n    strong_count = np.sum(Pxx>2.5)\n    avg_amp = np.mean(Pxx)\n    sum_amp = np.sum(Pxx)\n    std_amp = np.std(Pxx)\n    median_amp = np.median(Pxx)\n    return [Pxx[ix], f[ix], strong_count, avg_amp, sum_amp, std_amp, median_amp]\n\npower_spectrum_summary = train_df.apply(welch_max_power_and_frequency, result_type="expand")')


# In[ ]:


power_spectrum_summary = power_spectrum_summary.T.rename(columns={0:"max_amp", 1:"max_freq", 2:"strong_amp_count", 3:"avg_amp", 
                                                                  4:"sum_amp", 5:"std_amp", 6:"median_amp"})
power_spectrum_summary.head()


# In[ ]:


power_spectrum_summary.index = power_spectrum_summary.index.astype(int)
train_meta_df = train_meta_df.merge(power_spectrum_summary, left_on="signal_id", right_index=True)
train_meta_df.head()


# In[ ]:


X_cols = ["phase"] + train_meta_df.columns[4:].tolist()
X_cols


# In[ ]:


Fvals, pvals = f_classif(train_meta_df[X_cols], train_meta_df["target"])

print("F-value | P-value | Feature Name")
print("--------------------------------")

for i, col in enumerate(X_cols):
    print("%.4f"%Fvals[i]+" | "+"%.4f"%pvals[i]+" | "+col)


# So as expected phase is a useless feature on its own, but interestingly std_amp, median_amp, signal_std, max_amp may not be extremely useful variables because we cannot reject the null with a significance of 0.01 for these. However the features signal_mean, signal_sum, max_freq, strong_amp_count, avg_amp, and sum_amp all look like very useful features, even on their own.

# In[ ]:


def mcc(y_true, y_pred, labels=None, sample_weight=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight).ravel()
    mcc = (tp*tn - fp*fn)/np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    return mcc

mcc_scorer = make_scorer(mcc)

lgbm_classifier = lgbm.LGBMClassifier(boosting_type='gbdt', max_depth=-1, subsample_for_bin=200000, objective="binary", 
                                      class_weight=None, min_split_gain=0.0, min_child_weight=0.001, subsample=1.0, 
                                      subsample_freq=0, random_state=rand_seed, n_jobs=1, silent=True, importance_type='split')

param_distributions = {
    "num_leaves": randint(16, 48),
    "learning_rate": expon(),
    "reg_alpha": expon(),
    "reg_lambda": expon(),
    "colsample_bytree": uniform(0.25, 1.0),
    "min_child_samples": randint(10, 30),
    "n_estimators": randint(50, 250)
}

clf = RandomizedSearchCV(lgbm_classifier, param_distributions, n_iter=100, scoring=mcc_scorer, fit_params=None, n_jobs=1, iid=True, 
                         refit=True, cv=5, verbose=1, random_state=rand_seed, error_score=-1.0, return_train_score=True)
clf.fit(train_meta_df[X_cols], train_meta_df["target"])


# In[ ]:


print(clf.best_score_)


# In[ ]:


clf.best_estimator_


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(xsize, ysize)

lgbm.plot_importance(clf.best_estimator_, ax=ax)

plt.show()


# These results are interesting. The features signal_std, signal_mean, and avg_amp seem to be the most important. This makes sense intuitively because a faulty line will have more noise in its signal than a non faulty line, so for a faulty line we would expect a abnormally large signal_std, a signal_mean that is outside of the normal range due to outliers, and abnormally large avg_amp that results from lower frequencies becoming more present due to the noise from a faulty line. The next set of important features, max_amp, strong_amp_count, std_amp, and median_amp while not as important still support the current hypothesis of what the lgbm model is capturing. Finally signal_sum, sum_amp, max_freq are not important features because sum and median are robust to large outliers hence why they are not important and as determined earlier phase is not an important feature at all. 
# 
# 
# So the moral of this brief EDA is that we should look for features that quantify the abnormal "noise of a signal," and we expect faulty lines to have large amounts of noise and not faulty lines to have low amounts of noise. Thanks for reading this kernel and good luck in detecting faulty power lines!
# 
# 
# *Correction*: Versions 1 and 2 of this kernel incorrectly calculated median_amp by instead calculating the std of power spectrum amplitudes (effectively creating two std_amp features). Fixing this goof doesn't change my hypothesis about what kinds of features will do well in this competition, but it does even out the distribution of feature importance a bit. If you see any other issues with the kernel let me know or if you have any questions or discussion bits let me know.

# In[ ]:




