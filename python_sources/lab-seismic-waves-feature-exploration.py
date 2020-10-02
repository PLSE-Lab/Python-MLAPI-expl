#!/usr/bin/env python
# coding: utf-8

# ## General Infomation
# 
# Correct prediction and warning on earthquake indirectly saves lifes from damage of buildings.
# 
# In this competition, we train on large dataset having several laboratory earthquakes continuously,
# and predict `time_to_failure` in each last sample of segment to next laboratory earthquake.
# Each segment we usually have about 150,000 continous samples.
# 
# ## Summary on EDA
# 
# * Sampling rate is about 3.853 MHz
# 
# * `time_to_failure` within a lab earthquake is in stairs-like descending order
# 
# * About 0.31sec after each huge fluctuations, next laboratory earthquake comes.
# 
# * After Fast Fourier Transformation, the main difference between huge fluctuation chunks and others is their distribution of real numbers.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import gc # garbage collection

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# In[ ]:


# change plot style and default figure size
# plt.style.use('seaborn')
plt.rc("font", size=13)
plt.rc("figure", figsize=(14.4, 8.1), dpi=72)
# plt.rc("savefig", dpi=72)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = pd.read_csv("../input/train.csv", dtype={\'acoustic_data\': \'int16\', \'time_to_failure\': \'float64\'})\ngc.collect()\n# print(len(train_df))  # >> length of full_dataframe: 629,145,480')


# In[ ]:


# plot slice in dataframe
def df_slice_plot(df, start=0, stop=None, step=1, figsize=None):

    if not stop:
        start, stop = 0, start
    if start == 0 and start == stop:
        stop = len(df)

    ps = ""
    if step > 1:
        ps = "(down sampled)"

    df_slice = df.iloc[start:stop:step]

    df_slice.acoustic_data.plot(kind='line', label="Signals", figsize=figsize, legend=True)
    df_slice.time_to_failure.plot(kind='line', label="TimeToFailure", figsize=figsize, legend=True, secondary_y=True)
    plt.title("Signals and time_to_failure within a slice from {:,d} to {:,d} {}".format(start, stop, ps))
    plt.show()

    gc.collect()


def before_next_lab_earthquake(df, start=0, stop=None):
    
    if not stop:
        start, stop = 0, start
    
    df_slice = df.iloc[start:stop].copy().reset_index()
        
    sig_top_at = df_slice.acoustic_data.idxmax()
    sig_bot_at = df_slice.acoustic_data.idxmin()
    t_2_fail_min_at = df_slice.time_to_failure.idxmin()
    
    top_2_closest = df_slice.time_to_failure.iloc[sig_top_at] - df_slice.time_to_failure.min()
    bot_2_closest = df_slice.time_to_failure.iloc[sig_bot_at] - df_slice.time_to_failure.min()
    
    top_2_closest_x = t_2_fail_min_at - sig_top_at
    bot_2_closest_x = t_2_fail_min_at - sig_bot_at
    
    sample_rate = top_2_closest_x / top_2_closest
    
    print((
        "In slice from {:,d} to {:,d}:\n"
        "Time from top value of signal to the bottom value of time_to_failure: {:.8f} sec.\n"
        "Time from bottom value of signal to the bottom value of time_to_failure: {:.8f} sec.").format(
            start, stop, top_2_closest, bot_2_closest))
    print((
        "The location of top val is {:,d} samples to the bottom value of time_to_failure.\n"
        "The location of bot val is {:,d} samples to the bottom value of time_to_failure.").format(
            top_2_closest_x, bot_2_closest_x))
    print("The sample rate is about: {} Hz.\n".format(sample_rate))


def plot_transformed(signals):

    zc = np.fft.fft(signals)
    freq = np.fft.fftfreq(signals.shape[-1])

    fig, axes = plt.subplots(2, 1, figsize=(14.4, 16.2))

    axes[0].plot(zc.real, zc.imag)
    axes[0].set_aspect("equal")
    axes[0].set_title("Fourier transformed signals")

    axes[1].plot(freq, zc.real, label="Real", alpha=0.6)
    axes[1].plot(freq, zc.imag, label="Image", alpha=0.5)
    axes[1].legend()
    axes[1].set_title("Fourier transformed signals in freq")

    plt.show()
    
    print("Mean in real: {:f}, in image: {:.16f}".format(zc.real.mean(), zc.imag.mean()))
    print("Std in real: {:f}, in image: {:.16f}".format(zc.real.std(), zc.imag.std()))


# ## Explore time_to_failure

# In[ ]:


train_df.time_to_failure.iloc[:50_000].plot(kind='line', title="time_to_failure within a lab earthquake (stairs-like)")
plt.show()

train_df.time_to_failure.iloc[:4_001].plot(kind='line', title="time_to_failure within a stair")
plt.show()

gc.collect()

train_df.time_to_failure.iloc[:50_000].diff().plot(kind='line', title="Diffs of time_to_failure within a lab earthquake")
plt.show()

train_df.time_to_failure.iloc[5_656_570:5_656_580].plot(
    kind='bar', logy=True,
    title="Logarithmic(base 10) time_to_failure between 2 laboratory earthquakes")
plt.show()

gc.collect()


# ## Explore the acoustic_data

# In[ ]:


# Down sampling is risky to loss peak in huge fluctuations about 0.31sec before next laboratory earthquake (compared with fig1 and fig4)
df_slice_plot(train_df, 10_000_000)  # fig1
df_slice_plot(train_df, 10_000_000, 30_000_000)  # fig2
df_slice_plot(train_df, 30_000_000, 60_000_000)  # fig3
df_slice_plot(train_df, step=20, figsize=(24, 10))  # fig4


# In[ ]:


slice_signal = train_df.acoustic_data.iloc[4_435_000:4_445_000]
slice_signal.plot(kind='line', title="Signals within huge fluctuations.")
plt.show()

plot_transformed(slice_signal)

gc.collect()


# ## Short before next lab earthquake

# In[ ]:


before_next_lab_earthquake(train_df, 4_000_000, 6_000_000)
before_next_lab_earthquake(train_df, 47_500_000, 51_000_000)

gc.collect()


# ## Explore first 5 test segments

# In[ ]:


seg_lst = list(os.listdir("../input/test/"))
for f in seg_lst[:5]:
    seg_df = pd.read_csv(f"../input/test/{f}")
    seg_df.plot()
    plt.title(f"{f}")
    plt.show()
    plot_transformed(seg_df.acoustic_data)
    del seg_df
    
gc.collect()


# ## Explore test segments having huge fluctuations

# In[ ]:


c = 0
for f in seg_lst:
    seg_df = pd.read_csv(f"../input/test/{f}")
    if seg_df.max().values > 2000:
        print(f"Huge fluctuations in {f}")
        seg_df.plot()
        plt.title(f"{f}")
        plt.show()
        plot_transformed(seg_df.acoustic_data)
        c += 1
    del seg_df
    if c >= 5:
        break
        
gc.collect()


# ## Is it time to build models?
# 
# * It's such a huge amount of samples that machine cannot learn quickly
# 
# * So we aggregate on features to make derived dataset

# In[ ]:


# drop the dataframe
del train_df
gc.collect()


# ## Feature Engineering
# 
# * Using threshold which does not gain feature importance does no helps.
# 
# * `min`, `max` have extremely high correlation coefficient with `std`.

# In[ ]:


derived_trn = pd.DataFrame()
train_rd = pd.read_csv("../input/train.csv", iterator=True, chunksize=150000)


# Derive from `acoustic_data`
def get_features(chunk):

    curr_df = pd.DataFrame()

    curr_df["mean"] = [chunk.acoustic_data.mean()]
    curr_df["std"] = [chunk.acoustic_data.std()]
    curr_df["kurtosis"] = [chunk.acoustic_data.kurtosis()]
    curr_df["skew"] = [chunk.acoustic_data.skew()]
    curr_df["quantile_05"], curr_df["quantile_25"], curr_df["quantile_75"], curr_df["quantile_95"] =         [[x] for x in chunk.acoustic_data.quantile([0.05, 0.25, 0.75, 0.95])]

    # Attempt to apply window function on `acoustic_data`
    win_width = 100
    slepian_width = 51
    windowed = chunk.acoustic_data.rolling(win_width, win_type='slepian').mean(width=slepian_width).dropna()
    curr_df[f"window_{win_width}_mean"] = [windowed.mean()]
    curr_df[f"window_[win_width]_std"] = [windowed.std()]

    # Fast Fourier Transformed arrays
    fft = np.fft.fft(chunk.acoustic_data)
    curr_df["fft_real_mean"] = [fft.real.mean()]
    curr_df["fft_image_mean"] = [fft.imag.mean()]
    curr_df["fft_real_std"] = [fft.real.std()]
    curr_df["fft_image_std"] = [fft.imag.std()]

    # Aggregate on beginning number of samples.
    # If a chunk (the last chunk) is not more than the num of samples, use all samples in the chunk.
    sig_first = chunk.acoustic_data.iloc[:20000]
    curr_df["mean_first"] = [sig_first.mean()]
    curr_df["std_first"] = [sig_first.std()]
    curr_df["kurtosis_first"] = [sig_first.kurtosis()]
    curr_df["skew_first"] = [sig_first.skew()]
    curr_df["quantile_05_first"], curr_df["quantile_25_first"],     curr_df["quantile_75_first"], curr_df["quantile_95_first"] = [[x] for x in sig_first.quantile([0.05, 0.25, 0.75, 0.95])]
    
    windowed_first = sig_first.rolling(win_width, win_type='slepian').mean(width=slepian_width).dropna()
    curr_df[f"window_{win_width}_mean_first"] = [windowed_first.mean()]
    curr_df[f"window_{win_width}_std_first"] = [windowed_first.std()]

    fft_first = np.fft.fft(sig_first)
    curr_df["fft_real_mean_first"] = [fft_first.real.mean()]
    curr_df["fft_image_mean_first"] = [fft_first.imag.mean()]
    curr_df["fft_real_std_first"] = [fft_first.real.std()]
    curr_df["fft_image_std_first"] = [fft_first.imag.std()]

    # Aggregate on last number of samples.
    sig_last = chunk.acoustic_data.iloc[-20000:]
    curr_df["mean_last"] = [sig_last.mean()]
    curr_df["std_last"] = [sig_last.std()]
    curr_df["kurtosis_last"] = [sig_last.kurtosis()]
    curr_df["skew_last"] = [sig_last.skew()]
    curr_df["quantile_05_last"], curr_df["quantile_25_last"],     curr_df["quantile_75_last"], curr_df["quantile_95_last"] = [[x] for x in sig_last.quantile([0.05, 0.25, 0.75, 0.95])]
    
    windowed_last = sig_last.rolling(win_width, win_type='slepian').mean(width=slepian_width).dropna()
    curr_df[f"window_{win_width}_mean_last"] = [windowed_last.mean()]
    curr_df[f"window_{win_width}_std_last"] = [windowed_last.std()]

    fft_last = np.fft.fft(sig_last)
    curr_df["fft_real_mean_last"] = [fft_last.real.mean()]
    curr_df["fft_image_mean_last"] = [fft_last.imag.mean()]
    curr_df["fft_real_std_last"] = [fft_last.real.std()]
    curr_df["fft_image_std_last"] = [fft_last.imag.std()]

    return curr_df


for chunk in train_rd:
    curr_df = get_features(chunk)
    curr_df["target"] = chunk.time_to_failure.iloc[-1]
    derived_trn = pd.concat([derived_trn, curr_df], ignore_index=True)
    gc.collect()


# In[ ]:


derived_trn.to_csv("derived_train.csv", index=False)
display(derived_trn.head())
display(derived_trn.loc[(derived_trn.target <= 0.33) & (derived_trn.target >= 0.3)].head())
features = [col for col in derived_trn.columns if col != "target"]


# In[ ]:


test_seg_list = list(os.listdir("../input/test/"))
derived_test = pd.DataFrame()

for csv_file in test_seg_list:
    curr_test_df = pd.read_csv(f"../input/test/{csv_file}", dtype={"acoustic_data": "int16"})

    curr_df = get_features(curr_test_df)
    curr_df["seg_id"] = [csv_file.split(".")[0]]

    derived_test = pd.concat([derived_test, curr_df], ignore_index=True)

    gc.collect()


# In[ ]:


derived_test.to_csv("derived_test.csv", index=False)
display(derived_test.head())


# ## Exploration on Derived datasets -- Is it worthy to train? Why?

# In[ ]:


display(derived_trn.describe())
display(derived_test.describe())


# In[ ]:


nbins=None
threshold=0.35
    
for col in features:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.distplot(derived_trn[col].loc[derived_trn["target"]>=threshold], bins=nbins, label=r"`target`>={}".format(threshold), ax=axes[0])
    sns.distplot(derived_trn[col].loc[derived_trn["target"]<threshold], bins=nbins, label=r"`target`<{}".format(threshold), ax=axes[0])
    axes[0].set_title("derived_trn")
    axes[0].legend()

    sns.distplot(derived_test[col], bins=nbins, ax=axes[1])
    axes[1].set_title("derived_test")
    plt.tight_layout()
    plt.show()

    gc.collect()


# In[ ]:


fig, axes = plt.subplots(2, 1, figsize=(22, 40))
sns.heatmap(data=derived_trn.corr().abs().round(2), ax=axes[0], annot=False)
axes[0].set_title("Correlation coefficients in derived_trn")
sns.heatmap(data=derived_test.corr().abs().round(2), ax=axes[1], annot=False)
axes[1].set_title("Correlation coefficients in derived_test")
plt.show()


# ## Model building and training -- Don't overfit

# In[ ]:


from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import mean_absolute_error

params = dict(
    objective='regression_l1',  # loss func, candidates: huber, regression_l1, regression_l2, fair
    n_estimators=5000, learning_rate=0.01,
    num_leaves=31, max_depth=-1,
#     min_split_gain=10,
#     min_child_weight=1,
#     min_child_samples=90,
#     subsample=1., subsample_freq=0,
#     colsample_bytree=1.,
    reg_alpha=0.3, reg_lambda=0.5,
#     ramdom_state=42,
    metric='mae',
)


# samples augmentation
def augment(df, threshold=None, repeat_times=1):
    
    df = df.copy()
    
    for _ in range(repeat_times):
        df_copy = (df.loc[df["target"] <= threshold].copy()
                   if threshold is not None
                   else df.copy())
        df = pd.concat([df, df_copy], ignore_index=True)
    
    df = df.sample(frac=1, random_state=42)  # shuffle samples
    return df


# In[ ]:


param_grid = dict(
    n_estimators=[1000], learning_rate=[0.01],
#     min_split_gain=[0.001, 10.],
)


def search_params(estimator, param_grid, params):
    
    ori_n_estimators = params["n_estimators"]
    
    gscv = GridSearchCV(estimator, cv=5, param_grid=param_grid, scoring='neg_mean_absolute_error', verbose=3)
    best = gscv.fit(derived_trn[features], derived_trn["target"])
    
    params.update(gscv.best_params_)
    params["n_estimators"] = ori_n_estimators  # keep n_estimators not changed
    print(gscv.best_params_)
    
    return params


# params = search_params(LGBMRegressor(**params), param_grid, params)  # No output if this line is commented


# In[ ]:


n_splits = 5
n_repeats = 25

rskf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=42)

predictions = pd.DataFrame()
feature_importances = pd.DataFrame()
oof = np.zeros((len(derived_trn), n_repeats))

for nf, (trn_idx, val_idx) in enumerate(rskf.split(derived_trn, (derived_trn['target'] <= 1.2))):
    
#     augmented_df = augment(derived_trn.iloc[trn_idx], 1, 2)
#     trn_features, trn_targets = augmented_df[features], augmented_df["target"]
    trn_features, trn_targets = derived_trn[features].iloc[trn_idx], derived_trn["target"].iloc[trn_idx]
    val_features, val_targets = derived_trn[features].iloc[val_idx], derived_trn["target"].iloc[val_idx]
    
    print("\nNf: {}, train on {} samples, val on {} samples".format(nf, len(trn_targets), len(val_targets)))
    
    model = LGBMRegressor(**params)
    gc.collect()

    fit_params = dict(
        eval_set=[(trn_features, trn_targets), (val_features, val_targets)],
        verbose=1000,
        early_stopping_rounds=1500,
    )

    model.fit(trn_features, trn_targets, **fit_params)
    
    curr_importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    feature_importances = pd.concat([feature_importances, curr_importance])
    
    oof[val_idx, nf // n_splits] = model.predict(val_features)
    
    curr_test_pred = model.predict(derived_test[features])
    predictions[f"pred_{nf}"] = curr_test_pred


# ## MAE in trainset and Feature importances

# In[ ]:


print("OOF's mae: {}".format(mean_absolute_error(derived_trn.target, oof.mean(axis=1))))
feature_importances.groupby("feature").mean().sort_values(by="importance", ascending=True).plot(kind="barh", figsize=(14.4, 10.8))
plt.show()


# ## Submission

# In[ ]:


submission_y = predictions.mean(axis=1).values
submission = pd.DataFrame({"seg_id": derived_test["seg_id"], "time_to_failure": submission_y})
submission.to_csv("submission.csv", index=False)
display(submission.head())
display(submission.tail())
print("Minimum time_to_failure in submission:", submission["time_to_failure"].min(), "sec.")


# ## Discussion
# 
# * How to exactly apply window function in signal processing?(I doubt that I did not exactly apply)
# 
# * Would smaller chunksize in trainset help to improve model?
