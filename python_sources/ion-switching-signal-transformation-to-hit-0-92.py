#!/usr/bin/env python
# coding: utf-8

# # Ion switching: signal transformation to hit 0.92
# 
# 1. EDA and manual batch detection
# 2. Trend removal
# 3. Noise detection and filtering
# 4. Unsupervised cluster learning
# 5. Target prediction

# ## 1. EDA and manual batch detection
# 
# There are batches with different durations (not only 500_000 samples) - let's set them manually.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df_train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
df_test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
df_train.shape, df_test.shape


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize=(16,4))
plt.title('Train signal and trend')
plt.plot(df_train['signal'], label='signal')
plt.plot(df_train['open_channels'], label='target', alpha=0.5)
plt.legend()


# In[ ]:


plt.figure(figsize=(16,4))
plt.title('Test signal')
plt.plot(df_test['signal'])


# In[ ]:


def prepare(df, limits):
    df['batch_idx'] = np.zeros_like(df['signal'], dtype=np.int8)
    df['local_time'] = np.zeros_like(df['signal'], dtype=np.int8)
    for idx, start, finish in zip(range(len(limits)-1), limits[:-1], limits[1:]):
        mask = np.arange(start, finish)
        df.loc[mask, 'batch_idx'] = idx
        df.loc[mask, 'local_time'] = df.loc[mask, 'time'] - df.loc[mask, 'time'].min()


# In[ ]:


TRAIN_LIMITS = [
    0,
    500_000,
    600_000,
    1_000_000,
    1_500_000,
    2_000_000,
    2_500_000,
    3_000_000,
    3_500_000,
    # 3_642_000,
    # 3_823_000,
    4_000_000,
    4_500_000,
    5_000_000,
]
prepare(df_train, TRAIN_LIMITS)
plt.plot(df_train['batch_idx'])
df_train['batch_idx'].unique().shape


# In[ ]:


TEST_LIMITS = [
    0,
    100_000,
    200_000,
    300_000,
    400_000,
    500_000,
    600_000,
    700_000,
    800_000,
    900_000,
    1_000_000,
    1_500_000,
    2_000_000,
]
prepare(df_test, TEST_LIMITS)
plt.plot(df_test['batch_idx'])
df_test['batch_idx'].unique().shape


# In[ ]:


df_train.head()


# ## 2. Trend removal
# 
# The idea is very simple - let's just fit different linear regression models with degrees 0 (constant), 1 (linear) and 2 (quadratic) for different batches and then substract this trend from the signal

# In[ ]:


TRAIN_DEGREES = [0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]
TEST_DEGREES = [1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 2, 0]


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


def calculate_trend(df, degrees, roll=None):
    if roll is not None:
        df['trend'] = np.zeros(df.shape[0])
    df['trend_degree'] = df['batch_idx'].apply(lambda x: degrees[x]).astype(np.int)
    for cn in ['trend_lr', 'w0', 'w1', 'w2']:
        df[cn] = np.zeros(df.shape[0])
    for idx in df['batch_idx'].unique():
        print('series #', idx)
        mask = df['batch_idx'] == idx
        time = df.loc[mask, 'local_time'].values.reshape(-1, 1)
        time_2 = np.hstack([time, time ** 2])
        signal = df.loc[mask, 'signal']
        if roll is not None:
            print('rolling...')
            df.loc[mask, 'trend'] = signal.rolling(roll, center=True, win_type='parzen').mean()
        print('regression...')
        reg = LinearRegression()
        if degrees[idx] == 2:
            reg.fit(time_2, signal)
            df.loc[mask, 'trend_lr'] = reg.predict(time_2)
            df.loc[mask, 'w2'] = reg.coef_[1]
        else:
            reg.fit(time, signal)
            df.loc[mask, 'trend_lr'] = reg.predict(time)
        df.loc[mask, 'w1'] = reg.coef_[0]
        df.loc[mask, 'w0'] = reg.intercept_
        print('coefs: ', reg.coef_, reg.intercept_)
    if roll is not None:
        df['signal-trend'] = df['signal'] - df['trend']
    df['signal-trend_lr'] = df['signal'] - df['trend_lr']


# In[ ]:


calculate_trend(df_train, TRAIN_DEGREES, roll=None)


# In[ ]:


df_train.head()


# In[ ]:


def plot_graph(df, test=False):
    plt.figure(figsize=(16,4))
    plt.plot(df['signal'], label='signal', alpha=0.5)
    plt.plot(df['batch_idx'], label='batch_idx', alpha=0.8)
    if 'trend' in df.columns:
        plt.plot(df['trend'], label='trend', alpha=0.8)
        plt.plot(df['signal-trend'], label='signal-trend', alpha=0.5)
    if 'trend_lr' in df.columns:
        plt.plot(df['trend_lr'], label='trend_lr', alpha=0.8)
        plt.plot(df['signal-trend_lr'], label='signal-trend_lr', alpha=0.5)
    if not test:
        plt.plot(df['open_channels'], label='open_channels', alpha=0.5)
    plt.legend(loc='upper left')


# In[ ]:


plot_graph(df_train)


# In[ ]:


plot_graph(df_train[996_000:998_000])


# In[ ]:


plot_graph(df_train[15_250:15_800])


# In[ ]:


plot_graph(df_train[503_000:505_000])


# In[ ]:


calculate_trend(df_test, TEST_DEGREES, roll=None)


# In[ ]:


plot_graph(df_test, test=True)


# ## 3. Noise detection and filtering
# 
# Core ideas used in this section:
# 
# - usage of Fast Fourier Transform (FFT) to detect the local peaks in the signal and the target;
# - if the local peak is present in the signal and absent in the target - it's a noise (systematic error) and has to be filtered

# In[ ]:


from scipy.fft import fft, fftfreq, ifft


# In[ ]:


def total_fft(df, sig='signal-trend_lr', limits=(-100, 100), logy=False, d=1e-4):
    sig_fft = fft(df[sig].values)
    power = np.abs(sig_fft)
    freq = fftfreq(df.shape[0], d=d)
    plt.figure(figsize=(12,4))
    l, r = limits
    mask = np.where((l <= freq) & (freq <= r))
    plt.plot(freq[mask], power[mask])
    plt.grid()
    if logy:
        plt.yscale('log')


# In[ ]:


total_fft(df_train)


# In[ ]:


total_fft(df_test)


# - 50Hz local peaks in train and test signals

# In[ ]:


total_fft(df_train, 'open_channels', (45, 55), logy=False)


# - there is no 50Hz local peak in target

# In[ ]:


def batch_ffts(df, sig='signal-trend_lr', limits=(-5_000, 5_000), logy=False, d=1e-4):
    n_rows, n_cols = 4, 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20,20))
    for bi, dfg in df.groupby('batch_idx'):
        idx_row, idx_col = bi // n_cols, bi % n_cols
        ax = axs[idx_row][idx_col]
        ax.set_title(f'batch {bi}')
        sig_fft = fft((dfg[sig] - dfg[sig].mean()).values)
        power = np.abs(sig_fft)
        freq = fftfreq(dfg.shape[0], d=d)
        l, r = limits
        mask = np.where((l <= freq) & (freq <= r))
        ax.plot(freq[mask], power[mask])
        ax.grid()
        if logy:
            ax.set_yscale('log')


# In[ ]:


batch_ffts(df_train, limits=(45, 55))


# In[ ]:


batch_ffts(df_train, limits=(0, 0.2))


# - frequencies to filter:
#   - 50Hz - all batches
#   - 0.04Hz - batches 7-10 (quadratic)

# In[ ]:


batch_ffts(df_train, 'open_channels', (45, 55), logy=True)


# In[ ]:


batch_ffts(df_train, 'open_channels', (0, 0.2), logy=True)


# - no 50Hz or 0.04Hz local peaks in target

# In[ ]:


batch_ffts(df_test, limits=(45, 55))


# In[ ]:


batch_ffts(df_test, limits=(0, 0.2))


# #### Naive filtering: setting frequency amp to zero

# In[ ]:


np.random.seed(42)

def filter_freq(df, sig='signal-trend_lr', means=[50], widths=[0.4],
                batch_idxs=None, limits=(-100, 100), d=1e-4):
    total_fft(df, sig, limits)
    for bi, dfg in df.groupby('batch_idx'):
        sig_fft = fft(dfg[sig].values)
        freq = fftfreq(dfg.shape[0], d=d)
        for mean, width in zip(means, widths):
            if batch_idxs is not None and bi not in batch_idxs:
                continue
            l, r = mean - width/2, mean + width/2
            print('batch', bi, 'limits', l, r)
            mask = (np.abs(freq) > l) & (np.abs(freq) <= r)
            sig_fft[mask] = 0
        power = np.abs(sig_fft)
        mask = df['batch_idx'] == bi
        df.loc[mask, sig+'-f'] = ifft(sig_fft)
    df[sig+'-f'] = df[sig+'-f'].apply(lambda x: x.real)
    total_fft(df, sig+'-f', limits)


# In[ ]:


filter_freq(df_train, means=[50, 0.04], 
            widths=[0.4, 0.01], limits=(45, 55))


# In[ ]:


batch_ffts(df_train, 'signal-trend_lr-f', (45, 55))


# In[ ]:


filter_freq(df_test, means=[50, 0.04], 
            widths=[0.4, 0.01], limits=(0, 0.2))


# In[ ]:


batch_ffts(df_test, 'signal-trend_lr-f', (0, 0.2))


# In[ ]:


plt.title('Signal delta')
plt.plot(df_train.loc[np.arange(2_000_000, 2_005_000), 'signal-trend_lr'] - df_train.loc[np.arange(2_000_000, 2_005_000), 'signal-trend_lr-f'])


# In[ ]:


plt.title('Signal delta')
plt.plot(df_train.loc[np.arange(1_998_000, 2_002_000), 'signal-trend_lr'] - df_train.loc[np.arange(1_998_000, 2_002_000), 'signal-trend_lr-f'])


# In[ ]:


def plot_batch_filtered(df, batch_idx=0):
    dfb = df[df['batch_idx'] == batch_idx]
    plt.figure(figsize=(16,4))
    plt.title(f'Batch {batch_idx}')
    plt.plot(dfb['signal-trend_lr'], label='signal', alpha=0.5)
    plt.plot(dfb['signal-trend_lr-f'], label='filtered', alpha=0.5)
    plt.legend()


# In[ ]:


plot_batch_filtered(df_train, 7)
plot_batch_filtered(df_train, 10)


# In[ ]:


plot_batch_filtered(df_train, 0)
plot_batch_filtered(df_train, 2)


# ## 4. Unsupervised cluster learning
# 
# Core ideas used in this section:
# - undersampling to achieve the balance of the classes
# - gaussian distributions of the samples in each class:
#   - usage of the gaussian mixture (GM) models to find the cluster means on train and to separate the samples on test
#   - different models for batches with different number of the classes
#     - for batches with number of the classes > `6` (batch_idx == 5 or 10) let's use model with n_components=9 (ignoring classes `0` and `1` because there is very few samples of classes `0` and `1`) and then perform calculation of the means of the classes `0` and `1` using the mean of the class deltas
# - calculation of the "signal ground" (the mean of the class `0`) and then substracting it from the signal
# - final single GM model for all the batches
#   - usage of the predefined means for 'good' class labels (target-like)

# ###### Target classes distributions in batches

# In[ ]:


def batch_signals(df, logy=False):
    n_rows, n_cols = 4, 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20,20))
    for bi in df['batch_idx'].unique():
        print(f'----- batch {bi} -----')
        mask = df['batch_idx'] == bi
        idx_row, idx_col = bi // n_cols, bi % n_cols
        ax = axs[idx_row][idx_col]
        ax.set_title(f'batch {bi}')
        if 'open_channels' in df.columns:
            means = []
            for label in np.sort(df.loc[mask, 'open_channels'].unique()):
                mask_label = df['open_channels'] == label
                means += [df.loc[mask & mask_label, 'signal-trend_lr-f'].mean()]
                ax.hist(df.loc[mask & mask_label, 'signal-trend_lr-f'], bins=100, alpha=0.6, label=label)
            ax.legend()
            print('means:', means)
        else:
            ax.hist(df.loc[mask, 'signal-trend_lr-f'], bins=200)
        ax.grid()
        if logy:
            ax.set_yscale('log')


# In[ ]:


batch_signals(df_train)


# In[ ]:


batch_signals(df_train, logy=True)


# In[ ]:


batch_signals(df_test)


# In[ ]:


batch_signals(df_test, logy=True)


# ###### Fitting different GM models

# In[ ]:


TRAIN_COMPS = [20, 20, 20, 25, 40, 90, 60, 25, 40, 60, 90]
TEST_COMPS = [20, 40, 60, 20, 25, 90, 60, 90, 20, 40, 20, 20]


# In[ ]:


train_comp_dict = {k: v for k, v in zip(range(11), TRAIN_COMPS)}
test_comp_dict = {k: v for k, v in zip(range(12), TEST_COMPS)}


# In[ ]:


from collections import defaultdict
train_comp_dict_inv = defaultdict(list)
test_comp_dict_inv = defaultdict(list)


# In[ ]:


for k, v in train_comp_dict.items():
    train_comp_dict_inv[v].append(k)
for k, v in test_comp_dict.items():
    test_comp_dict_inv[v].append(k)


# In[ ]:


train_comp_dict_inv, test_comp_dict_inv


# In[ ]:


from imblearn.under_sampling import RandomUnderSampler
from sklearn.mixture import GaussianMixture


# In[ ]:


plt.figure(figsize=(12,4))
plt.title('Means')

df_train['signal_ground'] = np.zeros(df_train.shape[0], dtype=np.float32)
df_test['signal_ground'] = np.zeros(df_test.shape[0], dtype=np.float32)
df_train['n_components'] = np.zeros(df_train.shape[0], dtype=np.byte)
df_test['n_components'] = np.zeros(df_test.shape[0], dtype=np.byte)

deltas = []
means_dict = {}
for group in sorted(train_comp_dict_inv.keys()):
    n_components = group // 10
    train_batch_idxs = train_comp_dict_inv[group]
    mask = df_train['batch_idx'].isin(train_batch_idxs)
    if n_components == 9:
        mask = mask & df_train['open_channels'].isin(range(2, 11))
    X_train = df_train.loc[mask, 'signal-trend_lr-f'].values.reshape(-1, 1)
    y_train = df_train.loc[mask, 'open_channels'].values.reshape(-1, 1)
    samp = RandomUnderSampler(random_state=42)
    X_train_samp, _ = samp.fit_sample(X_train, y_train)
    print(f'----- Resampled to {X_train_samp.shape[0]} samples -----')
    print(f'Fitting mixture of {n_components} ...')
    gm = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=42, verbose=1)
    gm.fit(X_train_samp)
    means = gm.means_[:, 0]
    means_dict[group] = means
    print('Means(sorted):', sorted(means))
    plt.bar(means - np.min(means), [n_components] * means.shape[0], 
            alpha=0.5, width=0.1, label=group)
    deltas += [b - a for a, b in zip(sorted(means)[:-1], sorted(means)[1:])]
    ground = np.min(means)
    mask = df_train['batch_idx'].isin(train_batch_idxs)
    df_train.loc[mask, ['signal_ground', 'n_components']] = ground, n_components
    test_batch_idxs = test_comp_dict_inv[group]
    mask = df_test['batch_idx'].isin(test_batch_idxs)
    df_test.loc[mask, ['signal_ground', 'n_components']] = ground, n_components

plt.legend()


# In[ ]:


plt.hist(deltas, bins=5)


# In[ ]:


delta = np.median(deltas)
delta


# In[ ]:


df_train['gm_label'] = np.zeros(df_train.shape[0], dtype=np.int8)
df_test['gm_label'] = np.zeros(df_test.shape[0], dtype=np.int8)

for group in sorted(train_comp_dict_inv.keys()):
    n_components = group // 10
    if n_components != 9:
        continue
    train_batch_idxs = train_comp_dict_inv[group]
    mask = df_train['batch_idx'].isin(train_batch_idxs)
    df_train.loc[mask, 'signal_ground'] -= 2*delta
    test_batch_idxs = test_comp_dict_inv[group]
    mask = df_test['batch_idx'].isin(test_batch_idxs)
    df_test.loc[mask, 'signal_ground'] -= 2*delta


# In[ ]:


df_train['signal_ground'].unique()


# In[ ]:


df_train['signal-trend_lr-ground'] = df_train['signal-trend_lr-f'] - df_train['signal_ground']
df_test['signal-trend_lr-ground'] = df_test['signal-trend_lr-f'] - df_test['signal_ground']


# ###### Fitting final single GM model for all samples

# In[ ]:


X_train = df_train['signal-trend_lr-ground'].values.reshape(-1, 1)
y_train = df_train['open_channels'].values.reshape(-1, 1)
samp = RandomUnderSampler(random_state=42)
X_train_samp, _ = samp.fit_sample(X_train, y_train)
plt.figure()
plt.title('Resampled train')
plt.hist(X_train_samp, bins=100)
gm = GaussianMixture(n_components=11, means_init=np.arange(11).reshape(-1, 1)*delta,
                     random_state=42, verbose=1)
gm.fit(X_train_samp)
df_train['gm_label'] = gm.predict(X_train)
means = gm.means_[:, 0]
print('Means:', means)
print('Mean weights:', [f'{k} : {v}' for k, v in zip(means, gm.weights_)])
plt.figure()
plt.title('Means GMM')
plt.bar(means - np.min(means), [n_components] * means.shape[0], 
        width=0.1, label=group)
deltas = [b - a for a, b in zip(sorted(means)[:-1], sorted(means)[1:])]
print('Deltas', deltas)
print('Delta', np.median(deltas))
X_test = df_test['signal-trend_lr-ground'].values.reshape(-1, 1)
df_test['gm_label'] = gm.predict(X_test)


# In[ ]:


df_train['gm_label'].unique()


# In[ ]:


df_test['gm_label'].unique()


# In[ ]:


set(df_train['gm_label'].unique()).difference(df_test['gm_label'].unique())


# In[ ]:


set(df_test['gm_label'].unique()).difference(df_train['gm_label'].unique())


# In[ ]:


plt.figure(figsize=(6,6))
plt.imshow(df_train.corr())
plt.xticks(range(df_train.shape[1]), df_train.columns, rotation=75)
plt.yticks(range(df_train.shape[1]), df_train.columns)
plt.colorbar()


# In[ ]:


df_train.corr()['open_channels']


# In[ ]:


error = df_train['open_channels'] - df_train['gm_label']


# In[ ]:


plt.figure(figsize=(16,4))
plt.plot(error.loc[np.arange(2_000_600,2_000_700)])


# In[ ]:


np.mean(df_train['open_channels'] == df_train['gm_label'])


# In[ ]:


from sklearn.metrics import f1_score


# In[ ]:


f1_score(df_train['open_channels'], df_train['gm_label'], average='macro')


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(df_train['open_channels'], df_train['gm_label'], digits=3))


# In[ ]:


def plot_graph(df, mask, levels=None, show_batch=False):
    plt.figure(figsize=(16, 8))
    plt.plot(df.loc[mask, 'signal-trend_lr-ground'], label='signal', alpha=0.9)
    if levels is not None:
        for lid, (level, level_next) in enumerate(zip(levels[:-1], levels[1:])):
            color = 'red' if lid % 2 == 0 else 'blue'
            plt.axhline((level + level_next) / 2, color=color, alpha=0.5, linestyle='--')
    if 'open_channels' in df.columns:
        plt.plot(df.loc[mask, 'open_channels'], label='open_channels', alpha=0.9)
    if 'gm_label' in df.columns:
        plt.plot(df.loc[mask, 'gm_label'], label='gm_label', alpha=0.5)
    if show_batch:
        plt.plot(df.loc[mask, 'batch_idx'], label='batch_idx', alpha=0.8)
    plt.grid()
    plt.legend(loc='upper left')


# In[ ]:


plot_graph(df_train, np.arange(2_000_600, 2_000_700), means)


# In[ ]:


plot_graph(df_train, np.arange(2_001_600, 2_001_700), means)


# ## 5. Target prediction
# 
# Here is a GM label used as a target prediction

# In[ ]:


y_pred = df_test['gm_label']


# In[ ]:


subm = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')


# In[ ]:


subm['open_channels'] = y_pred


# In[ ]:


SUBM = 'subm_111'
FN = f'{SUBM}.csv'
FN_ZIP = f'{FN}.zip'
compression_opts = dict(method='zip', archive_name=FN) 
subm.to_csv(FN_ZIP, index=False,
            float_format='%.4f',
            compression=compression_opts)


# In[ ]:





# In[ ]:





# In[ ]:




