#!/usr/bin/env python
# coding: utf-8

# # Import libs and load data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
print(os.listdir("../input"))


# In[ ]:


# read data
df = pd.read_csv('../input/pmsm_temperature_data.csv')
df.head(10)
target_features = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']


# # Measurement session lengths
# The plot below shows that all measurement sessions range from 20 minutes to around 6 hours.
# The two short session ids "46" and "47" might be not very representative as temperatures inside electric motors need time to vary.

# In[ ]:


fig = plt.figure(figsize=(17, 5))
grpd = df.groupby(['profile_id'])
_df = grpd.size().sort_values().rename('samples').reset_index()
ordered_ids = _df.profile_id.values.tolist()
sns.barplot(y='samples', x='profile_id', data=_df, order=ordered_ids)
tcks = plt.yticks(2*3600*np.arange(1, 8), [f'{a} hrs' for a in range(1, 8)]) # 2Hz sample rate


# # Linear correlations

# In[ ]:


corr = df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.figure(figsize=(14,14))
_ = sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# We observe a very high positve linear correlation between *i_q* and *torque*.
# Moreover, *u_d* is highly negative linearly correlated with *torque* and *i_q*.
# Indeed, for the former insight we can refer to electric drive theory, where either higher torque is exclusively dependent on *i_q* in case of similar sized inductances in *d*- and *q*-axis, or increasing with higher *i_q* and slightly decreasing *i_d* elsewise (more common in practice).

# # Distributions

# In[ ]:


reduced_df = df.drop(['profile_id', 'stator_yoke', 'stator_tooth', 'stator_winding'], axis=1)
matplotlib.rcParams.update({'font.size': 22})
g = sns.PairGrid(reduced_df.sample(frac=0.01))
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);


# The kde-plots give a somewhat clearer hunch how the data is distributed and where the linear correlation comes from.
# Note, however, that these distributions where calculated on a random sample of 1% of the data, so skews and shifts might take place for the full sample.
# 
# We find
# * distributions are not very gaussian,
# * we often have multi-modal distributions,
# * the high negative linear correlation between *u_d* and *torque* or *i_q* happens on a fairly concentrated value range of *u_d*. Thus it's not of much help,
# * same observation with the lin. correlation between *torque* and *i_q*.
# 

# # Timeseries gestalt
# We plot a few recording ids to get an idea what the data looks like.
# In particular, we'll have a look at more or less representative recordings that are picked uniformly with respect to their session length.

# In[ ]:


def pick_equidistant_elems(m, n):
    """From a set of n elements, pick m which have equal distance to each other"""
    return [i*n//m + n//(2*m) for i in range(m)]

matplotlib.rcParams.update({'font.size': 12})
ids_to_plot = np.asarray(ordered_ids)[pick_equidistant_elems(6, len(ordered_ids))]
fig2 = plt.figure(figsize=(17, 10))
cols = len(ids_to_plot)
for i, (sess_id, _df) in enumerate([g for g in grpd if g[0] in ids_to_plot]):
    _df = _df.reset_index(drop=True)
    plt.subplot(4, cols, i+1)
    plt.xticks([])
    for target in target_features:
        plt.plot(_df[target], label=target)
    if i == 0:
        plt.legend(loc='upper right', bbox_to_anchor=(-0.2, 1.0))
    plt.subplot(4, cols, i+1+cols)
    plt.plot(_df['motor_speed'], color='green', label='motor_speed')
    plt.xticks([])
    if i == 0:
        plt.legend(loc='upper right', bbox_to_anchor=(-0.2, 1.0))
    plt.subplot(4, cols, i+1+cols*2)
    plt.plot(_df['torque'], color='yellow', label='torque')
    plt.xticks([])
    if i == 0:
        plt.legend(loc='upper right', bbox_to_anchor=(-0.2, 1.0))
    plt.subplot(4, cols, i+1+cols*3)
    plt.plot(_df['coolant'], color='cyan', label='coolant')
    k = int(len(_df)> 4*3600) + 1
    plt.xticks(_df['coolant'].index.values[::k*3600], _df.coolant.index.values[::k*3600] / (2*3600))
    plt.xlabel('time in hours')
    if i == 0:
        plt.legend(loc='upper right', bbox_to_anchor=(-0.2, 1.0))
    
plt.tight_layout()


# We find:
# * While motor excitations (motor_speed, torque, coolant) are sometimes of high dynamic, sometimes of stepwise nature, target temperatures always exhibit low-pass behavior with exponential rise and falls,
# * Coolant temperature suffers from measurement artefacts expressed by sharp drops in temperature, which recover as fast,
# * PM (Permanent Magnet -> Rotor) temperature expresses the slowest time constant and follows stator temperatures

# ## Dimensionality reduced visualization
# We further depict all recording sessions in terms of their principal component axes, shifting the color from blue to red as the permanent magnet temperature rises.
# 
# ### All profiles separated
# Showing the two most significant principal components.

# In[ ]:


_df = df.loc[~df.profile_id.isin([46, 47])].reset_index(drop=True)
transformed = PCA().fit_transform(_df.drop(['profile_id']+target_features, axis=1))
N = len(_df.profile_id.unique())
cols = min(10, N)
rows = np.ceil(N/10)
plt.figure(figsize=(2*cols, 2*rows))
for i, (sess_id, sess_df) in enumerate(_df.groupby(['profile_id'])):
    plt.subplot(rows, cols, i+1)
    _trans = transformed[sess_df.index, :]
    plt.scatter(_trans[:, 0], _trans[:, 1], c=_df.loc[sess_df.index, 'pm'].values, cmap=plt.get_cmap('coolwarm'), marker='.', vmin=_df['pm'].min(), vmax=_df['pm'].max())
    plt.xlim(-6, 6)
    plt.ylim(-5, 5)
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                            labelbottom=False, right=False, left=False,
                            labelleft=False)
    plt.annotate(str(sess_id), (4.8, -4.8))
plt.tight_layout()


# It becomes obvious that lower profile IDs are of simpler driving cycles, not moving much in feature space.
# Higher profile IDs are driving cycles of high dynamics - excitation happened through random walks in the motor_speed-torque-plane.
# 
# ### All profiles condensed 
# Showing increasing principal component significance from right to left.

# In[ ]:


fig = plt.figure(figsize=(17, 3))
cols = 4
for i in range(cols):
    plt.subplot(1, cols, i+1)
    plt.scatter(transformed[:, i], transformed[:, i+1], c=_df.pm.values, cmap=plt.get_cmap('coolwarm'), marker='.', vmin=_df['pm'].min(), vmax=_df['pm'].max())
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
plt.show()


# We find:
# * No gaussian distributions are recognizable,
# * nor is the target temperature distinguishable in the spatial dimension,
# * good features need to be found, different from the raw sensor data.
# 
# # Baseline Predictions with Ordinary Least Squares and Random Forest Regression
# In the following we try to predict the permanent magnet temperature *pm*.

# In[ ]:


def evaluate_baseline(data):
    # Use k-fold CV to measure generalizability
    trainset = data.loc[:, [c for c in data.columns if c not in ['profile_id']+target_features]]
    target = data.loc[:, 'pm']

    ols = LinearRegression(fit_intercept=False)
    print('Start fitting OLS...')
    scores = cross_val_score(ols, trainset, target, cv=5, scoring='neg_mean_squared_error')
    print(f'OLS MSE: {-scores.mean():.4f} (+/- {scores.std()*2:.3f})\n')  # mean and 95% confidence interval

    rf = RandomForestRegressor(n_estimators=20, n_jobs=-1)
    print('Start fitting RF...')
    scores = cross_val_score(rf, trainset, target, cv=5, scoring='neg_mean_squared_error')
    print(f'RF MSE: {-scores.mean():.4f} (+/- {scores.std()*2:.3f})\n')  # mean and 95% confidence interval


# ## Fitting on raw data

# In[ ]:


evaluate_baseline(df)


# ## Feature Engineering
# We add some static features, that we can infer from the given raw signals and might help expose relevant patterns.

# In[ ]:


extra_feats = {
     'i_s': lambda x: np.sqrt(x['i_d']**2 + x['i_q']**2),  # Current vector norm
     'u_s': lambda x: np.sqrt(x['u_d']**2 + x['u_q']**2),  # Voltage vector norm
     'S_el': lambda x: x['i_s']*x['u_s'],                  # Apparent power
     'P_el': lambda x: x['i_d'] * x['u_d'] + x['i_q'] *x['u_q'],  # Effective power
     'i_s_x_w': lambda x: x['i_s']*x['motor_speed'],
     'S_x_w': lambda x: x['S_el']*x['motor_speed'],
}
df = df.assign(**extra_feats)


# Moreover, the trend in the raw signals is of very high information as has been mentioned in literature (see [ResearchGate Paper](https://www.researchgate.net/publication/331976678_Empirical_Evaluation_of_Exponentially_Weighted_Moving_Averages_for_Simple_Linear_Thermal_Modeling_of_Permanent_Magnet_Synchronous_Machines)).
# We can compute the trend by calculating _exponentially weighted moving averages_ (EWMA). Note, that this is nothing more than low-pass filtering the signals.

# In[ ]:


spans = [6360, 3360, 1320, 9480]  # these values correspond to cutoff-frequencies in terms of low pass filters, or half-life in terms of EWMAs, respectively
max_span = max(spans)
enriched_profiles = []
for p_id, p_df in df.groupby(['profile_id']):
    target_df = p_df.loc[:, target_features].reset_index(drop=True)
    # pop out features we do not want to calculate the EWMA from
    p_df = p_df.drop(target_features + ['profile_id'], axis=1).reset_index(drop=True)
    
    # prepad with first values repeated until max span in order to get unbiased EWMA during first observations
    prepadding = pd.DataFrame(np.zeros((max_span, len(p_df.columns))),
                              columns=p_df.columns)
    temperature_cols = [c for c in ['ambient', 'coolant'] if c in df]
    prepadding.loc[:, temperature_cols] = p_df.loc[0, temperature_cols].values

    # prepad
    prepadded_df = pd.concat([prepadding, p_df], axis=0, ignore_index=True)
    ewma = pd.concat([prepadded_df.ewm(span=s).mean().rename(columns=lambda c: f'{c}_ewma_{s}') for s in spans], axis=1).astype(np.float32)
    ewma = ewma.iloc[max_span:, :].reset_index(drop=True)  # remove prepadding
    assert len(p_df) == len(ewma) == len(target_df), f'{len(p_df)}, {len(ewma)}, and {len(target_df)} do not match'
    new_p_df = pd.concat([p_df, ewma, target_df], axis=1)
    new_p_df['profile_id'] = p_id
    enriched_profiles.append(new_p_df.dropna())
enriched_df = pd.concat(enriched_profiles, axis=0, ignore_index=True)  

# normalize
p_ids = enriched_df.pop('profile_id')
scaler = StandardScaler()
enriched_df = pd.DataFrame(scaler.fit_transform(enriched_df), columns=enriched_df.columns)
# please note that we standardize the full data here, yet this is not statistically sound procedure.
# In order to get an unflawed generalization measure of any model evaluated on the data only the training set should be used for fitting the scaler.
# Depending on the CV used this might mean to scale repeatedly with different subsets


# ## Fitting on engineered data

# In[ ]:


enriched_df.head()


# In[ ]:


evaluate_baseline(enriched_df)


# 
