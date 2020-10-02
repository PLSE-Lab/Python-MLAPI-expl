#!/usr/bin/env python
# coding: utf-8

# # **On Monte Carlo Simulations**

# This kernel will cover: 
# * Metropolis-Hastings algorithm
# * Hamiltonian Monte Carlo

# > # **1. On Metropolis-Hastings**

# ## What is Metropolis-Hastings?
# 
# Metropolis-Hastings approach allows us to obtain random samples from a distribution where direct samping is difficult. They are used for sampling from multi-dimensional distributions.

# ## Our application 

# I am taking a **VERY SIMPLE** approach and using Ridge and Logistic regression.

# I have also taken a look at <a href="https://bair.berkeley.edu/blog/2017/08/02/minibatch-metropolis-hastings/">Bair's approach towards minibatch Metropolis.</a> Please take a look there if you wish to learn more.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Reading in the data.

# In[ ]:


df=pd.read_csv('../input/liverpool-ion-switching/train.csv')
test = pd.read_csv('../input/liverpool-ion-switching/test.csv')


# ## Implementing Metropolis

# In[ ]:


def log_posterior(theta, X, p_mean, p_cov, x_var):
    inv_cov = np.linalg.inv(p_cov)
    determinant = np.prod(np.diagonal(p_cov))
    prior = np.log(1. / (2*np.pi*np.sqrt(determinant)))

    ll_constant = 1. / (np.sqrt(2 * np.pi * x_var))
    L = 0.5 * np.exp(-(1./(2*x_var)) * (X - theta[0])**2) +         0.5 * np.exp(-(1./(2*x_var)) * (X - (theta[0]+theta[1]))**2)
    L *= ll_constant
    log_likelihood = np.sum(np.log(L))
    
    assert not np.isnan(prior + log_likelihood)
    return np.squeeze(prior + log_likelihood)
    

def get_noise(eps):
    """ Returns a 2-D multivariate normal vector with covariance matrix diag(eps,eps). """
    return (np.random.multivariate_normal(np.array([0,0]), eps*np.eye(2))).reshape((2,1))


def get_info_for_contour(K=100):
    """ For building the contour plots. """
    xlist = np.linspace(-1.5, 2.5, num=K)
    ylist = np.linspace(-3, 3, num=K)
    X_a,Y_a = np.meshgrid(xlist, ylist)
    Z_a = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            theta = np.array( [[X_a[i,j]],[Y_a[i,j]]] )
            Z_a[i,j] = log_posterior(theta, X, prior_mean, prior_cov, sigmax_sq) 
    return X_a, Y_a, Z_a


# In[ ]:


data = pd.merge(df, test)
x_train, y_train, x_test, y_test = train_test_split(df.head(2000000), test, test_size=0.15)


# In[ ]:


def logistic_run():
    clf = LogisticRegression(random_state=0)
    clf.fit(np.array(x_train.head(300000)['open_channels']).reshape(-1, 1), np.array(y_train.open_channels).reshape(-1, 1))
    x = clf.predict(np.array(df['open_channels']).reshape(-1, 1))
    x = np.array(x)
    x = pd.DataFrame({'time': df.time, 'open_channels': x})
    x.to_csv('log_sub.csv')
    x.head()


# In[ ]:


logistic_run()


# ## LGBM

# ## Credit to Rob Mulla

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import lightgbm as lgb
#from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_squared_error, f1_score
from datetime import datetime
import os

###########
# SETTINGS
###########


TARGET = 'open_channels'

TOTAL_FOLDS = 5
RANDOM_SEED = 529
MODEL_TYPE = 'LGBM'
LEARNING_RATE = 0.009
SHUFFLE = True
NUM_BOOST_ROUND = 500_000
EARLY_STOPPING_ROUNDS = 100
N_THREADS = -1
OBJECTIVE = 'regression'
METRIC = 'rmse'
NUM_LEAVES = 2**8+1
MAX_DEPTH = -1
FEATURE_FRACTION = 1
BAGGING_FRACTION = 1
BAGGING_FREQ = 0

####################
# READING IN FILES
####################

train = pd.read_csv('../input/liverpool-ion-switching/train.csv')
test = pd.read_csv('../input/liverpool-ion-switching/test.csv')
ss = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')
train['train'] = True
test['train'] = False
tt = pd.concat([train, test], sort=False).reset_index(drop=True)
tt['train'] = tt['train'].astype('bool')

###########
# TRACKING
###########

run_id = "{:%m%d_%H%M}".format(datetime.now())


def update_tracking(
        run_id, field, value, csv_file="./tracking.csv",
        integer=False, digits=None, nround=6,
        drop_broken_runs=False):
    """
    Tracking function for keep track of model parameters and
    CV scores. `integer` forces the value to be an int.
    """
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except:
        df = pd.DataFrame()
    if drop_broken_runs:
        df = df.dropna(subset=['1_f1'])
    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    df.loc[run_id, field] = value  # Model number is index
    df = df.round(nround)
    df.to_csv(csv_file)


# Update Tracking

update_tracking(run_id, 'model_type', MODEL_TYPE)
update_tracking(run_id, 'seed', RANDOM_SEED, integer=True)
update_tracking(run_id, 'nfolds', TOTAL_FOLDS, integer=True)
update_tracking(run_id, 'lr', LEARNING_RATE)
update_tracking(run_id, 'shuffle', SHUFFLE)
update_tracking(run_id, 'boost_rounds', NUM_BOOST_ROUND)
update_tracking(run_id, 'es_rounds', EARLY_STOPPING_ROUNDS)
update_tracking(run_id, 'threads', N_THREADS)
update_tracking(run_id, 'objective', OBJECTIVE)
update_tracking(run_id, 'metric', METRIC)
update_tracking(run_id, 'num_leaves', NUM_LEAVES)
update_tracking(run_id, 'max_depth', MAX_DEPTH)
update_tracking(run_id, 'feature_fraction', FEATURE_FRACTION)
update_tracking(run_id, 'bagging_fraction', BAGGING_FRACTION)
update_tracking(run_id, 'bagging_freq', BAGGING_FREQ)

###########
# FEATURES
###########

# # Include batch
tt = tt.sort_values(by=['time']).reset_index(drop=True)
tt.index = ((tt.time * 10_000) - 1).values
tt['batch'] = tt.index // 50_000
tt['batch_index'] = tt.index - (tt.batch * 50_000)
tt['batch_slices'] = tt['batch_index'] // 5_000
tt['batch_slices2'] = tt.apply(lambda r: '_'.join(
    [str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

# 50_000 Batch Features
tt['signal_batch_min'] = tt.groupby('batch')['signal'].transform('min')
tt['signal_batch_max'] = tt.groupby('batch')['signal'].transform('max')
tt['signal_batch_std'] = tt.groupby('batch')['signal'].transform('std')
tt['signal_batch_mean'] = tt.groupby('batch')['signal'].transform('mean')
tt['mean_abs_chg_batch'] = tt.groupby(['batch'])['signal'].transform(
    lambda x: np.mean(np.abs(np.diff(x))))
tt['abs_max_batch'] = tt.groupby(
    ['batch'])['signal'].transform(lambda x: np.max(np.abs(x)))
tt['abs_min_batch'] = tt.groupby(
    ['batch'])['signal'].transform(lambda x: np.min(np.abs(x)))

tt['range_batch'] = tt['signal_batch_max'] - tt['signal_batch_min']
tt['maxtomin_batch'] = tt['signal_batch_max'] / tt['signal_batch_min']
tt['abs_avg_batch'] = (tt['abs_min_batch'] + tt['abs_max_batch']) / 2

# 5_000 Batch Features
tt['signal_batch_5k_min'] = tt.groupby(
    'batch_slices2')['signal'].transform('min')
tt['signal_batch_5k_max'] = tt.groupby(
    'batch_slices2')['signal'].transform('max')
tt['signal_batch_5k_std'] = tt.groupby(
    'batch_slices2')['signal'].transform('std')
tt['signal_batch_5k_mean'] = tt.groupby(
    'batch_slices2')['signal'].transform('mean')
tt['mean_abs_chg_batch_5k'] = tt.groupby(['batch_slices2'])[
    'signal'].transform(lambda x: np.mean(np.abs(np.diff(x))))
tt['abs_max_batch_5k'] = tt.groupby(['batch_slices2'])[
    'signal'].transform(lambda x: np.max(np.abs(x)))
tt['abs_min_batch_5k'] = tt.groupby(['batch_slices2'])[
    'signal'].transform(lambda x: np.min(np.abs(x)))

tt['range_batch_5k'] = tt['signal_batch_5k_max'] - tt['signal_batch_5k_min']
tt['maxtomin_batch_5k'] = tt['signal_batch_5k_max'] / tt['signal_batch_5k_min']
tt['abs_avg_batch_5k'] = (tt['abs_min_batch_5k'] + tt['abs_max_batch_5k']) / 2


# add shifts
tt['signal_shift+1'] = tt.groupby(['batch']).shift(1)['signal']
tt['signal_shift-1'] = tt.groupby(['batch']).shift(-1)['signal']
tt['signal_shift+2'] = tt.groupby(['batch']).shift(2)['signal']
tt['signal_shift-2'] = tt.groupby(['batch']).shift(-2)['signal']

for c in ['signal_batch_min', 'signal_batch_max',
          'signal_batch_std', 'signal_batch_mean',
          'mean_abs_chg_batch', 'abs_max_batch',
          'abs_min_batch',
          'range_batch', 'maxtomin_batch', 'abs_avg_batch',
          'signal_shift+1', 'signal_shift-1',
          'signal_batch_5k_min', 'signal_batch_5k_max',
          'signal_batch_5k_std',
          'signal_batch_5k_mean', 'mean_abs_chg_batch_5k',
          'abs_max_batch_5k', 'abs_min_batch_5k',
          'range_batch_5k', 'maxtomin_batch_5k',
          'abs_avg_batch_5k','signal_shift+2','signal_shift-2']:
    tt[f'{c}_msignal'] = tt[c] - tt['signal']


# FEATURES = [f for f in tt.columns if f not in ['open_channels','index','time','train','batch',
#                                                'batch_index','batch_slices','batch_slices2']]


FEATURES = ['signal',
            'signal_batch_min',
            'signal_batch_max',
            'signal_batch_std',
            'signal_batch_mean',
            'mean_abs_chg_batch',
            #'abs_max_batch',
            #'abs_min_batch',
            #'abs_avg_batch',
            'range_batch',
            'maxtomin_batch',
            'signal_batch_5k_min',
            'signal_batch_5k_max',
            'signal_batch_5k_std',
            'signal_batch_5k_mean',
            'mean_abs_chg_batch_5k',
            'abs_max_batch_5k',
            'abs_min_batch_5k',
            'range_batch_5k',
            'maxtomin_batch_5k',
            'abs_avg_batch_5k',
            'signal_shift+1',
            'signal_shift-1',
            # 'signal_batch_min_msignal',
            'signal_batch_max_msignal',
            'signal_batch_std_msignal',
            # 'signal_batch_mean_msignal',
            'mean_abs_chg_batch_msignal',
            'abs_max_batch_msignal',
            'abs_min_batch_msignal',
            'range_batch_msignal',
            'maxtomin_batch_msignal',
            'abs_avg_batch_msignal',
            'signal_shift+1_msignal',
            'signal_shift-1_msignal',
            'signal_batch_5k_min_msignal',
            'signal_batch_5k_max_msignal',
            'signal_batch_5k_std_msignal',
            'signal_batch_5k_mean_msignal',
            'mean_abs_chg_batch_5k_msignal',
            'abs_max_batch_5k_msignal',
            'abs_min_batch_5k_msignal',
            #'range_batch_5k_msignal',
            'maxtomin_batch_5k_msignal',
            'abs_avg_batch_5k_msignal',
            'signal_shift+2',
            'signal_shift-2']

print('....: FEATURE LIST :....')
print([f for f in FEATURES])

update_tracking(run_id, 'n_features', len(FEATURES), integer=True)
update_tracking(run_id, 'target', TARGET)

###########
# Metric
###########


def lgb_Metric(preds, dtrain):
    labels = dtrain.get_label()
    print(preds.shape)
    print(preds)
    preds = np.argmax(preds, axis=0)
#     score = metrics.cohen_kappa_score(labels, preds, weights = 'quadratic')
    score = f1_score(labels, preds, average='macro')
    return ('KaggleMetric', score, True)


###########
# MODEL
###########
tt['train'] = tt['train'].astype('bool')
train = tt.query('train').copy()
test = tt.query('not train').copy()
train['open_channels'] = train['open_channels'].astype(int)
X = train[FEATURES]
X_test = test[FEATURES]
y = train[TARGET].values
sub = test[['time']].copy()
groups = train['batch']

if OBJECTIVE == 'multiclass':
    NUM_CLASS = 11
else:
    NUM_CLASS = 1

# define hyperparammeter (some random hyperparammeters)
params = {'learning_rate': LEARNING_RATE,
          'max_depth': MAX_DEPTH,
          'num_leaves': NUM_LEAVES,
          'feature_fraction': FEATURE_FRACTION,
          'bagging_fraction': BAGGING_FRACTION,
          'bagging_freq': BAGGING_FREQ,
          'n_jobs': N_THREADS,
          'seed': RANDOM_SEED,
          'metric': METRIC,
          'objective': OBJECTIVE,
          'num_class': NUM_CLASS
          }

kfold = KFold(n_splits=TOTAL_FOLDS, shuffle=SHUFFLE, random_state=RANDOM_SEED)

oof_df = train[['signal', 'open_channels']].copy()
fi_df = pd.DataFrame(index=FEATURES)

fold = 1  # init fold counter
for tr_idx, val_idx in kfold.split(X, y, groups=groups):
    print(f'====== Fold {fold:0.0f} of {TOTAL_FOLDS} ======')
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_val, y_val)

    model = lgb.train(params,
                      train_set,
                      num_boost_round=NUM_BOOST_ROUND,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      valid_sets=[train_set, val_set],
                      verbose_eval=50)
    # feval=lgb_Metric)

    if OBJECTIVE == 'multi_class':
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        preds = np.argmax(preds, axis=1)
        test_preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds = np.argmax(test_preds, axis=1)
    elif OBJECTIVE == 'regression':
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        preds = np.round(np.clip(preds, 0, 10)).astype(int)
        test_preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds = np.round(np.clip(test_preds, 0, 10)).astype(int)

    oof_df.loc[oof_df.iloc[val_idx].index, 'oof'] = preds
    sub[f'open_channels_fold{fold}'] = test_preds

    f1 = f1_score(oof_df.loc[oof_df.iloc[val_idx].index]['open_channels'],
                  oof_df.loc[oof_df.iloc[val_idx].index]['oof'],
                  average='macro')
    rmse = np.sqrt(mean_squared_error(oof_df.loc[oof_df.index.isin(val_idx)]['open_channels'],
                                      oof_df.loc[oof_df.index.isin(val_idx)]['oof']))

    update_tracking(run_id, f'{fold}_best_iter', model.best_iteration, integer=True)
    update_tracking(run_id, f'{fold}_rmse', rmse)
    update_tracking(run_id, f'{fold}_f1', f1)
    fi_df[f'importance_{fold}'] = model.feature_importance()
    print(f'Fold {fold} - validation f1: {f1:0.5f}')
    print(f'Fold {fold} - validation rmse: {rmse:0.5f}')

    fold += 1

oof_f1 = f1_score(oof_df['open_channels'],
                  oof_df['oof'],
                  average='macro')
oof_rmse = np.sqrt(mean_squared_error(oof_df['open_channels'],
                                      oof_df['oof']))

update_tracking(run_id, f'oof_f1', oof_f1)
update_tracking(run_id, f'oof_rmse', oof_rmse)

###############
# SAVE RESULTS
###############

s_cols = [s for s in sub.columns if 'open_channels' in s]

sub['open_channels'] = sub[s_cols].median(axis=1).astype(int)
sub.to_csv(f'./pred_x_{oof_f1:0.6}.csv', index=False)
sub[['time', 'open_channels']].to_csv(f'./sub_x_{oof_f1:0.10f}.csv',
                                      index=False,
                                      float_format='%0.4f')

oof_df.to_csv(f'./oof_x_{oof_f1:0.6}.csv', index=False)

fi_df['importance'] = fi_df.sum(axis=1)
fi_df.to_csv(f'./fi_x_{oof_f1:0.6}.csv', index=True)


fig, ax = plt.subplots(figsize=(15, 30))
fi_df.sort_values('importance')['importance']     .plot(kind='barh',
          figsize=(15, 30),
          title=f'x - Feature Importance',
          ax=ax)
plt.savefig(f'./x__{oof_f1:0.6}.png')


# > # **2. On Hamiltonian Monte Carlo**

# **Hamiltonian Monte Carlo** can produce distant proposals for the **Metropolis Algorithm** (an algorithm which obtains random samples from a distribution where direct sampling is difficult).
# 
# **Hamiltonian Monte Carlo** avoids random walk behaviour by bringing in Hamiltonian Mechanics (where a system is defined by  a set of coordinates in space).
# 
# ---
# 
# ## Further results
# 
# After experimentation, I have found out that it is not the result of a HMC. However, I am still experimenting on Metroplis algorithm for this dataset.

# ## Implementation
# 
# We can implement **Hamiltonian Monte Carlo** with `pyhmc`.

# In[ ]:


get_ipython().system('pip install pyhmc --quiet')
get_ipython().system('pip install triangle --quiet')


# In[ ]:


import pyhmc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb


# As Markus proved, the data is simply the result of a Markov process with Gaussian noise.

# In[ ]:


df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

train_time   = df_train["time"].values.reshape(-1,500000)
train_signal = df_train["signal"].values.reshape(-1,500000)
train_opench = df_train["open_channels"].values.reshape(-1,500000)


# ### Why Hamiltonian Monte Carlo?

# One might ask, "why HMC instead of a regular Markov chain?". HMC is built **not** to facilitate random walk behaviours.

# In[ ]:


def markov_p(data):
    channel_range = np.unique(data)
    channel_bins = np.append(channel_range, 11)
    data_next = np.roll(data, -1)
    matrix = []
    for i in channel_range:
        current_row = np.histogram(data_next[data == i], bins=channel_bins)[0]
        current_row = current_row / np.sum(current_row)
        matrix.append(current_row)
    return np.array(matrix)
p03 = markov_p(train_opench[3])
p04 = markov_p(train_opench[4])

eig_values, eig_vectors = np.linalg.eig(np.transpose(p03))
print("Eigenvalues :", eig_values)


# 1. To really prove whether it's a HMC, we must identify the autocorrelation. 
# 2. How will this help?

# In[ ]:


p03 = p03.flatten()
p03


# In[ ]:


print("Auto correlation of HMC is:", pd.Series(p03).autocorr())


# In[ ]:


pd.Series(p03).hist()


# In[ ]:


pd.Series(p03).plot()


# In[ ]:


len(p03)


# In[ ]:


def create_axes_grid(numplots_x, numplots_y, plotsize_x=6, plotsize_y=3):
    fig, axes = plt.subplots(numplots_y, numplots_x)
    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig, axes
    
def set_axes(axes, use_grid=True, x_val = [0,100,10,5], y_val = [-50,50,10,5]):
    axes.grid(use_grid)
    axes.tick_params(which='both', direction='inout', top=True, right=True, labelbottom=True, labelleft=True)
    axes.set_xlim(x_val[0], x_val[1])
    axes.set_ylim(y_val[0], y_val[1])
    axes.set_xticks(np.linspace(x_val[0], x_val[1], np.around((x_val[1] - x_val[0]) / x_val[2] + 1).astype(int)))
    axes.set_xticks(np.linspace(x_val[0], x_val[1], np.around((x_val[1] - x_val[0]) / x_val[3] + 1).astype(int)), minor=True)
    axes.set_yticks(np.linspace(y_val[0], y_val[1], np.around((y_val[1] - y_val[0]) / y_val[2] + 1).astype(int)))
    axes.set_yticks(np.linspace(y_val[0], y_val[1], np.around((y_val[1] - y_val[0]) / y_val[3] + 1).astype(int)), minor=True)


# In[ ]:


fig, axes = create_axes_grid(1,1,10,10)
set_axes(axes, x_val=[-4,2,1,.1], y_val=[-4,2,1,.1])

axes.set_aspect('equal')
data = train_signal[4]
plt.scatter(np.roll(data, -1), data)
plt.savefig("results.png")


# We get rather odd results from the scatter plot.

# In[ ]:


train_signal[4]


# In[ ]:


prob = [.3, .3, .2, .1, .05, .05]
data = train_signal[4][1:7]
len(data)
stat = []
for i in range(1, 1000):

    # Choose random inputs for the sales targets and percent to target
    mc_signal = np.random.choice(data, 500, p=prob)

    # Build the dataframe based on the inputs and number of reps
    df = pd.DataFrame(index=range(500), data={'mc_signal': mc_signal,})

    # We want to track sales,commission amounts and sales targets over all the simulations
    stat.append([df['mc_signal'].sum().round(0)])


# In[ ]:


plt.plot(df.head(11))
arsig = np.array(df.mc_signal)


# In[ ]:


from scipy.stats import kstest as ks
ks(arsig, 'norm')


# In[ ]:


from scipy.stats import kstest as ks
ks(p03, 'norm')


# Seems like our Monte Carlo predicted value and our p03 array are not similar. So, it seems Monte Carlo is not so reliable after all. 
# 
# So, it seems we do not have a Hamiltonian Markov Chain, because this is obviously random whereas a HMC is built not to become random.

# Some resources to help you in this competition:
# 
# * What is Drift? :- [Chris Deotte](https://www.kaggle.com/c/liverpool-ion-switching/discussion/137537)
# * Viterbi Algorithm explained:- [Markus F.](https://www.kaggle.com/c/liverpool-ion-switching/discussion/137388)
# * What a Markov chain actually is:- [Roman](https://www.kaggle.com/c/liverpool-ion-switching/discussion/137366)

# ## Credits: Markus F.
