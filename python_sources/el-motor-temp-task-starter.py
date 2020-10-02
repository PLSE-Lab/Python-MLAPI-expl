#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression as OLS
from sklearn.preprocessing import StandardScaler
import os
import multiprocessing
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# In[ ]:


# read data
df = pd.read_csv('/kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv')
df.drop('torque', axis=1, inplace=True)
target_features = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']
PROFILE_ID_COL = 'profile_id'

df.head(10)


# In[ ]:


extra_feats = {
     'i_s': lambda x: np.sqrt(x['i_d']**2 + x['i_q']**2),  # Current vector norm
     'u_s': lambda x: np.sqrt(x['u_d']**2 + x['u_q']**2),  # Voltage vector norm
     'S_el': lambda x: x['i_s']*x['u_s'],                  # Apparent power
     #'P_el': lambda x: x['i_d'] * x['u_d'] + x['i_q'] *x['u_q'],  # Effective power
     'i_s_x_w': lambda x: x['i_s']*x['motor_speed'],
     'S_x_w': lambda x: x['S_el']*x['motor_speed'],
}
df = df.assign(**extra_feats)
x_cols = [x for x in df.columns.tolist() if x not in target_features + [PROFILE_ID_COL]]


# In[ ]:


spans = [6360, 3360, 1320, 9480]  # these values correspond to cutoff-frequencies in terms of low pass filters, or half-life in terms of EWMAs, respectively

def dig_into_rolling_features(_df):
    """_df corresponds to a unique measurement session"""

    # get max lookback
    max_lookback = max(spans)
    # prepad default values until max lookback in order to get unbiased
    # rolling lookback feature during first observations
    dummy = pd.DataFrame(np.zeros((max_lookback, len(_df.columns))),
                         columns=_df.columns)

    temperature_cols = [c for c in ['ambient', 'coolant'] if c in _df]
    dummy.loc[:, temperature_cols] = _df.loc[0, temperature_cols].values

    # prepad
    _df = pd.concat([dummy, _df], axis=0, ignore_index=True)

    ew_mean = [_df.ewm(span=lb).mean()
                   .rename(columns=lambda c: c+'_ewma_'+str(lb))
               for lb in spans]
    ew_std = pd.concat(
        [_df.ewm(span=lb).std().fillna(0).astype(np.float32)
             .rename(columns=lambda c: c+'_ewms_'+str(lb))
         for lb in spans], axis=1)

    concat_l = [pd.concat(ew_mean, axis=1).astype(np.float32),
                ew_std,
                ]
    ret = pd.concat(concat_l, axis=1).iloc[max_lookback:, :]        .reset_index(drop=True)
    return ret


# In[ ]:


# smooth input temperatures (mitigate artifacts)
cols_to_smooth = ['ambient', 'coolant']
smoothing_window = 100
orig_x = df.loc[:, cols_to_smooth]
x_smoothed = [x.rolling(smoothing_window,
                        center=True).mean() for p_id, x in
              df[cols_to_smooth + [PROFILE_ID_COL]]
                  .groupby(PROFILE_ID_COL)]
df.loc[:, cols_to_smooth] = pd.concat(x_smoothed).fillna(orig_x)
# We depend on the grp iterator to always yield the same order of grps
p_df_list = [df.drop(PROFILE_ID_COL, axis=1).reset_index(drop=True)
             for _, df in df[x_cols + [PROFILE_ID_COL]].groupby([PROFILE_ID_COL])]
# add EWMA and EWMS
df = pd.concat([df, 
                pd.concat([dig_into_rolling_features(p) for p in p_df_list], ignore_index=True)],
               axis=1).dropna().reset_index(drop=True)

x_cols = [x for x in df.columns.tolist() if x not in target_features + [PROFILE_ID_COL]]


# ### helper functions and classes

# In[ ]:





# In[ ]:


from sklearn.metrics import mean_squared_error as mse, mean_squared_log_error    as msle, mean_absolute_error as mae, r2_score


def print_scores(y_true, y_pred):
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    print(f'MSE: {mse(y_true, y_pred):.6} ')
    print(f'MAE: {mae(y_true, y_pred):.6} ')
    print(f'MaxAbsDev: {np.max(np.abs(y_pred - y_true)):.6} ')
    print(f'R2 : {r2_score(y_true, y_pred):.6}')

class Report:
    """Summary of an experiment/trial"""

    param_map = {'pm': '{PM}',
                 'stator_tooth': '{ST}',
                 'stator_yoke': '{SY}',
                 'stator_winding': '{SW}',
                 'motor_speed': 'motor speed',
                 'ambient': 'ambient temperature',
                 'coolant': 'coolant temperature'}
    output_param_map = {'pm': 'magnet temperature',
                        'stator_tooth': 'stator tooth temperature',
                        'stator_yoke': 'stator yoke temperature',
                        'stator_winding': 'stator winding temperature'}

    def __init__(self, uid, yhat=None, actual=None, history=None,
                 used_loss=None, model=None,):

       
        self.yhat_te = yhat
        self.actual = actual
        self.history = history
        self.uid = uid
        self.yhat_tr = None
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.used_loss = used_loss
        self.model = model
        self.cfg_blob = {}

 
    def _format_plot(self, y_lbl='temp', x_lbl=True, legend=True,
                     legend_loc='best'):
        if x_lbl:
            plt.xlabel('Time in h')

        if y_lbl in ['temp', 'coolant', 'ambient', 'pm', 'stator_yoke',
                     'stator_teeth', 'stator_winding']:
            plt.ylabel('Temperature')
        elif y_lbl == 'motor_speed':
            plt.ylabel('Rot. speed')
        elif y_lbl == 'torque':
            plt.ylabel('Torque')
        elif y_lbl.startswith('i_'):
            plt.ylabel('Current')

        if legend:
            plt.legend(loc=legend_loc)
        plt.xlim(-1000, np.around(len(self.actual), -3) + 300)
        tcks = np.arange(0, np.around(len(self.actual), -3), 7200)
        tcks_lbls = tcks // 7200 if x_lbl else []
        plt.xticks(tcks, tcks_lbls)

    def plot_history(self):
        if self.history is not None:
            history = self.history.history
            plt.figure(figsize=(6, 4))
            plt.plot(history['loss'], label='train loss')
            plt.plot(history['val_loss'], label='validation loss')
            plt.xlabel('epoch')
            plt.ylabel(f'{self.used_loss}')
            plt.title(f'Training/Validation Score over Epochs of Experiment '
                      f'{self.uid}')
            plt.legend()

    def plot(self, show=True, with_input=False):
        print_scores(self.actual, self.yhat_te)
        #self.plot_overlapping_testset_error(with_input)
        self.plot_history()
        self.plot_testset_error(with_input)
        self.plot_residual_over_y_range()
        try:
            self.plot_residual_histogram()
        except Exception as err:
            print(err)
            print('cant plot residual plot (histogram)')

        if show:
            plt.show()

    def plot_signal_measured_and_estimated(self, x_lbl, column, ax, title=True):
        diff = self.yhat_te[column] - self.actual[column]
        if title:
            plt.title('Measured and estimated ' + self.output_param_map[column])
        plt.plot(self.actual[column], color='green',
                 label=r'$\vartheta_{}$'.format(self.param_map[column]),
                 linestyle='-')
        plt.plot(self.yhat_te[column], lw=2, color='navy',
                 label=r'$\hat \vartheta_{}$'.format(self.param_map[column]),
                 linestyle='-')
        self._format_plot(x_lbl=x_lbl, legend_loc='lower right',
                          legend=True)
        plt.text(0.6, 0.9,
                 s=f'MSE: {(diff ** 2).mean():.2f}',
                 bbox={'facecolor': 'white',
                       'edgecolor': 'black'}, transform=ax.transAxes,
                 verticalalignment='top', horizontalalignment='center')

    def plot_signal_estimation_error(self, x_lbl, column, ax, title=True):
        diff = self.yhat_te[column] - self.actual[column]
        if title:
            plt.title('Estimation error ' + self.output_param_map[column])
        plt.plot(diff, color='red',
                 label='Estimation error ' +
                       r'$\vartheta_{}$'.format(self.param_map[column]))
        self._format_plot(x_lbl=x_lbl, legend_loc='lower center',
                          legend=False)
        plt.text(0.6, 0.9,
                 bbox={'facecolor': 'white', 'edgecolor': 'black'},
                 transform=ax.transAxes,
                 s=r'$||e||_\infty$: ' + f'{diff.abs().max():.2f}',
                 verticalalignment='top', horizontalalignment='center')

    def plot_testset_error(self, with_input=True):
        n_targets = len(self.actual.columns)
        if with_input:
            n_targets += 2

        plt.figure(figsize=(16, 2.3 * n_targets))
        for i, c in enumerate(self.actual):
            ax = plt.subplot(n_targets, 2, 2 * i + 1)
            x_lbl = not with_input and len(self.actual.columns) == (i + 1)
            self.plot_signal_measured_and_estimated(x_lbl, c, ax)
            ax = plt.subplot(n_targets, 2, 2 * (i + 1))
            self.plot_signal_estimation_error(x_lbl, c, ax)


    def plot_residual_histogram(self):
        for c in self.actual:
            diff = self.yhat_te[c] - self.actual[c]
            diff = np.clip(diff, a_min=-10, a_max=10)
            plt.figure(figsize=(5, 3))
            sns.distplot(diff)
            plt.xlabel(c + ' error')
            plt.tight_layout()
            plt.title('Residual histogram')

    def plot_signal_residual_over_y_range(self, column, color='b', name=None):
        name = name or column
        residuals =             (pd.DataFrame({column + '_true': self.actual[column],
                           column + '_pred': self.yhat_te[column]})
             .sort_values(column + '_true')
             # .set_index(c+'_true', inplace=False)
             )
        # sns.relplot(data=residuals, s=3, alpha=0.9, marker='+')
        plt.scatter(residuals[f'{column}_true'],
                    residuals[f'{column}_pred'] - residuals[f'{column}_true'],
                    s=1, c=color, label=name)

        plt.axhline(color='black', ls='--')
        plt.xlabel(r'$\vartheta_{}$'.format(self.param_map[column]) +
                   ' ground truth')
        plt.ylabel(r'$\vartheta_{}$'.format(self.param_map[column]) +
                   ' prediction error')
        plt.tight_layout()
        plt.title('Error Residuals')

    def plot_residual_over_y_range(self):
        for c in self.actual:
            plt.figure(figsize=(5, 5))
            self.plot_signal_residual_over_y_range(c)
        plt.legend(markerscale=5.0, loc='lower left', ncol=len(self.actual))


# ### Model training

# In[ ]:


test_set_profiles = [65, 72]
trainset = df.loc[~df.profile_id.isin(test_set_profiles), :].reset_index(drop=True)
testset = df.loc[df.profile_id.isin(test_set_profiles), :].reset_index(drop=True)

x_train = trainset.loc[:, x_cols]
y_train = trainset.loc[:, target_features]
x_test = testset.loc[:, x_cols]
y_test = testset.loc[:, target_features]

# standardize (targets are already standardized)
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_cols)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_cols)

ols = OLS(fit_intercept=False)
print('Start fitting OLS...')
ols.fit(x_train, y_train)
print('Predict with OLS...')
pred = ols.predict(x_test)
pred = pd.DataFrame(pred, columns=y_test.columns)


# In[ ]:


# The Report class can be used to have a quick performance overview
report = Report('OLS', pred, y_test)
report.plot()


# In[ ]:


# I suspect a bug for the profile no. 72: Prediction is way worse there than on on the first profile and on my local machine. I am looking forward to any feedback.

