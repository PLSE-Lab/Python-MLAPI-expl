#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, FileLink
import seaborn as sns

import doctest
import copy
import functools
# tests help notebooks stay managable
def autotest(func):
    globs = copy.copy(globals())
    globs.update({func.__name__: func})
    doctest.run_docstring_examples(
        func, globs, verbose=True, name=func.__name__)
    return func


# # Setup
# ## Download the latest table from the floodlight page

# In[ ]:


fl_ds_url = 'https://dataset.floodlightopen.com/public-blobs-prod/complete_dataset.csv'
get_ipython().system('wget {fl_ds_url}')


# ## Process and Reformat

# In[ ]:


fl_df = pd.read_csv(fl_ds_url.split('/')[-1])
fl_df['participantCreatedOn'] = pd.to_datetime(fl_df['participantCreatedOn'], errors='coerce')
fl_df['testResultMetricCreatedOn'] = pd.to_datetime(fl_df['testResultMetricCreatedOn'], errors='coerce')
fl_df['measurementDate'] = fl_df['testResultMetricCreatedOn'].dt.strftime('%B %d, %Y')
fl_df['full_test_name'] = fl_df.apply(lambda x: '{testName}-{testMetricName}'.format(**x), 1)
print(fl_df.shape, fl_df.columns)
display(fl_df.head(3))


# In[ ]:


# clean up types
for c_col in ['testResultMetricValue', 'participantWeightLbs', 
              'participantHeightCms', 'participantBirthYear']:
    fl_df[c_col] = pd.to_numeric(fl_df[c_col],errors='coerce')
display(fl_df.describe())


# In[ ]:


ds_overview_df = fl_df.groupby(['floodlightOpenId', 'participantIsControl', 
                                'participantSex', 'participantBirthYear']).\
  size().reset_index(name='count').sort_values('count', ascending=False)
ds_overview_df.head(5)


# In[ ]:


part_cols = [x for x in fl_df.columns if x.startswith('participant')]
test_cols = [x for x in fl_df.columns if x.startswith('test')]
print('User', part_cols)
print('Test', test_cols)


# In[ ]:


raw_measurement_df = fl_df.pivot_table(index=['floodlightOpenId', 'measurementDate']+part_cols,
            columns=['full_test_name'],
            values='testResultMetricValue')
raw_measurement_df.sample(3)


# In[ ]:


pred_names = raw_measurement_df.index.names
x_vars = raw_measurement_df.columns


# In[ ]:


measurement_df = raw_measurement_df.reset_index()
measurement_df['participantIsControl'] = measurement_df['participantIsControl'].str.strip().    map(lambda x:    x.upper().startswith('T') if isinstance(x, str) else x).    map(lambda x: 1.0 if x else 0)
measurement_df.to_csv('clean_measure_table.csv', index=False)


# In[ ]:


part_df = measurement_df[pred_names].copy()
part_df['IsFemale'] = part_df['participantSex'].map(lambda x: x=='female')
part_df['participantIsControl'] = part_df['participantIsControl'].map(lambda x: 'Healthy' if x else 'MS')
sns.pairplot(data=part_df, hue='participantIsControl', diag_kind="kde")


# ## Compare Positive and Control Groups

# ### Time change

# In[ ]:


from scipy.interpolate import interp1d, pchip
from scipy import interpolate
from collections import defaultdict
from sklearn.linear_model import LinearRegression
day_steps = 24*3600*np.linspace(0, 300, 40)
out_vec_dict = defaultdict(dict)
small_df = fl_df[['floodlightOpenId','participantIsControl', 'full_test_name', 
                  'testResultMetricCreatedOn','testResultMetricValue']].copy()
i=0
debug_plot=0
fancy_inter = False
for (test_name, is_control), c_rows in small_df.dropna().groupby(['full_test_name', 
                                                         'participantIsControl']):
    x_vals = []
    y_vals = []
    y_all_std = c_rows['testResultMetricValue'].dropna().std()
    y_all_mean = c_rows['testResultMetricValue'].dropna().mean()
    for _, n_rows in c_rows.groupby('floodlightOpenId'):
        t_rows = n_rows.sort_values('testResultMetricCreatedOn').dropna()
        t_vec = t_rows['testResultMetricCreatedOn']
        x_vals += [(t_vec-t_vec.iloc[0]).dt.total_seconds().values]
        y_vec = t_rows['testResultMetricValue'].values
        y_vals += [(y_vec-y_all_mean)/y_all_std]
    x_vals = np.concatenate(x_vals, 0)
    x_vals += np.random.uniform(0, 5, size=x_vals.shape)
    y_vals = np.concatenate(y_vals, 0)
    i_vals = np.argsort(x_vals)
    x_vals = x_vals[i_vals]
    y_vals = y_vals[i_vals]
    if fancy_inter:
        p_xy = interpolate.interp1d(x=x_vals, y=y_vals, 
                                    kind='linear', 
                                    fill_value="extrapolate",
                                    assume_sorted=False)
    else:
        lr_model = LinearRegression().fit(x_vals.reshape((-1, 1)), y_vals)
        p_xy = lambda x: lr_model.predict(x.reshape((-1, 1)))
  
  
    out_vec_dict[is_control][test_name] = p_xy(day_steps)
    if debug_plot>0:
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x_vals/(24*3600),  y_vals , '.', alpha=0.1)
        ax1.plot(day_steps/(24*3600),  p_xy(day_steps) , '-')

        ax1.set_title('{}'.format((is_control, test_name)))
        i+=1
        if i==debug_plot:
            break


# In[ ]:


sns.set_style("whitegrid", {'axes.grid' : False})
show_traces=False
fig, m_axs = plt.subplots(len(out_vec_dict), 2 if show_traces else 1, 
                          figsize=(30 if show_traces else 20, 10))
for n_axs, (k1, v1) in zip(m_axs, out_vec_dict.items()):
    if show_traces:
        c_ax, d_ax = n_axs
        d_ax.plot(out_img.T)
        d_ax.set_ylim(-2, 2)
    else:
        c_ax = n_axs
    
    v1k = sorted(v1.keys())
    c_ax.set_title('Is Control: {}'.format(k1))
    out_img = np.stack([v1[k2] for k2 in v1k], 0)
    c_ax.imshow(out_img, cmap='RdBu', vmin=-2, vmax=2)
    c_ax.set_yticks(range(len(v1k)))
    c_ax.set_yticklabels(v1k)
    c_ax.set_xticks(range(day_steps.shape[0]))
    c_ax.set_xticklabels(['{:2.0f}'.format(x) for x in day_steps/(24*3600)])
    c_ax.set_xlabel('Days')
  


# In[ ]:


fig, m_axs = plt.subplots(3, 4, figsize=(15, 20))
for c_ax, k1 in zip(m_axs.flatten(), out_vec_dict[True].keys()):
    c_ax.set_title(k1.replace('-', '\n'))
    for k2, v2 in out_vec_dict.items():
        c_ax.plot(day_steps/(24*3600), v2[k1], label='Is Control: {}'.format(k2) )
    c_ax.legend()
    c_ax.set_ylim(-2, 2)


# ## Build a Simple Model

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score, roc_curve
def two_way_roc(gt, pred_val):
    if roc_auc_score(gt, pred_val)<0.5:
        return roc_curve(gt, -pred_val)[:2]+(roc_auc_score(gt, -pred_val),)
    else:
        return roc_curve(gt, pred_val)[:2]+(roc_auc_score(gt, pred_val),)


# ### Logistic Regression

# In[ ]:


scaled_lr = make_pipeline(RobustScaler(), # scale the values
                          SimpleImputer(missing_values=np.nan, strategy='mean'), # fix missing values
                          LogisticRegression() 
                         )


# In[ ]:


scaled_lr.fit(measurement_df[x_vars], measurement_df['participantIsControl'])


# Show the predictions vs the real values

# In[ ]:


pred_ms = scaled_lr.predict_proba(measurement_df[x_vars])
sns.violinplot(x=measurement_df['participantIsControl'], 
               y=pred_ms[:, 1])


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
tpr, fpr, _ = roc_curve(measurement_df['participantIsControl'], pred_ms[:, 1])
ax1.plot(tpr, fpr, 'k.-', label='LR Model AUC:{:2.1%}'.format(roc_auc_score(measurement_df['participantIsControl'], pred_ms[:, 1])))
c_df = measurement_df.ffill().bfill()

for x in sorted(x_vars, key = lambda x: -1*two_way_roc(measurement_df['participantIsControl'], c_df[x])[2]):
  tpr, fpr, auc = two_way_roc(measurement_df['participantIsControl'],  c_df[x])
  ax1.plot(tpr, fpr, '-', label='{} AUC:{:2.1%}'.format(x, auc))
ax1.legend(bbox_to_anchor=(1.1, 1.05))


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(18, 8))
lr_model = scaled_lr.steps[-1][1]
ax1.bar(range(len(x_vars)), lr_model.coef_[0])
ax1.set_xticks(range(len(x_vars)))
ax1.set_xticklabels([x.replace('-', '\n') for x in x_vars], rotation=90);


# In[ ]:


var_df = pd.DataFrame([{'Variable': x, 'Coefficient': v} for x, v in zip(x_vars, lr_model.coef_[0])])
var_df.assign(energy=np.power(var_df['Coefficient'], 2)).  sort_values('energy', ascending=False).  drop(['energy'],1)


# ### Native Bayes

# In[ ]:


scaled_nb = make_pipeline(RobustScaler(), # scale the values
                          SimpleImputer(missing_values=np.nan, strategy='mean'), # fix missing values
                          BernoulliNB()
                         )
scaled_nb.fit(measurement_df[x_vars], measurement_df['participantIsControl'])


# In[ ]:


pred_nb_ms = scaled_lr.predict_proba(measurement_df[x_vars])
sns.boxplot(x=measurement_df['participantIsControl'], 
               y=pred_nb_ms[:, 1])


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
tpr, fpr, _ = roc_curve(measurement_df['participantIsControl'], pred_ms[:, 1])
ax1.plot(tpr, fpr, '-', label='LR Model AUC:{:2.1%}'.format(roc_auc_score(measurement_df['participantIsControl'], pred_ms[:, 1])))
tpr, fpr, _ = roc_curve(measurement_df['participantIsControl'], pred_nb_ms[:, 1])
ax1.plot(tpr, fpr, 'k.-', label='Naive Bayes AUC:{:2.1%}'.format(roc_auc_score(measurement_df['participantIsControl'], pred_nb_ms[:, 1])))

c_df = measurement_df.ffill().bfill()

for x in sorted(x_vars, key = lambda x: -1*two_way_roc(measurement_df['participantIsControl'], c_df[x])[2]):
  tpr, fpr, auc = two_way_roc(measurement_df['participantIsControl'],  c_df[x])
  ax1.plot(tpr, fpr, '-', label='{} AUC:{:2.1%}'.format(x, auc))
ax1.legend(bbox_to_anchor=(1.1, 1.05))


# In[ ]:




