#!/usr/bin/env python
# coding: utf-8

# # Modules, Styling, & Helpful Functions

# In[ ]:


import warnings; warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
import pylab as pl,seaborn as sn
import scipy as sp,keras as ks
import statsmodels.api as sm
from statsmodels.tsa import ar_model
from IPython.display import display,HTML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error,mean_absolute_error
from sklearn.metrics import r2_score,explained_variance_score


# In[ ]:


thp=[('font-size','16px'),('text-align','center'),
     ('font-weight','bold'),('padding','5px 5px'),
     ('color','#66FF66'),('background-color','slategray')]
tdp=[('font-size','14px'),('padding','5px 5px'),
     ('text-align','center'),('color','darkgreen'),
     ('background-color','silver')]
style_dict=[dict(selector='th',props=thp),
            dict(selector='td',props=tdp)]
pl.style.use('seaborn-pastel')
pl.style.use('seaborn-whitegrid')


# In[ ]:


def reg_plot(targets,preds):
    pl.figure(figsize=(11,6))
    mse=mean_squared_error(targets,preds)
    pl.plot(targets,'o',label='real data',
            markersize=3,markeredgecolor='darkgreen',
            markerfacecolor="None",markeredgewidth=1)
    pl.plot(preds,'o',label='predictions',
            markersize=3,markeredgecolor='#66ff66',
            markerfacecolor="None",markeredgewidth=1)
    pl.legend(); pl.title('mse = %.6f'%mse);


# # Data Loading & Exploration

# In[ ]:


c_bank=pd.read_csv('../input/c_bank.csv')
finance=c_bank.drop(['date','monetary_gold',
                     'foreign_exchange_reserves'],1)
metal_list=['silver','palladium','platinum','gold']
dual_metal_list=['dual_currency_basket','silver',
                 'palladium','platinum','gold']
display(c_bank.head().T.style        .set_table_styles(style_dict))
finance.describe().T.style.set_table_styles(style_dict)


# In[ ]:


pl.figure(figsize=(11,6))
c_bank.gold.plot()
c_bank.platinum.plot()
c_bank.palladium.plot()
pl.xlabel('metals'); pl.ylabel('prices')
pl.title('Metal Prices')
pl.legend(['gold','platinum','palladium']);


# In[ ]:


pl.figure(figsize=(11,7))
sn.distplot(c_bank.gold,bins=100)
sn.distplot(c_bank.platinum,bins=100)
sn.distplot(c_bank.palladium,bins=100)
pl.xlabel('metals')
pl.ylim(0.,.0035); pl.ylabel('distribution')
pl.title('Distributions of Metal Prices')
pl.legend(['gold','platinum','palladium']);


# In[ ]:


hd={'color':'SlateGrey','bins':50}
axes=pd.plotting.scatter_matrix(finance,alpha=.3,
                                figsize=(11,11),diagonal='hist',
                                c='#1b2c45',hist_kwds=hd)
corr=finance.corr().as_matrix()
for i,j in zip(*np.triu_indices_from(axes,k=1)):
    axes[i,j].annotate("%.3f"%corr[i,j],(.7,.1),fontsize=12,
                       color='darkgreen',xycoords='axes fraction',
                       ha='center',va='center');


# In[ ]:


features=RobustScaler().fit_transform(finance[metal_list])
targets=RobustScaler().fit_transform(finance['dual_currency_basket']                     .values.reshape(-1,1))
X_train,X_test=features[:850],features[850:]
y_train,y_test=targets[:850],targets[850:]


# # Models
# #### `Numpy Solutions` (the least-squares solution to a linear matrix equation) & `OLS Model`

# In[ ]:


coef,total_error,_,_=np.linalg.lstsq(features,targets)
rmse=np.sqrt(total_error[0]/len(targets))
preds=np.dot(features,coef); n=len(targets)
print(coef.reshape(-1)); print('rmse = %.6f'%rmse)
ols_model=sm.OLS(targets,features)
ols_results=ols_model.fit()
ols_preds=ols_results.predict(features)
ols_coef=ols_results.params
ols_r2=ols_results.rsquared
print(ols_coef); print('r2 = %.6f'%ols_r2)
reg_plot(targets,ols_preds)


# #### Structural time series models

# In[ ]:


data=c_bank[dual_metal_list]
ids=[pd.Timestamp(c_bank['date'][i]) for i in range(n)]
data=data.set_index(c_bank['date'].values)
restricted_model={'level':'smooth trend','cycle': True,
                  'damped_cycle': True,'stochastic_cycle':True}
dual_restricted_mod=sm.tsa.UnobservedComponents(data['dual_currency_basket'],**restricted_model)
dual_restricted_res=dual_restricted_mod.fit(method='powell',disp=False)
print(dual_restricted_res.summary())


# In[ ]:


fig=dual_restricted_res.plot_components(legend_loc='higher left',figsize=(11,11));

