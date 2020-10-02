#!/usr/bin/env python
# coding: utf-8

# # Forecasting BTC using N-BEATS 
# In this kernel, I implement the N-BEATS[0] architecture that is the current SOTA for time series forecasting as far as I know. For more details on what the architecture is about, please read my medium blog on it.
# 
# I modified [fastai](https://www.fast.ai/)'s tabular model/learner and use their one cycle learning to speed up training among other things!

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from fastai.tabular import *
from fastai.callbacks import *
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


path = Path(os.getcwd())


# In[ ]:


df = pd.read_csv('/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv')
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
df = df.groupby('date')
df = df['Weighted_Price'].mean(); df = df.to_frame() #convert series to df

#testing to see if logging makes the model better: https://stats.stackexchange.com/questions/298/in-linear-regression-when-is-it-appropriate-to-use-the-log-of-an-independent-va
df['Weighted_Price'] = df['Weighted_Price'].apply(lambda x: math.log(x))
print(df.shape); df.plot()


# In[ ]:


#What we will train the model on
df.iloc[:int(len(df)*.9)].plot()


# In[ ]:


#What we will want the model to predict
df.iloc[int(len(df)*.9):].plot()


# In[ ]:


def prep_df(df,lag=1):
  "Extend df sideways to allow a longer more timesteps while taking advantage of fastai's dataset/loader"
  df_org = df.copy(deep=True)

  for i in range(lag):
    df_lag = df_org.shift(i+1)
    df_lag = df_lag.add_suffix('_M' + str(i+1))
    #not taking last column of lagged values because cannot use for competition
    df = df.merge(df_lag, left_index=True, right_index=True ,suffixes=(False, False))
    
    df.dropna(inplace=True)
  
  return df


# In[ ]:


df_com = prep_df(df,5); df_com.reset_index(drop=True,inplace=True); df_com.head(2)


# In[ ]:


dep_var = 'Weighted_Price'; cat_names =[]


# In[ ]:


procs = [FillMissing, Categorify, Normalize]
valid_idx = range(int(len(df_com)*.9), len(df_com))


# In[ ]:


data = TabularDataBunch.from_df(path, df_com, dep_var, valid_idx=valid_idx, procs=procs, 
                                cat_names=cat_names,bs=64)

#making shuffling false so that there is no data leakage
data.train_dl = data.train_dl.new(shuffle=False)


# In[ ]:


#preview of data for sanity checks
x,y = next(iter(data.train_dl))
(cat_x,cont_x),y = next(iter(data.train_dl))
for o in (cat_x, cont_x, y): print(to_np(o[:5]))


# In[ ]:


class gen_block(nn.Module):
  def __init__(self,n_in, n_hidden, theta_dim, n_out, bn:bool=True, ps:float=0., actn:Optional[nn.Module]=None):
    super().__init__()
    self.FC1 = nn.Sequential(*bn_drop_lin(n_in,n_hidden,bn,ps,actn))
    self.FC2 = nn.Sequential(*bn_drop_lin(n_hidden,n_hidden,bn,ps,actn))
    self.FC3 = nn.Sequential(*bn_drop_lin(n_hidden,n_hidden,bn,ps,actn))
    self.FC4 = nn.Sequential(*bn_drop_lin(n_hidden,n_hidden,bn,ps,actn))
    self.Fcst = nn.Sequential(*(bn_drop_lin(n_hidden,theta_dim,bn,ps,actn)+bn_drop_lin(theta_dim,n_out,bn,ps))) #forecast output shouldnt have relu
    self.Bcst = nn.Sequential(*(bn_drop_lin(n_hidden,theta_dim,bn,ps,actn)+bn_drop_lin(theta_dim,n_in,bn,ps))) #same for backcast

  def forward(self, x):
    x1 = self.FC1(x)
    x1 = self.FC2(x1)
    x1 = self.FC3(x1)
    x1 = self.FC4(x1)
    x2 = self.Fcst(x1)
    x3 = self.Bcst(x1)

    return (x-x3, x2)


# In[ ]:


class FFModel(Module):
    "Modified tabular model from fastai for embedding projections, if needed"
    def __init__(self, n_hidden, theta_dim, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__()
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        self.n_in  = self.n_emb + self.n_cont
        self.n_out = out_sz
        self.n_hidden = n_hidden
        self.theta_dim = theta_dim

        self.blk1 = gen_block(n_in=self.n_in,n_hidden=self.n_hidden,theta_dim=self.theta_dim,n_out=self.n_out,ps=ps,actn=nn.ReLU(inplace=True))
        self.blk2 = gen_block(n_in=self.n_in,n_hidden=self.n_hidden,theta_dim=self.theta_dim,n_out=self.n_out,ps=ps,actn=nn.ReLU(inplace=True))
        self.blk3 = gen_block(n_in=self.n_in,n_hidden=self.n_hidden,theta_dim=self.theta_dim,n_out=self.n_out,ps=ps,actn=nn.ReLU(inplace=True))
        self.blk4 = gen_block(n_in=self.n_in,n_hidden=self.n_hidden,theta_dim=self.theta_dim,n_out=self.n_out,ps=ps,actn=nn.ReLU(inplace=True))
        self.blk5 = gen_block(n_in=self.n_in,n_hidden=self.n_hidden,theta_dim=self.theta_dim,n_out=self.n_out,ps=ps,actn=nn.ReLU(inplace=True))
        self.blk6 = gen_block(n_in=self.n_in,n_hidden=self.n_hidden,theta_dim=self.theta_dim,n_out=self.n_out,ps=ps,actn=nn.ReLU(inplace=True))


    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont

        x, f1 = self.blk1(x)
        x, f2 = self.blk2(x)
        x, f3 = self.blk3(x)
        x, f4 = self.blk4(x)
        x, f5 = self.blk5(x)
        x, f6 = self.blk6(x)        

        x = f1+f2+f3+f4+f5+f6

        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]

        return x


# In[ ]:


def FFLearner(data:DataBunch, n_hidden, theta_dim, layers:Collection[int], emb_szs:Dict[str,int]=None, metrics=None,
        ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, **learn_kwargs):
    "Get a `Learner` using `data`, with `metrics`, including a `TabularModel` created using the remaining params."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    model = FFModel(n_hidden, theta_dim, emb_szs, len(data.cont_names), out_sz=data.c, layers=layers, ps=ps, emb_drop=emb_drop,
                    y_range=y_range, use_bn=use_bn)
    return Learner(data, model, metrics=metrics, **learn_kwargs)


# In[ ]:


y_range=None
#y_range = (df_com['Weighted_Price'].min(), df_com['Weighted_Price'].max()); print(y_range)
#del df_com; gc.collect()


# In[ ]:


learn = FFLearner(data, n_hidden=512,theta_dim=8,layers=[0], metrics=mean_absolute_error,
            emb_drop=0, y_range=y_range, 
                  callback_fns=[ShowGraph,partial(CSVLogger, append=True)], ps=0.1)


# In[ ]:


learn.model


# In[ ]:


learn.lr_find(); learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, 1e-2)
#, callbacks=[SaveModelCallback(learn, every='epoch', monitor='mean_absolute_error')])


# Pretty muted predictions at first

# In[ ]:


preds, y = learn.get_preds()
plt.plot(preds, label = 'Predictions'); plt.plot(y, label = 'Actuals')
plt.legend(); plt.show()


# In[ ]:


learn.lr_find(); learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4, 1e-2)


# We can see the model getting more expressive as we train more.

# In[ ]:


preds, y = learn.get_preds()
plt.plot(preds, label = 'Predictions'); plt.plot(y, label = 'Actuals')
plt.legend(); plt.show()


# In[ ]:


learn.save('NBEATS_LAG5_LOG')


# ## References
# [0] Oreshkin et al. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
