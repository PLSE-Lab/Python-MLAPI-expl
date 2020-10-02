#!/usr/bin/env python
# coding: utf-8

# [Lesson Video Link](https://course.fast.ai/videos/?lesson=6)
# 
# [Lesson resources and updates](https://forums.fast.ai/t/lesson-6-official-resources-and-updates/31441)
# 
# [Lesson chat](https://forums.fast.ai/t/lesson-6-in-class-discussion/31440)
# 
# [Further discussion thread](https://forums.fast.ai/t/lesson-6-advanced-discussion/31442)
# 
# Note: This is a mirror of the FastAI Lesson 6 Nb. 
# Please thank the amazing team behind fast.ai for creating these, I've merely created a mirror of the same here
# For complete info on the course, visit course.fast.ai

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


from fastai import *
from fastai.tabular import *


# In[ ]:


path = Path('data/rossmann/')
dest = path
dest.mkdir(parents=True, exist_ok=True)


# In[ ]:


get_ipython().system('cp -r ../input/* {path}/')


# In[ ]:


path.ls()


# # Rossmann

# ## Data preparation

# To create the feature-engineered train_clean and test_clean from the Kaggle competition data, run `rossman_data_clean.ipynb`. One important step that deals with time series is this:
# 
# ```python
# add_datepart(train, "Date", drop=False)
# add_datepart(test, "Date", drop=False)
# ```

# In[ ]:


train_df = pd.read_pickle(path/'train_clean')


# In[ ]:


train_df.head().T


# In[ ]:


n = len(train_df); n


# ### Experimenting with a sample

# In[ ]:


idx = np.random.permutation(range(n))[:2000]
idx.sort()
small_train_df = train_df.iloc[idx[:1000]]
small_test_df = train_df.iloc[idx[1000:]]
small_cont_vars = ['CompetitionDistance', 'Mean_Humidity']
small_cat_vars =  ['Store', 'DayOfWeek', 'PromoInterval']
small_train_df = small_train_df[small_cat_vars + small_cont_vars + ['Sales']]
small_test_df = small_test_df[small_cat_vars + small_cont_vars + ['Sales']]


# In[ ]:


small_train_df.head()


# In[ ]:


small_test_df.head()


# In[ ]:


categorify = Categorify(small_cat_vars, small_cont_vars)
categorify(small_train_df)
categorify(small_test_df, test=True)


# In[ ]:


small_test_df.head()


# In[ ]:


small_train_df.PromoInterval.cat.categories


# In[ ]:


small_train_df['PromoInterval'].cat.codes[:5]


# In[ ]:


fill_missing = FillMissing(small_cat_vars, small_cont_vars)
fill_missing(small_train_df)
fill_missing(small_test_df, test=True)


# In[ ]:


small_train_df[small_train_df['CompetitionDistance_na'] == True]


# ### Preparing full data set

# In[ ]:


train_df = pd.read_pickle(path/'train_clean')
test_df = pd.read_pickle(path/'test_clean')


# In[ ]:


len(train_df),len(test_df)


# In[ ]:


procs=[FillMissing, Categorify, Normalize]


# In[ ]:


cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw']

cont_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']


# In[ ]:


dep_var = 'Sales'
df = train_df[cat_vars + cont_vars + [dep_var,'Date']].copy()


# In[ ]:


test_df['Date'].min(), test_df['Date'].max()


# In[ ]:


cut = train_df['Date'][(train_df['Date'] == train_df['Date'][len(test_df)])].index.max()
cut


# In[ ]:


valid_idx = range(cut)


# In[ ]:


df[dep_var].head()


# In[ ]:


data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs)
                   .split_by_idx(valid_idx)
                   .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
                   .databunch())


# In[ ]:


#doc(FloatList)


# ## Model

# In[ ]:


max_log_y = np.log(np.max(train_df['Sales'])*1.2)
y_range = torch.tensor([0, max_log_y], device=defaults.device)


# In[ ]:


learn = tabular_learner(data, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04, 
                        y_range=y_range, metrics=exp_rmspe)


# In[ ]:


learn.model


# In[ ]:


len(data.train_ds.cont_names)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, 1e-3, wd=0.2)


# In[ ]:


learn.save('1')


# In[ ]:


learn.recorder.plot_losses(last=-1)


# In[ ]:


learn.load('1');


# In[ ]:


learn.fit_one_cycle(5, 3e-4)


# In[ ]:


learn.fit_one_cycle(5, 3e-4)


# (10th place in the competition was 0.108)

# ## fin

# In[ ]:




