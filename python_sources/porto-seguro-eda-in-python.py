#!/usr/bin/env python
# coding: utf-8

# This is the copy of kernel "Steering Wheel of Fortune - Porto Seguro EDA"   of Heads or Tails. I wanted to crete it in python using seaborn plots.
# For analysis details you can visit the actual kernel: https://www.kaggle.com/headsortails/steering-wheel-of-fortune-porto-seguro-eda
# I have tried my best to create plots in python ,if you find any best way please suggest.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import binom_test
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('../input/train.csv', na_values=["-1","-1.0"])
test = pd.read_csv('../input/test.csv', na_values=["-1","-1.0"])
sample_submit = pd.read_csv('../input/sample_submission.csv')


# In[3]:


# binom_test(15006,(15006+345846))
from scipy.stats import beta

def get_binCI(success, total, confint=0.95):
    quantile = (1 - confint) / 2.
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    return (lower, upper)

def gett(col):
    t = train.groupby([col,'target'])['id'].count().unstack()
    t = t.assign(frac_claim=t[1]/(t[1]+t[0])*100,
             lwr=get_binCI(t[1],(t[1]+t[0]))[0]*100,
             upr=get_binCI(t[1],(t[1]+t[0]))[1]*100)
    t = t.reset_index()
    return t


# In[4]:


train.describe()


# In[5]:


test.describe()


# In[6]:


train.isnull().sum().sum()


# In[7]:


test.isnull().sum().sum()


# #Reformating Features 
# We will turn the categorical features into factors and the binary ones into logical values. For the target variable we choose a factor format.

# ### 3.4 Reformating Features

# In[8]:


train['ps_calc_20_bin']
catcols = [x for x in train.columns if x[-3:]=='cat']
bincols =[x for x in train.columns if x[-3:]=='bin']

for i in catcols:
    train[i] = train[i].astype('category')
for i in bincols:
    train[i] = train[i].astype('bool')


# ### 3.5 combining dataframes

# In[9]:


combine = pd.concat([train.assign(dset='train'),test.assign(dset='test',target=np.nan )],axis=0)
combine['dset'] = combine['dset'].astype('category')


# ## 4.Individual Feature visualizations
# 

# ### 4.1 Binary features part 1

# In[13]:


f, axes = plt.subplots(2, 4, figsize=(10, 7), sharex=False)
plt.style.use('ggplot')
sns.despine(left=True)
sns.countplot(x="ps_ind_06_bin", data=train,ax=axes[0,0])
sns.countplot(x="ps_ind_07_bin", data=train,ax=axes[0,1])
sns.countplot(x="ps_ind_08_bin", data=train,ax=axes[0,2])
sns.countplot(x="ps_ind_09_bin", data=train,ax=axes[0,3])
sns.countplot(x="ps_ind_10_bin", data=train,ax=axes[1,0])
sns.countplot(x="ps_ind_11_bin", data=train,ax=axes[1,1])
sns.countplot(x="ps_ind_12_bin", data=train,ax=axes[1,2])
sns.countplot(x="ps_ind_13_bin", data=train,ax=axes[1,3])
# plt.setp(axes, yticks=[])
plt.tight_layout()


# ### 4.2 Binary features part 2

# In[14]:


f, axes = plt.subplots(2, 5, figsize=(10, 5), sharex=False)
plt.style.use('ggplot')
sns.despine(left=True)
sns.countplot(x="ps_ind_16_bin", data=train,ax=axes[0,0])
sns.countplot(x="ps_ind_17_bin", data=train,ax=axes[0,1])
sns.countplot(x="ps_ind_18_bin", data=train,ax=axes[0,2])
sns.countplot(x="ps_calc_15_bin", data=train,ax=axes[0,3])
sns.countplot(x="ps_calc_16_bin", data=train,ax=axes[0,4])
sns.countplot(x="ps_calc_17_bin", data=train,ax=axes[1,0])
sns.countplot(x="ps_calc_18_bin", data=train,ax=axes[1,1])
sns.countplot(x="ps_calc_19_bin", data=train,ax=axes[1,2])
sns.countplot(x="ps_calc_20_bin", data=train,ax=axes[1,3])
# plt.setp(axes, yticks=[])
plt.tight_layout()


# ### 4.3 Categorcial features part 1

# In[15]:


fig = plt.figure(figsize=(15, 8))
plt.style.use('ggplot')
sns.despine(left=True)
gs = GridSpec(3, 2)

def getcatwithna(t_series):
    t = t_series.cat.add_categories("NA").fillna('NA').value_counts().reset_index()
    t.columns = (['index',col])
    return t

col = 'ps_ind_02_cat'
t=getcatwithna(train[col])
sns.barplot(x=t['index'],y=t[col],ax=plt.subplot(gs[0, 0]),log=True)
# plt.yscale('log',basey=10)

col = 'ps_ind_04_cat'
t=getcatwithna(train[col])
sns.barplot(x=t['index'],y=t[col],ax=plt.subplot(gs[0, 1]),log=True)
# plt.yscale('log',basey=10) 

col = 'ps_ind_05_cat'
t=getcatwithna(train[col])
sns.barplot(x=t['index'],y=t[col],ax=plt.subplot(gs[1, 0]),log=True)
# plt.yscale('log',basey=10) 


col = 'ps_car_01_cat'
t=getcatwithna(train[col])
sns.barplot(x=t['index'],y=t[col],ax=plt.subplot(gs[1, 1]),log=True)
# plt.yscale('log',basey=10) 


col = 'ps_car_02_cat'
t=getcatwithna(train[col])
sns.barplot(x=t['index'],y=t[col],ax=plt.subplot(gs[2, 0]),log=True)
# plt.yscale('log',basey=10) 

col = 'ps_car_03_cat'
t=getcatwithna(train[col])
sns.barplot(x=t['index'],y=t[col],ax=plt.subplot(gs[2, 1]),log=True)
# plt.yscale('log',basey=10) 

# plt.setp(axes, yticks=[])
plt.tight_layout()


# ### 4.4 Categorcial features part 2

# In[16]:


# f, axes = plt.subplots(1, 4, figsize=(10, 8), sharex=False)
fig = plt.figure(figsize=(15, 10))
plt.style.use('ggplot')
gs = GridSpec(4, 4)
col = ["ps_car_04_cat","ps_car_05_cat","ps_car_06_cat","ps_car_07_cat","ps_car_08_cat",
      "ps_car_09_cat","ps_car_10_cat","ps_car_11_cat",]
sns.despine(left=True)

# sns.countplot(x=pd.DataFrame(np.log10(train["ps_ind_02_cat"].value_counts(dropna=False))),ax=axes[0,0])
sns.countplot(train[col[0]].cat.add_categories("NA").fillna('NA'),ax=plt.subplot(gs[0, 0:2]) ,log=True)
sns.countplot(train[col[1]].cat.add_categories("NA").fillna('NA'),ax=plt.subplot(gs[0, 2]) ,log=True)
sns.countplot(train[col[3]].cat.add_categories("NA").fillna('NA'),ax=plt.subplot(gs[0, 3]) ,log=True)
sns.countplot(train[col[2]].cat.add_categories("NA").fillna('NA'),ax=plt.subplot(gs[1, 0:2]) ,log=True)
sns.countplot(train[col[4]].cat.add_categories("NA").fillna('NA'),ax=plt.subplot(gs[1, 2:4]),log=True)
sns.countplot(train[col[5]].cat.add_categories("NA").fillna('NA'),ax=plt.subplot(gs[2, 0:2]),log=True)
sns.countplot(train[col[6]].cat.add_categories("NA").fillna('NA'),ax=plt.subplot(gs[2, 2:4]),log=True)

sns.countplot(train[col[7]].cat.add_categories("NA").fillna('NA'),ax=plt.subplot(gs[3, 0:4]),log=True )
plt.tight_layout()


# ### 4.5 Integer features part 1:ind and car

# In[17]:


# f, axes = plt.subplots(1, 4, figsize=(10, 8), sharex=False)
fig = plt.figure(figsize=(15, 7))
plt.style.use('ggplot')
gs = GridSpec(2, 4)
sns.despine(left=True)

# sns.countplot(x=pd.DataFrame(np.log10(train["ps_ind_02_cat"].value_counts(dropna=False))),ax=axes[0,0])
sns.countplot(x="ps_ind_01", data=train,ax=plt.subplot(gs[0, 0:2]) )
sns.countplot(x="ps_ind_03", data=train,ax=plt.subplot(gs[0, 2:]) )

sns.countplot(x="ps_ind_14", data=train,ax=plt.subplot(gs[1, 0  ]),log=True)
sns.countplot(x="ps_ind_15", data=train,ax=plt.subplot(gs[1, 1:3]))
sns.countplot(x="ps_car_11", data=train,ax=plt.subplot(gs[1, 3  ]))

plt.tight_layout()


# ### 4.6 Integer features part 2:calc

# In[18]:


fig = plt.figure(figsize=(15, 10))
plt.style.use('ggplot')
gs = GridSpec(3, 4)
sns.despine(left=True)

sns.countplot(x="ps_calc_04", data=train,ax=plt.subplot(gs[0, 0]) )
sns.countplot(x="ps_calc_05", data=train,ax=plt.subplot(gs[0, 1]) )
sns.countplot(x="ps_calc_06", data=train,ax=plt.subplot(gs[0, 2]) )
sns.countplot(x="ps_calc_07", data=train,ax=plt.subplot(gs[0, 3]) )

sns.countplot(x="ps_calc_08", data=train,ax=plt.subplot(gs[1, 0]) )
sns.countplot(x="ps_calc_09", data=train,ax=plt.subplot(gs[1, 1]) )
sns.countplot(x="ps_calc_10", data=train,ax=plt.subplot(gs[1, 2]) ,color='blue')
# plt.subplot(gs[1, 2]).hist(train['ps_calc_10'])
sns.countplot(x=train["ps_calc_11"],ax=plt.subplot(gs[1, 3]),color='blue')
# plt.subplot(gs[1, 3]).hist(train["ps_calc_11"])

sns.countplot(x="ps_calc_12", data=train,ax=plt.subplot(gs[2, 0  ]))
sns.countplot(x="ps_calc_13", data=train,ax=plt.subplot(gs[2, 1]))
sns.countplot(x="ps_calc_14", data=train,ax=plt.subplot(gs[2, 2:  ]),color='blue')
# plt.subplot(gs[2, 2:  ]).hist(train['ps_calc_14'])
plt.tight_layout()


# ### 4.7 Integer features part 1: reg and calc

# In[20]:


fig = plt.figure(figsize=(15, 7))
plt.style.use('ggplot')
gs = GridSpec(2, 3)
sns.despine(left=True)

# sns.countplot(x="ps_reg_01", data=train,ax=plt.subplot(gs[0, 0]) ,color='green')
# sns.countplot(x="ps_reg_02", data=train,ax=plt.subplot(gs[0, 1]) ,color='green')
# sns.countplot(x=train[train["ps_reg_03"].isnull()==False]["ps_reg_03"], data=train,ax=plt.subplot(gs[0, 2]) ,color='green')

# sns.countplot(x="ps_calc_01", data=train,ax=plt.subplot(gs[1, 0]) ,color='blue')
# sns.countplot(x="ps_calc_02", data=train,ax=plt.subplot(gs[1, 1]) ,color='blue')
# sns.countplot(x="ps_calc_03", data=train,ax=plt.subplot(gs[1, 2]) ,color='blue')

plt.subplot(gs[0, 0]).hist(train[train["ps_reg_01"].isnull()==False]["ps_reg_01"],color='green',bins=20)
plt.subplot(gs[0, 1]).hist(train[train["ps_reg_02"].isnull()==False]["ps_reg_02"],color='green',bins=20)
plt.subplot(gs[0, 2]).hist(train[train["ps_reg_03"].isnull()==False]["ps_reg_03"],color='green',bins=50)

plt.subplot(gs[1, 0]).hist(train[train["ps_calc_01"].isnull()==False]["ps_calc_01"],color='blue')
plt.subplot(gs[1, 1]).hist(train[train["ps_calc_02"].isnull()==False]["ps_calc_02"],color='blue')
plt.subplot(gs[1, 2]).hist(train[train["ps_calc_03"].isnull()==False]["ps_calc_03"],color='blue')
plt.tight_layout()



# ### 4.8 Float features part 2: car

# In[22]:


fig = plt.figure(figsize=(15, 10))
plt.style.use('ggplot')
gs = GridSpec(2, 2)
sns.despine(left=True)

# sns.countplot(x="ps_reg_01", data=train,ax=plt.subplot(gs[0, 0]) ,color='green')
# sns.countplot(x="ps_reg_02", data=train,ax=plt.subplot(gs[0, 1]) ,color='green')
# sns.countplot(x=train[train["ps_reg_03"].isnull()==False]["ps_reg_03"], data=train,ax=plt.subplot(gs[0, 2]) ,color='green')

# sns.countplot(x="ps_calc_01", data=train,ax=plt.subplot(gs[1, 0]) ,color='blue')
# sns.countplot(x="ps_calc_02", data=train,ax=plt.subplot(gs[1, 1]) ,color='blue')
# sns.countplot(x="ps_calc_03", data=train,ax=plt.subplot(gs[1, 2]) ,color='blue')

plt.subplot(gs[0, 0]).hist(train[train["ps_car_12"].isnull()==False]["ps_car_12"],color='red',bins=20)
plt.subplot(gs[0, 1]).hist(train[train["ps_car_13"].isnull()==False]["ps_car_13"],color='red',bins=20)
plt.subplot(gs[1, 0]).hist(train[train["ps_car_14"].isnull()==False]["ps_car_14"],color='red',bins=50)

plt.subplot(gs[1, 1]).hist(train[train["ps_car_15"].isnull()==False]["ps_car_15"],color='red',bins=50)
plt.tight_layout()


# ### 4.9 Target variable

# ### More details on missing values

# In[23]:


fig = plt.figure(figsize=(8, 7))
plt.style.use('ggplot')
sns.countplot(x="target", data=train,color='green')


# ## 5 Claim rates for individual features

# ### 5.1 Binary features part 1

# In[24]:


fig = plt.figure(figsize=(15, 7))
plt.style.use('ggplot')
rowno = 2
colno = 4
gs = GridSpec(rowno,colno)
sns.despine(left=True)

t_columns=['ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin',
          'ps_ind_12_bin','ps_ind_13_bin']
i = 0
for col in t_columns:
    t = train.groupby([col,'target'])['id'].count().unstack()
    t = t.assign(frac_claim=t[1]/(t[1]+t[0])*100,
             lwr=get_binCI(t[1],(t[1]+t[0]))[0]*100,
             upr=get_binCI(t[1],(t[1]+t[0]))[1]*100)
    t = t.reset_index()
    yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
    sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[i//colno, i%colno])).set(ylabel='Claims [%]')
    plt.subplot(gs[i//colno, i%colno]).errorbar(t[col], t['frac_claim'], yerr=yerr,fmt='o',capsize=10,color='black',capthick=2,elinewidth=2)
    i += 1
plt.tight_layout()


# ### 5.2 Binary features part 2

# In[25]:


fig = plt.figure(figsize=(15, 7))
plt.style.use('ggplot')
rowno = 2
colno = 5
gs = GridSpec(rowno,colno)
sns.despine(left=True)

t_columns=['ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin',
          'ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin',
          'ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin',]
i = 0
for col in t_columns:
    t = train.groupby([col,'target'])['id'].count().unstack()
    t = t.assign(frac_claim=t[1]/(t[1]+t[0])*100,
             lwr=get_binCI(t[1],(t[1]+t[0]))[0]*100,
             upr=get_binCI(t[1],(t[1]+t[0]))[1]*100)
    t = t.reset_index()
    yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
    sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[i//colno, i%colno])).set(ylabel='Claims [%]')
    plt.subplot(gs[i//colno, i%colno]).errorbar(t[col], t['frac_claim'], yerr=yerr,fmt='o',capsize=10,color='black',capthick=2,elinewidth=2)
    i += 1
plt.tight_layout()



# ### 5.3 Categorical features part 1

# In[26]:


fig = plt.figure(figsize=(15, 7))
plt.style.use('ggplot')
rowno = 3
colno = 2
gs = GridSpec(rowno,colno)
sns.despine(left=True)

t_columns=['ps_ind_02_cat','ps_ind_04_cat','ps_ind_05_cat',
          'ps_car_01_cat','ps_car_02_cat','ps_car_03_cat']

temp = train[t_columns+['target','id']]

for i in t_columns:
    temp[i] = temp[i].cat.add_categories("NA").fillna('NA')


i = 0
for col in t_columns:
    t = temp.groupby([col,'target'])['id'].count().unstack()
    t = t.assign(frac_claim=t[1]/(t[1]+t[0])*100,
             lwr=get_binCI(t[1],(t[1]+t[0]))[0]*100,
             upr=get_binCI(t[1],(t[1]+t[0]))[1]*100)
    t = t.reset_index()
    yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
    sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[i//colno, i%colno])).set(ylabel='Claims [%]')
    plt.subplot(gs[i//colno, i%colno]).errorbar(t[col].astype('str'), t['frac_claim'], yerr=yerr,fmt='o',capsize=10,color='black',capthick=2,elinewidth=2)
    i += 1
plt.tight_layout()



# ### 5.4 Categorical features part 2

# In[27]:


fig = plt.figure(figsize=(15, 7))
plt.style.use('ggplot')
rowno = 4
colno = 4
gs = GridSpec(rowno,colno)
sns.despine(left=True)



def gett(col):
    if train[col].isnull().sum() > 0:
        temp = train[[col,'target','id']]
        temp[col] =temp[col].cat.add_categories("NA").fillna('NA')
        t = temp.groupby([col,'target'])['id'].count().unstack()
    else:
        t = train.groupby([col,'target'])['id'].count().unstack()
    t = t.assign(frac_claim=t[1]/(t[1]+t[0])*100,
             lwr=get_binCI(t[1],(t[1]+t[0]))[0]*100,
             upr=get_binCI(t[1],(t[1]+t[0]))[1]*100)
    t = t.reset_index()
    return t


col ='ps_car_04_cat'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[0, 0:2])).set(ylabel='Claims [%]')
plt.subplot(gs[0, 0:2]).errorbar(t[col].astype('str'), t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='black',capthick=2,elinewidth=2)

col ='ps_car_05_cat'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[0, 2])).set(ylabel='Claims [%]')
plt.subplot(gs[0, 2]).errorbar(t[col].astype('str'), t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='black',capthick=2,elinewidth=2)
col = 'ps_car_07_cat'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[0, 3])).set(ylabel='Claims [%]')
plt.subplot(gs[0, 3]).errorbar(t[col].astype('str'), t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='black',capthick=2,elinewidth=2)

col ='ps_car_06_cat'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 0:2])).set(ylabel='Claims [%]')
plt.subplot(gs[1, 0:2]).errorbar(t[col].astype('str'), t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='black',capthick=2,elinewidth=2)
col ='ps_car_08_cat'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 2:])).set(ylabel='Claims [%]')
plt.subplot(gs[1, 2:]).errorbar(t[col].astype('str'), t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='black',capthick=2,elinewidth=2)
col = 'ps_car_09_cat'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[2, 0:2])).set(ylabel='Claims [%]')
plt.subplot(gs[2, 0:2]).errorbar(t[col].astype('str'), t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='black',capthick=2,elinewidth=2)
col = 'ps_car_10_cat'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[2, 2:])).set(ylabel='Claims [%]')
plt.subplot(gs[2, 2:]).errorbar(t[col].astype('str'), t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='black',capthick=2,elinewidth=2)
col = 'ps_car_11_cat'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[3,:])).set(ylabel='Claims [%]')
plt.subplot(gs[3, :]).errorbar(t[col].astype('str'), t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=5,color='black',capthick=1)
plt.tight_layout()


# ### 5.5 Integers features part 1

# In[28]:


fig = plt.figure(figsize=(15, 7))
plt.style.use('ggplot')
rowno = 2
colno = 4
gs = GridSpec(rowno,colno)
sns.despine(left=True)


def gett(col):
    t = train.groupby([col,'target'])['id'].count().unstack()
    t = t.assign(frac_claim=t[1]/(t[1]+t[0])*100,
             lwr=get_binCI(t[1],(t[1]+t[0]))[0]*100,
             upr=get_binCI(t[1],(t[1]+t[0]))[1]*100)
    t = t.reset_index()
    return t

col ='ps_ind_01'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[0, 0:2])).set(ylabel='Claims [%]')
plt.subplot(gs[0, 0:2]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='orange',capthick=2,elinewidth=2)
plt.subplot(gs[0, 0:2]).set(ylabel='Claims [%]',xlabel=col)

col ='ps_ind_03'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[0, 2:])).set(ylabel='Claims [%]')
plt.subplot(gs[0, 2:]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='orange',capthick=2,elinewidth=2)
plt.subplot(gs[0, 2:]).set(ylabel='Claims [%]',xlabel=col)

col = 'ps_ind_14'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 0])).set(ylabel='Claims [%]')
plt.subplot(gs[1, 0]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='orange',capthick=2,elinewidth=2)
plt.subplot(gs[1, 0]).set(ylabel='Claims [%]',xlabel=col)

col ='ps_ind_15'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 1:3])).set(ylabel='Claims [%]')
plt.subplot(gs[1, 1:3]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='orange',capthick=2,elinewidth=2)
plt.subplot(gs[1, 1:3]).set(ylabel='Claims [%]',xlabel=col)


col ='ps_car_11'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 3])).set(ylabel='Claims [%]')
plt.subplot(gs[1, 3]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='red',capthick=2,elinewidth=2)
plt.subplot(gs[1, 3]).set(ylabel='Claims [%]',xlabel=col)


plt.tight_layout()


# ### 5.6 Integers features part 2

# In[29]:


fig = plt.figure(figsize=(15, 7))
plt.style.use('ggplot')
rowno = 3
colno = 4
gs = GridSpec(rowno,colno)
sns.despine(left=True)


col ='ps_calc_04'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[0, 0:2])).set(ylabel='Claims [%]')
plt.subplot(gs[0, 0]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='blue',capthick=2,elinewidth=2)
plt.subplot(gs[0, 0]).set(ylabel='Claims [%]',xlabel=col)

col ='ps_calc_05'
t= gett(col)
t = t[t[col]<6]
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[0, 2:])).set(ylabel='Claims [%]')
plt.subplot(gs[0, 1]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='orange',capthick=2,elinewidth=2)
plt.subplot(gs[0, 1]).set(ylabel='Claims [%]',xlabel=col)

col = 'ps_calc_06'
t= gett(col)
t = t[t[col]>2]
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 0])).set(ylabel='Claims [%]')
plt.subplot(gs[0, 2]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='orange',capthick=2,elinewidth=2)
plt.subplot(gs[0, 2]).set(ylabel='Claims [%]',xlabel=col)

col ='ps_calc_07'
t= gett(col)
t =t[t[col]<8]
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 1:3])).set(ylabel='Claims [%]')
plt.subplot(gs[0, 3]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='orange',capthick=2,elinewidth=2)
plt.subplot(gs[0, 3]).set(ylabel='Claims [%]',xlabel=col)


col ='ps_calc_08'
t= gett(col)
t =t[t[col]>2]
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 3])).set(ylabel='Claims [%]')
plt.subplot(gs[1, 0]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='red',capthick=2,elinewidth=2)
plt.subplot(gs[1, 0]).set(ylabel='Claims [%]',xlabel=col)


col ='ps_calc_09'
t= gett(col)
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 3])).set(ylabel='Claims [%]')
plt.subplot(gs[1, 1]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='red',capthick=2,elinewidth=2)
plt.subplot(gs[1, 1]).set(ylabel='Claims [%]',xlabel=col)

col ='ps_calc_10'
sns.distplot(train[col],ax=plt.subplot(gs[1, 2]),hist_kws={"alpha": 0.5}).set(ylabel='Claims [%]')

col ='ps_calc_11'
sns.distplot(train[col],ax=plt.subplot(gs[1, 3]),hist_kws={"alpha": 0.5}).set(ylabel='Claims [%]')


col ='ps_calc_12'
t= gett(col)
t =t[t[col]<9]
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 3])).set(ylabel='Claims [%]')
plt.subplot(gs[2, 0]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='red',capthick=2,elinewidth=2)
plt.subplot(gs[2, 0]).set(ylabel='Claims [%]',xlabel=col)

col ='ps_calc_13'
t= gett(col)
t =t[t[col]<12]
yerr = (-t['lwr']+t['frac_claim'],t['upr']-t['frac_claim'])
# sns.barplot(t[col],t['frac_claim'],ax=plt.subplot(gs[1, 3])).set(ylabel='Claims [%]')
plt.subplot(gs[2, 1]).errorbar(t[col], t['frac_claim'], yerr=yerr,
                                            fmt='o',capsize=10,color='red',capthick=2,elinewidth=2)
plt.subplot(gs[2, 1]).set(ylabel='Claims [%]',xlabel=col)

col ='ps_calc_14'
sns.distplot(train[col],ax=plt.subplot(gs[2, 2:])).set(ylabel='Claims [%]')



plt.tight_layout()


# ### 5.7 Float features part 1

# In[30]:


fig = plt.figure(figsize=(15, 7))
plt.style.use('ggplot')
rowno = 2
colno = 3
gs = GridSpec(rowno,colno)
sns.despine(left=True)

t_columns=['ps_reg_01','ps_reg_02','ps_reg_03',
          'ps_calc_01','ps_calc_02','ps_calc_03']
i = 0
for col in t_columns:
#     sns.distplot(train[train[col].isnull()==False][col],ax=plt.subplot(gs[i//colno, i%colno]),hist=False).set(ylabel='Density')
    sns.kdeplot(train[(train[col].isnull()==False) & (train['target']==0)][col],
                 ax=plt.subplot(gs[i//colno, i%colno]),
                 color="green",
                 shade=True)
    
    sns.kdeplot(train[(train[col].isnull()==False) & (train['target']==1)][col],
                ax=plt.subplot(gs[i//colno, i%colno]), color="blue", shade=True)

    i +=1


# ### 5.8 Float features part 2

# In[31]:


fig = plt.figure(figsize=(15, 7))
plt.style.use('ggplot')
rowno = 2
colno = 2
gs = GridSpec(rowno,colno)
sns.despine(left=True)

t_columns=['ps_car_12','ps_car_13','ps_car_14',
          'ps_car_15']
i = 0
for col in t_columns:
#     sns.distplot(train[train[col].isnull()==False][col],ax=plt.subplot(gs[i//colno, i%colno]),hist=False).set(ylabel='Density')
    sns.kdeplot(train[(train[col].isnull()==False) & (train['target']==0)][col],
                 ax=plt.subplot(gs[i//colno, i%colno]),
                 color="green",
                 shade=True)
    
    sns.kdeplot(train[(train[col].isnull()==False) & (train['target']==1)][col],
                ax=plt.subplot(gs[i//colno, i%colno]), color="blue", shade=True)

    i +=1


# ## 6 Multi-feature comparisons

# ### 6.1 Correlation overview

# In[32]:


# df[df.columns.difference(['b'])]
# df.drop('b', axis=1)
t =train.drop(list(train.columns[train.columns.str[:7]=='ps_calc'])+
           ['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_car_10_cat', 'id'],axis=1)
t.reset_index(drop=True,inplace=True)
t = t[t.isnull().sum(axis=1)==0]

intcol = t.columns[(t.columns.str[-3:]=='cat') | (t.columns.str[-3:]=='bin') |(t.columns.str=='target')]
for col in intcol:
    t[col] = t[col].astype(np.int32)

# Compute the correlation matrix
corr = t.corr(method='spearman')

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
plt.style.use('ggplot')

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, mask=mask,vmax=1,center=0,
            square=False, linewidths=.5, cbar_kws={"shrink": .5})

plt.tight_layout()


# In[33]:


# train %>%
#   select(ps_ind_12_bin, ps_ind_14, ps_ind_16_bin, ps_ind_17_bin, ps_ind_18_bin, ps_reg_02,
#          ps_reg_03, ps_car_12, ps_car_13, ps_car_14, ps_car_15, ps_car_02_cat, ps_car_04_cat) %>%
#   mutate_at(vars(ends_with("cat")), funs(as.integer)) %>%
#   mutate_at(vars(ends_with("bin")), funs(as.integer)) %>%
#   cor(use="complete.obs", method = "spearman") %>%
#   corrplot(type="lower", tl.col = "black",  diag=FALSE, method = "number")

t =train[['ps_ind_12_bin', 'ps_ind_14', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_02',
         'ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_car_02_cat', 'ps_car_04_cat']]
t = t[t.isnull().sum(axis=1)==0]

intcol = t.columns[(t.columns.str[-3:]=='cat') | (t.columns.str[-3:]=='bin')]
for col in intcol:
    t[col] = t[col].astype(np.int32)

# Compute the correlation matrix
corr = t.corr(method='spearman')

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
plt.style.use('ggplot')

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, mask=mask,vmax=1,center=0,
            square=False, linewidths=.5, cbar_kws={"shrink": .5})
plt.tight_layout()


# Skipping 6.2 to 6.4 as I found very difficult to implement in python

# In[ ]:





# In[ ]:





# In[ ]:




