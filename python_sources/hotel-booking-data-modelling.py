#!/usr/bin/env python
# coding: utf-8

# # Hotel Booking Data Analysis
# 
# ### - Data
# #### This data set contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.
# 
# ### - Question
# #### We would like to predict if a new reservation is likely to be canceled by customers and what's the probability of it? Also, how can we utilise this infomation and what can we do about it?
# 
# ### - Goal
# #### The aim is to do a complete data analysis including exploratory data analysis, feature engineering and finally choose the best model to solve our question by model comparison and parameter tuning.
# 
# 
# ### - Content of this notebook
# #### This notebook is divided into 2 parts. 
# #### The fist part includes data visulization, exploration and some feature engineering to help us better understand our data. Can be accessed here:https://www.kaggle.com/feilinhu/hotel-booking-data-visualisation
# #### This second part is the data modelling part and we will build and compare different models to satisfy different business needs.

# # Load Data

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

htl = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
pd.set_option('display.max_columns', None)
htl.head()


# # Data Overview

# In[ ]:


# a general overview of data

htl.describe(include="all").T


# In[ ]:


# basic description about all attributes we have, including col_name,data type,NA count and unique value count.
describe = []
for col in htl.columns:
    describe.append(len(htl[col].value_counts()))
describe = pd.DataFrame(list(zip(htl.columns,htl.dtypes,htl.isnull().sum(),htl.describe(include="all").T['unique'],describe)),columns = ['col','type','NA','Ucount','count'])
describe


# In[ ]:


# find numercial viariables which can be turned to categorical viariables. 
# This time we choose 200 as we discovered some variables like days_in_waiting_list are actually categorical variables too.

to_cate = describe[describe['Ucount'].isna()][describe['count']<200]
to_cate = to_cate[1:] # y=is_canceled stays as numeric.

to_cate


# # Feature Engineering - Create and select variables

# In[ ]:


# the agent / country/ company is just too many categories and many of them just appeared once in data.
# since we want to build a stable model, we'd better do something to group the small data together to avoid overfitting.
# re-group the agent and contry and company group

agent_cnt = htl['agent'].value_counts().reset_index()
agent_cnt['sum']=agent_cnt['agent'].cumsum()
agent_cnt['sum%']=agent_cnt['sum']/agent_cnt.loc[332,'sum']
agent_cnt['re_grp'] = agent_cnt[['index','sum%']].apply(lambda x : x['index'] if x['sum%'] <= 0.8 else 'OTH', axis =1)
agent_cnt.columns = ['agent', 'count', 'acc_count', 'acc_count%','agent_grp'] 

cnty_cnt = htl['country'].value_counts().reset_index()
cnty_cnt['sum']=cnty_cnt['country'].cumsum()
cnty_cnt['sum%']=cnty_cnt['sum']/cnty_cnt.loc[176,'sum']
cnty_cnt['re_grp'] = cnty_cnt[['index','sum%']].apply(lambda x : x['index'] if x['sum%'] <= 0.8 else 'OTH', axis =1)
cnty_cnt.columns = ['country', 'count', 'acc_count', 'acc_count%','cnty_grp'] 


cmp_cnt = htl['company'].value_counts().reset_index()
cmp_cnt['sum']=cmp_cnt['company'].cumsum()
cmp_cnt['sum%']=cmp_cnt['sum']/cmp_cnt.loc[351,'sum']
cmp_cnt['re_grp'] = cmp_cnt[['index','sum%']].apply(lambda x : x['index'] if x['sum%'] <= 0.8 else 'OTH', axis =1)
cmp_cnt.columns = ['company', 'count', 'acc_count', 'acc_count%','cmp_grp'] 

print(agent_cnt)
print(cnty_cnt)
print(cmp_cnt)


# #### For each attribute we regroup the rest 20% of data as 'OTH', the top 80% data stays the same.

# In[ ]:


# To better record our actions, create a record book to map the 'before' and 'after' value

data_clean = []

data_clean.append(agent_cnt[['agent_grp']].set_index(agent_cnt['agent']).to_dict())
data_clean.append(cnty_cnt[['cnty_grp']].set_index(cnty_cnt['country']).to_dict())
data_clean.append(cmp_cnt[['cmp_grp']].set_index(cmp_cnt['company']).to_dict())


# In[ ]:


# take a look
data_clean[0]['agent_grp']


# In[ ]:


# take a look at days_in_waiting_list

plt.hist(htl['days_in_waiting_list'], bins=50)


# In[ ]:


htl['days_in_waiting_list'].value_counts()


# #### Since it has too many 0, we create a new variable for it.

# In[ ]:


# change the variable type, create new variables.
# Note we did create some new variables in Part One, but we can skip it for now because we will go back and look at the categorical variables again.

data = htl.copy()
'''
data['is_children'] = ['Y' if x > 0 else 'N' for x in data['children']]
data['is_baby'] = ['Y' if x > 0 else 'N' for x in data['babies']]
data['is_agent'] = ['Y' if x > 0 else 'N' for x in data['agent']]
data['is_company'] = ['Y' if x > 0 else 'N' for x in data['company']]
data['is_parking'] = ['Y' if x > 0 else 'N' for x in data['required_car_parking_spaces']]
data['is_request'] = ['Y' if x > 0 else 'N' for x in data['total_of_special_requests']]
data['is_canceled_before'] = ['Y' if x > 0 else 'N' for x in data['previous_cancellations']]
data['is_changed'] = ['Y' if x > 0 else 'N' for x in data['booking_changes']]
'''
data['is_waited'] = ['Y' if x > 0 else 'N' for x in data['days_in_waiting_list']]

data['is_room_changed'] = data[['reserved_room_type','assigned_room_type']].apply(lambda x:x['reserved_room_type'] != x['assigned_room_type'], axis=1)

for i in to_cate['col']:
    data[i] = data[i].astype('str')
    
data['is_room_changed'] = data['is_room_changed'].astype('str')


# In[ ]:


# mapping back company, country and agent values we created before

data['agent'] = data['agent'].map(data_clean[0]['agent_grp'])
data['country'] = data['country'].map(data_clean[1]['cnty_grp'])
data['company'] = data['company'].map(data_clean[2]['cmp_grp'])


# In[ ]:


# Fill NA with 0: NA data for agent/company just means the customer is not from an agent/company. 
# The country value is simply missing and we can fill it this way and may intergrate it into other groups later.

data.agent.fillna(value='0', inplace=True)
data.company.fillna(value='0', inplace=True)
data.country.fillna(value= 'MISSING', inplace=True)


# In[ ]:


# Write into record

data_clean.append({"agentNaN":'0'})
data_clean.append({"countryNaN":'MISSING'})
data_clean.append({"companyNaN":'0'})


# In[ ]:


# make them categorical

data['agent'] = data['agent'].astype('str')
data['company'] = data['company'].astype('str')

# and now check the new data

cate = [var for var in data.columns if data[var].dtypes == 'object']
num = [var for var in data.columns if data[var].dtypes != 'object']

describe_new = []
for col in data.columns:
    describe_new.append(len(data[col].value_counts()))
describe_new = pd.DataFrame(list(zip(data.columns,data.dtypes,data.isnull().sum(),data.describe(include="all").T['unique'],describe_new)),columns = ['col','type','NA','Ucount','count'])
describe_new


# ### Only lead_time and adr are continuous variables.

# In[ ]:


# check correlation for continuous variables

cor = data.corr(method='pearson')
cor


# In[ ]:


# Also there are reservation_status: 100% relate to 'is_canceled'.
# And we need to remove 'reservation_status_date' and 'days_in_waiting_list' becuase we just built new categorical variable.

cate = cate[0:23] + cate[24:27] + cate[29:] # remove  'reservation_status', 'reservation_status_date','days_in_waiting_list'
num = num[1:] #remove Y


# In[ ]:


# Create new dataset

y = 'is_canceled'
X = data[cate + num].copy()

Y = data[y].copy()


# In[ ]:


# Here we select variables using WOE and IV value. I used this function to fast calculate the value. Thanks to the author!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score

__author__ = 'Denis Surzhko'


class WoE:
    """
    Basic functionality for WoE bucketing of continuous and discrete variables
    :param self.bins: DataFrame WoE transformed variable and all related statistics
    :param self.iv: Information Value of the transformed variable
    """
    def __init__(self, qnt_num=16, min_block_size=16, spec_values=None, v_type='c', bins=None, t_type='b'):
        """
        :param qnt_num: Number of buckets (quartiles) for continuous variable split
        :param min_block_size: minimum number of observation in each bucket (continuous variables)
        :param spec_values: List or Dictionary {'label': value} of special values (frequent items etc.)
        :param v_type: 'c' for continuous variable, 'd' - for discrete
        :param bins: Predefined bucket borders for continuous variable split
        :t_type : Binary 'b' or continous 'c' target variable
        :return: initialized class
        """
        self.__qnt_num = qnt_num  # Num of buckets/quartiles
        self._predefined_bins = None if bins is None else np.array(bins)  # user bins for continuous variables
        self.type = v_type  # if 'c' variable should be continuous, if 'd' - discrete
        self._min_block_size = min_block_size  # Min num of observation in bucket
        self._gb_ratio = None  # Ratio of good and bad in the sample
        self.bins = None  # WoE Buckets (bins) and related statistics
        self.df = None  # Training sample DataFrame with initial data and assigned woe
        self.qnt_num = None  # Number of quartiles used for continuous part of variable binning
        self.t_type = t_type  # Type of target variable
        if type(spec_values) == dict:  # Parsing special values to dict for cont variables
            self.spec_values = {}
            for k, v in spec_values.items():
                if v.startswith('d_'):
                    self.spec_values[k] = v
                else:
                    self.spec_values[k] = 'd_' + v
        else:
            if spec_values is None:
                self.spec_values = {}
            else:
                self.spec_values = {i: 'd_' + str(i) for i in spec_values}

    def fit(self, x, y):
        """
        Fit WoE transformation
        :param x: continuous or discrete predictor
        :param y: binary target variable
        :return: WoE class
        """
        # Data quality checks
        if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
            raise TypeError("pandas.Series type expected")
        if not x.size == y.size:
            raise Exception("Y size don't match Y size")
        # Calc total good bad ratio in the sample
        t_bad = np.sum(y)
        if t_bad == 0 or t_bad == y.size:
            raise ValueError("There should be BAD and GOOD observations in the sample")
        if np.max(y) > 1 or np.min(y) < 0:
            raise ValueError("Y range should be between 0 and 1")
        # setting discrete values as special values
        if self.type == 'd':
            sp_values = {i: 'd_' + str(i) for i in x.unique()}
            if len(sp_values) > 100:
                raise type("DiscreteVarOverFlowError", (Exception,),
                           {"args": ('Discrete variable with too many unique values (more than 100)',)})
            else:
                if self.spec_values:
                    sp_values.update(self.spec_values)
                self.spec_values = sp_values
        # Make data frame for calculations
        df = pd.DataFrame({"X": x, "Y": y, 'order': np.arange(x.size)})
        # Separating NaN and Special values
        df_sp_values, df_cont = self._split_sample(df)
        # # labeling data
        df_cont, c_bins = self._cont_labels(df_cont)
        df_sp_values, d_bins = self._disc_labels(df_sp_values)
        # getting continuous and discrete values together
        self.df = df_sp_values.append(df_cont)
        self.bins = d_bins.append(c_bins)
        # calculating woe and other statistics
        self._calc_stat()
        # sorting appropriately for further cutting in transform method
        self.bins.sort_values('bins', inplace=True)
        # returning to original observation order
        self.df.sort_values('order', inplace=True)
        self.df.set_index(x.index, inplace=True)
        return self

    def fit_transform(self, x, y):
        """
        Fit WoE transformation
        :param x: continuous or discrete predictor
        :param y: binary target variable
        :return: WoE transformed variable
        """
        self.fit(x, y)
        return self.df['woe']

    def _split_sample(self, df):
        if self.type == 'd':
            return df, None
        sp_values_flag = df['X'].isin(self.spec_values.keys()).values | df['X'].isnull().values
        df_sp_values = df[sp_values_flag].copy()
        df_cont = df[np.logical_not(sp_values_flag)].copy()
        return df_sp_values, df_cont

    def _disc_labels(self, df):
        df['labels'] = df['X'].apply(
            lambda x: self.spec_values[x] if x in self.spec_values.keys() else 'd_' + str(x))
        d_bins = pd.DataFrame({"bins": df['X'].unique()})
        d_bins['labels'] = d_bins['bins'].apply(
            lambda x: self.spec_values[x] if x in self.spec_values.keys() else 'd_' + str(x))
        return df, d_bins

    def _cont_labels(self, df):
        # check whether there is a continuous part
        if df is None:
            return None, None
        # Max buckets num calc
        self.qnt_num = int(np.minimum(df['X'].unique().size / self._min_block_size, self.__qnt_num)) + 1
        # cuts - label num for each observation, bins - quartile thresholds
        bins = None
        cuts = None
        if self._predefined_bins is None:
            try:
                cuts, bins = pd.qcut(df["X"], self.qnt_num, retbins=True, labels=False)
            except ValueError as ex:
                if ex.args[0].startswith('Bin edges must be unique'):
                    ex.args = ('Please reduce number of bins or encode frequent items as special values',) + ex.args
                    raise
            bins = np.append((-float("inf"), ), bins[1:-1])
        else:
            bins = self._predefined_bins
            if bins[0] != float("-Inf"):
                bins = np.append((-float("inf"), ), bins)
            cuts = pd.cut(df['X'], bins=np.append(bins, (float("inf"), )),
                          labels=np.arange(len(bins)).astype(str))
        df["labels"] = cuts.astype(str)
        c_bins = pd.DataFrame({"bins": bins, "labels": np.arange(len(bins)).astype(str)})
        return df, c_bins

    def _calc_stat(self):
        # calculating WoE
        # stat = self.df.groupby("labels")['Y'].agg({'mean': np.mean, 'bad': np.count_nonzero, 'obs': np.size}).copy()
        stat = self.df.groupby("labels")["Y"].agg([np.mean, np.count_nonzero, np.size])
        stat = stat.rename(columns={'mean': 'mean', 'count_nonzero':'bad', 'size':'obs'})
        if self.t_type != 'b':
            stat['bad'] = stat['mean'] * stat['obs']
        stat['good'] = stat['obs'] - stat['bad']
        t_good = np.maximum(stat['good'].sum(), 0.5)
        t_bad = np.maximum(stat['bad'].sum(), 0.5)
        stat['woe'] = stat.apply(self._bucket_woe, axis=1) + np.log(t_good / t_bad)
        iv_stat = (stat['bad'] / t_bad - stat['good'] / t_good) * stat['woe']
        self.iv = iv_stat.sum()
        # adding stat data to bins
        self.bins = pd.merge(stat, self.bins, left_index=True, right_on=['labels'])
        label_woe = self.bins[['woe', 'labels']].drop_duplicates()
        self.df = pd.merge(self.df, label_woe, left_on=['labels'], right_on=['labels'])

    def transform(self, x):
        """
        Transforms input variable according to previously fitted rule
        :param x: input variable
        :return: DataFrame with transformed with original and transformed variables
        """
        if not isinstance(x, pd.Series):
            raise TypeError("pandas.Series type expected")
        if self.bins is None:
            raise Exception('Fit the model first, please')
        df = pd.DataFrame({"X": x, 'order': np.arange(x.size)})
        # splitting to discrete and continous pars
        df_sp_values, df_cont = self._split_sample(df)

        # function checks existence of special values, raises error if sp do not exist in training set
        def get_sp_label(x_):
            if x_ in self.spec_values.keys():
                return self.spec_values[x_]
            else:
                str_x = 'd_' + str(x_)
                if str_x in list(self.bins['labels']):
                    return str_x
                else:
                    raise ValueError('Value ' + str_x + ' does not exist in the training set')
        # assigning labels to discrete part
        df_sp_values['labels'] = df_sp_values['X'].apply(get_sp_label)
        # assigning labels to continuous part
        c_bins = self.bins[self.bins['labels'].apply(lambda z: not z.startswith('d_'))]
        if not self.type == 'd':
            cuts = pd.cut(df_cont['X'], bins=np.append(c_bins["bins"], (float("inf"), )), labels=c_bins["labels"])
            df_cont['labels'] = cuts.astype(str)
        # Joining continuous and discrete parts
        df = df_sp_values.append(df_cont)
        # assigning woe
        df = pd.merge(df, self.bins[['woe', 'labels']], left_on=['labels'], right_on=['labels'])
        # returning to original observation order
        df.sort_values('order', inplace=True)
        return df.set_index(x.index)

    def merge(self, label1, label2=None):
        """
        Merge of buckets with given labels
        In case of discrete variable, both labels should be provided. As the result labels will be marget to one bucket.
        In case of continous variable, only label1 should be provided. It will be merged with the next label.
        :param label1: first label to merge
        :param label2: second label to merge
        :return:
        """
        spec_values = self.spec_values.copy()
        c_bins = self.bins[self.bins['labels'].apply(lambda x: not x.startswith('d_'))].copy()
        if label2 is None and not label1.startswith('d_'):  # removing bucket for continuous variable
            c_bins = c_bins[c_bins['labels'] != label1]
        else:
            if not (label1.startswith('d_') and label2.startswith('d_')):
                raise Exception('Labels should be discrete simultaneously')
            bin1 = self.bins[self.bins['labels'] == label1]['bins'].iloc[0]
            bin2 = self.bins[self.bins['labels'] == label2]['bins'].iloc[0]
            spec_values[bin1] = label1 + '_' + label2
            spec_values[bin2] = label1 + '_' + label2
        new_woe = WoE(self.__qnt_num, self._min_block_size, spec_values, self.type, c_bins['bins'], self.t_type)
        return new_woe.fit(self.df['X'], self.df['Y'])

    def plot(self,figsize):
        """
        Plot WoE transformation and default rates
        :return: plotting object
        """
        index = np.arange(self.bins.shape[0])
        bar_width = 0.8
        woe_fig = plt.figure(figsize = figsize)
        plt.title('Number of Observations and WoE per bucket')
        ax = woe_fig.add_subplot(111)
        ax.set_ylabel('Observations')
        plt.xticks(index + bar_width / 2, self.bins['labels'])
        plt.bar(index, self.bins['obs'], bar_width, color='b', label='Observations')
        ax2 = ax.twinx()
        ax2.set_ylabel('Weight of Evidence')
        ax2.plot(index + bar_width / 2, self.bins['woe'], 'bo-', linewidth=4.0, color='r', label='WoE')
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        plt.legend(handles, labels)
        woe_fig.autofmt_xdate()
        return woe_fig

    def optimize(self, criterion=None, fix_depth=None, max_depth=None, cv=3):
        """
        WoE bucketing optimization (continuous variables only)
        :param criterion: binary tree split criteria
        :param fix_depth: use tree of a fixed depth (2^fix_depth buckets)
        :param max_depth: maximum tree depth for a optimum cross-validation search
        :param cv: number of cv buckets
        :return: WoE class with optimized continuous variable split
        """
        if self.t_type == 'b':
            tree_type = tree.DecisionTreeClassifier
        else:
            tree_type = tree.DecisionTreeRegressor
        m_depth = int(np.log2(self.__qnt_num))+1 if max_depth is None else max_depth
        cont = self.df['labels'].apply(lambda z: not z.startswith('d_'))
        x_train = np.array(self.df[cont]['X'])
        y_train = np.array(self.df[cont]['Y'])
        x_train = x_train.reshape(x_train.shape[0], 1)
        start = 1
        cv_scores = []
        if fix_depth is None:
            for i in range(start, m_depth):
                if criterion is None:
                    d_tree = tree_type(max_depth=i)
                else:
                    d_tree = tree_type(criterion=criterion, max_depth=i)
                scores = cross_val_score(d_tree, x_train, y_train, cv=cv)
                cv_scores.append(scores.mean())
            best = np.argmax(cv_scores) + start
        else:
            best = fix_depth
        final_tree = tree_type(max_depth=best)
        final_tree.fit(x_train, y_train)
        opt_bins = final_tree.tree_.threshold[final_tree.tree_.threshold > 0]
        opt_bins = np.sort(opt_bins)
        new_woe = WoE(self.__qnt_num, self._min_block_size, self.spec_values, self.type, opt_bins, self.t_type)
        return new_woe.fit(self.df['X'], self.df['Y'])

    @staticmethod
    def _bucket_woe(x):
        t_bad = x['bad']
        t_good = x['good']
        t_bad = 0.5 if t_bad == 0 else t_bad
        t_good = 0.5 if t_good == 0 else t_good
        return np.log(t_bad / t_good)


# In[ ]:


# Select the continuous variables by VI value 

iv_n = {}
for col in num:
    try:
        iv_n[col] = WoE(v_type='c',t_type='b',qnt_num=5, spec_values={0: '0'} ).fit(data[col].copy(),data['is_canceled'].astype(int).copy()).iv 
    except ValueError:
        try:
            iv_n[col] = WoE(v_type='c',t_type='b',qnt_num=2, spec_values={0: '0'} ).fit(data[col].copy(),data['is_canceled'].astype(int).copy()).iv
            print(col,'qnt_num=2')
        except ValueError:
            print(col)

sort_iv_n = pd.Series(iv_n).sort_values(ascending=False)


# In[ ]:


# Sort the IV value and filter by 0.02

var_n_s =  list(sort_iv_n[sort_iv_n > 0.02].index)
var_n_s


# ['lead_time', 'adr'] They are both important according to Information Value.

# In[ ]:


# But we also want to double check by statistics tests

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

sample_df = data.sample(3000) # We need to sample the data otherwise the P-value will be nonsense.
sample_y = Y[sample_df.index]

ols_n = {}
for i in num:  #date not included
    print(i,":")       
    a = i
    b = Y.name
    formula = "{} ~ C({})".format(a,b)
    fit = sm.stats.anova_lm(ols(formula=formula,data=sample_df).fit())
    print(fit)
    ols_n[i] = fit.iloc[0,4]
    print('===============================================================')
sort_ols_n = pd.Series(ols_n).sort_values(ascending=True)
sort_ols_n


# In[ ]:


# filter by P<0.001

var_ols_n = list(sort_ols_n[sort_ols_n < 0.001].index)
var_ols_n


# Only ['lead_time'] left. Okay, we will see later if 'adr' is important or not.
# 
# Now let's look at the categorical variables.

# In[ ]:


# Select the categorical variables by VI value 

iv_c = {}
for i in cate:
    iv_c[i] = WoE(v_type='d').fit(X[i].copy(), Y.copy()).iv


sort_iv_c = pd.Series(iv_c).sort_values(ascending = False)


# In[ ]:


# Sort the IV value and filter by 0.02

var_c_s =  list(sort_iv_c[sort_iv_c > 0.02].index)

var_c_s


# okay, these categorical variables can stay. For now.

# In[ ]:


# Tidy up and create new dataset to operate on

X = data[var_c_s + var_n_s].copy()
Y = data[y].copy()

X_rep = X.copy()


# # Feature Engineering - Variable Exploration

# In[ ]:


# Continous Variable Exploration

X_rep[var_n_s].describe().T


# In[ ]:


# Check the outliers and data distribution

plt.hist(X_rep['lead_time'], bins=20)


# Lead_time seems clean.

# In[ ]:


plt.hist(X_rep['adr'], bins=20)


# 'adr' clearly has outliers

# In[ ]:


X_rep['adr'].describe().T


# In[ ]:


# Take a look at the outliers

print(data[data['adr']>5000].T) 
print('==========================================================')
print(data[data['adr']<0].T)


# In[ ]:


# Assign 0 to negative value and drop the 'over 5000' data. It's ok to not do anything here becuase we will do variable discretization later.

X_rep[X_rep['adr']<0]
X_rep.loc[14969,'adr'] = 0.0

X_rep = X_rep.drop([48515])
Y = Y.drop([48515])

data_clean.append({'adr<0':'0'})
data_clean.append({'adr>5000':'DROP'})


# In[ ]:


# Take a look again

fig = plt.figure()
for i in var_n_s:
    print(i)
    ax = fig.add_subplot(2,1,var_n_s.index(i)+1)
#    plt.hist(X_rep[i], bins=20)
    sns.distplot(X_rep[i], kde=True, fit=stats.norm, ax = ax)


# In[ ]:


# Continuous variable discretization

for i in var_n_s:
    try:
        X_rep[i+'_bins'] = pd.qcut(X_rep[i],5) 
        print(Y.astype('int64').groupby(X_rep[i+'_bins']).agg(['count', 'mean']))
    except ValueError:
        print('=======================================')
        print(i,ValueError)
        print('=======================================')


# #### We can see a positive correlation between lead_time and is_canceled, meaning the earlier people book their reservation, the more likely they will have second thoughts.

# In[ ]:


# append to data_clean. And map back to the dataset.

for i in var_n_s:
    data_clean.append(X_rep[[i,i+"_bins"]].drop_duplicates().set_index(i).to_dict())
    del X_rep[i]
    X_rep.rename(columns={i+"_bins":i},inplace=True)


# #### Now we've transfered the continous variables to categorical variables.

# In[ ]:


# Start to explore categorical variables. 
# We try to figure out if the original groups are suitable for modelling or if we need to regoup them like what we did for agent/country/company.
# So we calculate the mean(since the Y is 0 or 1, the mean is just the % of Y=1) and count(frequency) for each group

var_c_ex = {}
for i in var_c_s:
    print(i,':',str(var_c_s.index(i)),':')
    DemCluster_grp = data[[i,'is_canceled']].groupby(i,as_index = False)
    DemC_C = DemCluster_grp['is_canceled'].agg({'mean' : 'mean',
                                                   'count':'count'}).sort_values("mean")
    var_c_ex[var_c_s.index(i)]=DemC_C
    print(DemC_C)
    print('===================================')


# #### For each attribute, we group by its group, the % of Y=1 and its frequency. Sorted by cancel rate.
# And we get some ideas about the regrouping.
# 
# The followings are:
# variable index, regroup num, vairable name
# * var_c_ex[1], 2, required_car_parking_spaces
# * var_c_ex[3], 2, previous_cancellations
# * var_c_ex[4], 7, agent
# * var_c_ex[6], 4, market_segment
# * var_c_ex[7], 3, total_of_special_requests
# * var_c_ex[8], 4, assigned_room_type
# * var_c_ex[9], 2, booking_changes
# * var_c_ex[10], 4, distribution_channel
# * var_c_ex[11], 2, company
# * var_c_ex[12], 2, previous_bookings_not_canceled
# * var_c_ex[15], 5, stays_in_week_nights 0 1 2 3 OTH
# * var_c_ex[16], DEL, arrival_date_week_number
# * var_c_ex[19], 3, adults
# * var_c_ex[20], 4, reserved_room_type

# In[ ]:


for i in [6,8,10,20]:
    var_c_ex[i]["count_cumsum"]=var_c_ex[i]["count"].cumsum()
    var_c_ex[i]["new_"+ var_c_ex[i].columns[0]] = var_c_ex[i]["count_cumsum"].apply(lambda x: x//(len(data)/4)).astype(int)
    
for i in [1,3,9,11,12]:
    var_c_ex[i]["count_cumsum"]=var_c_ex[i]["count"].cumsum()
    var_c_ex[i]["new_"+ var_c_ex[i].columns[0]] = var_c_ex[i]["count_cumsum"].apply(lambda x: x//(len(data)/2)).astype(int)

for i in [7,19]:
    var_c_ex[i]["count_cumsum"]=var_c_ex[i]["count"].cumsum()
    var_c_ex[i]["new_"+ var_c_ex[i].columns[0]] = var_c_ex[i]["count_cumsum"].apply(lambda x: x//(len(data)/3)).astype(int)
        
var_c_ex[4]["count_cumsum"]=var_c_ex[4]["count"].cumsum()
var_c_ex[4]["new_"+ var_c_ex[4].columns[0]] = var_c_ex[i]["count_cumsum"].apply(lambda x: x//(len(data)/7)).astype(int)

var_c_ex[15]["count_cumsum"]=var_c_ex[15]["count"].cumsum()
var_c_ex[15]["new_"+ var_c_ex[15].columns[0]] = var_c_ex[i]["count_cumsum"].apply(lambda x: x//(len(data)/5)).astype(int)

var_c_ex


# In[ ]:


# Double check the regrouping. Unfortunatly we have to manually assign some group names.

var_c_ex[3]['new_previous_cancellations'] = [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1]
var_c_ex[4]['new_agent'] = [1,1,1,1,1,1,1,0,2,2,2,2,2,3,4,5,6,6,6,6,6,6,6,6]
var_c_ex[6]['new_market_segment'][4] = 4
var_c_ex[7]['new_total_of_special_requests'][1] = 1
var_c_ex[8]['new_assigned_room_type'][11] = 3
var_c_ex[9]['new_booking_changes'] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1]
var_c_ex[10]['new_distribution_channel'] = [0,0,1,2,2]
var_c_ex[11]['new_company'][[50,60,14,16,39,43]] = [0,0,0,0,0,0]
var_c_ex[12]['new_previous_bookings_not_canceled'][[18,39,54,43]] = [0,0,0,0]
var_c_ex[15]['new_stays_in_week_nights'][1] = 1
var_c_ex[15]['new_stays_in_week_nights'][12] = 2
var_c_ex[15]['new_stays_in_week_nights'][19] = 3
var_c_ex[15]['new_stays_in_week_nights'][0:10] = 5
var_c_ex[15]['new_stays_in_week_nights'][11:13] = 5
var_c_ex[15]['new_stays_in_week_nights'][14:17] = 5
var_c_ex[15]['new_stays_in_week_nights'][18:22] = 5
var_c_ex[15]['new_stays_in_week_nights'][23:] = 5
var_c_ex[19]['new_adults'] = [0,0,1,0,2,0,0,0,0,0,0,0,0,0]
var_c_ex[20]['new_reserved_room_type'][9] = 3


# In[ ]:


# add to clean data

for i in [1,3,4,6,7,8,9,10,11,12,15,19,20]:
    data_clean.append(var_c_ex[i][[var_c_ex[i].columns[0],"new_"+ var_c_ex[i].columns[0]]].set_index(var_c_ex[i].columns[0]).to_dict())


# In[ ]:


# map back to dataset

j = 10
for i in [1,3,4,6,7,8,9,10,11,12,15,19,20]:
    X_rep[var_c_ex[i].columns[0]] = X_rep[var_c_ex[i].columns[0]].map(data_clean[j]["new_"+ var_c_ex[i].columns[0]])
    j = j+1


# In[ ]:


var_c_s = var_c_s[0:16] + var_c_s[17:]  # del arrival_date_week_number, we've already had the month attribute.


# In[ ]:


# Double check the data

X_rep[var_c_s].describe().T


# In[ ]:


# Since we transfer the continous variables to categorical ones, now we combine them together

var_c = var_c_s+var_n_s
for i in var_c:
    print(i,": ",len(X_rep[var_c][i].value_counts()))


# In[ ]:


# put them in statistic test: chisq test

sample_df = X_rep.sample(3000)
sample_y = Y[sample_df.index]

ols_c = {}
for col in var_c:
    print(col,":")
    cross_table = pd.crosstab(sample_df[col],sample_y)

    print(''' chisq = %6.4f \n p-value = %6.4f \n dof = %i \n expected_freq = %s'''
               %stats.chi2_contingency(cross_table))  
    ols_c[col] = stats.chi2_contingency(cross_table)[1]
    print('==========================================')
        
    
sort_ols_c = pd.Series(ols_c).sort_values(ascending=True)
sort_ols_c


# In[ ]:


# And filter the variables by P value

var_ols_c = list(sort_ols_c[sort_ols_c < 0.001].index)
var_ols_c


# #### Now we have narrowed down both continous and categorical variables

# In[ ]:


# Clear up the data

X_rep = X_rep[var_ols_c]


# In[ ]:


# Transfer all the variables back to continous ones.

for i in var_ols_c:
    X_rep[i+"_woe"] = WoE(v_type='d').fit_transform(X_rep[i],Y)


# In[ ]:


# append to clean data and delete the extra colomns

for i in var_ols_c:
    data_clean.append(X_rep[[i,i+"_woe"]].drop_duplicates().set_index(i).to_dict())
    del X_rep[i]
    X_rep.rename(columns={i+"_woe":i},inplace=True)

X_rep[var_ols_c].describe().T


# In[ ]:


# Use Random Forest to get feature importance

import sklearn.ensemble as ensemble

rfc = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=100, max_features=0.5, min_samples_split=100)
rfc_model = rfc.fit(X_rep, Y)
rfc_model.feature_importances_
rfc_fi = pd.DataFrame()
rfc_fi["features"] = list(X_rep.columns)
rfc_fi["importance"] = list(rfc_model.feature_importances_)
rfc_fi=rfc_fi.set_index("features",drop=True)
var_sort = rfc_fi.sort_values(by="importance",ascending=False)
var_sort.plot(kind="bar")


# In[ ]:


# filter by 0.02

var_x = list(var_sort.importance[var_sort.importance > 0.02].index)
var_x


# In[ ]:


# Now we get the final attributes.

X_rep_reduc = X_rep[var_x].copy()
X_rep_reduc.head()


# # Model Comparison

# In[ ]:


# Split to test and train

import sklearn.model_selection as model_selection

ml_data = model_selection.train_test_split(X_rep_reduc, Y, test_size=0.3, random_state=2333)
train_data, test_data, train_target, test_target = ml_data


# ### Decision Tree

# In[ ]:


from sklearn.model_selection import ParameterGrid, GridSearchCV
import sklearn.tree as tree
import sklearn.metrics as metrics

param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[7,8,10,12,15],
    'min_samples_split':[10,20,50,100,200] 
}
clf = tree.DecisionTreeClassifier(random_state = 233)
clfcv = GridSearchCV(estimator=clf
                     ,param_grid=param_grid
                     ,scoring='roc_auc'
                     ,cv=10)

clfcv.fit(train_data, train_target)

print(clfcv.best_estimator_)
print("best accuracy:%f" % clfcv.best_score_) #best accuracy:0.916438


# In[ ]:


# Create confusion matrix. We focus on precision and recall value when Y = 1

train_est  = clfcv.predict(train_data)  
train_est_p= clfcv.predict_proba(train_data)[:,1]  
test_est   = clfcv.predict(test_data) 
test_est_p = clfcv.predict_proba(test_data)[:,1] 

print(metrics.confusion_matrix(test_target, test_est,labels=[0,1]))
print(metrics.classification_report(test_target, test_est))


# In[ ]:


# Plot the ROC Curve

fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)

plt.figure(figsize=[6,6])
plt.plot(fpr_test, tpr_test, 'b-')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.text(0.4, 0.8, 'AUC = %6.4f' %metrics.auc(fpr_test, tpr_test), ha='center')
plt.show()
print('Train AUC = %6.4f' %metrics.auc(fpr_train, tpr_train)) 
print('Test AUC = %6.4f' %metrics.auc(fpr_test, tpr_test))


# In[ ]:


# How's the classification?

red, blue = sns.color_palette("Set1",2)

sns.kdeplot(test_est_p[test_target==1], shade=True, color=red)
sns.kdeplot(test_est_p[test_target==0], shade=True, color=blue)


# #### Both the Y = 1 and Y = 0 are classified pretty well when the probabilty is >0.8 or <0.2. But there are some blend-in in between.

# ### Random Forest

# In[ ]:


# Use default model

rf1 = ensemble.RandomForestClassifier(oob_score=True,random_state = 233)
rf1.fit(train_data, train_target)

print("accuracy:%f"%rf1.oob_score_)


# In[ ]:


# Now we try to tune the model. And we do it step by step.

# 1.test estimators

param_test1 = {"n_estimators":range(50,101,10)}
gsearch1 = GridSearchCV(estimator=rf1
                        ,param_grid=param_test1
                        ,scoring='roc_auc'
                        ,cv=10)

gsearch1.fit(train_data, train_target)

print(gsearch1.best_params_) 
print("best accuracy:%f" % gsearch1.best_score_) 


# In[ ]:


# 2.test criterion

param_test2 = {"criterion":['entropy','gini']}
rf2 = ensemble.RandomForestClassifier(n_estimators = 100
                                      ,random_state = 233)
gsearch2 = GridSearchCV(estimator=rf2
                        ,param_grid=param_test2
                        ,scoring='roc_auc'
                        ,cv=10)

gsearch2.fit(train_data, train_target)

print(gsearch2.best_params_) 
print("best accuracy:%f" % gsearch2.best_score_) 


# In[ ]:


# 3.test max_depth and min_sample_split

param_test3 = {'max_depth':range(11,16,2), 'min_samples_split':range(20,201,20)}

rf3 = ensemble.RandomForestClassifier(n_estimators = 100
                                      ,random_state = 233
                                      ,criterion = 'entropy')

gsearch3 = GridSearchCV(estimator = rf3
                        ,param_grid = param_test3
                        ,scoring='roc_auc'
                        ,cv=10)

gsearch3.fit(train_data, train_target)
gsearch3.best_params_, gsearch3.best_score_


# In[ ]:


# Fit test data and calculate the AUC

test_est   = gsearch3.predict(test_data)
test_est_p = gsearch3.predict_proba(test_data)[:,1] 
train_est  = gsearch3.predict(train_data)
train_est_p= gsearch3.predict_proba(train_data)[:,1] 

fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_est_p)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_est_p)


print('Train AUC = %.4f' %metrics.auc(fpr_train, tpr_train)) 
print('Test AUC = %.4f' %metrics.auc(fpr_test, tpr_test)) 
print(metrics.confusion_matrix(test_target, test_est, labels=[0, 1]))


# In[ ]:


# ROC Curve

plt.figure(figsize=[4, 4])
plt.plot(fpr_test, tpr_test, 'b-')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.text(0.4, 0.8, 'AUC = %6.4f' %metrics.auc(fpr_test, tpr_test), ha='center')
plt.show()


# ### Logistic Regression

# In[ ]:


# Need to scale the data first

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

train_data_scale = min_max_scaler.fit_transform(train_data)
test_data_scale = min_max_scaler.fit_transform(test_data)


# In[ ]:


# Fit the best model

import sklearn.linear_model as linear_model

logistic_model = linear_model.LogisticRegression(class_weight=None
                                                 ,dual=False
                                                 ,fit_intercept=True
                                                 ,intercept_scaling=1
                                                 ,penalty='l1'
                                                 ,random_state=233
                                                 ,solver='liblinear'
                                                 ,tol=0.001)
C = np.logspace(-3,1,base=10) 

param_grid = {'C': C}

clf_cv = GridSearchCV(estimator=logistic_model, 
                      param_grid=param_grid, 
                      cv=4, 
                      scoring='roc_auc')

clf_cv.fit(train_data_scale, train_target)


# In[ ]:


# Calculate AUC

print('C:',clf_cv.best_score_)

test_predict = clf_cv.predict(test_data_scale)
train_predict = clf_cv.predict(train_data_scale)
test_proba = clf_cv.predict_proba(test_data_scale)[:,1]
train_proba = clf_cv.predict_proba(train_data_scale)[:,1]

fpr_test, tpr_test, th_test = metrics.roc_curve(test_target, test_proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train_target, train_proba)

print('AUC = %6.4f' %metrics.auc(fpr_test, tpr_test)) 


# In[ ]:


# ROC Curve

plt.figure(figsize=[4, 4])
plt.plot(fpr_test, tpr_test, 'b-')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.text(0.4, 0.8, 'AUC = %6.4f' %metrics.auc(fpr_test, tpr_test), ha='center')
plt.show()


# In[ ]:


# Check if the logistic model is as good as trees. Can it distinguish the two class?

red, blue = sns.color_palette("Set1",2)

sns.kdeplot(test_proba[test_target==1], shade=True, color=red)
sns.kdeplot(test_proba[test_target==0], shade=True, color=blue)


# #### Looks like the logistic model doesn't perform as good as the trees.

# In[ ]:


# Compare different threshold and different precision/recall score.

for i in [0.25, 0.35, 0.5, 0.6, 0.75]:
    prediction = (test_proba > i).astype('int')
    confusion_matrix = pd.crosstab(test_target,prediction,
                                   margins = True)
    precision = confusion_matrix.iloc[1, 1] /confusion_matrix.loc['All', 1]
    recall = confusion_matrix.iloc[1, 1] / confusion_matrix.loc[1, 'All']
    Specificity = confusion_matrix.iloc[0, 0] /confusion_matrix.loc[0,'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold: %s, precision: %.2f, recall:%.2f ,Specificity:%.2f , f1_score:%.2f'%(i, precision, recall, Specificity,f1_score))


# ### We can set our own threshold to meet business needs.
# * If the hotel want to provide better service for those who are not likely to cancel their reservation but has little resource. We can set the threshold low, so that we only focus on the 'most likely to come' customers.
# * If the hotel want to 'oversell' the rooms but doesn't want to lose credibility, we can set the threshold high:0.75, so that the precison is 99%. It's unlikely to hurt a loyal customer.
# * If the hotel want to have a double check with the customers, but doesn't know who to call, we can have the 'most likely no-show' customer list, so our personnel can call them first.
# 
# #### We can utilise this model to different business scenarios instead of just output a Yes or No.

# ### To recap, we utilised this dataset to predict future booking cancelations. 
# During data manipulation, we discovered:
# * deposit_type
# * country
# * lead_time
# * market_segment
# * customer_type
# * is_room_changed
# * total_of_special_requests
# * required_car_parking_spaces
# * agent
# * booking_changes
# are the most relavent attributes to predict cancelation.
# 
# We visualised:
# * continuous data
# * categorical data
# * time seires
# * geo map
# * relation between attributes and Y
# 
# 
# We dealed with:
# * missing data
# * outlier
# * create new features
# * regroup categorical variables
# * hyperparameter tuning
# 
# And finnaly we use the model to make business decisions, by setting different threshold, we can have vaires of usage scenarios.
# 

# # End
# 
# ### Feel free to contact me if you have any questions.
# ### Please upvote and fork if you find this notebook useful! Many thanks
