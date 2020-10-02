#!/usr/bin/env python
# coding: utf-8

# The below lines are for automatic loading and reloading of the kernel

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's import some libraries. The fastai.imports only contains some imports. We can see this thing in the github repository. The other libraries we will import as required

# In[2]:


from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from IPython.display import display
from sklearn import metrics


# Lets specify a path first

# In[3]:


PATH = '../input/'


# In[4]:


df_raw = pd.read_csv(f'{PATH}Train/Train.csv', low_memory=False, parse_dates=['saledate'])


# We make a function for displaying the full dataframe below. For this we need pd.option_context and set its display.max_rows and display.max_columns to 1000

# In[5]:


def display_all(df):
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000):
        display(df)


# In[6]:


display_all(df_raw.tail().T)


# In[7]:


display_all(df_raw.describe(include='all').T)


# The evaluation metric is RMLSE. So we have to take the log of the dependent variable for better prediction

# In[8]:


df_raw.SalePrice = np.log(df_raw.SalePrice)


# As the dataframe has categorical variables so the classifier can't fit. We can't use categorical variables in the model. So we have to convert them to numeric. We will do that from the next cells.

# In[9]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)


# First we change the saledate variable as it is a date variable. For this we use add_datepart. It will make the date variable into various numerical columns

# In[10]:


add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()


# In[11]:


get_ipython().run_line_magic('pinfo2', 'train_cats')


# The train_cats change any columns of strings in a panda's dataframe to a column of categorical values. This applies the changes inplace

# In[12]:


train_cats(df_raw)


# In[13]:


df_raw.UsageBand.cat.categories


# In[14]:


df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)


# If we use .codes the categorical variables will be converted into numerical variables

# In[15]:


df_raw.UsageBand = df_raw.UsageBand.cat.codes


# In[16]:


display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# In[17]:


get_ipython().run_line_magic('pinfo2', 'proc_df')


# proc_df takes a data frame df and splits off the response variable, and changes the df into an entirely numeric dataframe

# In[18]:


df, y, nas = proc_df(df_raw, 'SalePrice')


# Lets make a simple model now. The n_jobs=-1 is used to take the advantage of multiple cores of the CPU.

# In[19]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df, y)


# We break the dataframe into train and variable set using split_vals

# In[20]:


def split_vals(a, n):
    return a[:n].copy(), a[n:].copy()
n_valid = 12000
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
X_train.shape, y_train.shape, X_valid.shape


# A function for rmse and printing the score

# In[21]:


def rmse(x, y):
    return math.sqrt(((x-y)**2).mean())
def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid), m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)


# In[22]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# We take a subset of 30000 for training set to make the model fast.

# In[23]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000, na_dict=nas)
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)


# In[24]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# Lets visualize the model with 1 tree. bootstrap=False is telling it to sample observations with or without replacement - it should still sample when it's False, just without replacement

# In[25]:


m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[26]:


draw_tree(m.estimators_[0], df_trn, precision=3)


# In[27]:


m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


#  ***BAGGING***
#  
#  Bagging is the combining of various bad estimators which turns into a good model.
# 

# Lets see the baseline model

# In[28]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# The np.stack joins a sequence of arrays produced by m.estimators_ along a new axis.

# In[29]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:, 0], np.mean(preds[:, 0]), y_valid[0]


# In[30]:


preds.shape


# In[31]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])


# In[32]:


m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[33]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[34]:


m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# ***OUT OF BAG(OOB) SCORE***

# Sometimes we dont have enough data for making validation sets. For that purpose we use the OOB score. What it does is that the trees which are not used in bagging are passed to the bagged model and its score is noted. It will give us a good enough score which is as close as the score of validation set

# In[35]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[36]:


df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)


# set_rf_samples(n) changes Scikit learn's random forests to give each tree a random sample of n random rows.

# In[44]:


set_rf_samples(20000)


# In[45]:


m = RandomForestRegressor(n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[46]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[47]:


reset_rf_samples()


# Now let's do some hyperparameter tuning. The various parameters in RandomForestRegressor are: 
# 1. n_estimators
# 2. min_samples_leaf
# 3. max_features

# n_estimators tells how many trees are we ensembling together

# In[48]:


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# min_samples_leaf is the minimum number of samples required to be at a leaf node. If leaf nodes are equal to this, the tree will not split for those nodes any further.

# In[49]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# max_features is the number of features to consider when looking for the best split. In this case, we have taken 0.5 meaning that half of the features will be used in the trees while splitting

# In[50]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:




