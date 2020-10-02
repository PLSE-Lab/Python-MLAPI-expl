#!/usr/bin/env python
# coding: utf-8

# # Mercari Price Suggestion Challenge Data Preparation
# 
# This notebook is for initial preprocessing of data and creating custom sub datasets and train/test sets.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.imports import *
from fastai.column_data import *
from fastai.structured import *

from scipy import stats

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


DATA_PATH = Path('../input/')


# ### Functions

# In[ ]:


def split_df(df, test_mask):
    df_train, df_test = df[~test_mask], df[test_mask]
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    return df_train, df_test

# na category names are just replaced with 'missing'
def split_cat(text):
    try:
        return text.split('/')
    except AttributeError:
        return tuple(['missing'] * 3)

# replace na or no description values with 'missing'
def fix_desc(text):
    return 'missing' if not isinstance(text, str) or text == 'No description yet' else text


# In[ ]:


split_cat('Men/Coats & Jackets/Flight/Bomber')


# In[ ]:


tuple(['missing'] * 3)


# ### Fixup Raw Data

# In[ ]:


train = pd.read_csv(DATA_PATH/'train.tsv', sep='\t')
test = pd.read_csv(DATA_PATH/'test_stg2.tsv', sep='\t')


# In[ ]:


test.rename(columns={'test_id': 'train_id'}, inplace=True)


# There are up to four categories that can be seperated from this string but I decided only to consider the first three.

# In[ ]:


train.category_name.str.count('/').max()


# In[ ]:


train[train.category_name.str.count('/') == 3].category_name.unique()


# Remove prices below `$3` as Merari does not allow postings below `$3` which makes it an error. There are `874` entries like that. Mercari also does not allow prices above `$2,000` but there are only 3 entries like that with only a few dollars more which are likely shipping fees. So removing them is not required.

# In[ ]:


train = train.drop(train[train['price'] < 3].index)


# Extract subcategories from the main `category_name` and remove it after as we don't need it.

# In[ ]:


train['main_cat'], train['sub_cat1'], train['sub_cat2'] = zip(*train['category_name'].apply(split_cat))                                                              
test['main_cat'], test['sub_cat1'], test['sub_cat2'] = zip(*test['category_name'].apply(split_cat))

train.drop('category_name', inplace=True, axis=1)
test.drop('category_name', inplace=True, axis=1)


# Replace `na` values in `brand_name` column with `missing`.

# In[ ]:


train['brand_name'].fillna(value='missing', inplace=True)
test['brand_name'].fillna(value='missing', inplace=True)


# The `name` column has nothing missing, but this is added just in case.

# In[ ]:


train['name'].fillna(value='missing', inplace=True)
test['name'].fillna(value='missing', inplace=True)


# Convert `item_condition_id` and `shipping` column to `str` for easy conversion using FastAI's `proc_df`.

# In[ ]:


train['shipping'] = train['shipping'].astype('str')
test['shipping'] = test['shipping'].astype('str')

train['item_condition_id'] = train['item_condition_id'].astype('str')
test['item_condition_id'] = test['item_condition_id'].astype('str')


# Replace `na` values and `No description yet` values in `item_description` with `missing`.

# In[ ]:


train['item_description'] = train['item_description'].apply(fix_desc)
test['item_description'] = test['item_description'].apply(fix_desc)


# Combine `name` and `item_description` into one field where the name and description are separated by a newline.

# In[ ]:


train['full_desc'] = train['name'].str.cat(train['item_description'], sep='\n')
test['full_desc'] = test['name'].str.cat(test['item_description'], sep='\n')


# Drop these two columns since they are no longer needed.

# In[ ]:


train.drop('name', axis=1, inplace=True)
train.drop('item_description', axis=1, inplace=True)

test.drop('name', axis=1, inplace=True)
test.drop('item_description', axis=1, inplace=True)


# ### ! Replace training sets `price` column with its `np.log1p` !

# In[ ]:


train['price'] = np.log1p(train['price'])


# This is done so that the values for index and `train_id` are not the same and that index reflects the true length of the dataframe such that the last index is of the value `len(df)-1`

# In[ ]:


train.reset_index(inplace=True, drop=True)


# In[ ]:


train.columns


# In[ ]:


print(train['full_desc'][np.random.randint(0, len(train))])


# ### Create custom dataset from only the struct columns of the dataset

# For now, use only the columns of the dataset, in addition the datasets contain `train_id` and `price`.

# ### Extract and create the sub-datasets

# In[ ]:


dep = ['price']
rid = ['train_id']
struct_vars = ['item_condition_id', 'brand_name', 'shipping', 'main_cat', 'sub_cat1', 'sub_cat2']


# In[ ]:


test.columns


# In[ ]:


for s in struct_vars: print (len(train[s].unique()))


# In[ ]:


price = train[dep].as_matrix().flatten()
train = train[rid + struct_vars + dep]
test =  test[rid +  struct_vars]


# ### Split in training and test sets

# The idea is to have a 10% data for test and 90% for train (and validation). The datasets are created as follows:
# 1. Get a random test mask of length 10% of the total training data
# 2. Extract the dependent variables for train and test using the mask
# 3. Extract train and test for each of the datasets

# In[ ]:


test_mask = train.index.isin(get_cv_idxs(n = len(train), val_pct=0.1))
y_test = price[test_mask]


# In[ ]:


my_train, my_test = split_df(train, test_mask)
my_test.drop('price', axis=1, inplace=True)


# # Mercari Price Suggestion Challenge Structured Data

# ## Introduction

# In this experiment, I consider all variables except `name` and `item_description` as part of the training features and label them as categorical (structured) data and create entity embeddings for them. This is part of the abalation study of how discarding `name` and `item_description` variables affects performance.

# ## My Definitions

# ### Functions

# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
        
def rmsle(y_pred, targ):
    '''Root Mean Squared Logarithmic Error'''
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(targ))**2))
        
def RMSE(preds, targs):
    assert(len(preds) == len(targs))
    return np.sqrt(mean_squared_error(targs, preds))    


# Total number of epochs formula from [here](http://forums.fast.ai/t/n-epochs-cycle-len-cycle-mult-n-cycles/19106)
# 
# $n\_epochs = cycle\_len \cdot (1 + cycle\_mult + cycle\_mult^{2} + \cdots + cycle\_mult^{(n_{cycles}-1)})$

# In[ ]:


def get_epochs(n_cycle, cycle_len, cycle_mult):
    n_epochs = 0
    for cycle in range(n_cycle):
        n_epochs += cycle_mult ** cycle
    
    return cycle_len * n_epochs


# ## Load data and get validation indices

# In[ ]:


X_train = train.copy()
X_test = test.copy()


# In[ ]:


X_train.set_index('train_id', inplace=True)
X_test.set_index('train_id', inplace=True)


# #### Change any columns of strings in a panda's dataframe to a column of categorical values. Apply changes inplace.

# In[ ]:


train_cats(X_train) 
apply_cats(X_test, X_train)


# In[ ]:


df_train, y_train, nas = proc_df(X_train, 'price')
df_test, _, nas = proc_df(X_test, na_dict=nas)


# In[ ]:


val_idxs = get_cv_idxs(len(df_train), val_pct=0.15, seed=None)
y_range = (0, np.max(y_train) * 1.5)


# ## DL Model

# ### Experimenting

# In[ ]:


cat_vars = ['item_condition_id', 'brand_name', 'shipping', 'main_cat', 'sub_cat1', 'sub_cat2']

cat_sz = [(c, len(X_train[c].cat.categories)+1) for c in cat_vars]


# In[ ]:


emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
print (emb_szs)


# In[ ]:


PATH = '../working/'


# In[ ]:


md = ColumnarModelData.from_data_frame(PATH,
                                       val_idxs, 
                                       df_train,
                                       y_train.astype(np.float32),
                                       cat_flds=cat_vars,
                                       bs=128, 
                                       test_df=df_test)


# In[ ]:


m = md.get_learner(emb_szs,
                   n_cont=0,
                   emb_drop=0.04,
                   out_sz=1,
                   szs=[1000, 500],
                   drops=[0.001, 0.01],
                   y_range=y_range)


# In[ ]:


# %%time
# m.lr_find()


# In[ ]:


# m.sched.plot(1000)


# In[ ]:


lr=1e-3


# In[ ]:


# bk = PlotDLTraining(m)


# In[ ]:


get_ipython().run_cell_magic('time', '', "m.fit(lr, n_cycle=4, metrics=[RMSE], best_save_name='mercari_best')")


# In[ ]:


x,y=m.predict_with_targs()


# In[ ]:


RMSE(x,y)


# In[ ]:


pred_test=m.predict(is_test=True)


# In[ ]:


submission = pd.DataFrame(np.exp(pred_test)).reset_index()


# In[ ]:


submission.columns = ['test_id', 'price']


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




