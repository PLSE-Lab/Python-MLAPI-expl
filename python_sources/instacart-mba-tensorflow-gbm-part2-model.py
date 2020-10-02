#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

##changing directory
os.chdir('../input/')


# In[2]:


all_products = pd.read_csv('products.csv') #49688
all_orders = pd.read_csv('orders.csv')
prior_orders = pd.read_csv('order_products__prior.csv')
train_set = pd.read_csv('order_products__train.csv')


# In[3]:


prior_orders_extended = prior_orders.merge(all_orders,on='order_id',how='left')


# ## FEATURES

# In[4]:


prior_orders_extended.columns


# In[5]:


prior_orders_extended.head()


# ### CUSTOMER

# In[6]:


cust_attribs1 = prior_orders_extended.groupby('user_id').agg({'product_id':['count','nunique'],'order_id':'nunique','reordered':'sum'})
cust_attribs1.columns = cust_attribs1.columns.map('_'.join)
cust_attribs1 = cust_attribs1.reset_index().rename(columns=    {'product_id_count':'total_basket_size','product_id_nunique':'unique_basket_size','order_id_nunique':'user_num_prior_orders','reordered_sum':'reordered_basket_size'})
cust_attribs1.head()


# In[7]:


#cust_attribs2 = prior_orders_extended.groupby('user_id')['product_id'].apply(set).reset_index().rename(columns={'product_id':'products_list'})
#cust_attribs2.head()


# In[8]:


cust_attribs3 = all_orders.groupby('user_id').agg({'days_since_prior_order' : 'mean','order_id':'count'})
cust_attribs3 = cust_attribs3.reset_index().rename(columns=    {'days_since_prior_order':'user_avg_frequency','order_id':'user_num_orders'})
cust_attribs3.head()


# In[9]:


#cust_attribs = cust_attribs1.merge(cust_attribs2,how='left',on='user_id').\
#     merge(cust_attribs3,how='left',on='user_id')
#del cust_attribs1,cust_attribs2,cust_attribs3
cust_attribs = cust_attribs1.merge(cust_attribs3,how='left',on='user_id')
del cust_attribs1,cust_attribs3

cust_attribs['average_basket_size'] = cust_attribs['total_basket_size']/cust_attribs['user_num_prior_orders']
cust_attribs['user_repeatability'] = cust_attribs['reordered_basket_size']/cust_attribs['total_basket_size']
cust_attribs.head()


# ### CUST-PRODUCT

# In[10]:


t = prior_orders_extended.groupby(['user_id','product_id']).agg({'order_number':['max','count'],'add_to_cart_order':'mean','reordered':'sum'})
t.columns = t.columns.map('_'.join)
t = t.reset_index().rename(columns=    {'order_number_max':'user_prod_last_order_num','order_number_count':'user_prod_num_orders',     'add_to_cart_order_mean':'avg_prod_basket_position','reordered_sum':'user_prod_num_reorders'})
t.head()


# In[11]:


user_prod_attribs = prior_orders_extended[['user_id','product_id','order_id','order_number']].merge(t,how='inner',left_on=['user_id','product_id','order_number'],right_on=['user_id','product_id','user_prod_last_order_num'])
user_prod_attribs.head()


# User product frequency would make more sense than user average frequency, as quantity of product is not given. Eg. A users usual order quanity can 1 kg of Pulses which he orders once a week, another's might be 2 kg ordered once in two weeks. So this might be a valuable feature. //Work later on bringing this - requires quite a bit of tweaking as order dates are not given as timestamps

# In[12]:


user_prod_attribs.drop('order_number',axis=1,inplace=True)
user_prod_attribs.rename(columns={'order_id':'user_prod_last_order_id'},inplace=True)
user_prod_attribs['user_prod_repeatability'] = user_prod_attribs['user_prod_num_reorders']/     user_prod_attribs['user_prod_num_orders']


# ### PRODUCT

# In[13]:


prod_attribs = prior_orders_extended.groupby('product_id').agg({'reordered':['sum','count']})
prod_attribs.columns = prod_attribs.columns.map('_'.join)
prod_attribs = prod_attribs.reset_index().rename(columns=    {'reordered_sum':'prod_num_reorders','reordered_count':'prod_num_orders'})
prod_attribs['prod_repeatability'] = prod_attribs['prod_num_reorders']/     prod_attribs['prod_num_orders']
prod_attribs.head()


# ### PREPARING TRAINING DATA

# In[14]:


print(train_set.columns)
test_orders = all_orders[all_orders.eval_set == 'test']
train_orders = all_orders[all_orders.eval_set == 'train']
all_orders.head()


# In[15]:


print("Number of orders in train_set : ",train_set.order_id.nunique())
print("Number of orders in train_orders : ",train_orders.order_id.nunique())
print("Train set: ",train_set.shape)
print("Train orders: ",train_orders.shape)


# In[16]:


train_set.reordered.value_counts()


# The training data set given and test data set dervied from orders have different columns. We'll pick order_id, product_id, reordered or not from the training set given, map it to train tagged data from all_orders set and there on add features which we generated. A user-product can be taken as index, rest as features and reordered/not as label and model can be trained.

# In[17]:


train_df = train_set.merge(train_orders,how='left',on='order_id').    merge(prod_attribs,how='left',on='product_id').    merge(cust_attribs,how='left',on='user_id').    merge(all_products,how='left',on='product_id').    merge(user_prod_attribs,how='left',on=['user_id','product_id'])

print(train_orders.shape)
print(train_df.shape)
print(train_df.columns)
print("Number of orders in train_df : ",train_df.order_id.nunique())
print("Number of users in train_df : ",train_df.user_id.nunique())
print("Any NaNs in target var : ",sum(train_df.reordered != train_df.reordered))
train_df.set_index(['user_id','product_id'],inplace=True)
train_df.head()


# In[18]:


train_df.isnull().sum()


# The NaNs in these variables resulted from user_prod df meaning the user product combination in the training data is new. Verifying it. 
# For imputing these missing values - we can ignore the user_prod_last_order_id,user_prod_last_order_num as we are not using them anywhere now. For rest, 
# Since we've a single order per user, we can replace user_prod_num_orders with 1, avg basket position with the current basked position,user_prod_num_reorders and repeatability as 0
# 
# 9 rows have NaNs on product related columns, which means the product appeared for the first time. Dropping these rows. 

# In[19]:


train_df[train_df['user_prod_last_order_id'].isnull()].head(5)


# In[20]:


prior_orders_extended[(prior_orders_extended['user_id']==112108) & (prior_orders_extended['product_id']==10246)]


# In[21]:


train_df.user_prod_num_orders.fillna(1,inplace=True)
train_df.avg_prod_basket_position = train_df.add_to_cart_order.where(        train_df.avg_prod_basket_position.isnull(),train_df.avg_prod_basket_position)
train_df.user_prod_num_reorders.fillna(0,inplace=True)
train_df.user_prod_repeatability.fillna(0,inplace=True)
train_df.dropna(subset=['prod_num_orders'],inplace=True,axis=0)
train_df.isnull().sum()


# In[22]:


num_features = ['order_hour_of_day', 'days_since_prior_order', 'total_basket_size',
       'unique_basket_size', 'user_num_prior_orders', 'reordered_basket_size','user_avg_frequency', 'user_num_orders',
       'average_basket_size', 'user_repeatability',    'user_prod_num_orders', 'avg_prod_basket_position',
       'user_prod_num_reorders', 'user_prod_repeatability',
       'prod_num_reorders', 'prod_num_orders', 'prod_repeatability']
cat_features = ['aisle_id','department_id','order_dow']
tot_features = list(set(num_features)|set(cat_features))


# In[23]:


len(num_features)


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(nrows = 6, ncols = 3 ,figsize=(30,30))
flat_ax = [item for sublist in ax for item in sublist]
for i,feature in enumerate(num_features):
     sns.distplot(train_df[feature],ax=flat_ax[i])


# In[25]:


def get_normalization_parameters(traindf, features):
    """Get the normalization parameters (E.g., mean, std) for traindf for 
    features. We will use these parameters for training, eval, and serving."""

    def z_score_params(column):
        mean = np.float32(traindf[column].mean())
        std = np.float32(traindf[column].std())
        min = np.float32(traindf[column].min())
        max = np.float32(traindf[column].max())

        return {'mean': mean, 'std': std,'min':min,'max':max}

    normalization_parameters = {}
    for column in features:
        normalization_parameters[column] = z_score_params(column)
    return normalization_parameters
#normalization_parameters = get_normalization_parameters(x_train, num_features)
normalization_parameters = get_normalization_parameters(train_df, num_features)
print(normalization_parameters)


# In[26]:


num_features_s = []
for feature in num_features:
    train_df[feature+'_s'] = (train_df[feature]-normalization_parameters[feature]['min'])/(normalization_parameters[feature]['max']-normalization_parameters[feature]['min'])
    num_features_s.extend([feature+'_s'])
    print(feature)
tot_features_s = list(set(num_features_s)|set(cat_features))    


# In[27]:


from sklearn.model_selection import train_test_split
x_train, x_eval, y_train, y_eval = train_test_split(train_df[tot_features_s], train_df['reordered'], test_size=0.2)


# In[28]:


del train_df,all_orders,prior_orders_extended,prior_orders,train_set,all_products


# In[29]:


import tensorflow as tf

tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)


# In[30]:


fc = tf.feature_column
all_features = []

def std_scaler(feature,mean,std):
      return (feature - mean)/std

for feature_name in cat_features:
  # Need to one-hot encode categorical features.
    vocabulary = x_train[feature_name].unique()
    all_features.append(fc.indicator_column(
      fc.categorical_column_with_vocabulary_list(feature_name,vocabulary)))

for feature_name in num_features_s:
#     fet_mean = normalization_parameters[feature_name]['mean']
#     fet_std = normalization_parameters[feature_name]['std']
    all_features.append(fc.numeric_column(feature_name,dtype=tf.float32,           # normalizer_fn=lambda x:((x-fet_mean)/fet_std)\
                       ))
#Normalizing is creating a problem - check later


# In[31]:


print(len(all_features))
all_features


# In[32]:


example = x_train.head(1)


# In[33]:


fc.input_layer(dict(example), all_features)


# In[34]:


x_train.shape,x_eval.shape


# In[35]:


NUM_EXAMPLES = len(y_train) #1107693

def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
          dataset = dataset.shuffle(NUM_EXAMPLES)
        # For training, cycle thru dataset as many times as need (n_epochs=None).    
        dataset = dataset.repeat(n_epochs)  
        # In memory training doesn't use batching.
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn

train_input_fn = make_input_fn(x_train.head(800), y_train.head(800))
eval_input_fn = make_input_fn(x_eval.head(1000), y_eval.head(1000), shuffle=False, n_epochs=1)


# In[36]:


linear_est = tf.estimator.LinearClassifier(all_features)

# Train model.
linear_est.train(train_input_fn, max_steps=10)

# Evaluation.
results = linear_est.evaluate(eval_input_fn)
print('Accuracy : ', results['accuracy'])
print('Dummy model: ', results['accuracy_baseline'])


# In[ ]:


def populate_features(df,chosen_features):
    df_new = df.merge(cust_attribs,how='left',on='user_id').        merge(user_prod_attribs,how='left',on='user_id').        merge(prod_attribs,how='left',on='product_id').        merge(train_set,how='left',on=['order_id','product_id']).        merge(all_products,how='left',on='product_id')
    return df_new[chosen_features]

