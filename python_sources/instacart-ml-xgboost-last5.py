#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For data manipulation
import pandas as pd              

# Garbage Collector to free up memory
import gc                         
gc.enable()                       # Activate 


# In[ ]:


orders = pd.read_csv('../input/orders.csv' )
order_products_train = pd.read_csv('../input/order_products__train.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
products = pd.read_csv('../input/products.csv')
aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')


# In[ ]:


orders.head()


# In[ ]:


order_products_train.head()


# In[ ]:


order_products_prior.head()


# In[ ]:


products.head()


# In[ ]:


aisles.head()


# In[ ]:


departments.head()


# In[ ]:


# We convert character variables into category. 
# In Python, a categorical variable is called category and has a fixed number of different values
aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')


# In[ ]:


#Create a DataFrame with the orders and the products that have been purchased on prior orders (op)
#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs
op = orders.merge(order_products_prior, on='order_id', how='inner')
op.head()


# # 2. Create Predictor Variables
# We are now ready to identify and calculate predictor variables based on the provided data. We can create various types of predictors such as:
# * <b>User predictors</b> describing the behavior of a user e.g. total number of orders of a user.
# * <b>Product predictors</b> describing characteristics of a product e.g. total number of times a product has been purchased.
# * <b>User & product predictors</b> describing the behavior of a user towards a specific product e.g. total times a user ordered a specific product.

# In[ ]:


#Number of orders per customer
# Create distinct groups for each user, identify the highest order number in each group, save the new column to a DataFrame
user = op.groupby('user_id')['order_number'].max().to_frame('u_total_orders')
# Reset the index of the DF so to bring user_id from index to column (pre-requisite for step 2.4)
user = user.reset_index()
user.head()


# In[ ]:


#How frequent a customer has reordered products
u_reorder = op.groupby('user_id')['reordered'].mean().to_frame('u_reordered_ratio')
u_reorder = u_reorder.reset_index()
u_reorder.head()


# In[ ]:


user = user.merge(u_reorder, on='user_id', how='left')

del u_reorder
gc.collect()

user.head()


# **Create product predictors**

# In[ ]:


#Number of purchases for each product
# Create distinct groups for each product, count the orders, save the result for each product to a new DataFrame  
prd = op.groupby('product_id')['order_id'].count().to_frame('p_total_purchases')
prd = prd.reset_index()
prd.head()


# **What is the probability for a product to be reordered**

# In[ ]:


#Remove products with less than 40 purchases
# the x on lambda function is a temporary variable which represents each group
# shape[0] on a DataFrame returns the number of rows
p_reorder = op.groupby('product_id').filter(lambda x: x.shape[0] >40)
p_reorder.head()


# In[ ]:


#Group products, calculate the mean of reorders
p_reorder = p_reorder.groupby('product_id')['reordered'].mean().to_frame('p_reorder_ratio')
p_reorder = p_reorder.reset_index()
p_reorder.head()


# In[ ]:


#Merge the prd DataFrame with reorder
prd = prd.merge(p_reorder, on='product_id', how='left')

#delete the reorder DataFrame
del p_reorder
gc.collect()

prd.head()


# In[ ]:


prd['p_reorder_ratio'] = prd['p_reorder_ratio'].fillna(value=0)
prd.head()


# In[ ]:


# Create distinct groups for each combination of user and product, count orders, save the result for each user X product to a new DataFrame 
uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought')
# Reset the index of the DF so to bring user_id & product_id rom indices to columns (pre-requisite for step 2.4)
uxp = uxp.reset_index()
uxp.head()


# In[ ]:


#last 5 orders
op['order_number_back'] = op.groupby('user_id')['order_number'].transform(max) - op.order_number +1 
op5 = op[op.order_number_back <= 5]
last_five = op5.groupby(['user_id','product_id'])[['order_id']].count()
last_five.columns = ['times_last5']
last_five['times_last5_ratio'] = last_five.times_last5 / 5
#last_five = last_five.drop(['times_last5_y','times_last5_ratio_y'])
#############
uxp = uxp.merge(last_five , on=['user_id', 'product_id'], how='left')
del [last_five]
gc.collect()
uxp.head()


# In[ ]:


uxp['times_last5'] = uxp['times_last5'].fillna(value=0)
uxp['times_last5_ratio'] = uxp['times_last5_ratio'].fillna(value=0)
uxp.head()


# **How frequently a customer bought a product after its first purchase**

# Calculating the numerator - How many times a customer bought a product? ('Times_Bought_N')

# In[ ]:


times = op5.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
times.head()


# **Calculating the denumerator**

# In[ ]:


#The total number of orders for each customer ('total_orders')
total_orders = op5.groupby('user_id')['order_number'].max().to_frame('total_orders')
total_orders.head()


# #### 2.3.2.2.b The order number where the customer bought a product for first time ('first_order_number')
# Where for first_order_number we .groupby( ) by both user_id & product_id. As we want to get the order when a product has been purchases for first time, we select the order_number column and we retrieve with .min( ) aggregation function, the earliest order.

# In[ ]:


first_order_no = op5.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')
first_order_no  = first_order_no.reset_index()
first_order_no.head()


# We merge the first order number with the total_orders DataFrame. As total_orders refers to all users, where first_order_no refers to unique combinations of user & product, we perform a right join
# 

# In[ ]:


span = pd.merge(total_orders, first_order_no, on='user_id', how='right')
span.head()


# #### 2.3.2.2.c For each product get the total orders placed since its first order ('Order_Range_D')
# The denominator now can be created with simple operations between the columns of results DataFrame:

# In[ ]:


# The +1 includes in the difference the first order were the product has been purchased
span['Order_Range_D'] = span.total_orders - span.first_order_number + 1
span.head()


# ### 2.3.2.3 Create the final ratio "uxp_order_ratio"
# #### 2.3.2.3.a Merge the DataFrames of numerator & denumerator
# 

# In[ ]:


uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')
uxp_ratio.head()


# ####  2.3.2.3.b  Perform the final division
# Now we divide theTimes_Bought_N by the Order_Range_D for each user and product.

# In[ ]:


uxp_ratio['uxp_order_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D
uxp_ratio.head()


# ####  2.3.2.3.c Keep the final feature
# Here we select to keep only the 'user_id', 'product_id' and the final feature 'uxp2_order_ratio'
# 

# In[ ]:


uxp_ratio = uxp_ratio.drop(['Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D'], axis=1)
uxp_ratio.head()


# In[ ]:


#Remove temporary DataFrames
del [times, first_order_no, span,op5]


# ### 2.3.2.4 Merge the final feature with uxp DataFrame
# 

# In[ ]:


uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')

del uxp_ratio
uxp.head()


# ## 2.4 Merge all features
# We now merge the DataFrames with the three types of predictors that we have created (i.e., for the users, the products and the combinations of users and products).
# 
# We will start from the **uxp** DataFrame and we will add the user and prd DataFrames. We do so because we want our final DataFrame (which will be called **data**) to have the following structure: 
# 
# <img style="float: left;" src="https://i.imgur.com/mI5BbFE.jpg" >
# 
# 
# 
# 
# 

# ### 2.4.1 Merge uxp with user DataFrame
# Here we select to perform a left join of uxp with user DataFrame based on matching key "user_id"
# 
# <img src="https://i.imgur.com/WlI84Ud.jpg" width="400">
# 
# Left join, ensures that the new DataFrame will have:
# - all the observations of the uxp (combination of user and products) DataFrame 
# - all the **matching** observations of user DataFrame with uxp based on matching key **"user_id"**
# 
# The new DataFrame as we have already mentioned, will be called **data**.

# In[ ]:


#Merge uxp features with the user features
#Store the results on a new DataFrame
data = uxp.merge(user, on='user_id', how='left')
data.head()


# ### 2.4.1 Merge data with prd DataFrame
# In this step we continue with our new DataFrame **data** and we perform a left join with prd DataFrame. The matching key here is the "product_id".
# <img src="https://i.imgur.com/Iak6nIz.jpg" width="400">
# 
# Left join, ensures that the new DataFrame will have:
# - all the observations of the data (features of userXproducts and users) DataFrame 
# - all the **matching** observations of prd DataFrame with data based on matching key **"product_id"**

# In[ ]:


#Merge uxp & user features (the new DataFrame) with prd features
data = data.merge(prd, on='product_id', how='left')
data.head()


# ### 2.4.2 Delete previous DataFrames

# The information from the DataFrames that we have created to store our features (op, user, prd, uxp) is now stored on **data**. 
# 
# As we won't use them anymore, we now delete them.

# In[ ]:


del op, user, prd, uxp
gc.collect()


# # 3. Create train and test DataFrames
# 

# In[ ]:


orders_last = orders[(orders.eval_set=='train') | (orders.eval_set=='test') ]


# In[ ]:


data = data.merge(orders_last, on='user_id', how='left')
data.head()


# In[ ]:


data_train = data[data.eval_set=='train']

data_train = data_train.merge(order_products_train, on=['product_id', 'order_id'], how='left' )

data_train = data_train.drop(['order_id','eval_set', 'add_to_cart_order'], axis=1)
data_train = data_train.fillna(0)
data_train.head()


# In[ ]:


data_test = data[data.eval_set=='test']
data_test = data_test.drop(['eval_set', 'order_id'], axis=1)
data_test = data_test.fillna(0)
data_test.head()


# In[ ]:


del data
del orders_last
gc.collect()


# In[ ]:


data_train = data_train.set_index(['user_id', 'product_id'])
data_test = data_test.set_index(['user_id', 'product_id'])


# In[ ]:


import xgboost
from sklearn.model_selection import train_test_split
data_train.loc[:, 'reordered'] = data_train.reordered.fillna(0)


# subsample
X_train, X_val, y_train, y_val = train_test_split(data_train.drop('reordered', axis=1), data_train.reordered,
                                                    test_size=0.2, random_state=42)

del data_train
gc.collect()

d_train = xgboost.DMatrix(X_train, y_train)
xgb_params = {
    "objective"         : "reg:logistic"
    ,"eval_metric"      : "logloss"
    ,"eta"              : 0.1
    ,"max_depth"        : 6
    ,"min_child_weight" :10
    ,"gamma"            :0.70
    ,"subsample"        :0.76
    ,"colsample_bytree" :0.95
    ,"alpha"            :2e-05
    ,"lambda"           :10
}

watchlist= [(d_train, "train")]
bst = xgboost.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)
xgboost.plot_importance(bst)


# In[ ]:


del [X_train, X_val, y_train, y_val]
gc.collect()


# In[ ]:


d_test = xgboost.DMatrix(data_test)

data_test = data_test.reset_index()
data_test = data_test[['product_id', 'user_id']]

data_test["reordered"] = (bst.predict(d_test) > 0.21).astype(int)

del bst


# In[ ]:


orders_test = orders[orders.eval_set=='test']


# In[ ]:


data_test = data_test.merge(orders_test[["user_id", "order_id"]], on='user_id', how='left')
data_test.head()


# In[ ]:


del orders
del orders_test
gc.collect()


# In[ ]:


data_test['product_id'] = data_test.product_id.astype(int)
data_test = data_test.drop('user_id', axis=1)
gc.collect()


# In[ ]:


d = dict()
for row in data_test.itertuples():
    if row.reordered == 1:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

for order in data_test.order_id:
    if order not in d:
        d[order] = 'None'
        
gc.collect()


# In[ ]:


sub = pd.DataFrame.from_dict(d, orient='index')
sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']

sub.to_csv('sub.csv', index=False)


# In[ ]:


submission = pd.read_csv("../working/sub.csv")
submission.head()


# In[ ]:


submission.shape[0]

