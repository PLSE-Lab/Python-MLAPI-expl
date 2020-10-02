#!/usr/bin/env python
# coding: utf-8

# ### The objectives of this notebook are threefold:
# ##### 1. To explore large data by storing it in disk (using sqlite) 
# ##### 2. To Visualizing large data using Seaborn 
# ##### 3. To attempt multiple techniques (market basket analysis, collaborative filtering, poisson regression etc.)
# 
# ### Notebook is organized in Four Sections
# ##### Section 1: I will setup the environment. 
# ##### Section 2: I will import the data from  CSV to sqlite on disk.
# ##### Section 3: I will start with our basic data exploration
# ##### Section 4: I will attempt machine learning algorithms
# 
# Credits:
# Inspired from 3 Kernels and below website
# https://plot.ly/python/big-data-analytics-with-pandas-and-sqlite/

# **Section 1**
# -------------

# In[16]:


import pandas as pd
import numpy as np
import os
import warnings
from math import pi
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas.io.sql as pd_sql
from sqlalchemy import create_engine
import sqlite3 as sql
from datetime import datetime
sns.set(color_codes=True)
warnings.filterwarnings('ignore') # silence annoying warningsn


# In[17]:


### setup the current working directory 
import os
path = "/"
app_input="/kaggle/input"
app_output="../"
# Check current working directory.
print (os.getcwd())
print("\n")
#os.remove('kaggle_instakart.db')
# Now change the directory
os.chdir( app_input )

# Check current working directory.
print ("Directory changed successfully \n")
print (os.getcwd())

#print the list of csv files in the input folder
from subprocess import check_output
print ("\n")
print(check_output(["ls", "/kaggle/input"]).decode("utf8"))
os.chdir( "/kaggle/working" )
conn = create_engine('sqlite:///kaggle_instakart.db')


# **Section 2**
# -------------

# # I have tried 3 different approaches to read large csv files to SQLite Database
# ## Approach 1: pd.read_csv and pd.to_sql for complete files
# #### this approach failed with memory error on pd.to_sql().
# 
# aisles = pd.read_csv('../input/aisles.csv', engine='c')
# 
# full_sqllite_path = app_input + "/kaggle_instakart.db"
# 
# aisles.to_sql('aisles',conn,if_exists = 'replace',index=False )

# ## Approach 2: pd.read_csv and pd.to_sql chunk by chunk
# #### Created a function read_large_csv_to_sqlite for this approach

# In[18]:


def read_large_csv_to_sqlite(filename,disk_engine, tablename):
    start = datetime.now()
    chunksize = 20000
    j = 0
    index_start = 1
    drop_table="DROP TABLE IF EXISTS %s ;" %(tablename)
    conn.execute(drop_table)        
    
    for df in pd.read_csv(filename, chunksize=chunksize, iterator=True, encoding='utf-8'):
        df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) # Remove spaces from columns
        df.index += index_start
        j+=1
        if j<= 50:
            df.to_sql(tablename, disk_engine, if_exists='append')
            if j%10 == 0:
                print ('{} Seconds: Completed rows {}'.format((datetime.now()-start).total_seconds(),j*chunksize))
            index_start = df.index[-1] + 1
        else:
            print("Limiting Data for exploratory analysis")
            break
    
    # Created indexes on all id columns. this is a life saver
    index_columns = [col for col in df.columns if col.find("_id")>-1]
    for col in index_columns:
        create_indexes="CREATE INDEX index_%s on %s (%s);" %(col+"_"+tablename,tablename,col)
        conn.execute(create_indexes)        


# In[19]:


# Read CSV into SQLite database
list_files1 = [file  for file in os.listdir(app_input) if file.find(".csv") != -1 ]
print (list_files1)
for file in list_files1:    
    tablename = file.split(".")[0]
    print('\n Started Processing Table {}'.format(tablename))
    filename = app_input + "/"+file
    #print(filename)
    read_large_csv_to_sqlite(filename,conn,tablename)
    print (' Processing of Table {} ended'.format(tablename))


# ## Approach 3: read csv using native csv reader and write to sqlite in batches 
# #### pd.read_csv fails when the csv file size was 2 GB or greater
# #### code for this approach is beyond the scope of current project. Therefore, skipping it.

# **Section 3**
# --

# In[20]:


"""The table name is self-explanatory"""
# Explore Aisles
aisles = pd.read_sql_query("SELECT * FROM aisles LIMIT 5 ;", conn)
print('Total aisles: {}'.format(aisles.shape[0]))
print("Aisles Table\n",aisles.head(),"\n\n")


# Explore Departments
departments = pd.read_sql_query("SELECT * FROM departments LIMIT 5 ;", conn)
print('Total departments: {}'.format(departments.shape[0]))
print('Departments Table\n',departments.head(),"\n\n")

# Explore Products csv
products = pd.read_sql_query("SELECT * FROM products LIMIT 5 ;", conn)
print('Total products: {}'.format(products.shape[0]))
print("Products Table\n",products.head(),"\n\n")

# Explore orders csv
orders = pd.read_sql_query("SELECT * FROM orders LIMIT 5 ;", conn)
print('Total orders: {}'.format(orders.shape[0]))
""" This table contains the order and user linkage. Also, this table contains the day of week, 
  hour_of_day kind of fields"""
print("Orders Table\n",orders.head(5),"\n\n")

# Explore orders_train csv
orders_products_train = pd.read_sql_query("SELECT * FROM order_products__train LIMIT 5 ;", conn)
print('Total orders for training: {}'.format(orders_products_train.shape[0]))
""" This table contains linkage between product table and the orders table. 
  Also, this table will be usefule for prediction for reorder""" 
print("Orders_products_train\n",orders_products_train.head(5),"\n\n")

# Explore order_product_prior.csv
orders_products_prior = pd.read_sql_query("SELECT * FROM order_products__prior LIMIT 5;", conn)
""" This table contains linkage between product table and the orders table. 
  Also, this table will be usefule for prediction for reorder""" 
print('Total prior orders: {}'.format(orders_products_prior.shape[0]))
print("orders_products_prior\n",orders_products_prior.head(5),"\n\n")

del aisles,departments,products,orders,orders_products_train,orders_products_prior


# In[21]:


# Combine aisles, departments and products to create goods table
"""The purpose of creating goods table is to analyze combinations of aisle, department and products"""
start = datetime.now()
drop_goods_table=""" DROP TABLE IF EXISTS goods;"""
conn.execute(drop_goods_table)
join_prod_dep_sql = """    CREATE TABLE goods AS
    SELECT p.*, d.department, a.aisle
    FROM products p
    INNER JOIN departments d ON p.department_id = d.department_id
    INNER JOIN aisles a ON p.aisle_id = a.aisle_id;
    """
conn.execute(join_prod_dep_sql)
goods = pd.read_sql_query("SELECT * FROM goods Limit 5;", conn)
goods_step_time = datetime.now()
print('\nTotal Time taken to delete and create goods table {}\n'.format((goods_step_time-start).total_seconds()))
print("Goods Table \n", goods.head(),"\n\n")


#Combine orders and the orders_prior dataframe
#creating indexes earlier reduced time from 8 minutes to 3 minutes
drop_orders_combined_table=""" DROP TABLE IF EXISTS orders_combined;"""
conn.execute(drop_orders_combined_table)
join_ordProdPrior_sql = """    CREATE TABLE orders_combined AS
    SELECT o.*, op.product_id, op.add_to_cart_order,op.reordered
    FROM orders o
    INNER JOIN order_products__prior op ON o.order_id = op.order_id;
    """
conn.execute(join_ordProdPrior_sql)
orders_combined = pd.read_sql_query("SELECT * FROM orders_combined Limit 5;", conn)

index_columns = [col for col in orders_combined.columns if col.find("_id")>-1]
for col in index_columns:
    create_indexes="CREATE INDEX index_%s on orders_combined (%s);" %(col+"_orders_combined",col)
    conn.execute(create_indexes)        
        
orders_combined_time = datetime.now()
print('\nTotal Time taken to delete and create orders_combined table {}\n'.format((orders_combined_time-goods_step_time).total_seconds()))
print("Orders_Combined Table \n", orders_combined.head(),"\n\n")


#create datamart with combined tables
drop_prior_datamart_table=""" DROP TABLE IF EXISTS prior_datamart;"""
conn.execute(drop_prior_datamart_table)
join_prior_datamart_sql = """    CREATE TABLE prior_datamart AS
    SELECT o.*, gd.product_id, gd.product_name,gd.department,gd.aisle
    FROM orders_combined o
    INNER JOIN goods gd
    ON o.product_id = gd.product_id;
    """
conn.execute(join_prior_datamart_sql)
prior_datamart = pd.read_sql_query("SELECT * FROM prior_datamart Limit 5;", conn)

index_columns = [col for col in orders_combined.columns if col.find("_id")>-1]
for col in index_columns:
    create_indexes="CREATE INDEX index_%s on prior_datamart (%s);" %(col+"_prior_datamart",col)
    conn.execute(create_indexes)        
        
prior_datamart_time = datetime.now()
print('\nTotal Time taken to delete and create prior_datamart table {}\n'.format((prior_datamart_time-orders_combined_time).total_seconds()))
print("prior_datamart Table \n", prior_datamart.head(),"\n\n")

del goods, orders_combined,prior_datamart


# In[26]:


"""
Univariate Analysis
1. When do people order (Distribution of Time of Day) ?
2. Day of Week (Distribution of day_of_week)?
3. When do they order again (Distribution of Time Since Prior Order)?
4. How many prior orders are there (Distribution of Reorders)?
"""

#read the data from sqlite database to dataframe
orders = pd.read_sql_query("SELECT order_id, order_dow,days_since_prior_order,order_hour_of_day                            FROM orders;", conn)
goods = pd.read_sql_query("SELECT product_id,product_name,department,aisle FROM goods;", conn)
prior_datamart =  pd.read_sql_query("SELECT order_id, order_dow,days_since_prior_order,                                    order_hour_of_day,                                    product_id, product_name,department, aisle,reordered                                    FROM prior_datamart;", conn)
                                     

temp_df02 = pd.DataFrame(prior_datamart.groupby(['product_name']).agg({'order_id':pd.Series.nunique})
                         .rename(columns={'order_id':'cnt_ord_by_prod'})).reset_index()
temp_df03 = pd.DataFrame(prior_datamart.groupby(['department']).agg({'order_id':pd.Series.nunique})
                         .rename(columns={'order_id':'cnt_ord_by_dep'})).reset_index()
temp_df04 = pd.DataFrame(prior_datamart.groupby(['aisle']).agg({'order_id':pd.Series.nunique})
                         .rename(columns={'order_id':'cnt_ord_by_aisle'})).reset_index()

top_10_products = temp_df02.nlargest(10,'cnt_ord_by_prod')['product_name']
top_10_departments = temp_df03.nlargest(10,'cnt_ord_by_dep')['department']
top_10_aisle = temp_df04.nlargest(10,'cnt_ord_by_aisle')['aisle']

temp_df05= prior_datamart[prior_datamart['product_name'].isin(top_10_products)]
temp_df06= prior_datamart[prior_datamart['department'].isin(top_10_departments)]
temp_df07= prior_datamart[prior_datamart['aisle'].isin(top_10_aisle)]


days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
orders['day_of_week']=pd.Series([days[dow] for dow in orders['order_dow']]) 
prior_datamart['day_of_week']=pd.Series([days[dow] for dow in prior_datamart['order_dow']]) 

plt.close('all')
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax7)) = plt.subplots(nrows=4,ncols=2)
fig.set_size_inches(30,36)
ax1=sns.countplot(x="day_of_week", ax=ax1,  data=orders,palette="Blues_d")        
ax2=sns.countplot(x="days_since_prior_order",ax=ax2,   data=orders,palette="Reds_d")
ax3=sns.countplot(x="order_hour_of_day",ax=ax3,   data=orders,palette="Greens_d")
ax4=sns.countplot(x="reordered",ax=ax4, data=prior_datamart,palette="Blues_d")
ax5=sns.countplot(x="product_name",ax=ax5, data=temp_df05,palette="Blues_d")
ax6=sns.countplot(x="department",ax=ax6, data=temp_df06,palette="Blues_d")
ax7=sns.countplot(x="aisle",ax=ax7, data=temp_df07,palette="Blues_d")
fig.tight_layout(h_pad=4,w_pad=4,pad=4)

del temp_df02,temp_df03,temp_df04,temp_df05,temp_df06,temp_df07


# In[27]:


"""
Bivariate Analysis
1. When do people order (Distribution of Time of Day by orders) ?
2. Day of Week (Distribution of day_of_week by orders)?
"""
temp_df_01 = pd.DataFrame(orders.groupby(['day_of_week','order_hour_of_day'])
                          .agg({'order_id':pd.Series.nunique})
                          .rename(columns={'order_id':'count_of_orders'})).reset_index()
plt.close('all')
sns.factorplot(x="order_hour_of_day", y="count_of_orders",
               col="day_of_week", data=temp_df_01, kind="swarm",col_wrap=3,size=5);

del temp_df_01


# In[ ]:


"""
Bivariate Analysis
Mondays and Tuesdays are busy days. It will be interesting to look for type of products ordered on Monday and Tuesdays
during the peak time
"""
temp_groupby_01= prior_datamart.groupby(['day_of_week','order_hour_of_day','product_name',
                                         'reordered']).agg({'order_id':
                                                            pd.Series.nunique}).rename(columns={'order_id':
                                                                                                'count_of_reorders'})
temp_groupby_02 = temp_groupby_01['count_of_reorders'].groupby(level=0, group_keys=False)

temp_df_01 = pd.DataFrame(temp_groupby_01).reset_index()
temp_df_02 = pd.DataFrame(temp_groupby_02.nlargest(50)).reset_index()

top_10_products_by_DayAndTime = temp_df_02[temp_df_02['reordered']==1]
print("\n Top 10 products for Day and Time Combination\n",top_10_products_by_DayAndTime.head(5))

#Limit data to only those products which are in top 10 category by any time and day combination
temp_df_01 = temp_df_01[temp_df_01['product_name'].isin(top_10_products_by_DayAndTime['product_name'])]

#Monday and Tuesday are of interest
#temp_df_01 = temp_df_01[temp_df_01['day_of_week'].isin(['Monday','Tuesday'])] 

#Most orders are between 6 and 20
#temp_df_01 = temp_df_01[(temp_df_01['order_hour_of_day'] >= 6) & (temp_df_01['order_hour_of_day'] <= 20) ]

plt.close('all')
g = sns.factorplot(x='product_name', y='count_of_reorders',
                   #col="day_of_week", data=temp_df_01[temp_df_01['reordered']==1], kind="swarm",col_wrap=3,size=5);
                   col="day_of_week", data=temp_df_01, kind="swarm",col_wrap=3,size=5);
g.set_xticklabels(rotation=90)


# In[ ]:


"""
Bivariate Analysis
1. When do people order (Distribution of Time of Day by orders) ?
2. Day of Week (Distribution of day_of_week by orders)?
"""
temp_df_01 = pd.DataFrame(prior_datamart.groupby(['day_of_week','order_hour_of_day','reordered'])
                          .agg({'order_id':pd.Series.nunique})
                          .rename(columns={'order_id':'count_of_reorders'})).reset_index()
plt.close('all')
sns.factorplot(x='order_hour_of_day', y='count_of_reorders',
               col="day_of_week", data=temp_df_01[temp_df_01['reordered']==1], kind="swarm",col_wrap=3,size=5);

del temp_df_01


# In[ ]:


"""
Bivariate Analysis
Mondays and Tuesdays are busy days. It will be interesting to look for type of products ordered on Monday and Tuesdays
during the peak time
"""
temp_groupby_01= prior_datamart.groupby(['day_of_week','order_hour_of_day','product_name',
                                         'reordered']).agg({'order_id':
                                                            pd.Series.nunique}).rename(columns={'order_id':
                                                                                                'count_of_reorders'})
temp_groupby_02 = temp_groupby_01['count_of_reorders'].groupby(level=0, group_keys=False)

temp_df_01 = pd.DataFrame(temp_groupby_01).reset_index()
temp_df_02 = pd.DataFrame(temp_groupby_02.nlargest(50)).reset_index()

top_10_products_by_DayAndTime = temp_df_02[temp_df_02['reordered']==1]
print("\n Top 10 products for Day and Time Combination\n",top_10_products_by_DayAndTime.head(5))

#Limit data to only those products which are in top 10 category by any time and day combination
temp_df_01 = temp_df_01[temp_df_01['product_name'].isin(top_10_products_by_DayAndTime['product_name'])]

#Monday and Tuesday are of interest
#temp_df_01 = temp_df_01[temp_df_01['day_of_week'].isin(['Monday','Tuesday'])] 

#Most orders are between 6 and 20
#temp_df_01 = temp_df_01[(temp_df_01['order_hour_of_day'] >= 6) & (temp_df_01['order_hour_of_day'] <= 20) ]

plt.close('all')
g = sns.factorplot(x='product_name', y='count_of_reorders',
                   #col="day_of_week", data=temp_df_01[temp_df_01['reordered']==1], kind="swarm",col_wrap=3,size=5);
                   col="day_of_week", data=temp_df_01, kind="swarm",col_wrap=3,size=5);
g.set_xticklabels(rotation=90)

