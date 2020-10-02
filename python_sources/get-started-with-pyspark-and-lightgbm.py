#!/usr/bin/env python
# coding: utf-8

# # 0. Setup
# 
# 1. `pip install lightgbm pyspark pandas` and install JDK(>= 1.8).
# 2. Put your data into folder `data` .
# 3. Start a jupyter notebook and run the following code.

# # 1. Data Warehousing

# ```python
# import pandas as pd
# import numpy as np
# 
# from pyspark.sql import SparkSession
# from pyspark.sql.types import *
# from pyspark import StorageLevel
# 
# from IPython.core.magic import register_line_cell_magic
# from string import Template
# from os import path
# 
# 
# replaces = {}  # replacable variables in SQL
# print_sql = False # whether to debug when running SQL in notebook
# 
# 
# ctx = SparkSession.builder.appName("MDD Cup 2018").config("spark.driver.memory", "6G").master('local[7]').getOrCreate()
# 
# 
# 
# 
# def table(name):
#     return ctx.read.table(name)
# 
# def saveTable(tb, name):
#     dataFile = "data/" + name
#     tb.write.parquet(dataFile, mode = 'overwrite')
# 
# @register_line_cell_magic
# def sql_as(line, cell=None, format='parquet'):
#     val = cell if cell else line 
#     sp = line.split()
#     tb_name = sp[0]
#     load_prev_data = False
#     if len(sp) > 1 and sp[1].lower() in ['true', '1']:
#         load_prev_data = True
#     dataFile = "data/"+tb_name
#     if path.exists(dataFile) and load_prev_data:
#         return ctx.read.parquet(dataFile).persist(StorageLevel.MEMORY_AND_DISK).createOrReplaceTempView(tb_name)
#     else:
#         rp = Template(val).substitute(replaces)
#         if print_sql:
#             print(rp)
#         # persist to speed up
#         tb = ctx.sql(rp).persist(StorageLevel.MEMORY_AND_DISK)
#         tb.createOrReplaceTempView(tb_name)
#         return tb.write.parquet(dataFile, mode = 'overwrite') 
# 
# from pyspark.sql.functions import lit
# 
# def loadData(table_name):
#     tb = ctx.read.option("delimiter", "\t").csv("data/{}.txt".format(table_name), header = True, inferSchema = True, nullValue = 'NULL')
#     tb.createOrReplaceTempView(table_name)
#     return tb
# 
# deal_poi_test = loadData('deal_poi_test').withColumn('sales', lit(None))
# deal_sales = loadData('deal_sales_train').union(deal_poi_test)
# poi_info = loadData('poi_info')
# merchant_recommended_dish = loadData('merchant_recommended_dish')
# user_recommended_dish = loadData('user_recommended_dish')
# deals = loadData("deal_train").union(loadData("deal_test"))
# dishes = loadData("dish_train").union(loadData("dish_test"))
# poi_deal_pv = loadData("poi_deal_pv_train").union(loadData("poi_deal_pv_test"))
# 
# deals.createOrReplaceTempView("deals")
# dishes.createOrReplaceTempView("dishes")
# deal_sales.createOrReplaceTempView("deal_sales")
# poi_deal_pv.createOrReplaceTempView("poi_deal_pv")
# 
# 
# cols = []
# tables = [deal_sales, poi_info, deals, dishes, poi_deal_pv]
# dup_cols = []
# # remove duplicate columns
# for t in tables:
#     for c in t.columns:
#         if c not in dup_cols:
#             if c in cols:
#                 dup_cols.append(c)
#                 cols.remove(c)
#             else:
#                 cols.append(c)
#     
# replaces['cols'] = ','.join(cols)
# ```

# Run the following code in a separate cell.

# ```sql
# %%sql_as wide_table 1
# 
# select l.deal_id,
#        l.poi_id,
#         
#        (hash(poi_rank) % 10) + 10 as poi_rank_hash,
#        price/market_price discount, 
#        price/deal_avg_num price_per_person,
#         
#         $cols
#   from deal_sales l
# 
#   left join deals d
#     on d.deal_id = l.deal_id
#   left join poi_info p
#     on p.poi_id = l.poi_id
#   
#   left join dishes ds
#     on ds.deal_id= l.deal_id
# 
#   left join poi_deal_pv pv 
#     on pv.deal_id = l.deal_id 
#     and pv.poi_id = l.poi_id
#     and datediff(begin_date, partition_date)  = 1
# 
# ```

# # 2. Denoising
# 

# # 3. EDA & Feature Enginnering
# 
# 

# # 4. Trainning Data Preparing
# 

# ```python
# wide_table = table('wide_table')
# 
# exclude_cols = 'sales begin_date partition_date day_unavailable time_available weekday_unavailable dish_tag poi_rank'.split()
# label_key = 'sales'
# features = np.setdiff1d(wide_table.columns, [exclude_cols]).tolist()
# 
# test_data = wide_table.where('sales is null')
# train_data, val_data = wide_table.where('sales is not null').randomSplit([0.9, 0.1], seed=1)
# 
# x_train = np.array(train_data[features].collect())
# y_train = np.array(train_data.select(label_key).collect()).flatten()
# 
# x_val = np.array(val_data[features].collect())
# y_val = np.array(val_data.select(label_key).collect()).flatten()
# 
# x_test = np.array(test_data[features].collect())
# 
# ctx.catalog.clearCache()
# ```

# # 5. Modeling

# ```python
# pred_output = {
#     'deal_id':np.array(test_data[['deal_id']].collect()).flatten(), 
#     'poi_id':np.array(test_data[['poi_id']].collect()).flatten(), 
#     }
# 
# def save_csv(sales, file_name):
#     pred_output['sales'] = sales
#     output = pd.DataFrame(pred_output, columns=['deal_id', 'poi_id','sales'])
#     output['deal_poi'] = output['deal_id'].astype(str) + '_'  + output['poi_id'].astype(str) 
#     output[['deal_poi', 'sales']].groupby('deal_poi').mean().to_csv(file_name,index=True)
# ```

# ```python
# import lightgbm as gb
# 
# from lightgbm.plotting import *
# from matplotlib import pyplot
# 
# cat_cols = 'poi_rank_hash'.split()
# cat_ft = [i for i in (cat_cols) if i in features]
# 
# dtrain = gb.Dataset(x_train, y_train, feature_name = features, categorical_feature = cat_ft)
# deval = gb.Dataset(x_val, y_val, feature_name = features, categorical_feature = cat_ft, reference = dtrain)
# 
# 
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'MAE',
#     'num_iterations':5000,
#     'early_stopping_round': 200,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.6,
#     'verbose': 1,
#     'nthread': 7,
#     'train_metric': 'true'
# }
#     
# gbm = gb.train(params,
#                 dtrain, 
#                 valid_sets=[dtrain, deval],
#                 verbose_eval=params['num_iterations']/20,
#               )
# 
# gbm.save_model('lightgbm.model')
# save_csv(gbm.predict(x_test), 'lightgbm_submit.csv')
# 
# ax = plot_importance(gbm, max_num_features=20)
# pyplot.show()
# 
# ```
# 

# public leaderboard:  108
# 
# # Good luck ^o^
