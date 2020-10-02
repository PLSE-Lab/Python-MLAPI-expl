#!/usr/bin/env python
# coding: utf-8

# This kernel is based on [the starter](https://www.kaggle.com/sirtorry/bigquery-ml-template-intersection-congestion) and it's still under developed. Welcome for any comments and ideas. 

# In[ ]:


from google.cloud import bigquery
from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials

PROJECT_ID = 'kaggle-bqml-256206'

client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('bqml_example', exists_ok=True)
# using magics.context object to set credentials
magics.context.credentials = KaggleKernelCredentials()
# using magics.context object to set project
magics.context.project = PROJECT_ID


# Kaggle competition dataset (`kaggle-competition-datasets`) is a private dataset which doesn't allow public viewing. In order to acquire "read" access, one should follow the instruction file, `BigQuery-Dataset-Access.md`, listed in Data folder to join [the specific google group](https://groups.google.com/d/forum/bigquery-geotab ). 
# 
# A minor notice is that you need to join the `bigquery-geotab` google group with the google acount which is also the BigQuery project owner. 
# My case is signing in Kaggle with one google account and using another for BigQuery cloud project. Without explicitly telling which account for use, it might automatically re-direct to the Kaggle account by default. Either opening a clean tab/window, or explicitly logging into the BigQuery user, and then join google group for dataset access permission. 

# In[ ]:


# create a reference to our table
train_table = "kaggle-competition-datasets.geotab_intersection_congestion.train"
test_table = 'kaggle-competition-datasets.geotab_intersection_congestion.test'


# In[ ]:


import os
import json
import numpy as np
DATA_FOLDER = '../input/bigquery-geotab-intersection-congestion'
with open(os.path.join(DATA_FOLDER, 'submission_metric_map.json'), 
          'rt') as fp:
    submission_map = json.load(fp)
LABELS = {val:key for key, val in submission_map.items()}
print(LABELS)


# In[ ]:


feature_cols = ['RowId', 'IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName', 
                'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend', 'Month', 
                'Path', 'City']
table = client.get_table(train_table)
for field in table.schema:
    if field.name in feature_cols:
        print(field.name, field.field_type)


# Based on the BigQuery API references about ["create model" statement](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create), the BigQuery ML model will handle minimal feature engineering as long as the inputs are recogniable types. 
# 
# The supported input can be referenced from the table given within the same page under **Supported inputs** title. Once you get all input types supported, you can get the input transformation for free. You can check the corresponding transformations under **Input variable transformations** title. 
# 
# Unfortunately, sometimes the default option might not be the best one. So we are going to check on model features to ensure it's a good choice for our best understanding. 

# In[ ]:


import pandas as pd
from itertools import repeat, chain
from google.cloud.bigquery.magics import _run_query 

def stack_results(results):
    keys = list(chain.from_iterable(
        [repeat(k, len(v)) for k, v in results]))
    values = [v for k, v in results] 
    frame = pd.concat(values)
    if len(keys) == len(frame):
        frame['label'] = keys
        frame = frame.set_index('label')
    return frame

def make_query(query_text, job_config=None, **kwargs):
    query = _run_query(
        client, query_text.format(**kwargs),
        job_config=job_config)
    return query.to_dataframe()

def make_queries(query_text, configs=None, **kwargs):
    results = []
    for label in submission_map.values():
        percent = label[-2:]
        if label.startswith('T'):
            model = "bqml_example.model_TTS"
        elif label.startswith('D'):
            model = "bqml_example.model_DF"
        model += percent
        if configs is None:
            jobConfig=None
        else:
            query_params = configs[model]
            jobConfig = bigquery.QueryJobConfig()
            jobConfig.query_parameters = query_params
        df = make_query(query_text,
                        job_config=jobConfig,
                        model_name=model,
                        label_name=label,
                        **kwargs)
        results.append((label, df))
    return results
    
def iterate_query(query_text, configs=None, **kwargs):
    results = make_queries(query_text, configs=configs,
                           **kwargs)
    frame = stack_results(results)
    return frame

def change_columns(df, model_num):
    df['RowId'] = df['RowId'].apply(str) + '_%s'%(model_num)
    df.rename(columns={'RowId': 'TargetId', 
                       submission_map[model_num]: 'Target'}, 
              inplace=True)


# In[ ]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# # Train-Validation Split
# Using `RowId` to split the train / validatio will result in a biased set. As you can see in the following figure in which I plot the response value against `RowId`. You can see a simple `RowId` split will give oversampling result. In this result, the low values (close to zero) will be over-representative. For any learning algorithm such as "early stopping" depending on the performance of a validation set, picking a right validation set is very important.   
# A better way is to create validation set from the same distribution. You can use whatever sampling scheme you want. Once you have the validation set at hand, you can create a boolean column to hold the split result. When creating the BigQueryML model, using "CUSTOM" split and supply the 

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'rowids', 'SELECT \n    DISTINCT RowId,\n    TotalTimeStopped_p20,\n    TotalTimeStopped_p50,\n    TotalTimeStopped_p80, \n    DistanceToFirstStop_p20,\n    DistanceToFirstStop_p50,\n    DistanceToFirstStop_p80\nFROM\n`kaggle-competition-datasets.geotab_intersection_congestion.train`\nORDER BY RowId ASC')


# In[ ]:


# find a good split given the evalution fraction
eval_frac = 0.1
split_index = int(len(rowids)*0.9)
split_rowid = rowids.iloc[split_index][0]
internal_split = rowids.iloc[int(((rowids.RowId <= split_rowid).sum())*0.9)][0]
print("using the RowId = {} as splitting point\n"
      "training set fraction   = {:.3}\n"
      "evaluation set fraction = {:.3}".format(
          split_rowid, 
          ((rowids.RowId <= split_rowid).sum() / len(rowids)),
          ((rowids.RowId > split_rowid).sum() / len(rowids))
     ))


# In[ ]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(8, 6))
for i, label in enumerate(list(LABELS.keys())[3:]):
  plt.sca(axes[i])
  plt.title("RowId vs %s" % label)
  plt.plot(rowids['RowId'], rowids[label])
  axes[i].axvline(split_rowid, ls='--', color='r')
  top = max(rowids[label]) + 200
  axes[i].text(split_rowid + 10, top,"test split", fontsize=12,
               ha="left", rotation=30)
  axes[i].axvline(internal_split, ls='--', color='g')
  axes[i].text(internal_split  + 10, top,"valid split", fontsize=12,
               ha="left", rotation=30)
  
plt.tight_layout()


# # Feature Engineering 

# In[ ]:


# create process feature table 
proced_feature_stmt = """
CREATE TABLE IF NOT EXISTS `{dest_table}` AS 
WITH ProcedFeatures AS ( # create a table with processing string feautres
    SELECT * REPLACE(
     REGEXP_REPLACE(
         REGEXP_REPLACE(
             IFNULL(EntryStreetName, "Unkonwn"), 
                     r"St.?( |$)","Street"),
         r"Ave.?( |$)", "Avenue"
     ) AS EntryStreetName,
     REGEXP_REPLACE(
        REGEXP_REPLACE(
             IFNULL(ExitStreetName, "Unkonwn"),
                 r"St.?( |$)","Street"),
         r"Ave.?( |$)", "Avenue"
     ) AS ExitStreetName)
    FROM
      `{source_table}`
      ORDER BY RowId),
    DirectionEnc AS ( # creating a 8x8=64 direction to angle mapping
      SELECT 
        entry.directions AS EntryDirection,
        entry.angles AS EntryAngle,
        exit.directions AS ExitDirection,
        exit.angles AS ExitAngle
      FROM 
        (SELECT ARRAY<STRUCT<directions STRING, angles INT64>>[
            ("N", 0), ("NE", 45), ("E", 90), ("SE", 135), 
            ("S", 180), ("SW", 225), ("W", 270), ("NW", 325)] as code)
            AS book,
        UNNEST(book.code) as entry,
        UNNEST(book.code) as exit),
    AngleJoin AS( # create a table grouped by entryheading and exitheading 
    SELECT
      EntryHeading,
      EntryAngle,
      ExitHeading,
      Exitangle,
      heading
      FROM 
       (SELECT 
            EntryHeading, 
            ExitHeading,
            ARRAY_AGG(
              STRUCT(RowId, EntryStreetName, ExitStreetName) ORDER BY Rowid) as heading
         FROM ProcedFeatures
         GROUP BY EntryHeading, ExitHeading)
         INNER JOIN
         DirectionEnc As d
         on d.EntryDirection=EntryHeading 
         AND d.ExitDirection=ExitHeading
       ),
    PackedGeo AS ( # create a table grouped longitutde and latitude by City
       SELECT 
          City,
          ST_CENTROID_AGG(ST_GeogPoint(Longitude,Latitude)) AS Center,
          ARRAY_AGG(STRUCT(Longitude, Latitude, RowId) ORDER BY RowId) AS Points
       FROM ProcedFeatures
    GROUP BY City
  )
SELECT * FROM
(
  SELECT
  u.RowId,
  distance,
  EntryAngle,
  ExitAngle,
  (ExitAngle - EntryAngle) AS rotation,
  (u.EntryStreetName = u.ExitStreetName) AS OntoSameStreet
FROM 
  AngleJoin,
  UNNEST(AngleJoin.heading) AS u
  INNER JOIN
  (SELECT 
    ST_DISTANCE(Center, ST_GeogPoint(
     citypoint.Longitude, citypoint.Latitude)) AS distance,
     City,
     RowId
   FROM 
     PackedGeo,
     UNNEST(Points) as citypoint) As d
   ON u.RowId = d.RowId
   ORDER BY u.RowId
   )
   INNER JOIN ProcedFeatures
   USING (RowId)
   {mask_stmt}
"""
mask_stmt="""
   INNER JOIN `kaggle-bqml-256206.bqml_example.mask_train`
   USING (RowId)
"""
def create_processed_table(source_table):
    suffix = source_table.split('.')[-1]
    table_name = "proced_%s" % suffix
    dest_table = "{project_id}.{dataset_id}.{table_name}".format(
            project_id=dataset.project,
            dataset_id=dataset.dataset_id,
            table_name=table_name)
    if suffix == 'train':
        query_text = proced_feature_stmt.format(
            source_table=source_table,
            dest_table=dest_table,
            mask_stmt=mask_stmt)
    else:
        query_text = proced_feature_stmt.format(
            source_table=source_table,
            dest_table=dest_table,
            mask_stmt=''
    )
    job_config = bigquery.QueryJobConfig()
    job_config.dry_run = False
    job_config.use_query_cache = True
    _ = _run_query(client, query_text, 
                   job_config=job_config)
    return dest_table


# In[ ]:


proc_train = create_processed_table(train_table)
proc_test  = create_processed_table(test_table)


# # Feature Engineering and Selection

# In[ ]:


no_proc_stmt="""
FROM
  `{table}`
"""
proc_stmt="""
   ,distance,
    OntoSameStreet,
    EntryAngle,
    ExitAngle,
    rotation
FROM 
    `{table}`
"""
feature_stmt="""
        Weekend,
        #Hour,
        CAST(Hour AS STRING) as hour,
        #Month,
        CAST(Month AS STRING) as month,
        City,
        #Path,
        #IntersectionId,
        #CAST(IntersectionId AS STRING) as intersection,
        EntryHeading,
        ExitHeading,
        #EntryStreetName, 
        #IFNULL(EntryStreetName, "Unkonwn") AS EntryStreetNameNL,
        #ExitStreetName
        #IFNULL(ExitStreetName, "Unkonwn") AS ExitStreetNameNL
        FORMAT("%d %d", month, hour) as day,
        FORMAT("%s %d", City, IntersectionId) as cityinter
        {proc_stmt}
"""
no_proc_features = feature_stmt.format(
    proc_stmt=no_proc_stmt.format(table=train_table))
with_proc_features =  feature_stmt.format(
    proc_stmt=proc_stmt.format(table=proc_train))


# In[ ]:


eval_train_template = """SELECT
  *
FROM
  ML.TRAINING_INFO(MODEL `{model_name}`) 
ORDER BY iteration 
"""

eval_model_template="""
SELECT
  mean_squared_error,
  r2_score,
  explained_variance
FROM ML.EVALUATE(MODEL `{model_name}`, (
  SELECT
    #RowId,
    {label_name}_imask,
    {label_name} as label,
    {feature_stmt}
  WHERE
    #RowId > {cutoff_stmt}
    {label_name}_mask = True
    ))
"""

feature_exam_template = """
SELECT
  *
FROM
  ML.FEATURE_INFO(MODEL `{model_name}`)
"""
predict_with_correct_stmt="""
SELECT
  RowId,
  {label_name},
  predicted_label
FROM
  ML.PREDICT(MODEL `{model_name}`,
    (
    SELECT
        RowId,
        {label_name},
        {feature_stmt}
    )
  )
  ORDER BY RowId ASC
"""
predict_template="""
SELECT
  RowId,
  predicted_label AS {label_name}
FROM
  ML.PREDICT(MODEL `{model_name}`,
    (
    SELECT
        RowId,
        {feature_stmt}
    )
  )
  ORDER BY RowId ASC
"""


# In[ ]:


experimental = True
if experimental:
    create_stmt = "CREATE OR REPLACE MODEL"
else: 
    create_stmt = "CREATE MODEL IF NOT EXISTS"
create_model_template = """
{is_experimental} `{model_name}`
    OPTIONS(MODEL_TYPE = 'LINEAR_REG',
            OPTIMIZE_STRATEGY = 'BATCH_GRADIENT_DESCENT',
            LEARN_RATE_STRATEGY = 'LINE_SEARCH',
            L1_REG = @reg_value,
            LS_INIT_LEARN_RATE = @init_lr,
            MAX_ITERATIONS = 10, 
            EARLY_STOP = FALSE,
            #MIN_REL_PROGRESS = 0.001,
            DATA_SPLIT_METHOD = 'CUSTOM',
            DATA_SPLIT_COL = @split_col
            #DATA_SPLIT_METHOD = 'SEQ',
            #DATA_SPLIT_COL = 'RowId',
            #DATA_SPLIT_EVAL_FRACTION = 0.1 # 0.2 by default
) AS
SELECT
    #RowId,
    {label_name}_imask,
    {label_name} as label,
    {feature_stmt}
WHERE
    #RowId <= {cutoff_stmt}
    {label_name}_mask = False
ORDER BY RowId ASC
"""
configs= {
    "bqml_example.model_TTS20":[
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.2),
        bigquery.ScalarQueryParameter("reg_value", "FLOAT64", 100),
        bigquery.ScalarQueryParameter("split_col", "STRING", "TotalTimeStopped_p20_imask")
    ],
    "bqml_example.model_TTS50":[
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.2),
        bigquery.ScalarQueryParameter("reg_value", "FLOAT64", 1000),
        bigquery.ScalarQueryParameter("split_col", "STRING", "TotalTimeStopped_p50_imask")
    ],
    "bqml_example.model_TTS80":[
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.2),
        bigquery.ScalarQueryParameter("reg_value", "FLOAT64", 1000),
        bigquery.ScalarQueryParameter("split_col", "STRING", "TotalTimeStopped_p80_imask")
    ],
    "bqml_example.model_DF20": [
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.2),
        bigquery.ScalarQueryParameter("reg_value", "FLOAT64", 10000),
        bigquery.ScalarQueryParameter("split_col", "STRING", "DistanceToFirstStop_p20_imask")
    ],
    "bqml_example.model_DF50": [
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.2),
        bigquery.ScalarQueryParameter("reg_value", "FLOAT64", 10000),
        bigquery.ScalarQueryParameter("split_col", "STRING", "DistanceToFirstStop_p50_imask")
    ],
    "bqml_example.model_DF80": [
        bigquery.ScalarQueryParameter("init_lr", "FLOAT64", 0.2),
        bigquery.ScalarQueryParameter("reg_value", "FLOAT64", 1000),
        bigquery.ScalarQueryParameter("split_col", "STRING", "DistanceToFirstStop_p80_imask")
    ]
}


# In[ ]:


import re
feature_set = ''
model='bqml_example.model_DF80'
idx2feature = []
for i, each_line in enumerate(
        #no_proc_features.split('\n')):
        with_proc_features.split('\n')):
    each_line = each_line.strip()
    if each_line.startswith('FROM'):
        break
    if len(each_line) > 0 and each_line.find('#') < 0:
        job_config = bigquery.QueryJobConfig()
        job_config.query_parameters = configs[model]
        feature_set += each_line
        assert(feature_set.rstrip(', ')[-1]!=',')
        idx2feature.append(
            feature_set.rstrip(', ') + \
            no_proc_stmt.format(table=proc_train))
        model_name = 'bqml_example.model_DF80_%d' % len(idx2feature)
        # creating model
        _ = make_query(create_model_template, 
            job_config=job_config,
            model_name=model_name,
            label_name='DistanceToFirstStop_p80',
            is_experimental=create_stmt,
            feature_stmt=idx2feature[-1],
            cutoff_stmt=split_rowid)
        print(model_name, "is complete")


# In[ ]:


print(idx2feature[-1])


# In[ ]:


del_models = False
for model in client.list_models('kaggle-bqml-256206.bqml_example'):
    print(model.path)
    if del_models :
        client.delete_model(model)


# In[ ]:


feature2loss = {}
for i, this_feature in enumerate(idx2feature): 
    # evaluating model
    train_info = make_query(
            eval_train_template,
            model_name='bqml_example.model_DF80_%d' % (i + 1),
            label_name='DistanceToFirstStop_p80')
    eval_info = make_query(
            eval_model_template,
            model_name='bqml_example.model_DF80_%d' % (i + 1),
            label_name='DistanceToFirstStop_p80',
            feature_stmt=this_feature,
            cutoff_stmt=split_rowid)
    feature2loss[i] = {'eval': eval_info, 
                       'train':train_info.loc[train_info['iteration'].idxmax(),
                                              ['loss', 'eval_loss']]}
    print(i + 1, "train_loss (train, eval)= ",
              *train_info.loc[
            train_info['iteration'].idxmax(),
            ['loss', 'eval_loss']])


# In[ ]:


last_feat_info = make_query(
            feature_exam_template,
            model_name='bqml_example.model_DF80_%d' % len(idx2feature))


# In[ ]:


last_feat_info


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
eval_features = pd.concat([feature2loss[idx]['eval'] for idx in feature2loss], ignore_index=True)
loss = pd.concat([feature2loss[idx]['train'] for idx in feature2loss], axis=1)
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
axes[0].plot(eval_features.index.values + 1, eval_features['r2_score'], label='r2_score', marker='x')
axes[0].plot(eval_features.index.values + 1, eval_features['explained_variance'], label='explained_variance',
            marker='o')
axes[0].set_xticklabels(eval_features.index.values + 1)
axes[0].set_xticks(eval_features.index.values + 1)
axes[0].set_xlabel('feature set number')
axes[0].set_ylabel('r2_score / explained_variance')
axes[0].set_title('Different Feature Set vs Model capability')
axes[0].legend();
train_loss = loss.loc['loss'].reset_index()
train_loss.columns = ['max_iteration', 'loss']
axes[1].plot(train_loss.index.values + 1, train_loss['loss'].apply(np.sqrt), 
             label='train_RMSD', marker='x');
eval_loss = loss.loc['eval_loss'].reset_index()
eval_loss.columns = ['max_iteration', 'loss']
axes[1].plot(eval_loss.index.values + 1, eval_loss['loss'].apply(np.sqrt), 
             label='eval', marker='o');
axes[1].plot(eval_features.index.values + 1, 
             eval_features['mean_squared_error'].apply(np.sqrt), 
             label='test', marker='*');
axes[1].set_xticklabels(eval_loss.index.values + 1)
axes[1].set_xticks(eval_loss.index.values + 1)
axes[1].set_xlabel('feature set number')
axes[1].set_ylabel('RMSD')
axes[1].set_title('Different Feature Set vs Model loss (RMSE)')
axes[1].legend();


# In[ ]:


_ = iterate_query(create_model_template, 
                  configs=configs,
                  is_experimental=create_stmt,
                  #feature_stmt=with_geo_features,
                  #feature_stmt=no_geo_features,
                  feature_stmt=idx2feature[-1],
                  cutoff_stmt=split_rowid)


# In[ ]:


for label in submission_map.values():
    percent = label[-2:]
    if label.startswith('T'):
        model_name = "bqml_example.model_TTS"
    elif label.startswith('D'):
        model_name = "bqml_example.model_DF"
    model_name += percent
    # examine models for learning rate 
    model = client.get_model("kaggle-bqml-256206.%s" % model_name)
    model_settings = model.training_runs[0].training_options
    results = model.training_runs[0].results
    print(model_name, model_settings.initial_learn_rate, 
          results[0].learn_rate)


# In[ ]:


feat_info = iterate_query(feature_exam_template)


# In[ ]:


#feat_info
feat_info.loc['DistanceToFirstStop_p80']


# In[ ]:


for label in LABELS:
    # path is the concatenation of 'EntryStreetName', 'EntryHeading'
    # 'ExitStreetName', 'ExitHeading' 
    percent = label[-2:]
    if label.startswith('T'):
        model = "bqml_example.model_TTS"
    elif label.startswith('D'):
        model = "bqml_example.model_DF"
    model += percent
    print("{model_name} has the number of features = {}".format(
        int(feat_info.loc[label]['category_count'].fillna(1.0).sum()), 
        model_name=model)
         )


# A way to diagnose a model is to visualize weights. BigQuery ML provides a **ML.WEIGHTS** function for this purpose. One can follow [this link](https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-weights#mlweights_function) to read the official reference. It shows examples of how to call **ML.WEIGHTS** and what will be returned. 
# 
# The following SQL statement is a lazy person's query. It will return all the columns from a **ML.WEIGHTS** query. They are **'processed_input'**, **'weight'** and **'category_weights'**. If your feature is numeric, then **'category_weights'** will be an empty list. Based on the reference, a numeric feature will return NULL for **'category_weights'** column. If your feature is categorical and encoded as **one-hot-encoding**. **'weight'** column is NaN value but **'category_weights'** column will be a list of dict objects. 
# 
# Each entry of this list is a mapping of category name and weight. Returning a list is because often more than one category exists.  If you only want to examine one category, you can use the example given in the reference. That example will UNNEST an ARRAY structure based on the **processed_input** given in the WHERE clause. The ARRAY structure is what selecting category_weights will return. 
# 
# But I'm more a pandas faithful user than a SQL expert, I decided to "unnest" myself through pandas. 
# 
# The last thing is about `STRUCT(false AS standardize)`. It is a statement to ensure the weights returned to be standardized beforehand when setting true. 

# In[ ]:


# reading weight
weight_info_template="""
SELECT
  *
FROM
  ML.WEIGHTS(MODEL `{model_name}`,
    STRUCT(false AS standardize))
"""
results = make_queries(weight_info_template)
results[-1]


# In[ ]:


def handle_weight_result(weights_info, model_name):
    # numerical one
    numerical_weights = weights_info[~weights_info['weight'].isnull()]
    numerical_weights = numerical_weights[
        numerical_weights.columns[:-1]]
    numerical_weights.loc[:, 'category'] =         numerical_weights.loc[:, 'processed_input']
    numerical_weights.loc[:, 'processed_input'] = 'Numeric'
    caterical_list = []
    for _, frame in weights_info[
            weights_info['weight'].isnull()].iterrows():
        cate_frame = pd.DataFrame(frame['category_weights'])
        cate_frame['processed_input'] = frame['processed_input']
        caterical_list.append(cate_frame)
    caterical_weights = pd.concat(caterical_list)
    # total weight
    ttl_weights = pd.concat([numerical_weights, caterical_weights], 
                       sort=False, ignore_index=True)
    ttl_weights['model'] = model_name
    return  ttl_weights

model_weights = pd.concat([handle_weight_result(res, k) 
                           for k, res in results],
                         sort=False, ignore_index=True)


# In[ ]:


# box-plot for each model
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.boxenplot(x='model', y='weight', data=model_weights[['weight', 'model']])
fig = plt.gcf()
fig.autofmt_xdate();
fig.set_size_inches(10, 6)
ax.set_title('Weight Distribution Across 6 Models');


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(12, 10))
for i, (model_name, weights_per_model) in enumerate(model_weights.groupby('model')):
    row = i // 3
    col = i % 3
    axes[row, col].set_title('histogram of weights \n for predicting %s' % model_name)
    plt.sca(axes[row, col])
    for grp_name, members in  weights_per_model['weight'].groupby(
        weights_per_model['processed_input']):
        sns.distplot(members, bins=25, hist_kws={'alpha': 0.6}, label=grp_name,
                     hist=True, kde=False, norm_hist=False) 
        axes[row, col].legend();
plt.tight_layout()


# In[ ]:


train_info = iterate_query(eval_train_template)


# In[ ]:


train_info
#train_info.loc['DistanceToFirstStop_p80', ['loss', 'eval_loss']]


# In[ ]:


# plot the learning rate
markers = ['x', 'o', '*', 'v', '+', 's']
for i, (model_name, sub_info) in enumerate(
    train_info.groupby(train_info.index)):
    plt.plot(sub_info['iteration'], sub_info['learning_rate'], 
             label=model_name, marker=markers[i], linestyle='dashed')
plt.title('Learning rate change along iteration')
plt.legend(bbox_to_anchor=(1.5, 1.0));


# In[ ]:


# plot the training curve
fig, axes = plt.subplots(2, 3, figsize=(16, 12))
for i, (model_name, sub_info) in enumerate(
        train_info.groupby(train_info.index)):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    ax.plot(sub_info['iteration'], np.sqrt(sub_info['loss']), label='train')
    ax.plot(sub_info['iteration'], np.sqrt(sub_info['eval_loss']), label='evaluation')
    ax.set_ylabel('RMSD')
    ax.set_title('training curve for predicting\n %s' % model_name)
    ax.legend();


# In[ ]:


eval_info = iterate_query(eval_model_template,
                          feature_stmt=idx2feature[-1],
                          cutoff_stmt=split_rowid)


# In[ ]:


eval_info


# In[ ]:


loss_and_weights = pd.merge(eval_info['mean_squared_error'],
         model_weights['weight'].abs().groupby(
             model_weights['model']).sum(), 
         left_index=True, right_index=True)
loss_and_weights['loss_to_WeightSum_ratio'] =     loss_and_weights['mean_squared_error'] / loss_and_weights['weight']
loss_and_weights['percent_of_zero_weights'] = (model_weights['weight'] == 0).astype(
    np.float).groupby(model_weights['model']).mean()
loss_and_weights


# In[ ]:


# random split will give lower RMSD, a better but optimistic result
random_eval = np.sqrt(train_info['eval_loss'].mean())
# sequential split will give higher RMSD, a better but pessimistic result
seq_eval = np.sqrt(eval_info['mean_squared_error'].mean())
print('validation result = {:.7}'.format(random_eval))
print('test set result = {:.7}'.format(seq_eval))
print('average result = {:.7}'.format((random_eval + seq_eval) / 2))


# In[ ]:


# making predictions for training set
results = make_queries(predict_with_correct_stmt,
                       feature_stmt=idx2feature[-1]
                      )


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(12, 12))
for i, (label_name, pred_res) in enumerate(results):
    row = i // 3
    col = i % 3
    axes[row, col].plot(pred_res[label_name], pred_res[label_name], 
                       ls=':', lw=2, color='r')
    axes[row, col].set_title(label_name)
    axes[row, col].scatter(pred_res[label_name], pred_res['predicted_label'])
    axes[row, col].set_ylim(0, pred_res['predicted_label'].max() + 1)


# In[ ]:


with_proc_test =  feature_stmt.format(
    proc_stmt=proc_stmt.format(table=proc_test))
results = make_queries(predict_template,
                       feature_stmt=with_proc_test
                      )
# execute this if you want a single query
#results = make_query(predict_template,
#    model_name='bqml_example.model_TTS80',
#    label_name='TotalTimeStopped_p80')


# In[ ]:


predictions = [vframe.copy(deep=True) for _, vframe in results]
keys = [k for k, _ in results]
for k, frame in zip(keys, predictions):
    change_columns(frame, LABELS[k])
df = pd.concat(predictions)


# In[ ]:


df.to_csv('bq_submission.csv', index=False)

