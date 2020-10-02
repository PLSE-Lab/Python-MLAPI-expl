#!/usr/bin/env python
# coding: utf-8

# This tutorial introduces data scientists to BigQuery ML and is based
# on the [official documentation tutorial](https://cloud.google.com/bigquery/docs/bigqueryml-scientist-start). BigQuery ML enables
# users to create and execute machine learning models in BigQuery
# using SQL queries. The goal is to democratize machine learning by enabling SQL
# practitioners to build models using their existing tools and to increase
# development speed by eliminating the need for data movement.
# 
# In this tutorial, you use the sample
# [Google Analytics sample dataset for BigQuery](https://support.google.com/analytics/answer/7586738?hl=en&ref_topic=3416089)
# to create a model that predicts whether a website visitor will make a
# transaction. For information on the schema of the Analytics dataset, see
# [BigQuery export schema](https://support.google.com/analytics/answer/3437719)
# in the Google Analytics Help Center.
# 

# ## Objectives
# 
# In this tutorial, you use:
# 
# + BQML to create a binary logistic regression model using the
#   `CREATE MODEL` statement
# + The `ML.EVALUATE` function to evaluate the ML model
# + The `ML.PREDICT` function to make predictions using the ML model

# ## Costs
# 
# This tutorial uses billable components of Cloud Platform,
# including:
# 
# + BigQuery
# + BigQuery ML
# 
# 
# For more information on BigQuery costs, see the [Pricing](https://cloud.google.com/bigquery/pricing)
# page.
# 
# For more information on BigQuery ML costs, see the [BQML pricing](https://cloud.google.com/bigquery/bqml-pricing)
# section of the pricing page.

# ## Step one: Setup and create your dataset
# 
# Next, you create a BigQuery dataset to store your
# ML model.

# In[ ]:


# Set your own project id here
PROJECT_ID = 'bigquerytestdefault'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('bqml_tutorial', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID


# ## Step two: Create your model
# 
# Next, you create a logistic regression model using the Google Analytics sample
# dataset for BigQuery. The model is used to predict whether a
# website visitor will make a transaction. The standard SQL query uses a
# `CREATE MODEL` statement to create and train the model.

# The BigQuery Python client library provides a magic command that
# allows you to run queries with minimal code. To load the magic commands from the
# client library, enter the following code.

# In[ ]:


get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# Note: `%load_ext` is one of the many Jupyter built-in magic commands. See the
# [Jupyter documentation](https://ipython.readthedocs.io/en/stable/interactive/magics.html) for more
# information about `%load_ext` and other magic commands.
# 
# The BigQuery client library provides a cell magic,
# `%%bigquery`, which runs a SQL query and returns the results as a Pandas
# DataFrame. Enter the following standard SQL query in the cell. The `#standardSQL`
# prefix is not required for the client library. Standard SQL is the default
# query syntax.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'CREATE MODEL IF NOT EXISTS `bqml_tutorial.sample_model`\nOPTIONS(model_type=\'logistic_reg\') AS\nSELECT\n  IF(totals.transactions IS NULL, 0, 1) AS label,\n  IFNULL(device.operatingSystem, "") AS os,\n  device.isMobile AS is_mobile,\n  IFNULL(geoNetwork.country, "") AS country,\n  IFNULL(totals.pageviews, 0) AS pageviews\nFROM\n  `bigquery-public-data.google_analytics_sample.ga_sessions_*`\nWHERE\n  _TABLE_SUFFIX BETWEEN \'20160801\' AND \'20170630\'')


# The query takes several minutes to complete. After the first iteration is
#     complete, your model (`sample_model`) appears in the navigation panel of the
#     BigQuery UI. Because the query uses a `CREATE MODEL` statement to create a
#     table, you do not see query results. The output is an empty string.

# ## Step three: Get training statistics
# 
# To see the results of the model training, you can use the
# [`ML.TRAINING_INFO`](/bigquery/docs/reference/standard-sql/bigqueryml-syntax-train)
# function, or you can view the statistics in the BigQuery UI.
# In this tutorial, you use the `ML.TRAINING_INFO` function.
# 
# A machine learning algorithm builds a model by examining many examples and
# attempting to find a model that minimizes loss. This process is called empirical
# risk minimization.
# 
# Loss is the penalty for a bad prediction &mdash; a number indicating
# how bad the model's prediction was on a single example. If the model's
# prediction is perfect, the loss is zero; otherwise, the loss is greater. The
# goal of training a model is to find a set of weights that have low
# loss, on average, across all examples.
# 
# To see the model training statistics that were generated when you ran the
# `CREATE MODEL` query, run the following:

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `bqml_tutorial.sample_model`)')


# Note: Typically, it is not a best practice to use a `SELECT *` query. Because the model output is a small table, this query does not process a large amount of data. As a result, the cost is minimal.
# 
# The `loss` column represents the loss metric calculated after the given iteration
#     on the training dataset. Since you performed a logistic regression, this column
#     is the [log loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression).
#     The `eval_loss` column is the same loss metric calculated on
#     the holdout dataset (data that is held back from training to validate the model).
# 
# For more details on the `ML.TRAINING_INFO` function, see the
#     [BQML syntax reference](https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-train).

# ## Step four: Evaluate your model
# 
# After creating your model, you evaluate the performance of the classifier using
# the [`ML.EVALUATE`](/bigquery/docs/reference/standard-sql/bigqueryml-syntax-evaluate)
# function. You can also use the [`ML.ROC_CURVE`](/bigquery/docs/reference/standard-sql/bigqueryml-syntax-roc)
# function for logistic regression specific metrics.
# 
# A classifier is one of a set of enumerated target values for a label. For
# example, in this tutorial you are using a binary classification model that
# detects transactions. The two classes are the values in the `label` column:
# `0` (no transactions) and not `1` (transaction made).
# 
# To run the `ML.EVALUATE` query that evaluates the model, run the following:

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `bqml_tutorial.sample_model`, (\n  SELECT\n    IF(totals.transactions IS NULL, 0, 1) AS label,\n    IFNULL(device.operatingSystem, "") AS os,\n    device.isMobile AS is_mobile,\n    IFNULL(geoNetwork.country, "") AS country,\n    IFNULL(totals.pageviews, 0) AS pageviews\n  FROM\n    `bigquery-public-data.google_analytics_sample.ga_sessions_*`\n  WHERE\n    _TABLE_SUFFIX BETWEEN \'20170701\' AND \'20170801\'))')


# Because you performed a logistic regression, the results include the following columns (click to learn more):
# 
# + [`precision`](https://developers.google.com/machine-learning/glossary/#precision)
# + [`recall`](https://developers.google.com/machine-learning/glossary/#recall)
# + [`accuracy`](https://developers.google.com/machine-learning/glossary/#accuracy)
# + [`f1_score`](https://en.wikipedia.org/wiki/F1_score)
# + [`log_loss`](https://developers.google.com/machine-learning/glossary/#Log_Loss)
# + [`roc_auc`](https://developers.google.com/machine-learning/glossary/#AUC)

# ## Step five: Use your model to predict outcomes
# 
# Now that you have evaluated your model, the next step is to use it to predict
# outcomes. You use your model to predict the number of transactions made by
# website visitors from each country. And you use it to predict purchases per user.
# 
# To run the query that uses the model to predict the number of transactions:

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  country,\n  SUM(predicted_label) as total_predicted_purchases\nFROM ML.PREDICT(MODEL `bqml_tutorial.sample_model`, (\n  SELECT\n    IFNULL(device.operatingSystem, "") AS os,\n    device.isMobile AS is_mobile,\n    IFNULL(totals.pageviews, 0) AS pageviews,\n    IFNULL(geoNetwork.country, "") AS country\n  FROM\n    `bigquery-public-data.google_analytics_sample.ga_sessions_*`\n  WHERE\n    _TABLE_SUFFIX BETWEEN \'20170701\' AND \'20170801\'))\n  GROUP BY country\n  ORDER BY total_predicted_purchases DESC\n  LIMIT 10')


# In the next example, you try to predict the number of transactions each website
# visitor will make. This query is identical to the previous query except for the
# `GROUP BY` clause. Here the `GROUP BY` clause &mdash; `GROUP BY fullVisitorId`
# &mdash; is used to group the results by visitor ID.
# 
# To run the query that predicts purchases per user:

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  fullVisitorId,\n  SUM(predicted_label) as total_predicted_purchases\nFROM ML.PREDICT(MODEL `bqml_tutorial.sample_model`, (\n  SELECT\n    IFNULL(device.operatingSystem, "") AS os,\n    device.isMobile AS is_mobile,\n    IFNULL(totals.pageviews, 0) AS pageviews,\n    IFNULL(geoNetwork.country, "") AS country,\n    fullVisitorId\n  FROM\n    `bigquery-public-data.google_analytics_sample.ga_sessions_*`\n  WHERE\n    _TABLE_SUFFIX BETWEEN \'20170701\' AND \'20170801\'))\n  GROUP BY fullVisitorId\n  ORDER BY total_predicted_purchases DESC\n  LIMIT 10')


# ## Cleanup
# To avoid incurring charges to your Google Cloud Platform account for the resources used in this tutorial:
# 
# + You can delete the project you created.
# + Or you can keep the project and delete the dataset.
