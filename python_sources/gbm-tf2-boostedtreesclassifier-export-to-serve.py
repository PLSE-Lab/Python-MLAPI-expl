#!/usr/bin/env python
# coding: utf-8

# # Purpose
# * To introduce `BoostedTreesClassifier` and how to save a `tf.estimator` model in the ProtoBuf (.pb) format for serving.
# 
# * *For Reference:*   
# [Introduction to GradientBoosting - YouTube/StatQuest with Josh Starmer](https://youtu.be/3CC4N4z3GJc?t=5)   
# [Why Estimators?](https://www.tensorflow.org/guide/estimator)  
# [What is serving?](https://www.tensorflow.org/tfx/guide/serving)  
# [Protocol Buffers](https://en.wikipedia.org/wiki/Protocol_Buffers)
# 

# In[ ]:


import pandas as pd
import tensorflow as tf

seed = 1010
tf.random.set_seed(seed)


# ## Load the dataset 

# In[ ]:


df1 = pd.read_csv('../input/flight-delay-prediction/Jan_2019_ontime.csv')
df2 = pd.read_csv('../input/flight-delay-prediction/Jan_2020_ontime.csv')
df1.shape, df2.shape


# In[ ]:


# Merge the datasets
df = pd.concat([df1, df2], ignore_index=True)
df.info()


# In[ ]:


df.head()


# ## Prepare the dataset for training

# In[ ]:


# Remove the columns not used in training
cols_to_drop = []
cols_to_drop.append('OP_UNIQUE_CARRIER')
cols_to_drop.append('OP_CARRIER_AIRLINE_ID')
cols_to_drop.append('OP_CARRIER')
cols_to_drop.append('ORIGIN_AIRPORT_SEQ_ID')
cols_to_drop.append('ORIGIN')
cols_to_drop.append('DEST_AIRPORT_SEQ_ID')
cols_to_drop.append('DEST')
cols_to_drop.append('DEP_TIME')
cols_to_drop.append('ARR_TIME')
cols_to_drop.append('ARR_DEL15')
cols_to_drop.append('CANCELLED')
cols_to_drop.append('DIVERTED')
cols_to_drop.append('DISTANCE')
cols_to_drop.append('Unnamed: 21')

df.drop(columns=cols_to_drop, inplace=True)


# In[ ]:


df.isna().any()


# In[ ]:


totalrows = df.shape[0]

for col in df.columns:
    nas = sum(df[col].isna())
    if nas:
        print(f'Column {col} has {nas} ({(nas/totalrows)*100:.2f}% of total) NAs.')
print('Done looking for NAs')


# In[ ]:


df.dropna(inplace=True)
df.shape


# ## Split dataset for training and evaluation 

# In[ ]:


split   = 0.8 # 80/20 split for training and evaluation
dftrain = df.sample(frac=split, random_state=seed) 
dfeval  = df.drop(dftrain.index)
dftrain.shape, dfeval.shape


# In[ ]:


# Extract and remove the label (to be predicted) set
y_train = dftrain.pop('DEP_DEL15')
y_eval  = dfeval.pop('DEP_DEL15')
y_train.shape, y_eval.shape


# ## Using Estimator

# In[ ]:


dftrain.dtypes


# #### Create Feature Columns

# In[ ]:


def onehot_catgcol(df, column):
    fc = tf.feature_column
    values  = df[column].unique()
    cat_col = fc.categorical_column_with_vocabulary_list(column, values)
    return (fc.indicator_column(cat_col))


# In[ ]:


categorical_cols = ['TAIL_NUM', 'DEP_TIME_BLK']
numeric_cols = [i for i in dftrain.columns if i not in categorical_cols]

categorical_cols, numeric_cols


# In[ ]:


# tf.estimator requires the features to be Tensors
fc = tf.feature_column
features = [fc.numeric_column(i)  for i in numeric_cols]
fc_catgs = [onehot_catgcol(df, i) for i in categorical_cols]
features.extend(fc_catgs)

len(features)


# #### Create Input Function for both Training and Evaluation

# In[ ]:


def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


# #### Set up Hyper Parameters

# In[ ]:


params = {
    'n_trees':50,
    'max_depth':3,
    'n_batches_per_layer':1,
    'center_bias':True
}
classifier = tf.estimator.BoostedTreesClassifier(features, **params)


# #### Train

# In[ ]:


classifier.train(
    input_fn=lambda: input_fn(dftrain, y_train, training=True),
    max_steps=100)


# #### Evaluation

# In[ ]:


eval_result = classifier.evaluate(
                input_fn=lambda: input_fn(dfeval, y_eval, training=False))

print(f'Evaluation set accuracy = {eval_result["accuracy"]*100:.2f}%')


# In[ ]:


pd.Series(eval_result).to_frame()


# ## Saving the Estimator Model

# In[ ]:


def serving_fn():
    day_of_month      = tf.Variable([], dtype=tf.int64, name='DAY_OF_MONTH')
    day_of_week       = tf.Variable([], dtype=tf.int64, name='DAY_OF_WEEK')
    tail_num          = tf.Variable([], dtype=tf.string,name='TAIL_NUM')
    op_carrier_fl_num = tf.Variable([], dtype=tf.int64, name='OP_CARRIER_FL_NUM')
    origin_airport_id = tf.Variable([], dtype=tf.int64, name='ORIGIN_AIRPORT_ID')
    dest_airport_id   = tf.Variable([], dtype=tf.int64, name='DEST_AIRPORT_ID')
    dep_time_blk      = tf.Variable([], dtype=tf.string,name='DEP_TIME_BLK')
    
    reqd_inputs =  {'DAY_OF_MONTH':day_of_month,
                    'DAY_OF_WEEK':day_of_week,
                    'TAIL_NUM':tail_num,
                    'OP_CARRIER_FL_NUM':op_carrier_fl_num,
                    'ORIGIN_AIRPORT_ID':origin_airport_id,
                    'DEST_AIRPORT_ID':dest_airport_id,
                    'DEP_TIME_BLK':dep_time_blk}
    
    fn = tf.estimator.export.build_raw_serving_input_receiver_fn(reqd_inputs)
    return fn


# In[ ]:


get_ipython().system("rm -r '../output/kaggle/working/'")


# In[ ]:


# Note that we are using serving_fn as a function () while passing as arg
classifier.export_saved_model('../output/kaggle/working/', serving_fn())


# #### View the saved model files
# Using `find` as an alternate to `tree` based on solution from: [Stackoverflow/Bryan Heden](https://stackoverflow.com/questions/54228819/tree-command-not-found)

# In[ ]:


get_ipython().system('find ../output/kaggle/working/ -print | sed -e "s;[^/]*/;|____;g;s;____|; |;g"')

