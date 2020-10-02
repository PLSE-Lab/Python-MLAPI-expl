# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

tfe = tf.contrib.eager
EMBEDDING_SIZE = 8 # K
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status",
                       "occupation", "relationship", "race",
                       "gender", "native_country"]
CONTINUOUS_COLUMNS = [ "age","fnlwgt","education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]
                      
data = pd.read_csv("../input/adult.data",names=COLUMNS,skipinitialspace=True,dtype={
    **dict(zip(CONTINUOUS_COLUMNS,["int32"]*len(CATEGORICAL_COLUMNS))),
    **dict(zip(CATEGORICAL_COLUMNS,["category"]*len(CATEGORICAL_COLUMNS)))
    })
data[LABEL_COLUMN] = data["income_bracket"].apply(lambda x: ">50K" in x).astype("int32")

cat_data = data[CATEGORICAL_COLUMNS].apply(lambda x:x.cat.codes)
cont_data = data[CONTINUOUS_COLUMNS]

dataset = tf.data.Dataset.from_tensor_slices((dict(cat_data),dict(cont_data),data[LABEL_COLUMN])).batch(32)


feat_embeddings_weight = tfe.Variable(
    tf.random_normal([len(CATEGORICAL_COLUMNS),EMBEDDING_SIZE],0.0,0.01),
    name="feat_embeddings_weight") # F * K

feat_bias_weight = tfe.Variable(
    tf.random_uniform([len(CATEGORICAL_COLUMNS),1],0.0,0.01),name ="feat_bias_weight" 
    )
    
num_epochs = 200

for epoch in range(num_epochs):
    for X_cat,X_cont,y in dataset:
        order1 = tf.reduce_sum(tf.multiply(X_cat,feat_bias_weight),1)
        
        tf.multiply(tf.square(feat_embeddings_weight),tf.square(X_cat))
        
        
        
        
print(feat_bias_weight) 