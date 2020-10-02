import tensorflow as tf
import tensorflow.contrib as contrib
import pandas as pd
import tempfile
import numpy as np
import tempfile



#WAGE PREDICTION USING A COMBINATION OF WIDE AND DEEP LEARNING ALGORITHMS

#DEFINE PATHS TO DATA FILES

train_file = "../input/adult-training.csv"

test_file = "../input/adult-test.csv"

#DEFINE COLUMNS
COLUMNS = ["age", "workclass", "fnlwgt", "education", 

"education_num",
           "marital_status", "occupation", "relationship", "race", 

"gender",
           "capital_gain", "capital_loss", "hours_per_week", 

"native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", 

"occupation",
                       "relationship", "race", "gender", 

"native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", 

"capital_loss",
                      "hours_per_week"]

df_train = pd.read_csv(train_file, names = COLUMNS, 

skipinitialspace = True,engine= "python")
df_test = pd.read_csv(test_file,names = COLUMNS,skipinitialspace = True, skiprows=1, engine = "python")

#Remove NaN

df_train.dropna(how="any",axis = 0)
df_test.dropna(how="any", axis = 0)

#Set LABEL COLUMN
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: 

">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: 

">50K" in x)).astype(int)

#CREATE CONTINUOS COLUMNS

age = contrib.layers.real_valued_column("age")
age_buckets = contrib.layers.bucketized_column(age,boundaries=[18, 

25, 30, 35, 40, 45,
                                                        50, 55, 60, 

65])
education_num = contrib.layers.real_valued_column("education_num")
capital_gain = contrib.layers.real_valued_column("capital_gain")
capital_loss = contrib.layers.real_valued_column("capital_loss")
hours_per_week = contrib.layers.real_valued_column("hours_per_week")

#CREATE CATEGORICAL COLUMNS
workclass = contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size= 100)
education = contrib.layers.sparse_column_with_hash_bucket("education",hash_bucket_size=100)
marital_status = contrib.layers.sparse_column_with_hash_bucket("marital_status",hash_bucket_size=100)
occupation = contrib.layers.sparse_column_with_hash_bucket("occupation",hash_bucket_size=1000)
relationship = contrib.layers.sparse_column_with_hash_bucket("relationship",hash_bucket_size=100)
race = contrib.layers.sparse_column_with_hash_bucket("race",hash_bucket_size=100)
native_country = contrib.layers.sparse_column_with_hash_bucket("education",hash_bucket_size=1000)
gender = contrib.layers.sparse_column_with_keys("gender",keys=["male","female"])

#CROSS COLUMNS
education_occupation = contrib.layers.crossed_column(columns=[education,occupation],hash_bucket_size= int(1e4))
age_education_occupation = contrib.layers.crossed_column(columns=[age_buckets,education,occupation], hash_bucket_size= int(1e6))
native_country_occupation = contrib.layers.crossed_column(columns= [native_country,occupation], hash_bucket_size= int(1e4))
race_occupation = contrib.layers.crossed_column(columns = [race,occupation], hash_bucket_size = int(1e4))

#WIDE COLUMNS
wide_columns = [age,age_buckets,education_num,capital_gain,capital_loss,hours_per_week,workclass,education
                

,marital_status,occupation,relationship,race,native_country,gender,education_occupation,age_education_occupation
                ,native_country_occupation,race_occupation]

deep_columns = [age,education_num,capital_gain,capital_loss,hours_per_week,
                contrib.layers.embedding_column(workclass,dimension=8),
                contrib.layers.embedding_column(education,dimension=8),
                contrib.layers.embedding_column(marital_status,dimension=8),
                contrib.layers.embedding_column(occupation,dimension=8),
                contrib.layers.embedding_column(relationship,dimension=8),
                contrib.layers.embedding_column(race,dimension=8),
                contrib.layers.embedding_column(native_country,dimension=8),
                contrib.layers.embedding_column(gender,dimension=8)
                ]

def input_function(df):
    continuos_cols = {k: tf.constant(df[k].values) for k in 

CONTINUOUS_COLUMNS}

    categorical_cols = {k: tf.SparseTensor(indices= [[i,0] for i in 

range(df[k].size)],
                                           values= df[k].values,
                                           dense_shape= [df

[k].size,1])
                        for k in CATEGORICAL_COLUMNS}

    label = tf.constant(df[LABEL_COLUMN].values)

    feature_cols = dict(continuos_cols)
    feature_cols.update(categorical_cols)
    return feature_cols,label

#DEFINE MODEL DIR
model_dir = tempfile.mkdtemp()


#BUILD AND TRAIN MODEL

m = contrib.learn.DNNLinearCombinedClassifier(model_dir = 
model_dir,linear_feature_columns=wide_columns,dnn_feature_columns=deep_columns,
                                              dnn_hidden_units= [100,50],fix_global_step_increment_bug=True)


m.fit(input_fn= lambda: input_function(df_train),steps= 200)

results = m.evaluate(input_fn = lambda: input_function(df_test),steps = 1)

for key in sorted(results):
    print(key, end= " ")
    print(results[key])