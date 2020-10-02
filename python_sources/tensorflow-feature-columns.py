#!/usr/bin/env python
# coding: utf-8

# # Tensorflow feature columns
# 
# Tensorflow nodes operate on numbers (integers or floats), but your data may feature categorical or other types of information. Tensorflow tensors must have types, and those types must ultimately resolve to said numbers, but it  includes some functions for reconstituting features.
# 
# If the feature is already numeric you can encode it using little more than `tf.feature_column.numeric_column`.

# In[ ]:


import tensorflow as tf
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength", dtype=tf.float32)


# If the tensor is a matrix, you can provide a `shape` expressing the dimensions.

# In[ ]:


matrix_feature_column = tf.feature_column.numeric_column(key="MyMatrix", shape=[10,5])


# Partitioning a numerical column into a set of indicator categoricals can be done using `bucketized_column`:
# 

# In[ ]:


bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000])


# An identity categorical (each unique category gets its own categorical column) can be done using `categorical_column_with_identity`. This is exactly the same as `pd.to_dummies`.

# In[ ]:


tf.feature_column.categorical_column_with_identity


# Strings may be transformed into categoricals using a volcabulary list, using either `categorical_column_with_vocabulary_list` or `categorical_column_with_vocabulary_file`. Which of these two is more convenient depends on how many strings you have; enumerating a long list quickly becomes cumbersome. This function is barely different than `categorical_column_with_identity`; the only difference seems to be that it produces explicitly enumerated output?

# In[ ]:


tf.feature_column.categorical_column_with_vocabulary_file


# The first interesting tool IMO is the hash bucket. Hash bucketing is a random partitioning scheme that assigns classes to buckets randomly, meant for cases when there are way too many classes to work with in a categorical encoding. It apparently can work surprisingly well in practice, which is very counterinuitive...you can hash bucket your categories using `tf.feature_column.categorical_column_with_hash_bucket`.
# 
# This hashing idea also comes up in feature crossing. You can cross useful features, like latitude and longitude, using `tf.feature_column.crossed_column`, potentially with prior bucketization to deal with high numeracy. If this still results in too many categories, you can still have a huge sparse matrix. You can condense the result by hashing it!  This can be done as follows:

# In[ ]:


# Cross the bucketized columns, using 5000 hash bins.
crossed_lat_lon_fc = tf.feature_column.crossed_column([bucketized_feature_column, bucketized_feature_column], 5000)


# The docs say that you should still provide the original features to the model as well, as they can help distinguish between classes in the case of hash collisions.
# 
# I wonder how many hash collisions an algorithm can tolerate before it becomes too many hash collisions. Might be worth exploring that in a separate notebook.
# 
# Finally, Tensorflow includes a `tf.feature_column.embedding_column` builtin for generating an embedding.
# 
# Overall, I'm quixotic about the Tensorflow feature column utilities. It seems to me like you can achieve more using dedicated transform tools, and though you need to know how to work with feature columns because your inputs to the model must be wrapped in them you shouldn't rely on tensorflow to handle your feature transforms.
