#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Extended (TFX)
# - In this kernel we try to use Tensorflow data validation library to easily visualize the statistics and given schema,the other main purpose of this tool is to compare train and test statistics (Train -Test Skew)
# - This tensorflow data validation packages uses Facets internally to render interactive visualization
# - This packages was used in production system to validate the data for anomalies before passing into 
#     our model (ML/DL model -since model expects data in particular schema)
# - For further information kindly go through the **TFX documentation** 
# 
# [TFX](https://www.tensorflow.org/tfx)
# 
# [Tensorflow Data validation](https://www.tensorflow.org/tfx/data_validation/get_started)
# 
# 
# *
# Disclaimer: we can analyse the Descriptive nature of the data by rich set of tools already available in the python 
# ecosystem, this kernel purpose is to introduce the new tool (from Tensorflow ecosytem) which can be useful in analysing descriptive nature and production system also .*

# In[ ]:


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


# # Installing Tensorflow data validation
# - Tfdv uses Apache Beam to full pass the dataset (Big datasets) and for aggregations 
# - Apache Beam has some nice characteristics such as 
#     - **unfied programing model **for Batch and stream processing
#     - **portable** Execute pipelines on multiple execution environments.
#     - Apache beam supports different language SDKs
#     - Apache beam runs with different backend runners such as 
#         - Apache Flink
#         - Apache Spark
#         - Apache Gearpump
#         - Apache Nemo
#         - Apache Samza
#         - Google cloud dataflow
#         - Direct runners (without parallelism in local machines)
#         
# *Note this kernel uses Direct runners this may be slow while getting statistics from the dataset.***

# In[ ]:


get_ipython().system('pip install --user tensorflow_data_validation')


# In[ ]:


import tensorflow_data_validation as tfdv


# # Generating Statistics
# Satistics is nothing but tfdv full passes the dataset and collects the characteristics (aggregation metrics) of 
# the dataset and use this statistics (Train statistics ) to validate the statistics of the test data in prediction or serving time.
# - Currently, We can prepare Satistics of the dataset in three ways :
#     - Directly from the CSVs
#     - From pandas Dataframe
#     - From Tensorflow specific- TFRecord

# In[ ]:


dipole_moments=pd.read_csv("../input/dipole_moments.csv")
magnetic_shielding_tensors=pd.read_csv("../input/magnetic_shielding_tensors.csv")
mulliken_charges=pd.read_csv("../input/mulliken_charges.csv")
potential_energy=pd.read_csv("../input/potential_energy.csv")
structures=pd.read_csv("../input/structures.csv")
scalar_coupling_contributions=pd.read_csv("../input/scalar_coupling_contributions.csv")
#dipole_moments=pd.read_csv("../input/dipole_moments.csv")


# In[ ]:


#dipole_moments=tfdv.generate_statistics_from_csv('../input/dipole_moments.csv')
#magnetic_shielding_tensors=tfdv.generate_statistics_from_csv('../input/magnetic_shielding_tensors.csv')
#mulliken_charges=tfdv.generate_statistics_from_csv('../input/mulliken_charges.csv')
#potential_energy=tfdv.generate_statistics_from_csv('../input/potential_energy.csv')
#scalar_coupling_contributions=tfdv.generate_statistics_from_csv('../input/scalar_coupling_contributions.csv')


# Run any of the above cells to generate statistics, i observed generating statistics from the pandas dataframe was 
# bit faster than generating from the CSVs.

# # Visualizing the Generated Statistics in Facets 
# - This is as simple as calling **tfdv.visualize_statistics** api

# In[ ]:


dipole_moments_stats=tfdv.generate_statistics_from_dataframe(dipole_moments)
tfdv.visualize_statistics(dipole_moments_stats)


# In[ ]:


magnetic_shielding_tensors_stats=tfdv.generate_statistics_from_dataframe(magnetic_shielding_tensors)
tfdv.visualize_statistics(magnetic_shielding_tensors_stats)


# In[ ]:


mulliken_charges_stats=tfdv.generate_statistics_from_dataframe(mulliken_charges)
tfdv.visualize_statistics(mulliken_charges_stats)


# In[ ]:


potential_energy_stats=tfdv.generate_statistics_from_dataframe(potential_energy)
tfdv.visualize_statistics(potential_energy_stats)


# In[ ]:


structures_stats=tfdv.generate_statistics_from_dataframe(structures)
tfdv.visualize_statistics(structures_stats)


# In[ ]:


scalar_coupling_contributions_stats=tfdv.generate_statistics_from_dataframe(scalar_coupling_contributions)
tfdv.visualize_statistics(scalar_coupling_contributions_stats)


# # Saving schema for training statistics
# 
# - Schema is nothing but the expected features /schema of the given dataset
#     - Schema can be prepared in two ways:
#         - Manually write the schema
#         - USe the generated Satistics to infer schema.
# - You can save/serialize the schema of your training dataset in to the disk and later you load and validate 
#     the statistis of the test data

# In[ ]:


dipole_moments_schema=tfdv.infer_schema(dipole_moments_stats)
tfdv.write_schema_text(dipole_moments_schema,"dipole_moments_schema")

magnetic_shielding_tensors_schema=tfdv.infer_schema(magnetic_shielding_tensors_stats)
tfdv.write_schema_text(magnetic_shielding_tensors_schema,"magnetic_shielding_tensors_schema")

mulliken_charges_schema=tfdv.infer_schema(mulliken_charges_stats)
tfdv.write_schema_text(mulliken_charges_schema,"mulliken_charges_schema")

potential_energy_schema=tfdv.infer_schema(potential_energy_stats)
tfdv.write_schema_text(potential_energy_schema,"potential_energy_schema")

structures_schema=tfdv.infer_schema(structures_stats)
tfdv.write_schema_text(structures_schema,"structures_schema")

scalar_coupling_contributions_schema=tfdv.infer_schema(scalar_coupling_contributions_stats)
tfdv.write_schema_text(scalar_coupling_contributions_schema,"scalar_coupling_contributions_schema")


# Schemas are stored in Output DIR = '.'

# In[ ]:


print(os.listdir(".")),tfdv.load_schema_text('magnetic_shielding_tensors_schema')


# # Validate Generated Schema
# - Look at the generated schema, and validate the data type of the each feature, if everything is allright you can
# save the schema and use this schema to validate the test data

# In[ ]:


tfdv.display_schema(dipole_moments_schema)


# In[ ]:


tfdv.display_schema(magnetic_shielding_tensors_schema)


# In[ ]:


tfdv.display_schema(mulliken_charges_schema)


# In[ ]:


tfdv.display_schema(potential_energy_schema)


# In[ ]:


tfdv.display_schema(scalar_coupling_contributions_schema)


# In[ ]:


tfdv.display_schema(structures_schema)


# In[ ]:


train=pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


test=pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)


# In[ ]:


train.shape,test.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Lets Generate Statistics for Train and Test datasets 
# - This time use all the train features in single dataframe and generate statistics 
# - Use the above generated Train statistics to the test datasets to identify anomalies or data deviations

# In[ ]:


train_stats=tfdv.generate_statistics_from_dataframe(train)


# In[ ]:


test_stats=tfdv.generate_statistics_from_dataframe(test)


# In[ ]:


train_schema=tfdv.infer_schema(train_stats)
tfdv.write_schema_text(train_schema,'train_schema')


# In[ ]:


test_schema=tfdv.infer_schema(test_stats)
tfdv.write_schema_text(test_schema,'test_schema')


# # Validation

# In[ ]:


anomalies=tfdv.validate_statistics(test_stats,train_schema)
tfdv.display_anomalies(anomalies)


# In[ ]:


tfdv.visualize_statistics(lhs_statistics=test_stats, rhs_statistics=train_stats,
                          lhs_name='TEST_DATASET', rhs_name='TRAIN_DATASET')

