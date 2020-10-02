#!/usr/bin/env python
# coding: utf-8

# ## Simple Memory-Based Collaborative Filtering Technique
# 
# ### Overview
# 
# The below code details our (after-the-fact) efforts to build a memory-based collaborative filtering (CF) recommendation system for the Santander Product Recommendation challenge, which ran a couple of years ago. The top submissions were based on classic Kaggle-winning algorithms like random forests and boosting algorithms, but we wanted to build something more in-line with the recommendation systems literature. The below can thus be viewed as a basic how-to on building a recommendation engine on difficultly-structured data; the main difficulties in this dataset arise from the fact that product information is sparse (in a traditional CF problem, we have ratings or purchase volumes attached to each product, rather than the binary data format we are faced with here). 
# 
# A good introduction to CF can be found at https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0
# 
# The presence of demographic data also posed a challenge: what is the best way to incorporate these informative variables in the CF model? We loosely followed the approach laid out in this paper: https://pdfs.semanticscholar.org/a621/441e7b688580707af3a4bf0ebff8c9e3d640.pdf
# 
# We approached the problem in 8 main steps:
# 
# 1. ** Data cleaning: ** 
# this took the form of dropping columns deemed surplus to requirements, imputing missing data.
# 2. ** Feature engineering:**
# binning continuous data into factor variables and mutating the product ownership variables to give an indication of purchase and ownership in the previous month.
# 3. ** Data subsetting: **
# Based on other kernels and our own investigations, we only used June 2015 data (i.e. one year before the test data) as a predictor.
# 4. ** De-duplicate the data: **
# there are >900,000 test individuals, but most of these will have identical purchase histories given the data sparsity: we thus de-duplicate based on the variables that will be considered for CF, and create an index to map results back to the unique users. This de-duplication is employed in all the major algorithms we used, and substantially sped up processing time.
# 5. ** Build a demographic-based and memory-based similarity matrix: **
# because of the large data size, we calculated the Manhattan distance of each test row to the training rows and used the inverse distance to derive similarities/purchase probabilities...
# 6. ** ...combine demographic-based and memory-based probabilities: **
# we combine the two models using a range of candidate weights, and nullified the probability of products each user already owns...
# 7. ** ...and derive recommendations: **
# using the derived probabilities, we output the top seven recommendations. We iterate steps 5-7 over each unique test user profile, and ascertain the optimal demographic/memory mixing parameter.
# 8. ** Re-run model using all training data for optimal mixing parameter **
# 
# Of course this solution is not exhaustive, and steps for algorithmic improvement are outlined at the end of the Kernel. We hope to be able to implement these improvements in-kernel at a future date!
# 
# Special thanks to George Hartshorn for sharing the data-cleaning code this notebook is based on!

# In[ ]:


#Import modules

import numpy as np
import pandas as pd
import csv
import datetime
from operator import sub
import xgboost as xgb
from sklearn import preprocessing, ensemble, metrics
import os
import gc
import psutil
import math
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from scipy.spatial.distance import pdist, wminkowski, squareform

pd.options.display.max_rows = 100
pd.options.display.max_columns = None

# Check data library

import os
print(os.listdir("../input"))


# In[ ]:


#Import data
path = '../input/'
traindat = pd.read_csv(path + 'train_ver2.csv', low_memory = True)
testdat = pd.read_csv(path + 'test_ver2.csv', low_memory = True)


# ## Step 1: Data Cleaning
# Now that the data is read in, we can get going with everyone's favourite bit: ** data cleaning! **

# In[ ]:


#Define columns of interest, based on other kernels' output 

demographic_cols = ['fecha_dato',
 'ncodpers','ind_empleado','pais_residencia','sexo','age','fecha_alta','ind_nuevo','antiguedad','indrel',
 'indrel_1mes','tiprel_1mes','indresi','indext','canal_entrada','indfall',
 'tipodom','cod_prov','ind_actividad_cliente','renta','segmento']

notuse = ["ult_fec_cli_1t","nomprov"]

product_col = [
 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
 'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1',
 'ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
 'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1',
 'ind_nom_pens_ult1','ind_recibo_ult1']

train_cols = demographic_cols + product_col

# Create trimmed datasets

traindat = traindat.filter(train_cols)
testdat  = testdat.filter(train_cols)


# In[ ]:


#Identify columns with missing data

traindat.isnull().sum()


# Missing value variables can be broken down into three categories:
# 1. Missing data for factor variables: we either impute the most common factor level, as the missing variables are a small subset of the total data, or set to a new 'missing' level if this will imbalance the factor classes.
# 2. Missing data for numerical variables: we can use a more granular imputation, as other kernels have, by setting the missing value equal to the average for each province.
# 3. Missing data for product variables: these are NA as the customers are not eligible to purchase the product. Thus they don't have the product, and their ownership status can be set to 0.

# In[ ]:


#Impute training data

traindat.age = pd.to_numeric(traindat.age, errors='coerce')
traindat.renta = pd.to_numeric(traindat.renta, errors='coerce')
traindat.antiguedad = pd.to_numeric(traindat.antiguedad, errors='coerce')

traindat.loc[traindat['ind_empleado'].isnull(),'ind_empleado'] = 'N'
traindat.loc[traindat['pais_residencia'].isnull(),'pais_residencia'] = 'ES'
traindat.loc[traindat['sexo'].isnull(),'sexo'] = 'V'
traindat.fecha_alta = traindat.fecha_alta.astype('datetime64[ns]')
traindat.loc[traindat['fecha_alta'].isnull(), 'fecha_alta'] = pd.Timestamp(2011,9,1)
traindat.loc[traindat['ind_nuevo'].isnull(), 'ind_nuevo'] = 0
traindat.loc[traindat['indrel'].isnull(), 'indrel'] = 1
traindat.indrel_1mes = traindat.indrel_1mes.astype('str').str.slice(0,1)
traindat.loc[traindat['indrel_1mes'].isnull(), 'indrel_1mes'] = '1'
traindat.loc[traindat['tiprel_1mes'].isnull(), 'tiprel_1mes'] = 'I'
traindat.loc[traindat['indresi'].isnull(), 'indresi'] = 'S'
traindat.loc[traindat['indext'].isnull(), 'indext'] = 'N'
traindat.loc[traindat['canal_entrada'].isnull(), 'canal_entrada'] = 'MIS'
traindat.loc[traindat['indfall'].isnull(), 'indfall'] = 'N'
traindat.loc[traindat['tipodom'].isnull(), 'tipodom'] = 0.0
traindat.loc[traindat['cod_prov'].isnull(), 'cod_prov'] = 28.0
traindat.loc[traindat['ind_actividad_cliente'].isnull(), 'ind_actividad_cliente'] = 0.0
traindat["renta"] = traindat[['renta','cod_prov']].groupby("cod_prov").transform(lambda x: x.fillna(x.mean())) #Replace renta with provincial mean
traindat["age"] = traindat[['age','cod_prov']].groupby("cod_prov").transform(lambda x: x.fillna(x.mean())) #Replace age with provincial mean
traindat["antiguedad"] = traindat[['antiguedad','cod_prov']].groupby("cod_prov").transform(lambda x: x.fillna(x.mean())) #Replace antiguedad with provincial mean
traindat.loc[traindat['segmento'].isnull(), 'segmento'] = '02 - PARTICULARES'
traindat.loc[traindat['ind_nomina_ult1'].isnull(), 'ind_nomina_ult1'] = 0
traindat.loc[traindat['ind_nom_pens_ult1'].isnull(), 'ind_nom_pens_ult1'] = 0

#Impute test data

testdat.age = pd.to_numeric(testdat.age, errors='coerce')
testdat.antiguedad = pd.to_numeric(testdat.antiguedad, errors='coerce')
testdat.renta = pd.to_numeric(testdat.renta, errors='coerce')

testdat.loc[testdat['sexo'].isnull(),'sexo'] = 'V'
testdat.indrel_1mes = testdat.indrel_1mes.astype('str').str.slice(0,1)
testdat.loc[testdat['indrel_1mes'].isnull(), 'indrel_1mes'] = '1'
testdat.loc[testdat['tiprel_1mes'].isnull(), 'tiprel_1mes'] = 'I'
testdat.loc[testdat['canal_entrada'].isnull(), 'canal_entrada'] = 'MIS'
testdat.loc[testdat['cod_prov'].isnull(), 'cod_prov'] = 28.0
testdat.loc[testdat['segmento'].isnull(), 'segmento'] = '02 - PARTICULARES'
testdat["renta"] = testdat[['renta','cod_prov']].groupby("cod_prov").transform(lambda x: x.fillna(x.mean())) #Replace renta with provincial mean
testdat["age"] = testdat[['age','cod_prov']].groupby("cod_prov").transform(lambda x: x.fillna(x.mean())) #Replace age with provincial mean
testdat["antiguedad"] = testdat[['antiguedad','cod_prov']].groupby("cod_prov").transform(lambda x: x.fillna(x.mean())) #Replace antiguedad with provincial mean


# In[ ]:


#Check to make sure all missing data has been filled
traindat.isnull().sum()


# ## Step 2: Feature Engineering
# 
# Feature engineering is one of the most important parts of good Kaggle performance, and in this competition it was vital. Our first step was to bin the continuous variables, so we could treat all variables as binary in the distance matrix, easily scaling demographic and ownership variables.

# In[ ]:


# some more data cleaning

traindat["fecha_alta"] = traindat["fecha_alta"].astype("datetime64")
testdat["fecha_alta"] = testdat["fecha_alta"].astype("datetime64")

# Observation: based on (omitted) EDA, a pre/post 2011 split would make sense for fecha_alta; as credit recovered following the 2008 crash, we may expect to see different user types

# Observation: on a log scale, the salary data is broadly normal. We can take low-medium-high bounds using quartiles

traindat["renta"] = np.log(traindat["renta"])
testdat["renta"] = np.log(testdat["renta"])

# bin the continuous variables

bins_dt = pd.date_range('1994-01-01', freq='16Y', periods=3)
bins_str = bins_dt.astype(str).values
labels = ['({}, {}]'.format(bins_str[i-1], bins_str[i]) for i in range(1, len(bins_str))]

traindat['fecha_alta'] = pd.cut(traindat.fecha_alta.astype(np.int64)//10**9,
                   bins=bins_dt.astype(np.int64)//10**9,
                   labels=labels)

testdat['fecha_alta'] = pd.cut(testdat.fecha_alta.astype(np.int64)//10**9,
                   bins=bins_dt.astype(np.int64)//10**9,
                   labels=labels)


bins_renta = [0,np.percentile(traindat.renta, 25),np.percentile(traindat.renta, 75),25]

traindat['renta'] = pd.cut(traindat.renta,
                   bins=bins_renta)

testdat['renta'] = pd.cut(testdat.renta,
                   bins=bins_renta)


bins_age = [0,25,42,60,1000]
labels_age = ['young','middle','older','old']

traindat['age'] = pd.cut(traindat.age,
                   bins=bins_age,
                   labels=labels_age)

testdat['age'] = pd.cut(testdat.age,
                   bins=bins_age,
                   labels=labels_age)


bins_anti = [-1,220,300]
labels_anti = ['new','old']

#remove negative antiguedad values
traindat.antiguedad[traindat.antiguedad<0] = 0

traindat['antiguedad'] = pd.cut(traindat.antiguedad,
                   bins=bins_anti,
                   labels=labels_anti)

testdat['antiguedad'] = pd.cut(testdat.antiguedad,
                   bins=bins_anti,
                   labels=labels_anti)


# ## Step 3: Data Subsetting 
# Jumping quickly ahead to ** step 3 **: now we have dealt with continuous variables, we can subset the data to only the months of interest. We use June 2015 (and its associated lagged month) in the final model, as well as the lagged May 2016 for the response variable. The data merge that follows breaks the 16GB kernel RAM limit if we don't bring the date subsetting forward.
# 

# In[ ]:


traindat = traindat[traindat.fecha_dato.isin(['2015-05-28','2015-06-28','2016-05-28'])]


# This frees up space, allowing us to finish our feature engineering. The data as presented to us was based on ownership in each month, whereas we are interested in purchase in each month: our second step was to create two new features for each variable, which were ownership in the *previous* month and whether the product was purchased in each month. We used the latter as our response variables, and the former as our main explanatory variables to construct the distance matrix.

# In[ ]:


# similar to a SQL window function, we want to join each user with itself in the previous month. We first sort data based on key columns...
traindat = traindat.sort_values(['ncodpers','fecha_dato'],ascending=[True,True]).reset_index(drop=True)
print('sort completed')

# ...then create a new dataset where the index is incremented...
traindat['new'] = traindat.index
train_index = traindat.copy()
train_index['new'] += 1

# ...then merge the dataset with itself to add each user's purchases in the previous month (there is definitely a quicker way of doing this - I am still relatively new to Python!)
# we rename these new columns with a '_previous' suffix 
merge_drop_cols = demographic_cols.copy()
merge_drop_cols.remove('ncodpers')
traindat_use = pd.merge(traindat,train_index.drop(merge_drop_cols,1), on=['new','ncodpers'],how='left',suffixes=['','_previous'])
print('merge completed')

# replace current with (current - previous) to obtain what we want: purchase indicators
for i in product_col:
    traindat_use[i] = traindat_use[i]-traindat_use[i+"_previous"]
    # replace negative values with 0: if a user gets rid of a product from month x to month x+1, this registers as no purchase in the evaluation metric, so we also treat it as no purchase made
    traindat_use[i][traindat_use[i] < 0] = 0

# fill in na values created by merge
traindat_use[product_col] = traindat_use[product_col].fillna(0)
new_product_col = [i + "_previous" for i in product_col]
traindat_use[new_product_col] = traindat_use[new_product_col].fillna(0)

# delete redundant objects to free up memory
del train_index


# In[ ]:


# We also want to add purchase history columns to the test data set, for the purposes of making predictions

test_col = product_col + ['ncodpers']
testdat_use = pd.merge(testdat,traindat[traindat.fecha_dato=='2016-05-28'][test_col],on='ncodpers',how='left',suffixes=['','_previous'])

testdat_use.rename(
    columns={i:j for i,j in zip(product_col,new_product_col)}, inplace=True
)

testdat_use[new_product_col] = testdat_use[new_product_col].fillna(0)

# delete redundant objects to free up memory
del traindat, testdat
gc.collect()


# ## Step 4: De-duplicate the Data
# Building a similarity matrix for all test users takes a very long time and repeats a lot of work. Instead, we work out the similarity of each ** unique ** test user with all the training users, speeding up computation time enormously. 

# In[ ]:


# pull through variables for memory-based CF

traindat_purchases = traindat_use[traindat_use.fecha_dato == '2015-06-28'][product_col].copy()
traindat_final = traindat_use[traindat_use.fecha_dato == '2015-06-28'][new_product_col].copy()

# pull through variables for demographic-based CF

demog_col = ['sexo','age','fecha_alta','ind_nuevo','indrel','indresi','indfall','tipodom','ind_actividad_cliente']
traindat_demog_final = traindat_use[traindat_use.fecha_dato == '2015-06-28'][demog_col].copy()

# transform demographic factor variables into binary format

sexo_map = {'V': 1,'H': 0}
age_map = {'old': 1,'young': 0}
fecha_alta_map = {'(1994-12-31, 2010-12-31]': 1,'(2010-12-31, 2026-12-31]': 0}
indresi_map = {'S': 1,'N': 0}
indfall_map = {'S': 1,'N': 0}

traindat_demog_final.loc[traindat_demog_final['age']=='older', 'age'] = 'old'
traindat_demog_final.loc[traindat_demog_final['age']=='middle', 'age'] = 'young'
traindat_demog_final.sexo = [sexo_map[item] for item in traindat_demog_final.sexo]
traindat_demog_final.age = [age_map[item] for item in traindat_demog_final.age]
traindat_demog_final.fecha_alta = [fecha_alta_map[item] for item in traindat_demog_final.fecha_alta]
traindat_demog_final.indresi = [indresi_map[item] for item in traindat_demog_final.indresi]
traindat_demog_final.indfall = [indfall_map[item] for item in traindat_demog_final.indfall]

# we want all the observed combinations of purchase history
new_product_col_aug = new_product_col + ['ncodpers']
testdat_final = testdat_use[new_product_col_aug].copy()
testdat_final_unique = testdat_final.drop('ncodpers',1).drop_duplicates().copy().reset_index(drop=True)
#testdat_final_unique.shape
# 6510 unique combinations of purchase history

# transform demographic factor variables into binary format

demog_col_aug = demog_col + ['ncodpers']
testdat_demog_final = testdat_use[demog_col_aug].copy()

testdat_demog_final.loc[testdat_demog_final['age']=='older', 'age'] = 'old'
testdat_demog_final.loc[testdat_demog_final['age']=='middle', 'age'] = 'young'
testdat_demog_final.sexo = [sexo_map[item] for item in testdat_demog_final.sexo]
testdat_demog_final.age = [age_map[item] for item in testdat_demog_final.age]
testdat_demog_final.fecha_alta = [fecha_alta_map[item] for item in testdat_demog_final.fecha_alta]
testdat_demog_final.indresi = [indresi_map[item] for item in testdat_demog_final.indresi]
testdat_demog_final.indfall = [indfall_map[item] for item in testdat_demog_final.indfall] 

testdat_demog_final_unique = testdat_demog_final.drop('ncodpers',1).drop_duplicates().copy().reset_index(drop=True)

#testdat_demog_final_unique.shape
#114 unique combinations of demographics


# We now have everything we need to perform memory-based and demographic-based CF. If we want to run either of these models, we can use all the relevant data to get the strongest possible submission. However, if we want to combine these models, we need to use an evaluation metric to ascertain optimal combination weights. To do this, we can train a first-stage model using 80% of the training data as our 'training' and 20% as our 'test' data, find optimal weights, and run the final model on all the data using these weights.

# In[ ]:


# Split the training data into 'training' and 'test' sets

# create 80% index
traindat_index = np.random.rand(len(traindat_final)) < 0.8
# create traindat_train
traindat_train = traindat_final[traindat_index]
# create traindat_test
traindat_test = traindat_final[~traindat_index]
# make traindat_test unique
traindat_test_unique = traindat_test.drop_duplicates().copy().reset_index(drop=True)
# create traindat_purchases
traindat_purchases_train = traindat_purchases[traindat_index]
# create traindat_purchases_test for verification
traindat_purchases_test = traindat_purchases[~traindat_index]
# create training ncodpers index
traindat_ncodpers = traindat_use[traindat_use.fecha_dato == '2015-06-28'][traindat_index][['fecha_dato','ncodpers']]
traindat_test_ncodpers = traindat_use[traindat_use.fecha_dato == '2015-06-28'][~traindat_index][['fecha_dato','ncodpers']]

# repeat for demographic columns
# create traindat_demog_train
traindat_demog_train = traindat_demog_final[traindat_index]
# create traindat_demog_test
traindat_demog_test = traindat_demog_final[~traindat_index]
# make traindat_demog_test unique
traindat_demog_test_unique = traindat_demog_test.drop_duplicates().copy().reset_index(drop=True)
# purchase indices are the same as for memory-based model data


# ## Step 5: Build Demographic-Based and Memory-Based Similarity Matrices
# A similarity matrix, even on a few thousand unique data points, would be vast, so we iterated over each of the unique test profiles. The steps of the algorithm should be clear in the below code, but at a high level we calculated distances to each training point, weighted training purchases by the inverse distances to obtain a purchase probability, and nullified the purchase probability of owned items.

# In[ ]:


predict_product_col = [i + "_predict" for i in new_product_col]

def probability_calculation(dataset,training,training_purchases,used_columns,metric,test_remap,print_option=False):
    # 'dataset' takes the unique test data with purchase/demographic history; 'training' are the training data that we calculate distances to
    n = dataset.shape[0]
    for index, row in dataset.iterrows():
        if print_option == True:
            print(str(index) + '/' + str(n))
        row_use = row.to_frame().T
        #store purchase history for the test users
        row_history = row_use[used_columns]
        #calculate distances between the test point and each training point based on selected binary features
        #use 'manhattan' when data was binary - when weighted against demographics, use Euclidean
        distances = metrics.pairwise_distances(row_use,training,metric=metric) + 1e-6
        #normalise distances: previously used 24-distances, and 1/(1+distances), but the asymptotic behaviour of 1/distances gives the most accurate predictions.
        norm_distances = 1/distances
        #take dot product between distance to training point and training point's purchase history to obtain ownership likelihood matrix
        sim = pd.DataFrame(norm_distances.dot(training_purchases)/np.sum(norm_distances),columns = new_product_col)
        if(index == 0):
            probabilities = sim
        else:
            probabilities = probabilities.append(sim)
    print("probabilities calculated")
    # reindex users for join
    reindexed_output = probabilities.reset_index().drop('index',axis=1).copy()
    indexed_unique_test = dataset.reset_index().drop('index',axis=1).copy()
    output_unique = indexed_unique_test.join(reindexed_output,rsuffix='_predict')
    output_final = pd.merge(test_remap,output_unique,on=used_columns,how='left')
    # only select relevant products
    output_final = output_final.drop(used_columns,1)
    output_final.columns = output_final.columns.str.replace("_predict", "")
    output_final.columns = output_final.columns.str.replace("_previous", "_predict")
    # now we have all test probabilities - can average and compare with results
    return output_final


# ## Step 6: Combine Demographic-Based and Memory-Based Probabilities
# Using the probabilities derived in step 5, we can derive a weighted average (for 5 candidate weights, ranging from 0-1) to incorporate both data sources into our model. 

# In[ ]:


# calculate memory-based similarities
probabilities_memory = probability_calculation(traindat_test_unique,traindat_train,traindat_purchases_train,new_product_col,'manhattan',traindat_test)
# calculate demographic-based similarities
probabilities_demog = probability_calculation(traindat_demog_test_unique,traindat_demog_train,traindat_purchases_train,demog_col,'manhattan',traindat_demog_test)
# average predictions for a range of mixing probabilities
probabilities_avg_90 = 0.9*probabilities_memory + 0.1*probabilities_demog
probabilities_avg_70 = 0.7*probabilities_memory + 0.3*probabilities_demog
probabilities_avg_50 = 0.5*probabilities_memory + 0.5*probabilities_demog


# Now that we have probabilities, we want to remove the possibility of predicting a product that a user already owns. Nullifying previous purchases can be done at step 5, but we do it here to avoid repeating work.

# In[ ]:


predict_col = [i + "_predict" for i in product_col]
predict_previous_col = predict_col + new_product_col

def purchase_nullifier(probabilities,purchase_history,print_option=False):
    # function to 'nullify' any probabilities that would lead to an owned product being predicted
    # probabilities should have 24 columns with suffix 'predict', purchase_history should have 24 columns with suffix 'previous'
    # join two datasets together
    purchase_history = purchase_history.reset_index(drop=True)
    joined_data = purchase_history.join(probabilities)
    # shrink dataset to deal with large-scale data
    unique_data = joined_data.drop_duplicates().copy().reset_index(drop=True)
    n = unique_data.shape[0]
    print("data joined")
    for index,row in unique_data.iterrows():
        if print_option == True:
            print(str(index) + "/" + str(n))
        row = row.to_frame().T
        # subset dataframe and rename columns for nullification
        row_purchases = row[new_product_col]
        row_purchases.columns = row_purchases.columns.str.replace("_previous","")
        row_probabilities = row[predict_col]
        row_probabilities.columns = row_probabilities.columns.str.replace("_predict","")
        prob_norm = (1-row_purchases).multiply(row_probabilities,axis=0)
        if(index == 0):
            output_norm = prob_norm
        else:
            output_norm = output_norm.append(prob_norm)
    print("nullification complete")
    # duplicate back up to original dataset
    # add columns to enable merge
    output_index = output_norm.reset_index(drop=True)
    prob_predict = output_index.join(unique_data)
    scaled_predict = pd.merge(joined_data,prob_predict,how='left')
    output = scaled_predict[product_col]
    output.columns = output.columns.str.replace("ult1","ult1_predict")
    return output

# can output these probabilities for model averaging with other outputs or cast to predictions for a submission


# In[ ]:


nulled_probabilities_100 = purchase_nullifier(probabilities_memory,traindat_test)
nulled_probabilities_90 = purchase_nullifier(probabilities_avg_90,traindat_test)
nulled_probabilities_70 = purchase_nullifier(probabilities_avg_70,traindat_test)
nulled_probabilities_50 = purchase_nullifier(probabilities_avg_50,traindat_test)
nulled_probabilities_0 = purchase_nullifier(probabilities_demog,traindat_test)


# ## Step 7: Derive Recommendations
# Keeping purchase probabilities rather than simply outputting the recommendations is very useful for further analysis, e.g. model averaging. In this step we use the probabilities to derive recommendations.

# In[ ]:


def probabilities_to_predictions(probabilities,ncodpers,print_option=False):
# ncodpers is a dataframe with two columns: fecha_dato and ncodpers (corresponding to probabilities order)    
    # we make probabilities unique to speed upc calculations
    unique_probabilities = probabilities.drop_duplicates().copy().reset_index(drop=True)
    print(unique_probabilities.shape)
    n = unique_probabilities.shape[0]
    for index, row in unique_probabilities.iterrows():
        if print_option == True:
            print(str(index) + '/' + str(n))
        row_use = row.to_frame().T
        # rank list of product recommendations
        arank = row_use.apply(np.argsort, axis=1)
        ranked_cols = row_use.columns.to_series()[arank.values[:,::-1][:,:7]]
        new_frame = pd.DataFrame(ranked_cols)
        #concatenate all 7 predictions
        recoms = new_frame[0] + ' ' + new_frame[1] + ' ' + new_frame[2] + ' ' + new_frame[3] + ' ' + new_frame[4] + ' ' + new_frame[5] + ' ' + new_frame[6]
        recoms_final = recoms.str.replace('_predict', '', regex=True)
        if(index == 0):
            predictions = recoms_final
        else:
            predictions = predictions.append(recoms_final)
    # merge predictions back to initial indices for full dataset
    mapped_predictions = predictions.to_frame().rename(columns={0:'added_products'}).reset_index(drop=True)
    output_unique = mapped_predictions.join(unique_probabilities)
    output_final = pd.merge(probabilities,output_unique,on=predict_col,how='left')
    # add ncodpers for final submission file
    no_index_ncodpers = ncodpers.copy().reset_index(drop=True)
    output_ncodpers = no_index_ncodpers.join(output_final['added_products']).drop('fecha_dato',axis=1)
    return output_ncodpers


# In[ ]:


predictions_output_100 = probabilities_to_predictions(nulled_probabilities_100,traindat_test_ncodpers)
predictions_output_90 = probabilities_to_predictions(nulled_probabilities_90,traindat_test_ncodpers)
predictions_output_70 = probabilities_to_predictions(nulled_probabilities_70,traindat_test_ncodpers)
predictions_output_50 = probabilities_to_predictions(nulled_probabilities_50,traindat_test_ncodpers)
predictions_output_0 = probabilities_to_predictions(nulled_probabilities_0,traindat_test_ncodpers)


# The final piece of the puzzle is an evaluation metric, to ascertain the performance of each model (i.e. each mixing probability).

# In[ ]:


evaluation_col = product_col + ['added_products']

def evaluation_metric(predictions,reality,print_option=False):
    # predictions is a list of the top seven purchase likelihood indicators; reality is the actual purchases
    reality = reality.reset_index(drop=True)
    # find unique combinations to speed up function: merge data, group_by, count (then multiply results at the end)
    reality['added_products'] = predictions['added_products']
    data_unique = reality.drop_duplicates().copy().reset_index(drop=True)
    predictions_unique = data_unique['added_products'].to_frame()
    reality_unique = data_unique.drop('added_products',1)
    n = predictions_unique.shape[0]
    for index, row in predictions_unique.iterrows():
        if print_option == True:
            print(str(index) + '/' + str(n))
        prediction_use = row.to_frame().T['added_products'].str.split(' ',expand=True).T
        prediction_use = prediction_use.rename(columns={list(prediction_use)[0]:'predict_products'})
        #print(prediction_use)
        # only take top 7 products purchased
        reality_use = reality_unique.iloc[index].to_frame()
        reality_use = reality_use.rename(columns={list(reality_use)[0]:'added_products'})
        reality_use['product_name'] = reality_use.index
        reality_use = reality_use[reality_use.added_products==1]
        reality_use['ind'] = 1
        #print(reality_use)
        if reality_use.empty:
            P = [0]
        else:
            # calculate precision @7: what average proportion of our predictions are purchased?
            P = [precision_at_k(prediction_use,reality_use)]
        if index == 0:
            eval_sum = P
        else:
            eval_sum.extend(P)
    # duplicate back up
    print('precisions calculated')
    data_unique['precision'] = eval_sum
    reality_final = pd.merge(reality,data_unique,on=evaluation_col,how='left')
    U = predictions.shape[0]
    output = sum(reality_final.precision)/U
    return output

def precision_at_k(prediction,reality):
    # 'prediction' is a data frame with a column 'predict_products' containing our 7 predictions
    # 'reality' is a data frame with a column 'added_products' containing any products purchased (always non-empty)
    summand = min(prediction.shape[0],7)
    sum_prec = 0
    for k in range(summand):
        # for each k, calculate precision at k (careful with 0 index)
        top_k_predictions = prediction.head(k+1)
        # join additions to reduced predictions
        add_vs_pred = pd.merge(reality,top_k_predictions,left_on='product_name',right_on='predict_products',how='inner')
        sum_prec = sum_prec + sum(add_vs_pred.ind)/top_k_predictions.shape[0]
    denom = min(reality.shape[0],7)
    # always defined as in evaluation_metric function 'reality_use' is always non-empty
    output = sum_prec/denom
    return output 


# In[ ]:


evaluation_100 = evaluation_metric(predictions_output_100,traindat_purchases_test)
evaluation_90 = evaluation_metric(predictions_output_90,traindat_purchases_test)
evaluation_70 = evaluation_metric(predictions_output_70,traindat_purchases_test)
evaluation_50 = evaluation_metric(predictions_output_50,traindat_purchases_test)
evaluation_0 = evaluation_metric(predictions_output_0,traindat_purchases_test)


# In[ ]:


print("all memory: " + str(evaluation_100) + '\n' + 
      "90% memory: " + str(evaluation_90) + '\n' + 
      "70% memory: " + str(evaluation_70) + '\n' + 
      "50% memory: " + str(evaluation_50) + '\n' + 
      "all demographics: " + str(evaluation_0))


# ## Step 8: Re-run model using all training data for optimal mixing parameter
# 90% memory gives the strongest results: the optimum seems to lie somewhere around 85%. Now that we have an optimal mixing probability, we can iterate through the entire process, using all of the training data to return our optimal prediction that leverages both demographic and purchase data.

# In[ ]:


# calculate probabilities
probability_85_memory = probability_calculation(testdat_final_unique,traindat_final,traindat_purchases,new_product_col,'manhattan',testdat_final)
probability_85_demog = probability_calculation(testdat_demog_final_unique,traindat_demog_final,traindat_purchases,demog_col,'manhattan',testdat_demog_final)

# average probabilities
probability_avg_85 = 0.85*probability_85_memory + 0.15*probability_85_demog

# write csv of averaged probabilities
probability_avg_85.to_csv("probabilities_85_avg.csv",index=False)

# null previous ownership
nulled_probability_85 = purchase_nullifier(probability_avg_85[predict_col],testdat_final[new_product_col])

# map to predictions - check dimensions
testdat_ncodpers = testdat_use[['fecha_dato','ncodpers']]
predictions_output_85 = probabilities_to_predictions(nulled_probability_85,testdat_ncodpers)

# send predictions to csv
predictions_output_85.to_csv('submission.csv',index=False)


# The submission in its current form has a private score of 0.02644 and a public score of 0.02619, which is just outside the top 50% of scores. Not bad with no model-based input! This compares to respective scores of 0.02633 and 0.02610 when no mixing probabilities are used (i.e. we only use memory-based cf); a small improvement, but significant in a Kaggle context.

# ## Next Steps
# This notebook is very much intended as an introduction to recommendation systems. There are several lines of analysis that could be pursued to improve upon this model's score:
# * ** Test different distance metrics and demographic weightings: ** in our final algorithm we use a very straightforward distance metric, and weight purchases by taking the inverse sum of this straightforward metric. There are improvements to be made by sharpening up this methodology.
# * ** Include item-based CF and average results: ** the products are real, and we can derive product features manually. This could help us to build an item-based CF approach, which could be combined with our algorithmically-defined memory-based probabilities to enhance to model. 
# * ** Try model-based CF: ** memory-based CF is the most straightforward technique, and it will be interesting to compare the performance of these models with the performance of out memory-based algorithm.
# 
