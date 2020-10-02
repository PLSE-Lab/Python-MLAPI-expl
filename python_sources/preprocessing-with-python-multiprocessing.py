#!/usr/bin/env python
# coding: utf-8

# # Preprocessing with Multiprocessing
# 
# This kernel uses the Python multiprocessing module to make use of all 4 cores you get in a Kaggle CPU kernel. The input data is split into four chunks, and each is processed using the given feature extraction function in parallel. The results are stored on disk in both scaled and non-scaled format. Output compression is used the avoid kernel output size limits.
# 
# Kaggle CPU kernels have 4 CPU cores, while GPU kernels have only 2 CPU cores. Running pre-processing in a separate kernel like this helps use both kernel types more optimally. CPU kernel with more cores to pre-process, GPU kernel to build and experiment with models using the CPU kernel output as data source.
# 
# The features in this version base on the ones in https://www.kaggle.com/braquino/5-fold-lstm-attention-fully-commented-0-694, and a few lag/diff ones I played with. Should be simple to tune for any other features/processing.
# 
# All the features are created in function "summarize_df_np". Change that to produce different features. 
# 
# This kernel produces a set of output files as follows:
# 
# - my_train.csv.gz: The raw data as processed (features, buckets, whatever you call them). Each signal separately in 160 rows per signal. "raw" as in not scaled.
# - my_train_scaled.csv.gz: The same data as my_train.csv.gz, scaled using min-max-scaler with -1 to 1 scale.
# - my_train_combined_scaled.csv.gz: The scaled data but all 3 signals per measurement on a single row.
# - my_test.csv.gz: Same as above but for test data.
# - my_test_scaled.csv.gz: Same as above but for test data.
# - my_test_combined_scaled.csv.gz: Same as above but for test data.
# 

# In[ ]:


import pandas as pd
import pyarrow.parquet as pq
import os
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


train_meta = pd.read_csv("../input/metadata_train.csv")
#train_meta.head(6)


# In[ ]:


test_meta = pd.read_csv("../input/metadata_test.csv")
#test_meta.head(6)


# ## Feature Generation

# The function to process all the chunks and generate features in the parallel running processes:

# In[ ]:


#I use bkt as short for bucket, rather than bin for bin since I tend to read bin as binary
bkt_count = 160
data_size = 800000
bkt_size = int(data_size/bkt_count)

def summarize_df_np(meta_df, data_type, p_id):
    count = 0
    measure_rows = []

    for measurement_id in meta_df["id_measurement"].unique():
        count += 1
        idx1 = measurement_id * 3
        input_col_names = [str(idx1), str(idx1+1), str(idx1+2)]
        df_sig = pq.read_pandas('../input/'+data_type+'.parquet', columns=input_col_names).to_pandas()
        df_sig = df_sig.clip(upper=127, lower=-127)

        df_diff = pd.DataFrame()
        for col in input_col_names:
            df_diff[col] = df_sig[col].diff().abs()
        
        data_measure = df_sig.values
        data_diffs = df_diff.values
        sig_rows = []
        sig_ts_rows = []
        for sig in range(0, 3):
            #take the data for each 3 signals in a measure separately
            data_sig = data_measure[:,sig]
            data_diff = data_diffs[:,sig]
            bkt_rows = []
            diff_avg = np.nanmean(data_diff)
            for i in range(0, data_size, bkt_size):
                # cut data to bkt_size (bucket size)
                bkt_data_raw = data_sig[i:i + bkt_size]
                bkt_avg_raw = bkt_data_raw.mean() #1
                bkt_sum_raw = bkt_data_raw.sum() #1
                bkt_std_raw = bkt_data_raw.std() #1
                bkt_std_top = bkt_avg_raw + bkt_std_raw #1
                bkt_std_bot = bkt_avg_raw - bkt_std_raw #1

                bkt_percentiles = np.percentile(bkt_data_raw, [0, 1, 25, 50, 75, 99, 100]) #7
                bkt_range = bkt_percentiles[-1] - bkt_percentiles[0] #1
                bkt_rel_perc = bkt_percentiles - bkt_avg_raw #7

                bkt_data_diff = data_diff[i:i + bkt_size]
                bkt_avg_diff = np.nanmean(bkt_data_diff) #1
                bkt_sum_diff = np.nansum(bkt_data_diff) #1
                bkt_std_diff = np.nanstd(bkt_data_diff) #1
                bkt_min_diff = np.nanmin(bkt_data_diff) #1
                bkt_max_diff = np.nanmax(bkt_data_diff) #1

                raw_features = np.asarray([bkt_avg_raw, bkt_std_raw, bkt_std_top, bkt_std_bot, bkt_range])
                diff_features = np.asarray([bkt_avg_diff, bkt_std_diff, bkt_sum_diff])
                bkt_row = np.concatenate([raw_features, diff_features, bkt_percentiles, bkt_rel_perc])
                bkt_rows.append(bkt_row)
            sig_rows.extend(bkt_rows)
        measure_rows.extend(sig_rows)
    df_sum = pd.DataFrame(measure_rows)
    #df_sum = df_sum.astype("float32")
    return df_sum


# Function process_subtrain() is passed to the Python multiprocessing for all four cores/chunks. 
# 
# It calls the feature processing function for the training data:

# In[ ]:


def process_subtrain(arg_tuple):
    meta, idx = arg_tuple
    df_sum = summarize_df_np(meta, "train", idx)
    return idx, df_sum


# The scaler to produce the scaled version on of the data once all four parallel chunks have finished processing:

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler(feature_range=(-1,1))


# ## Multiprocessing

# Function to create the chunks sizes/indices to split the data into chunks. Used for both train and test data:

# In[ ]:


def create_chunk_indices(meta_df, chunk_idx, chunk_size):
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size
    meta_chunk = meta_df[start_idx:end_idx]
    print("start/end "+str(chunk_idx+1)+":" + str(start_idx) + "," + str(end_idx))
    print(len(meta_chunk))
    #chunk_idx in return value is used to sort the processed chunks back into original order,
    return (meta_chunk, chunk_idx)


# ## Training dataset processing

# Actual code to call multiprocessing for the training data:

# In[ ]:


from multiprocessing import Pool

num_cores = 4

def process_train():
    #splitting here by measurement id's to get all signals for a measurement into single chunk
    measurement_ids = train_meta["id_measurement"].unique()
    df_split = np.array_split(measurement_ids, num_cores)
    chunk_size = len(df_split[0]) * 3
    
    chunk1 = create_chunk_indices(train_meta, 0, chunk_size)
    chunk2 = create_chunk_indices(train_meta, 1, chunk_size)
    chunk3 = create_chunk_indices(train_meta, 2, chunk_size)
    chunk4 = create_chunk_indices(train_meta, 3, chunk_size)

    #list of items for multiprocessing, 4 since using 4 cores
    all_chunks = [chunk1, chunk2, chunk3, chunk4]
    
    pool = Pool(num_cores)
    #this starts the (four) parallel processes and collects their results
    #-> process_subtrain() is called concurrently with each item in all_chunks 
    result = pool.map(process_subtrain, all_chunks)
    #parallel processing can be non-deterministic in timing, so here I sort results by their chunk id
    #to maintain results in same order as in original files (to match metadata from other file)
    print("sorting")
    result = sorted(result, key=lambda tup: tup[0])
    print("sorted")
    sums = [item[1] for item in result]
    
    df_train = pd.concat(sums)
    df_train = df_train.reset_index(drop=True)
    #np.save() would be another option but this works for now
    df_train.to_csv("my_train.csv.gz", compression="gzip")

    df_train_scaled = pd.DataFrame(minmax.fit_transform(df_train))
    df_train_scaled.to_csv("my_train_scaled.csv.gz", compression="gzip")
    return df_train, df_train_scaled


# In[ ]:


ps = process_train()


# ## Training dataset processing validation

# And some brief look at the data itself, to see it is valid:

# In[ ]:


#first 10 rows of raw feature data
ps[0].head(10)


# In[ ]:


#same first 10 rows in scaled format
ps[1].head(10)


# In[ ]:


ps[1].values.shape


# The above shows the shape of the generated data. In this case we have 22 features, so 22 columns. 
# 
# 160 rows per signal (one "bucket" per row) so overall size matches:

# In[ ]:


bkt_count*len(train_meta)


# The feature processing function in this kernel generates these features:
# * 5 for general bucket/bin statistics: bkt_mean, bkt_std, bkt_std_top, bkt_std_bottom, bkt_range
# * 3 for diff/lag in bucket/bin: bkt_diff_mean, bkt_diff_std, bkt_diff_sum
# * 7 percentiles: 0, 1, 25, 50, 75, 99, 100
# * 7 relative percentiles: 0, 1, 25, 50, 75, 99, 100
# -> total of 22 "features"
# 
# To show a bit how they all look together after scaled to range -1 to 1, a look at the first signal as processed into 160 buckets from 800k:

# In[ ]:


ps[1][0:160].plot(figsize=(8,5))


# All that is bit of a mess, so just the first bullet on its own -
# * 5 for general bucket/bin statistics: bkt_mean, bkt_std, bkt_std_top, bkt_std_bottom, bkt_range

# In[ ]:


ps[1].iloc[:,0:5][0:160].plot()


# Next the diff features:
# * 3 for diff/lag in bucket/bin: bkt_diff_mean, bkt_diff_std, bkt_diff_sum

# In[ ]:


ps[1].iloc[:,5:8][0:160].plot()


# Sum and average are overlapping, which is why only 2 lines show. So scaled average is the same as a scaled sum in this case. Should probably look at the othe features more closely as well, but that would be another story.

# Next the percentiles:
# * 7 percentiles: 0, 1, 25, 50, 75, 99, 100

# In[ ]:


ps[1].iloc[:,8:15][0:160].plot()


# And the relative percentiles:
# * 7 relative percentiles: 0, 1, 25, 50, 75, 99, 100

# In[ ]:


ps[1].iloc[:,15:22][0:160].plot()


# ## Combined dataset from above single-signal data

# The above showed an example for one signal with all the 22 features total.
# 
# As another dataset, I combine for each measurement id, the 3 signals into one row as features.
# 
# This allows running models where all 3 signals for a measirement id are treated as unified features, such as in the kernel I linked at the beginning.
# 
# ## Training-dataset combining:
# 
# The combine 3 signals code for the training dataset:

# In[ ]:


measurement_ids = train_meta["id_measurement"].unique()
rows = []
for mid in measurement_ids:
    idx1 = mid*3
    idx2 = idx1 + 1
    idx3 = idx2 + 1
    sig1_idx = idx1 * bkt_count
    sig2_idx = idx2 * bkt_count
    sig3_idx = idx3 * bkt_count
    sig1_data = ps[1][sig1_idx:sig1_idx+bkt_count]
    sig2_data = ps[1][sig2_idx:sig2_idx+bkt_count]
    sig3_data = ps[1][sig3_idx:sig3_idx+bkt_count]
    #this combines the above read 3*160 rows for 3 signals into 1 combined set with with 160 rows
    #and from 22 features on 3*160 to 66 (=22*3) features on 160 rows.
    row = np.concatenate([sig1_data, sig2_data, sig3_data], axis=1).flatten().reshape(bkt_count, sig1_data.shape[1]*3)
    rows.append(row)
df_train_combined = pd.DataFrame(np.vstack(rows))
df_train_combined.to_csv("my_train_combined_scaled.csv.gz", compression="gzip")


# ### Verification

# For verification, a look at the results to check the signal combination has worked:
# 
# First the 2 first signals in the single signal dataset:

# In[ ]:


#slot 1 (measurement 1, signal 1, rows 0-159) for single signal version
ps[1].iloc[:,15:22][0:160].plot()


# In[ ]:


#slot 2 (measurement 1, signal 2, rows 160-319) for single signal version
ps[1].iloc[:,15:22][160:320].plot()


# Now the same 2 signals in the combined dataset:

# In[ ]:


#slot 1 (measurement 1, signal 1) for combined signal version
df_train_combined.iloc[:,15:22][0:160].plot()


# In[ ]:


#slot 2 (measurement 1, signal 2) for combined signal version
df_train_combined.iloc[:,37:44][0:160].plot()


# One more, a look at the data contents to check also the values:

# In[ ]:


#signal 1, single signal version
ps[1].iloc[0:4]


# In[ ]:


#signal 2, single signal version
ps[1].iloc[160:164]


# In[ ]:


#signal 3, single signal version
ps[1].iloc[320:324]


# In[ ]:


#signal 4 (or signal 1 for measurement id 2))
ps[1].iloc[480:484]


# For comparison, the signals 1-3 for measurement id 1 in combined set:
# 
# With 22 features, signal 1 from above should be in columns 0-21, signal 2 in columns 22-43, and signal 3 in columns 44-65.

# In[ ]:


df_train_combined.iloc[0:4]


# And the second bucket at 160-320 rows should have signal 4 (or signal 1 for measurement id 2) from above, in columns 0-21:

# In[ ]:


df_train_combined.iloc[160:164]


# All the above checks match, so I judge this as working as intended.
# 
# Since this should now all be saved to disk, clean up some memory:

# In[ ]:


del ps
del df_train_combined


# ## Test-dataset Processing and Features

# Now, the same multiprocessing elements for the test set as above for the training set:

# In[ ]:


def process_subtest(arg_tuple):
    meta, idx = arg_tuple
    df_sum = summarize_df_np(meta, "test", idx)
    return idx, df_sum


# In[ ]:


from multiprocessing import Pool

num_cores = 4

def process_test():
    measurement_ids = test_meta["id_measurement"].unique()
    df_split = np.array_split(measurement_ids, num_cores)
    chunk_size = len(df_split[0]) * 3
    
    chunk1 = create_chunk_indices(test_meta, 0, chunk_size)
    chunk2 = create_chunk_indices(test_meta, 1, chunk_size)
    chunk3 = create_chunk_indices(test_meta, 2, chunk_size)
    chunk4 = create_chunk_indices(test_meta, 3, chunk_size)

    all_chunks = [chunk1, chunk2, chunk3, chunk4]
    
    pool = Pool(num_cores)
    result = pool.map(process_subtest, all_chunks)
    result = sorted(result, key=lambda tup: tup[0])

    sums = [item[1] for item in result]

    df_test = pd.concat(sums)
    df_test = df_test.reset_index(drop=True)
    df_test.to_csv("my_test.csv.gz", compression="gzip")

    df_test_scaled = pd.DataFrame(minmax.transform(df_test))
    df_test_scaled.to_csv("my_test_scaled.csv.gz", compression="gzip")
    return df_test, df_test_scaled


# In[ ]:


pst = process_test()


# In[ ]:


pst[0].head(10)


# In[ ]:


pst[1].head(10)


# In[ ]:


pst[1].values.shape


# ## A look at the processed test data

# All-in-one figure for the mess:

# In[ ]:


pst[1][0:160].plot()


# * The 5 general bucket/bin statistics:

# In[ ]:


pst[1].iloc[:,0:5][0:160].plot()


# * The 3 for diff/lag in bucket/bin:

# In[ ]:


pst[1].iloc[:,5:8][0:160].plot()


# * The 7 percentiles:

# In[ ]:


pst[1].iloc[:,8:15][0:160].plot()


# * The 7 relative percentiles:

# In[ ]:


pst[1].iloc[:,15:22][0:160].plot()


# ## Test-dataset combinations

# Similarly, combine 3 signals into one row as features for the combined version for test data:

# In[ ]:


measurement_ids = test_meta["id_measurement"].unique()
start = measurement_ids[0]
rows = []
for mid in measurement_ids:
    #test measurement id's start from 2904 and indices at 0, so need to align
    mid = mid - start
    idx1 = mid*3
    idx2 = idx1 + 1
    idx3 = idx2 + 1
    sig1_idx = idx1 * bkt_count
    sig2_idx = idx2 * bkt_count
    sig3_idx = idx3 * bkt_count
    sig1_data = pst[1][sig1_idx:sig1_idx+bkt_count]
    sig2_data = pst[1][sig2_idx:sig2_idx+bkt_count]
    sig3_data = pst[1][sig3_idx:sig3_idx+bkt_count]
    row = np.concatenate([sig1_data, sig2_data, sig3_data], axis=1).flatten().reshape(bkt_count, sig1_data.shape[1]*3)
    rows.append(row)
df_test_combined = pd.DataFrame(np.vstack(rows))
df_test_combined.to_csv("my_test_combined_scaled.csv.gz", compression="gzip")


# ### A look at combined test-dataset

# Brief look at the combined test set signals to see it makes sense:

# In[ ]:


pst[1].iloc[0:4]


# vs.

# In[ ]:


df_test_combined.head(4)


# Check shape of combined signals to match number of unique measurements in test data:

# In[ ]:


df_test_combined.shape


# ## Importing the Results to Other Kernels

# To use the kernel output as a dataset in another (e.g., GPU) kernel, create a kernel and select "+ Add Data" on the right side panel.

# In[ ]:





# In[ ]:




