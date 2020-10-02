#!/usr/bin/env python
# coding: utf-8

# The training ata for this competition is provided as a several hundred million line CSV with two columns. I find this a rather awkward data format to deal with. 
# 
# Simply attempting to load the csv via pandas read_csv takes a long time and isn't very memory efficient. When beginning to experiment with this data in kaggle kernels the pd.read_csv caused a dead kernel when the compute instance ran out of memory. Instead of suffering through a long running custom python loop to do all my feature engineering I have decided to reformat the training data to make the loading faster, more memory efficient and formatted in a way that more closely mimics the way the data is used to generate predictions on the testing segments. 
# 
# Since the "time_to_failure" column varies only very slowly it really needn't be provided for every single row separately, this is doubly true since we have been asked to predict only one such time per stretch of 150,000 acoustic data points in the test segements. By storing just one time_to_failure value per 150,000 training data rows we can save roughly a factor of 2 in memory usage right away, not to mention the extra convenience of having our training and testing data in a similar format. 
# 
# The acoustic data has a maximum dynamic range which is small enough to allow us to represent it with arrays of type int16 which saves us a factor of 4 in terms of memory footprint versus storing the data as int64 (this by itself was not enough to. 
# 
# Finally we can store the data out in the hdf5 data format which will dramatically improve the time required to load the data from disk (roughly 1,000x speedup relative to a pd.read_csv call).
# 
# You can use this alternate format for the training data in your kernels by adding the output of this kernel as an additional data source (see https://www.kaggle.com/product-feedback/45472 )

# In[ ]:


import os
import time
import h5py
import numpy as np
import pandas as pd


# count the number of lines in the training file.

# In[ ]:


get_ipython().system('wc -l ../input/train.csv')


# In[ ]:


n_lines_train = 629145481


# In[ ]:


chunk_size = 150000
n_segments = (n_lines_train-1)//chunk_size
n_segments


# In[ ]:


leftover = n_lines_train - n_segments*chunk_size
leftover


# In[ ]:


leftover/n_lines_train


# Unfortunately the training data doesn't neatly divide into an integer number of the test segment size chunks. But when dealing with 600+ million time points a few tens of thousand more or less probably won't make much of a difference (it makes up just one ten thousandth part of the training data). So I will just ignore the last few training data time points. 

# In[ ]:


input_dir = "../input"
output_dir = ""


# we create the hdf5 file and then will create named datasets within the file and then iterate over the csv and fill in the dataset rows one by one. We also could have first loaded the data into numpy arrays and then simply assigned the arrays directly out to the hdf5 file which is often a more convenient interface, but doing it this way allows us to deal with files which are much too large to fit directly into memory.

# In[ ]:


#create the hdf5 file
h5_file = h5py.File(os.path.join(output_dir, "train.h5"), "w")


# In[ ]:


#create datasets within the top level group in that file
sound_dset = h5_file.create_dataset("sound", shape=(n_segments, chunk_size), dtype=np.int16)
ttf_dset = h5_file.create_dataset("ttf", shape=(n_segments,), dtype=np.float32)


# In[ ]:


#iterate over all 629 million lines and save them out in chunks of 150,000
#this takes a while ...
chunk_size = 150000
lines_to_read = chunk_size*n_segments

printed_warning = False
with open(os.path.join(input_dir, "train.csv")) as f:
    x_stack, y_stack = [], []
    last_ttf = np.inf
    hdr = f.readline()
    for line_idx in range(lines_to_read):
        cx, cy = f.readline().split(",")
        cx = int(cx)
        if np.abs(cx) > 32767:
            if not printed_warning:
                printed_warning = True
                print("line {} is too big to be an int16".format(line_idx+1))
        cy = float(cy)
        if cy < last_ttf:
            last_ttf = cy
            y_stack.append(cy)
        x_stack.append(cx)
        if line_idx % chunk_size == chunk_size-1:
            sound_dset[line_idx//chunk_size] = np.array(x_stack).astype(np.int16)
            ttf_dset[line_idx//chunk_size] = np.mean(y_stack)
            x_stack, y_stack = [], []
            last_ttf = np.inf


# In[ ]:


h5_file.close()


# Now that we have a nice hdf5 file lets do some comparisons versus the csv file. First off the h5 file on disk takes up only 1.2 Gb versus 8.9 Gb for the uncompressed csv. Most of this difference can be attributed to the fact that the time_to_failure column takes a lot of characters to express as a character string and we are storing just 1/150,000th as many numbers and in a binary format. 

# What about the time to read in the data from disk? Lets load the first 100 segments from our hdf5 file and compare that to using pandas.read_csv on the raw data.

# In[ ]:


start_time = time.time()
hf = h5py.File(os.path.join(output_dir, "train.h5"))
segs = np.array(hf["sound"][:100])
seg_ttf = np.array(hf["ttf"][:100])
hf.close()
end_time = time.time()
print("{} seconds".format(end_time-start_time))


# now compare this with the time to read in the data from the csv using pandas.

# In[ ]:


start_time = time.time()
df = pd.read_csv(
    os.path.join(input_dir, "train.csv"), 
    nrows=chunk_size*100,#limit to the first 100 segments 
    dtype={"acoustic_data":np.int16, "time_to_failure":np.float32}
)
end_time = time.time()
print("{} seconds".format(end_time-start_time))


# After putting the data into the hdf5 file the read time is fast, the memory usage is reduced and the training data is neatly formatted into 150,000 length segments and single time labels just like at testing time. 
# 
# To maximize convenience we can also turn the test segment data into a hdf5 file with the same format. 

# In[ ]:


test_files = os.listdir(os.path.join(input_dir, "test"))

with h5py.File(os.path.join(output_dir, "test.h5"), "w") as h5_file:
    sound_dset = h5_file.create_dataset("sound", (len(test_files), chunk_size))
    seg_ids = []

    for fidx, fname in enumerate(test_files):
        cdata = pd.read_csv(os.path.join(input_dir, "test", fname), dtype=np.int16)["acoustic_data"].values
        sound_dset[fidx] = cdata
        seg_ids.append(fname.split(".")[0])

    h5_file["seg_id"] = np.array(seg_ids).astype(np.string_)


# While HDF5 is fantastic for storing numerical data it can be somewhat painful to use for storing text data. Unfortunately HDF5 likes storing string arrays as ascii and numpy likes string arrays to be either of object type, fixed length ascii (which works for hdf5 just fine) or unicode (which doesn't work with hdf5). 
# 
# So when we write out the seg_id's we need to use np.string_ data format which will encode the strings as ascii. Then when we load the seg_id's back in we need to explicitly cast the numpy array back to the "str" type so that python will treat the resulting array as strings instead of as type "bytes". 

# In[ ]:


#loading back in the test data segment strings
hf = h5py.File(os.path.join(output_dir, "test.h5"))


# In[ ]:


#without a .astype(str) the resulting array is of type bytes
seg_ids = np.array(hf["seg_id"])
seg_ids


# In[ ]:


#adding a astype call gets us nice unicode strings
#that play well with python 3
seg_ids = np.array(hf["seg_id"]).astype(str)
seg_ids


# In[ ]:


hf.close()


# Happy kaggling.
