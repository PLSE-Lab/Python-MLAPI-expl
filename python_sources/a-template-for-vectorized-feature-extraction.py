#!/usr/bin/env python
# coding: utf-8

# 

# ### Intro
# 
# Looking through the kernels in the competition I noticed most of the kernels were using an iterative way to calculate features. So I'm sharing a template for vectorizing the feature extraction to improve the performance. Hope it would be useful for you. 

# In[ ]:


import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}) ")


# ### The iterative way 
# As you might have seen in many of the kernels, one way of extracting features is to calculate segments of data and then iterate through each segment to calculate the features. Here I put a simple example (to extract only 6 features) 

# In[ ]:


# Segmenting 
rows = 150_000
segments = int(np.floor(train_df.shape[0] / rows))
print("Number of segments: ", segments)


# In[ ]:


def create_features(seg_id, seg, X):
    xc = pd.Series(seg['acoustic_data'].values)
    
    X.loc[seg_id, 'mean'] = xc.mean()
    X.loc[seg_id, 'std'] = xc.std()
    X.loc[seg_id, 'max'] = xc.max()
    X.loc[seg_id, 'min'] = xc.min()
    X.loc[seg_id, 'sum'] = xc.sum()
    X.loc[seg_id, 'median'] = xc.median()
    
    
    return X


# In[ ]:


train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])


# In[ ]:


get_ipython().run_cell_magic('time', '', "for seg_id in tqdm_notebook(range(segments)):\n    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]\n    create_features(seg_id, seg, train_X)\n    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]")


# ### The vectorize way
# So let's take a look at anothe way of handling the featur extraction. 
# 
# We all know that Numpy is known for its performance in handling matrices, so let's leverage it to calcualte the features.
# 
# In this project, we only have one input parameter (acoustic data) and it makes it much easier to prepare the input matrix. 
# The idea is very seimple, we trim the input data to be dividable by our desired number rows (in this case 150,000) and then turn the 1D matrix into a 2D matrix and calculate features amongs its vertical axis. 
# 
# So assuming our input data has **n** rows,  so our input matrix is (n x 1), we make it (m x 1) where  m = n -  n mod 150,000 and then reshape it to (150,000, x) where x = m / 150,000 
# 
# Now we can calculate features amongst the axis =0 of the new matrix. 
# 

# In[ ]:


# This function makes sure the input matrix is dividable by the target number of rows
def prep_df_for_separation(df,rows):
    mod_value = df.shape[0] % rows 
    if mod_value > 0:
        lastRow = df.shape[0] - mod_value
        df = df.iloc[:lastRow]
    return df 


# The following function is a very simple example of vectorized feature extraction. We know the input matrix shape is (rows, x) and we want to calculate features for each row so our output can be (x, n_features) 

# In[ ]:


# an example of vactorized feature exraction function
def vectorized_features(data):
    n_features = 6
    output_matrix = np.empty(shape=(data.shape[1], n_features))

    output_matrix[:,0] = np.mean(data,axis=0)
    output_matrix[:,1] = np.std(data,axis=0)
    output_matrix[:,2] = np.max(data,axis=0)
    output_matrix[:,3] = np.min(data,axis=0)
    output_matrix[:,4] = np.sum(data,axis=0)
    output_matrix[:,5] = np.median(data,axis=0)
    
    return output_matrix


# we use *prep_df_for_separation* function we defined earlier to prepare our train data frame and then separate x and y and process them separately. 
# 
# For y, in this case we know the time_to_failure is reducing during each 150,000 section, so we can simply use min function to get the desired value (notice we used axis= 0) 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = prep_df_for_separation(train_df,rows)\ndata_matrix = train_df.acoustic_data.values.reshape(-1,rows).T\noutput_matrix_all = train_df.time_to_failure.values.reshape(-1,rows).T\noutput_matrix = np.min(output_matrix_all,axis=0)\n\nprint("data matrix shape", data_matrix.shape)\nprint("output matrix shape", output_matrix.shape)')


# Finally we can calculate features. You can compare its process time with the iterative method above and consider this is only for 6 features and if you would incrase the number of features, the difference woud be much more significant. 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'features = vectorized_features(data_matrix)\nprint("data matrix shape", data_matrix.shape, "\\t| Output matrix shape:", output_matrix.shape)')


# ### Conclusion
# Some of the features are a bit harder to calculate using this method. Moving averag for instance, is possible but requires forming a larger matrix which consumes memory. I'd add more feature ideas later on if requested. 

# In[ ]:



