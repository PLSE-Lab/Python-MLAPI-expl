#!/usr/bin/env python
# coding: utf-8

# # Objective
# Import the FAF freight matrices provided with FAF into AequilibraE's matrix format
# 
# aequilibrae https://github.com/AequilibraE/aequilibrae-examples/
# 
# http://aequilibrae.com/
# 
# ## Input data
# 
# * FAF: https://faf.ornl.gov/fafweb/
# * Matrices: https://faf.ornl.gov/fafweb/Data/FAF4.4_HiLoForecasts.zip
# * Zones System: http://www.census.gov/econ/cfs/AboutGeographyFiles/CFS_AREA_shapefile_010215.zip
# * FAF User Guide: https://faf.ornl.gov/fafweb/data/FAF4%20User%20Guide.pdf
# * The blog post (with data): http://www.xl-optim.com/matrix-api-and-multi-class-assignment

# # The code
# We import all libraries we will need, including the AequilibraE, after putting it in our Python path 

# In[95]:


import sys
# On Linux
# sys.path.append('/home/pedro/.qgis2/python/plugins/AequilibraE')
# On Windows
sys.path.append("../input/")
import pandas as pd
import numpy as np
import os

path = "../input/" # specify directory where are the files 

fileList = os.listdir(path) # get file list in the path directory
# list files
for f in fileList: 
    print(f)


# Importing aequilibrae

# In[97]:


get_ipython().system('pip install aequilibrae')


# In[99]:


import aequilibrae
from aequilibrae.matrix import AequilibraeMatrix
from scipy.sparse import coo_matrix


# In[ ]:





# In[ ]:


import sys
print (sys.version)


# In[ ]:





# In[ ]:





# In[ ]:





# Now we set all the paths for files and parameters we need

# In[ ]:


data_folder = '../input/'
data_file = 'FAF4.4_HiLoForecasts.csv'
sctg_names_file = 'sctg_codes.csv'  # Simplified to 50 characters, which is AequilibraE's limit
output_folder = data_folder


# We import the the matrices

# In[ ]:


matrices = pd.read_csv(os.path.join(data_folder, data_file), low_memory=False)
print (matrices.head(10))


# We import the sctg codes

# In[101]:


sctg_names = pd.read_csv(os.path.join(data_folder, sctg_names_file), low_memory=False)
sctg_names.set_index('Code', inplace=True)
sctg_descr = list(sctg_names['Commodity Description'])
print (sctg_names.head(5))
sctg_names.shape

sctg_names.info()
sctg_names.count()

type (sctg_names)


# We now process the matrices to collect all the data we need, such as:
# * the list of zones
# * CSTG codes
# * Matrices/scenarios we are importing

# In[ ]:


# lists the zones
all_zones = np.array(sorted(list(set( list(matrices.dms_orig.unique()) + list(matrices.dms_dest.unique())))))

# Count them and create a 0-based index
num_zones = all_zones.shape[0]
idx = np.arange(num_zones)

# Creates the indexing dataframes
origs = pd.DataFrame({"from_index": all_zones, "from":idx})
dests = pd.DataFrame({"to_index": all_zones, "to":idx})

# adds the new index columns to the pandas dataframe
matrices = matrices.merge(origs, left_on='dms_orig', right_on='from_index', how='left')
matrices = matrices.merge(dests, left_on='dms_dest', right_on='to_index', how='left')

# Lists sctg codes and all the years/scenarios we have matrices for
mat_years = [x for x in matrices.columns if 'tons' in x]
sctg_codes = matrices.sctg2.unique()


# We now import one matrix for each year, saving all the SCTG codes as different matrix cores in our zoning system
# 
# 

# In[ ]:


# aggregate the matrix according to the relevant criteria
agg_matrix = matrices.groupby(['from', 'to', 'sctg2'])[mat_years].sum()

# returns the indices
agg_matrix.reset_index(inplace=True)


for y in mat_years:
    mat = AequilibraeMatrix()
    
    kwargs = {'file_name': os.path.join(output_folder, y + '.aem'),
              'zones': num_zones,
              'matrix_names': sctg_descr}
    
    mat.create_empty(**kwargs)
    mat.index[:] = all_zones[:]
    # for all sctg codes
    for i in sctg_names.index:
        prod_name = sctg_names['Commodity Description'][i]
        mat_filtered_sctg = agg_matrix[agg_matrix.sctg2 == i]
        
        m = coo_matrix((mat_filtered_sctg[y], (mat_filtered_sctg['from'], mat_filtered_sctg['to'])),
                                           shape=(num_zones, num_zones)).toarray().astype(np.float64)
        
        mat.matrix[prod_name][:,:] = m[:,:]
        
    

