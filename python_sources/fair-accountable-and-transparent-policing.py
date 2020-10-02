#!/usr/bin/env python
# coding: utf-8

# Fair Accountable and Transparent Policing
# =====================

# In[ ]:


import pandas as pd
import os


# In[ ]:


get_ipython().system('ls ../input/cpe-data')


# In[ ]:


ACS_variable_descriptions = pd.read_csv("../input/cpe-data/{}.csv".format("ACS_variable_descriptions"), index_col=0, header=None)


# In[ ]:


ACS_variable_descriptions.shape


# In[ ]:


ACS_variable_descriptions.head()


# In[ ]:


ACS_variable_descriptions.loc['HC01_VC03']


# In[ ]:


get_ipython().system('ls -R ../input/cpe-data/Dept_11-00091')


# In[ ]:


def read_metadata(file):
    return pd.read_csv(file, index_col=0, header=None)
    
def read_data(file):
    return pd.read_csv(file, index_col=[0, 1, 2], header=[0,1], na_values=['(X)', '-', '**'])

def read_prepped(file):
    return pd.read_csv(file, header=[0,1])

def ingnore_DS_Store(directory):
    return filter(lambda f: f != '_DS_Store', os.listdir(directory))

def collect_info_for_dep (dept_dir):
    """
    This function collects the '.csv' files into pandas dataframes.
    The return value is a hash where the keys refer to the original file names.
    """
    base_dir = "../input/cpe-data/{}".format(dept_dir)
    data_directories = list(filter(lambda f: f.endswith("_data"), os.listdir(base_dir)))
    info = {'dept' : dept_dir}
    assert len(data_directories) == 1, "found {} data directories".format(len(data_directories))
    for dd in data_directories:
        directory = "{}/{}".format(base_dir, dd)
        dd_directories = ingnore_DS_Store(directory)
        #print(dd_directories)
        for ddd in dd_directories:
            ddd_directory = "{}/{}".format(directory, ddd)
            files = list(ingnore_DS_Store(ddd_directory))
            #print(files)
            assert len(files) == 2, "found {} files in {}".format(len(files), directory)
            full_file_names = ["{}/{}".format(ddd_directory, file) for file in files]
            dataframes = [read_metadata(file) if file.endswith('_metadata.csv') else read_data(file) for file in full_file_names]
            info[ddd] = dict(zip(files, dataframes))
    prepped_files = list(filter(lambda f: f.endswith("_prepped.csv"), os.listdir(base_dir)))
    for pf in prepped_files:
        info[pf] = read_prepped("{}/{}".format(base_dir, pf))
    return info


# In[ ]:


Dept_11_00091_info = collect_info_for_dep('Dept_11-00091')

Dept_11_00091_info.keys()


# In[ ]:


Dept_11_00091_info['11-00091_ACS_education-attainment'].keys()


# In[ ]:


Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_metadata.csv'].info()


# In[ ]:


Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_metadata.csv'].head()


# In[ ]:


Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_with_ann.csv'].info()


# In[ ]:


Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_with_ann.csv'].head()


# In[ ]:


Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_with_ann.csv'].loc['1400000US25027700100', 'HC01_MOE_VC02']


# In[ ]:


desc = Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_with_ann.csv'].describe()
desc.loc[:, desc.loc['count', :] > 0]


# In[ ]:


Dept_11_00091_info['11-00091_ACS_education-attainment']['ACS_16_5YR_S1501_with_ann.csv'].shape


# In[ ]:


# 768 - 606 columns are empty (all values are -NA-)


# In[ ]:


def investigate_dept(dept):
    print(dept['dept'])
    print('=' * 20)
    print(dept.keys())


# In[ ]:


investigate_dept(Dept_11_00091_info)


# In[ ]:


department_names = [
    'Dept_11-00091',
    'Dept_23-00089',
    'Dept_35-00103',
    'Dept_37-00027',
    'Dept_37-00049',
    'Dept_49-00009',
]

departments = {dep: collect_info_for_dep(dep) for dep in department_names}


# In[ ]:


for dep in departments.keys():
    investigate_dept(departments[dep])
    print()


# In[ ]:


get_ipython().system('ls -R ../input/cpe-data/Dept_35-00103')


# In[ ]:


departments['Dept_37-00049']['37-00049_UOF-P_2016_prepped.csv']


# In[ ]:


departments['Dept_37-00049']['37-00049_UOF-P_2016_prepped.csv'].info()


# ## Thoughts as of this point
# 
# * We got demographics of several departments/districts/Census Tract.
# For example, how many educated people live there. Then how many of them are males, and how many are females etc.
# Some numbers were given with confidence margin (e.g. add +/- 100 to the number).
# * Then we have some maps. Other demonstrated that those geographic resources also contain the location of the stations. Locations are important, I have not touched those yet at all.
# * The 'prepped' files deals with report of arrests / violence, and include also geographic location. We did not get 'prepped' files for all departments.
# 
# ### What's next? Understand locations better to tie the 'prepped' files to the demographic information.... 

# In[ ]:




