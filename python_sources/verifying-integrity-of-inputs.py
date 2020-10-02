#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports

import pathlib
import geopandas


# # Verifying integrity of inputs
# 
# Hello there! Given that lots of inputs for this challenge were retrieved manually (all of them?), here is a little kernel to verify their integrity.

# In[ ]:


INPUTS_DIR = pathlib.Path('../input/cpe-data/')
dept_dirs = [x for x in INPUTS_DIR.iterdir() if x.is_dir()]


# # 1. ACS Directories
# 
# Let's see if all ACS Directories are named as expected.

# In[ ]:


EXPECTED_DIRECTORIES = [
    '{name}_ACS_education-attainment',
    '{name}_ACS_education-attainment-over-25',
    '{name}_ACS_employment',
    '{name}_ACS_income',
    '{name}_ACS_owner-occupied-housing',
    '{name}_ACS_poverty',
    '{name}_ACS_race-sex-age',
]

def get_dept_name(dept_dir):
    return dept_dir.name[5:]

def get_acs_data_dir(dept_dir):
    name = get_dept_name(dept_dir)
    return dept_dir / f'{name}_ACS_data'

def get_shapefile_dir(dept_dir):
    name = get_dept_name(dept_dir)
    return dept_dir / f'{name}_Shapefiles'


# In[ ]:


for dept_dir in dept_dirs:
    name = get_dept_name(dept_dir)
    expected_dirs = set([x.format(name=name) for x in EXPECTED_DIRECTORIES])
    found_dirs = set([x.name for x in get_acs_data_dir(dept_dir).iterdir()])

    if expected_dirs != found_dirs:
        left = expected_dirs - found_dirs
        right = found_dirs - expected_dirs
        
        msg = f"\n"
        msg += f"{name}\n"
        msg += f"Expected dirs not found: {left}\n"
        msg += f"Found dirs not expected: {right}"
        print(msg)


# # 2. ACS Files
# 
# And if the files are what we would expect

# In[ ]:


# same as EXPECTED_DIRECTORIES

EXPECTED_TABLES = [
    'S1501',
    'B15003',
    'S2301',
    'S1903',
    'S2502',
    'S1701',
    'DP05',
]

for dept_dir in dept_dirs:
    name = get_dept_name(dept_dir)
    # we will use pathlib objects
    acs_dir = get_acs_data_dir(dept_dir)
        
    expected_dirs = [acs_dir / x.format(name=name) for x in EXPECTED_DIRECTORIES]    
    expected_files = []
    for x_dir, x_table in zip(expected_dirs, EXPECTED_TABLES):
        x_file1 = x_dir / f'ACS_15_5YR_{x_table}_with_ann.csv'
        x_file2 = x_dir / f'ACS_15_5YR_{x_table}_metadata.csv'
        expected_files += [x_file1, x_file2]
        
    found_files = []
    for f_dir in acs_dir.iterdir():
        found_files += list(f_dir.iterdir())

    expected_files = set(str(x.relative_to(acs_dir)) for x in expected_files)
    found_files = set(str(x.relative_to(acs_dir)) for x in found_files)
    if expected_files != found_files:
        left = expected_files - found_files
        right = found_files - expected_files
        
        msg = f"\n"
        msg += f"{name}\n"
        msg += f"Expected files not found: {left}\n"
        msg += f"Found files not expected: {right}"
        print(msg)


# # 3. Shapefiles
# 
# Now, checking if the shapefiles can be loaded (no exceptions = okay).

# In[ ]:


for dept_dir in dept_dirs:
    name = get_dept_name(dept_dir)
    shape_dir = get_shapefile_dir(dept_dir)
    try:
        df = geopandas.read_file(str(shape_dir))
    except Exception as ex:
        raise ValueError(f"Could not load Department {name} shapefile") from ex


# # 4. Outer files

# In[ ]:


for dept_dir in dept_dirs:
    name = get_dept_name(dept_dir)
    acs_dir = get_acs_data_dir(dept_dir)
    shape_dir = get_shapefile_dir(dept_dir)
    
    files = [x.relative_to('../input') 
             for x in dept_dir.iterdir() 
             if x not in (acs_dir, shape_dir)]
    
    msg = f"{name}\n"
    for file in files:
        msg += f"{file}\n"
    print(msg)


# Department 49-00009 seems to have a duplicate.

# In[ ]:


get_ipython().system('ls -l ../input/cpe-data/Dept_49-00009/*_UOF.csv')


# # Notes
# 
# Please note that the scripts here only check how the files look on the outside (except the shapefiles that are loaded).
# 
# I will try to keep this updated with each dataset version.
# 
# Also, feel free to suggest any other tests for the inputs (you may also fork this and implement yourself ha!).
# 
# Good bye
