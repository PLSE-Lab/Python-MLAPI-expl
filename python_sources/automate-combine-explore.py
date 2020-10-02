#!/usr/bin/env python
# coding: utf-8

# The Center for Policing Equity has the mission of helping law enforcement professionals to better protect and serve the citizens of their cities. The path that was chosen is to use data to find and document situations where the disparity in treatment can compromise the mission of *protect and serve*.
# 
# The data sources available to accomplish such mission are:
# * census data
# * policing data about use of force or other police-related actions
# * maps
# 
# The challenge is to effectively combine these sources in order to make easy for a field expert to discover and document critical situations. The primary objective is to combine police data with census data and, as we will see, the shapefiles are a way of achieving this. In this notebook, I will present some descriptive statistics about census data and police data in order to display how I think the automation process should be. It is important to remember that any conclusion that didn't pass through several levels of skepticism has to be considered incorrect. For example, knowing that the majority of arrests involves a specific race means nothing without knowing how the underlying population is composed. Even then, the available data can be enough to prove that there is a significant difference from one segment of the population to the other but they are not telling us yet the reason behind this difference. Therefore, I strongly invite the reader to keep a skeptic eye towards what follows and please let me know if there is something methodologically wrong or other silly mistakes.
# 
# A quick look at the available data reveals that, for how much effort it was spent to standardize the input of such analysis, it is not obvious to immediately get what data is available and what isn't. The United States of America is a place with wildly diverse situations and the data describing different places represent this diversity not only in the content, but also in the format.
# 
# In other words, given a department and a suitable amount of time, it is not difficult to generate a report with the available data. However, the work has to be repeated almost from scratch once that we want to analyze the next department. On the other hand, investing time and resources in maintaining a healthy database can be a frustrating job because the data are coming from many different sources.
# 
# Therefore, the solution proposed here aims to make easier to spot a problem in the data (thus before spending precious hours starting an analysis that can not be finished), have a quick overview of the relevant available information, and thus pave the way to combine and better explore a given department. It would foolish of me thinking that the proposed exploration is in any way satisfactory for an expert, simply because I do not have that expertise in this field. Here I want to propose a process that can facilitate the work of an expert, not to replace it. For this reason, every function was written with the intent of being customizable (either due to other preferences, different needs, or changes in the data). 
# 
# The notebook can be summarized as follows:
# 
# * Check the quality of the data.
# * Import every piece of information available for a department
# * Get an overview of the census data
# * Get an overview of the police data
# * A simple way of combining these two sources of data
# * A case study to show how to combine census and police data and example analysis.
# 
# The 2 cities used here are **Dallas** and **Indianapolis**. The former used just to introduce the various functions, the latter will also be subject of an analysis.
# 
# The code written for this kernel will be soon available at my github: https://github.com/lucabasa/cpe-automate

# In[ ]:


from os.path import join, isfile
from os import path, scandir, listdir

# standard
import pandas as pd
import numpy as np

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import geopandas as gp
from shapely.geometry import Point

import gc


# # Structure of the input data and health check
# 
# The first function is simply a helper to speed up every exploration of the data folder. 

# In[ ]:


def list_all_files(location='../input/', pattern=None, recursive=True):
    """
    This function returns a list of files at a given location (including subfolders)
    
    - location: path to the directory to be searched
    - pattern: part of the file name to be searched (ex. pattern='.csv' would return all the csv files)
    - recursive: boolean, if True the function calls itself for every subdirectory it finds
    """
    subdirectories= [f.path for f in scandir(location) if f.is_dir()]
    files = [join(location, f) for f in listdir(location) if isfile(join(location, f))]
    if recursive:
        for directory in subdirectories:
            files.extend(list_all_files(directory))
    if pattern:
        files = [f for f in files if pattern in f]
    return files


# In[ ]:


list_all_files()[11:16]  # printing only 5 files for simplicity


# Next, we saw during this competition (and every other competition or realistic situation) that the folder structure of the data was somewhat clear but can present some minor annoying peculiarity. Therefore, the next set of functions scans through the input files and check that some consistency requirements are matched. This step was crucial to quickly find small and big criticalities and has the merit of being fairly scalable for being a 0.0.1 version.
# 
# The checks implemented are:
# 
# * Has every department the same set of topics (education, poverty, etc.)?
# * Are the geographic Id's consistent across the census data? 
# * Do we have police data and shapefiles in good order?
# 
# A serious error handling procedure is still missing, but this is good enough to give us a quick overview of what is available and what might require more attention.

# In[ ]:


def _get_topics(topics_list):
    for topic in topics_list:
        topics_list[topics_list.index(topic)] = topic.split('_')[-1]
    return topics_list


def check_topics(dept_num, base_topics, test_topics):
    """
    This function checks that a department has all the topics (education, poverty, etc.)
    that are present in the other departments.
    
    - dept_num: string identifying the department
    - base_topics: list of topics that the other departments have (if empty, it is created)
    - test_topics: topics found for the given department
    
    If there are new topics, the function updates base_topics and returns it
    """
    test_topics = _get_topics(test_topics)
    if len(base_topics) < 1:
        base_topics = test_topics  # the first time just create the list
    # check if something is missing
    mis_topics = [top for top in base_topics if top not in test_topics]
    if len(mis_topics) > 0:
        print(f"Department {dept_num} does not have data about the following topics:")
        print(mis_topics)
    # check if something is new
    new_topics = [top for top in test_topics if top not in base_topics]
    if len(new_topics) > 0:
        print(f"Department {dept_num} has data about the following new topics:")
        print(new_topics)
        print("The departments previously checked do not have these data")
        # updating the base_topics
        base_topics = list(set(base_topics + test_topics))
    return base_topics


def check_ids(base_ids, data):
    """
    This function checks that, across the topics, the id's are consistent
    """
    tmp_ids = data['GEO.id2'].unique()
    if len(tmp_ids) != data.shape[0]:
        print(f"In {file} inconsistent id's")
    if len(base_ids) < 1: # the first time it creates the "base" of id's
        base_ids = tmp_ids
    if set(tmp_ids) != set(base_ids):
        print(f"In {file} inconsistent id's with the other files")
    return base_ids


def data_quality(location='../input/data-science-for-good/cpe-data/'):
    """
    This is the main function, it checks every department at the given location,
    assuming that every department is in a separate directory.
    
    It checks for:
    - presence of police related data
    - consistency of the ACS files (and relative metadata)
    - presence of all the necessary shapefiles
    """
    # Get the list of the departments
    dept_list = [d.path for d in scandir(location) if d.is_dir()]
    topics = []  # needed to check if we have all
    
    # loop over departments
    for dept in dept_list:
        dept_num = dept.split('_')[1]
        print("_"*40)
        print(f'Checking department {dept_num}')
        
        # Check if we have some kind of data about crime or police-------------
        crime_files = list_all_files(dept, pattern='.csv', recursive=False)
        if len(crime_files) < 1:
            print(f"Department {dept_num} does not have data about police interventions")
        else:
            print("Department {} has {} file(s) about police interventions".format(dept_num, 
                                                                                   len(crime_files)))
            
        # Check the ACS data consistency -------------------------------------------
        data_path = dept + '/' + dept_num + '_ACS_data/'
        # Check if we have all the topics (poverty, education, etc)
        temp_topics = [d.path for d in scandir(data_path) if d.is_dir()]
        topics = check_topics(dept_num, topics, temp_topics)
        
        # Check if the data have consistent id's and columns
        files = list_all_files(data_path, pattern='_with_ann.csv')
        ids = []  # needed to check if we have all
        for file in files:
            data = pd.read_csv(file, skiprows=[1], low_memory=False, nrows=3)  # nrows is for speed
            meta = file.replace('_with_ann.csv', '_metadata.csv')
            metadata = pd.read_csv(meta, header=None, names=['key', 'description'])
            if not data.columns.all() in list(metadata['key']):
                print("In {} inconsistent metadata".format(file))
            ids = check_ids(ids, data)
        
        # Check the Shapefiles consistency ------------------------------------------
        data_path = dept + '/' + dept_num + '_Shapefiles/'
        extensions = ['.shp', '.shx', '.dbf', '.prj']
        for ext in extensions:
            files = list_all_files(data_path, pattern=ext)
            if len(files) < 1:
                print("Department {} does not have the {} file".format(dept_num, 
                                                                       ext))
            if len(files) > 1:
                print("Department {} has {} files with extension {}".format(dept_num, 
                                                                            len(files), 
                                                                            ext))
        print("\n")
    print("Done!")


# In[ ]:


data_quality()


# We see immediately that Dept_37-00027 is missing its projection file (as we know already thanks to other popular kernels) and that some departments have slightly different ways of calling some of the topics (poverty vs poverty-status, etc.). Before the cleaning done by the organizers, other inconsistencies were found by this function.
# 
# # Import a department
# 
# Once that we know how the folder structure is, we need to have all the available data in one convenient structure. I chose a dictionary of DataFrames to be that structure because dictionaries are awesome.
# 
# An issue that can be quickly spotted is that the ACS files have a lot of columns and often they are missing the entries. On top of that, the missing entries are not a simple *NaN* but rather some symbol.  I wanted to take care of that after I spent so much time to check what the individual column codes meant to then find out it was little to no information in them.
# 
# Therefore, the import process is also a generic cleaning of the data with missing values. As of now, every column with more than 30% of missing entries is not imported. In the future, there can be a method to change this parameter.

# In[ ]:


def import_topic(path, tolerance=0.7):
    """
    Imports the file at a given location,
    Coerces the values to be numerical in order to easily spot the missing values
    Drops every column with enough missing values, the threshold is set by the parameter tolerance
    
    It returns 2 DataFrames: one with the data, one with the metadata.
    """
    # find the file with the ACS data and load it
    datafile = list_all_files(path, pattern='_with_ann.csv')[0]
    data = pd.read_csv(datafile, skiprows=[1], low_memory=False)
    # take out the ids momentarily
    ids = data[[col for col in data.columns if 'GEO' in col]]
    rest = data[[col for col in data.columns if 'GEO' not in col]]
    # convert to numeric and force na's if necessary
    rest = rest.apply(pd.to_numeric, errors='coerce')
    # put data together again
    data = ids.join(rest)
    print(f'Shape: {data.shape}')
    cols = data.columns
    nrows = data.shape[0]
    removed = 0
    for col in cols:
        mis = data[col].isnull().sum() / nrows
        if mis > tolerance:
            removed += 1
            del data[col]
    if removed > 0:
        print("Removed {} columns because more than {}% of the values are missing".format(removed, 
                                                                                      tolerance*100))
        print(f"New shape: {data.shape}")
    meta = datafile.replace('_with_ann.csv', '_metadata.csv')
    metadata = pd.read_csv(meta, header=None, names=['key', 'description'])
    return data, metadata


def import_dept(location):
    """
    Imports all the police files, the ACS, the shapefiles at a given location
    
    It returns a dictionary of DataFrames
    """
    dept_num = location.split('_')[1]
    print(f'Importing department {dept_num}')
    print('\n')
    data_list = {}
    # Police data ------------------------
    print("Importing police data...")
    crime_files = list_all_files(location, pattern='.csv', recursive=False)
    crm_count = 1
    for crm in crime_files:
        crm_name = "police_" + str(crm_count)
        data_list[crm_name] = pd.read_csv(crm, skiprows=[1], low_memory=False)
        print("File {}, shape: {}".format(crm_count,
                                         data_list[crm_name].shape))
        crm_count += 1
    # ACS -------
    data_path = location + '/' + dept_num + '_ACS_data/'
    topics = listdir(data_path)
    for topic in topics:
        topic_name = topic.split('_')[-1]
        print(f'Importing {topic_name}...')
        data, meta = import_topic(data_path + topic, tolerance=0.3)  # I am being more strict than the default
        data_list[topic_name] = data
        data_list[topic_name + '_meta'] = meta    
    # Shapefiles -----
    print("Importing shapefile(s)...")
    data_path = location + '/' + dept_num + '_Shapefiles/'
    shapes = list_all_files(data_path, pattern='.shp')
    shapes = [shp for shp in shapes if shp.endswith('.shp')]
    shp_count = 1
    for shp in shapes:
        shp_name = 'shapefile_' + str(shp_count)
        data_list[shp_name] = gp.read_file(shp)
        print("File {}, shape: {}".format(shp_count,
                                         data_list[shp_name].shape))
        shp_count += 1
    gc.collect() # in case some of the files were really big
    return data_list


# The merit of this function is that it spots further inconsistencies (if it fails, something is odd) and makes everything available very quickly.
# 
# Let's see what departments are available once more and import a random one to show how the function works. Re-running the next cell with a different department can give you more information regarding how flexibles are the methods here presented (and, of course, what are their limitations).

# In[ ]:


dept_list = [d.path for d in scandir('../input/data-science-for-good/cpe-data/') if d.is_dir()]
dept_list


# In[ ]:


dept = import_dept(dept_list[9])


# In[ ]:


dept.keys()


# Did I mention that dictionaries are awesome? 
# 
# Let's have a look at one of the files

# In[ ]:


poverty = dept['poverty']
poverty.head(2)


# As we see both from the log messages of the import and from this table, the census data are very wide tables.
# 
# I personally struggled a lot to find out what information I could take away from DataFrames with so many non-descriptive columns. Luckily we have the metadata and I made up this simply routine to find out what is in there.

# In[ ]:


pov_meta = dept['poverty_meta']
# selecting only the columns that survived the import
pov_meta = pov_meta.loc[pov_meta.key.isin(list(poverty.columns))]  

for i in range(0,5):  # to have a full view use range(0,200)
    name = 'EST_VC' + str(i).zfill(2)  # focusing on the estimates
    pov_est = pov_meta[pov_meta.key.str.contains(name)].copy()
    desc = pov_est.description.values
    if len(desc) > 0 :
        print(name)
        print(desc)
        print("\n")


# With a few lines of code we know what the columns mean. For example:
# 
# * HC01 is about Total numbers
# * HC02 is about Totals below the poverty level
# * HC03 is about the percentage below the poverty level.
# 
# Then we know that each VC code is corresponding to a different segment of the population (either by race, gender, education, age, etc.).
# 
# It is an overwhelming feeling the one of having to look into so many different segments. That's why I propose the following selection
# 
# # ACS summary
# 
# This section is aimed to create one DataFrame with all the information that I found relevant by reading their description. In doing so, it also gives a quick overview of the ACS data.
# 
# The proposed columns are (and it is very easy to add or remove some of them):

# In[ ]:


poverty_list = {'HC01_EST_VC01' : 'p_total_est',  # these are just those we know the poverty level of
                'HC01_MOE_VC01' : 'p_total_moe',
                'HC02_EST_VC01' : 'p_below_pov_est',
                'HC02_MOE_VC01' : 'p_below_pov_moe',
                'HC03_EST_VC01' : 'p_below_pov_perc_est',
                'HC03_MOE_VC01' : 'p_below_pov_perc_moe',
                'HC02_EST_VC14' : 'p_males_below_pov_est',
                'HC02_MOE_VC14' : 'p_males_below_pov_moe',
                'HC03_EST_VC14' : 'p_males_below_pov_perc_est',
                'HC03_MOE_VC14' : 'p_males_below_pov_perc_moe',
                'HC02_EST_VC15' : 'p_females_below_pov_est',
                'HC02_MOE_VC15' : 'p_females_below_pov_moe',
                'HC03_EST_VC15' : 'p_females_below_pov_perc_est',
                'HC03_MOE_VC15' : 'p_females_below_pov_perc_moe',
                'HC02_EST_VC18' : 'p_white_below_pov_est',
                'HC02_MOE_VC18' : 'p_white_below_pov_moe',
                'HC03_EST_VC18' : 'p_white_below_pov_perc_est',
                'HC03_MOE_VC18' : 'p_white_below_pov_perc_moe',
                'HC02_EST_VC19' : 'p_black_below_pov_est',
                'HC02_MOE_VC19' : 'p_black_below_pov_moe',
                'HC03_EST_VC19' : 'p_black_below_pov_perc_est',
                'HC03_MOE_VC19' : 'p_black_below_pov_perc_moe',
                'HC02_EST_VC20' : 'p_native_below_pov_est',
                'HC02_MOE_VC20' : 'p_native_below_pov_moe',
                'HC03_EST_VC20' : 'p_native_below_pov_perc_est',
                'HC03_MOE_VC20' : 'p_native_below_pov_perc_moe',
                'HC02_EST_VC21' : 'p_asian_below_pov_est',
                'HC02_MOE_VC21' : 'p_asian_below_pov_moe',
                'HC03_EST_VC21' : 'p_asian_below_pov_perc_est',
                'HC03_MOE_VC21' : 'p_asian_below_pov_perc_moe',
                'HC02_EST_VC22' : 'p_islander_below_pov_est',
                'HC02_MOE_VC22' : 'p_islander_below_pov_moe',
                'HC03_EST_VC22' : 'p_islander_below_pov_perc_est',
                'HC03_MOE_VC22' : 'p_islander_below_pov_perc_moe',
                'HC02_EST_VC23' : 'p_other_race_below_pov_est',
                'HC02_MOE_VC23' : 'p_other_race_below_pov_moe',
                'HC03_EST_VC23' : 'p_other_race_below_pov_perc_est',
                'HC03_MOE_VC23' : 'p_other_race_below_pov_perc_moe',
                'HC02_EST_VC26' : 'p_hispanic_below_pov_est',
                'HC02_MOE_VC26' : 'p_hispanic_below_pov_moe',
                'HC03_EST_VC26' : 'p_hispanic_below_pov_perc_est',
                'HC03_MOE_VC26' : 'p_hispanic_below_pov_perc_moe'
                }


race_list = {'HC01_VC03': 'total_population',
             'HC01_VC04': 'total_males',
             'HC01_VC05': 'total_females',
             'HC01_VC49': 'total_white',
             'HC03_VC49': 'perc_white',
             'HC01_VC50': 'total_black',
             'HC03_VC50': 'perc_black',
             'HC01_VC51': 'total_native',  # sorry for not including the individual tribes
             'HC03_VC51': 'perc_native',
             'HC01_VC56': 'total_asian',
             'HC03_VC56': 'perc_asian', # sorry for not aknowledging that Asia is a huge place
             'HC01_VC64': 'total_islander',
             'HC03_VC64': 'perc_islander',
             'HC01_VC69': 'total_other_race',
             'HC03_VC69': 'perc_other_race',
             'HC01_VC88': 'total_hispanic',
             'HC03_VC88': 'perc_hispanic'
            }


housing_list = {'HC01_EST_VC01': 'h_total_houses_est',
                'HC01_MOE_VC01': 'h_total_houses_moe',
                'HC02_EST_VC01': 'h_owner_houses_est',
                'HC02_MOE_VC01': 'h_owner_houses_moe',
                'HC03_EST_VC01': 'h_rented_houses_est',
                'HC03_MOE_VC01': 'h_rented_houses_moe',
                'HC04_EST_VC04': 'h_total_houses_white_est',
                'HC04_MOE_VC04': 'h_total_houses_white_moe',
                'HC02_EST_VC04': 'h_owner_houses_white_est',
                'HC02_MOE_VC04': 'h_owner_houses_white_moe',
                'HC03_EST_VC04': 'h_rented_houses_white_est',
                'HC03_MOE_VC04': 'h_rented_houses_white_moe',
                'HC05_EST_VC05': 'h_total_houses_black_est',
                'HC05_MOE_VC05': 'h_total_houses_black_moe',
                'HC02_EST_VC05': 'h_owner_houses_black_est',
                'HC02_MOE_VC05': 'h_owner_houses_black_moe',
                'HC03_EST_VC05': 'h_rented_houses_black_est',
                'HC03_MOE_VC05': 'h_rented_houses_black_moe',
                'HC06_EST_VC06': 'h_total_houses_native_est',
                'HC06_MOE_VC06': 'h_total_houses_native_moe',
                'HC02_EST_VC06': 'h_owner_houses_native_est',
                'HC02_MOE_VC06': 'h_owner_houses_native_moe',
                'HC03_EST_VC06': 'h_rented_houses_native_est',
                'HC03_MOE_VC06': 'h_rented_houses_native_moe',
                'HC07_EST_VC07': 'h_total_houses_asian_est',
                'HC07_MOE_VC07': 'h_total_houses_asian_moe',
                'HC02_EST_VC07': 'h_owner_houses_asian_est',
                'HC02_MOE_VC07': 'h_owner_houses_asian_moe',
                'HC03_EST_VC07': 'h_rented_houses_asian_est',
                'HC03_MOE_VC07': 'h_rented_houses_asian_moe',
                'HC08_EST_VC08': 'h_total_houses_islander_est',
                'HC08_MOE_VC08': 'h_total_houses_islander_moe',
                'HC02_EST_VC08': 'h_owner_houses_islander_est',
                'HC02_MOE_VC08': 'h_owner_houses_islander_moe',
                'HC03_EST_VC08': 'h_rented_houses_islander_est',
                'HC03_MOE_VC08': 'h_rented_houses_islander_moe',
                'HC09_EST_VC09': 'h_total_houses_other_race_est',
                'HC09_MOE_VC09': 'h_total_houses_other_race_moe',
                'HC02_EST_VC09': 'h_owner_houses_other_race_est',
                'HC02_MOE_VC09': 'h_owner_houses_other_race_moe',
                'HC03_EST_VC09': 'h_rented_houses_other_race_est',
                'HC03_MOE_VC09': 'h_rented_houses_other_race_moe',
                'HC12_EST_VC12': 'h_total_houses_hispanic_est',
                'HC12_MOE_VC12': 'h_total_houses_hispanic_moe',
                'HC02_EST_VC12': 'h_owner_houses_hispanic_est',
                'HC02_MOE_VC12': 'h_owner_houses_hispanic_moe',
                'HC03_EST_VC12': 'h_rented_houses_hispanic_est',
                'HC03_MOE_VC12': 'h_rented_houses_hispanic_moe'
               }


income_list = {'HC01_EST_VC02': 'i_total_income_est',
               'HC01_MOE_VC02': 'i_total_income_moe',
               'HC02_EST_VC02': 'i_median_income_est',
               'HC02_MOE_VC02': 'i_median_income_moe',
               'HC01_EST_VC04': 'i_total_income_white_est',
               'HC01_MOE_VC04': 'i_total_income_white_moe',
               'HC02_EST_VC04': 'i_median_income_white_est',
               'HC02_MOE_VC04': 'i_median_income_white_moe',
               'HC01_EST_VC05': 'i_total_income_black_est',
               'HC01_MOE_VC05': 'i_total_income_black_moe',
               'HC02_EST_VC05': 'i_median_income_black_est',
               'HC02_MOE_VC05': 'i_median_income_black_moe',
               'HC01_EST_VC06': 'i_total_income_native_est',
               'HC01_MOE_VC06': 'i_total_income_native_moe',
               'HC02_EST_VC06': 'i_median_income_native_est',
               'HC02_MOE_VC06': 'i_median_income_native_moe',
               'HC01_EST_VC07': 'i_total_income_asian_est',
               'HC01_MOE_VC07': 'i_total_income_asian_moe',
               'HC02_EST_VC07': 'i_median_income_asian_est',
               'HC02_MOE_VC07': 'i_median_income_asian_moe',
               'HC01_EST_VC08': 'i_total_income_islander_est',
               'HC01_MOE_VC08': 'i_total_income_islander_moe',
               'HC02_EST_VC08': 'i_median_income_islander_est',
               'HC02_MOE_VC08': 'i_median_income_islander_moe',
               'HC01_EST_VC09': 'i_total_income_other_race_est',
               'HC01_MOE_VC09': 'i_total_income_other_race_moe',
               'HC02_EST_VC09': 'i_median_income_other_race_est',
               'HC02_MOE_VC09': 'i_median_income_other_race_moe',
               'HC01_EST_VC12': 'i_total_income_hispanic_est',
               'HC01_MOE_VC12': 'i_total_income_hispanic_moe',
               'HC02_EST_VC12': 'i_median_income_hispanic_est',
               'HC02_MOE_VC12': 'i_median_income_hispanic_moe'
              }


employment_list = {'HC04_EST_VC01': 'e_unempl_rate_est',
                   'HC04_MOE_VC01': 'e_unempl_rate_moe',
                   'HC04_EST_VC15': 'e_unempl_rate_white_est',
                   'HC04_MOE_VC15': 'e_unempl_rate_white_moe',
                   'HC04_EST_VC16': 'e_unempl_rate_black_est',
                   'HC04_MOE_VC16': 'e_unempl_rate_black_moe',
                   'HC04_EST_VC17': 'e_unempl_rate_native_est',
                   'HC04_MOE_VC17': 'e_unempl_rate_native_moe',
                   'HC04_EST_VC18': 'e_unempl_rate_asian_est',
                   'HC04_MOE_VC18': 'e_unempl_rate_asian_moe',
                   'HC04_EST_VC19': 'e_unempl_rate_islander_est',
                   'HC04_MOE_VC19': 'e_unempl_rate_islander_moe',
                   'HC04_EST_VC20': 'e_unempl_rate_other_race_est',
                   'HC04_MOE_VC20': 'e_unempl_rate_other_race_moe',
                   'HC04_EST_VC23': 'e_unempl_rate_hispanic_est',
                   'HC04_MOE_VC23': 'e_unempl_rate_hispanic_moe',
                   'HC04_EST_VC28': 'e_unempl_rate_males_est',
                   'HC04_MOE_VC28': 'e_unempl_rate_males_moe',
                   'HC04_EST_VC29': 'e_unempl_rate_females_est',
                   'HC04_MOE_VC29': 'e_unempl_rate_females_moe'
                  }


education_list = {'HC02_EST_VC42': 'ed_perc_hs_white_est',
                  'HC04_EST_VC42': 'ed_perc_hs_white_male_est',
                  'HC06_EST_VC42': 'ed_perc_hs_white_female_est',
                  'HC02_EST_VC43': 'ed_perc_ba_white_est',
                  'HC04_EST_VC43': 'ed_perc_ba_white_male_est',
                  'HC06_EST_VC43': 'ed_perc_ba_white_female_est',
                  'HC02_EST_VC46': 'ed_perc_hs_black_est',
                  'HC04_EST_VC46': 'ed_perc_hs_black_male_est',
                  'HC06_EST_VC46': 'ed_perc_hs_black_female_est',
                  'HC02_EST_VC47': 'ed_perc_ba_black_est',
                  'HC04_EST_VC47': 'ed_perc_ba_black_male_est',
                  'HC06_EST_VC47': 'ed_perc_ba_black_female_est',
                  'HC02_EST_VC50': 'ed_perc_hs_native_est',
                  'HC04_EST_VC50': 'ed_perc_hs_native_male_est',
                  'HC06_EST_VC50': 'ed_perc_hs_native_female_est',
                  'HC02_EST_VC51': 'ed_perc_ba_native_est',
                  'HC04_EST_VC51': 'ed_perc_ba_native_male_est',
                  'HC06_EST_VC51': 'ed_perc_ba_native_female_est',
                  'HC02_EST_VC54': 'ed_perc_hs_asian_est',
                  'HC04_EST_VC54': 'ed_perc_hs_asian_male_est',
                  'HC06_EST_VC54': 'ed_perc_hs_asian_female_est',
                  'HC02_EST_VC55': 'ed_perc_ba_asian_est',
                  'HC04_EST_VC55': 'ed_perc_ba_asian_male_est',
                  'HC06_EST_VC55': 'ed_perc_ba_asian_female_est',
                  'HC02_EST_VC58': 'ed_perc_hs_islander_est',
                  'HC04_EST_VC58': 'ed_perc_hs_islander_male_est',
                  'HC06_EST_VC58': 'ed_perc_hs_islander_female_est',
                  'HC02_EST_VC59': 'ed_perc_ba_islander_est',
                  'HC04_EST_VC59': 'ed_perc_ba_islander_male_est',
                  'HC06_EST_VC59': 'ed_perc_ba_islander_female_est',
                  'HC02_EST_VC62': 'ed_perc_hs_other_race_est',
                  'HC04_EST_VC62': 'ed_perc_hs_other_race_male_est',
                  'HC06_EST_VC62': 'ed_perc_hs_other_race_female_est',
                  'HC02_EST_VC63': 'ed_perc_ba_other_race_est',
                  'HC04_EST_VC63': 'ed_perc_ba_other_race_male_est',
                  'HC06_EST_VC63': 'ed_perc_ba_other_race_female_est',
                  'HC02_EST_VC70': 'ed_perc_hs_hispanic_est',
                  'HC04_EST_VC70': 'ed_perc_hs_hispanic_male_est',
                  'HC06_EST_VC70': 'ed_perc_hs_hispanic_female_est',
                  'HC02_EST_VC71': 'ed_perc_ba_hispanic_est',
                  'HC04_EST_VC71': 'ed_perc_ba_hispanic_male_est',
                  'HC06_EST_VC71': 'ed_perc_ba_hispanic_female_est'
                 }


# You can tell I love the dictionaries, can't you?
# 
# Moving on, the next function (with 2 helpers), is preparing a unique DataFrame. The reason this is done (effectively) column by column is that not every column is necessarily present in the imported data. We thus sacrify some speed for more flexibility.

# In[ ]:


def _add_ACS_column(data, column, output):
    try:
        to_add = data[['GEO.id', 'GEO.id2', 'GEO.display-label', column]].copy()
        output = pd.merge(output, to_add, on=['GEO.id', 'GEO.id2', 'GEO.display-label'])
    except KeyError:
        pass
    return output


def _add_ACS_topic(data, output, col_list):
    data = data.rename(columns=col_list)
    col_list = list(col_list.values())
    for col in col_list:
        output = _add_ACS_column(data, col, output)
    return output


def prepare_ACS(dept):
    """
    This function merges together the chosen columns for all the topics in the census data
    """
    topics = [topic for topic in list(dept.keys()) if '_meta' not in topic 
              and 'police' not in topic and 'shapefile' not in topic 
              and 'education-attainment-over-25' not in topic]  # it is redundant
    
    switcher = {  # this is because python is cool by I still miss a switch statement
        'poverty': poverty_list,
        'poverty-status': poverty_list,
        'owner-occupied-housing': housing_list,
        'race-sex-age': race_list,
        'race-age-sex': race_list,
        'income': income_list,
        'education-attainment': education_list,
        'employment': employment_list
        }
    
    output = dept['education-attainment'][['GEO.id', 'GEO.id2', 'GEO.display-label']].copy()
    size = 0
    
    for topic in topics:
        col_list = switcher.get(topic)
        size += len(col_list.keys())
        output = _add_ACS_topic(dept[topic], output, col_list)
        
    print(f"Expected size of the output: {size} columns")
    print(f"Available data: {output.shape}")
    return output


# In[ ]:


df = prepare_ACS(dept) # it was imported above
df.head(3)


# As we see, not everything we wanted is then available. We have to live with that, I guess.
# 
# The next step is to quickly explore what we have created. This is done by the following set of functions (wrapped by `summarize_ACS()`).

# In[ ]:


def _wavg(data, column, weight):
    return np.average(data[column], weights=data[weight])


def _print_stats(data, col_list, total='total_population'):
    try:
        tmp = data[[total] + col_list].fillna(0)
        for col in col_list:
            print('{}: {}'.format(col, round(_wavg(tmp, col, total),3)))
    except KeyError:
        print('Total population unavailable, the means are not weighted')
        tmp = data[col_list].fillna(0)
        for col in col_list:
            print('{}:{}'.format(col, round(tmp[col].mean(),3)))

            
def _print_perc(data, col_list):
    for col in col_list:
        min_perc = data[col].min()
        med_perc = data[col].median()
        max_perc = data[col].max()
        print(col)
        print(f'\t Min: {min_perc}')
        print(f'\t Median: {med_perc}')
        print(f'\t Max: {max_perc}')


def overview_ACS(data):
    tot_pop = data[[col for col in data.columns if col.startswith('total_')]].sum()
    tot_pop = round(tot_pop / tot_pop[0] * 100, 2)
    print(tot_pop)
    print("_"*40)

    race_perc = [col for col in data.columns if col.startswith('perc_')]
    _print_perc(data, race_perc)


def unemployment_ACS(data):
    unemp_cols = [col for col in data.columns if 'e_unemp' in col and '_est' in col]
    _print_stats(data, unemp_cols)
            

def poverty_ACS(data):
    pov_cols = [col for col in data.columns if 'below_pov_perc_est' in col]
    _print_stats(data, pov_cols)
    print("_"*40)
    _print_perc(data, pov_cols)


def income_ACS(data):
    inc_cols = [col for col in data.columns if 'median_income' in col and '_est' in col]
    mean_inc = round(data[inc_cols].mean(),1)
    max_inc = round(data[inc_cols].max(), 1)
    min_inc = round(data[inc_cols].min(), 1)
    print('Mean of medians: ' + '-'*10)
    print(mean_inc)
    print('Max of medians: ' + '-'*10)
    print(max_inc)
    print('Min of medians: ' + '-'*10)
    print(min_inc)
    
    
def education_ACS(data):
    ed_cols = [col for col in data.columns if 'ed_perc_' in col and
               'male' not in col and 'female' not in col]
    _print_perc(data, ed_cols)
        

def summarize_ACS(data):
    print("Population overview (estimated totals)")
    overview_ACS(data)
    print('\n')
    
    print("Unemployment rate (weighted averages)")
    unemployment_ACS(data)
    print('\n')
    
    print('Below poverty level (weighted averages)')
    poverty_ACS(data)
    print('\n')
    
    print('Median income (means and ranges)')
    income_ACS(data)
    print('\n')
    
    print('Education (estimated percentages)')
    education_ACS(data)
    print('\n')


# In[ ]:


summarize_ACS(df)


# In seconds we know that in this department 60% of the population is white, but there are places where the percentage of black people is 98% and the Hispanic ethnicity is very present. We also see that black people have a higher unemployment rate and more of them are below the poverty level. At last, the access to higher education looks unevenly distributed across races.
# 
# Naturally, this is a very superficial analysis but in my mind this is what you do the first time you open a file. Moreover, it is not hard to extend this quick report with more accurate analysis with automatic tests of statistical significance and graphs. All that being said, the main (and, let's face it, the only) merit of this approach is that it is very flexible across the departments and a simple loop on what I called `dept_list` can give you insights about what department is really lacking information (its output would be much shorter).
# 
# # Police data
# 
# The main data source for this problem is coming from this data and it is time to adopt the same approach, aimed towards speed and flexibility, to have a first look 

# In[ ]:


def _drop_columns(feats, additional=None):
    """
    This function takes a list of features and removes DETAILS and ID.
    The user can provide an additional list to remove more features
    """
    to_drop = ['DETAILS', 'ID']
    if additional:
        to_drop = to_drop + additional
    feats = [feat for feat in feats if feat not in to_drop]
    return feats       


def _get_columns(data):
    """
    This helper finds the columns regarding subjects and officers.
    The prefix SUBJECT_ and OFFICER_ are removed.
    It returns a list of columns regarding subjects, one regarding officers
    and one with their intersection
    """
    subj = [col.replace('SUBJECT_', '') for col in data.columns if 'SUBJECT' in col]
    off = [col.replace('OFFICER_', '') for col in data.columns if 'OFFICER' in col]
    conf = list(set(subj).intersection(off))
    conf = _drop_columns(conf)
    return subj, off, conf


def subj_v_off(data, conf):
    """
    This function takes the data and a list of columns describing both subjects and columns
    Accordingly to the nature of the columns, it produces side by side plots and prints some 
    descriptive statistics (count, crosstabs)
    """
    num = len(conf)
    # 2 plots side by side for each category
    fig, ax = plt.subplots(num,2, figsize=(15,5*num))
    i = 0
    for feat in conf:
        off = 'OFFICER_' + feat
        subj = 'SUBJECT_' + feat
        print(feat)
        if feat in ['GENDER', 'RACE', 'HOSPITALIZATION', 'INJURY', 'INJURY_TYPE']:
            print('Officers: ' + '-'*40)
            print(data[off].value_counts(dropna=False, normalize=True).head(10))
            
            print("Subjects: " + '-'*40)
            print(data[subj].value_counts(dropna=False, normalize=True).head(10))
            
            if (len(data[subj].unique()) > 10 or len(data[off].unique()) > 5):
                print("Too many unique values, crosstab not printed")
            else:
                print("Crosstab: " + '-'*40)
                print(pd.crosstab(data[subj], data[off], 
                                  dropna=False, margins=True))
                print(pd.crosstab(data[subj], data[off], 
                                  dropna=False, normalize=True, margins=True))
            print("_"*40)
            print("\n")
            if num == 1: # dirty escape for poor usage of subplots
                sns.countplot(x=off, data=data, ax=ax[0], 
                              order=data[off].value_counts().iloc[:5].index) # plot only top 5 
                sns.countplot(x=subj, data=data, ax=ax[1], 
                              order=data[subj].value_counts().iloc[:5].index)
            else:
                sns.countplot(x=off, data=data, ax=ax[i][0], 
                              order=data[off].value_counts().iloc[:5].index)
                sns.countplot(x=subj, data=data, ax=ax[i][1], 
                              order=data[subj].value_counts().iloc[:5].index)
                i = i + 1
                
        elif feat in ['AGE']:
            print('Officers: ' + '-'*40)
            print(f"\t- mean: {data[off].mean()}")
            print(f"\t- median: {data[off].median()}")
            print(f"\t- range: {data[off].min()}--{data[off].max()}")
            print(f"\t- std: {data[off].std()}")
            
            print('Subjects: ' + '-'*40)
            print(f"\t- mean: {data[subj].mean()}")
            print(f"\t- median: {data[subj].median()}")
            print(f"\t- range: {data[subj].min()}--{data[subj].max()}")
            print(f"\t- std: {data[subj].std()}")
            print("_"*40)
            print("\n")
            if num == 1:
                sns.distplot(data[off].dropna(), bins = 30, ax=ax[0])
                sns.distplot(data[subj].dropna(), bins = 30, ax=ax[1])
            else:
                sns.distplot(data[off].dropna(), bins = 30, ax=ax[i][0])
                sns.distplot(data[subj].dropna(), bins = 30, ax=ax[i][1])
                i = i + 1


def _cross_cat_cont(data, cont, cat, title=None):
    """
    This function plots a histogram of a continuous variable by segmenting it
    according to a categorical variable
    """
    g = sns.FacetGrid(data, hue=cat, height= 5)
    g.map(plt.hist, cont, alpha= 0.3, bins=30)
    g.add_legend()
    if title:
        plt.title(title)
        

def _experience_segm(data, segment, col=None):
    """
    This function plots the officers year on force, segmented by 2 categories (if provided)
    """
    g = sns.FacetGrid(data, col=col, hue=segment, height= 5)
    g.map(plt.hist, 'OFFICER_YEARS_ON_FORCE', alpha= 0.3, bins=30)
    g.add_legend()
    

def individuals(data, feats, role='SUBJECT'):
    """
    Prints and plots a summary of the features regarding subjects and officers
    The output depends on what is available
    """
    condition = all(x in feats for x in ['AGE', 'RACE'])
    if condition:
        _cross_cat_cont(data, role + '_AGE', role + '_RACE')
    
    condition = all(x in feats for x in ['AGE', 'GENDER'])
    if condition:
        _cross_cat_cont(data, role + '_AGE', role + '_GENDER')
    
    condition = all(x in feats for x in ['RACE', 'WAS_ARRESTED'])
    if condition:
        print(pd.crosstab(data[role + '_RACE'], data[role + '_WAS_ARRESTED'], 
                                  dropna=False, normalize='index', margins=True))
        print("_"*40)
        print('\n')
        
    condition = all(x in feats for x in ['RACE', 'INJURY'])
    if condition:
        print(pd.crosstab(data[role + '_RACE'], data[role + '_INJURY'], 
                                  dropna=False, normalize='index', margins=True))
        print("_"*40)
        print('\n')
        
    condition = all(x in feats for x in ['RACE', 'HOSPITALIZATION'])
    if condition:
        print(pd.crosstab(data[role + '_RACE'], data[role + '_HOSPITALIZATION'], 
                                  dropna=False, normalize='index', margins=True))
        print("_"*40)
        print('\n')
        
    condition = all(x in feats for x in ['YEARS_ON_FORCE', 'INJURY'])
    if condition:
        _cross_cat_cont(data, role + '_YEARS_ON_FORCE', role + '_INJURY')
    
    condition = all(x in feats for x in ['YEARS_ON_FORCE'])
    if condition:
        try:
            _experience_segm(data, 'SUBJECT_RACE', col='SUBJECT_INJURY')
        except KeyError:
            _cross_cat_cont(data, role + '_YEARS_ON_FORCE', 'SUBJECT_RACE')
        try:
            _experience_segm(data, 'SUBJECT_GENDER', col='SUBJECT_INJURY')
        except KeyError:
            _cross_cat_cont(data, role + '_YEARS_ON_FORCE', 'SUBJECT_GENDER')
        
            
def explore_police(data):
    """
    Wrapper for the functions above, calls the appropriate function given 
    what is available
    """
    subj, off, conf = _get_columns(data)
    if len(subj) > 0:
        try:
            individuals(data, subj, 'SUBJECT')
        except Exception as e:
            print("Something went wrong in exploring the subjects")
            print(e)
            pass
    if len(off) > 0:
        try:
            individuals(data, off, 'OFFICER')
        except Exception as e:
            print("Something went wrong in exploring the officers")
            print(e)
            pass
    if len(conf) > 0:
        try:
            subj_v_off(data, conf)
        except Exception as e:
            print("Something went wrong in comparing subjects and officers")
            print(e)
            pass
    print(f"Subject related variables found: {subj}")
    print(f"Officer related variables found: {off}")


# In[ ]:


pol_df = dept['police_1'].copy()
explore_police(pol_df)


# Again, this is very basic and superficial but within seconds we can extract every available information (among the ones that were preselected) about the use of force in this department. We see that the majority of the subjects were black in a population that is predominantly white (as represented by the race of the officers, fairly in line with the 60% we know from the census data). We can see that the police officer is a male-dominated profession. 
# 
# We also see that most of the times there is no injury (either for the subjects or the officers) but when the injury occurs is most likely to happen to the subject rather than to the officer and that white subjects have an injury rate above average. It would be interesting to investigate the nature of this cases.
# 
# This is the spirit that led me to the creation of these summaries: simplify the overwhelming quantity and variety of data in order to start a more serious investigation. This is why **it would be a mistake to stop here and come to any sort of conclusion**.
# 
# Most of the files containing data regarding police actions also have a column that regards a district. For reasons that will be clear later, it is interesting to summarize the police data by district. So it is time for a new function. I will focus on summarizing race, gender, injuries, and consequences because the nature of the entries makes them easy to pass through this process automatically

# In[ ]:


col_list = ['SUBJECT_RACE', 'SUBJECT_GENDER', 'SUBJECT_INJURY', 'OFFICER_INJURY', 
            'SUBJECT_WAS_ARRESTED', 'SUBJECT_HOSPITALIZATION']


def _summary_cleanup(data, distr_col):
    """
    Keep only the columns selected above (plus the district)
    """
    feats = [distr_col] + [col for col in data.columns if col in col_list]
    return data[feats]


def police_by_distr(data, distr_col):
    """
    The police data are reduced to the one selected above
    and summarized according to the distr_col column.
    
    Returns a datafram with the aggregated data.
    """
    data = _summary_cleanup(data, distr_col)
    
    try:
        tot_df = data[[distr_col, data.columns[-1]]].groupby(distr_col, as_index=False).count()
        tot_df.columns = [distr_col, 'total_records']
    except ValueError:
        print("Insufficient data to aggregate")
        return data.head()
    sum_cols = [col for col in data.columns if
                ('RACE' not in col) and ('GENDER' not in col)]
    sum_df = data[sum_cols].groupby(distr_col, as_index=False).agg('sum')
    summary = pd.merge(tot_df, sum_df)
    
    if 'SUBJECT_RACE' in data.columns:
        race = data.groupby([distr_col, 'SUBJECT_RACE']).size().unstack().reset_index().fillna(0)
        summary = pd.merge(summary, race, on=distr_col)
        
    if 'SUBJECT_GENDER' in data.columns:
        gender = data.groupby([distr_col, 'SUBJECT_GENDER']).size().unstack().reset_index().fillna(0)
        summary = pd.merge(summary, gender, on=distr_col)
    
    return summary


# Since in this particular district the injuries are coded with Yes/No, I need to do a little of processing first. 

# In[ ]:


conversion = {'Yes': 1, 'No': 0}

pol_df.SUBJECT_INJURY = pol_df.SUBJECT_INJURY.map(conversion)
pol_df.OFFICER_INJURY = pol_df.OFFICER_INJURY.map(conversion)
pol_df.SUBJECT_WAS_ARRESTED = pol_df.SUBJECT_WAS_ARRESTED.map(conversion)


# In[ ]:


police_by_distr(pol_df, 'LOCATION_DISTRICT')


# This result will be useful once that we will do the same kind of aggregation on the census data. For now, we can only see that there are districts with more records than others, that the subject injury rate is more or less constant across the departments while the officer's one has some interesting cases where it is much lower. 
# 
# In some districts it may look like the race is playing a role but, since we don't have any information about the population of those districts yet, it is a merely descriptive result.
# 
# Now, how do we link this result to the census data explored above? While we have some geographical Id on the census data, we don't have a way to link it to a specific location. 
# 
# The strategy is to use shapefiles associated with census data, which can be found at https://www.census.gov/geo/maps-data/data/cbf/cbf_tracts.html, and use it to link it to the police data. We can do so in various ways:
# 
# * If we have the address, one can use the Google API to get the coordinates (if not the district name directly)
# * If we have the coordinates, one can automatically check if they fall inside a specific location (labeled with the geographical Id of the census data)
# * Or we can use the provided shapefiles, intersect them with the census one, get the district names and aggregate to finally put all our data in one table.
# 
# In this particular case, we have coordinates and address, but it is fairly an anomaly. In the case study below we will explore how to implement the third option with yet another attempt to automating the process.
# 
# But for now we move on to the third source of data.
# 
# # Shapefiles
# 
# Some preliminary exploration of the shapefiles comes with the quality check and the import. Thus we already know that some department have all we need: 3 mandatory files (`['.shp', '.shx', '.dbf']`) and the equally useful `'.prj'`. The first three are necessary to make the gelocalization api of choice work but the last one is the one that tells us the coordinate system of the shapefile, without it we will have difficulties in plotting something on top of our map.
# 
# Let's see what columns we have in these shapefiles with a simple routine similar to the one above (yes, the function `list_all_files` is something I like as much as the python dictionaries)

# In[ ]:


shapes = list_all_files(location='../input/data-science-for-good/cpe-data/',pattern='.shp')
shapes = [file for file in shapes if file.endswith('.shp')]
shapes


# In[ ]:


for file in shapes[:3]: # printing them all would be horrible to see
    print(file)
    sh = gp.read_file(file)
    print(sh.columns)
    print("\n")


# We see that we have always the column `geometry`, which contains the polygons or the points for the shape, something about districts or zones (can be called `name`, `district`, `precint`, or variations of this). In particular, the random one we have already imported si one of the simplest among the available.

# In[ ]:


dept['shapefile_1']


# We can even plot it.

# In[ ]:


dept['shapefile_1'].plot(column='Name')


# As mentioned before, the police data for the city of Dallas, TX contains longitude and latitude for almost every event. If we are interested in knowing if different districts behave differently, we could overlap these coordinates with the shapefile provided by the organizers. There is no way I can do it better than Pavan Kumar Kulkarni and I won't try to top that excellent kernel that I invite you to read (https://www.kaggle.com/pavankumarkulkarni/measure-of-justice-comprehensive-framework).
# 
# What I want to do instead is to combine it with the census data. Having the coordinates, the combination of the two sources will be as granular as possible, i.e. any other case will require a higher level of aggregation (with the approximation that comes with that).
# 
# As an external source, I put the shapes provided by the census website.
# 

# In[ ]:


ref_id = gp.read_file('../input/texas-shape/cb_2015_48_tract_500k.shp')
ref_id.to_crs({'init': 'epsg:32118'},inplace=True) # because meters are better
ref_id = ref_id[['GEOID', 'geometry']].copy()
ref_id.rename(columns={'GEOID' : 'GEO.id2'}, inplace=True)
ref_id['GEO.id2'] = pd.to_numeric(ref_id['GEO.id2'])
ref_id.head()


# Now we have all the shapes of Texas

# In[ ]:


ref_id.plot()


# We can then get all the shapes of Dallas

# In[ ]:


dallas_sh = ref_id[ref_id['GEO.id2'].isin(df['GEO.id2'])]
dallas_sh.plot()


# With the police data, we can find all the locations we need

# In[ ]:


pol_sh = pol_df.dropna(subset = ['LOCATION_LONGITUDE','LOCATION_LATITUDE']).copy()
pol_sh['geometry'] = list(zip(pol_sh['LOCATION_LONGITUDE'],pol_sh['LOCATION_LATITUDE']))
pol_sh['geometry'] = pol_sh['geometry'].apply(Point)
pol_sh = gp.GeoDataFrame(pol_sh, geometry = 'geometry',
                         crs ={'init': 'epsg:4326'} )
pol_sh.to_crs({'init': 'epsg:32118'},inplace = True)

pol_sh = pol_sh.dropna(subset = ['geometry'])

pol_sh.plot()


# There are problems with packages and dependencies when I try to run my code on Kaggle. Thus, I propose here a dirty workaround, something that works but it is slow and  disgraceful

# In[ ]:


def find_geoid(city, pol):
    geoids = []
    for point in pol.geometry:
        try:
            geoids.append(city[city.geometry.contains(point)]['GEO.id2'].values[0])
        except IndexError:
            geoids.append(np.nan)
    return geoids


# In[ ]:


pol_sh['GEO.id2'] = find_geoid(dallas_sh, pol_sh)
pol_sh.head()


# And now we have all we wanted: geographic id's on the police data, ready to be merged with the census data for a more in-depth analysis. Moreover, a further overlap with the district shapefile is still possible (and it will be done in the case study below)
# 
# As said before, this is a very benevolent case because the coordinates are not always provided. To see what happens in a more common case, let's have a look at Indianapolis for our case study.
# 
# # Case study - Marion County, Indiana
# 
# Here I pick a department that can give us enough to have a taste of most of the functionalities here presented, while still having some obstacles that will give us the chance of seeing how much effort is required to overcome them.
# 
# Let's start with the import.

# In[ ]:


dept = import_dept(dept_list[0])
dept.keys()


# We see that in some cases we drop quite a lot of columns, there is not much we can do about it.
# 
# Running `explore_police` now leads to catching some exceptions and the reason is that there is a typo in the columns. I fix it before moving forward

# In[ ]:


pol_df = dept['police_1'].copy()
pol_df.rename(columns={'SUBJECT_RACT' : 'SUBJECT_RACE'}, inplace=True)
explore_police(pol_df)


# Let's quickly summarize this first result:
# 
# * The arrest rate looks fairly constant across races. There are a few exceptions but, looking at the count plot for the subject race, those races are very little represented in our data.
# * Injury and hospitalization rates are again skewed towards the subjects, which makes sense due to the fact that officers are trained to first protect themselves. As before, white subjects have a higher injury rate than the average and this is also observed in the hospitalization rate. We don't have information about the injury types from this summary and it will require manual inspection.
# * There are some errors in the ages of subjects and officers because 2 years old officers would be silly (although maybe appropriate to chase down 2 years old subjects, I guess)
# * The officers are overwhelmingly white, while the subjects are mostly black.
# * Gender-wise, males dominate both the categories.
# * The age patterns reveals that the distribution is slightly more skewed towards the younger ages for black subjects and for female subjects.
# * The years of experience of the officers do not seem to matter in terms of how subjects of different races are treated or in terms of injuries.
# 
# This summary does not cover everything is available (although it could with some more development of the methods), so let's see what else is available.

# In[ ]:


pol_df.columns


# The CHARGE column can be very interesting.

# In[ ]:


pol_df.CHARGE.value_counts().head()


# I wonder why there are 2 Resisting Law Enforcement columns, maybe gender?

# In[ ]:


pol_df[pol_df.CHARGE == 'Resisting Law Enforcement (M)'].SUBJECT_GENDER.value_counts()


# No, it requires more study.
# 
# We can see that geographically we have expected districts and other values that will probably require recoding

# In[ ]:


pol_df['LOCATION_DISTRICT'].value_counts().head(8)


# Before moving on, there are some columns that got my curiosity and were not included in the summary. I want to spend some time to have a look at differences in reasons for the use of force and type of use of force across different races

# In[ ]:


pol_df['REASON_FOR_FORCE'].value_counts()


# In[ ]:


pd.crosstab(pol_df['SUBJECT_RACE'], pol_df['REASON_FOR_FORCE'])


# In[ ]:


pd.crosstab(pol_df['SUBJECT_RACE'], pol_df['REASON_FOR_FORCE'], normalize='index', margins=True)


# We can't observe any noticeable difference between the reason of force across the races. The main reason is always Resisting Arrest, followed by Non-Compliant and Combative Suspect.
# 
# To see the type of force used, we need to do some processing

# In[ ]:


types_code = ['Physical', 'Lethal', 'Less Lethal']  # the order is sadly crucial or everything is lethal

pol_df.TYPE_OF_FORCE_USED = pol_df.TYPE_OF_FORCE_USED.fillna('Other')

for code in types_code:
    pol_df.loc[pol_df.TYPE_OF_FORCE_USED.str.contains(code), 'TOF_code'] = code

pol_df.TOF_code = pol_df.TOF_code.fillna('Other')

pol_df.TOF_code.value_counts()


# In[ ]:


pd.crosstab(pol_df['SUBJECT_RACE'], pol_df['TOF_code'])


# In[ ]:


pd.crosstab(pol_df['SUBJECT_RACE'], pol_df['TOF_code'], normalize='columns', margins=True)  
# warning, the normalization is across races this time


# While at a first look there is not any statistically significant difference, it is hard to not notice that of the 18 lethal uses of force, 13 had a black person as a victim. We, unfortunately, do not have further information about the situation that led to these outcomes but we can say that, **while in any other type of force used roughly half of the subjects were black, in the case of lethal use of force 72% is black**. 
# 
# The situation starts looking even worse if we then look at the reason for force, which is mostly *Fleeing*. 

# In[ ]:


pd.crosstab(pol_df['REASON_FOR_FORCE'], pol_df['TOF_code'], margins=True)


# This can include car crashes while attempting an escape. So let's have a look at the not recoded data

# In[ ]:


pol_df[pol_df.TOF_code == 'Lethal'][['SUBJECT_DETAILS', "TYPE_OF_FORCE_USED", "REASON_FOR_FORCE", "CHARGE",
                                     "SUBJECT_RACE", 'OFFICER_RACE', 'SUBJECT_AGE', 'OFFICER_YEARS_ON_FORCE']]


# Not as many car crashes as I was hoping for (weird sentence, I know). I wish I could say more about these cases, but I have no data to do so.
# 
# Let's move on to the census data, we can use the functions to put them together and summarize according to the selection above.

# In[ ]:


df_ACS = prepare_ACS(dept)
summarize_ACS(df_ACS)


# * The population is mostly white, with a 63% that is fairly in line with the country rate. For this kind of things there is always some ambiguity regarding the Hispanic ethnicity and I still look for a solid solution.
# * The maximum and minimum in the race percentages by location reveals that there are places where racial diversity is not achieved at all. This indicates that any aggregation at the city level is limited since it misses the diverse situation across this department.
# * The unemployment rate is not uniformly distributed, having a higher rate for black people (and males).
# * Not surprisingly given the previous result, the poverty rate for black people is higher. On the other side, females have a higher poverty rate regardless of being more employed (although it is very close to the average)
# * The Hispanic ethnicity seems to be more vulnerable economically, this is an interesting result because it goes across races (being an ethnicity). Unfortunately, my selection is limiting the investigation and we would need to go back to the imported file to know more.
# * For everyone but white people, there is at least one location with 100% poverty rate. This can be due to numerically low representatives in some locations but, looking at the medians, we see that Blacks and Hispanics are more vulnerable in this sense.
# * The median incomes might indicate more disparity among black people, which reach a much lower minimum and the slightly higher maximum.
# * Education-wise, white and black people seem to have a similar behavior up to the high school but for the higher education we observe some disparity. It is too little to say if this determines or is determined by some of the results summarized above.
# * The Hispanic ethnicity seems to have a bigger problem already at the high school level.
# 
# Now, we have seen that we need to summarize these data at a different granular level. We thus need the census shapefiles for Indiana and the appropriate census track. Here it is provided as an external source.

# In[ ]:


ref_id = gp.read_file('../input/indiana-shape/cb_2015_18_tract_500k.shp')
ref_id.to_crs({'init': 'epsg:32118'},inplace=True) # because meters are better
ref_id = ref_id[['GEOID', 'geometry']].copy()
ref_id.rename(columns={'GEOID' : 'GEO.id2'}, inplace=True)
ref_id.head()


# And if we plot it we see that is really the State of Indiana

# In[ ]:


ref_id.plot()


# Yep, that looks like Indiana. To merge it with our census data we need a small correction but after that we can give a polygon to every geographical Id

# In[ ]:


ref_id['GEO.id2'] = pd.to_numeric(ref_id['GEO.id2'])
acs = pd.merge(df_ACS, ref_id, on='GEO.id2', how='left')


# We now use the provided shapefiles, which looks like this

# In[ ]:


sh = dept['shapefile_1'].copy()
sh.plot(column='DISTRICT')


# We immediately see that the projection is different, but luckily we can simply change it

# In[ ]:


sh.to_crs({'init': 'epsg:32118'},inplace=True)
sh.plot(column='DISTRICT')


# What I want to do is to overlap this shapefile with the other one so that I know in which district each geographic Id falls in and in which proportion.

# In[ ]:


sh = sh[['DISTRICT', 'geometry']].copy()

inter_shape = []
for index, crim in sh.iterrows():
    for index2, popu in acs.iterrows():
        if crim['geometry'].intersects(popu['geometry']):
            inter_shape.append({'geometry': crim['geometry'].intersection(popu['geometry']),
                         'district': crim['DISTRICT'],
                         'GEO.id2' : popu['GEO.id2'],
                         'area':crim['geometry'].intersection(popu['geometry']).area})
            
inter_shape = gp.GeoDataFrame(inter_shape,columns=['geometry', 'district', 'GEO.id2','area'])

inter_shape.head()


# We need another step to have what we want. We need to find if each Geo.Id falls inside a district completely or it is shared. Some simple (an inelegant) table manipulation gives the answer

# In[ ]:


tmp = inter_shape[['GEO.id2', 'district', 'area']].groupby(['GEO.id2', 'district'], as_index=False).sum()
tmp_2 = inter_shape[['GEO.id2', 'area']].groupby(['GEO.id2'], as_index=False).sum()
inter_shape = pd.merge(tmp, tmp_2, on='GEO.id2')
inter_shape['fraction'] = inter_shape['area_x'] / inter_shape['area_y']
del inter_shape['area_x']
del inter_shape['area_y']
del tmp
del tmp_2
inter_shape.head()


# In[ ]:


# and finally merge census and shapefile so that we have the districts
acs_merged = pd.merge(df_ACS, inter_shape)
acs_merged.head()


# The LOCATION_DISTRICT of the police file has some different names and I suspect that the one that looked somewhat weird can be all categorized as Excluded. The next cell takes care of that

# In[ ]:


renaming = {'East District': 'East',
           'Southwest District': 'Southwest',
           'Northwest District': 'Northwest',
           'North District': 'North',
           'Southeast District': 'Southeast',
           'Downtown  District': 'Downtown'}

pol_df.LOCATION_DISTRICT = pol_df.LOCATION_DISTRICT.map(renaming).fillna('Excluded')

pol_df.LOCATION_DISTRICT.value_counts(dropna=False)


# In[ ]:


# recall again the police_by_district aggregation
pol = police_by_distr(pol_df, "LOCATION_DISTRICT")
pol


# Now, so far we have prepared the census data so that we have a district and we made sure that the police data are aggregated by the same districts. Before finally put everything together, we need to aggregate the census data as well. In doing so, we have to take into account what fraction of a GEO.Id falls into a given district. This is done by the following function

# In[ ]:


def acs_by_district(data):
    data = data[[col for col in data.columns if 'GEO' not in col 
                 and 'geometry' not in col]].copy()
    
    # applying the fraction to the merged dataframe
    fraction = data['fraction']
    del data['fraction']
    cols = [col for col in data.columns if 'district' not in col]
    data[cols] = data[cols].multiply(fraction, axis="index")
    
    sel = ['district'] + [col for col in data.columns if '_est' in col 
                          or col.startswith('total_')]
    
    # grouping the totals
    tot_cols = [col for col in sel if 'perc' not in col
               and 'rate' not in col and 'median' not in col]
    totals = data[tot_cols].groupby('district', as_index=False).sum()
    
    # grouping the proportions
    try:
        prp = [col for col in sel if 'perc' in col
                                   or 'rate' in col]
        prop_cols = ['district'] + prp
        props = data[prop_cols].copy()
        # make them proportions
        props[prp] = props[prp].multiply(0.01, axis='index')
        # groupby with weighted average
        wm = lambda x: np.average(x, weights=data.loc[x.index, "total_population"])
        props = props.groupby('district', as_index=False).agg(wm)    
    except KeyError:
        print("Total population unavailable, percentages and rates can't be summarized")
        
    # mergin together
    summary = pd.merge(totals, props, on='district')
    
    return summary


# In[ ]:


acs_agg = acs_by_district(acs_merged)
acs_agg


# We are finally ready to put everything together

# In[ ]:


pol.rename(columns={'LOCATION_DISTRICT': 'district'}, inplace=True) # To merge easily
tot_agg = pd.merge(acs_agg, pol, on='district')
tot_agg


# We have achieved our goal: put police and census data together at some level of granularity when we don't know were exactly the accidents took place. Let's explore the final result.

# In[ ]:


max_rec = tot_agg.total_records.max()
min_rec = tot_agg.total_records.min()
tot_rec = tot_agg.total_records.sum()
print(tot_rec, max_rec, min_rec)


# In[ ]:


tot_agg['records_pp'] = tot_agg.total_records / tot_agg.total_population
tot_agg[['district', 'total_records', 'total_population', 'records_pp']]


# Which makes sense because Downtown is reasonably an area with more police control.

# In[ ]:


tot_agg['pol_white_perc'] = tot_agg.White / tot_agg.total_records # percentages of the total police records, by race
tot_agg['pol_black_perc'] = tot_agg.Black / tot_agg.total_records
tot_agg['pol_hispanic_perc'] = tot_agg.Hispanic / tot_agg.total_records

tot_agg['records_pp_white'] = tot_agg.White / tot_agg.total_white  # crime rate by race
tot_agg['records_pp_black'] = tot_agg.Black / tot_agg.total_black
tot_agg['records_pp_hispanic'] = tot_agg.Hispanic / tot_agg.total_hispanic

tot_agg[['district', 'records_pp', 'total_white', 'total_black', 'total_hispanic', 
         'pol_white_perc', 'pol_black_perc', 'pol_hispanic_perc', 'records_pp_white', 'records_pp_black', 'records_pp_hispanic']]


# It would be interesting to see how other socio-demographic factors play a role in having different crime rates in different districts (*Why Downtown has such a high crime rate?*, etc). 
# 
# Another interesting question regards the different distribution of the population (the southern districts are mostly white citizens, while black citizens live in the northern ones)
# 
# I will reserve these studies to a future kernel since I don't have time to do so before the end of the competition.
# 
# 
# # Conclusions and next steps
# 
# This kernel has great margins of improvements and customization. The cases here presented (Dallas for the exposition of the base functions and Indianapolis for the case study) were chosen fairly randomly and I am particularly happy about being able to change an index on the top of the page and get similar results without too much work. All that being said, choices were made along the way and were biased by my curiosity, interest, and ability of execution. Therefore I see this as a first step and I really hope it will inspire someone to make it better or to perform a more accurate analysis.
# 
# One thing that I take away from this kernel is the following. Summary statistics and aggregations are necessary to peek into the message contained in very large datasets. However, it has to be kept in mind that the story they tell is always an approximation of a complex situation. **Different levels of granularity can tell different stories**. Pretty much in the same way people say *Africa is poor* or *Europe is rich*, completely ignoring that Africa is a very big place with very different stories to tell.
# 
# I say that because we often observe that the percentages of crimes by race are not proportional to the underlying population and any statistical test of goodness of fit will tell that no, that sample is not randomly extracted from the population. When this happens, the counter-argument is sometimes related to different crime rates and socioeconomic indicators, etc. All of this is true but they are all a simplification of a very complex argument, ignoring relations of cause and effect while focusing on correlations.
# 
# On the other side, the response of trained officials should be guaranteed by a democratic State and, if we observe that it is not uniform if the color of the skin of a suspect is different, then we have a problem. Here, we have seen that the police was behaving fairly uniformly in any use of force, except for the lethal use of force. This is my definition of injustice.
# 
# However, for the sake of completeness, even this result is partial. Not only a table does not contain the nuances of a situation but also does not contain all the counterfactuals. In other words, we don't know from this data how many times an officer could have used lethal force but didn't as well as we don't know what would have happened if the lethal force was not used. In my vision, a complete analysis would require that as well, or it will risk being biased in searching for a bias.
# 
# 
# Thank you for reading this far, please let the feedback come, it is a very important topic that requires the attention of everyone.
