#!/usr/bin/env python
# coding: utf-8

#   **First script **
# 
# On this script we  modify the data in order to order to add useful columns or delete unnecessary ones, and we merge the date of the five files we are using into one single data frame that we save as a CSV file that we will use later.  (this last part works if the script is launched localy but doesn't work if launched in this platform)
# 

# In[5]:


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:33:09 2018

@author: machinelearning
"""
import time
#import numpy as np
import pandas as pd
#from pandas.io.json import json_normalize
#import matplotlib.pyplot as plt

timestart = time.time()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 40)

# Create Data Frames from the given data
donations = pd.read_csv(('../input/Donations.csv'), 
                       parse_dates=["Donation Received Date"])
donors = pd.read_csv(('../input/Donors.csv'), dtype={'Donor_Zip': str},
                     low_memory=False)
schools = pd.read_csv(('../input/Schools.csv'), dtype={'School_Zip': str},
                      error_bad_lines=False)
teachers = pd.read_csv(('../input/Teachers.csv'), error_bad_lines=False)
projects = pd.read_csv(('../input/Projects.csv'), error_bad_lines=False, 
                       warn_bad_lines=False, 
                       parse_dates=["Project Posted Date",
                                    "Project Fully Funded Date", 
                                    "Project Expiration Date"])

# Drop unnecessary columns 
# (or that we simply decided not to use to avoid over-complexity)
teachers = teachers.drop('Teacher First Project Posted Date', axis=1)
donations = donations.drop('Donation ID', axis=1)
projects = projects.drop(['Project Title','Project Essay',
                          'Project Posted Date',
                          'Project Current Status',
                          'Project Fully Funded Date',
                          'Project Short Description',
                          'Project Need Statement'], axis=1)

# Remove or replace NaN values
teachers = teachers.dropna()
donors['Donor City'] = donors['Donor City'].fillna('Unknown')
donors = donors.dropna()#(subset=['Donor Zip'])
schools['School City'] = schools['School City'].fillna('Unknown')
schools = schools.dropna()
projects = projects.dropna()

# Modify or Add columns that could be useful later 
projects['Is_teachers_first_project'] = projects[
        'Teacher Project Posted Sequence']==1
donations['Type_of_Donor'] = pd.cut(donations['Donor Cart Sequence'],
         [0, 1, 6, 30000], labels=['Once', 'Occational', 'Regular'])
schools['School_Percentage_Free_Lunch'] = pd.cut(schools[
        'School Percentage Free Lunch'], 5, labels=False)
schools['School_Percentage_Free_Lunch'] = schools[
        'School_Percentage_Free_Lunch'].fillna(0.0).astype(int)
projects['Project Cost'] = (projects['Project Cost'].replace(
        '[\$,)]','', regex=True).replace('[(]','-', regex=True).astype(float))

projects['Project Subject Category Tree'] = projects[
        'Project Subject Category Tree'].str.replace(',', '_')
projects['Project Subject Subcategory Tree'] = projects[
        'Project Subject Subcategory Tree'].str.replace(',', '_')

# Drop more unnecessary columns 
projects = projects.drop('Teacher Project Posted Sequence', axis=1)
schools = schools.drop('School Percentage Free Lunch', axis=1) 

# Replace spaces by '_' on column names
donations.columns = donations.columns.str.replace(" ", "_")
donors.columns = donors.columns.str.replace(" ", "_")
schools.columns = schools.columns.str.replace(" ", "_")
teachers.columns = teachers.columns.str.replace(" ", "_")
projects.columns = projects.columns.str.replace(" ", "_")

# Merge donation data with donor data 
donors_donations = donations.merge(donors, on='Donor_ID', how='inner')


# Merge projects data with teachers data 
teachers_projects = projects.merge(teachers, on='Teacher_ID', how='inner')


# Merge the rest (except Resourses):
TPS = teachers_projects.merge(schools, on='School_ID', how='inner')
TPSDD = TPS.merge(donors_donations, on='Project_ID', how='inner')

print('\n All merged DataFrame (except Resourses):')
print(TPSDD.head(6))
print(TPSDD.describe(include='all'))


# Here we create a CSV file with our merged Data Frame. In reality the CSV file should be created with all the lines of the Data Frame. But since in Kaggle the Kernels only give you 1GB of disk space and the output file should have 1.8 GB, we are taking here only the first 2255000 lines.  

# In[ ]:



# Create a single modified and unified CSV file

#TPSDD.to_csv('TPSDD.csv')
TPSDD.iloc[0:2255000, :].to_csv('TPSDD.csv')


timeend = time.time()
print('\n----------------------------------------------')
print(' Run time of the script:')
print('  %d minutes ' % ((timeend - timestart)/60))
print('----THE END----\n ')


# Note: 
# We considered that Resourses.csv wasn't useful for the purpose of the classification, so we don't use it.
