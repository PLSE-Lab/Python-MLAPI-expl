#!/usr/bin/env python
# coding: utf-8

# **Objective**
# 
# The objective is to predict the project that a particular donor would donate towards. This will be used to decide what projects are recommended to a donor, in the hopes of increasing the likelihood of receiving a donation from the donor.
# 
# **Approach**
# 
# The approach here to predict whether the donor would donate to a new project is to use the attributes of projects that a donor has previously donated towards. This is based on the belief that past behaviour predicts future behaviour.
# 
# **Data**
# 
# Data version when code written: version 8
# 
# Load data and verify that it is loaded correctly.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime # convert string to date

import gc # for garbage collection to reduce RAM load

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# Load data
Donations = pd.read_csv('../input/Donations.csv', parse_dates=['Donation Received Date'])
Donors = pd.read_csv('../input/Donors.csv', dtype={'Donor Zip': 'str'})
Projects = pd.read_csv('../input/Projects.csv', parse_dates=['Project Posted Date', 'Project Expiration Date', 'Project Fully Funded Date'])
Resources = pd.read_csv('../input/Resources.csv')
Schools = pd.read_csv('../input/Schools.csv')
Teachers = pd.read_csv('../input/Teachers.csv', parse_dates=['Teacher First Project Posted Date'])
print('Data loaded')


# In[3]:


# Check data loaded properly - column type
print('Donations')
print(Donations.dtypes)

print(' ')
print('Donors')
print(Donors.dtypes)

print(' ')
print('Projects') # WARNING: Column Project Essay contains line breaks
print(Projects.dtypes)

print(' ')
print('Resources')
print(Resources.dtypes)

print(' ')
print('Schools')
print(Schools.dtypes)

print(' ')
print('Teachers')
print(Teachers.dtypes)


# **Preprocessing**
# 
# Each row in the column Project Subject Category Tree actually contains one or two category values. The different combinations of category values make it appear that there are more categories than there really are. Some preprocessing will be done to separate the individual category values. The same applies to the subcategory values in the column Project Subject Subcategory Tree.
# 
# Besides that, the data will be separated using a date and a sample will be taken. The date is required as a reference point. Relative to the date, the past data will define each donor's past behaviour. Projects that are live at the reference date will be used in training for flagging projects that a donor donates towards. Future projects that are not live at the reference date (Project Posted Date > reference date) can be used to validate the prediction model. The sampling is required because each donor will be flagged by whether they donated to a particular project, and the full data of about 2 million donors vs. 9,000 projects may be too large to handle or take too long to process.
# 
# One other condition is the number of past donations by a donor for inclusion in training. A donor with one or two past donations will not provide very useful insights into their donation behaviour or into generalising to other donors.
# 
# * Date = 2018-01-01
# * Subset approximate number of donors = 10,000
# * Subset approximate number of "current" live projects = 100
# * Seed value for random subset selection = 404
# * Minimum number of donations by donor for inclusion in training = 5
# 
# **Separating the category and subcategory values**

# In[4]:


# Select parameters for creating the subset later
Date = '2018-01-01'
SubsetNumDonors = 10000
SubsetNumProj = 100
RndSeed = 404
MinDonations = 5
print('Date: ' + Date)
print('Subset number of donors: ' + str(SubsetNumDonors))
print('Subset number of projects live at chosen date: ' + str(SubsetNumProj))
print('Random seed value for sampling subset: ' + str(RndSeed))
print('Minimum number of donations for inclusion in training: ' + str(MinDonations))


# In[5]:


# Process subject category
ProjCat = Projects[['Project ID', 'Project Subject Category Tree']]
ProjCat['Project Subject Category Tree'] = ProjCat['Project Subject Category Tree'].str.replace(', ', '--')
ProjCat['Project Subject Category Tree'] = ProjCat['Project Subject Category Tree'].str.replace('h--', 'h, ')
ProjCat.loc[:, 'Cat_Pri'],ProjCat.loc[:, 'Cat_Sec'] = ProjCat['Project Subject Category Tree'].str.split('--').str

#idx_contains_warmth = ProjCat['Project Subject Category Tree'].str.contains('Warmth', na=False)
#ProjCat[idx_contains_warmth].head()
ProjCat.head()


# In[6]:


#Process project subcategory
ProjSubcat = Projects[['Project ID', 'Project Subject Subcategory Tree']]
ProjSubcat['Project Subject Subcategory Tree'] = ProjSubcat['Project Subject Subcategory Tree'].str.replace(', ', '--')
ProjSubcat['Project Subject Subcategory Tree'] = ProjSubcat['Project Subject Subcategory Tree'].str.replace('h--', 'h, ')
ProjSubcat.loc[:, 'Subcat_Pri'],ProjSubcat.loc[:, 'Subcat_Sec'] = ProjSubcat['Project Subject Subcategory Tree'].str.split('--').str

ProjSubcat.head()


# In[7]:


#Replace the category and subcategory columns with the new split columns
Projects_adj = (Projects.merge(ProjCat.drop(columns=['Project Subject Category Tree']), on='Project ID', how='inner')
                .merge(ProjSubcat.drop(columns=['Project Subject Subcategory Tree']), on='Project ID', how='inner')
               .drop(columns=['Project Subject Category Tree'])
               .drop(columns=['Project Subject Subcategory Tree']))

Projects_adj.head()


# **Create training data**
# 
# The donors for training data are found by:
# 1. Filter the projects by the reference date on Project Fully Funded Date and Project Expiration Date (fully funded or expire after the reference date)
# 2. Filter the donations by the reference date on Donation Received Date (received before the reference date)
# 3. Count the number of donations by each donor in the filtered donation table
# 4. Filter out donors who do not have the minimum number of donations
# 5. Select a random subset of donors
# 
# The live projects (as at the reference date) for training data are found by:
# 1. Filter projects by the reference date on Project Posted Date (posted on or before reference date), Project Fully Funded Date and Project Expiration Date (fully funded or expire on or after the reference date)
# 2. Select a random subset of live projects
# 
# The final training table is created by:
# 1. Cross join the subset of donors with the subset of live projects. As the current data do not show which projects a donor is aware of, it is assumed that each donor is aware of all live projects and only donates to the ones they are most interested in.

# In[8]:


#Filter projects by date
print('Date: ' + Date)

# Seed value for random subset selection = 404
Projects_t = Projects_adj[(Projects_adj['Project Fully Funded Date'] < Date) | (Projects_adj['Project Expiration Date'] < Date)]
Projects_t.head()


# In[9]:


#Filter donations to those associated with the historic set Project ID
#TrainingProjectID = pd.DataFrame(Projects_t['Project ID'])
Donations_t = Donations[(Donations['Donation Received Date'] < Date)]
Donations_t.head()


# In[10]:


print('Subset number of donors: ' + str(SubsetNumDonors))
print('Minimum number of donations for inclusion in training: ' + str(MinDonations))
print('Random seed value for sampling subset: ' + str(RndSeed))

#Count the number of donations per donor in the training set
#DonationsPerDonor = Donations_t[['Donor ID', 'Donation ID']].groupby(['Donor ID']).agg('count')
#DonationsPerDonor.head()

#Filter out donors with not enough donations in the data
ValidDonors = Donations_t.groupby(['Donor ID']).filter(lambda x: len(x) >= MinDonations)
#ValidDonors.head()

#Choose a random subset of donors who have made enough donations
Donors_t = (Donors.merge(pd.DataFrame(ValidDonors['Donor ID']), on='Donor ID', how='inner')
           .sample(n=SubsetNumDonors, random_state=RndSeed))
Donors_t.head()


# In[11]:


print('Subset number of projects live at chosen date: ' + str(SubsetNumProj))
print('Random seed value for sampling subset: ' + str(RndSeed))

#Find live projects at reference date
LiveProjects = (Projects_adj[(Projects_adj['Project Posted Date'] <= Date) & 
                            ((Projects_adj['Project Fully Funded Date'] >= Date) | (Projects_adj['Project Expiration Date'] >= Date))]
               .sample(n=SubsetNumProj, random_state=RndSeed))
#LiveProjects[LiveProjects['Project Fully Funded Date'].isnull()].head()
LiveProjects.head()


# In[12]:


#Cross join donors with live projects
Donors_t_tmp = Donors_t
Donors_t_tmp['Cross_join_key'] = 1

LiveProjects_tmp = LiveProjects
LiveProjects_tmp['Cross_join_key'] = 1

DonorVsLiveProj = (Donors_t_tmp.merge(LiveProjects_tmp, on='Cross_join_key', how='outer')
                   .drop(columns=['Cross_join_key'])
                  )

del Donors_t_tmp
del LiveProjects_tmp
gc.collect()
print(DonorVsLiveProj.shape)


# **Variable calculation **
# 
# Suppose the name "donor-project pair" is used to refer to a donor and a project which the donor may or may not donate towards. The following variables can be calculated from a donor's history and are the same for any donor-project pair in the training set:
# 1. Days_since_last_donation: the number of days since the donor last made a donation
# 2. Num_hist_donations: the number of donations made by a donor in the training set (i.e. not the total number of donations over all time)
# 3. Avg_days_between_donations: the average number of days between a donor's donations
# 4. Avg_donations_per_proj: the average number of donations made per project
# 5. Interest_in_optional: how interested a donor is in giving optional donations; it is the fraction of donations by the donor that include optional donation
# 
# The following variables can be calculated from a donor's history and the exact value to use depends on the donor-project pair in the training set:
# 6. Interest_in_cat: how interested a donor is in a particular subject category; it is the fraction of donations by the donor that are in the subject category
# 7. Interest_in_subcat: how interested a donor is in a particular subject subcategory; it is the fraction of donations by the donor that are in the subject subcategory
# 8. Interest_in_res_cat: how interested a donor is in a particular resource category; it is the fraction of donations by the donor that are in the resource category
# 9. Interest_in_grade_cat: how interested a donor is in a particular grade category; it is the fraction of donations by the donor that are in the grade category
# 10. Interest_in_metro: how interested a donor is in a particular metro type; it is the fraction of donations by the donor that are in the metro type
# 11. Interest_in_proj_type: how interested a donor is in a particular project type; it is the fraction of donations by the donor that are in the project type
# 12. Interest_in_month: how interested a donor is in donating in a particular month; it is the fraction of donations by the donor that are in the month
# 13. Total_project_interest: overall interest of a donor in a project; it is the sum of items 10 to 15
# 
# The following variables are independent of a donor's history and the exact value depends on the donor-project pair in the training set:
# 14. Donor_vs_school_city: value is 1 if the donor and school are in the same city and state, otherwise 0
# 15. Donor_vs_school_state: value is 1 if the donor and school are in the same state, otherwise 0
# 16. Project_posted_month: the month of Project Posted Date
# 17. Project_posted_day_of_month: the day of the month of Project Posted Date
# 
# 

# In[13]:


#Create variables
#Join tables to get donor and school columns together
DonorVsDonation = Donors_t.merge(Donations_t, on='Donor ID', how='left')
DonorVsDonationVsProject = DonorVsDonation.merge(Projects_t, on='Project ID', how='left')
DonorVsDonationVsProjectVsSchool = DonorVsDonationVsProject.merge(Schools, on='School ID', how='left')

print('Donor vs donation vs project vs school table dimensions')
print(DonorVsDonationVsProjectVsSchool.shape)


# In[14]:


#Historical donations count in the data
NumHistDonations = (DonorVsDonation[['Donor ID', 'Donation ID']]
                    .groupby(['Donor ID'])
                    .agg('count')
                    .reset_index()
                    .rename(columns={'Donation ID': 'Num_hist_donations'})
                   )
NumHistDonations.head()


# In[15]:


#Most recent donation date in the data
LastDonations = (DonorVsDonation[['Donor ID', 'Donation Received Date']]
                     .groupby(['Donor ID'])
                     .agg('max')
                     .reset_index()
                     .rename(columns={'Donation Received Date': 'Last_donation_date'})
                    )

#Days since most recent donation
LastDonations['Days_since_last_donation'] = (datetime.strptime(Date, '%Y-%m-%d') - LastDonations['Last_donation_date']) / np.timedelta64(1, 'D')
LastDonations.head()


# In[16]:


#Earliest donation date in the data
EarliestDonations = (DonorVsDonation[['Donor ID', 'Donation Received Date']]
                     .groupby(['Donor ID'])
                     .agg('min')
                     .reset_index()
                     .rename(columns={'Donation Received Date': 'Earliest_donation_date'})
                    )

#Average number of days between donations
DonationFreq = (NumHistDonations
                .merge(LastDonations, on='Donor ID', how='inner')
                .merge(EarliestDonations, on='Donor ID', how='inner')
               )

DonationFreq['Avg_days_between_donations'] = (((DonationFreq['Last_donation_date'] - DonationFreq['Earliest_donation_date']) / DonationFreq['Num_hist_donations'])
                                              / np.timedelta64(1, 'D')
                                             )
del EarliestDonations
gc.collect()
DonationFreq.head()


# In[17]:


#Average number of donations per project
NumDonationsPerProj = (DonorVsDonation[['Donor ID', 'Project ID', 'Donation ID']]
                       .groupby(['Donor ID', 'Project ID'])
                       .agg('count')
                       .reset_index()
                       .rename(columns={'Donation ID': 'Num_donations_per_proj'})
                      )

AvgDonationsPerProj = (NumDonationsPerProj[['Donor ID', 'Num_donations_per_proj']]
                       .groupby(['Donor ID'])
                       .agg('mean')
                       .reset_index()
                       .rename(columns={'Num_donations_per_proj': 'Avg_donations_per_proj'})
                      )

del NumDonationsPerProj
gc.collect()
AvgDonationsPerProj.head()


# In[18]:


# Donor interest in optional donations
NumOptional = (DonorVsDonation[['Donor ID', 'Donation Included Optional Donation']]
                    .replace({'Yes': 1, 'No': 0})
                    .groupby(['Donor ID'])
                    .agg('sum')
                    .reset_index()
                    .rename(columns={'Donation Included Optional Donation': 'Num_optional'})
                   )

InterestInOptional = NumHistDonations.merge(NumOptional, on='Donor ID', how='left')
InterestInOptional['Interest_in_optional'] = InterestInOptional['Num_optional'] / InterestInOptional['Num_hist_donations']

del NumOptional
gc.collect()
InterestInOptional.head()


# In[19]:


#Interest in subject category
InterestInCat_tmp = pd.concat(
                    [DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Cat_Pri']].rename(columns={'Cat_Pri': 'Cat'}),
                     DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Cat_Sec']].rename(columns={'Cat_Sec': 'Cat'})]
                    , axis=0
                    , join='outer'
                    )

NumCat = (InterestInCat_tmp
          .dropna()
          .groupby(['Donor ID', 'Cat'])
          .agg('count')
          .reset_index()
          .rename(columns={'Donation ID': 'Num_cat'})
         )

InterestInCat = NumHistDonations.merge(NumCat, on='Donor ID', how='left')
InterestInCat['Interest_in_cat'] = InterestInCat['Num_cat'] / InterestInCat['Num_hist_donations']

del InterestInCat_tmp
del NumCat
gc.collect()
InterestInCat.head()


# In[20]:


#Interest in subject subcategory
InterestInSubcat_tmp = pd.concat(
                    [DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Subcat_Pri']].rename(columns={'Subcat_Pri': 'Subcat'}),
                     DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Subcat_Sec']].rename(columns={'Subcat_Sec': 'Subcat'})]
                    , axis=0
                    , join='outer'
                    ).dropna()

NumSubcat = (InterestInSubcat_tmp
          .groupby(['Donor ID', 'Subcat'])
          .agg('count')
          .reset_index()
          .rename(columns={'Donation ID': 'Num_subcat'})
         )

InterestInSubcat = NumHistDonations.merge(NumSubcat, on='Donor ID', how='left')
InterestInSubcat['Interest_in_subcat'] = InterestInSubcat['Num_subcat'] / InterestInSubcat['Num_hist_donations']

del InterestInSubcat_tmp
del NumSubcat
gc.collect()
InterestInSubcat.head()


# In[21]:


#Interest in resource category
InterestInRes_tmp = (DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Project Resource Category']]
                     .groupby(['Donor ID', 'Project Resource Category'])
                     .agg('count')
                     .reset_index()
                     .rename(columns={'Donation ID': 'Num_res'})
                    )

InterestInRes = NumHistDonations.merge(InterestInRes_tmp, on='Donor ID', how='left')
InterestInRes['Interest_in_res'] = InterestInRes['Num_res'] / InterestInRes['Num_hist_donations']

del InterestInRes_tmp
gc.collect()
InterestInRes.head()


# In[22]:


#Interest in grade level category
InterestInGrade_tmp = (DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Project Grade Level Category']]
                     .groupby(['Donor ID', 'Project Grade Level Category'])
                     .agg('count')
                     .reset_index()
                     .rename(columns={'Donation ID': 'Num_grade'})
                    )

InterestInGrade = NumHistDonations.merge(InterestInGrade_tmp, on='Donor ID', how='left')
InterestInGrade['Interest_in_grade'] = InterestInGrade['Num_grade'] / InterestInGrade['Num_hist_donations']

del InterestInGrade_tmp
gc.collect()
InterestInGrade.head()


# In[23]:


#Interest in project type
InterestInProjType_tmp = (DonorVsDonationVsProject[['Donor ID', 'Donation ID', 'Project Type']]
                          .groupby(['Donor ID', 'Project Type'])
                          .agg('count')
                          .reset_index()
                          .rename(columns={'Donation ID': 'Num_proj_type'})
                         )

InterestInProjType = NumHistDonations.merge(InterestInProjType_tmp, on='Donor ID', how='left')
InterestInProjType['Interest_in_proj_type'] = InterestInProjType['Num_proj_type'] / InterestInProjType['Num_hist_donations']

del InterestInProjType_tmp
gc.collect()
InterestInProjType.head()


# In[24]:


#Interest in school metro type
InterestInMetro_tmp = (DonorVsDonationVsProjectVsSchool[['Donor ID', 'Donation ID', 'School Metro Type']]
                          .groupby(['Donor ID', 'School Metro Type'])
                          .agg('count')
                          .reset_index()
                          .rename(columns={'Donation ID': 'Num_metro'})
                         )

InterestInMetro = NumHistDonations.merge(InterestInMetro_tmp, on='Donor ID', how='left')
InterestInMetro['Interest_in_metro'] = InterestInMetro['Num_metro'] / InterestInMetro['Num_hist_donations']

del InterestInMetro_tmp
gc.collect()
InterestInMetro.head()


# In[ ]:


#Interest in month
InterestInMonth_tmp = DonorVsDonation[['Donor ID', 'Donation ID', 'Donation Received Date']]
InterestInMonth_tmp['Donation_month'] = InterestInMonth_tmp['Donation Received Date'].dt.month

InterestInMonth_tmp_2 = (InterestInMonth_tmp.drop(columns=['Donation Received Date'])
                         .groupby(['Donor ID', 'Donation_month'])
                         .agg('count')
                         .reset_index()
                         .rename(columns={'Donation ID': 'Num_donations_in_month'})
                        )

InterestInMonth = NumHistDonations.merge(InterestInMonth_tmp_2, on='Donor ID', how='left')
InterestInMonth['Interest_in_month'] = InterestInMonth['Num_donations_in_month'] / InterestInMonth['Num_hist_donations']

del InterestInMonth_tmp
del InterestInMonth_tmp_2
gc.collect()
InterestInMonth.head()


# **Finish creating the training set**
# 
# The final training set is created by attaching the variable values to the corresponding project attributes in the training set and removing the columns that will not be used.
# 
# The following columns are not used in the training set:
# 1. Project Essay - no skill in natural language processing to cluster the many distinct values into a few distinct values
# 2. Project Short Description - same as above
# 3. Project Need Statement - same as above
# 4. Project Title - same as above
# 5. School Name - same as above
# 6. Resource Item Name - same as above
# 7. School District - too many distinct values and no idea how to cluster the many distinct values into a few distinct values
# 8. School Zip - too many distinct values; data is obscured for privacy reasons, and it would have been nice to be able to match donors and schools on zip code level
# 9. Donor Zip - too many distinct values

# #Kernel crashes at this stage
# 
# #To do to DonorVsLiveProj
# #1. Calculate project specific variables
# #2. Attach pre-calculated variable values according to matching attributes
# #3. Remove columns that will not be used
# 
# #Function to check matching state
# def MatchState(row):
#     if row['Donor State'] == row['School State']:
#         return 1
#     return 0
# 
# #Function to check matching state and city
# def MatchCity(row):
#     if row['Donor_vs_school_state'] == 1 and row['Donor City'] == row['School City']:
#         return 1
#     return 0
# 
# #Whole training set - drop columns
# WholeTrainigSet_tmp = (DonorVsLiveProj
#                        .merge(Schools, on='School ID', how='left')
#                        .merge(Resources, on='Project ID', how='left')
#                        .merge(Teachers, on='Teacher ID', how='left')
#                        .drop(columns=['Project Essay', 'Project Short Description', 'Project Need Statement', 'Project Title', 'School Name', 
#                                       'Resource Item Name', 'School District', 'School Zip', 'Donor Zip'])
#                       )
# 
# #Whole training set - calculate project specific variables
# WholeTrainingSet_tmp['Donor_vs_school_state'] = WholeTrainigSet_tmp.apply(lambda row: MatchState(row), axis=1)
# WholeTrainigSet_tmp['Donor_vs_school_city'] = WholeTrainigSet_tmp.apply(lambda row: MatchCity(row), axis=1)
# 
# #Project month and day of month
# WholeTrainingSet_tmp['Project_posted_month'] = pd.Series(WholeTrainingSet_tmp['Project Posted Date'].dt.month, index=Projects_t.index)
# WholeTrainingSet_tmp['Project_posted_day_of_month'] = pd.Series(WholeTrainingSet_tmp['Project Posted Date'].dt.day, index=Projects_t.index)
# 
# #Apply subject category interest to primary and secondary categories
# TotalInterestInCat_tmp = (DonorVsLiveProj[['Donor ID', 'Project ID', 'Cat_Pri', 'Cat_Sec']]
#                           .merge(InterestInCat, left_on=['Donor ID', 'Cat_Pri'], right_on=['Donor ID', 'Cat'])
#                           .merge(InterestInCat, left_on=['Donor ID', 'Cat_Sec'], right_on=['Donor ID', 'Cat'])
#                          )
# TotalInterestInCat_tmp['Total_interest_in_cat'] = TotalInterestInCat_tmp['Interest_in_cat_x'] + TotalInterestInCat_tmp['Interest_in_cat_y']
# TotalInterestInCat = TotalInterestInCat_tmp[['Donor ID', 'Project ID', 'Total_interest_in_cat']]
# 
# TotalInterestInSubcat_tmp = (DonorVsLiveProj[['Donor ID', 'Project ID', 'Subcat_Pri', 'Subcat_Sec']]
#                              .merge(InterestInSubcat, left_on=['Donor ID', 'Subcat_Pri'], right_on=['Donor ID', 'Subcat'])
#                              .merge(InterestInSubcat, left_on=['Donor ID', 'Subcat_Sec'], right_on=['Donor ID', 'Subcat'])
#                             )
# TotalInterestInSubcat_tmp['Total_interest_in_subcat'] = TotalInterestInSubcat_tmp['Interest_in_subcat_x'] + TotalInterestInSubcat_tmp['Interest_in_subcat_y']
# TotalInterestInSubcat = TotalInterestInSubcat_tmp[['Donor ID', 'Project ID', 'Total_interest_in_subcat']]
# 
# #Whole training set - attach pre-calculated values according to attributes
# WholeTrainingSet = (DonorVsLiveProj
#                     .merge(NumHistDonations, on='Donor ID', how='left')
#                     .merge(LastDonations, on='Donor ID', how='left')
#                     .merge(DonationFreq, on='Donor ID', how='left')
#                     .merge(AvgDonationsPerProj, on='Donor ID', how='left')
#                     .merge(InterestInOptional, on='Donor ID', how='left')
#                     .merge(TotalInterestInCat, left_on=['Donor ID', 'Project ID'], right_on=['Donor ID', 'Project ID'], how='left')
#                     .merge(TotalInterestInSubcat, left_on=['Donor ID', 'Project ID'], right_on=['Donor ID', 'Project ID'], how='left')
#                     .merge(InterestInRes, left_on=['Donor ID', 'Project Resource Category'], right_on=['Donor ID', 'Project Resource Category'], how='left')
#                     .merge(InterestInGrade, left_on=['Donor ID', 'Project Grade Level Category'], right_on=['Donor ID', 'Project Grade Level Category'], how='left')
#                     .merge(InterestInProjType, left_on=['Donor ID', 'Project Type'], right_on=['Donor ID', 'Project Type'], how='left')
#                     .merge(InterestInMetro, left_on=['Donor ID', 'School Metro Type'], right_on=['Donor ID', 'School Metro Type'], how='left')
#                     .merge(InterestInMonth, left_on=['Donor ID', 'Project_posted_month'], right_on=['Donor ID', 'Donation_month'], how='left')
#                    )
# 
# #Clean up columns one last time
# 
# del WholeTrainigSet_tmp
# del TotalInterestInCat_tmp
# del TotalInterestInSubcat_tmp
# gc.collect()
# 
# print(WholeTrainingSet.dtypes)
# WholeTrainingSet.head()

# 
# 17. Total_project_interest: overall interest of a donor in a project; it is the sum of items 10 to 15

# **Prediction model**
# 
# Predictors used in model:
# 

# In[ ]:


#Prediction model


# **Possible improvements**
# 
# The following steps could be taken to improve upon the work presented in this notebook:
# 1. The columns Project Essay, Project Short Description, Project Need Statement, Project Title, School Name, and Resource Item Name were ignored because of my absence of skill in natural language processing. Natural language processing could be used to categorise these columns to reduce the number of possible values, thereby making them useful for the prediction model.
# 2. Potentially useful additional data
#     * Classify donors and schools based on the socioeconomic status of their area, possibly through census data
#     * Classify schools as private or public
#     * Classify schools as religious associated or not religious associated
#     * Classify schools as single-sex or co-ed
# 3. Cluster similar donors or projects, then use cluster ID as a predictor
# 4. Find variables that are more predictive
# 5. Better optimisation of prediction model parameters
# 6. The work here assumed that each donor is aware of all live projects when deciding to make donation. It may be useful to estimate the live projects of which a donor is aware. For example, if donors are sent emails about live projects, the exact projects in the email and the email date could be recorded.
# 
