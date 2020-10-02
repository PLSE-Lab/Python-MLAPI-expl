#!/usr/bin/env python
# coding: utf-8

# This kernel is using the data from DonorChoose.org and is trying to answer the following questions:
#  
# If we select donors who made exactly two donations, how many of them made a first donation to a school in the same city where they live?  how many in the second donation?
# 
# How many donors, who made a local donation in their fist donation, changed to a school in a different city for the second donation?

# In[2]:


import numpy as np 
import pandas as pd 


# In[3]:


projects = pd.read_csv("../input/Projects.csv")
donations = pd.read_csv("../input/Donations.csv")
donors = pd.read_csv("../input/Donors.csv")
schools = pd.read_csv("../input/Schools.csv")


# In[4]:


#Counting the number of donations for each donor
DonorsCount1 = pd.merge(donors,donations,how='inner',on='Donor ID').groupby('Donor ID')
DonorsCount2 = DonorsCount1.count()['Donor City'].reset_index()
DonorsCount2['NoDonations'] = DonorsCount2['Donor City']


# In[5]:


#Merging donors table with DonorsCount2
#And adding a new column with the number of donation for each row
FinalDonors = pd.merge(donors,DonorsCount2,how = 'inner', on = 'Donor ID')


# In[6]:


#To create a dataframe containing the corresponding project for each donation
DonationsProjects = pd.merge(donations,projects[['Project ID','School ID']],how ="left",on="Project ID")


# In[7]:


#To create a dataframe containing the corresponding donor for each donation 
DonationsProjectsDonors = pd.merge(DonationsProjects,donors, how = "left", on = "Donor ID")


# In[8]:


#To create a dataframe containing the corresponding school for each donation
DonationsProjectsDonorsSchools = pd.merge(DonationsProjectsDonors,schools, how = "left", on = "School ID")


# In[9]:


#New column indicating if the donor city is the same as the school city
DonationsProjectsDonorsSchools["IsLocal"]=DonationsProjectsDonorsSchools.apply(lambda row: row["School City"] == row["Donor City"],axis=1)


# In[12]:


#To add a new column with the number of donations for each donor
DonationsProjectsDonorsSchoolsNoDonations = pd.merge(DonationsProjectsDonorsSchools,FinalDonors[["Donor ID","NoDonations"]], how = "left", on = "Donor ID")
df = DonationsProjectsDonorsSchoolsNoDonations


# In[14]:


#Donors with exact 2 donations
#This dataframe contains two records for each donor, first and second donation
df2 = pd.DataFrame(df[df['NoDonations']==2])[["Donor ID","Donation Received Date",'IsLocal']]


# In[15]:


FirstDonation = pd.DataFrame(df2.groupby('Donor ID').min()).reset_index()
SecondDonation = pd.DataFrame(df2.groupby('Donor ID').max()).reset_index()


# In[16]:


#First and second donation in the same row
FirstAndSecondDonation = pd.merge(FirstDonation,SecondDonation,how="inner",on='Donor ID')


# In[17]:


FirstAndSecondDonation.head(10)


# Percentage of donors who donated to a local school (in the same  donor's city) in their first donation

# In[18]:


len(FirstDonation[FirstDonation['IsLocal']==True]) / len(FirstDonation)


# Percentage of donors who donated to a local school (in the same donor's city) in their second donation

# In[19]:


len(SecondDonation[SecondDonation['IsLocal']==True]) / len(SecondDonation)


# Percentage of donors who changed from a local school in their first donation to a non-local school in their second donation

# In[20]:


len(FirstAndSecondDonation[(FirstAndSecondDonation['IsLocal_x']==True) & (FirstAndSecondDonation['IsLocal_y']==False)])/len(FirstAndSecondDonation)


# Percentage of donors who changed from a non-local school in their first donation to a local school in their second donation

# In[21]:


len(FirstAndSecondDonation[(FirstAndSecondDonation['IsLocal_x']==False) & (FirstAndSecondDonation['IsLocal_y']==True)])/len(FirstAndSecondDonation)


# **Conclusions** (if the code is correct and I haven't done any mistake)
# 
# First time donors who donated to a local school will donate to a local school in their second donation too.
# 
# 10% of the first time donors who donated to a school in a different city to where they live, changed to a local school in their second donation.
# 
# A more general conclusion could be (although this can't be concluded for this analysis only):
# 
# Donors tend to donate to a closer schools as they make more donations.
# 
