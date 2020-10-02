#!/usr/bin/env python
# coding: utf-8

# # Exploring Open Data for CPE Challenge 
# After looking through some of the EDA kernels for this competition, I noticed that the data is not complete for all police departments in questions. Some of them are missing Use of Force data, others don't have complete police district shapefiles. Perhaps the point is to find all the data if it is available. So, the purpose of this Kernel is to collate additional datasets (outside of the provided ones) that are relevant to the analysis in the competition. While I provide these additional files as Kaggle datasets, you are welcome to download and use local copies based on the link I share in every section.
# 
# This kernel is split into 6 parts, with each part covering relevant, publicly available data pertaining to each city's police department:
# 1. [Boston - Dept_11](#boston)
# 2. [Indianapolis - Dept_23](#india)
# 3. [Charlotte-Mecklenburg (a.k.a. CMPD) - Dept_35](#charlotte)
# 4. [Austin - Dept_3700027](#austin)
# 5. [Dallas - Dept_37-00049](#dallas)
# 6. [Seattle - Dept_49](#sea)

# <a id="boston"></a>
# ## 1. Boston
# What is available in the competition folder is the census data and the police distcits borders shapefiles. What is missing is the actual police reports to show which police officers responded to which suspects/individuals. Luckily, City of Boston has a website called [Analyze Boston](https://data.boston.gov/) that contains datasets on many aspects of city operations, including public safety. Specifically, there is a group of datasets here titled "[BPD FIELD INTERROGATION AND OBSERVATION (FIO)](https://data.boston.gov/dataset/boston-police-department-fio)" that go into details of ethnicity, race, gender, and age of private individuals that Boston Police Department interracted with around 2016. 

# In[ ]:


import pandas as pd
import seaborn as sns

# Looking at Boston police interrogation and observation files
individuals = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/1-Boston/Boston_Field_Contact_Demographics_2016.csv')
calls = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/1-Boston/Boston_Field_Contact_Report_2016.csv')


# The individuals file provides demographics of persons of interest during the interactions of the officers with private citizens, while the calls file provides details of the interactions along with the names of the officers (though doesn't give officer's demographics).

# In[ ]:


print('There were {} Boston police interactions with individuals in 2016'.format(individuals.shape[0]))


# In[ ]:


# Quick bargraph of the race distribution
sns.countplot(y='race', data=individuals, orient='h');


# <a id="india"></a>
# ## 2. Indianapolis
# For Indianapolis we again have the police districts in shapefiles and the census data, but we are missing the police activity. For that there is [Project Comport](https://www.projectcomport.org/) website that has a Use of Force report for Indianapolis, along with Officer Invovled Shootings. These files cover time period from 2014 to 2018 - so you can even  do a longitudional study.

# In[ ]:


# Looking at Boston police interrogation and observation files
ind_uof = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/2-Indianapolis/Indianapolis_Police_Use_of_Force.csv')
ind_psi = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/2-Indianapolis/Indianapolis_Police_Shootings.csv')


# In[ ]:


print('There are {} records in Use of Force file for Indianapolis, from 2014 to 2018'.format(ind_uof.shape[0]))


# Both files need some cleaning, but should be pretty straight foraward to get useful information from them. For example, narrowing down to just incidents in 2016 could be done as follows:

# In[ ]:


# Convert date column from string to datetime
ind_uof['occurredDate'] = pd.to_datetime(ind_uof['occurredDate'])

# Plot distribution by race of incidents in 2016
sns.countplot(y='residentRace', data=ind_uof[ind_uof['occurredDate'].dt.year == 2016], orient='h');


# <a id="charlotte"></a>
# ## 3. Charlotte-Mecklenburg
# Here Kaggle gives us Officer Invovled Shooting data and the division offices (not sure how useful that is) in the shapefiles. Lucky for us, there is data.gov, where [CMPD posted shapefiles](https://catalog.data.gov/dataset/cmpd-police-divisions) for the police districts in the metro area.  However, I wasn't able to find the Use of Force reports. They may be available on the [City of Charlotte website](http://charlottenc.gov/Pages/Home.aspx), but the resource seems to be down at the time of writing (i.e. I can't load any pages on their website - if you can please let me know).

# In[ ]:


# Read and plot the shapefile
import geopandas as gpd 

charlotte_map = gpd.GeoDataFrame.from_file('../input/cpe-external-data/cpe_external_data/cpe_external_data/3-Charlotte/CMPD_Police_Divisions.shp')
charlotte_map.plot();


# <a id="austin"></a>
# ## 4. Austin
# Kaggle provided a fairly complete dataset for Austin. The only item I would add is the Officer Invovled Shootings (OIS), which is available on the [open data portal for Austin, Texas](https://data.austintexas.gov/Public-Safety/2008-17-OIS-Subjects/u2k2-n8ez). These 3 CSV files contain details of the incidents, officers involved, and the private citizens invovled, including demographic variables. 

# In[ ]:


# Load Austin datasets
aus_ois = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/4-Austin/Austin_2008-17_OIS_Incidents.csv')
aus_ois_officers = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/4-Austin/Austin_2008-17_OIS_Officers.csv')
aus_ois_subjects = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/4-Austin/Austin_2008-17_OIS_Subjects.csv')


# _Case #_ can be used to join the 3 datasets together for a complete picture of peope involved in each incident. For now, I'll just do a quick bargraph.

# In[ ]:


sns.countplot(y='Subject Race/Ethnicity', data=aus_ois_subjects, orient='h');


# <a id="dallas"></a>
# ## 5. Dallas
# First, it's important to note that the shapefiles posted for Dallas police divisions are a _subset_ of all police divisions. Specifically, theses are "EPIC Focus Areas" as noted [here](https://gis.dallascityhall.com/shapefileDownload.aspx). Now, I wasn't able to find the shapefiles for all police divisions. Best I could do is this [interactive map](http://dpdcau.maps.arcgis.com/apps/InformationLookup/index.html?appid=b9f5abf9068b4d9c8c705a00895ce9c8) on the Dallas PD website. However, I was able to dig up the [Officer Invovled Shooting data](https://www.dallasopendata.com/Public-Safety/Dallas-Police-Officer-Involved-Shootings/4gmt-jyx2) and a dataset on the [Use of Force](https://www.dallasopendata.com/Public-Safety/Police-Response-to-Resistance-2016/99fn-pvaf). The latter actually has demographics for both police officer and private individual built in.

# In[ ]:


# Load Dallas datasets
dal_ois = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/5-Dallas/Dallas_Police_Officer-Involved_Shootings.csv')
dal_uof = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/5-Dallas/Police_Response_to_Resistance_-_2016.csv')


# In[ ]:


sns.countplot(y='CitRace', data=dal_uof, orient='h');


# Use of force file has all the demograhics you need. On first look, I thougt there was no demographic data in the OIS file. 

# In[ ]:


dal_ois.head()


# However, after examinig the Subject(s) column more closely, I noticed the suffix that identifies skin complexion and sex. For example, in the first row we have Latin/Male, in fourth it's 3 Black/Males. So, there is something to think your teeth into here after a bit of feature extraction.

# <a id="sea"></a>
# ## 6. Seattle
# Seattle is the only city on the list that I've been to :) Kaggle provides the complete police districts shapefiles along with the census data, but no use of force or officer invovled shooting data. These were easy to find thanks to [Seattle Open Data portal](https://data.seattle.gov/). 

# In[ ]:


# Load Seattle datasets
sea_ois = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/6-Seattle/Seattle_Officer_Involved_Shooting.csv')
sea_uof = pd.read_csv('../input/cpe-external-data/cpe_external_data/cpe_external_data/6-Seattle/Seattle_Use_Of_Force.csv')


# Data is fairly well layed out in these files, without much need for feature extraction.

# In[ ]:


sns.countplot(y='Subject_Race', data=sea_uof, orient='h');


# ## Conclusion
# Two things that I learned from doing this kernel are that a) there is tons of open data out there that municipalities and police are willing to share; and b) Kaggle could have done a better job with the due dilligence on this competition and provide all this data from the beginning. 
# 
# Hope you found this useful. Let me know if you are using these files in your analysis.

# 
