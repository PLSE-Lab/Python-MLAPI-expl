#!/usr/bin/env python
# coding: utf-8

# # Effect of COVID on publication ouput in Physics
# 
# The goal of this project is to investigate the possibility of a disproportionate impact of COVID on the academic productivity of women physicists

# # Input files used
# 
# I used two datasets containing the information on arXiv submissions between January 1st, 2012 and July 1st, 2020 in the Condensed Matter and Astrophysics categories respectively. arXiv is an open-access repository of electronic preprints where researchers commonly submit the manuscripts concerning their findings while they awais publication on peer-reviewed journals. The fact that arXiv submissions usually appear on the website within 24 hours allows us to take a glimpse on the impact of the lockdown caused by COVID in real time, without the need to wait for the longer time required for official publication. arXiv also provides a convenient centralized location where to get information on a large number of works that were eventually published on a whole spectrum of scientific journals. 
# To scrape the data I used the code in the (commented) cell below

# In[ ]:


# !pip install arxivscraper

# import arxivscraper
# import pandas as pd

# scraper_cond = arxivscraper.Scraper(category='physics:cond-mat', date_from='2012-01-01', date_until='2020-07-01',timeout=10000)
# output_cond = scraper_cond.scrape()
# dfcond = pd.DataFrame(output_cond,columns=cols)

# scraper_astro = arxivscraper.Scraper(category='physics:astro-ph', date_from='2012-01-01', date_until='2020-07-01',timeout=10000)
# output_astro = scraper_astro.scrape()
# dfastro = pd.DataFrame(output_astro,columns=cols)

# cols = ('categories', 'created', 'authors')

# dfcond = pd.DataFrame(output_cond,columns=cols)
# dfcond.to_csv('Data/arxiv_cond_2012_2020.csv',index=False)
# dfastro = pd.DataFrame(output_astro,columns=cols)
# dfastro.to_csv('Data/arxiv_astro_2012_2020.csv',index=False)


# # Description of the project
# 
# This project takes inspiration from the [work](https://github.com/drfreder/pandemic-pub-bias) of Megan Frederickson concerning the effect of the pandemic on the academic publication output of scientists. Following an [article](https://www.nature.com/articles/d41586-020-01135-9) in Nature suggesting that the restrictions imposed by the lockdown might disproportionally be affecting the productivity of women in the field, Prof. Frederickson collected information on the preprint submissions to arXiv and bioRxiv in the first few months of 2019 and 2020 to investigate this aspect further. From her analysis it appears that the impact of the pandemic has indeed negatively affected women scientists to a larger extent. 
# 
# In order to shed more light on this importan issue, I here carry out a similar analysis on a subset of arXiv submission categories, namely publications in the condensed matter and astrophysics fields, but on a longer time scale (2012-2020). I limit my attention to first and last authors as I believe that they provide the most important information on the impact on publication output.
# I believe that an analysis that considers a longer period of time is important as it provides information on the overall publication trend and establishes the magnitude of the variance in the male/female publication rates. 
# 
# The steps of the analysis are explained throughout the 'Data_analysis' notebook. As mentioned above, the 'Data_scraping' notebook provides the code that I used to scrape arXiv for the authors' information. As the scraping process is rather time consuming, the 'Data_analysis' notebook directly imports the scraped data stored in the 'Data' folder. 
# 
# In order to extract the gender of the authors I use the [gender-guesser](https://pypi.org/project/gender-guesser/) package. As mentioned also by Prof. Frederickson, this type of approach has limitations and will not provide a perfectly accurate gender assignment for each entry in the data set. Nevertheless, they can provide a good aggregate overview of the male/female split. 

# # Comments
# 
# This is my first repository and Python data science project. I appreciate any comment and suggestion on how to improve it, tips on how to better present the findings in the repository, and particularly inputs on aspects of the analysis where I might have committed mistakes. 
# Thanks! 

# In[ ]:


get_ipython().system('pip install gender_guesser')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import gender_guesser.detector as gender

get_ipython().run_line_magic('matplotlib', 'inline')


# # Import previously scraped data

# In[ ]:


dfcond = pd.read_csv('../input/arxiv-scrape/arxiv_cond_2012_2020.csv',converters={'authors': eval})
dfastro = pd.read_csv('../input/arxiv-scrape/arxiv_astro_2012_2020.csv',converters={'authors': eval})


# We then consider the three data sets separately using the same analysis in order to compare the trends within the different topics

# In[ ]:


dfcond['first_author']=dfcond['authors'].apply(lambda x:x[0].split(' ')[0])
dfastro['first_author']=dfastro['authors'].apply(lambda x:x[0].split(' ')[0])


# We will start by taking a look at the data from the condensed matter category and then we will repeat the analysis for the astrophysics one as a measure of consistency

# # Analysis of condensed matter data
# 
# ## Data preparation
# 
# First we prepare the dataset for the analysis by dropping some entries that cannot be used.
# 
# For many arXiv submissions the authors only provide initials for their first and middle name. These entries cannot be used to build a statistics concerning the gender of the authors and I therefore proceed with removing them. 
# 
# To simplify I only check that the first author has entered a full name assuming that if only initials are used it is likely that that would be the case for all authors. Remember that we are also only concerned with the first and last author in this analysis. 

# In[ ]:


# First we split the author list and returning the first name of the first and last authors in a new columns
# We then drop the authors column for clarity
dfcond['first_author']=dfcond['authors'].apply(lambda x:x[0].split(' ')[0]) 
dfcond['last_author']=dfcond['authors'].apply(lambda x:x[-1].split(' ')[0])
dfcond.drop('authors',inplace=True,axis=1)

#Drop all the rows for which the author's name is shorter than 3 characters 
#This is done to include punctuation in the initials since the split of the author list is done using a single space
#This passage can be improves as it removes from the dataset also authors with a two-letter first name
dfcond.drop(dfcond[dfcond['first_author'].map(len)<3].index, inplace=True)

#We transform the created columns into a datetime column 
#We also drop all the entries that were created before 2012 but entered the data set because modified after 2012
dfcond['created'] = pd.to_datetime(dfcond['created'],format='%Y-%m-%d')
dfcond.drop(dfcond[dfcond['created'].dt.year<2012].index,inplace=True)

#Finally we reset the dataframe index
dfcond.reset_index(drop=True, inplace=True)


# We then proceed with detecting the gender of the first and last author of the manuscript using the gender.Detector method of the gender_guesser library. 

# In[ ]:


# We capitalize the names as this is required for the gender guesser method to work correctly
dfcond['first_author'] = dfcond['first_author'].apply(lambda x:x.capitalize())
dfcond['last_author'] = dfcond['last_author'].apply(lambda x:x.capitalize())

# We instantiate a gender detector object and run it on the first and last author columns separately
detect = gender.Detector()
dfcond['gender_first']=dfcond['first_author'].map(detect.get_gender)
dfcond['gender_last']=dfcond['last_author'].map(detect.get_gender)

# We then split the dataframe in two to better handle separately the two data sets
# This is useful to more easily handle cases where the gender of the first author is clearly identifyed 
# but not that of the last author, or viceversa. This approach can likely be improved
# We also rename the columns for simplicity 

dfcondfirst = dfcond.loc[:,['created','first_author','gender_first']]
dfcondfirst.rename(columns={'created':'date','first_author':'author','gender_first':'gender'},inplace=True)
dfcondlast = dfcond.loc[:,['created','last_author','gender_last']]
dfcondlast.rename(columns={'created':'date','last_author':'author','gender_last':'gender'},inplace=True)

# Finally we map the results of the gender guesser so that 'mostly male' and 'mostly female' 
# are transformed to 'male' and 'female' respectively. The gender guesser also return 'andy' when the name
# has equal likelihood of being male or female and 'uknown' when no record of the name is found
# We drop all the rows associated to names with these last two tags

dfcondfirst['gender'] = dfcondfirst['gender'].map({'male':'male','mostly_male':'male','mostly_female':'female',
                                                       'female':'female'})
dfcondfirst.dropna(inplace=True)

dfcondlast['gender'] = dfcondlast['gender'].map({'male':'male','mostly_male':'male','mostly_female':'female',
                                                       'female':'female'})
dfcondlast.dropna(inplace=True)


# We check what we are left with

# In[ ]:


dfcondlast.info()


# So it looks like we have a decently sized data set that we can work with.

# ### Some simple data visualization
# 
# Simple  histogram plot of the total number of submissions by year and by gender of the first author. We limit this to the year for which we have a full dataset

# In[ ]:


plt.figure(figsize = (10,6))

sns.countplot(x=dfcondfirst[dfcondfirst['date'].dt.year<2020]['date'].dt.year,
              data = dfcondfirst[dfcondfirst['date'].dt.year<2020],hue='gender');

plt.xlabel('Year', fontsize=15);
plt.ylabel('Publication Count', fontsize=15);
plt.xticks(size=13);
plt.yticks(size=13);
plt.legend(fontsize=15);


# Simple histogram plot of the total number of submissions by year and by gender of the last author

# In[ ]:


plt.figure(figsize = (10,6))

sns.countplot(x=dfcondlast[dfcondlast['date'].dt.year<2020]['date'].dt.year,
              data = dfcondlast[dfcondlast['date'].dt.year<2020],hue='gender');

plt.xlabel('Year', fontsize=15);
plt.ylabel('Publication Count', fontsize=15);
plt.xticks(size=13);
plt.yticks(size=13);
plt.legend(fontsize=15);


# Histogram plots of the total number of submissions separated by month by year and by gender of the first author.
# 
# This allows us to see more in details the effect on the first months of 2020.

# In[ ]:


fig, axes = plt.subplots(2,3, figsize=(20,12))
d = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June'}

for key,month in d.items():

        sns.countplot(dfcondfirst[dfcondfirst['date'].dt.month==key]['date'].dt.year,
              data = dfcondfirst[dfcondfirst['date'].dt.month==key], hue='gender',ax=axes[int(np.floor((key-1)/3)),
                                                                                          int((key-1)-np.floor((key-1)/3)*3)])
        axes[int(np.floor((key-1)/3)),
             int((key-1)-np.floor((key-1)/3)*3)].set_title('arXiv publication for the month of %s' % month, size = 15);
        axes[int(np.floor((key-1)/3)),
             int((key-1)-np.floor((key-1)/3)*3)].set_xlabel('Year', size = 13);
        axes[int(np.floor((key-1)/3)),
             int((key-1)-np.floor((key-1)/3)*3)].set_ylabel('Publications',size=13);


# ### Gender ratio 
# 
# We now want to extract the ratio of female authors to analyze how this factor has changed in time in the past 8 years

# In[ ]:


# We create a small function to create a dataframe for the evolution of the gender publication ratio  
# we sample the data by month

def GenderRatio(df):
    
    ratio = []
    years = df['date'].dt.year.unique()
    months = df['date'].dt.month.unique()

    for year in years:

        try:
            for month in months:

                [a,b] = df[(df['date'].dt.year==year)
                             & (df['date'].dt.month==month)]['gender'].value_counts()
                ratio.append(b/(a+b))
        except:
            break
        
    time = pd.date_range(start=str(months[0])+'/'+str(years[0]),end=str(month)+'/'+str(year),freq='m');

    return pd.DataFrame(ratio,time,columns=['Percentage Female Author'])


# In[ ]:


# Use function to create the dataframe

GndRtCondFirst = GenderRatio(dfcondfirst)


# In[ ]:


# Visualization of the female/male first author ratio between 2012 and 2020 

fig = plt.figure(figsize = (10,8))

sns.set_style('darkgrid')
sns.lineplot(data=GndRtCondFirst,lw=2,legend=False)

plt.title('Gender ratio in first authorship',size=25)
plt.xlabel('Time',size=18);
plt.ylabel('Percentage of female first authors',size=18);
plt.xticks(fontsize=15);
plt.yticks(fontsize=15);


# In[ ]:


# We look at some descriptive statistics to better understand the variability of the data

STD = GndRtCondFirst.groupby(GndRtCondFirst.index.year).std()
AVE = GndRtCondFirst.groupby(GndRtCondFirst.index.year).mean()
AVE.rename(columns={'Percentage Female Author':'Mean'},inplace=True)
STD.rename(columns={'Percentage Female Author':'Standard Deviation'},inplace=True)

STAT = pd.concat((AVE,STD),axis=1)
STAT


# In[ ]:


# We use the function previously created to repeat the analysis for the last author

GndRtCondLast = GenderRatio(dfcondlast)


# In[ ]:


fig = plt.figure(figsize = (10,8))

sns.set_style('darkgrid')
chart = sns.lineplot(data=GndRtCondLast,lw=2,legend=False)

plt.title('Gender ratio in last authorship',size=25)
plt.xlabel('Time',size=18);
plt.ylabel('Percentage of female last authors',size=18);
plt.xticks(fontsize=15);
plt.yticks(fontsize=15);


# In[ ]:


# We look at some descriptive statistics to better understand the variability of the data

STD = GndRtCondLast.groupby(GndRtCondLast.index.year).std()
AVE = GndRtCondLast.groupby(GndRtCondLast.index.year).mean()
AVE.rename(columns={'Percentage Female Author':'Mean'},inplace=True)
STD.rename(columns={'Percentage Female Author':'Standard Deviation'},inplace=True)

STAT = pd.concat((AVE,STD),axis=1)
STAT


# We now repeat the analysis for the publication data in the astrophysics category
# 
# # Analysis of astro data
# 
# ### Data preparation

# In[ ]:


dfastro['first_author']=dfastro['authors'].apply(lambda x:x[0].split(' ')[0]) 
dfastro['last_author']=dfastro['authors'].apply(lambda x:x[-1].split(' ')[0])
dfastro.drop('authors',inplace=True,axis=1)

dfastro.drop(dfastro[dfastro['first_author'].map(len)<3].index, inplace=True)

dfastro['created'] = pd.to_datetime(dfastro['created'],format='%Y-%m-%d')
dfastro.drop(dfastro[dfastro['created'].dt.year<2012].index,inplace=True)

dfastro.reset_index(drop=True, inplace=True)


# In[ ]:


dfastro['first_author'] = dfastro['first_author'].apply(lambda x:x.capitalize())
dfastro['last_author'] = dfastro['last_author'].apply(lambda x:x.capitalize())

detect = gender.Detector()
dfastro['gender_first']=dfastro['first_author'].map(detect.get_gender)
dfastro['gender_last']=dfastro['last_author'].map(detect.get_gender)

dfastrofirst = dfastro.loc[:,['created','first_author','gender_first']]
dfastrofirst.rename(columns={'created':'date','first_author':'author','gender_first':'gender'},inplace=True)
dfastrolast = dfastro.loc[:,['created','last_author','gender_last']]
dfastrolast.rename(columns={'created':'date','last_author':'author','gender_last':'gender'},inplace=True)

dfastrofirst['gender'] = dfastrofirst['gender'].map({'male':'male','mostly_male':'male','mostly_female':'female',
                                                       'female':'female'})
dfastrofirst.dropna(inplace=True)

dfastrolast['gender'] = dfastrolast['gender'].map({'male':'male','mostly_male':'male','mostly_female':'female',
                                                       'female':'female'})
dfastrolast.dropna(inplace=True)


# In[ ]:


dfastrolast.info()


# ### Data visualization

# In[ ]:


fig = plt.figure(figsize = (10,6))

sns.countplot(x=dfastrofirst[dfastrofirst['date'].dt.year<2020]['date'].dt.year,
              data = dfastrofirst[dfastrofirst['date'].dt.year<2020],hue='gender');

plt.xlabel('Year', fontsize=15);
plt.ylabel('Publication Count', fontsize=15);
plt.xticks(size=13);
plt.yticks(size=13);
plt.legend(fontsize=15);


# In[ ]:


fig = plt.figure(figsize = (10,6))

sns.countplot(x=dfastrolast[dfastrolast['date'].dt.year<2020]['date'].dt.year,
              data = dfastrolast[dfastrolast['date'].dt.year<2020],hue='gender');

plt.xlabel('Year', fontsize=15);
plt.ylabel('Publication Count', fontsize=15);
plt.xticks(size=13);
plt.yticks(size=13);
plt.legend(fontsize=15);


# ### Breakdown by month

# In[ ]:


fig, axes = plt.subplots(2,3, figsize=(20,12),sharey=True)
d = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June'}

for key,month in d.items():

        sns.countplot(dfastrofirst[dfastrofirst['date'].dt.month==key]['date'].dt.year,
              data = dfastrofirst[dfastrofirst['date'].dt.month==key], 
                      hue='gender',hue_order = ['male','female'], ax=axes[int(np.floor((key-1)/3)),
                                                                            int((key-1)-np.floor((key-1)/3)*3)])
        axes[int(np.floor((key-1)/3)),
             int((key-1)-np.floor((key-1)/3)*3)].set_title('arXiv publication for the month of %s' % month, size = 15);
        axes[int(np.floor((key-1)/3)),
             int((key-1)-np.floor((key-1)/3)*3)].set_xlabel('Year', size = 13);
        axes[int(np.floor((key-1)/3)),
             int((key-1)-np.floor((key-1)/3)*3)].set_ylabel('Publications',size=13);


# ### Gender ratio

# In[ ]:


# Use function to create the dataframe

GndRtAstroFirst = GenderRatio(dfastrofirst)


# In[ ]:


# Visualization of the female/male first author ratio between 2012 and 2020 

fig = plt.figure(figsize = (10,8))

sns.set_style('darkgrid')
sns.lineplot(data=GndRtAstroFirst,lw=2,legend=False)

plt.title('Gender ratio in first authorship',size=25)
plt.xlabel('Time',size=18);
plt.ylabel('Percentage of female first authors',size=18);
plt.xticks(fontsize=15);
plt.yticks(fontsize=15);


# In[ ]:


# We look at some descriptive statistics to better understand the variability of the data

STD = GndRtAstroFirst.groupby(GndRtAstroFirst.index.year).std()
AVE = GndRtAstroFirst.groupby(GndRtAstroFirst.index.year).mean()
AVE.rename(columns={'Percentage Female Author':'Mean'},inplace=True)
STD.rename(columns={'Percentage Female Author':'Standard Deviation'},inplace=True)

STAT = pd.concat((AVE,STD),axis=1)
STAT


# In[ ]:


# Last author

GndRtAstroLast = GenderRatio(dfastrolast)


# In[ ]:


# Visualization of the female/male Last author ratio between 2012 and 2020 

fig = plt.figure(figsize = (10,8))

sns.set_style('darkgrid')
sns.lineplot(data=GndRtAstroLast,lw=2,legend=False)

plt.title('Gender ratio in last authorship',size=25)
plt.xlabel('Time',size=18);
plt.ylabel('Percentage of female last authors',size=18);
plt.xticks(fontsize=15);
plt.yticks(fontsize=15);


# In[ ]:


# We look at some descriptive statistics to better understand the variability of the data

STD = GndRtAstroLast.groupby(GndRtAstroLast.index.year).std()
AVE = GndRtAstroLast.groupby(GndRtAstroLast.index.year).mean()
AVE.rename(columns={'Percentage Female Author':'Mean'},inplace=True)
STD.rename(columns={'Percentage Female Author':'Standard Deviation'},inplace=True)

STAT = pd.concat((AVE,STD),axis=1)
STAT


# # Conclusions
# 
# From this analysis we can conclude that, at least in the subfields considered here, there is not a statistically relevant change in the ratio of female-to-male authorship (both in the case of first and last authors) in the first few months of 2020, when this data is compared to the trend of the past 8 years. 
# 
# We should nonetheless consider that the impact on publications could still be delayed by many months and this analysis will have to be regularly repeated to consider this possibility.
