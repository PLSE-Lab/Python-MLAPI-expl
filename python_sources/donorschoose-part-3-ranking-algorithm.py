#!/usr/bin/env python
# coding: utf-8

# ## DonorsChoose- Part 3. Ranking Algorithm Reasoning & Methods

# #### First, the approach used for the e-mail recommendation system is explained.  Next I will explain how the donors' behaviors support this approach.  Afterwards, the recommender system will be implemented and validated. 

# <a href='#1'>1. Introduction</a><br>
# <a href='#2'>2. Why does the ranking system work?</a><br>
# <a href='#3'>3. Building the Recommender</a><br>
# <a href='#4'>4. Recommender Performance</a><br>
# <a href='#5'>5. Future Enhancements</a><br>

# ### <a id='1'>1. Introduction</a>

# #### Background
# DonorsChoose.org is an organization that helps teachers fundraise for American classrooms.  The goal of the project is to enable DonorsChoose.org to build targeted email campaigns recommending specific classroom requests to prior donors.  The organization has supplied anonymized [data](https://www.kaggle.com/donorschoose/io) on donor giving from the past five years, starting from January 2013 to May 1, 2018.   
# 
# In Part 1 of the notebooks, features were generated to capture the characteristics regarding the donors, the schools, as well as the projects.  The features related to donors are normalized to reflect an donor's interest relative to the rest of the donors. The project features are one-hot encoded.  A major feature that proved to be very useful is geographic location of the donor and the schools.  A main component of the reommendation algorithm is Geo Search, where possible recommendations are filtered and sorted by distance or around certain geo-locations.
# 
# #### Recommendation Algorithm
# The recommendation algorithm is based on a combination of Filtering and Ranking.  Here are the implementation steps:
# <Br>
# <font color= navy> Prepare Project Features Matrix and Donor Features Matrix </font>
# 
# 1. Generate the Project Features Matrix: The project features encode project characteristics, such as the project category, school metro type, project grade level, project resource, the percentage of students receiving free lunch, and etc. Some features are one-hot encoded, some are binned. There are a total of 71 dimensions. (discussed in Part 1 of Notebooks)
# 
# 2. Generate the Donor Features Matrix: The Donor Features Matrix not only has features that measures donor's interest in every metrics in the Project Features Matrix, but it includes donation history based features.  Some examples of the donation based features include statistics on the number of carts, projects, schools, & teachers that the the donor had ever donated to, and the distance features between school location and donor location.  Additionally, The features in the Donor Feature Matrix are normalized either by comparing the donor to the the mean of all the donors, or are based on the percentile ranking against the other donors. There are a total of 96 dimensions.  (discussed in Part 1 of Notebooks)
# 
# 
# <font color= navy> Filtering </font>
# 1. Obtain projects that are eligible for recommendations, which would be projects that are not yet fully funded, and have not expired. We will refer to these projects as the "Initial Project Universe." 
# 2. Next, we will narrow down the "Initial Project Universe."  We will refer to the remaining projects as the "Project Universe." The criterias in determining the Project Universe are as follows:
#     1. Obtain the "School City, School State" of the projects that the donor has ever donated to in the past, and filter the "Initial Project Universe" by these locations. 
#     2. Obtain projects in the "Initial Project Universe" that are located in the donor's "Donor City, Donor State".
#     3. Obtain the "Teacher IDs" that the donor has ever donated to in the past, and filter the "Initial Project Universe" by these "Teacher IDs"<br>
#     
# 3. The Project Features Matrix that will be used to calculate similarity measures between donor and projects will only include the projects obtained above, and these are the projects that we will rank.
# 
# <font color= navy> Ranking </font> <br>
# The relevance of a project is based on the following attributes: <br>
# 1. Calculate the Similarity Score between the Project Features Matrix and the corresponding features in the Donor Features Matrix by taking the dot product between the two.
# 
# 2. The Final Score is based on a combination of:
#     1.  Similarity Score from step 1.  
#     2. School Bonus: +1 point if the project is from a school that the donor has previously donated to
#     3. Teacher Bonus: +1 point if the project is from a teacher that the donor has previously donated to 
#     4. Home City Bonus: +1 point if the project is located in the Donor City.   
#     5. Project Location Bonus: +1 point if the project is located in a city that the donor has previously donated to
#     6. Frequent Location Bonus: Reward the projects that are originated from the cities that the donor has made the most contribution to in the past (based on dollar amount). The rational behind this bonus is that the likelihood of the donor donating to these cities again is higher than the likelihood of the donor donating to the other cities. <br> 
#     <br>To calculate the Frequent Location Bonus, you would multiply the sum of A+B+C+D+E by a scaling factor. The scaling factor is designed in such as a way that the effect of the boost will be higher as the donor donates to more city.  The boost will be minimal if the donor has only donated to a few cities.
#     
#     *Example:* 
#     <br>A location ranked #1 will receive a boost of 0.5 if the donor have donated to 50 cities.  The boost will decrease to 0.1 if the donor have only donated to 10 cities.  
#     
#     *Formula: *
#     $$scaling\:factor =\frac{1}{100}*\frac{Number\:of\:Cities}{City\:Ranking\:(highest\:ranks\:No.1)}$$
#     
#     **Ranking Score Calculation:** <br>
#     Preliminary score = similarity score + school bonus + teacher bonus + home city bonus + project location bonus<br>
#     Frequent Location bonus = scaling factor x Preliminary score <br>
#     Final score = Preliminary score + Frequent location bonus <br>

# #### Validation Scheme
# Use data prior to 2018 to generate donor profile.  Verify recommendation against actual donations by the donor in January 2018 <br>
# * Number of donors who donated both prior to 2018 and in January 2018: 31,362
# * Number of projects that donors donated to in Jan 2018: 32,716
# * Number of projects that donors can choose from (projects that expires after Dec. 31, 2017): 47,816
# 

# #### Result

# 1000 random sample of donors are drawn from the pool of 31,362 donors: <br>
# With 47,816 projects to choose from, the recommender was able to recommend the correct project 32% of times. <br>
# * % Hit in top 1 Recommendations: 32.4
# * % Hit in top 5 Recommendations: 54.5
# * % Hit in top 10 Recommendations: 61.0
# * % Hit in top 25 Recommendations: 66.3
# 

# #### The Analysis below shows why location, school, teacher, and Owner Interest are used as the criterias.

# In[6]:


import numpy as np
import pandas as pd 
import os
import datetime as dt
from sklearn import cluster
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
py.offline.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


# define columns to import
projectCols = ['Project ID', 'School ID', 'Teacher ID',
               'Teacher Project Posted Sequence', 'Project Type',
               'Project Subject Category Tree', 'Project Subject Subcategory Tree',
               'Project Grade Level Category', 'Project Resource Category',
               'Project Cost', 'Project Posted Date', 'Project Expiration Date',
               'Project Current Status', 'Project Fully Funded Date']

resourcesCols = ['Project ID','Resource Quantity','Resource Unit Price', 'Resource Vendor Name']

# import files
donations = pd.read_csv('../input/io/Donations.csv', dtype = {'Donation Amount': np.float32, 'Donor Cart Sequence': np.int32})
donors = pd.read_csv('../input/io/Donors.csv', dtype = {'Donor Zip':'str'})
projects = pd.read_csv('../input/io/Projects.csv', usecols = projectCols, dtype = {'Teacher Project Posted Sequence': np.float32, 'Project Cost': np.float32})
resources = pd.read_csv('../input/io/Resources.csv', usecols = resourcesCols, dtype = {'Resource Quantity': np.float32,'Resource Unit Price': np.float32})
schools = pd.read_csv('../input/io/Schools.csv', dtype = {'School Zip': 'str'})
teachers = pd.read_csv('../input/io/Teachers.csv')

# These are files from Part I:
donorFeatureMatrixNoAdj = pd.read_csv('../input/part-1-preprocessing-feature-engineering/donorFeatureMatrixNoAdj.csv')
donorFeatureMatrix =  pd.read_csv('../input/part-1-preprocessing-feature-engineering/donorFeatureMatrix.csv')
donorsMapping = pd.read_csv('../input/part-1-preprocessing-feature-engineering/donorsMapping.csv') 
schoolsMapping = pd.read_csv('../input/part-1-preprocessing-feature-engineering/schoolsMapping.csv')
projFeatures = pd.read_csv('../input/part-1-preprocessing-feature-engineering/projFeatures.csv')
distFeatures = pd.read_csv('../input/part-1-preprocessing-feature-engineering/distFeatures.csv')

# donations
donations['Donation Received Date'] = pd.to_datetime(donations['Donation Received Date'])
donations['Donation Included Optional Donation'].replace(('Yes', 'No'), (1, 0), inplace=True)
donations['Donation Included Optional Donation'] = donations['Donation Included Optional Donation'].astype('bool')
donations['Donation_Received_Year'] = donations['Donation Received Date'].dt.year
donations['Donation_Received_Month'] = donations['Donation Received Date'].dt.month
donations['Donation_Received_Day'] = donations['Donation Received Date'].dt.day

# donors
donors['Donor Is Teacher'].replace(('Yes', 'No'), (1, 0), inplace=True)
donors['Donor Is Teacher'] = donors['Donor Is Teacher'].astype('bool')

# projects
cols = ['Project Posted Date', 'Project Fully Funded Date']
projects.loc[:, cols] = projects.loc[:, cols].apply(pd.to_datetime)
projects['Days_to_Fullyfunded'] = projects['Project Fully Funded Date'] - projects['Project Posted Date']

# teachers
teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])

##
# Name the dataframes
##
def name_dataframes(dfList, dfNames):
    '''
    give names to a list of dataframes. 
    Argument:
        dfList = list of dataframes,
        dfNames = list of names for the dataframes
    Return:
        None
    '''
    for df, name in zip(dfList, dfNames):
        df.name = name
    
    return

dfList = [donations, donors, projects, resources, schools, teachers]
dfNames = ['donations', 'donors', 'projects', 'resources', 'schools', 'teachers']
name_dataframes(dfList, dfNames)

##
#  Remove rows in the datasets that cannot be mapped
##
projects = projects.loc[projects['School ID'].isin(schools['School ID'])]
projects = projects.loc[projects['Project ID'].isin(resources['Project ID'])]
donations = donations.loc[donations['Project ID'].isin(projects['Project ID'])]
donations = donations.loc[donations['Donor ID'].isin(donors['Donor ID'])]
donors = donors.loc[donors['Donor ID'].isin(donations['Donor ID'])]

##
#  We will add features we created in Part I to the donors and schools dataset.
##
donors = donors.merge(donorsMapping, left_on = 'Donor ID', right_on = 'Donor ID', how = 'left')
donors = donors.drop(['Unnamed: 0'], axis=1)
schools = schools.merge(schoolsMapping.filter(items = ['School ID', 'School_Lon', 'School_Lat']), left_on = 'School ID', right_on = 'School ID', how = 'left')

projFeatures = projFeatures.drop(['Unnamed: 0'], axis=1)


# ### <a id='2'>2. Why does the ranking system work?</a>

# ### Donors have favorite schools that they prefer to donate to

# #### 60% of donors who donated more than once have donated to the same school at least once.  
# Donors have tendancy to donate to selected schools.  60% of donors who donated to more than 1 project, have project count higher than # of schools the donor donates to.  This means that the donor donated to the same school at least once. <br>
# Some possible explanations could be that donor donates to a local school, or a school that their children attends.

# In[3]:


def donor_summary(donations, projects):
    ''' 
    Generate features to refelect donor's previous donation history:
    'num_proj', 'num_donation', 'num_cart', 'donation_median',
    'donation_mean', 'donation_sum', 'donation_std', 'School ID_count',
    'Teacher ID_count', 'schoolConcentration', 'TeacherConcentration',
    '''
    donations = donations.set_index('Donor ID', drop = False)
    donorSummary = pd.DataFrame(donations['Donor ID']).drop_duplicates(keep = 'first') 
    
    #### Obtain number of projects, # of donations, and the max cart number for each  donor
    countProj = donations.groupby(donations.index).agg({'Project ID':'nunique','Donor Cart Sequence':'max', 'Donation ID':'count'})
    countProj.columns = ['num_proj', 'num_donation','num_cart']
    donorSummary  = donorSummary.merge(countProj, left_index = True,  right_index=True, how = 'left')
    
    #### Count # of schools and # of teachers that a donor donates to
    school_teacher = donations[['Project ID', 'Donation Amount', 'Donor ID']].merge(projects[['Project ID', 'School ID', 'Teacher ID']], left_on = 'Project ID', right_on = 'Project ID', how = 'left')
    concentration = school_teacher.groupby('Donor ID').agg({'School ID':'nunique', 'Teacher ID':'nunique'})
    concentration.columns = concentration.columns + '_count'
    donorSummary  = donorSummary.merge(concentration, left_index = True,  right_index=True, how = 'left')
    
    #### Design feature to capture the concentration of donation to schools.
    #### feature that captures doners that donates to multiple schools, and not just have one favorite school
    schoolSum = school_teacher.groupby(['Donor ID', 'School ID'])['Donation Amount'].sum().reset_index(drop = False)
    schoolSum = schoolSum.groupby(['Donor ID'])['Donation Amount'].agg(['sum', 'max'])
    schoolSum['SchoolConcentration'] = schoolSum['max']/schoolSum['sum']
    donorSummary['schoolConcentration'] = schoolSum['SchoolConcentration']
    
    #### Design feature to capture the concentration of donation to a teacher.  
    TeacherSum = school_teacher.groupby(['Donor ID', 'Teacher ID'])['Donation Amount'].sum().reset_index(drop = False)
    TeacherSum = TeacherSum.groupby(['Donor ID'])['Donation Amount'].agg(['sum', 'max'])
    TeacherSum['TeacherConcentration'] = TeacherSum['max']/TeacherSum['sum']
    donorSummary['TeacherConcentration'] = TeacherSum['TeacherConcentration']
    
    return donorSummary


# In[64]:


donorSummary = donor_summary(donations, projects)


# In[11]:


chartData = donorSummary.loc[donorSummary['num_proj']>1]
chartData['School Bias'] = chartData['School ID_count'] < chartData['num_proj']
Breakdown = chartData['School Bias'].value_counts()/len(chartData)
Breakdown.plot(kind='bar', stacked=True, legend = True)
plt.title("Most donors have donated to the same school")
Breakdown


# Suppose a donor does not have preference for school, if the donor made donation to 5 projects, the 5 projects would likely to be from diffierent schools and the amount of donation that each school received would be roughly 20%.  However, in reality, for each donor, the school that received the most money from the donor is much higher than 20%. 
# 
# Looking at the chart below, the pink line represents equally distribution of donations amongst all the donations.  The blue dots are donors whose number of projects equales to number of projects.  The orange dots are donors with project counts greater than school counts.  The blue and orange lines are regression lines for each respective groups.  The futher down the regression lines are away from the pink line, the higher the bias of donors concentrating the donation to one specific school.  
# 
# Suppose a donor donated to 5 projects, the donor would lie on y = 0.2.  If donation is equally distributed, the school receiving the highest donation from the donor would be 20%.  However, looking at the chart below, the school receiving the most donations got a lot more than 20%.  The regression estimate of donation going to the favorite school for donors whose school count < project count is about 75%.  The estimated donation for the donors whose school count = project count is about 25%.

# In[12]:


chartData['1/NumProj'] = 1/chartData['num_proj']
chartData = chartData.sample(1000)
sns.lmplot(data = chartData, x= 'schoolConcentration', y= '1/NumProj', hue = 'School Bias', fit_reg=True)
plt.xlabel("% donation donated to the favorite school") 
plt.ylabel("1 / Number of Projects")
plt.title("Donors have Favorite Schools")
plt.ylim(0, 1)
plt.xlim(0, 1)
x1, y1 = [0, 1], [0, 1]
plt.plot(x1, y1, dashes=[6, 2], color = "pink")


# ### Some donors donate to the same teacher despite the teacher moved to a different school

# #### There are about 4% of donors whose preference towards teachers is above their preference for schools.
# One explanation for these set of donors is that the donors could be personally related to the teacher.  The donors could be the teacher's friends and family.  

# For donors that donated to more than 1 projects, if they donate to 2 teachers, they would usually donate to 2 different schools.  There are about 4% of the donors with school count smaller than teacher count.  These could signify prefreference to a teacher.  Looking at the chart below, there are a small number of donors whose $ donated to their favorite school is lower than the dollar donated to their favorite teacher.  Those donors are the ones above the slope of 1.

# In[15]:


chartData = donorSummary.loc[donorSummary['num_proj']>1]
chartData['Teacher Bias'] = (chartData['School ID_count']) > chartData['Teacher ID_count']
Breakdown = chartData['Teacher Bias'].value_counts()/len(chartData)
Breakdown.plot(kind='bar', stacked=True, legend = True)
plt.title('Most donors are not biased towards a specific teacher')
print(Breakdown)

chartData = chartData.sample(5000)
sns.lmplot(data = chartData, x= 'schoolConcentration', y= 'TeacherConcentration', hue = 'Teacher Bias', fit_reg=False)
plt.xlabel("% donation donated to the favorite school") 
plt.ylabel("% donation donated to the favorite teacher")
plt.title("Few donors are biased towards specific teachers")
plt.ylim(0, 1)
plt.xlim(0, 1)
x1, y1 = [0, 1], [0, 1]
plt.plot(x1, y1)


# ### Donors donate locally

# Most of the donors donate to local schools, within the same city or very close to each other.
# 
# * Approximately 38% of donation amount have the matching city/state between donors and the schools.
# * 68% of donotions are from under 20 miles between donor and school
# * The median donation distance is 16 miles.
# 
# Please note that the above analysis is limited to donors who have valid Donor City and Donor State data.
# 

# In[16]:


# Prepare Chart Data
chartData = donations[['Donor ID', 'Project ID', 'Donation Amount']].merge(distFeatures[['Donor ID', 'Project ID', 'dist']], left_on = ['Donor ID', 'Project ID'], right_on = ['Donor ID', 'Project ID'], how = 'left')
chartData = chartData.merge(donors[['Donor ID', 'no_mismatch']], left_on = 'Donor ID', right_on = 'Donor ID', how = 'left')
chartData = chartData.loc[chartData['no_mismatch'] == 1]
chartData['dist_cut'] = pd.cut(chartData['dist'], bins = [-1, 0, 5, 10, 20, 50, 100, 15000], labels = ['0', '1-5', '6-10', '11-20', '21-50', '51-100', '>100'])
chartData['Total Amount'] = chartData.groupby(['Donor ID', 'Project ID'])['Donation Amount'].transform('sum')
chartData = chartData.drop_duplicates(subset=['Project ID', 'Donor ID'], keep='first')
chart = pd.DataFrame(chartData.groupby('dist_cut')['Total Amount'].agg('sum')/chartData['Donation Amount'].sum()*100).reset_index(drop = False)

# Plot chart
plot = sns.barplot(x = 'dist_cut', y = 'Total Amount', data = chart)
plot.set_title("% of donations relative to distances between donor and schools")
plot.set_ylabel('% donations')
plot.set_xlabel('distances in miles')

print("The median distance in miles between project/donor pair is:", chartData['dist'].median())
print(chart)


# ### Donors have preferences for the project types that they donate to.

# K-means clustering of the donors show the following groups:
# * Literacy only
# * Math, Science, Applied Science
# * Music & Art
# * Literacy, Math, & Science
# * Applied Learning, Sports, Special Needs
# 

# In[18]:


# Run K-means model on donor FeatureMatrix 
k_means = cluster.KMeans(n_clusters=5)

# Group using project categories
colsInclude = list(donorFeatureMatrix.loc[:,'ProjCat_Applied Learning': 'ProjCat_Warmth, Care & Hunger'].columns)
result = k_means.fit(donorFeatureMatrix[colsInclude])

# Get the k-means grouping label
clusterLabel = result.labels_


# #### % of donors in each cluster profile

# In[19]:


pd.DataFrame(clusterLabel)[0].value_counts(normalize=True)


# In[20]:


def plot_cluster_traits(donorFeatureMatrix, col_category, clusterLabel):
    '''
    col_category are the filters for the column names in the donorFeatureMatrix
    values could be: 
    'Project Type', 'School Metro Type', 'Project Grade Level Category',
    'Project Resource Category', 'lunchAid', 'ProjCat', 'Dist', 'Percentile'
    
    clusterLabel is labels from the output of k-means
    '''
    
    # get columns to chart
    chart = donorFeatureMatrix.filter(regex='^'+col_category, axis=1).copy()
    chart['label'] = clusterLabel
    
    # for each column, get mean of each cluster
    chart = chart.groupby(['label']).mean().reset_index()
    chart_melt = pd.melt(chart, id_vars = ['label'], value_vars = chart.columns[1:], var_name='category', value_name = 'mean')
    chart_melt['color'] = np.where(chart_melt['mean']<0, 'orange', 'pink')
    chart_melt = chart_melt.sort_values(by = ['label', 'category']).reset_index(drop = True)
    
    # delete the col_category from column names for the chart
    chart_melt['category'] = chart_melt['category'].str.replace(col_category+'_','')
    
    # plot chart using Seaborn
    if chart_melt['category'].nunique()>8:
        g = sns.FacetGrid(chart_melt, row = 'label', size=1.5, aspect=8)  # size: height, # aspect * size gives the width
        g.map(sns.barplot, 'category', 'mean', palette="Set1")
        g.set_xticklabels(rotation=90)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Cluster Preferences- ' + col_category)
    else:
        g = sns.FacetGrid(chart_melt, row = 'label', size=1.5, aspect=4)
        g.map(sns.barplot, 'category', 'mean', palette="Set2")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Cluster Preferences- ' + col_category)
    return g


# ##### Preferences for Project Categories (adjusted by mean)

# In[21]:


plot_cluster_traits(donorFeatureMatrix, 'ProjCat', clusterLabel)


# ##### What are the features used to generate the donor profiles

# In[ ]:


donorFeatureMatrix.columns


# ### <a id='3'>3. Building the Recommender</a>

# ### A. Set-Up Validation Framework

# Use data prior to 2018 to generate donor profile.  Check if recommendations match the donor donations in January 2018.  

# #### Filter for donors who satisfies the following conditions:
# 1. Donated both in January of 2018 and prior to 2018

# In[13]:


donorID_2018 = donations[(donations['Donation_Received_Year'] == 2018) & (donations['Donation_Received_Month'] == 1)]['Donor ID'].unique()
donorID_prior_2018 = donations[donations['Donation_Received_Year'] < 2018]['Donor ID'].unique()
test_ID = list(set(donorID_2018).intersection(donorID_prior_2018))
print("# of IDs matching the criterias:", len(test_ID))


# #### Filter donations into groups: prior to 2018, and Jan 2018

# In[14]:


donations_prior = donations[(donations['Donation_Received_Year'] < 2018) & (donations['Donor ID'].isin(test_ID))]
donations_2018 = donations[(donations['Donation_Received_Year'] == 2018) & (donations['Donation_Received_Month'] == 1) & (donations['Donor ID'].isin(test_ID))]
print('donations_prior:', donations_prior.shape)
print('donations_Jan 2018:', donations_2018.shape)


# #### Filter donor feature matrix

# In[15]:


donorFeatureMatrixNoAdj = donorFeatureMatrixNoAdj.loc[donorFeatureMatrixNoAdj['Donor ID'].isin(test_ID)]
donorFeatureMatrix = donorFeatureMatrix.loc[donorFeatureMatrix['Donor ID'].isin(test_ID)]
print('Donor Feature Matrix Unscaled Shape:', donorFeatureMatrixNoAdj.shape)
print('Donor Feature Matrix Scaled Shape:', donorFeatureMatrix.shape)


# #### Filter for projects that qualifies for recommendation 
# 1. Projects that have not been fully funded as of 1/1/2018
# 2. Project Expiration Date after 1/1/2018
# 3. Projects posted before 1/31/2018

# In[16]:


projectsID = set(projects[(projects['Project Fully Funded Date'] >= '1/1/2018') & (projects['Project Expiration Date'] >= '1/1/2018') & (projects['Project Posted Date'] <= '1/31/2018')]['Project ID'])
print("# of Project IDs matching the criterias:", len(projectsID))


# #### Ensure all the projects that the donors donated to are in the projectFeatures

# In[17]:


projects_2018 = list(donations[(donations['Donation_Received_Year'] == 2018) & (donations['Donation_Received_Month'] == 1)]['Project ID'].unique())
print("# of projects that people donated to in Jan 2018:", len(projects_2018))

projectsID = set(projects_2018).union(set(projectsID))
print("# of projects that people could donate to in Jan 2018:", len(projectsID))


# #### Create ProjectFeatures matrix

# In[18]:


projectFeatures = projFeatures.loc[projFeatures['Project ID'].isin(projectsID)]
projectFeatures = projectFeatures.merge(projects.filter(items = ['Project ID', 'School ID', 'Teacher ID']), left_on = 'Project ID', right_on = 'Project ID', how = 'left')
projectFeatures = projectFeatures.merge(schools.filter(items = ['School ID', 'School_Lon', 'School_Lat', 'School City', 'School State']), left_on = 'School ID', right_on = 'School ID', how = 'left')
projectFeatures = projectFeatures.drop_duplicates(subset=['Project ID'], keep='first')
projectFeatures['CityState'] = projectFeatures['School City'].map(str)+', '+projectFeatures['School State']
print("Number of projects in projectFeatures:", len(projectFeatures))


# ### B. Define Recommendation Function

# In[19]:


def merge_data(ID, donations, donors = donors, projects = projects, schools = schools):
    ''' 
    Filter data based on a list of Donor ID.  Merge all data together into one dataframe.  
    Arguments: list of 'Donor ID'
    Returns: dataframe 
    '''
    
    # To ensure ID parameter works for both a list and a string
    if isinstance(ID, list):
        temp = donations[donations['Donor ID'].isin(ID)].reset_index(drop = True)
    else:
        temp = donations[donations['Donor ID'] == ID].reset_index(drop = True)
    
    temp = temp.merge(donors, on = 'Donor ID', how='left')
    temp = temp.merge(projects, on = 'Project ID', how = 'left')
    temp = temp.merge(schools, on = 'School ID', how = 'left')
    
    return temp


# In[20]:


def recommender(ID,
                donorDonations = None,
                donations_prior = donations_prior, 
                donors = donors, 
                projects = projects, 
                schools = schools, 
                donorFeatureMatrix = donorFeatureMatrix,
                projectFeatures = projectFeatures):
    '''
    Filter potential projects based on donor's previous donations, and donor's home location
    Rank projects based on dot product score of project features matrix, and donor feature matrix 
    
    Arguments: "Donor ID", 
                donations - the donation history (do not include future donations)
                donorFeatureMatrix - compiled using donor history to ensure no look ahead bias
    Returns: 
    Donor ID
    Ranking- Ranking of the project that the donor actually donated to 
    Project ID- Project ID of the project that the donor donated to
    Filter Size- # of projects that met location criterias
    Universe- dataframe of projectFeatures with scoring
    '''
    
    #print("Donor ID:", ID)
    
    ##
    # Merge donations data filtered by ID with donors, projects, schools
    ##
    
    if donorDonations is None:
        donorDonations = merge_data(ID, donations_prior)
    else:
        donorDonations = donorDonations.loc[donorDonations['Donor ID'] == ID]

    ##
    #  Get Previous School ID, Teacher ID, School Location (City/State), Donor's Location
    ##
    
    # previous locations of donations
    byLocation = donorDonations.groupby(['School City', 'School State'])['Donation Amount'].sum().sort_values(ascending = False)
    locMap = donorDonations.drop_duplicates(subset = ['School City', 'School State']).set_index((['School City', 'School State']))
    byLocation = byLocation.reset_index()
    
    # calculate the rank of city according to amount of donation received
    byLocation['byLocation rank']= byLocation['Donation Amount'].rank(ascending = False) # rank from largest to smallest
    
    # calculate the Frequent location bonus scoring scale
    byLocation['byLocation scale'] = 1/(byLocation['byLocation rank']/len(byLocation))/100
    
    # add longitude and latitude data for reference later
    byLocation = byLocation.set_index(['School City', 'School State'])
    byLocation = byLocation.merge(locMap.filter(items = ['School_Lon', 'School_Lat']), left_index = True, right_index = True, how = 'left')
    byLocation = byLocation.reset_index()
    byLocation['CityState'] = byLocation['School City'].map(str)+', '+byLocation['School State']
    
    # donor's location
    donorLocation = donors.loc[donors['Donor ID'] == ID, ['Donor City', 'Donor State', 'Donor_Lat', 'Donor_Lon']].reset_index(drop = True)
    donorLocation['CityState'] = donorLocation['Donor City'].map(str)+', '+donorLocation['Donor State']
    
    # get the ID masks for filtering donor donations
    schoolIDs = donorDonations['School ID'].unique()
    teacherIDs = donorDonations['Teacher ID'].unique()
    locationIDs = byLocation['CityState'].unique()
    homeIDs = donorLocation['CityState'].unique() 
    
    ##
    # Filter projectFeatures for Schools, Teachers, Locations that the donor has ever donated to
    ##
    
    projSchool = list(projectFeatures[projectFeatures['School ID'].isin(schoolIDs)]['Project ID'])
    projTeacher = list(projectFeatures[projectFeatures['Teacher ID'].isin(teacherIDs)]['Project ID'])
    projLocation = list(projectFeatures[projectFeatures['CityState'].isin(locationIDs)]['Project ID'])
    projHomeLoc = list(projectFeatures[projectFeatures['CityState'].isin(homeIDs)]['Project ID'])
    projAll = set(projSchool).union(set(projTeacher)).union(set(projLocation)).union(set(projHomeLoc))
    
    ##
    #  Filter the project Features based on 'Project IDs'
    ##
    
    projUniverse = projectFeatures.loc[projectFeatures['Project ID'].isin(projAll)]
    #print('Number of potential projects:', len(projUniverse))
    
    ##
    # Generate Donor Feature Matrix, Calculate Score, Rank Recommendations
    ##
    
    # Do the following if there are 1 or more potential projects to choose from
    if len(projAll) >=1: 
        
        #### Get Donor FeatureMatrix #### 
        
        donorFeatureMatrix = donorFeatureMatrix.set_index('Donor ID')  # if donorFeatureMatrix did not have index set
        y = donorFeatureMatrix.loc[ID, 'Project Type_Professional Development':'ProjCat_Warmth, Care & Hunger'] 
        y = y.values.reshape(len(y), 1) 

        #### Calculate Similarity Score
        score = np.dot(projUniverse.loc[:, 'Project Type_Professional Development':'ProjCat_Warmth, Care & Hunger'], y)

        #### Add more Scoring Attributes
        
        # score from dot product of similarity
        projUniverse['Score_interest'] = score
        
        # Flag as 1 if the project matches the conditions
        projUniverse['Score_School'] = projUniverse['Project ID'].isin(projSchool)
        projUniverse['Score_Teacher'] = projUniverse['Project ID'].isin(projTeacher)
        projUniverse['Score_homeLoc'] = projUniverse['Project ID'].isin(projHomeLoc)
        projUniverse['Score_priorLoc'] = projUniverse['Project ID'].isin(projLocation)
        
        # merge the scoring matrix
        projUniverse = projUniverse.merge(byLocation.filter(items = ['byLocation rank', 'byLocation scale', 'CityState']), on = 'CityState', how = 'left')

        #### Rank Recommendations #### 
        cols = ['Score_interest', 'Score_School', 'Score_Teacher','Score_homeLoc', 'Score_priorLoc']
        projUniverse['Score_Total_Unadjusted'] = projUniverse.loc[:, cols].sum(axis = 1)
        
        projUniverse['location_Premium'] = projUniverse['Score_Total_Unadjusted']*projUniverse['byLocation scale']

        cols = ['Score_interest', 'Score_School', 'Score_Teacher','Score_homeLoc', 'Score_priorLoc', 'location_Premium']
        projUniverse['Score_Total'] = projUniverse.loc[:, cols].sum(axis = 1)
        
        projUniverse['Rank'] = projUniverse['Score_Total'].rank(ascending = False)
        projUniverse = projUniverse.sort_values(by = 'Rank' )
        projUniverse = projUniverse.set_index('Project ID')
        
        # Length of potential project selections
        lenUniv = len(projUniverse)

    else:
        lenUniv = 0
    
    ##
    # Identify the actual project that the donor donated and find the recommender ranking of the correct project
    ##
    
    #### Get the actual project that the donor donated to
    ans = donations_2018[donations_2018['Donor ID'] == ID]['Project ID']
    
    # loop through multiple projects for donors that donated to multiple projects in the testing timeframe
    for i in range(len(ans)):
        proj = ans.values[i]

        ### Get ranking of the project that the donor donated to
        try:
            ranking = projUniverse.loc[proj]['Rank']
            
            # if project matches, skip the rest of the search
            break  
            
        except:
            # if cannot find matching project
            ranking = np.nan 
    #print('Ranking of correct response:', ranking)
        
    ### Return dictionary 
    response = { 'Donor ID': ID, 'Ranking': ranking, 'Donor Project ID': proj, 
                'Filter Size': lenUniv, 'Universe': projUniverse, 'Donor Donations': donorDonations,
               'Prior Location Count': len(locationIDs)}
    
    return response


# ###  An example of the recommender ouput

# In[21]:


response = recommender('01487813310e283992cfd5249c6cd722')


# In[22]:


print("The project chosen by the Donor is ranked:", response['Ranking'])
print("The donor donated to Project ID:", response['Donor Project ID'])
print("Total projects in the filtered Universe is:", response['Filter Size'])
print("Number of projects the donor donated to prior to 2018:", len(response['Donor Donations']))


# #### Top 5 recommendations from Recommender

# In[23]:


cols = ['Score_interest', 'Score_School', 'Score_Teacher', 'Score_homeLoc','Score_priorLoc', 'byLocation rank', 'byLocation scale','Score_Total_Unadjusted', 'location_Premium', 'Score_Total', 'Rank']
responseScores = response['Universe'][cols][0:5]
picks = response['Universe'].index[0:5]
recommendedProj = projects.loc[projects['Project ID'].isin(picks)]
TopRecommendations = recommendedProj.merge(schools, left_on= 'School ID', right_on = 'School ID', how = 'left')
TopRecommendations = TopRecommendations.merge(responseScores, left_on= 'Project ID', right_on = 'Project ID', how = 'left')
TopRecommendations.sort_values(by = 'Rank', ascending = True)


# #### Recommender score for Donor's chosen project

# In[24]:


i = response['Ranking'].astype('int')
cols = ['Score_interest', 'Score_School', 'Score_Teacher', 'Score_homeLoc','Score_priorLoc', 'byLocation rank', 'byLocation scale','Score_Total_Unadjusted', 'location_Premium', 'Score_Total', 'Rank']
responseScores = response['Universe'][cols][i-1: i]
picks = response['Donor Project ID']
correctProj = projects.loc[projects['Project ID'] == picks]
correctProj = correctProj.merge(schools, left_on= 'School ID', right_on = 'School ID', how = 'left')
correctProj = correctProj.merge(responseScores, left_on= 'Project ID', right_on = 'Project ID', how = 'left')
correctProj 


# #### Most recent projects that donor donated to

# In[25]:


response['Donor Donations'].sort_values(by = 'Donation Received Date', ascending = False).head()


# #### Donor Profile

# In[26]:


donorSummary.loc[donorSummary['Donor ID'] == response['Donor ID']]


# ### <a id='4'>4. Recommender Performance</a>

# In[28]:


def accuracy(test_ID, 
             numSample, 
             donations_prior = donations_prior):
    
    donorDonations = merge_data(test_ID, donations_prior)
    recommendations = pd.DataFrame(columns=['Donor ID', 'Ranking', 'Donor Project ID', 
                                            'Filter Size', 'Prior Donation Count', 'Prior Location Count'])
    IDs = pd.Series(test_ID).sample(n= numSample, random_state = 513)
    recommendations['Donor ID'] = IDs
    recommendations = recommendations.set_index('Donor ID', drop = False)
    i = 1
    
    for ID in IDs:
        #print('Processing #:', i)
        response = recommender(ID, donorDonations)
        recommendations.loc[ID, 'Ranking'] = response['Ranking']
        recommendations.loc[ID, 'Donor Project ID'] = response['Donor Project ID']
        recommendations.loc[ID, 'Filter Size'] = response['Filter Size']
        recommendations.loc[ID, 'Prior Donation Count'] = len(response['Donor Donations'])
        recommendations.loc[ID, 'Prior Location Count'] = response['Prior Location Count']
        
        i+=1
    return recommendations


# #### Run samples  to observe the recommender results

# In[29]:


recommendations = accuracy(test_ID, 1000)


# In[30]:


recommendations.head()


# ###  Filtering projects using donor's previous donation history based on home city and school cities captured the future donation approximately  80% of the times.

# In[31]:


print('% of time the filter captured the correct project:', recommendations['Ranking'].notnull().sum()/len(recommendations))


# ### Frequency that the recommender ranks the donor's chosen project as No. 1

# In[32]:


chartData = recommendations.copy()
chartData['Ranking'] = chartData['Ranking'].astype('Float32')
chartData['Ranking Range'] = pd.cut(chartData['Ranking'], bins = [0, 1, 5, 10, 25, 50, 100, 70000], labels = ['1', '2-5', '6-10', '11-25', '26-50', '51-100', '>100'])
chartData['Ranking Range']= chartData['Ranking Range'].astype(str)
chartData = chartData.groupby('Ranking Range').agg('count')
chartData['Frequency'] = chartData['Donor ID']/chartData['Donor ID'].sum()*100
chartData = chartData.reindex(index = ['1', '2-5', '6-10', '11-25', '26-50', '51-100', '>100', 'nan'])

g =sns.barplot(chartData.index, chartData['Frequency'])
g.set(xlabel="Recommender's Ranking", ylabel='Frequency %', title = "Recommender's ranking of donor's chosen Project")

print('% Hit in top 1 Recommendations:', chartData['Frequency'][0:1].sum() )
print('% Hit in top 5 Recommendations:', chartData['Frequency'][0:2].sum() )
print('% Hit in top 10 Recommendations:', chartData['Frequency'][0:3].sum() )
print('% Hit in top 25 Recommendations:', chartData['Frequency'][0:4].sum() )


# ### <a id='5'>5. Future Enhancements</a>

# The current filtering method limits the search to cities that the donor either live in or have donated to. As discussed in the previous section, the filter underperforms when the donor donated to less cities especially in cities with low number of available projects.  To solve this problem, a minimal project universe can be set.  When the number of projects returned by the filter is lower than the threshold, the search can be expanded to to nearby cities within 15 miles radius of the donor's home. 15 miles was the median distance between donor's home and the schools that the donor donated to.   
# 
# To address the model not identifying donors with a specific cause in mind, such as helping the underperviledged schools and students with special needs.  The model could be modified to screen for special interest by using the features that measures donor's interest against the population mean.  These features are already created in the donorFeatureMatrix.  If the donor's interest is around 70% above the average donor, then the recommender should automatically recommend projects associated to these causes.  
# 
# Currently, the model uses a very simple weighting system in scoring. The different components of the scoring matrix could be tuned by identifying the best coefficients to boost the different dimensions. 
# 
# Futhermore, the current methodology only use the donor's prior donation history as a reference. Collaborative filtering can be introduced to compare the donor against similar donors. Similarity between donors could be measured by whether they live in the same city, donated to the same school, same teacher, same projects, or have broadly similar interest in terms of the other dimensions.
# 

# In[ ]:




