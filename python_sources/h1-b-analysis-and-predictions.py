#!/usr/bin/env python
# coding: utf-8

# In this kernel I will analyze the data of H-1B petitions during 2011-2016. I will find top companies having the most numbers of applications and their offering salary, average salary and salaries of some most popular jobs. Most popular jobs from top companies and from whole market are also identified. The geographic dependence of jobs market is also analyzed without/with regional price parity taken into account. There are detailed analysis and prediction for the hot Data Scientist job as well. Please enjoy the analysis and give comment below. Thanks.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from mpl_toolkits.basemap import Basemap

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
h1bdat = pd.read_csv('../input/h1b_kaggle.csv')
#Note: Column "Unnamed: 0" (another indexing starting from 1) will be removed
h1bdat = h1bdat.drop(["Unnamed: 0"], axis=1)


# In[ ]:


print('Number of entries:', h1bdat.shape[0])
print('Number of missing data in each column:')
print(h1bdat.isnull().sum())


# Except for location "loc" and "lat" columns, there are very few missing data for each columns. The number of missing data of location "loc" and "lat" columns are still small number (3.3%) in comparison with the whole data set.

# # Who are the main players?
# 
# Top 10 Applicants in term of the numbers of applications in 2011, in 2016 and from 2011 to 2016

# In[ ]:


ax1 = h1bdat['EMPLOYER_NAME'][h1bdat['YEAR'] == 2011].groupby(h1bdat['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Applicants, 2011")
ax1.set_ylabel("")
plt.show()
ax2 = h1bdat['EMPLOYER_NAME'][h1bdat['YEAR'] == 2016.0].groupby(h1bdat['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Applicant, 2016")
ax2.set_ylabel("")
plt.show()
ax3 = h1bdat['EMPLOYER_NAME'].groupby([h1bdat['EMPLOYER_NAME']]).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Applicant over 2011 to 2016")
ax3.set_ylabel("")
plt.show()


# We will analyze more about the number of applications from top 10 EMPLOYERS as functions of time. A question here is that how do we choose top 10 EMPLOYERS having the most applications!
# Above plots showed that if we use the accumulated numbers of applications to select 10 top EMPLOYERS, we will miss new comers (like CAPGEMINI AMERICA or TECH MAHINDRA), who are playing significant role in the present. So it may be better to use the accumulated number of applications during last 2 or 3 years. I will use 2.

# In[ ]:


topEmp = list(h1bdat['EMPLOYER_NAME'][h1bdat['YEAR'] >= 2015.0].groupby(h1bdat['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).index)
byEmpYear = h1bdat[['EMPLOYER_NAME', 'YEAR', 'PREVAILING_WAGE']][h1bdat['EMPLOYER_NAME'].isin(topEmp)]
byEmpYear = byEmpYear.groupby([h1bdat['EMPLOYER_NAME'],h1bdat['YEAR']])


# In[ ]:


markers=['o','v','^','<','>','d','s','p','*','h','x','D','o','v','^','<','>','d','s','p','*','h','x','D']
fig = plt.figure(figsize=(12,7))
for company in topEmp:
    tmp = byEmpYear.count().loc[company]
    plt.plot(tmp.index.values, tmp["PREVAILING_WAGE"].values, label=company, linewidth=2,marker=markers[topEmp.index(company)])
plt.xlabel("Year")
plt.ylabel("Number of Applications")
plt.legend()
plt.title('Number of Applications of Top 10 Applicants')
plt.show()


# We can see clearly from the figure that there are 2 new big players: CAPGEMINI AMERICA and TECH MAHINDRA.
# From this plot we can draw some interesting points:
#     - INFOSIS shows a very rapid development, especially during period from 2011 to 2013, where it came from about zero to 32k applications!
#     - TATA also shows a significant development.
#     - Except for 2 new comers, we can see that all top EMPLOYERS have peaks of numbers of applycations at the year of 2015 and a trend of decreasing to 2016. There could be a social or political event relating to this trend!
#     - All very top applicants are from India.
# 
# Those are companies who filed the most numbers of applications. What about their payments? Certainly, it depends on the specific job, but it is also informative to have a look at the average salary from each company.

# In[ ]:


fig = plt.figure(figsize=(10,7))
for company in topEmp:
    tmp = byEmpYear.mean().loc[company]
    plt.plot(tmp.index.values, tmp["PREVAILING_WAGE"].values, label=company, linewidth=2,  marker=markers[topEmp.index(company)])

plt.xlabel("Year")
plt.ylabel("Average Salary Offer (USD)")
plt.legend()
plt.title("Average Salary of Top 10 Applicants")
plt.show()


# It is very interesting to see a huge peak in 2014 from "IBM INDIA PRIVATE LIMITED", looking like something going wrong. The other suspicious peaks are from INFOSIS at 2012, ACCENTURE and WIPRO at 2011 and TECH MAHINDRA at 2016. We need to further analyze these peaks.
# We will see clearly below that there are some outliers (salary of billion dollars per year!) causing these weird behaviors. This outliers are very likely coming from the unit conversions (between salary rate per hours, day, week or month to per year) when the data were collected and converted.
# The numbers of outliers are very small in comparison to the number of entries for corresponding EMPLOYERS for each year, so I will remove these outliers. It should be safe to consider the salary higher than half a milion as outliers. There certainly jobs such as CEO having this kind of on the sky high salary. But they are not good representatives for the distribution since we have small number of entries, typically in order of 10000, for each EMPLOYER, therefor few extreme outliers can skew the distribution quite a lot.
# Note: The above analyses about the number of applications with time are still valid. They are independent with the errors on converting salary from different unit to a common one of USD per year.
# Bellow are example of outliers from "IBM INDIA PRIVATE LIMITED" and "ACCENTURE LLP":

# In[ ]:


for company in ['IBM INDIA PRIVATE LIMITED','ACCENTURE LLP']:
    print(h1bdat[['EMPLOYER_NAME','PREVAILING_WAGE','YEAR']][h1bdat['EMPLOYER_NAME']==company].sort_values(['PREVAILING_WAGE'], ascending=False).head(15))


# # Cleaning dataset
# We will remove all entries having PREVAILING_WAGE higher than 500000 as discussed above. 

# In[ ]:


h1bdat = h1bdat[h1bdat['PREVAILING_WAGE'] <= 500000]
byEmpYear = h1bdat[['EMPLOYER_NAME', 'YEAR', 'PREVAILING_WAGE']][h1bdat['EMPLOYER_NAME'].isin(topEmp)]
byEmpYear = byEmpYear.groupby([h1bdat['EMPLOYER_NAME'],h1bdat['YEAR']])


# In[ ]:


fig = plt.figure(figsize=(10,7))
for company in topEmp:
    tmp = byEmpYear.mean().loc[company]
    plt.plot(tmp.index.values, tmp["PREVAILING_WAGE"].values, label=company, linewidth=2,  marker=markers[topEmp.index(company)])
        
plt.ylim(50000,110000)
plt.xlabel("Year")
plt.ylabel("Average Salary Offer (USD)")
plt.legend()
plt.title("Salary From Top 10 Applicants")
plt.show()


# Plot looks reasonable now. We can say something from this plot: 
#     1. MICROSOFT CORPORATION is paying the highest offering salary and there is a gap between its offering salary with all other EMPLOYERS. Its offering salary is also increasing with  time. 
#     2. HCL AMERICA INC shows a moderate sudden jump of 15~20% between 2012 and 2013.
#     3. TATA is the 2nd biggest applicant but they has lowest offering salary among top 10 big applicants. Larger size of company does not warranty better salary.
#     4. The new comer CAPGEMINI AMERICA is very impressive. It shows both very high number of applications and offering salary. We should keep an eye on this guy for coming years.

# # What are the most popular jobs and their corresponding salary those top 10 companies hiring?

# In[ ]:


PopJobs = h1bdat[['JOB_TITLE', 'EMPLOYER_NAME', 'PREVAILING_WAGE']][h1bdat['EMPLOYER_NAME'].isin(topEmp)].groupby(['JOB_TITLE'])
topJobs = list(PopJobs.count().sort_values(by='EMPLOYER_NAME', ascending=False).head(30).index)
df = PopJobs.count().loc[topJobs].assign(mean_wage=PopJobs.mean().loc[topJobs])
fig = plt.figure(figsize=(10,12))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
width = 0.35
df.EMPLOYER_NAME.plot(kind='barh', ax=ax1, color='C0', width=0.4, position=0, label='# of Applications')
df.mean_wage.plot(kind='barh', ax=ax2, color='C7', width=0.4, position=1, label='Mean Salary')
ax1.set_xlabel('Number of Applications')
ax1.set_ylabel('')
ax1.legend(loc=(0.75,0.55))
ax2.set_xlabel('Mean Salary')
ax2.set_ylabel('Job Title')
ax2.legend(loc=(0.75,0.50))
plt.show()


# Top jobs from top 10 applications are mostly IT jobs. The most popular jobs are "Technology Lead" and "Technology Analyst" but their offering salary are not top, they are about middle. The top offering salary is "Manager", which is absolutely understandable. Bellow is the distribution of salary for top 20 popular jobs from top 10 applicants. The distribution has mean = 68.4k, median = 66.6k, mode = 60.0k and deviation = 13.5k.

# In[ ]:


ax = h1bdat[['JOB_TITLE', 'EMPLOYER_NAME', 'PREVAILING_WAGE']][h1bdat['EMPLOYER_NAME'].isin(topEmp)  & h1bdat['JOB_TITLE'].isin(topJobs)]['PREVAILING_WAGE'].hist(bins=100)
ax.set_ylabel('Offering Wage (USD/year)')
plt.title('Offering Salary Distribution of Popular Jobs from Top Applicants')
plt.show()


# ## How about popular jobs in entire jobs market, certainly evaluating via the H-1B applications as an approximate distribution.

# In[ ]:


PopJobsAll = h1bdat[['JOB_TITLE', 'EMPLOYER_NAME', 'PREVAILING_WAGE']].groupby(['JOB_TITLE'])
topJobsAll = list(PopJobsAll.count().sort_values(by='EMPLOYER_NAME', ascending=False).head(30).index)
dfAll = PopJobsAll.count().loc[topJobsAll].assign(mean_wage=PopJobsAll.mean().loc[topJobsAll])
fig = plt.figure(figsize=(10,12))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
width = 0.35
dfAll.EMPLOYER_NAME.plot(kind='barh', ax=ax1, color='C0', width=0.4, position=0, label='# of Applications')
dfAll.mean_wage.plot(kind='barh', ax=ax2, color='C7', width=0.4, position=1, label='Mean Salary')
ax1.set_xlabel('# of Applications')
ax1.set_ylabel('')
ax1.legend(loc=(0.75,0.55))
ax2.set_xlabel('Mean wage')
ax2.set_ylabel('Job Title')
ax2.legend(loc=(0.75,0.50))
plt.show()


# We can see that indeed IT jobs are the most popular ones. There are only few non-IT jobs in top 20. All top 5 polular jobs are IT. Note that the #6 is "Business analyst", which also needs many computer and analysis skills.
# Among IT jobs, we can see a very interesting fact that "Programmer analyst" and "Software engineer" are very dominant in the whole job market, especially the first one. But these 2 jobs are not that dominant in top 10 applicants (plot above). This implies that the job demand category of top 10 applicants is a bit different with that of whole market. Note that number of applications from top 10 applicants is about 14% of whole market, which is not very signicicant but indeed high.
# 
# It is very interesting to see "Assistant professor" in the list of the most popular jobs (#11, just miss top10!). I don't think we are expecting to see it in this list.
# 
# Another interesting observation here is that "Data Scientist" jobs is not in the list. People are talking about it everyday at every corners. Posted jobs for Data Scientist can be found plenty on any job searching site like Glassdoor. The reason for this missing is that, although 
# the number of Data Scientist jobs is increasing rapidly recently (see figure bellow), but it has just become very hot recently and it needs time to have large enough number of jobs to be in the top list!

# ## Data Scientist Job: Analysis and Prediction
# Since Data Scientist job is hot, so let do some analysis and prediction on the number of Data Scientist jobs. The regression will be used, where the feature is a power function of year.
# From figure bellow we can see that the number of Data scientist job is increasing rapidly (power of 2.32) and we expect to have about 1.5k and 2.1k Data Scientist jobs in 2017 and 2018.

# In[ ]:


dsj = h1bdat[['JOB_TITLE','YEAR']][h1bdat['JOB_TITLE'] == "DATA SCIENTIST"].groupby('YEAR').count()['JOB_TITLE']
X = np.array(dsj.index)
Y = dsj.values
def func(x, a, b, c):
    return a*np.power(x-2011,b)+c

popt, pcov = curve_fit(func, X, Y)
X1 = np.linspace(2011,2018,9)
X2 = np.linspace(2016,2018,3)
X3 = np.linspace(2017,2018,2)
fig = plt.figure(figsize=(7,5))
plt.scatter(list(dsj.index), dsj.values, c='C0', marker='o', s=120, label='Data')
plt.plot(X1, func(X1,*popt), color='C0', label='')
plt.plot(X2, func(X2,*popt), color='C5', linewidth=3, marker='s', markersize=1, label='')
plt.plot(X3, func(X3,*popt), color='C5', marker='s', markersize=10, label='Prediction')
plt.legend()
plt.title('Number of Data Scientist Jobs')
plt.xlabel('Year')
plt.show()


# ### Who hire data scientist the most and where do data scientists work?

# In[ ]:


ax1 = h1bdat[h1bdat['JOB_TITLE'] == "DATA SCIENTIST"]['EMPLOYER_NAME'].groupby(h1bdat['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Data Scientist Hiring Companies, 2011-2016")
ax1.set_ylabel("")
plt.show()
ax2 = h1bdat[h1bdat['JOB_TITLE'] == "DATA SCIENTIST"]['EMPLOYER_NAME'][h1bdat['YEAR'] == 2016.0].groupby(h1bdat['EMPLOYER_NAME']).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Data Scientist Hiring Companies, 2016")
ax2.set_ylabel("")
plt.show()
ax3 = h1bdat[h1bdat['JOB_TITLE'] == "DATA SCIENTIST"]['WORKSITE'].groupby(h1bdat['WORKSITE']).count().sort_values(ascending=False).head(10).plot(kind='barh', title="Top City Data Scientist Work In, 2011-2016")
ax3.set_ylabel("")
plt.show()


# This result is not very surprised when top Data Scientist hiring companies are tech and IT companies. But it is still a bit surprised that half of the number of applications for Data scientist from MICROSOFT CORPORATION was in 2016!
# 
# It is also interesting that Wal-mart is in the top list for 2011 to 2016 period. Someone may ask where is Amazon? The reason could be that, as I know, Amazon does not call "the job" as Data Scientist. Sometimes they call "the job" as research/applied scientist!
# 
# Since all top IT and tech companies have offices in Silicon Valley, New York and Seattle area, top cities Data scientists work in are all in California, New York and Washington.

# # Currently popular jobs on the market
# 
# Above analysis of popular jobs is based on accumulated number of jobs. Let have a look at the similar plot for the jobs in 2016 only.

# In[ ]:


PopJobs2016 = h1bdat[['JOB_TITLE', 'EMPLOYER_NAME', 'PREVAILING_WAGE','YEAR']]
PopJobs2016 = PopJobs2016[PopJobs2016['YEAR']==2016].groupby(['JOB_TITLE'])
topJobs2016 = list(PopJobs2016.count().sort_values(by='EMPLOYER_NAME', ascending=False).head(30).index)
df2016 = PopJobs2016.count().loc[topJobs2016].assign(mean_wage=PopJobs2016.mean().loc[topJobs2016]['PREVAILING_WAGE'])
fig = plt.figure(figsize=(10,12))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
width = 0.35
df2016.EMPLOYER_NAME.plot(kind='barh', ax=ax1, color='C0', width=0.4, position=0, label='# of Applications')
df2016.mean_wage.plot(kind='barh', ax=ax2, color='C7', width=0.4, position=1, label='Mean Salary')
ax1.set_xlabel('Number of Applications')
ax1.set_ylabel('')
ax1.legend(loc=(0.75,0.55))
ax2.set_xlabel('Mean Salary')
ax2.set_ylabel('Job Title')
ax2.legend(loc=(0.75,0.50))
plt.show()


# The picture is very similar to that of whole time period from 2011 - 2016, meaning that the structure of jobs in the market do not change significantly from 2011 to 2016.

# # Geographic differences of job market
# I will do some analyses for number of jobs and salary based on states. If we plot the data directly out from the lantitude and longtitude on a US map (kind of company based), some spots will have too many overlap data points, making it difficult to see. Therefore, I analyze the variation of numbers of jobs and salary from state to state. I will use state capitols for positions of each state. Since the data include applications from Puerto Rico and there are some entries with missing info of state as can be seen below, I will remove those entries before doing next analyses.

# In[ ]:


# I don't know how to add additional file to the folder here so I paste whole dictionary for state capitol positions here ^^
capitolpos = {'ALABAMA': (-86.79113, 32.806671), 'ALASKA': (-152.404419, 61.370716), 'ARIZONA': (-111.431221, 33.729759),
 'ARKANSAS': (-92.373123, 34.969704), 'CALIFORNIA': (-119.681564, 36.116203), 'COLORADO': (-105.311104, 39.059811),
 'CONNECTICUT': (-72.755371, 41.597782), 'DELAWARE': (-75.507141, 39.318523), 'DISTRICT OF COLUMBIA': (-77.026817, 38.897438),
 'FLORIDA': (-81.686783, 27.766279), 'GEORGIA': (-83.643074, 33.040619), 'HAWAII': (-157.498337, 21.094318),
 'IDAHO': (-114.478828, 44.240459), 'ILLINOIS': (-88.986137, 40.349457), 'INDIANA': (-86.258278, 39.849426),
 'IOWA': (-93.210526, 42.011539), 'KANSAS': (-96.726486, 38.5266), 'KENTUCKY': (-84.670067, 37.66814),
 'LOUISIANA': (-91.867805, 31.169546), 'MAINE': (-69.381927, 44.693947), 'MARYLAND': (-76.802101, 39.063946),
 'MASSACHUSETTS': (-71.530106, 42.230171), 'MICHIGAN': (-84.536095, 43.326618), 'MINNESOTA': (-93.900192, 45.694454),
 'MISSISSIPPI': (-89.678696, 32.741646), 'MISSOURI': (-92.288368, 38.456085), 'MONTANA': (-110.454353, 46.921925),
 'NEBRASKA': (-98.268082, 41.12537), 'NEVADA': (-117.055374, 38.313515), 'NEW HAMPSHIRE': (-71.563896, 43.452492),
 'NEW JERSEY': (-74.521011, 40.298904), 'NEW MEXICO': (-106.248482, 34.840515), 'NEW YORK': (-74.948051, 42.165726),
 'NORTH CAROLINA': (-79.806419, 35.630066), 'NORTH DAKOTA': (-99.784012, 47.528912), 'OHIO': (-82.764915, 40.388783),
 'OKLAHOMA': (-96.928917, 35.565342), 'OREGON': (-122.070938, 44.572021), 'PENNSYLVANIA': (-77.209755, 40.590752),
 'RHODE ISLAND': (-71.51178, 41.680893), 'SOUTH CAROLINA': (-80.945007, 33.856892), 'SOUTH DAKOTA': (-99.438828, 44.299782),
 'TENNESSEE': (-86.692345, 35.747845), 'TEXAS': (-97.563461, 31.054487), 'UTAH': (-111.862434, 40.150032),
 'VERMONT': (-72.710686, 44.045876), 'VIRGINIA': (-78.169968, 37.769337), 'WASHINGTON': (-121.490494, 47.400902),
 'WEST VIRGINIA': (-80.954453, 38.491226), 'WISCONSIN': (-89.616508, 44.268543), 'WYOMING': (-107.30249, 42.755966)}


# In[ ]:


h1bdat = h1bdat.assign(state=h1bdat['WORKSITE'].str.split(',').str.get(1).str.strip())
stlist = list(capitolpos.keys())
h1bdat[~h1bdat.state.isin(stlist)]['state'].value_counts()
h1bdat = h1bdat[h1bdat.state.isin(capitolpos.keys())]


# In[ ]:


sbystate = h1bdat[['state','PREVAILING_WAGE']].groupby(h1bdat['state']).count()['state']
X = []
Y = []
for state in list(sbystate.index):
    (lon, lan) = capitolpos[state]
    X.append(lon)
    Y.append(lan)
fig = plt.figure(figsize=(16,8))
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
m.drawcoastlines()
m.drawcountries(linewidth=3, color='C0')
m.drawstates(linewidth=1.5)
xmap, ymap = m(X, Y)
ax = m.scatter(xmap, ymap, s=1000, c=sbystate.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)
cb = plt.colorbar(ax)
plt.legend()
plt.title('Number of Applications by State (x1000)')
#plt.show()
sbystate = h1bdat[['state','PREVAILING_WAGE']].groupby(h1bdat['state']).mean()['PREVAILING_WAGE']
X = []
Y = []
for state in list(sbystate.index):
    (lon, lan) = capitolpos[state]
    X.append(lon)
    Y.append(lan)
fig = plt.figure(figsize=(16,8))
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
m.drawcoastlines()
m.drawcountries(linewidth=3, color='C0')
m.drawstates(linewidth=1.5)
xmap, ymap = m(X, Y)
ax = m.scatter(xmap, ymap, s=1000, c=sbystate.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)
cb = plt.colorbar(ax)
plt.legend()
plt.title('Average Salary by State (x1000 USD/year)')
plt.show()


# California filed the most applications for H1-B visa. The 2nd tie includes Taxes, New York, New Jersey, Illinois and Washington. This is because those states have big and very dynamic economic centers with many tech and IT companies residing in such as San Francisco, Chicago, New York and Seattle. For salary, California and Wahsington are leading then some north east states (New York, New Jersey, Massachusetts, ..), Oregon, North Dakota and Wyoming.
# 
# Wait! We all know that North Dakota and Wyoming are not on top economic centers. Howcome are they on the top of high salary?It turns out that the average salary (over all jobs) comparison we are doing here is not a good practice. Since there is regional dependence on type of jobs, i.e., west coast may have more IT and tech jobs, east coast may have more finance jobs, and mid-west may have more argricalture jobs, and different jobs have different salaries. For this specific case of Wyoming and North Dakota, there are also two more resons (will see right below): 1. The number of petitions from these 2 states are quite small (less than 1000) for a good statistics. 2. Quite some petitions are for very high pay jobs like doctor and physician (skewing the distribution, especially in case of small number of sample). 
# 
# Therefore, comparing salary of a same job in different states is more meaningful. If you have a set of skills suitable for some certain jobs and want to find a place to have as high as possible income, you need to plot the regional dependent data for those jobs.

# In[ ]:


print("Number of petitions: top 5 states having least peritions")
print(h1bdat['state'].groupby(h1bdat['state']).count().sort_values().head(5))
print("")
print('Top high salary jobs in Wyoming')
print(h1bdat[h1bdat['state']=='WYOMING'][['JOB_TITLE','PREVAILING_WAGE']].groupby(h1bdat['JOB_TITLE']).mean().sort_values(by='PREVAILING_WAGE',ascending=False).head(5))
print("")
print('Top high salary jobs in North Dakota')
print(h1bdat[h1bdat['state']=='NORTH DAKOTA'][['JOB_TITLE','PREVAILING_WAGE']].groupby(h1bdat['JOB_TITLE']).mean().sort_values(by='PREVAILING_WAGE',ascending=False).head(5))


# # Geographic Salary Dependence of some popular jobs
# Let check geographic dependence of salaries (average over whole period 2011-2016) of 2 most popular jobs.

# In[ ]:


sbystate = h1bdat[h1bdat['JOB_TITLE']=='PROGRAMMER ANALYST'][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).mean()
sbystate2 = h1bdat[h1bdat['JOB_TITLE']=='PROGRAMMER ANALYST'][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).count()
X = []
Y = []
for state in list(sbystate.index):
    (lon, lan) = capitolpos[state]
    X.append(lon)
    Y.append(lan)
fig = plt.figure(figsize=(16,8))
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
m.drawcoastlines()
m.drawcountries(linewidth=3, color='C0')
m.drawstates(linewidth=1.5)
xmap, ymap = m(X, Y)
ax = m.scatter(xmap, ymap, s=1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate.PREVAILING_WAGE.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)

cb = plt.colorbar(ax)
plt.legend()
plt.title('Average "Programmer Analyst" Salary by State (x1000 USD/year), 2011-2016')
####
sbystate2 = h1bdat[h1bdat['JOB_TITLE']=='SOFTWARE ENGINEER'][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).count()
X = []
Y = []
for state in list(sbystate.index):
    (lon, lan) = capitolpos[state]
    X.append(lon)
    Y.append(lan)

fig = plt.figure(figsize=(16,8))
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
m.drawcoastlines()
m.drawcountries(linewidth=3, color='C0')
m.drawstates(linewidth=1.5)
xmap, ymap = m(X, Y)
ax = m.scatter(xmap, ymap, s = 1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate.PREVAILING_WAGE.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)
cb = plt.colorbar(ax)
plt.legend()
plt.title('Average "Software Engineer" Salary by State (x1000 USD/year), 2011-2016')
plt.show()


# For the number of jobs, it looks as what we expect. There are more IT jobs in west and east coast. In term of salary for 2 most popular IT jobs, Oklahoma has the lowest salary. The 2nd lowest tie includes North Datoka, Whyoming, Montana and Idaho.

# # How about living expense? It is also varying from state to state!
# Let compare the real purchasing power based on salary of above 2 popular jobs. The real purchasing power is the income adjusted by Regional Price Parity (RPP), which measures the differences in price levels across states and can be found from here: https://www.bea.gov/newsreleases/regional/rpp/rpp_newsrelease.htm. RPP for 2016 has not been released yet, so I will analyze for 2015 data. I will plot the "original" and "adjusted" salaries plots together for above 2 popular jobs.

# In[ ]:


# Dictionary of 2015 RPP:
rpp2015 = {'ALABAMA': 0.868, 'ALASKA': 1.056, 'ARIZONA': 0.962, 'ARKANSAS': 0.874, 'CALIFORNIA': 1.134,
 'COLORADO': 1.032, 'CONNECTICUT': 1.087, 'DELAWARE': 1.004, 'DISTRICT OF COLUMBIA': 1.17, 'FLORIDA': 0.995,
 'GEORGIA': 0.926, 'HAWAII': 1.188, 'IDAHO': 0.934, 'ILLINOIS': 0.997, 'INDIANA': 0.907,
 'IOWA': 0.903, 'KANSAS': 0.904, 'KENTUCKY': 0.886, 'LOUISIANA': 0.906, 'MAINE': 0.98,
 'MARYLAND': 1.096, 'MASSACHUSETTS': 1.069, 'MICHIGAN': 0.935, 'MINNESOTA': 0.974, 'MISSISSIPPI': 0.862,
 'MISSOURI': 0.893, 'MONTANA': 0.948, 'NEBRASKA': 0.906, 'NEVADA': 0.98, 'NEW HAMPSHIRE': 1.05,
 'NEW JERSEY': 1.134, 'NEW MEXICO': 0.944, 'NEW YORK': 1.153, 'NORTH CAROLINA': 0.912, 'NORTH DAKOTA': 0.923,
 'OHIO': 0.892, 'OKLAHOMA': 0.899, 'OREGON': 0.992, 'PENNSYLVANIA': 0.979, 'RHODE ISLAND': 0.987,
 'SOUTH CAROLINA': 0.903, 'SOUTH DAKOTA': 0.882, 'TENNESSEE': 0.899, 'TEXAS': 0.968, 'UNITED STATES': 1.0,
 'UTAH': 0.97, 'VERMONT': 1.016, 'VIRGINIA': 1.025, 'WASHINGTON': 1.048, 'WEST VIRGINIA': 0.889,
 'WISCONSIN': 0.931, 'WYOMING': 0.962}


# In[ ]:


sbystate = h1bdat[h1bdat['JOB_TITLE']=='PROGRAMMER ANALYST']
sbystate = sbystate[sbystate['YEAR'] == 2015][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).mean()
sbystate2 = h1bdat[h1bdat['JOB_TITLE']=='PROGRAMMER ANALYST']
sbystate2 = sbystate2[sbystate2['YEAR'] ==2015][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).count()
X = []
Y = []
for state in list(sbystate.index):
    (lon, lan) = capitolpos[state]
    X.append(lon)
    Y.append(lan)
fig = plt.figure(figsize=(16,8))
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
m.drawcoastlines()
m.drawcountries(linewidth=3, color='C0')
m.drawstates(linewidth=1.5)
xmap, ymap = m(X, Y)
ax = m.scatter(xmap, ymap, s=1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate.PREVAILING_WAGE.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)
cb = plt.colorbar(ax)
plt.legend()
plt.title('Average "Programmer Analyst" Salary by State in 2015 (x1000 USD/year)')
########
sbystate3 = h1bdat[h1bdat['JOB_TITLE']=='PROGRAMMER ANALYST']
sbystate3 = sbystate3[sbystate3['YEAR'] ==2015][['state','PREVAILING_WAGE']]
sbystate3 = sbystate3.assign(adj_salary=sbystate3.apply(lambda x: x.PREVAILING_WAGE/rpp2015[x['state']], axis=1))
sbystate3 = sbystate3[['state','adj_salary']].groupby(h1bdat['state']).mean()
X = []
Y = []
for state in list(sbystate.index):
    (lon, lan) = capitolpos[state]
    X.append(lon)
    Y.append(lan)
fig = plt.figure(figsize=(16,8))
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
m.drawcoastlines()
m.drawcountries(linewidth=3, color='C0')
m.drawstates(linewidth=1.5)
xmap, ymap = m(X, Y)
ax = m.scatter(xmap, ymap, s=1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate3.adj_salary.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)
cb = plt.colorbar(ax)
plt.legend()
plt.title('Adjusted Average "Programmer Analyst" Salary by State in 2015 (x1000 USD/year)')
plt.show()


# In[ ]:


sbystate = h1bdat[h1bdat['JOB_TITLE']=='SOFTWARE ENGINEER']
sbystate = sbystate[sbystate['YEAR'] == 2015][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).mean()
sbystate2 = h1bdat[h1bdat['JOB_TITLE']=='SOFTWARE ENGINEER']
sbystate2 = sbystate2[sbystate2['YEAR'] ==2015][['state','PREVAILING_WAGE']].groupby(h1bdat['state']).count()
X = []
Y = []
for state in list(sbystate.index):
    (lon, lan) = capitolpos[state]
    X.append(lon)
    Y.append(lan)
fig = plt.figure(figsize=(16,8))
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
m.drawcoastlines()
m.drawcountries(linewidth=3, color='C0')
m.drawstates(linewidth=1.5)
xmap, ymap = m(X, Y)
ax = m.scatter(xmap, ymap, s=1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate.PREVAILING_WAGE.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)
cb = plt.colorbar(ax)
plt.legend()
plt.title('Average "Software Engineer" Salary by State in 2015 (x1000 USD/year)')
########
sbystate3 = h1bdat[h1bdat['JOB_TITLE']=='SOFTWARE ENGINEER']
sbystate3 = sbystate3[sbystate3['YEAR'] ==2015][['state','PREVAILING_WAGE']]
sbystate3 = sbystate3.assign(adj_salary=sbystate3.apply(lambda x: x.PREVAILING_WAGE/rpp2015[x['state']], axis=1))
sbystate3 = sbystate3[['state','adj_salary']].groupby(h1bdat['state']).mean()
X = []
Y = []
for state in list(sbystate.index):
    (lon, lan) = capitolpos[state]
    X.append(lon)
    Y.append(lan)
fig = plt.figure(figsize=(16,8))
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=32,lat_2=45,lon_0=-95)
m.drawcoastlines()
m.drawcountries(linewidth=3, color='C0')
m.drawstates(linewidth=1.5)
xmap, ymap = m(X, Y)
ax = m.scatter(xmap, ymap, s=1.5*sbystate2.state.values/np.min(sbystate2.state.values), c=sbystate3.adj_salary.values/1000, cmap='cool', vmin=sbystate.values.min()/1000, vmax=sbystate.values.max()/1000)
cb = plt.colorbar(ax)
plt.legend()
plt.title('Adjusted Average "Software Engineer" Salary by State in 2015 (x1000 USD/year)')
plt.show()


# The pictures change quite a lot,  especially between eastern states (like New York and New Jersey) with the others!  And now you really know where to go to have maximum money ^^
# 
# But wait! We still miss something obvious here. There are 7 states having zero income tax. Washington and Taxes will be even more attractive places to work when we taking state tax into account.
# I will update later about tax.

# ## Further work
#  I also found that there are some typos in company names in the data. For example, in 2016, you can find 6746 applications from "TECH MAHINDRA (AMERICAS),INC." and 1257 applications from "TECH MAHINDRA (AMERICAS), INC." The number of typos is not negligible so we may need to correct them for a better analysis. I will correct these typos later, maybe by some simple ML techniques on comparing words/phases or by using some simple regular expressions.
