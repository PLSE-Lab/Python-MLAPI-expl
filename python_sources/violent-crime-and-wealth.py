#!/usr/bin/env python
# coding: utf-8

# 
# # Introduction
# 
# This collection of python scripts is an excercise in accessing APIs, cleaning data, exploratory data analysis, and hypothesis testing. 
# 
# Wealth and wealth inequality are typical predictors of crime at country-level and city-level. Do these factors specifically influence the prevalence of violent property crime compared to nonviolent property crime? Also, is this correlation present at the local level? 
# 
# The cognative stress that leads to acts of violence or the threat of violence is, presumably, much different than than the stressors that motivate acts of theft.  In this notebook, I test whether the distributions of violent property crime (robbery) and nonviolent property crimes (burglary, auto theft, shoplifting, counterfeiting, etc) are significantly different with respect to the median income and Gini coefficient (https://en.wikipedia.org/wiki/Gini_coefficient) in the census tracts where the crimes took place. 
#  
# In this notebook I load the Los Angeles crime data plus data from two APIs. The FCC geo API allows me to identify the US census tracts where the crimes took place using the lattitude and longitude in the 'Location' column of the crime data. The US census API allows me to access the median household income and Gini coefficient of wealth inequality for the census tracts associated with each crime.
# 
# # Loading the Packages
# 
# I will use some standard data science packages, plus json and requests for handling the API interactions. I use ECDF for comparing the final distributions of the subject variables. I also use a package which performs the Mann-Whitney statistic calculations. 

# In[ ]:


import subprocess
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import requests
import glob, os
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import mannwhitneyu


# # Loading the Data
# 
# I load the main crime data in chunks because the FCC geo API takes some time and I want to moniter the progress as the crimes are matched with tracts. 
# 
# I also load the US census data here because we will want it available as we loop through chunks of crime data. Details on the url can be found in the US Census API guide: https://www.census.gov/content/dam/Census/data/developers/api-user-guide/api-guide.pdf
# 
# Since the crime data includes 2010 - 2017, I chose to use the ACS 5 year community survey data taking place through 2017. I chose the population, median household income, and Gini variables for all census tracts in LA County, CA. The API provides a json which I read in as tractDF. 
# 
# The next lines contain some maintenence on the tract dataframe. I use the second line as the column names because the first is just digits, I remove the line of digits, I rename the variable codes to the true names of the variables, I set the 'Tract' variable as the index to help with merging the data later, I drop some extra columns with redundant labels (these were helpful when inspecting to make sure I got the right data), and I change my variables of interest to numeric (I ended up not using population). 
# 

# In[ ]:


# Retreive Crime in Los Angeles data as chunked DF
process_chunksize = 100000
mainCSV = '../input/Crime_Data_2010_2017.csv'
mainDF = pd.read_csv(mainCSV, chunksize=process_chunksize)

# Retreive and clean tract census data
census_api_url='https://api.census.gov/data/2017/acs/acs5?key=65c608668f59dc2272076d077091ba8cd98a6286&get=NAME,B00001_001E,B19019_001E,B19083_001E&for=TRACT:*&in=state:06+COUNTY:037'
responseCensus = requests.get(census_api_url, headers={'Content-Type': 'application/json'})
textCensus=json.loads(responseCensus.content)
tractDF = pd.DataFrame(textCensus)
print('Shape of census tract file: ' + str(tractDF.shape))

tractDF.rename(columns = tractDF.iloc[0], inplace=True)
tractDF = tractDF.iloc[1:]
tractDF.rename(index=str,columns={'B00001_001E':'Pop','B19019_001E':'Income','B19083_001E':'Gini','tract':'Tract'}, inplace=True)
tractDF.set_index('Tract', inplace=True)
tractDF.drop(['NAME', 'state', 'county'],inplace=True, axis=1)
tractDF=tractDF[['Pop','Income','Gini']].apply(pd.to_numeric, errors='coerce')
print(tractDF.info())
print(tractDF.head(8))


# There are some bunk values in the census data. I'll filter these later. 
# 
# # Function to Retreive Census Tract using Crime Location
# 
# I now create a function to retreive the census tract for the lattitude and longitude of the crime. It accepts the crime location as a tuple and creates the url for retreiving the json containing the tract ID number. The response json is read and the correct entry is selected.
# 

# In[ ]:


#Define tract finding function
def getTract(lonlat):
    tract = -999
    api_url='https://geo.fcc.gov/api/census/area?lat='+str(lonlat[0])+'&lon='+str(lonlat[1])+'&format=json'
    headers = {'Content-Type': 'application/json'}
    response = requests.get(api_url, headers=headers)
    try:
        tract=json.loads(response.content.decode('utf-8'))['results'][0]['block_fips'][5:11]
    except:
        print('No Tract found from FCC geo API')
    return tract


# # Looping Through the Crime Dataframe
# 
# Now I loop through chunks of the crime dataframe while cleaning them and merging with my census data. I use some counters so I can see which lines I'm on. 
# 
# First, I drop NA from the crime code descriptions. I immediately select the entries that include 'ROBBERY|THEFT|STOLEN|BURGLARY|COUNTERFEIT'. These are the violent and nonviolent property crimes that I am interested in. I reject dates before 2013 so that the data and the census survey are compatible. 
# 
# Then I clean up the LA crime data. I change the values to numeric or datetime where appropriate, convert the 'Location' to a tuple, and drop some numbers I know I won't use. I keep many interesting variables that I dont' use this time around, but I would certainly like to look into them later. I expect violent versus nonviolent is highly correlated with time of day.
# 
# I also create a 'Violent' bool so that I can easily access distributions of each type of crime. Finally, I use my getTract function to find the tract for each crime, and I merge my tables on Tract with the crime data on the left. The chunk is appended to my main dataframe, df. 

# In[ ]:


df = pd.DataFrame()
itld=0
itlu=1
for mainDF_chunk in mainDF:
    nLinesMainCSVd=itld*process_chunksize
    nLinesMainCSVu=itlu*process_chunksize
    print('New group from main DF, lines '+str(nLinesMainCSVd) + ' to ' + str(nLinesMainCSVu) )
    
    # Pre Filtering to save memory
    mainDF_chunk.dropna(subset=['Crime Code Description'],inplace=True)
    mainDF_chunk= mainDF_chunk[mainDF_chunk['Crime Code Description'].str.contains('ROBBERY|THEFT|STOLEN|BURGLARY|COUNTERFEIT')]
    mainDF_chunk= mainDF_chunk[mainDF_chunk['Date Occurred'] > '12/31/2012']

    # Cleaning main
    mainDF_chunk.rename(columns=lambda x: x.replace(' ',''), inplace=True)
    mainDF_chunk.dropna(subset=['Location'],inplace=True)
    mainDF_chunk['Location']=mainDF_chunk['Location'].map(lambda x: eval(str(x)))
    mainDF_chunk[['DateReported','DateOccurred','TimeOccurred']].apply(pd.to_datetime,errors='coerce')
    mainDF_chunk[['DRNumber','AreaID','ReportingDistrict','CrimeCode','VictimAge','PremiseCode','WeaponUsedCode','CrimeCode1']].apply(pd.to_numeric)
    mainDF_chunk.drop(['CrimeCode2','CrimeCode3','CrimeCode4'],inplace=True, axis=1)
    print('Dimension of main DF group:' + str(mainDF_chunk.shape))
    
    # Add Tract from FCC geo AND violent bool
    mainDF_chunk['Violent']=mainDF_chunk['CrimeCodeDescription'].str.contains('ROBBERY')
    mainDF_chunk['Tract']=mainDF_chunk.Location.map(lambda x: getTract(x))

    df_group = pd.merge(mainDF_chunk,tractDF, how='left', on='Tract', sort=True)
    #print(df_group.head(5))
    
    df = df.append(df_group,sort=True)
    
    #print(df.head(3))
    
    itlu+=1
    itld+=1


# Now I remove bad tract info by selecting positive Gini and Income. This final dataframe can be used for many other studies with the numeric variables present. I proceed with my intended study by creating arrays of Gini and Income data for violent and nonviolent crimes. 

# In[ ]:


df=df[df['Income']>0.0]
df=df[df['Gini']>0.0]
df.reset_index(inplace=True)

print(df.shape)
print(df.info())

viSetGini = np.array(df[df['Violent']==1]['Gini'])
nonviSetGini = np.array(df[df['Violent']!=1]['Gini'])

viSetIncome = np.array(df[df['Violent']==1]['Income'])
nonviSetIncome = np.array(df[df['Violent']!=1]['Income'])


# # Visualization
# 
# Time for some plots. I start with simple distributions of the two variables from the two selections. A difference in features is apparent for the income distribution, and possibly for the Gini distribution, but both are fairly subtle. 

# In[ ]:


#Plots
fGini, fGiniPlots = plt.subplots(3,sharex=True)
fGiniPlots[0].hist([viSetGini,nonviSetGini], bins=20, range=[0.2,0.8], stacked=True, color=['r','b'])
fGiniPlots[0].set_ylabel('Total Incidents')
fGiniPlots[0].legend(('Robbery', 'Nonviolent Theft'),loc='best')

fGiniPlots[1].hist(viSetGini, bins=20, range=[0.2,0.8], color='r')
fGiniPlots[1].set_ylabel('Robbery')
fGiniPlots[2].hist(nonviSetGini, bins=20, range=[0.2,0.8], color='b')
fGiniPlots[2].set_xlabel('Gini Coefficient at Crime Location Census Tract')
fGiniPlots[2].set_ylabel('Non-violent Theft')

fIncome, fIncomePlots = plt.subplots(3,sharex=True)
fIncomePlots[0].hist([viSetIncome,nonviSetIncome], bins=15, range=[0.,200000.], stacked=True, color=['r','b'])
fIncomePlots[0].set_ylabel('Total Incidents')
fIncomePlots[1].hist(viSetIncome, bins=15, range=[0.,200000.], color='r')
fIncomePlots[1].set_ylabel('Robbery')

fIncomePlots[2].hist(nonviSetIncome, bins=15, range=[0.,200000.], color='b')
fIncomePlots[2].set_xlabel('Median Household Income (12 mo) at Crime Location Census Tract')
fIncomePlots[2].set_ylabel('Non-violent Theft')


# Income has less of a tail in the positive direction for violent crime and Gini has more of a tail. This is compatible with what we might expect, but its not clear, and the statistics for the violent sample are much weaker compared to the nonviolent sample. Some cumulative distributions might make this clearer. 

# In[ ]:


#ecdf comparison
ecdf, ecdfPlots = plt.subplots(2)
ecdf.subplots_adjust(hspace=0.45)

# Gini
ecdfViGini = ECDF(viSetGini)
ecdfNonviGini = ECDF(nonviSetGini)
ecdfPlots[0].plot(ecdfViGini.x,ecdfViGini.y,marker='.', linestyle='none',
                 color='red', alpha=0.5)
ecdfPlots[0].plot(ecdfNonviGini.x,ecdfNonviGini.y,marker='.', linestyle='none',
                 color='blue', alpha=0.5)
ecdfPlots[0].set_xlabel('Tract Gini Coefficient')
ecdfPlots[0].legend(('Robbery', 'Nonviolent Theft'),
           loc='best')

# Income
ecdfViIncome = ECDF(viSetIncome)
ecdfNonviIncome = ECDF(nonviSetIncome)
ecdfPlots[1].plot(ecdfViIncome.x,ecdfViIncome.y,marker='.', linestyle='none',
                 color='red', alpha=0.5)
ecdfPlots[1].plot(ecdfNonviIncome.x,ecdfNonviIncome.y,marker='.', linestyle='none',
                 color='blue', alpha=0.5)
ecdfPlots[1].set_xlabel('Tract Median Income')
ecdfPlots[1].legend(('Robbery', 'Nonviolent Theft'), loc='best')


# The sharper peak of violent crime at lower income tracts is now very clear. We see a little bit more detail in the Gini distribution as well, but is this statistically significant? 
# 
# # Testing
# 
# The violent sample is significantly smaller than the nonviolent, so we should test whether these distributions could have come from the same identically distributed set. Perhaps the tail in the Gini shape is an upward fluctuation. 
# 
# For examining differences between two non-normal distributions, the Mann-Whitney U statistic is appropriate. This non-parametric method uses the ranking of each value in the combined set to calculate a test statistic U, which is proportional to the sum of rankings for the sample with the lowest rankings. U is small (with respect to the product of sample sizes) for very separate distributions and large for identical distributions. A more relevant value is the corresponding p-value for U, which takes into account the sample sizes and returns p-value significance assuming a gaussian distribution for U if the samples are combined. 
# 
# A good intro to Mann-Whitney: http://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_nonparametric/BS704_Nonparametric4.html
# 
# The Mann-Whitney method tests the hypothesis of one stochastic variable being less/greater than another, so this must be specified in the mannwhitneyu function using both the order of the arrays and the 'alternative' variable. Using the function on two arrays without specifying 'less' or 'greater' is depracated.
# 
# The difference in distributions due to income is already pretty clear from our plots. I still include income in the testing section so that we can see an example of Mann-Whitney U for two very different distributions and check that we are using the tool correctly and seeing what we expect.

# In[ ]:


def getMWU(values1,values2):
    return mannwhitneyu(values1,values2,alternative='less').statistic
def getMWP(values1,values2):
    return mannwhitneyu(values1,values2,alternative='less').pvalue

# Mann-Whitney Test (non-parametric test)
nbnGini = len(viSetGini)*len(nonviSetGini)
nbnIncome = len(viSetIncome)*len(nonviSetIncome)
MWGiniU = getMWU(nonviSetGini,viSetGini)
MWGiniP = getMWP(nonviSetGini,viSetGini)
MWIncomeU = getMWU(viSetIncome,nonviSetIncome)
MWIncomeP = getMWP(viSetIncome,nonviSetIncome)

print('length violent: ' + str(len(viSetGini)))
print('length nonviolent: ' + str(len(nonviSetGini)))
print("Mann-Whitney test for Gini, statistic U: " + str(MWGiniU) + ",  p-value: " + str(MWGiniP) + ",  n1 x n2: " + str(nbnGini))
print("Mann-Whitney test for Income, statistic U: " + str(MWIncomeU) + ",  p-value: " + str(MWIncomeP) + ",  n1 x n2: " + str(nbnIncome))


# The miniscule p-value for income is what we expect, given that we could see a large difference just from inspecting the cumulative plots.
# 
# For Gini, the exclusion of the null-hypothesis is better than I expected, though we do not consider a value of ~.13 that significant. For 1% or 5% significance we would need more data, especially for violent crime. 
# 
# I also visualize this test by performing 5,000 trials where the samples are combined and permutated and the U values are calculated under the null-hypothesis. This is ostensibly a Monte-Carlo simulation of the Gaussian distribution that the mannwhitneyu function uses to calculate the p-value, so this is also a nice check of our significance. 

# In[ ]:


# Permutation test with test statistic
def PermutationTestStat(values1,values2,nTrials,testStatFunc):
    trials = np.empty(nTrials)
    concatValues = np.concatenate((values1,values2))
    for i in range(0,nTrials):
        permutedValues = np.random.permutation(concatValues)
        sampledValues1 = permutedValues[:len(values1)]
        sampledValues2 = permutedValues[len(values1):]
        trials[i] = testStatFunc(sampledValues1,sampledValues2)
    return trials

#p value mean diff comparison
testStat, testStatPlots = plt.subplots(2)
testStat.subplots_adjust(hspace=0.3)

trialsGini = PermutationTestStat(nonviSetGini,viSetGini,5000,getMWU)
measuredTestStatGini = getMWU(nonviSetGini,viSetGini)
pValGini = (trialsGini < measuredTestStatGini).sum()/len(trialsGini)
testStatPlots[0].hist(trialsGini,bins=30)
testStatPlots[0].axvline(measuredTestStatGini, color='red', linewidth=1)
testStatPlots[0].text(.07,.7,'p-value: '+ str(pValGini), {'color': 'r', 'fontsize': 12}, transform=testStatPlots[0].transAxes)

trialsIncome = PermutationTestStat(viSetIncome,nonviSetIncome,5000,getMWU)
measuredTestStatIncome = getMWU(viSetIncome,nonviSetIncome)
pValIncome = (trialsIncome < measuredTestStatIncome).sum()/len(trialsIncome)
testStatPlots[1].hist(trialsIncome,bins=30)
testStatPlots[1].axvline(measuredTestStatIncome, color='red', linewidth=1)
testStatPlots[1].text(.07,.7,'p-value: '+ str(pValIncome), {'color': 'r', 'fontsize': 12}, transform=testStatPlots[1].transAxes)
testStatPlots[1].set_xlabel('Mann-Whitney U Distribution')
plt.show()


# # Assumptions and Future Tests
# 
# This study used a few assumptions. Tract populations are of order ~300 and they vary quite a lot, so the Gini calculation is subject to fluctuation. Also, this study is focused on the effect of the wealth metrics at the scene of the crime specifically, so we are learning about wealth's effect on the surrounding area outright. Clearly we cannot determine the criminal's home tract, though including neighboring tract statistics would be an interesting addition. Finally, a single tract can change significantly over the 5 years that the ACS survey took place. This study assumes limited change so that we can use long-term comprehensive data that matches the period over which the crime data was gathered. 
# 
# I am very interested in feedback regarding my general coding techniques, python usage, plot quality, and the appropriateness of the Mann-Whitney test in this case. 
