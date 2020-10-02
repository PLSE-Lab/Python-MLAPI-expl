#!/usr/bin/env python
# coding: utf-8

# <font color='blue'><b>What are we looking for in this kernel?  </b></font> Main focus of this kernel is to identify the scope of conducting Experimental design and A/B testing. Following are the contents.<br/>
# 1) Choose a few countries.<br/>
# 2) Study distributions of continuous variables.<br/>
# 3) Univariate and Bivariate analysis.<br/>
# 4) Feature Visualization.<br/>
# 5) Study Immunization coverage, Lifeexpectancy and AdultMortality across each country through visualization using different types of plots.<br/>
# 6) Identify 10 countries with lowest percentange of immunization coverage for "HepatitisB"<br/>
# 7) Analysis that lead to conduct experimental testing.<br/>
# 8) A/B Testing: Analysis, Roll-out plan, Key metrics.<br/>
# 

# In[ ]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing dataset in csv to pandas data frame.
data = pd.read_csv('../input/Life Expectancy Data.csv', delimiter=',')
data.dataframeName = 'Life Expectancy Data.csv'


# <b>Dataset Description:</b><br/>
# Find details about this dataset here https://www.kaggle.com/kumarajarshi/life-expectancy-who

# <b>Data Cleaning and EDA</b>

# In[ ]:


# Removing spaces from the column names.
data.columns = data.columns.str.replace(' ','')


# In[ ]:


df = data.copy()
dd = data.copy()
all = data.copy()
data.head()


# In[ ]:


# Pick required countries.
data = data.loc[data['Country'].isin(['Tonga','Monaco','Oman','Qatar','Finland','Nepal','India','Mexico','Poland'])]


# <b>Univariate distributions.

# In[ ]:


# Drop rows with Null values.
data = data.dropna(axis=0)
data.head()


# In[ ]:


# Distributions of Immunization coverage for Polio, HepatitisB and Diphtheria
plt.figure(figsize=(8,14))
sns.set(style="ticks")

plt.subplot(3,1,1)
p1 = sns.distplot(data['Polio'],color='r')
p1.set_xlabel("Ploio",fontsize=14)
p1.set_ylabel("% Immunization Coverage",fontsize=14)
p1.set_title("Immunization for Polio, HepatitisB and Diphtheria",fontsize=14)

plt.subplot(3,1,2)
p2 = sns.distplot(data['HepatitisB'],color='b')
p2.set_xlabel("HepatitisB",fontsize=14)
p2.set_ylabel("% Immunization Coverage",fontsize=14)

plt.subplot(3,1,3)
p3 = sns.distplot(data['Diphtheria'],color='orange')
p3.set_xlabel("Diphtheria",fontsize=14)
p3.set_ylabel("% Immunization Coverage",fontsize=14)

plt.show()


# In[ ]:


# Drop rows with Null values from the main data frame.
all.isnull().sum()
all = all.dropna(axis=0)


# In[ ]:


# Verifying null values
all.isnull().sum()


# <b>Feature Visualization

# In[ ]:


columns_to_plot = all.columns[3:6].sort_values()
for column in columns_to_plot: 
    sns.kdeplot(all[column], shade=True)
    plt.title(column)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.show()
    print(all[column].describe())


# <b>Bivariate relationships

# In[ ]:


p1 = sns.lmplot(x='Lifeexpectancy',y='AdultMortality',data=data)
plt.title('Bivariate relationship')
plt.show()


# In[ ]:


p2 = sns.lmplot(x='Lifeexpectancy',y='infantdeaths',data=data)
plt.title('Bivariate relationship')
plt.show()


# <b>Immunization coverage, Lifeexpectancy and AdultMortality across each country.

# In[ ]:


# Immunization coverage, Lifeexpectancy and AdultMortality across each country.
plt.figure(figsize=(18,18))
sns.set(style='whitegrid')

plt.subplot(3,2,1)
plot1 = sns.pointplot(data=data,x=data.Country,y=data['Lifeexpectancy'])
plot1.set_title("Lifeexpectancy across each country",fontsize=13)

plt.subplot(3,2,2)
plot2 = sns.barplot(data=data,x=data.Country,y=data['AdultMortality'],hue='Status')
plot2.set_title("AdultMortality across each country",fontsize=13)

plt.subplot(3,2,3)
plot3 = sns.boxplot(data=data,x=data.Country,y=data['HepatitisB'],hue='Status')
plot3.set_title("HepatitisB across each country",fontsize=13)

plt.subplot(3,2,4)
plot4 = sns.violinplot(data=data,x=data.Country,y=data['Polio'],hue='Status')
plot4.set_title("Polio across each country",fontsize=13)

plt.subplot(3,2,5)
plot5 = sns.stripplot(data=data,y=data.Country,x=data['Diphtheria'],hue='Status',dodge=True,jitter=True,alpha=1,zorder=1)
sns.pointplot(x=data['Diphtheria'],y=data.Country,hue='Status',data=data,dodge=.532,join=False,palette='dark',scale=.75,ci=None)
plot5.set_title("Diphtheria across each country",fontsize=13)

plt.show()


# <b>Observation:</b><br/>
# - Mexico has highest Lifeexpectancy.<br/>
# - Nepal has highest AdultMortality.
# 

# In[ ]:


final = df[df['Country'].isin(['Norway','Italy','Singapore','Japan','Poland','Palau','Mexico','France'])]
final.dropna(inplace=True)
final.isnull().sum()


# In[ ]:


# Using Subplots
dims = (15,4)
fig, axs = plt.subplots(ncols=4,figsize=dims)
sns.distplot(final.Alcohol,ax=axs[0],color='r')
sns.distplot(final.AdultMortality,ax=axs[1],color='g')
sns.distplot(final.GDP,ax=axs[2])
sns.distplot(final.Population,ax=axs[3],color='orange')
plt.show()


# In[ ]:


dd.head()


# <b>10 countries with lowest percentange of immunization coverage for "HepatitisB"</b>

# In[ ]:


# Find 10 countries with lowest percentange of immunization coverage for "HepatitisB"
dd = dd.groupby('Country').mean().nsmallest(10,'HepatitisB').reset_index()
dd


# In[ ]:


# Visualizing 10 countries with lowest percentange of immunization coverage for "HepatitisB" using barplots.
plt.figure(figsize=(14,5))
sns.set(style='dark')

plt.subplot(1,2,1)
plot1 = sns.barplot(data=dd,x=dd.Country,y=dd['HepatitisB'])
plot1.set_title("10 countries with low immunization for HepatitisB",fontsize=13)
plot1.set_xticklabels(plot1.get_xticklabels(), rotation=90, ha="right", fontsize=12)

plt.subplot(1,2,2)
plot2 = sns.barplot(data=dd,x=dd.Country,y=dd['infantdeaths'])
plot2.set_title("infantdeaths across each country",fontsize=13)
plot2.set_xticklabels(plot2.get_xticklabels(), rotation=90, ha="right", fontsize=12)


#ax = plt.scatter(x='HepatitisB',y='infantdeaths',data=data)
plt.show()


# ![](http://)<font color="red"><b>Analysis:</b></font> From the above data and visualization, 'India' is the country with high infantdeaths and low immunization coverage for HepatitisB. Though 'MarshallIslands' is the country with least HepatitisB immunization coverage, the infantdeaths count for it is low(0 in this case). Among the top 10 countries with low immunization coverage to HepatitisB, only India has high infant deathcounts. Hence I've choosen India to conduct the experiment.
# 

# <b><font color="red">Experimental Design: A/B Testing</font></b>

# <font color="blue"><b>Hypothesis:</b></font> Increasing HepatitisB immunization coverage among infants would decrease infantdeaths.
# 
# <font color="blue"><b>Sample:</b></font>  1000 infants will be randomly selected from the total population with low income. Low income families consent to anonymous data collection and adherence to participation guidelines in exchange for a nominal financial incentive. 
# 
# <font color="blue"><b>Treatment:</b></font> The sample will be offered immunization coverage for HepatitisB and are put under observation for 1 year.

# <font color="red"><b>Rollout & Evaluation Plan</b></font><br/>
# 
# <font color="blue"><b>Impact Window:</b></font><br/>
# The first year post-partum.
# 
# <font color="blue"><b>Data Collection Approach:</b></font><br/>
# Mothers would report either to an assigned site - a nearby public healthcare facility or a preferred alternate (a nearby private non-profit clinic for example) - for all of their infants' incidental and routine healthcare needs throughout the experiment. These sites will already have electronic health record (EHR) systems implemented, to streamline data collection and will abide by Data Use agreements.
# 
# <font color="blue"><b>Metrics:</b></font><br/>
# Number of infant deaths per 1000 population.
# 
# <font color="blue"><b>Success Criteria:</b></font><br/>
# If exclusively immunized infant deaths is atleast 10% less than that of infants without immunization coverage for HepatitisB, the null hypothesis that HepatitisB vaccination has no impact on infant deaths can be rejected.
# 

# In[ ]:




