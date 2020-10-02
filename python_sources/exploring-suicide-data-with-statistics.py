#!/usr/bin/env python
# coding: utf-8

# # Exploring Suicide Data
# **<mark> 1. Basic Info & Data Clean</mark>**
#    - Axes Overview
#    - Data Type Optimization
#    - Missing Values
# 
# **<mark> 2. Data Analysis</mark>**
#    - Suicide Data by Countries in 2016
#    - Trend of Suicide Data (Time series)
#    - Suicide Data by Age and Gender
#    - More about Age Groups:
#      * The overall trends and Percentage changes (using stack bar chart)
#      * Using statistics: Inference Based on Two Independent Samples)
#    - More about Gender Groups (Using statistics: Hypothese Test Based on Two independent Samples)
#    - HDI level & Suicide Data
#    - GDP & Suicide Data 
#    - Correlations
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# load the suicide data and the overview of first five rows
suicide = pd.read_csv("../input/master.csv")
suicide.head()


# ### 1. Basic Info & Data Clean
# #### <p> 1.1 Axes Overview

# In[ ]:


display(suicide.index) # row label of the df
display(suicide.columns) # column labels of the df, both are not callable
display(suicide.keys()) # columns for df, same as suicide.column
display(suicide.axes) # return a list representing all the axes of df, not callable


# In[ ]:


len(suicide.columns)


# #### ** Comments on the columns:
# <p> 1. HDI human development index: 
# <blockquote>
#     <p> A summary measure of average achievement in key dimensions of human development: a long and healthy life, being knowledgeable and have a decent standard of living. The HDI is the geometric mean of normalized indices for each of the three dimensions.
# <p><img style="float: bottom;margin:5px 20px 5px 1px" src="http://hdr.undp.org/sites/default/files/hdi.png"></p>

# In[ ]:


suicide.ndim # number of axes


# In[ ]:


display(suicide.size)
suicide.shape


# In[ ]:


suicide.info()


# > - Note1. There is a white space before the gdp_for_year label.
# > - Note2. There are missing values in the column HDI only.

# In[ ]:


# so rename the labels as gdp_for_year to remove the space.
suicide.rename(columns={" gdp_for_year ($) ":"gdp_for_year","gdp_per_capita ($)":"gdp_per_capita"},inplace = True)


# In[ ]:


suicide.head(2)


# #### <p> 1.2 Data Types (Optimize datatype)

# In[ ]:


suicide.dtypes


# > Note: Datatype Conversion:
# - sex / age / generation can be converted into category.
# - gdp_for_year should be converted into int.

# In[ ]:


display(suicide.sex.nunique())
suicide.sex.unique()


# In[ ]:


display(suicide.age.nunique())
suicide.age.unique()


# In[ ]:


display(suicide.generation.nunique())
suicide.generation.unique()


# In[ ]:


suicide[["age","sex","generation"]] = suicide[["age","sex","generation"]].astype("category")
suicide.iloc[:,9] = suicide.iloc[:,9].str.replace(",","").astype("int")


# <mark>Take away: How to convert number strings with commas in pandas DataFrame to float?</mark>
# - method 1 has been shown above, using `.str.replace()`. 
# Reference: https://stackoverflow.com/questions/22137723/convert-number-strings-with-commas-in-pandas-dataframe-to-float
# - method 2: `df.read_csv('foo.tsv', sep='\t', thousands=',')`

# In[ ]:


suicide.info()
# memory usage decreased!


# In[ ]:


suicide.describe()


# #### <p> 1.3 Missing Values

# In[ ]:


suicide.isna().head()


# In[ ]:


suicide.isna().sum()


# > Note: we only have missing values in HDI for year (19456 missing values)

# In[ ]:


display(suicide["HDI for year"].max())
suicide["HDI for year"].min()


# > Deal with missing value: fill the Nan values with 0 which align with the data type and is explicitly different from other values

# In[ ]:


suicide["HDI for year"].fillna(0, inplace=True)


# In[ ]:


suicide.info()


# ### 2. Analysis

# #### 2.1 Suicide Data in 2016 by Countries

# In[ ]:


plt.figure(figsize=(15,5))
suicide.groupby("year")["country"].nunique().plot(kind = "bar")
plt.title('Number of Country by Year', fontsize=12)
plt.xlabel("")


# > - Note: The number of coutry in each year varies so that we cannot directly group by country and sum up the suicide data. The conclusion would not be meaningful with the comparison across year by country.
# > - Instead, we focus on one specific year and choose the most recent year 2016.

# In[ ]:


# only keep the relevant columns
s1 = suicide.iloc[:,:7]
s1.head()


# In[ ]:


# focus first on the most recent data
# suicide data in 2016 (most recent year)
mask_year = s1["year"] == 2016
s1_2016 = s1[mask_year].drop("year",axis = 1).groupby("country").sum()
s1_2016


# > Note: Grenasda only has population data for 2016, so we delete the record

# In[ ]:


s1_2016.drop("Grenada",inplace=True)
s1_2016


# In[ ]:


s1_2016["country"] = s1_2016.index


# > <mark>Take Away: how to avoid the ValueError: Could not interperet input 'Country' after the `group by 
# ` funtion?</mark>
# - The reason for the exception you are getting is that "country" becomes an index of the dataframes after the group_by operation.
# - Method 1: an easy solution is to add the index as a column (same as what we did above).
# - Method 2: use `.reset_index()` after the `group by` function so that the column contry becomes a column again rather than the index.
# - Method 3: use parameter of the `group by` function as_index = False
# 

# In[ ]:


fig,axes = plt.subplots(2,1,figsize = (10,10))
plt.subplots_adjust(wspace = 0,hspace = 0.2) # adjust the distance between subplots

ax1 = plt.subplot(2,1,1) 
# need to sort the data so that the output could be in the descending order
sns.barplot(x="suicides_no",y = "country", data = s1_2016.sort_values("suicides_no",ascending = False), palette="Blues_r")
# add text labels
locs1, labels1 = plt.yticks()
for a,b in zip(s1_2016["suicides_no"].sort_values(ascending=False),locs1):
    plt.text(a+20,b,'%.0f' % a, va="center",fontsize = 10)
# only keep the bottom left spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title("Suicide Numbers by Countries in 2016")
ax1.set_xlabel("")

ax2 = plt.subplot(2,1,2)
sns.barplot(x="suicides/100k pop",y = "country", data = s1_2016.sort_values("suicides/100k pop",ascending = False), palette="Blues_r")
locs2, labels2 = plt.yticks()
for a,b in zip(s1_2016["suicides/100k pop"].sort_values(ascending = False),locs2):
    plt.text(a+3,b,'%.0f' % a, va="center",fontsize = 10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title("Suicide Rates by Countries in 2016")


# #### Conclusion:
# 1. In terms of absolute number of suicides in 2016, Thailand, Romania and Neterlands are top three countries; In terms of suicides/100k population, Lithuania, Hungary and Croatia are the top three.
# 2. Qutar, Armenia and Cyprus have almost both low suicide rate and low absolute suicide number.

# #### 2.2 Suicide Timeseries

# In[ ]:


# count years
display(s1.year.max())
display(s1.year.min())
s1.year.max() - s1.year.min()


# before we draw the line graph to explore the trend, we need to only select a period in which we have enough data to explore.

# In[ ]:


# count the number of data available in each year because we may only use the years with enough data.
country_num_year = suicide.groupby("year",as_index = False).agg({"country":"nunique"})

# draw the line graph would be much clearer that over 70% years have more than 65 data available.
# therefore, we select years from 1995 to 2014 to do the following examination.
x = country_num_year["year"]
y = country_num_year["country"]
quantile30 = country_num_year["country"].quantile(.3)
display(quantile30)

fig = plt.figure(figsize = (10,6))
plt.plot(x,y,"k-o",markersize = 3)
plt.axhline(quantile30,color = 'r',linestyle = '--')
plt.fill_between(x,y,quantile30,where=(y>=quantile30), facecolor='lightcoral',alpha = 0.8)
plt.fill_between(x,y,quantile30,where=(y<=quantile30), facecolor='lightgrey',alpha = 0.8)
plt.title("Number of Countries In Each Year")
plt.grid(True)
# color reference: https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib


# > - Note 1: the red dot line represent 30% quantile, the red shadow are those years with enough data.
# > - Note 2: we could choose the period 1995-2014 because there are roughly 80 countries.

# In[ ]:


s2_time = suicide.groupby("year")["suicides/100k pop"].sum().loc["1995":"2014"].reset_index()
display(s2_time.head(),s2_time.tail())


# In[ ]:


# draw the line graph to see the general trends from 1995-2014
from matplotlib.ticker import MaxNLocator

fig,axes = plt.subplots(1, figsize=(9, 5))
ax3 = sns.lineplot(x="year",y = "suicides/100k pop", data = s2_time, marker = 'o')

ax3.set_xlabel("") # same as plt.xlabel("")
ax3.set_ylabel("") # same as plt.ylabel("")
ax3.xaxis.set_major_locator(MaxNLocator(integer=True)) # force the xtick label to be int for years
ax3.legend(["suicides per 100k population"])
# same as plt.legend(["suicides per 100k population"])
ax3.set_title("Suicides/100K population from 1995-2014")
# same as plt.set_title("Suicides/100K population from 1998-2014")


# #### 2.3 Suicide Data by Age and Gender

# In[ ]:


# to see different trend from age groups and sex groups
mask_end = suicide["year"] <= 2014
mask_start = suicide["year"] >= 1998
s2 = suicide[mask_start & mask_end]
s2_agetime = s2.groupby(["year","age"])["suicides/100k pop"].sum().reset_index()
s2_sextime = s2.groupby(["year","sex"])["suicides/100k pop"].sum().reset_index()

fig,axes = plt.subplots(2,1, figsize=(7, 7),sharex = True)
# set main title for all subplots
fig.suptitle("Different Trends for Age Groups and Gender Groups")
ax4 = plt.subplot(2,1,1)
sns.lineplot(x="year",y = "suicides/100k pop", data = s2_agetime, hue = "age",marker = "o",markersize = 5)
# adjust the legend position outside the graph
ax4.legend(loc = "upper right",bbox_to_anchor=(1.3,1))
ax4.set_xlabel("")

ax5 = plt.subplot(2,1,2)
sns.lineplot(x="year",y = "suicides/100k pop", data = s2_sextime, hue = "sex",marker = "*")
ax5.legend(loc = "upper right",bbox_to_anchor=(1.24,1))


# > <mark>Take Away: How to Specify Legend Position?</mark>
# - using `bbox_to_anchor` & `loc`
# - reference:https://stackoverflow.com/questions/44413020/how-to-specify-legend-position-in-matplotlib-in-graph-coordinates

# #### Conclusion:
# 1. Suicide/100k population decreases during the past 20 years.
# 2. In terms of age, suicide number increase with age.
# 3. In terms of gender, male are far more likely to commit suicide than female. In addition, it seems that there is an obvious down trend for male suicides when compared with female (would be explore in later part).

# #### 2.4 More about Age Group: Statistics

# In[ ]:


suicide.head(3)


# In[ ]:


# show some statistic info on age groups over the years
s2_ageyear = s2.groupby(["year","age"])["suicides/100k pop"].agg(["size","mean","sum","std"])
s2_ageyear.head(10)


# In[ ]:


s2_ageyear.index.names


# > Note: In order to sort the age index, we need to rename the "5-14 years" group. Otherwise, it would appear in the middle of the groups when using ascending order.

# In[ ]:


s2_ageyear.rename(index = {"5-14 years":"05-14 years"},inplace=True)
s2_ageyear.sort_index(ascending = [True,False],inplace = True)
s2_ageyear.head(10)


# we would focus on the mean data and draw the trend using stacked graph.

# In[ ]:


s2_ageyear.unstack()


# In[ ]:


ageyear_avg = s2_ageyear.unstack().loc[:,"mean"]
display(ageyear_avg.head())

y1 = ageyear_avg.iloc[:,0]
y2 = ageyear_avg.iloc[:,1]
y3 = ageyear_avg.iloc[:,2]
y4 = ageyear_avg.iloc[:,3]
y5 = ageyear_avg.iloc[:,4]
y6 = ageyear_avg.iloc[:,5]

fig = plt.figure(figsize = (10,5))
ax = plt.stackplot(ageyear_avg.index,y1,y2,y3,y4,y5,y6)
plt.legend(ageyear_avg.columns)
plt.title("The Average Suidice/100k Numbers in Different Age Groups from 1998-2014")

ageyear_avg.plot(kind = "bar",stacked = True)
plt.title("The Average Suidice/100k Numbers in Different Age Groups from 1998-2014")
plt.legend(loc = "upper right",bbox_to_anchor=(1.35,1))


# > - Note 1: We would focus on the percentage change across years. Therefore, we need to calculate the percentage of each age group in each year.
# > - Note 2: I prefer to use the average (mean data) because sum data are sensitive to numbers of countries  collected in each year.

# In[ ]:


ageyear_avg["total"] = ageyear_avg.apply(lambda row: row[0] + row[1] + row[2] + row[3] + row[4] + row[5], axis=1)

def getPercentage(row):
    for i in range(len(row)):
        row[i] = row[i]/row[6]
    return row

ageyear_avg_pct = ageyear_avg.apply(getPercentage, axis=1)
ageyear_avg_pct.drop(columns = ["total"], inplace = True)
ageyear_avg.drop(columns = ["total"], inplace = True)
display(ageyear_avg_pct.head())


# In[ ]:


fig = plt.figure()
ageyear_avg_pct.plot(kind = "bar",stacked = True)
plt.title("The Percentage of Average Suidice/100k in Different Age Groups from 1998-2014")
plt.legend(loc = "upper right",bbox_to_anchor=(1.35,1))


# > - Note 1: Now, we only want to focus on two specific age groups to determine whether there is a significant difference between these two age groups on the average suicide number.
# - Note 2: these two samples would be independent and evaluation is under the 95% confident level.

# In[ ]:


ageyear_sum = s2_ageyear.unstack().loc[:,"sum"]
ageyear_sum.describe()


# Inference between two independent samples:
# - assume that different age groups are independent from each other.
# - number of year is 17, which is less than 30 = small sample inference
# - we don't know the population variances, and use sample variance to estimate them.
# - the pooled variance is the weighted average of sample variance with degree of freedom as weights.
# - take two age groups, namely 15-24 years and 25-34 years, as an example
# - under 95% confidence interval, the critical value is 2.12.

# In[ ]:


xbar1 = ageyear_sum["15-24 years"].mean()
xbar2 = ageyear_sum["25-34 years"].mean()
var1 = ageyear_sum["15-24 years"].var()
var2 = ageyear_sum["25-34 years"].var()
df1 = df2 = 16
tscore = 2.12

var_sample = (var1 + var2)/2
ME = 2.12 * (var_sample*(1/8))**(0.5)
print("ME = ",ME)
print("xbar2 - xbar1 = ",xbar2-xbar1)
print("the confidence interval should be (",xbar2-xbar1-ME,",",xbar2-xbar1+ME,")")


# #### Conclusion:
# 0 is not included in the confidence interval, therefore, we have 95% confidence to say that the average suicide number in 15-24 years is significantly lower than that in 25-34 years.

# #### 2.5 More about Gender Group: Statistics

# In the previous part, it seems that there is a down trend for male suicides. We are going to evaluate that.

# In[ ]:


s2_genderyear = s2.groupby(["year","sex"])["suicides/100k pop"].agg(["size","mean","sum","std"])
display(s2_genderyear.head(2))
s2_genderyear.tail(2)


# > - Note 1: In 1998, the average suicide rate for male is 23.77, the number of sample is 474 and the std of the sample is 26.74.
# > - Note 2: In 2014, the average suicide rate for male is 17.56, the number of sample is 468 and the std of the sample is 18.72.
# > - Note 3: We should not make any conclusion before we know the distribution and do the Paired test.

# Hypothese Test between two independent samples:
# - use the difference between the two sample to test.
# - assume that H0: mean 1 = mean 2; Ha: the mean suicide rate in 2014 is lower than that in 1998.
# - n1 = 474 and n2 = 468 and we can use CLT.

# In[ ]:


meandiff = s2_genderyear.loc[(1998,"male"),"mean"]-s2_genderyear.loc[(2014,"male"),"mean"]
vardiff = s2_genderyear.loc[(1998,"male"),"std"]**2/474 + s2_genderyear.loc[(2014,"male"),"std"]**2/468
stddiff = vardiff ** 0.5

Zvalue = meandiff/stddiff
Zvalue > 1.645


# #### Conclusion:
# the test statistics is greater than z critical value, meaning that we have 95% confidence to say that the average suicide rate in 2014 is significantly lower than that in 1998.

# #### 2.6 HDI & Suicide Data

# In[ ]:


suicide.head()


# In[ ]:


# only keep those valid records (HDI > 0)
mask_hdi = suicide["HDI for year"] > 0
s3_hdi = suicide[mask_hdi].groupby(["HDI for year","sex","age"])["suicides/100k pop"].sum().reset_index()
s3_hdi.head()


# In[ ]:


# see the distribute of HDI (a general picture)
fig,axes = plt.subplots()
display(s3_hdi["HDI for year"].max(),s3_hdi["HDI for year"].min())
bins = [0.45,0.55,0.65,0.75,0.85,0.95]
axes = plt.hist(s3_hdi["HDI for year"],bins = bins,rwidth = 0.9)


# > according to http://hdr.undp.org/en/composite/HDI
# - HDI < 0.55 is defined as low human development
# - HDI < 0.7 is defined as medium human development
# - HDI < 0.8 is defined as high human development
# - HDI >=0.8 is defined as very high human development

# In[ ]:


def f(row):
    if row['HDI for year'] < 0.55:
        text = "low"
    elif row['HDI for year'] < 0.7:
        text = "medium"
    elif row['HDI for year'] < 0.8:
        text = "high"
    else:
        text = "very high"
    return text

suicide["HDI level"] = suicide.apply(f,axis = 1)


# In[ ]:


# add a new column referencing HDI level to the data
s3_hdiLevel = suicide[mask_hdi].groupby(["HDI for year","sex","age","HDI level"])["suicides/100k pop"].sum().reset_index()
s3_hdiLevel.head()


# In[ ]:


fig,axes = plt.subplots(3,1,figsize=(8,10),sharex = True)
ax1 = plt.subplot(3,1,1)
sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel)
ax1.set_xlabel("")

ax2 = plt.subplot(3,1,2)
sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel,hue = "sex")
ax2.set_xlabel("")

ax3 = plt.subplot(3,1,3)
sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel,hue = "age")


# In[ ]:


s3_hdiLevel1 = suicide[mask_hdi].groupby("HDI level")["suicides/100k pop"].sum().reset_index()
s3_hdiLevel1.head()


# In[ ]:


s3_hdiLevel2 = suicide[mask_hdi].groupby(["HDI level","sex"])["suicides/100k pop"].sum().reset_index()
s3_hdiLevel2


# In[ ]:


s3_hdiLevel3 = suicide[mask_hdi].groupby(["HDI level","age"])["suicides/100k pop"].sum().reset_index()
s3_hdiLevel3


# In[ ]:


fig,axes = plt.subplots(3,1,figsize=(8,10),sharex = True)
ax1 = plt.subplot(3,1,1)
sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel1)
ax1.set_xlabel("")

ax2 = plt.subplot(3,1,2)
sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel2,hue = "sex")
ax2.set_xlabel("")

ax3 = plt.subplot(3,1,3)
sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel3,hue = "age")


# #### Conclusion:
# 1. Suicide/100k population increases with HDI level in all ages and both genders.
# 2. On average, countries with very high HDI have similar suicides/100k population numbers as countries with high HDI, both of which have far more suicides numbers than countries in low and medium HDI levels.

# #### 2.7 GDP & Suicides

# In[ ]:


suicide.head()


# In[ ]:


s4_gdp = suicide.groupby(["gdp_per_capita"])["suicides/100k pop"].sum().reset_index()
s4_gdp.head()


# In[ ]:


# general picture of gdp_per_capita and suicides numbers
sns.scatterplot("gdp_per_capita","suicides/100k pop",data = s4_gdp,linewidth = 0)


# In[ ]:


# adding more groups criteria = sex and age
s4_gdp1 = suicide.groupby(["gdp_per_capita","sex"])["suicides/100k pop"].sum().reset_index()
s4_gdp1.head()


# In[ ]:


s4_gdp2 = suicide.groupby(["gdp_per_capita","age"])["suicides/100k pop"].sum().reset_index()
s4_gdp2.head()


# In[ ]:


fig,axes = plt.subplots(1,2,figsize=(12,6),sharey = True)
ax1 = plt.subplot(1,2,1)
sns.scatterplot("gdp_per_capita","suicides/100k pop",data = s4_gdp1,hue = "sex", style = "sex",linewidth = 0)

ax2 = plt.subplot(1,2,2)
sns.scatterplot("gdp_per_capita","suicides/100k pop",data = s4_gdp2,hue = "age", linewidth = 0)
ax2.set_ylabel("")


# we should put gdp_per_capita data into groups/bins so that we could evaluate the distribution and trend

# In[ ]:


import numpy as np
suicide['gdp/10k_per_capita'] = np.floor(suicide["gdp_per_capita"].div(10000)).astype(int)
#s4_gdp = suicide.groupby(["gdp_per_capita"])["suicides/100k pop"].sum().reset_index()
suicide["gdp/10k_per_capita"].value_counts()


# focus on a specific year = year 2016
# - GDP in different years may be affected by other factors and is not representative
# - therefore, we should focus on specific year when evaluating the GDP factor
# 

# In[ ]:


# general data about gdp group
s4_gdp3 = suicide[mask_year].groupby("gdp/10k_per_capita",as_index = False)["suicides/100k pop"].sum()
# multi-index = gdp group + sex group
s4_gdp4 = suicide[mask_year].groupby(["gdp/10k_per_capita","sex"])["suicides/100k pop"].sum()

fig = plt.subplots(2,1,figsize = (8,8))
plt.subplot(2,1,1)
sns.barplot("gdp/10k_per_capita","suicides/100k pop",data = s4_gdp3)
plt.subplot(2,1,2)
sns.barplot("gdp/10k_per_capita","suicides/100k pop",data = s4_gdp4.reset_index(),hue = "sex")


# <mark>Take Away: Another way to calculate the percentage within the group:</mark>
# 1. reference: https://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby
# 2. s4_gdp3 is a Series with a Multi Index -- so the main body of the table is only one column of suicide rates whose value is numeric. 
# 3. After you do the groupby, each x is the single data within that group. 

# In[ ]:


s4_gdp4.head()


# In[ ]:


s4_gdp4.index


# In[ ]:


sex_pct = s4_gdp4.groupby(level = 0).apply(lambda x: x / float(x.sum()))
sex_pct


# In[ ]:


sex_pct.unstack().head()


# In[ ]:


ax1 = sns.lineplot("gdp/10k_per_capita","suicides/100k pop",data = sex_pct.reset_index(),hue = "sex", style = "sex")
from matplotlib.ticker import FuncFormatter
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 


# In[ ]:


sex_pct.unstack().plot(kind = "bar", stacked = True)
plt.legend(loc = "upper right",bbox_to_anchor=(1.25,1))


# <mark>Take Away: how to adjust percentage yticklabels?</mark>
# 1. reference: https://stackoverflow.com/questions/31357611/format-y-axis-as-percent
# 2. `vals = ax1.get_yticks()`
# 3. `ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])`

# In[ ]:


# similar to the sex group, we then evaluate the age group
s4_gdp5 = suicide[mask_year].groupby(["gdp/10k_per_capita","age"])["suicides/100k pop"].sum()
age_pct = s4_gdp5.groupby(level = 0).apply(lambda x: x / float(x.sum()))
age_pct


# In[ ]:


ax2 = sns.lineplot("gdp/10k_per_capita","suicides/100k pop",data = age_pct.reset_index(),hue = "age", color = "age", marker = "o")

#the two methods below are same in adjusting percentage yticklabels
#reference: https://stackoverflow.com/questions/31357611/format-y-axis-as-percent
#vals = ax1.get_yticks()
#ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
from matplotlib.ticker import FuncFormatter
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
ax2.legend(loc = "upper right",bbox_to_anchor=(1.35,1))


# In[ ]:


age_pct.unstack().plot(kind = "bar", stacked = True)
plt.legend(loc = "upper right",bbox_to_anchor=(1.35,1))


# #### Conclusion:
# 1. Suicide/100k population decreases with GDP_per_capita.
# 2. In the most recent year(2016), it is also seen a pattern that suicide numbers decreases with GDP per capita. Moreover, when we evaluate the percentage changes, there is a distinct decrease in male suicides from 20k gdp per capita to 50k gdp per capita.
# 3. In the most recent year(2016), from the angle of age groups, the pattern varies. Suicide number drops quickly for the 75+ years group but increases markedly for younger groups (25-34 years group and 15-24 years group).

# #### 2.8 Correlations

# In[ ]:


# try to examine the correlations among different factors and suicides/100k population.
# using heatmap
# only keep relevant factors = suicides/100k pop & population & HDI for year & gdp_per_capita
suicide_corr = suicide[["suicides/100k pop","suicides_no","population", "HDI for year", "gdp_per_capita"]]
sns.heatmap(suicide_corr.corr(),annot = True)


# #### Conclusion:
# Not much correlation between factors and suicide numbers except for population.
