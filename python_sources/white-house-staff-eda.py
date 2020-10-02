#!/usr/bin/env python
# coding: utf-8

# ![](https://i.ytimg.com/vi/2BtuIfEAJ5s/maxresdefault.jpg)

# # Contents
# 1. [Introduction](#intro)
# 2. [Data Overview](#do)
# 3. [Staff Breakdown](#sb)
#  - [Tenured Veterans](#tv)
#  - [Gender Pay Gap](#gpg)
#  - [Status Matters](#sm)
#  - [First & Second Terms](#fst)
#  - [Transitions](#t)

# <a id="intro"></a>
# # 1. Introduction
# 
# - The White House Staff dataset is already (mostly) cleaned. Here we're going to load and prepare the data for exploration and analysis.
# - We added a 'president' column for easier filtering, and 'inflation_adjusted_salary' values.

# In[ ]:


#!pip install pywaffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pywaffle import Waffle 
import re
plt.style.use('fivethirtyeight')


# In[ ]:


whs = pd.read_csv('../input/white-house-staff-salaries-20172020/wh_staff_dataset.csv')
cpi = pd.read_csv('../input/inflation-data/cpi_dataset.csv', index_col='year', usecols=['year','average'])


# In[ ]:


whs.loc[whs['year'].between(1997,2000),'president'] = 'Clinton'
whs.loc[whs['year'].between(2001,2008),'president'] = 'Bush'
whs.loc[whs['year'].between(2009,2016),'president'] = 'Obama'
whs.loc[whs['year'].between(2017,2020),'president'] = 'Trump'


# 
# - The staff salaries need to be adjusted so they could be better compared. To calculate the inflation values (constant dollars) from nominal (current dollars) values we're using an [average CPI](https://inflationdata.com/Inflation/Consumer_Price_Index/HistoricalCPI.aspx) for each year:
# 
# <p style="text-align: center;">
# $Constant 2020 Salary$$=$$\frac{2020 CPI}{Historical CPI}$$\times$$Nominal Historical Salary$
# </p>

# In[ ]:


whs['inflation_adjusted_salary'] = ((whs.salary * cpi.loc[2020,'average']) / cpi.loc[whs.year,'average'].reset_index(drop=True)).round(2)


# <a id="do"></a>
# # 2. Data Overview

# - We get a sample of 5 rows as a reminder of data structure.

# In[ ]:


whs.sample(5)


# In[ ]:


whs.groupby('president')[['gender','status','pay_basis']].agg(['describe'])


# In[ ]:


print(whs.status.unique())
print(whs.pay_basis.unique())


# In[ ]:


pd.concat([whs[whs.status=='Employee (part-time)'], whs[whs.pay_basis=='Per Diem']])


# - Only Obama has the 'Employee (part-time)' status. We're going to consider them a normal employee.
# - Only Bush was paying some staffers Per Diem. Since it is impossible to know for how long a staffer was employed during one year we're going to interpolate their salary to a Per Annum value.

# In[ ]:


# Make a part-timer an employee
whs.loc[whs['status']=='Employee (part-time)','status'] = 'Employee'

# Per_Diem_value * 52 weeks * 5 days
whs.loc[whs['pay_basis']=='Per Diem','salary'] = whs[whs['pay_basis']=='Per Diem'].salary*52*5
whs.loc[whs['pay_basis']=='Per Diem','pay_basis'] = 'Per Annum'


# ----

# In[ ]:


fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(12,10))
whs.groupby('year').salary.mean().plot.line(color='green', xticks=range(1996,2021), ax=ax1)
whs.groupby('year').inflation_adjusted_salary.mean().plot.line(rot=45, color='lightgreen', ax=ax1)
ax1.set(ylabel='mean', yticks=range(50000,150000,20000))

whs.groupby('year').salary.median().plot.line(color='green', xticks=range(1996,2021), ax=ax2)
whs.groupby('year').inflation_adjusted_salary.median().plot.line(rot=45, color='lightgreen', ax=ax2)
ax2.set(ylabel='median', yticks=range(50000,150000,20000))
ax2.legend(['salary','inflation_adjusted_salary'])

whs.groupby('year').name.count().plot.line(xticks=range(1996,2021), ax=ax3)
ax3.set(ylabel='count')

whs.groupby('year').salary.sum().plot.line(rot=45, sharex=True, color='red',  xticks=range(1996,2021), ax=ax4)
whs.groupby('year').inflation_adjusted_salary.sum().plot.line(rot=45, sharex=True, color='darksalmon', ax=ax4)
ax4.set(ylabel='budget')
ax4.legend(['sum of salaries', 'sum of inflated salaries'])
plt.show()


# - On average the federal salary did increase over the years, even when adjusted for inflation.
# - The number of active staffers was lowest on the beginning of Bush's and Trump's term, while remaining constant during Clinton's and Obama's. 
# - The White House budget mostly remained the same, factoring in inflation, and despite the changes in staff. There is a spike in 2002, probably because of Bush's increase in staff and war on terror.
# 
# ----

# <a id="sb"></a>
# # 3. Staff Breakdown

# <a id="tv"></a>
# ## Tenured Veterans

# In[ ]:


whs.name.value_counts().value_counts().head(12)


# - Most staffers stay less or equal to 1 year, on average 2.
# - We're going to look more closely at people who worked at the WH for the longest (at least 24 years).

# In[ ]:


veterans = whs[whs.name.isin(whs.name.value_counts().head(6).index)].reset_index(drop=True)
veterans.name.unique()


# In[ ]:


df = veterans.pivot(index='year', columns='name', values='salary')
df.plot.line(figsize=(12,8), xticks=range(1997,2021), rot=45, style='.-', linewidth=2)
plt.show()


# - The salary of long term civil servants has been steadily increasing over the years. From 2010 to 2015 there is a stagnation that is attributed to the Obama federal salary freeze.
# 
# ------

# <a id="gpg"></a>
# ## Gender Pay Gap

# - Gender is mostly interpolated from the first name. Names with a similar probability of being male and female have been checked manually. 
# 
# - Source: National Data on the relative frequency of given names in the population of U.S. births where the individual has a Social Security Number (Tabulated based on Social Security records as of March 3, 2019)

# In[ ]:


fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,12))
df = whs.groupby(['year','gender']).salary.count().unstack()
df.plot.bar(rot=45, ax=ax1, legend=False)
ax1.set(ylabel='count')

df = whs.groupby(['year','gender']).salary.median().unstack().reset_index()
df.plot.scatter(x='year',y='Female',rot=45, ax=ax2, color='blue', xticks=range(1997,2021), s=30, marker='^')
df.plot.scatter(x='year',y='Male',rot=45, ax=ax2, color='red', s=20, marker='s')
ax2.set(ylabel='median salary')
ax2.legend(['Female', 'Male'])

# Basic regression line
m, b = np.polyfit(df.year,df.Male, 1)
ax2.plot(df.year, m*df.year + b, linewidth=1, color='salmon')
m, b = np.polyfit(df.year,df.Female, 1)
ax2.plot(df.year, m*df.year + b, linewidth=1, color='lightblue')

plt.show()


# - Even though the number of women in the White House has been overall high, there is an obvious gap in salaries compared to men.
# - Basic regression shows both genders have an upward slope, but it seems the gap is only getting larger.
# 
# ----

# <a id="sm"></a>
# ## Status Matters

# In[ ]:


# Who are the highest paid staffers in each administration?
whs.iloc[whs.groupby('president').salary.idxmax()]


# In[ ]:


# Top 10 highest payed staffers
whs.sort_values('salary', ascending=False).head(10)


# In[ ]:


# Top 10 highest paid staffers, adjusted for inflation
whs.sort_values('inflation_adjusted_salary', ascending=False).head(10)


# - It seems like many highly paid employees are detailees. According to [Quora](https://www.quora.com/What-is-a-White-House-staff-detailee):
# > A detailee is a civil servant from one agency/department/organization who is assigned temporarily ("on detail") to another agency/department/organization. Their home organization continues to pay their salary and usually retains some administrative responsibility for them throughout the assignment. The assignment has a predetermined length, and when that period ends, the employee returns to their home organization.
# 
# - White House employee pay is based on established pay rate tables and congressional mandates, but since detailees are temporarily assigned to a different workstation and work title they would be paid at the pay rate of his or her home agency but would be able to work at the White House. Making a staffer a detailee could be considered a loop-hole promotion.

# In[ ]:


fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(12, 6))
df = whs[whs.inflation_adjusted_salary>150000].groupby(['year','status']).salary.count().unstack()
df.plot.bar(rot=45, figsize=(12,8),stacked=True, ax=ax1)
ax1.set(ylabel='count')
ax1.get_legend().remove()
whs.groupby(['year','status']).salary.median().unstack().plot.bar(rot=45, figsize=(12,8),stacked=True, ax=ax2)
ax2.set(ylabel='median salary')
plt.show()


# - The number of highly paid employees (over \$150k, adjusted for inflation) has been similar over the years. Lowest point is 2001, and highest in 2002, as well as 2009 and 2010 at the beginning of Obama's term.
# - Ratio between detailee and employee is similar over the years.
# - Detailee get a higher salary than employees. Makes sense since they are usually highly experienced officials from different parts of government.

# In[ ]:


whsx = whs.drop_duplicates(subset=['name','status'], ignore_index=True)
whsy = whsx.groupby('name').status.count()
both = whsx.set_index('name').loc[whsy[whsy > 1].index,:].reset_index().drop_duplicates(subset=['name'], ignore_index=True)
df = both.president.value_counts().reset_index()

plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=df.president,
    labels=list(df['index']),
    colors=['royalblue','seagreen','orangered','darkmagenta'],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.2), 'framealpha': 0, 'ncol': len(df.president)},
    figsize=(12, 6)
)
plt.show()


# - All presidents had people who were employees and became detailees, and vice versa.
# - Trump has the biggest number of detailees in one term.

# ----

# <a id="fst"></a>
# ## First & Second Terms

# - We're going to compare the available data for the first and second terms.

# In[ ]:


fbush = whs[whs['year'].between(2001,2004)]
fobama = whs[whs['year'].between(2009,2012)]
ftrump = whs[whs['year'].between(2017,2020)] 

sclinton = whs[whs['year'].between(1997,2000)]
sbush = whs[whs['year'].between(2005,2008)]
sobama = whs[whs['year'].between(2013,2016)]

fdf = pd.concat([fbush,fobama,ftrump], ignore_index=True)
sdf = pd.concat([sclinton,sbush,sobama], ignore_index=True)


# In[ ]:


# Basic stats for first terms 
fdf.groupby('president').salary.agg(['mean','median','sum','max','count'])


# In[ ]:


# Basic stats for second terms
sdf.groupby('president').salary.agg(['mean','median','sum','max','count'])


# In[ ]:


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 6))
my_pal = {'Clinton':'darkmagenta','Bush':'seagreen','Obama':'royalblue','Trump':'orangered'}
sns.boxplot(x='president', y='inflation_adjusted_salary', data=fdf, 
            ax=ax1, palette=my_pal, linewidth=2, width=0.4)
ax1.set_title('First Term')
ax1.set_yticks(range(0,330000,30000))
sns.boxplot(x='president', y='inflation_adjusted_salary', data=sdf, 
            ax=ax2, palette=my_pal, linewidth=2, width=0.4)
ax2.set_title('Second Term')
ax2.set_yticks(range(0,330000,30000))
plt.show()


# - Trump's median salary is greater than all former administrations, for both terms, even when inflated. 
# - Bush had a remarkably frugal 2001, but but it seems that was in preparation of the heavy spending 2002. Number of WH staffers reached a maximum as well during that time.

# In[ ]:


#Full term employees
fte1 = []
for pres in [fbush,fobama,ftrump]:
    fte1.append(pres.name.value_counts().value_counts().sum() - pres.name.value_counts().value_counts().loc[4])
fte2 = []
for pres in [sclinton,sbush,sobama]:
    fte2.append(pres.name.value_counts().value_counts().sum() - pres.name.value_counts().value_counts().loc[4])

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
sns.pointplot(x=['Bush','Obama','Trump'], y=fte1, ax=ax1)
sns.pointplot(x=['Clinton','Bush','Obama'], y=fte2, ax=ax2)
ax1.set(yticks=range(700,900,40))
plt.show()


# - Trump had the least number employees who stayed the whole term. This data isn't viable for getting a good picture about turnover, but a basic understanding tells us that Bush had most people stay a total of 4 terms.
# 
# --------

# <a id="t"></a>
# ## Transitions
# 
# - We will take a look at salaries in two years between transition of power to the next administration.

# In[ ]:


whs.groupby('year')[['salary','inflation_adjusted_salary']].agg(['sum','count'])


# In[ ]:


fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(12, 16), sharex=True)

ax1.set_title('salary distribution')
whs.loc[whs['year']==2000,'salary'].plot.hist(bins=25, color='darkmagenta', alpha=0.5, ax=ax1)
whs.loc[whs['year']==2001,'salary'].plot.hist(bins=25, color='seagreen', alpha=0.5, ax=ax1)
ax1.legend(['Clinton','Bush'])

whs.loc[whs['year']==2008,'salary'].plot.hist(bins=25, color='seagreen', alpha=0.5, ax=ax2)
whs.loc[whs['year']==2009,'salary'].plot.hist(bins=25, color='royalblue', alpha=0.5, ax=ax2)
ax2.legend(['Bush','Obama'])

whs.loc[whs['year']==2016,'salary'].plot.hist(bins=25, color='royalblue', alpha=0.5, ax=ax3)
whs.loc[whs['year']==2017,'salary'].plot.hist(bins=25, color='orangered', alpha=0.5, ax=ax3)
plt.xlabel('salary')
ax3.legend(['Obama','Trump'])
plt.show()


# - Trump's number of staffers is lower the first two years - the documented reason are a lower number of confirmations by Congress, as well as an unorganized transition team. Despite of that, most staffers are in the bracket around $115k, substantially bigger than previous administrations.
# - There is a \$3-5m budget difference between ending and starting a new administration. Bush and Trump administrations were more frugal that first year - that may be attributed to economical stabiliy while starting their term.
# 
# ------

# # History
# - Version 1: Initial commit.
# - Version 2: TBD
# 
# I welcome all and any feedback.
