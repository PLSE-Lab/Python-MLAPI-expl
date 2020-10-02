#!/usr/bin/env python
# coding: utf-8

# ** Kiva - Understanding Poverty Levels of Borrowers **
# 
# Kiva is an international nonprofit organization, with a mission to connect people through lending to alleviate poverty. In order to set investment priorities, help inform lenders, and understand their target communities, knowing the level of wealth or poverty of each borrower is critical.
# 
# In Kaggle Datasets' inaugural Data Science for Good challenge, Kiva is inviting the Kaggle community to estimate and describe the welfare levels of residents in given regions using historical loans data combined with external data sources. 
# 
# The purpose of this notebook is to perform analysis of poverty levels (measured by MPI) of borrowers, by joining Kiva Loans dataset with the MPI dataset. 

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Kiva Loans dataset**

# In[ ]:


kiva_loans=pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
print(kiva_loans.shape)
kiva_loans.sample(5)


# **Kiva MPI dataset**

# In[ ]:


kiva_mpi_region_locations=pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
print(kiva_mpi_region_locations.shape)
kiva_mpi_region_locations.sample(10)


# In a random sample of 10 above, we see several rows of all NaN values and (1000.0, 1000.0) value in geo column. It is surprising that there are so many of these rows, I wonder why. Something wrong with the data? We'll have to do some cleaning to remove those rows, but let's look into it first.

# In[ ]:


kiva_mpi_region_locations.isnull().sum()      


# Out of 2772 rows, it looks like there are 1788 without a LocationName-MPI data pair. 65%! Two thirds of this dataset is practically empty. For now, let's remove those rows. It seems that the MPI dataset provided by Kiva won't give us a very rich source of povery data. Perhaps we shouldn't be surprised; after all this whole Kaggle challenge is about finding extenal data sources and building models to estimate poverty levels in areas where Kiva has active loans.
# 
# For now, we'll work with what we have. Let's clean the MPI data.

# In[ ]:


mpi = kiva_mpi_region_locations[['country','region', 'MPI']]
mpi = mpi.dropna()
print(mpi.shape)
print(mpi.sample(5))


# After removing all the NaN rows, we are left with 984 country-region-MPI triples.
# 
# Now let's clean the loans dataset. We will remove all rows with missing country or region values. We need country and region to be available so we can join our loans dataset with the MPI dataset.

# In[ ]:


loans = kiva_loans[['country','region','loan_amount','activity','sector','borrower_genders', 'repayment_interval']]
print(loans.shape)
loans = loans.dropna(subset = ['country','region'])
print(loans.shape)
loans.sample(5)


# After removing the NaN rows from loans data, we are left with 92% of rows out of the original 671205 rows.
# 
# Let's now enrich the loans data with the MPI data.

# In[ ]:


d= pd.merge(loans, mpi, how='left')
d.count()


# Dissapointingly, only a small number of loan entries in the loans dataset were assigned an MPI. Out of 614405 loans entries, we could only assign MPI value to 50955 of them, a mere 8%.
# 
# Let's only look at those 8% of loan entries for now. It is getting even more clear that we will need to find a different measure of poverty to be added to the loans dataset!

# In[ ]:


d = d.dropna(subset=['MPI'])
d.sample(5)


# Let's start the analysis! 
# 
# We first calculate the mean loan amount of all loans by country-region pair. 

# In[ ]:


d1=d.groupby(['country','region','MPI'])['loan_amount'].mean().reset_index(name='Mean Loan Amount')
plt.figure(figsize=(8,6))
sns.regplot(x = d1.MPI, y = d1['Mean Loan Amount'], fit_reg=True)
plt.title("MPI vs. Mean Loan Amount")
plt.show()


# The plot suggests that the higher the MPI of the region, the lower is the mean amount of loan given to the region.
# 
# Let's now look at the total of all loan amounts given to each region.

# In[ ]:


d2=d.groupby(['country','region','MPI'])['loan_amount'].sum().reset_index(name='Sum of Loan Amounts')
plt.figure(figsize=(8,6))
sns.regplot(x = d2.MPI, y = d2['Sum of Loan Amounts'], fit_reg=True)
plt.title("MPI vs. Sum of Loan Amounts")
plt.show()


# Similar result as above. The higher the MPI of the region, the lower is the total amount of loan given to the region.
# 
# What about the relationship between the MPI and the total number of loans given to each region?

# In[ ]:


d3=d.groupby(['country','region','MPI'])['loan_amount'].count().reset_index(name='Number of Loans')
plt.figure(figsize=(8,6))
sns.regplot(x = d3.MPI, y = d3['Number of Loans'], fit_reg=False)
plt.title("MPI vs. Number of Loans")
plt.show()


# There is an obvious outlier, a region with an MPI around 0.31 and a large number (~10 000) of loans. What region is that? 

# In[ ]:


d3.loc[d3['Number of Loans'] == d3['Number of Loans'].max()]


# The region with the higest number of loans is Nigeria, Kaduna. Except for this outlier, it looks like from the plot that the lower the MPI of the region, the higher the number of loans. 
# 
# Let's try visualize this in a slightly different way.

# In[ ]:


d3['Location']=d3['country'] + ", " + d3['region']
d3 = d3.set_index(['Location'])


# In[ ]:


d3 = d3.sort_values("Number of Loans", ascending=False)
d3 = d3.loc[d3['Number of Loans']>20]


# In[ ]:


fig = plt.figure(figsize=(17, 7)) 
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

d3[["MPI"]].plot(kind='bar', color='blue', ax=ax2, width=.4, position=0)
d3[["Number of Loans"]].plot(kind='bar', color='green', ax=ax, width=.4, position=1)

ax.set_ylabel('Number of Loans')
ax2.set_ylabel('MPI')
plt.show()


# We can see from the above plot that regions with higher MPI receive a smaller number of loans compared to regions with lower MPI..

# Now let's look at the gender of borrowers. We are interested in whether there are any differences in poverty levels of borrowers based on their gender.

# In[ ]:


df_gender = pd.DataFrame(d.borrower_genders.str.split(',').tolist())
#dd = pd.concat([df_gender[0], df_gender[1], df_gender[2], df_gender[3], df_gender[4], df_gender[5]], ignore_index=True).dropna()
d['gender'] = df_gender[0]
# This needs to be done better. Now I'm only taking the first column.


# In[ ]:


d.groupby(['gender'])['MPI'].mean()


# Based on the mean MPI per gender, it might be that women borrowes are from higher MPI locations than male borrowers. Let's double check this by plotting the distributions.

# In[ ]:


fig = plt.figure(figsize=(17, 7)) 
ax = fig.add_subplot(111) 
sns.distplot(d.loc[d['gender']=='female'].MPI, label='female', ax=ax, color='r', bins=50, kde=True)
sns.distplot(d.loc[d['gender']=='male'].MPI, label='male', ax=ax, color='b', bins=50, kde=True)
plt.legend()
plt.show()


# Distributions of MPIs for both males and females almost overlap, so we conclude there is no difference in povery levels between male and female borrowers.
# 
# Let's now look at the sectors. Are there any differences in povery levels of borrowes that that focus their activities in different sectors? 

# In[ ]:


df_sector = d.groupby(['sector'])['MPI'].mean().sort_values(ascending=True).reset_index(name="Average MPI")
plt.figure(figsize=(10,5))
g = sns.barplot(x='sector', y="Average MPI", data=df_sector);
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Average MPI per Sector")
g.set_xlabel("Sector")
plt.show()


# Borrowers from regions with the higest MPI borrow money for Agriculture, Personal Use and Education. Borrowers from regions with lowest MPI borrow money for Halth, Transportation and Manufactoring.
# 
# Activity level is much more granular than the sector. Let's look at which activities require people to borrow money in high and low MPI regions.

# In[ ]:


df_activity = d.groupby(['activity'])['MPI'].mean().sort_values(ascending=True).reset_index(name="Average MPI")
plt.figure(figsize=(25,5))
g = sns.barplot(x='activity', y="Average MPI", data=df_activity);
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Average MPI per Activity")
g.set_xlabel("Activity")
plt.show()


# In high MPI regions, people borrow money to fund Celebrations, Primary/Secondary Education and Farming. (Cement is an interesting activity, I wonder what this is.) People in low MPI regions borrow to pay for Childcare, Energy, Musical instruments and Technology.

# In[ ]:


df_repayment_interval = d.groupby(['repayment_interval'])['MPI'].mean().sort_values(ascending=True).reset_index(name="Average MPI")
plt.figure(figsize=(16,5))

plt.subplot(121)
g = sns.barplot(x='repayment_interval', y="Average MPI", data=df_repayment_interval);
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Average MPI vs Replayment Interval")
g.set_xlabel("Repayment Interval")

plt.subplot(122)
g1 = sns.violinplot(x='repayment_interval', y='MPI', data=d)
g1.set_title("MPI Distribution by Repayment Interval")
g1.set_xlabel("")
g1.set_ylabel("MPI")

plt.show()


# A bullet loan is a loan where a payment of the entire principal of the loan, and sometimes the principal and interest, is due at the end of the loan term. We see that the borrowes from high MPI regions receive bullet loans, while low MPI regions tend to get loans with irregular or monthly repayment intervals. Or, based on the plot on the right,  it could just be that borrowers from Nigeria, Kaduna (MPI ~ 0.31), where there was a large amount of given loans, all received a bullet loan, and this is skewing the average.
# 
# **Conclusion**
# 
# We looked at some basic analysis of Kiva loans and MPI levels of regions where borrowers are from. We found that regions with higher MPI levels tend to get less loans, as well as smaller amounts. We found no differences in povery levels of male or female borrowers. We identified which sectors and activities require funding in high versus low MPI regions.
# 
# The main takeaway, however, is that there wasn't really that much MPI data for country-region pairs - we could only assign MPI to 8% of Kiva loans. This is a really small proportion of the whole loans dataset. We will have to find some other external data sources in order to be better able to assign poverty levels to each loan and do a comprehensive analysis on the whole loans dataset.
