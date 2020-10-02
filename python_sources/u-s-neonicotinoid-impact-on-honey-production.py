#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set()
from matplotlib import style
from matplotlib.backends.backend_pdf import PdfPages
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use(['seaborn-dark'])
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
from IPython.display import display
from IPython.display import Image


# ### Key questions addressed:

# > Do neonics impact total production, yield per colony and number of colonies?
# 
# > What neonic compounds were most frequently used per state and over time?
# 
# > Has production value and stocks increased or decreased with neonic usage?

# *  For visualization purposes and ease of data handling I split the dataset into pre-and post neonics (2003) as it is stated in the data description that neonics were more heavily used after 2003. As you will see this distinction was not so clear-cut for all states and regions, with moderate use of neonics being deployed prior to 2003 before dramatically surging in use. 
# 
# 
# *  This data set contains information on neonic usage per state from 1991 to 2016.
# 
# *  I found this dataset extremely interesting and delightful to work with as mimimal cleaning was necessary and variable distribution was good. 
# 

# In[ ]:


df = pd.read_csv('../input/vHoneyNeonic_v03.csv')
# Show data frame
df.head()


# In[ ]:


#Checking for null values
df.isnull().sum()


# In[ ]:


#Fill all NaN with 0
df = df.fillna(0)


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


#Convert state, StateName, Region to category
#Select all columns of type 'object'
object_columns = df.select_dtypes(['object']).columns
object_columns


# In[ ]:


#Convert selected columns to type 'category'
for column in object_columns: 
    df[column] = df[column].astype('category')
df.dtypes


# In[ ]:


df.describe().T


# In[ ]:


df.corr()


# In[ ]:


#print unique features for each row
print("Feature, UniqueValues") 
for column in df:
    print(column + "," + str(len(df[column].unique())))


# ### Correlation heatmap

# In[ ]:


#Add new column determined by pre- and post-neonics (2003)
df['post-neonics(2003)'] = np.where(df['year']>=2003, 1, 0)


# In[ ]:


# Correlation matrix using code found on https://stanford.edu/~mwaskom/software/seaborn/examples/many_pairwise_correlations.html
#USA_youtube_df = pd.read_csv("USvideos.csv")
sns.set(style="white")

# Select columns containing continuous data
continuous_columns = df[['numcol','yieldpercol','totalprod','stocks','priceperlb','prodvalue','year','nCLOTHIANIDIN','nIMIDACLOPRID','nTHIAMETHOXAM','nACETAMIPRID','nTHIACLOPRID','nAllNeonic','post-neonics(2003)']].columns

# Calculate correlation of all pairs of continuous features
corr = df[continuous_columns].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom colormap - blue and red
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, vmax=1, vmin=-1,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.yticks(rotation = 0)
plt.xticks(rotation = 45)


# ### Exploring quantity of neonics used prior to 2003
# 

# In[ ]:


#Making a new dataframe containing all data before 2003
df_pre_2003 = df[(df['year']<2003)]


# In[ ]:


#Making a new dataframe containing all data including and after 2003
df_2003 = df[(df['year']>=2003)]


# In[ ]:


#Units of neonic used each year 
df.groupby(['year'])['nAllNeonic'].sum()


# In[ ]:


import seaborn as sns
plt.style.use(['seaborn-dark'])


# In[ ]:


df_pre_2003.groupby(['year'])['nAllNeonic'].sum().plot(color='green')
plt.title("Neonic usage prior to 2003")


# In[ ]:


df_2003.groupby(['year'])['nAllNeonic'].sum().plot(color='green')
plt.title("Neonic usage after 2003")


# In[ ]:


#Timeline of neonic usage
df.groupby(['year'])['nAllNeonic'].sum().plot(color='green')
plt.title("Complete timeline of neonic usage")


# In[ ]:


#bivariate distribution of all neonics vs. year
sns.jointplot(data=df, x='year', y='nAllNeonic', kind='reg', color='g')


# ### Neonic usage by state

# In[ ]:


#Resizing plots
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


# In[ ]:


df_pre_2003.groupby(['StateName'])['nAllNeonic'].sum().plot(kind='bar')
plt.title("Neonic usage by state prior to 2003")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


# In[ ]:


df_2003.groupby(['StateName'])['nAllNeonic'].sum().plot(kind='bar')
plt.title("Neonic usage by state post 2003")


# In[ ]:


df_pre_2003.groupby(['Region'])['nAllNeonic'].sum().plot(kind='bar')
plt.title("Neonic use prior to 2003")


# In[ ]:


df_2003.groupby(['Region'])['nAllNeonic'].sum().plot(kind='bar')
plt.title("Neonic usage after 2003")


# > ``California`` was leading the way in neonics usage prior to 2003, being the state most heavily invested in their use out of a total of 44 states.
# 
# > States in the ``NorthEast`` have consistently used low amounts of neonics- will be interesting to see their totalprod, yieldpercol and no. of colonies.
# 
# > It is interesting that ``South`` and ``West`` regions were using the most neonics prior to 2003 but this switched after 2003 and they are now using the least amount. Perhaps detrimental effects of neonics were evident and caused a decrease in colony numbers leading to a reduction in amount of neonics required or else colony sizes/honey production farming were reduced in these regions due to external reasons. 
# 
# > To check this hypothesis I will plot colony size of region by year.

# ### Number of colonies per region over time

# In[ ]:


g = sns.FacetGrid(df, col="Region") 
g.map(sns.regplot, "year", "numcol", line_kws={"color": "red"})


# > The ``Midwest`` is the only region is where the no. of colonies appear to be increasing despite a massive increase in neonic usage after 2003.
# 
# > No. of colonies in the ``NorthEast`` are consistent with no dramatic fluctuations, corresponding to a consistently small usage of neonics, but appear to be slightly decreasing overall.
# 
# > In the ``South`` a slight decrease in no. of colonies can be seen after 2003 which coincides with a decrease in neonic usage, although colony numbers appear to be on the rise/maintaining a consistent number in the future.
# 
# > A decrease in the no. of colonies can also be said for the ``West`` region.

# ### No. of colonies by state 

# In[ ]:


df_pre_2003.groupby(['StateName'])['numcol'].sum().plot(kind='bar')
plt.title('No. of colonies per state pre-2003')


# In[ ]:


df_2003.groupby(['StateName'])['numcol'].sum().plot(kind='bar')
plt.title('Number of colonies per state from 2003 to present')


# >It is interesting that North Dakota surpassed California after 2003 in terms of no. of colonies and were very conservative with neonic usage prior to 2003 when compared to California.
# 
# > Florida, Texas and Idaho's no. of colonies appear to have remained consistent over time.
# 
# >This information will be useful for further analysis to compare total production and yield per colony pre- and post-2003 by state.

# ### Popularity of neonic compound used by each state

# In[ ]:


df_pre_2003.groupby(['StateName'])['nIMIDACLOPRID'].sum().plot(kind='bar')
plt.title('nIMIDACLOPRID usage pre-2003')


# In[ ]:


df_pre_2003.groupby(['StateName'])['nTHIAMETHOXAM'].sum().plot(kind='bar')
plt.title('nTHIAMETHOXAM usage pre-2003')


# In[ ]:


df_pre_2003.groupby(['StateName'])['nACETAMIPRID'].sum().plot(kind='bar')
plt.title('nACETAMIPRID usage pre-2003')


# In[ ]:


df_pre_2003.groupby(['StateName'])['nAllNeonic'].sum().plot(kind='bar')
plt.title('All Neonic usage pre-2003')


# In[ ]:


df.groupby(['StateName'])['nAllNeonic'].sum().plot(kind='bar')
plt.title("All neonic usage per state")


# ### Findings so far:

# > The information provided with the data states that neonics were used in small amounts before 2003 and then in larger quantities afterwards. We can see that California was heavily using the neonics ``nACETAMIPRID``, ``nTHIAMETHOXAM``, ``nIMIDACLOPRID`` prior to 2003. 
# 
# >The neonics ``nTHIACLOPRID`` and ``nCLOTHIANIDIN`` had not been in use prior to 2003 or perhaps were not yet developed.
# 
# >Other states that were using neonics in large quantities were ``Texas``, ``Washington``, and ``Mississippi``, with the most popular neonic in usage in the years before 2003 being ``nTHIAMETHOXAM`` with quite a few states using it on their colonies.
# 
# >States located in the ``South`` and ``West`` regions were more heavily involved in neonic deployment before 2003. This is in contrast to after 2003 when the ``Midwest`` surpasses all other regions in neonic usage and the states in the ``South`` and ``West`` reduce their neonic usage.
# 
# > California, Illinois and Iowa are responsible for the most neonic deployment. 
# 
# 

# #### Yield per colony over time in each region

# In[ ]:


#Yield per colony over time in each region
g = sns.FacetGrid(df, col="Region") 
g.map(sns.regplot, "year", "yieldpercol", line_kws={"color": "red"})


# #### Total production per region over time

# In[ ]:


#Total production per region over time
g = sns.FacetGrid(df, col="Region") 
g.map(sns.regplot, "year", "totalprod", line_kws={"color": "red"})


# ### Most prevalent neonic compound in use in U.S.

# In[ ]:


plt.plot( 'year', 'nCLOTHIANIDIN', data=df.sort_values('year'), marker='', color='blue', linewidth=2)
plt.plot( 'year', 'nIMIDACLOPRID', data=df.sort_values('year'), marker='', color='olive', linewidth=2)
plt.plot( 'year', 'nTHIAMETHOXAM', data=df.sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")
plt.plot( 'year', 'nACETAMIPRID', data=df.sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")
plt.plot( 'year', 'nTHIACLOPRID', data=df.sort_values('year'), marker='', color='#cd71ff', linewidth=2, label="nTHIACLOPRID")
plt.legend()
plt.title("Neonic usage over time")


# ## Case study: 
# #### Interesting States- California, North Dakota, Washington, Florida, Texas, Mississippi. 

# In[ ]:


g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='California'])
g.set_axis_labels("Year", "Total Production")
plt.title('California')

g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='California'])
g.set_axis_labels("Year", "Yield per col")
plt.title('California')

g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='California'])
g.set_axis_labels("Year", "No. of colonies")
plt.title('California')


# In[ ]:


plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="California"].sort_values('year'), marker='', color='blue', linewidth=2)
plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="California"].sort_values('year'), marker='', color='olive', linewidth=2)
plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="California"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")
plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="California"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")
plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="California"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")
plt.legend()


# California's no. of colonies, yield per colony and total production have been decreasing consistently since their frequent heavy use of neonics in 1994, namely Imidacloprid.

# In[ ]:


g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Illinois'])
g.set_axis_labels("Year", "Total Production")
plt.title('Illinois')

g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Illinois'])
g.set_axis_labels("Year", "Yield per col")
plt.title('Illinois')

g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Illinois'])
g.set_axis_labels("Year", "No. of colonies")
plt.title('Illinois')


# In[ ]:


plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Illinois"].sort_values('year'), marker='', color='blue', linewidth=2)
plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Illinois"].sort_values('year'), marker='', color='olive', linewidth=2)
plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Illinois"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")
plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Illinois"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")
plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Illinois"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")
plt.title('Illinois')
plt.legend()


# It is interesting to see Illinois's yield per colony was increasing prior to 2003 when they were using no neonics. 

# In[ ]:


g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Iowa'])
g.set_axis_labels("Year", "Total Production")
plt.title('Iowa')

g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Iowa'])
g.set_axis_labels("Year", "Yield per col")
plt.title('Iowa')

g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Iowa'])
g.set_axis_labels("Year", "No. of colonies")
plt.title('Iowa')


# In[ ]:


plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Iowa"].sort_values('year'), marker='', color='blue', linewidth=2)
plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Iowa"].sort_values('year'), marker='', color='olive', linewidth=2)
plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Iowa"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")
plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Iowa"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")
plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Iowa"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")
plt.title("Iowa")
plt.legend()


# Similar to Illinois, Iowa's total production and number of colonies were decreasing before 2003., with yield per colony increasing. Their number of colonies, however, were rapidly decreasing. In both these cases, it would seem that the introduction of neonics was beneficial in slowing colony collapse and colony numbers are now on the rise. 

# In[ ]:


g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='North Dakota'])
g.set_axis_labels("Year", "Total Production")
plt.title('North Dakota')

g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='North Dakota'])
g.set_axis_labels("Year", "Yield per col")
plt.title('North Dakota')

g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='North Dakota'])
g.set_axis_labels("Year", "No. of colonies")
plt.title('North Dakota')


# In[ ]:


plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="North Dakota"].sort_values('year'), marker='', color='blue', linewidth=2)
plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="North Dakota"].sort_values('year'), marker='', color='olive', linewidth=2)
plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="North Dakota"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")
plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="North Dakota"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")
plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="North Dakota"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")
plt.legend()


# North Dakota is one of the only states where both the number of colonies and total production have increased over time, despite overall yield per colony plummeting. Perhaps this can be attributed to their conservative usage of Imidacloprid from 1996 onwards. The neonic compound most used by this state in recent years is Clothianidin. 

# In[ ]:


g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Idaho'])
g.set_axis_labels("Year", "Total Production")
plt.title('Idaho')

g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Idaho'])
g.set_axis_labels("Year", "Yield per col")
plt.title('Idaho')

g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Idaho'])
g.set_axis_labels("Year", "No. of colonies")
plt.title('Idaho')


# In[ ]:


plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Idaho"].sort_values('year'), marker='', color='blue', linewidth=2, label="nCLOTHIANIDIN")
plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Idaho"].sort_values('year'), marker='', color='olive', linewidth=2, label="nIMIDACLOPRID")
plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Idaho"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")
plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Idaho"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")
plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Idaho"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")
plt.legend()


# Upon further analysis, Idaho's no. of colonies are decreasing rapidly with increasing neonic usage. Yield per colony and total production is also decreasing. 

# In[ ]:


g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Washington'])
g.set_axis_labels("Year", "Total Production")
plt.title('Washington')

g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Washington'])
g.set_axis_labels("Year", "Yield per col")
plt.title('Washington')

g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Washington'])
g.set_axis_labels("Year", "No. of colonies")
plt.title('Washington')


# In[ ]:


plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Washington"].sort_values('year'), marker='', color='blue', linewidth=2, label="nCLOTHIANIDIN")
plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Washington"].sort_values('year'), marker='', color='olive', linewidth=2, label="nIMIDACLOPRID")
plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Washington"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")
plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Washington"].sort_values('year'), marker='', color='orange', linewidth=2, label="nACETAMIPRID")
plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Washington"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")
plt.legend()


# Washington's total production appears to have increased after 2003 but there are a lot of outliers so this conclusion could be questioned. Washington's no. of colonies has increased rapidly after 2003 but the yield per colony is decreasing. Washington's history of neonic usage is extensive and varied with Imidacloprid being the most heavily used.  

# In[ ]:


g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Florida'])
g.set_axis_labels("Year", "Total Production")
plt.title('Florida')

g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Florida'])
g.set_axis_labels("Year", "Yield per col")
plt.title('Florida')

g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Florida'])
g.set_axis_labels("Year", "No. of colonies")
plt.title('Florida')


# In[ ]:


plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Florida"].sort_values('year'), marker='', color='blue', linewidth=2)
plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Florida"].sort_values('year'), marker='', color='olive', linewidth=2)
plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Florida"].sort_values('year'), marker='', color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")
plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Florida"].sort_values('year'), marker='', color='orange', linewidth=2,linestyle='dashed', label="nACETAMIPRID")
plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Florida"].sort_values('year'), marker='', color='pink', linewidth=2, label="nTHIACLOPRID")
plt.legend()


# Florida's total production, yield per colony and number of colonies were stable and increasing before 2003 with only moderate use of Imidacloprid. Now however,  total production and yield per colony is at an all time low despite their no. of colonies increasing. 

# In[ ]:


g = sns.lmplot(x="year", y="totalprod", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Texas'])
g.set_axis_labels("Year", "Total Production")
plt.title('Texas')

g = sns.lmplot(x="year", y="yieldpercol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Texas'])
g.set_axis_labels("Year", "Yield per col")
plt.title('Texas')

g = sns.lmplot(x="year", y="numcol", hue="post-neonics(2003)",
               truncate=True, size=5, data=df[df.StateName =='Texas'])
g.set_axis_labels("Year", "No. of colonies")
plt.title('Texas')


# In[ ]:


plt.plot( 'year', 'nCLOTHIANIDIN', data=df[df.StateName=="Texas"].sort_values('year'), color='blue', linewidth=2)
plt.plot( 'year', 'nIMIDACLOPRID', data=df[df.StateName=="Texas"].sort_values('year'), color='olive', linewidth=2)
plt.plot( 'year', 'nTHIAMETHOXAM', data=df[df.StateName=="Texas"].sort_values('year'), color='red', linestyle='dashed', linewidth=2, label="nTHIAMETHOXAM")
plt.plot( 'year', 'nACETAMIPRID', data=df[df.StateName=="Texas"].sort_values('year'), color='orange', linewidth=2, label="nACETAMIPRID")
plt.plot( 'year', 'nTHIACLOPRID', data=df[df.StateName=="Texas"].sort_values('year'), color='pink', linewidth=2, label="nTHIACLOPRID")
plt.legend()
plt.title('Texas')


# A large number of outliers in the total production and no. of colonies columns make it hard to definitively conclude they are in fact increasing post-2003, although it it is certain that yield per colony is decreasing. 

# ### Economics of honey production

# In[ ]:


sns.jointplot(data=df, x='year', y='priceperlb', kind='reg', color='g')


# In[ ]:


sns.jointplot(data=df, x='year', y='stocks', kind='reg', color='g')


# In[ ]:


sns.jointplot(data=df, x='year', y='prodvalue', kind='reg', color='g')


# ### My interpretations:

# *  Neonics were introduced in the U.S. to combat parasitic mites that attack bees. Colony numbers appear to have been decreasing in a lot of states prior to the introduction of neonics. In the case of Texas, Florida, Washington, Idaho, Iowa, Illinois, their colony numbers were decreasing prior to heavy neonic deployment in 2003. In these states it would seem that the introduction of neonics was beneficial in slowing and preventing large scale colony collapse.
# 
# * Although a number of states have restored their number of colonies, it is clear that neonics are severly affecting yield per colony.
# 
# * North Dakota is an interesting case study whereby both the number of colonies and total production have increased over time, despite overall yield per colony plummeting. Perhaps this can be attributed to their conservative usage of Imidacloprid from 1996 onwards. The neonic compound most used by this state is clothianidin.
# 
# * California's heavy and consistent use of Imidacloprid from 1994 might have been a factor in them not regaining their colony numbers. Starting in 1995, Idaho also used increasing and consistent quantities of imidacloprid. States that still have decreasing number of colonies seemed to be using increasing amounts of Imidacloprid from a very early stage so perhaps this compound has a net cumulative negative effect. 
# 
# * Illinois and Iowa used the most neonics after California. Unlike California though their number of colonies are on the rise and they predominantly used the compound clothianidin.
# 
# * An overall trend is a decrease in neonic usage in the U.S. after 2014. Have neonics become less effective? Are more people becoming aware of their toxic effects? 
# 
# *  Overall, neonic pesticides are negatively impacting bee colonies in the U.S. This corresponds with the surmounting scientific evidence of the detrimental effects of neonics on both wild and farmed bees:
# 
# 
#          Whitehorn, Penelope R., et al. "Neonicotinoid pesticide reduces bumble bee colony growth 
#          and queen production." Science (2012): 1215025.
#                   
#          Di Prisco, Gennaro, et al. "Neonicotinoid clothianidin adversely affects insect immunity 
#          and promotes replication of a viral pathogen in honey bees." Proceedings of the 
#          National Academy of Sciences 110.46 (2013): 18466-18471.
#                 
#          Godfray, H. Charles J., et al. "A restatement of the natural science evidence base 
#          concerning neonicotinoid insecticides and insect pollinators." 
#          Proc. R. Soc. B 281.1786 (2014): 20140558.
# 
# * Price per lb and production value of honey is increasing while stocks are decreasing.
# 
# * Neonic usage in the U.S.A is highly varied by state and other factors such as climate and proximity of colonies to areas of high radiation need to be taken into consideration as well as neonics. 
# 
# * Although research points to all neonic pesticides being toxic, more analysis and information is required to deduce whether a particular neonic compound is more detrimental or beneficial to bee colonies. 
