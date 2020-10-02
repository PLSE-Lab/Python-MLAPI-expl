#!/usr/bin/env python
# coding: utf-8

# # Affordable Housing in Nashville/Davidson County

# ## Part 1: Data Wrangling Basics with pandas

# In[ ]:


from IPython.display import Image
Image("../input/pandas-logo/pandas_logo.png")


# The pandas library will be our Swiss Army knife for data analysis. We will be using pandas DataFrames, which are useful for manipulating tabular, nonhomogenous data.
# 
# First, import the pandas library with the alias pd:

# In[ ]:


import pandas as pd


# To create our first DataFrame, we need to point pandas to the csv file we want to import and use the `read_csv` method:

# In[ ]:


houses_2009 = pd.read_csv('../input/appraisal-data/2009SINGLEFAMILYSF.txt')


# We can now view the first few rows using the `.head()` method. We can also inspect the last few rows using `.tail()`.

# In[ ]:


houses_2009.head(n=10)


# In[ ]:


houses_2009.tail()


# We can check the dimensions of our DataFrame using `.shape`. This returns a tuple (number of rows, number of columns).

# In[ ]:


houses_2009.shape


# The method `.info()` gives us more information about each column.

# In[ ]:


houses_2009.info()


# Now, let's look at the column names:

# In[ ]:


houses_2009.columns


# We can access a column by using `houses_2009["<column name>"]`.

# In[ ]:


houses_2009["2009 TOTAL APPR"]


# Let's adjust the column names. It is a lot easier to work with columns that do not have spaces or start with a number. This will make using tab-completion easier, and will allow us to access a particular column using `houses_2009.<column_name>`.

# In[ ]:


houses_2009.columns = ['APN', 'DistrictCode', 'CouncilDistrict', 'AddressFullAddress',
       'AddressCity', 'AddressPostalCode', 'LAND', 'IMPR', 'TOTALAPPR',
       'TOTALASSD', 'FinishedArea']
houses_2009.columns


# In[ ]:


houses_2009.TOTALAPPR


# Now, let's read in the 2013 and 2017 files and change the column names in the same way as for the 2009 file.

# In[ ]:


houses_2013 = pd.read_csv('../input/appraisal-data/2013SINGLEFAMILYSF.txt')
houses_2017 = pd.read_csv('../input/appraisal-data/2017SINGLEFAMILYSF.txt')
houses_2013.columns = ['APN', 'DistrictCode', 'CouncilDistrict', 'AddressFullAddress',
       'AddressCity', 'AddressPostalCode', 'LAND', 'IMPR', 'TOTALAPPR',
       'TOTALASSD', 'FinishedArea']
houses_2017.columns = ['APN', 'DistrictCode', 'CouncilDistrict', 'AddressFullAddress',
       'AddressCity', 'AddressPostalCode', 'LAND', 'IMPR', 'TOTALAPPR',
       'TOTALASSD', 'FinishedArea']


# In[ ]:





# ### Part 2: Slicing, Counting, and Basic Plots

# If we want to see the different entries for a column, we can use the `.unique()` method:

# In[ ]:


houses_2009.AddressCity.unique()


# The `.value_counts()` method will give a tally of the entries in a particular column, sorted in descending order by default. For example, let's say we want to get a tally of homes by city.

# In[ ]:


houses_2009.AddressCity.value_counts()


# ### Filtering
# 
# One way to filter a pandas DataFrame is to slice it using `.loc`. The syntax will look like `houses_2009.loc[<boolean array>]` where the boolean array has the same length as our DataFrame. This will result in a DataFrame containing the rows corresponding to the Trues from the array.
# 
# For example, let's find all homes in Brentwood. Start by creating a boolean array.

# In[ ]:


houses_2009.AddressCity == 'BRENTWOOD'


# Then, use `.loc`.

# In[ ]:


houses_2009.loc[houses_2009.AddressCity == 'BRENTWOOD']


# A few more words on `.loc`. We can slice our DataFrame using `.loc` and a boolean series, as before, but we can also use `.loc` to slice by passing which index values we want (row, column, or both). This looks like `df.loc[<rows>,<columns>]`

# In[ ]:


houses_2009.loc[100:105,['AddressFullAddress', 'AddressCity']]


# In[ ]:


houses_2009.loc[[1000, 2000, 3000], 'CouncilDistrict']


# Time for some plots! Let's look at the number of single family homes assessed in each district. 
# 
# Plotting can be done using pandas DataFrame methods. Behind the scenes, this is done using the matplotlib library. In order to get our plots to display in our notebook, we can use the ipython magic command `%matplotlib inline`.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


houses_2009.CouncilDistrict.value_counts().plot.bar();


# The plots we create are highly customizable. For a (partial) list of stylistic options, see https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.plot.html.

# In[ ]:


fig = houses_2009.CouncilDistrict.value_counts().plot.bar(figsize = (14,6), width = 0.75,
                                                         rot = 0, color = 'plum')
fig.set_xlabel('District')
fig.set_title('Number of Single-Family Homes by District, 2009', fontweight = 'bold');


# The `.plot` method orders the bars in the order they apper in the given DataFrame. If we want to change the order, say in order by district number, we can reorder the rows of our DataFrame by using `.loc`.

# In[ ]:


houses_2009.CouncilDistrict.value_counts().loc[range(1,36)]


# In[ ]:


fig = houses_2009.CouncilDistrict.value_counts().loc[list(range(1,36))].plot.bar(figsize = (14,6), width = 0.75,
                                                         rot = 0, color = 'plum')
fig.set_xlabel('District')
fig.set_title('Number of Single-Family Homes by District, 2009', fontweight = 'bold');


# In[ ]:





# If we want to display the distribution of a variable, we can use a histogram. For example, let's say we want to look at the distribution of square footages.

# In[ ]:


fig = houses_2009.FinishedArea.plot.hist(figsize = (10,4))
fig.set_title('Distribution of Homes by Square Footage', fontweight = 'bold');


# We get some extreme square footages - let's investigate.

# Let's narrow down the dataset we use in order to get a more informative histogram. Notice too that we can adjust the number of bins to further improve the histogram.

# In[ ]:


houses_2009.loc[houses_2009.FinishedArea < 10000].FinishedArea.plot.hist(figsize = (10,4), bins = 50)
plt.title('Distribution of Homes by Square Footage', fontweight = 'bold');


# We can also put two histograms on the same plot in order to compare two distributions. Let's say we want to compare the distribution of appraisal values from 2009 to 2017.

# In[ ]:


houses_2009.TOTALAPPR.plot.hist();


# In order to determine a good cutoff, we can use the `.describe()` method which gives summary statistics on our DataFrame.

# In[ ]:


houses_2009.TOTALAPPR.describe()


# We can see that 75% of homes are appraised at \$220,000 or less.

# In[ ]:


houses_2009.loc[houses_2009.TOTALAPPR <= 750000].TOTALAPPR.plot.hist(bins = 50);


# With two histograms, we should set the alpha values lower to increase the transparency. It is also probably a good idea to normalize our histograms so they they are showing densities rather than counts.

# In[ ]:


fig = houses_2009.loc[houses_2009.TOTALAPPR <= 750000].TOTALAPPR.plot.hist(bins = 50, alpha = 0.6, density = True, label = '2009', figsize = (10,5))
houses_2017.loc[houses_2017.TOTALAPPR <= 750000].TOTALAPPR.plot.hist(bins = 50, alpha = 0.6, density = True, label = '2017');
fig.axes.get_yaxis().set_visible(False)
fig.set_title('Distribution of Appraisal Values, 2009 vs 2017')
fig.legend();


# In[ ]:





# ## `groupby` to Aggregate by Category
# 
# pandas makes it easy to calculate statistics by category. First, we need to specify the column or columns we want to group by and then we specify how we cant to calculate our summary statistics.

# In[ ]:


houses_2009.groupby('CouncilDistrict').count()


# Notice how it returns a count for each column. We can instead choose a single column.

# In[ ]:


houses_2009.groupby('CouncilDistrict').APN.count()


# In[ ]:


houses_2009.groupby('CouncilDistrict').TOTALAPPR.mean()


# We can also apply multiple aggregate functions by using the `.agg` method after applying `.groupby`.

# In[ ]:


houses_2009.groupby('CouncilDistrict').TOTALAPPR.agg(['mean', 'median'])


# We can also group by multiple columns to get more granular information:

# In[ ]:


agg_df = houses_2009.groupby(['CouncilDistrict', 'AddressPostalCode']).TOTALAPPR.median()
agg_df


# Let's bring in some income information. The file ACS.csv contains the number of households and the median household income by council district, obtained from the US Census Bureau's American Community Survey.

# In[ ]:


ACS = pd.read_csv('../input/census/ACS.csv')
ACS = ACS.set_index('district')
ACS.head()


# We can compare two metrics on the same plot by creating a scatter plot.

# In[ ]:


fig, ax = plt.subplots(figsize = (10,6))
plt.scatter(x = ACS.loc[ACS.year == 2017].median_income, 
         y = houses_2017.groupby('CouncilDistrict').TOTALAPPR.median(),
           alpha = 0.75)
plt.xlabel('Median Income')
plt.ylabel('Median Household Appraisal Value');


# Our plot would be a lot more useful if we knew which district corresponded to each point. 
# 
# To add our labels, we can use the matplotlib `annotate` function. We need to specify the text that we want (using the `s` parameter) and where we want to place the text (using the `xy` parameter).

# In[ ]:


fig, ax = plt.subplots(figsize = (10,6))
plt.scatter(x = ACS.loc[ACS.year == 2017].median_income, 
         y = houses_2017.groupby('CouncilDistrict').TOTALAPPR.median(),
           alpha = 0.75)
plt.xlabel('Median Income')
plt.ylabel('Median Household Appraisal Value')
for i in range(1,36):
    plt.annotate(xy = (ACS.loc[ACS.year == 2017].median_income.loc[i], houses_2017.groupby('CouncilDistrict').TOTALAPPR.median().loc[i]),
                s = str(i));


# ### Part 3: Appraisal Values

# We can get an idea of the distribution of appraisal values by using the `.describe()` method.

# In[ ]:


houses_2017.TOTALAPPR.describe()


# Where are the 5 most expensive houses? To answer this, we can use the `.nlargest()` method.

# In[ ]:


houses_2017.nlargest(n=5, columns='TOTALAPPR')


# In[ ]:





# ## Changes by District
# 
# Let's look at how much appraisal values increased from 2013 to 2017. First, we need to combine our three dataframes into one. To concatenate, we pass a list containing the DataFrames we want to combine to the function `pd.concat()`. The default (which is what we need) is to stack the DataFrames vertically, but we can also combine them horizontally by specifying `axis = 1`. 

# In[ ]:


houses_2009['year'] = 2009
houses_2013['year'] = 2013
houses_2017['year'] = 2017
houses = pd.concat([houses_2009, houses_2013, houses_2017])
houses.head()


# Now, let's calculate the percent change in median house value overall for Davidson County.

# In[ ]:


overall_pct_change = 100*(houses.loc[houses.year == 2017].TOTALAPPR.median() - houses.loc[houses.year==2013].TOTALAPPR.median()) / houses.loc[houses.year == 2013].TOTALAPPR.median()
overall_pct_change


# Let's use `.groupby` to calculate median appraisal values by district and year.

# In[ ]:


median_appr = houses.groupby(['CouncilDistrict','year']).TOTALAPPR.median().reset_index()
median_appr.head()


# Since we want to find the percent change in median appraisal value from 2013-2017, it would be helpful to have years as columns. We can accomplish this using `.pivot`.

# In[ ]:


median_appr = median_appr.pivot(index = 'CouncilDistrict', columns='year', values='TOTALAPPR')
median_appr.head()


# In[ ]:


median_appr.columns


# Notice that our column index consists of integers.

# In[ ]:


median_appr['pct_change'] = 100*(median_appr[2017] - median_appr[2013]) / median_appr[2013]
median_appr.head()


# In[ ]:


median_appr = median_appr.reset_index()
median_appr.head()


# Let's start with a basic plot showing the percent change per district.

# In[ ]:


fig, ax = plt.subplots()
median_appr.plot.scatter(x = 'CouncilDistrict', y = 'pct_change', figsize = (14,6), ax = ax, s=75)
ax.set_xticks(list(range(1,36)))
ax.set_ylim(0,140)
plt.grid(axis = 'both', linestyle = '--', alpha = 0.5, lw = 2)

plt.title('Percent Change in Median Appraisal Value, 2013 - 2017', fontweight = 'bold', fontsize = 14)
plt.xlabel('Council District', fontsize = 14)
plt.ylabel('Percent Change (%)', fontsize = 14)
plt.xticks(fontsize = 12, fontweight = 'bold')
plt.yticks(fontsize = 14);


# To see how each district compares to the county overall, we can add a horizontal line by using matplotlib's `axhline` function.

# In[ ]:


fig, ax = plt.subplots()
median_appr.plot.scatter(x = 'CouncilDistrict', y = 'pct_change', figsize = (14,6), ax = ax, s=75)
ax.set_xticks(list(range(1,36)))
ax.set_ylim(0,140)
plt.grid(axis = 'both', linestyle = '--', alpha = 0.5, lw = 2)
plt.axhline(y=overall_pct_change, color='r', lw = 1)


plt.title('Percent Change in Median Appraisal Value, 2013 - 2017', fontweight = 'bold', fontsize = 14)
plt.xlabel('Council District', fontsize = 14)
plt.ylabel('Percent Change (%)', fontsize = 14)
plt.xticks(fontsize = 12, fontweight = 'bold')
plt.yticks(fontsize = 14);


# Next, let's add line segments from each point to the median value to emphasize the difference from the overall value. To do this, we need to use the matplotlib `collections` module.

# In[ ]:


from matplotlib import collections  as mc


# We will be creating a `LineCollection` object for our line segments. We need to pass this a list of lists, one for each segment containing the coordinates for the start and endpoints of the segment. We can do this using a list comprehension and by zipping a list.
# 
# `zip` takes two iterables (eg. lists) and returns an iterable containg tuples (a,b), where a comes from teh first iterable and b comes from the second.

# In[ ]:


for x in zip([1,2,3], ['a', 'b', 'c']):
    print(x)


# In[ ]:


lines = [[(x,y),(x,overall_pct_change)] for x,y in zip(range(1,36), median_appr['pct_change'])]

fig, ax = plt.subplots()
median_appr.plot.scatter(x = 'CouncilDistrict', y = 'pct_change', figsize = (14,6), ax = ax, s=75)
ax.set_xticks(list(range(1,36)))
ax.set_ylim(0,140)
plt.grid(axis = 'both', linestyle = '--', alpha = 0.5, lw = 2)
plt.axhline(y=overall_pct_change, color='r', lw = 1)
lc = mc.LineCollection(lines, linewidths=2)
ax.add_collection(lc)

plt.title('Percent Change in Median Appraisal Value, 2013 - 2017', fontweight = 'bold', fontsize = 14)
plt.xlabel('Council District', fontsize = 14)
plt.ylabel('Percent Change (%)', fontsize = 14)
plt.xticks(fontsize = 12, fontweight = 'bold')
plt.yticks(fontsize = 14);


# Finally, let's add some annotations to let the reader know that the horizontal line corresponds to the overall percent change and the emphasize the largest value. If we use the `arrowstyle` argument to the `annotate` function, then we can add arrows to our plot along with text.

# In[ ]:


lines = [[(x,y),(x,overall_pct_change)] for x,y in zip(range(1,36), median_appr['pct_change'])]

fig, ax = plt.subplots()
median_appr.plot.scatter(x = 'CouncilDistrict', y = 'pct_change', figsize = (14,6), ax = ax, s=75)
ax.set_xticks(list(range(1,36)))
ax.set_ylim(0,140)
plt.grid(axis = 'both', linestyle = '--', alpha = 0.5, lw = 2)
plt.axhline(y=overall_pct_change, color='r', lw = 1)
lc = mc.LineCollection(lines, linewidths=2)
ax.add_collection(lc)

plt.title('Percent Change in Median Appraisal Value, 2013 - 2017', fontweight = 'bold', fontsize = 14)
plt.xlabel('Council District', fontsize = 14)
plt.ylabel('Percent Change (%)', fontsize = 14)
plt.xticks(fontsize = 12, fontweight = 'bold')
plt.yticks(fontsize = 14)

ax.annotate("Percent Change for\nDavidson County\n(" + "{:.1f}".format(overall_pct_change)+ "%)", xy=(36, overall_pct_change), 
            xytext=(33, 90), fontsize = 12, ha = 'center', va = 'center', color = 'red', fontweight = "bold",
            arrowprops=dict(arrowstyle="->", lw = 2))
ax.annotate("District 5\n(" + "{:.1f}".format(median_appr['pct_change'].max())+ "%)", xy=(5.5, median_appr['pct_change'].max()-1), 
            xytext=(9, 120), fontsize = 12, ha = 'center', va = 'center', color = 'red', fontweight = "bold",
            arrowprops=dict(arrowstyle="->", lw = 2));


# **Exercise:** Create a chart similar to the one above showing median appraisal value per square foot of finished area.

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution.
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_409.py')


# Now, let's try to see what the supply of affordable housing looks like. To do this, we will adopt HUD's definitions of "affordable" and "workforce" housing, which is based on the Area Median Income. See https://www.hud.gov/program_offices/comm_planning/affordablehousing/ 

# Area Median Income for Davidson County: 
# 
#     2009: $64,900  
#     
#     2013: $62,300  
#     
#     2017: $68,000  

# HUD declares a household to be **cost-burdened** if they are spending more than 30% of their income on housing costs.
# 
# **Affordable Housing:** Won't cost-burden households making less than 60% of AMI.
#     
# **Workforce Housing:** Won't cost people making between 60% and 120% of AMI.

# Let's classify these according to whether they are affordable to someone making 30%, 60%, 90%, or 120% of AMI. We will need to estimate the total yearly cost of each house. We can make this estimate based on its appraised value.

# In[ ]:


def find_mortgage_payment(TOTALAPPR,years = 30, rate = 4, down_payment = 20):
    P = TOTALAPPR * (1 - (down_payment / 100))
    n = 12 * years
    r = rate / (100 * 12)
    M = P * (r * (1 + r)**n) / ((1 + r)**n - 1)
    return M


# In[ ]:


houses['est_mortgage_cost'] = houses.TOTALAPPR.apply(lambda x: 12*find_mortgage_payment(x))


# We also need to factor in property taxes, which are determined by a houses's district code. The rates can be found at http://www.padctn.org/services/tax-rates-and-calculator/. These rates are applied to the houses's assessed value, which is contained in the `TOTALASSD` column.
# 

# In[ ]:


tax_rates = {'USD' : 3.155/100,
            'GSD' : 2.755/100,
             'GO' : 3.503 / 100,
             'FH' : 2.755/100,
             'OH' : 2.755/100,
             'BM' : 3.012/100,
             'BH' : 2.755/100,
            'CBID' : 3.2844 / 100,
            'GBID': 3.2631 / 100,
            'RT' : 3.437/100,
            'LW' : 2.755/100}


# In[ ]:


def calculate_property_taxes(row):
    return row.TOTALASSD * tax_rates[row.DistrictCode]


# In[ ]:


houses['est_property_tax'] = houses.apply(calculate_property_taxes, axis = 1)


# Oops! We get an error. It seems that some of the values in the DistrictCode have some extra white space. The string method `.strip()` removes any white space at the beginning or end of a sting. We can apply this to our DistrictCode column to correct the problem.

# In[ ]:


houses.DistrictCode = houses.DistrictCode.str.strip()


# In[ ]:


houses['est_property_tax'] = houses.apply(calculate_property_taxes, axis = 1)


# We also need to factor in insurance cost. We'll use \$60/month, or \$720/year as our estimate for homeowner's insurance.

# In[ ]:


houses['est_yearly_cost'] = houses.est_mortgage_cost + houses.est_property_tax + 720


# Now that we have an estimated yearly cost, we can put each house into a category. We'll use 5 categories:
#  * __AFF_1:__ not cost-burdening to those making 30% of AMI
#  * __AFF_2:__ not cost-burdening to those making 60% of AMI
#  * __WF_1:__ not cost-burdening to those making 90% of AMI
#  * __WF_2:__ not cost-burdening to those making 120% of AMI
#  * __AWF:__ Requires more than 120% of AMI

# In[ ]:


def classify_house(value, year):
    if year == 2009: AMI = 64900
    if year == 2013: AMI = 62300
    if year == 2017: AMI = 68000
    if value <= 0.3 * 0.3*AMI:
        return 'AFF_1'
    elif value <= 0.3 * 0.6 * AMI:
        return 'AFF_2'
    elif value <= 0.3* 0.9 * AMI:
        return 'WF_1'
    elif value <= 0.3 * 1.2*AMI:
        return 'WF_2'
    else:
        return 'AWF'


# In[ ]:


houses['category'] = houses.apply(lambda x: classify_house(x['est_yearly_cost'], x['year']), axis = 1)


# In[ ]:


plt.figure(figsize = (10,6))
houses.loc[houses.year == 2017].category.value_counts()[['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']].plot.bar(rot = 0)
plt.title('Number of Single-Family Homes by Category, 2017');


# In[ ]:





# Let's explore a couple of types of plots we can use to display our findings. First, let's look at a side-by-side bar chart. Here is an example DataFrame to demonstrate the pieces we need.

# In[ ]:


chess_players = pd.DataFrame({'player': ['Magnus Carlsen', 'Fabiano Caruana', 'Ding Liren'],
                       'wins': [962, 793,414],
                        'draws': [930,821,575],
                       'losses': [334,459,186]})
chess_players= chess_players.set_index('player')
chess_players


# In[ ]:


fig, ax = plt.subplots(figsize = (7,5))
chess_players.plot.bar(ax = ax, edgecolor = 'black', lw = 1.5, rot = 0, width = 0.8)
plt.title('Top 3 Chess Players by ELO', fontweight = 'bold')
plt.xlabel('')
ax.legend(bbox_to_anchor=(1, 0.6));


# What do we need to pull this off? A dataframe, indexed by the category with a column per year containing the count for that year.

# Need: A DataFrame with 5 rows (categories) and 3 columns (years).
# 
# We can start by using `.groupby` to count by category and by year.

# In[ ]:


category_count = houses.groupby(['category', 'year']).APN.count().reset_index()
category_count


# This is close to what we need except that here all of the years are contained in one column; whereas, we need to have one column per year. We can do this by using `.pivot`.

# In[ ]:


pivot_df = category_count.pivot(index = 'category', columns = 'year', values = 'APN')
pivot_df


# We can reorder our DataFrame using `.loc`:

# In[ ]:


pivot_df = pivot_df.loc[['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']]
pivot_df


# In[ ]:


fig, ax = plt.subplots(figsize = (10,6))
pivot_df.plot.bar(ax = ax, edgecolor = 'black', lw = 1.5, rot = 0, width = 0.8)
plt.title('Number of Single-Family Homes by Category', fontweight = 'bold')
plt.xlabel('')
plt.ylabel('Number of Homes')
ax.legend(bbox_to_anchor=(1, 0.6));


# In[ ]:





# Side-by-side bar charts are one option. An different way to display this data is to use a stacked bar chart.

# In[ ]:


chess_players


# In[ ]:


fig, ax = plt.subplots(figsize = (8,6))
chess_players.plot.bar(stacked=True, edgecolor = 'black', lw = 1.5, 
                       rot = 0, ax = ax, width = 0.75,title = 'Top 3 Chess Players by ELO')
plt.xlabel('')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 0.6));


# To make this chart easier to read, we can add annotations to each block.
# 
# To do this, we need to calculate the cumulative sum (height) for each chess player. The numpy method `.cumsum` does this for us.

# In[ ]:


import numpy as np


# In[ ]:


fig, ax = plt.subplots(figsize = (8,6))
chess_players.plot.bar(stacked=True, edgecolor = 'black', lw = 1.5, 
                       rot = 0, ax = ax, width = 0.75,title = 'Top 3 Chess Players by ELO')
plt.xlabel('')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 0.6))

rows = chess_players.iterrows()
for i in range(3):
    values = next(rows)[1]
    heights = np.array([0] + list(values.cumsum()[:-1])) + values/2
    for height, value in zip(heights,values):
        plt.text(x = i, y = height, s = f'{value:,}', color = 'white', ha = 'center', va = 'center', fontweight = 'bold');


# **Goal:** Create a stacked bar chart showing the number of homes by category with year on the horizontal axis.
# 
# **What we need:** a DataFrame with 3 rows (years) and 5 columns (categories).
# 
# We can accomplish this again by pivoting the `category_count` DataFrame, but this time using a different index. 

# In[ ]:


pivot_df_stack = category_count.pivot(index='year', columns='category', values='APN')


# In[ ]:


fig,ax = plt.subplots(figsize = (10,7))
pivot_df_stack.plot.bar(stacked=True, ax = ax, rot = 0, width = 0.75, edgecolor = 'black',lw=1.5)
plt.title('Davidson County Affordable Single-Family Homes Profile', fontweight = 'bold', fontsize = 14)
ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
plt.yticks(fontsize = 12)
plt.ylabel('Number of Homes', fontsize = 14)
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.xlabel('')

rows = pivot_df_stack.iterrows()
for i in range(3):
    values = next(rows)[1]
    heights = np.array([0] + list(values.cumsum()[:-1])) + values/2
    for height, value in zip(heights,values):
        plt.text(x = i, y = height, s = f'{value:,}', color = 'white', ha = 'center', va = 'center', fontweight = 'bold')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], 
              ['More Than 120% of AMI', 'At Least 120% of AMI', 'At Least 90% of AMI', 'At Least 60% of AMI', 'At Least 30% of AMI'], 
              bbox_to_anchor=(1, 0.6), title = 'Not Cost-Burderning to\nHouseholds Making', title_fontsize = 12);


# Finally, let's recreate the above plot but look by district. To improve our plot, we can add interactivity with ipywidets and a map with geopandas.

# In[ ]:


from ipywidgets import interact
import geopandas as gpd


# In[ ]:


council_districts = gpd.read_file('../input/shapefiles/Council_District_Outlines.geojson')


# In[ ]:


# %load ../input/exercisesolutions/soln_407.py
@interact(district = range(1,36))
def make_plot(district):
    data = houses.loc[houses.CouncilDistrict == district]
    
    count_by_category = data.groupby(['year', 'category']).APN.count().reset_index()
    
    pivot_df = count_by_category.pivot(index='year', columns='category', values='APN').fillna(0)
    pivot_df = pivot_df.loc[:,['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']]
    
    fig,ax = plt.subplots(figsize = (10,7))
    pivot_df.plot.bar(stacked=True, ax = ax, rot = 0, width = 0.75, edgecolor = 'black', lw = 1.5)
    plt.title('Affordable Housing Profile, District ' + str(district), fontweight = 'bold', fontsize = 14)
    
    ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
    plt.yticks(fontsize = 12)
    plt.ylabel('Number of Homes', fontsize = 14)
    plt.xticks(fontsize = 14, fontweight = 'bold')
    plt.xlabel('')
    
    def check_height(value):
        if value >= pivot_df.sum(axis = 1).max() * 0.03:
            return f'{int(value):,}'
        return ''
    
    rows = pivot_df.iterrows()
    for i in range(3):
        values = next(rows)[1]
        heights = np.array([0] + list(values.cumsum()[:-1])) + values/2
        for height, value in zip(heights,values):
            plt.text(x = i, y = height, s = check_height(value), color = 'white', ha = 'center', va = 'center', fontweight = 'bold')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], 
              ['More Than 120% of AMI', 'At Least 120% of AMI', 'At Least 90% of AMI', 'At Least 60% of AMI', 'At Least 30% of AMI'], 
              bbox_to_anchor=(1, 0.9), title = 'Not Cost-Burderning to\nHouseholds Making', title_fontsize = 12)
    
    cd = council_districts[['district', 'geometry']]
    
    cd.loc[:,'chosen_district'] = 0
    cd.loc[cd.district == str(district), 'chosen_district'] = 1
    
    mini_map = plt.axes([.9, .25, .25, .25]) #[left, bottom, width, height]
    cd.plot(column = 'chosen_district', ax = mini_map, legend = False, edgecolor = 'black', cmap = 'binary')
    plt.axis('off')
    plt.title('District ' + str(district)); 


# # Bonus Material

# ## Seaborn

# Seaborn is a visualization library built on top of matplotlib.

# In[ ]:


import seaborn as sns


# Seaborn gives us options to compare home appraisal values across districts or across years. For example, we can use a box plot.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.boxplot(data = houses_2017.loc[houses_2017.CouncilDistrict.isin([1,2,3,4])], 
            x = 'CouncilDistrict', 
            y = 'TOTALAPPR')
plt.title('Home Appriasal Values, 2017')
plt.ylim(0, 1000000);


# Another option is to use a violin plot, which includes a density estimation.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.violinplot(data = houses_2017.loc[houses_2017.CouncilDistrict.isin([1,2,3,4])], x = 'CouncilDistrict', 
               y = 'TOTALAPPR')
plt.title('Home Appraisal Values, 2017')
plt.ylim(0, 1000000);


# In[ ]:





# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.boxplot(data = houses, x = 'year', y = 'TOTALAPPR')
plt.title('Home Appraisal Values')
plt.ylim(0, 1000000);


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.violinplot(data = houses, x = 'year', y = 'TOTALAPPR')
plt.title('Home Appraisal Values')
plt.ylim(0, 1000000);


# In[ ]:





# What if we want to dynamically change the maximum value based on the district? We can use the numpy library to help us. Specifically, the numpy `percentile` function can be used to set a maximum value.
# 
# For example, to find the appraisal value for which 90% of homes are appraised below, use `np.percentile` and specify 90 as the percentile we wish to find.

# In[ ]:


np.percentile(houses.loc[houses.CouncilDistrict == 34, 'TOTALAPPR'], 90)


# In[ ]:


from ipywidgets import interact


# In[ ]:


@interact(district = range(1,36), plot_type = ['box', 'violin'])
def plot_dist(district, plot_type):
    fig = plt.figure(figsize = (10,6))
    if plot_type == 'box':
        sns.boxplot(data = houses.loc[houses.CouncilDistrict == district], x = 'year', y = 'TOTALAPPR')
    if plot_type == 'violin':
        sns.violinplot(data = houses.loc[houses.CouncilDistrict == district], x = 'year', y = 'TOTALAPPR')
    ymax = np.percentile(houses.loc[(houses.CouncilDistrict == district) & (houses.year == 2017), 'TOTALAPPR'], 99.9)
    plt.ylim(0, ymax)
    plt.title('Total Appraised Value, District ' + str(district));


# In[ ]:


cd = council_districts[['district', 'geometry']]


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


@interact(district = range(1,36), plot_type = ['box', 'violin'])
def plot_dist(district, plot_type):
    fig = plt.figure(figsize = (10,6))
    if plot_type == 'box':
        sns.boxplot(data = houses.loc[houses.CouncilDistrict == district], x = 'year', y = 'TOTALAPPR')
    if plot_type == 'violin':
        sns.violinplot(data = houses.loc[houses.CouncilDistrict == district], x = 'year', y = 'TOTALAPPR')
    ymax = np.percentile(houses.loc[(houses.CouncilDistrict == district) & (houses.year == 2017), 'TOTALAPPR'], 99.9)
    plt.ylim(0, ymax)
    plt.title('Total Appraised Value, District ' + str(district))
    
    cd['chosen_district'] = 0
    cd.loc[cd.district == str(district), 'chosen_district'] = 1
    
    mini_map = plt.axes([.85, .3, .4, .4]) #[left, bottom, width, height]
    cd.plot(column = 'chosen_district', ax = mini_map, legend = False, edgecolor = 'black', cmap = 'binary')
    plt.axis('off')
    plt.title('District ' + str(district));


# In[ ]:





# Seaborn also provides distplots, which can do histograms or kernel density estimates, which are essentially smoothed histograms.

# In[ ]:


df = houses.loc[houses.CouncilDistrict == 20]

target_0 = df.loc[df.year == 2009]
target_1 = df.loc[df.year == 2013]
target_2 = df.loc[df.year == 2017]

sns.distplot(target_0[['TOTALAPPR']], hist=False, label = '2009')
sns.distplot(target_1[['TOTALAPPR']], hist=False, label = '2013')
g = sns.distplot(target_2[['TOTALAPPR']], hist=False, label = '2017')

g.set(xlim=(0, 500000));


# In[ ]:


df = houses.loc[houses.CouncilDistrict == 20]

target_0 = df.loc[df.year == 2009]
target_1 = df.loc[df.year == 2013]
target_2 = df.loc[df.year == 2017]

sns.distplot(target_0[['TOTALAPPR']], hist=True, label = '2009')
sns.distplot(target_1[['TOTALAPPR']], hist=True, label = '2013')
g = sns.distplot(target_2[['TOTALAPPR']], hist=True, label = '2017')

g.set(xlim=(0, 500000));


# In[ ]:


@interact(district = range(1,36))
def make_dist_plot(district):
    plt.figure(figsize = (10,6))
    
    df = houses.loc[houses.CouncilDistrict == district]

    target_0 = df.loc[df.year == 2009]
    target_1 = df.loc[df.year == 2013]
    target_2 = df.loc[df.year == 2017]

    sns.distplot(target_0[['TOTALAPPR']], hist=False, label = '2009', kde_kws={'lw': 2.5})
    sns.distplot(target_1[['TOTALAPPR']], hist=False, label = '2013', kde_kws={'lw': 2.5})
    g = sns.distplot(target_2[['TOTALAPPR']], hist=False, label = '2017', kde_kws={'lw': 2.5}, color = 'purple')

    xmax = np.percentile(houses.loc[(houses.CouncilDistrict == district) & (houses.year == 2017), 'TOTALAPPR'], 95)

    g.set(xlim=(0, xmax))
    g.set(yticks = [])
    g.set(title="Distribution of Appraisal Values, District " + str(district));


# In[ ]:





# ## Plotly

# `plotly` is another visualization library which allows for more dynamic, interactive graphics. The `plotly express` module makes it easy to work with pandas dataframes to quickly produce dynamic, interactive plots.

# In[ ]:


import plotly.express as px


# In[ ]:


chess_players


# Plotly express requires our data to be "tidy". This means only one "observation" per row. Currently, our `chess_players` DataFrame is not tidy, because we should have only one type of observation per row. To tidy it, we can use the `.melt()` method.

# In[ ]:


melted_chess = chess_players.reset_index().melt(id_vars=['player'], var_name = 'outcome')
melted_chess


# In[ ]:


fig = px.bar(melted_chess, x='player', y='value', color = 'outcome', 
             width = 800, height = 500,
            category_orders = {'outcome' : ['wins', 'losses', 'draws']})
fig.update_layout(title_text = 'Top Rated Chess Players', title_font_size = 24)
fig.update_yaxes(title_text = 'Number of Games', title_font_size = 20, tickfont_size = 14,)
fig.update_xaxes(title_text = '', tickfont_size = 18)
fig.update_layout(legend_traceorder = 'reversed')
fig.show()


# In[ ]:





# In[ ]:


category_count.head()


# In[ ]:


category_count.year = category_count.year.astype('category')


# In[ ]:


@interact(district = range(1, 36))
def make_plot(district):
    df = houses.loc[houses.CouncilDistrict == district].groupby(['category', 'year']).APN.count().reset_index().rename(columns = {'APN' : 'count'})
    #df.year = df.year.astype('category')
    fig = px.bar(df, x='year', y='count', color = 'category', width = 800, height = 500,
                category_orders = {'category' : ['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']})
    fig.update_yaxes(title_text = 'Number of Homes', title_font_size = 18)
    fig.update_xaxes(title_text = '', tick0=2009, dtick=4, tickfont_size = 18)
    fig.update_layout(title_text = 'Affordable Housing Profile, District ' + str(district), title_font_size = 20)
    fig.update_layout(legend_traceorder = 'reversed')
    fig.show()


# In[ ]:


district_counts = houses.groupby(['year', 'CouncilDistrict', 'category']).APN.count().reset_index().rename(columns = {'APN' : 'num_homes'})
district_counts = district_counts.loc[district_counts.CouncilDistrict.isin(list(range(1,36)))]


# In[ ]:


@interact(year = [2009, 2013, 2017])
def make_plot(year):
    df = district_counts.loc[district_counts.year == year]
    fig = px.bar(df, x='CouncilDistrict', y='num_homes', color = 'category', width = 900, height = 500,
                category_orders = {'category' : ['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']})
    fig.update_yaxes(title_text = 'Number of Homes', title_font_size = 18, range = [0,8300])
    fig.update_xaxes(title_text = 'District', tick0=1, dtick=1, tickfont_size = 14, tickangle = 0)
    fig.update_layout(title_text = 'Davidson County Affordable Housing Profile by District, ' + str(year), title_font_size = 20)
    fig.update_layout(legend_traceorder = 'reversed')
    fig.show()


# In[ ]:


@interact(district = range(1,36))
def make_plotly(district):
    df = houses.loc[houses.CouncilDistrict == district]
    ymax = np.percentile(df.TOTALAPPR, 99.9)
    
    fig = px.box(df, x="year", y="TOTALAPPR", width = 800, height = 500)
    fig.update_yaxes(range=[0,ymax], title_text = 'Appraised Value', title_font_size = 18)
    fig.update_xaxes(title_text = '', tickfont_size = 18)
    fig.update_layout(title_text = 'Appraised Values, District ' + str(district), title_font_size = 20)

    fig.show()


# In[ ]:


@interact(district = range(1,36))
def make_plotly(district):
    df = houses.loc[houses.CouncilDistrict == district]
    ymax = np.percentile(df.TOTALAPPR, 99.9)
    
    fig = px.violin(df, x="year", y="TOTALAPPR", width = 800, height = 500, box = True)
    fig.update_yaxes(range=[0,ymax], title_text = 'Appraised Value', title_font_size = 18)
    fig.update_xaxes(title_text = '', tickfont_size = 18)
    fig.update_layout(title_text = 'Appraised Values, District ' + str(district), title_font_size = 20)

    fig.show()

