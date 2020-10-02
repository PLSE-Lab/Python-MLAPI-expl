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


# **Exercise:** Read in the 2013 and 2017 files and change the column names in the same way as for the 2009 file.

# In[ ]:


# Your code here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_003.py')


# In[ ]:





# ### Part 2: Slicing, Counting, and Basic Plots

# If we want to see the different entries for a column, we can use the `.unique()` method:

# In[ ]:


houses_2009.AddressCity.unique()


# If we just care about how many unique elements there are, we can use `.nunique()` instead.

# In[ ]:


houses_2009.AddressCity.nunique()


# The `.value_counts()` method will give a tally of the entries in a particular column, sorted in descending order by default. For example, let's say we want to get a tally of homes by city.

# In[ ]:


houses_2009.AddressCity.value_counts()


# **Exercise:** Use `value_counts()` to get a tally of homes by their full address for 2009.

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_101.py')


# How many addresses are duplicated? To answer this, we can use an inequality to return a Boolean series:

# In[ ]:


houses_2009.AddressFullAddress.value_counts() > 1


# Python lets us do arithmetic with Booleans. True = 1 and False = 0:

# In[ ]:


(houses_2009.AddressFullAddress.value_counts() > 1).sum()


# Let's investigate the most common address, 0 Edmondson Pike. 
# 
# One way to filter a pandas DataFrame is to slice it using `.loc`. The syntax will look like `houses_2009.loc[<boolean array>]` where the boolean array has the same length as our DataFrame. This will result in a DataFrame containing the rows corresponding to the Trues from the array.
# 
# For example, let's find all homes in Brentwood. Start by creating a boolean array.

# In[ ]:


houses_2009.AddressCity == 'BRENTWOOD'


# Then, use `.loc`.

# In[ ]:


houses_2009.loc[houses_2009.AddressCity == 'BRENTWOOD']


# **Exercise:** Create a boolean array that indicates whether the address in a particular row is equal to 0 Edmondson Pike. Then use this to slice the `houses_2009` DataFrame.

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_102.py')


# **Exercise:** Use `.loc` to determine how many times O EDMONDSON PIKE appears in the 2013 and 2017 datasets.

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_103.py')


# **Harder Exercise:** How many homes in our 2009 dataset have house number 0?  
# Hint: To use string methods on columns of DataFrames which are strings, access that column and then use .str. For example, try `houses_2009.AddressCity.str.lower()`

# In[ ]:


houses_2009.AddressCity.str.lower()


# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_104.py')


# It is not clear what is going on with these duplicated addresses, but we need to decide what to do with our duplicate addresses. One option is to drop all duplicates, which can be accomplished using the `.drop_dupliates()` method and specifying that we want to drop based on the `AddressFullAddress` column.
# 
# **Warning:** For the most part, pandas methods don't have side effects, meaning that they won't affect the original DataFrame when we use them. Thus, when we use a method and want the changes to persist, we must save the result back to the DataFrame.

# In[ ]:


houses_2009 = houses_2009.drop_duplicates('AddressFullAddress')
houses_2013 = houses_2013.drop_duplicates('AddressFullAddress')
houses_2017 = houses_2017.drop_duplicates('AddressFullAddress')


# A few more words on `.loc`. We can slice our DataFrame using `.loc` and a boolean series, as before, but we can also use `.loc` to slice by passing which index values we want (row, column, or both). This looks like `df.loc[<rows>,<columns>]`

# In[ ]:


houses_2009.loc[100:105,['AddressFullAddress', 'AddressCity']]


# In[ ]:


houses_2009.loc[[1000, 2000, 3000], 'CouncilDistrict']


# **Exercise:** Use `.loc` to find the appraised value and the finished area of the house in row 50000.

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_105.py')


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


# **Exercise:** Create a bar chart showing the number of single-family homes by zip code for 2009.

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_106.py')


# In[ ]:





# If we want to display the distribution of a variable, we can use a histogram. For example, let's say we want to look at the distribution of square footages.

# In[ ]:


fig = houses_2009.FinishedArea.plot.hist(figsize = (10,4))
fig.set_title('Distribution of Homes by Square Footage', fontweight = 'bold');


# We get some extreme square footages - let's investigate.

# **Exercise:** Determine the number of homes in the `houses_2009` DataFrame that have finished area of at least 15,000 sqft. Which district contains the most number of these homes?

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_107.py')


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


# **Exercise:** Create a histograms showing distributions of appraisal values for homes in Madison and for homes in Brentwood in 2009. Plot both on the same figure.

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_108.py')


# **Harder Exercise:** Show only homes with appraisal values less than $1,000,000.
# Hint: To slice a DataFrame with `.loc` using multiple boolean series, separate the series with & for AND or | for OR. For example, to find all homes in Antioch that are at least 4,000 square feet, you can use `houses_2009.loc[(houses_2009.AddressCity == "ANTIOCH") & (houses_2009.FinishedArea >= 4000)]`

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_109.py')


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


# We can also aggregate multiple columns by passing a _dictionary_ to `.agg()`.
# 
# A python dictionary consists of key-value pairs. We can create a dictionary by enclosing the key-value pairs in squiggly brackets { }.
# 
# To aggregate on multiple columns, we pass a dictionary whose keys are the columns we wish to aggregate and whose values are the aggregation functions we want to use.

# In[ ]:


houses_2009.groupby('CouncilDistrict').agg({'TOTALAPPR':['mean', 'median'], 'FinishedArea': ['mean', 'median']})


# Notice how the column labels look different now. That is because when doing multiple aggregations the resulting DataFrame will now have a _MultiIndex_. It is also possible to have a row MultiIndex.

# In[ ]:


agg_df = houses_2009.groupby('CouncilDistrict').agg({'TOTALAPPR':['mean', 'median'], 'FinishedArea': ['mean', 'median']})
agg_df.columns


# We can also get a MultiIndex by grouping by multiple columns:

# In[ ]:


agg_df = houses_2009.groupby(['CouncilDistrict', 'AddressPostalCode']).TOTALAPPR.median()
agg_df


# To access entries, we can pass tuples.

# In[ ]:


agg_df.loc[25]


# In[ ]:


agg_df.loc[(25, 37205)]


# **Exercise:** Create a plot showing the median appraisal value by district for 2009.

# In[ ]:


# Your Code Here


# In[ ]:


# Or Load the Solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_110.py')


# If we have two metrics we want to compare on the same plot, we can use the `twinx()` method to have two different vertical scales.

# In[ ]:


fig, ax = plt.subplots(figsize = (12,5))
ax2 = ax.twinx()

width = 0.4

houses_2009.groupby('CouncilDistrict').TOTALAPPR.mean().plot.bar(color='olive', ax=ax, width=width, position=1, edgecolor = 'black', rot = 0)
houses_2009.groupby('CouncilDistrict').FinishedArea.mean().plot.bar(color='lightcoral', ax=ax2, width=width, position=0, edgecolor= 'black')

ax.set_ylabel('Median Appraisal Value', color = 'olive', fontweight = 'bold')
ax2.set_ylabel('Average Square Footage', color = 'lightcoral', fontweight = 'bold')

plt.xlim(-1,35)
plt.title('Housing Snapshot by District, 2009', fontweight = 'bold');


# In[ ]:


ACS = pd.read_csv('../input/census/ACS.csv')
ACS = ACS.set_index('district')
ACS.head()


# **Exercise:** the ACS.csv file contains the number of households and the median household income by council district, obtained from the US Census Bureau's American Community Survey. Create a side-by-side bar plot comparing median income by district to median home price in 2017.

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_111.py')


# A bar plot is not necessarily the best way to display this data. Perhaps a scatter plot might be easier to read.

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


# ### Part 3: Introduction to GeoPandas

# Let's create a map of these districts so we can get a better idea of where the supply of single-family homes is located.  
# 
# We will be using the geopandas library, which provides tools for working with geospatial data.

# In[ ]:


import geopandas as gpd


# We need to load in a shape file that includes the boundaries for the council districts. We will be using a geojson file obtained from https://data.nashville.gov/General-Government/Council-District-Outlines-GIS-/m4q4-q7tc

# In[ ]:


council_districts = gpd.read_file('../input/shapefiles/Council_District_Outlines.geojson')


# In[ ]:


council_districts.head()


# In[ ]:





# We can plot the council districts by calling `.plot()` on the GeoDataFrame.

# In[ ]:


council_districts.plot();


# To adjust the size of our plot, we can call `plt.subplots()` and specify the figsize. The matplotlib `subplots` function creates a matplotlib figure and axis. We need to specify that we want to create our plot on the axis that we created.

# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
council_districts.plot(ax = ax);


# In[ ]:





# What if we want to label these districts? To do this, we'll need coordinates for the center of each district. The shapely library provides a `representative_point()` method which will, given a (multi)polygon, return a point within that polygon. We can use this method on the geometry column of our DataFrame and save the result as a new column.

# In[ ]:


council_districts['coords'] = council_districts.geometry.map(lambda x: x.representative_point().coords[:][0])


# In[ ]:


council_districts.head()


# Now, we need to add a label onto our map for each row in the dataset.
# 
# One way to access the rows in a dataframe one at a time is by using the `.iterrows()`, method, which produces a generator object. This is an object that we can iterate through. We can call `next()` on a generator object in order to iterate through its contents.

# In[ ]:


rows = council_districts.iterrows()


# In[ ]:


next(rows)


# When we call `next` we get a tuple which gives the index of the row and its values. We can unpack this tuple to extract what we want:

# In[ ]:


idx, row = next(rows)
print(idx)
print(row)


# In[ ]:


row['district']


# We can also iterate through all of the rows using a `for` loop:

# In[ ]:


for idx, row in rows:
    print(row['district'])


# In[ ]:





# **Exercise:** Use `.iterrows()` along with `plt.annotate` to add district makers to our plot.

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_203.py')


# Link to more detailed map: http://maps.nashville.gov/webimages/MapGallery/PDFMaps/2018%20Council%20Members.pdf

# Some of these labels are in a less than ideal location. Look at, for example, District 11 on the northeast part of town.  
# To correct these, we can write a helper function to nudge those points into a slightly better location.

# In[ ]:


def shift_coord(district, amount, direction):
    old_coord = council_districts.loc[council_districts.district == district, 'coords'].values[0]
    if direction == 'up':
        new_coord = (old_coord[0], old_coord[1] + amount)
    if direction == 'down':
        new_coord = (old_coord[0], old_coord[1] - amount)
    if direction == 'left':
        new_coord = (old_coord[0] - amount, old_coord[1])
    if direction == 'right':
        new_coord = (old_coord[0] + amount, old_coord[1])
    council_districts.loc[council_districts.district == district, 'lng'] = new_coord[0]
    council_districts.loc[council_districts.district == district, 'lat'] = new_coord[1]

    council_districts.loc[council_districts.district == district, 'coords'] = council_districts.loc[council_districts.district == district, ['lng', 'lat']].apply(tuple, axis = 1) 


# In[ ]:


shift_coord(district='15', amount = 0.005, direction = 'left')
shift_coord(district='9', amount = 0.005, direction = 'down')
shift_coord(district='15', amount = 0.02, direction = 'down')
shift_coord(district='28', amount = 0.003, direction = 'down')
shift_coord(district='6', amount = 0.005, direction = 'down')
shift_coord(district='27', amount = 0.004, direction = 'left')
shift_coord(district='27', amount = 0.005, direction = 'down')
shift_coord(district='11', amount = 0.01, direction = 'down')
shift_coord(district='18', amount = 0.005, direction = 'down')
shift_coord(district='22', amount = 0.01, direction = 'down')
shift_coord(district='25', amount = 0.006, direction = 'down')
shift_coord(district='21', amount = 0.005, direction = 'right')
shift_coord(district='24', amount = 0.005, direction = 'right')
shift_coord(district='3', amount = 0.01, direction = 'down')
shift_coord(district='3', amount = 0.005, direction = 'left')
shift_coord(district='7', amount = 0.015, direction = 'down')


# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
council_districts.plot(ax = ax)
for idx, row in council_districts.iterrows():
    plt.annotate(s=row['district'], xy=row['coords'],
                 horizontalalignment='center', fontweight = 'bold')


# In[ ]:





# Now, we need to combine this data frame with the number of homes per district.

# In[ ]:


homes_per_district = houses_2009.CouncilDistrict.value_counts()
homes_per_district


# In[ ]:


type(homes_per_district)


# Just using the `.value_counts()` method returns a pandas Series. We can turn in into a DataFrame by using `.reset_index()` on it.

# In[ ]:


homes_per_district = homes_per_district.reset_index()
homes_per_district.head(5)


# We can merge the counts with our council districts DataFrame. To make this easier, we can rename the columns so that the columns containing the district numbers match.

# In[ ]:


homes_per_district.columns = ['district', 'num_homes_2009']
homes_per_district.head(2)


# Now we can use the pandas merge function. We need to specify the two DataFrames we with to merge using the `left` and `right` arguments. If we don't tell it otherwise, it will attempt to merge on all columns that appear in both dataframes (in our case, the `district` column).

# In[ ]:


pd.merge(left = council_districts, right = homes_per_district)


# Oops - we get an error. Pandas won't let us merge columns with different types. We can change the data type of the district column using the `astype` method. This will then allow us to merge the two dataframes.

# In[ ]:


council_districts.district = council_districts.district.astype(int)


# In[ ]:


council_districts = pd.merge(left = council_districts, right = homes_per_district)


#     After the merge, `council_districts` remains a GeoDataFrame.

# In[ ]:


type(council_districts)


# In[ ]:





# When we call .plot() on a GeoDataFrame, we can create a choropleth by specifying `column = <column_name>`. Here, we will color by number of homes.

# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
council_districts.plot(ax = ax, column = 'num_homes_2009')
for idx, row in council_districts.iterrows():
    plt.annotate(s=row['district'], xy=row['coords'],
                 horizontalalignment='center', fontweight = 'bold');


# We can add a legend explaining the meaning of the colors by specifying `legend = True`.

# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
council_districts.plot(ax = ax, column = 'num_homes_2009', legend = True)
for idx, row in council_districts.iterrows():
    plt.annotate(s=row['district'], xy=row['coords'],
                 horizontalalignment='center', fontweight = 'bold')


# This colormap is not necessarily the best. We can specify a different one using the cmap argument. 
# 
# See https://matplotlib.org/tutorials/colors/colormaps.html to see the colormap options. 
# 
# If you don't like any of those, it is also possible to create you own (but it takes some work to do so).

# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
council_districts.plot(ax = ax, column = 'num_homes_2009', legend = True, cmap = 'YlOrRd',edgecolor = 'grey')
for idx, row in council_districts.iterrows():
    plt.annotate(s=row['district'], xy=row['coords'],
                 horizontalalignment='center', fontweight = 'bold')


# We end up with an oddly-sized colormap. To modify it, we can use a couple of helper tools; namely, `make_axes_locatable` and `cm`.

# In[ ]:


from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))

council_districts.plot(ax = ax, column = 'num_homes_2009', cmap = 'YlOrRd', edgecolor = 'grey')

for idx, row in council_districts.iterrows():
    plt.annotate(s=row['district'], xy=row['coords'],
                 horizontalalignment='center', fontweight = 'bold', color = 'black')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

cmap = cm.ScalarMappable(
      norm = Normalize(council_districts.num_homes_2009.min(), council_districts.num_homes_2009.max()), 
      cmap = 'YlOrRd')
cmap.set_array([])    
fig.colorbar(mappable=cmap, cax = cax);


# To make it more readable, let's create another helper function that will adjust the color of the district label based on its background.
# 
# **Exercise:** Write a function called `choose_color` which returns 'black' if its input is less than 5000 and returns 'white' otherwise. Then adjust the above code to use your function to set the label colors.

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution 
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_202.py')


# In[ ]:





# Just a few more adjustments. We can add a title using `plt.title`. Within this function, we can also specify the font size and make the text bold.  
# 
# Second, since they are not really all that informative in this case, we might as well remove the axes by using `plt.axis('off')`.

# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))

council_districts.plot(ax = ax, column = 'num_homes_2009', cmap = 'YlOrRd', edgecolor = 'grey')

for idx, row in council_districts.iterrows():
    plt.annotate(s=row['district'], xy=row['coords'],
                 horizontalalignment='center', fontweight = 'bold', color = choose_color(row['num_homes_2009']))

plt.title('Number of Single-Family Homes by Council District, 2009', fontweight = 'bold', fontsize = 14)
plt.axis('off')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

cmap = cm.ScalarMappable(
      norm = Normalize(council_districts.num_homes_2009.min(), council_districts.num_homes_2009.max()), 
      cmap = 'YlOrRd')
cmap.set_array([])    
fig.colorbar(mappable=cmap, cax = cax);


# In[ ]:





# ## Adding Interstates

# So far, we have seen geometric objects in the form of (multi)polygons and points, but there are also lines.  
# 
# We can make our map a little easier to read by adding interstates. We will be using a shapefile of major roads obtained from https://catalog.data.gov/dataset/tiger-line-shapefile-2016-nation-u-s-primary-roads-national-shapefile/resource/94e763bb-78a9-48bb-8759-2c5c98508636.

# In[ ]:


interstates = gpd.read_file('../input/shapefiles/tl_2016_us_primaryroads.shp')


# In[ ]:


interstates.head()


# In[ ]:


interstates.plot()


# GeoDataFrames come equipped with a coordinate reference system, or crs. These have to do with the particular projection used to create the geometries.

# In[ ]:


print(interstates.crs)
print(council_districts.crs)


# Notice how the two GeoDataFrames we are using have different coordinate reference systems. We need to fix this by converting using the `to_crs()` method.

# In[ ]:


interstates = interstates.to_crs(council_districts.crs)


# Now we can narrow our interstates GeoDataFrame down to just those that intersect the council districts, using a spatial join. We'll keep only those highway segments that intersect the council districts.

# In[ ]:


interstates = gpd.sjoin(interstates, council_districts, how="inner", op='intersects')


# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
council_districts.plot(ax = ax)
for idx, row in council_districts.iterrows():
    plt.annotate(s=row['district'], xy=row['coords'],
                 horizontalalignment='center', fontweight = 'bold')
interstates.plot(color = 'black', ax = ax);


# Notice how the segment of I-40 extends further east than we need. We can fix this by specifying the x and y limits. The matplotlib functions `.xlim` and `.ylim` will either return the current values for the x and y range or can be used to specify new limits. Here, we will use them twice: once to get the limits before we plot the interstates and then again to reset the limits after plotting them.

# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))
council_districts.plot(ax = ax)
xlims = plt.xlim()
ylims = plt.ylim()
for idx, row in council_districts.iterrows():
    plt.annotate(s=row['district'], xy=row['coords'],
                 horizontalalignment='center', fontweight = 'bold')
interstates.plot(color = 'black', ax = ax)
plt.xlim(xlims)
plt.ylim(ylims);


# Now, we can combine this with our previous plot.

# In[ ]:


fig, ax = plt.subplots(figsize = (10,10))

council_districts.plot(ax = ax, column = 'num_homes_2009', cmap = 'YlOrRd', edgecolor = 'grey')

xlims = plt.xlim()
ylims = plt.ylim()

interstates.plot(color = 'black', ax = ax)
plt.xlim(xlims)
plt.ylim(ylims)


for idx, row in council_districts.iterrows():
    plt.annotate(s=row['district'], xy=row['coords'],
                 horizontalalignment='center', fontweight = 'bold', color = choose_color(row['num_homes_2009']))

plt.title('Number of Single-Family Homes by Council District, 2009', fontweight = 'bold', fontsize = 14)
plt.axis('off')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

cmap = cm.ScalarMappable(
      norm = Normalize(council_districts.num_homes_2009.min(), council_districts.num_homes_2009.max()), 
      cmap = 'YlOrRd')
cmap.set_array([])    
fig.colorbar(mappable=cmap, cax = cax); 


# **Exercise:** Create a choropleth showing the average square footage per district in 2009.  

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution 
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_204.py')


# In[ ]:





# ### Part 4: Adding Interactivity with ipywidgets

# What if we want to see how this changes over time?

# In[ ]:


homes_per_district_2013 = pd.DataFrame(houses_2013.CouncilDistrict.value_counts().sort_values()).reset_index()
homes_per_district_2013.columns = ['district', 'num_homes_2013']

homes_per_district_2017 = pd.DataFrame(houses_2017.CouncilDistrict.value_counts().sort_values()).reset_index()
homes_per_district_2017.columns = ['district', 'num_homes_2017']


# In[ ]:


council_districts = pd.merge(left = pd.merge(left = council_districts, right = homes_per_district_2013), right = homes_per_district_2017)


# In[ ]:





# **Exercise:** Create a function named `generate_map` which takes as input a year and produces a plot of number of homes per district for that year.

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_302.py')


# In[ ]:


generate_map(2013)


# In[ ]:





# Having a function to generate the map is great, but what if we didn't want to change the input parameter and rerun the cell each time we want to view a different year. Also, wouldn't it be nice to only be able to choose years for which we have data? We can accomplish this by using the `ipywidgets` library to create interactive plots.

# In[ ]:


from ipywidgets import interact


# ### `interact` as a decorator:
# 
# We will use `interact` as a **decorator**, which means that we will add it above our function definition with the @ symbol. That is, we need to add `@interact()` the line above `def`. Inside of `interact`, we need to specify the arguments to our function and what values they can take. `interact` is usually pretty good at figuring out a good default range, but you can also completely specify by using widgets.
# 
# For example, let's create an interactive widget using the squaring function. First, we need to define our function:

# In[ ]:


@interact(x=5)
def square(x):
    return x**2


# We can also create interactive widgets for functions with more than one argument:

# In[ ]:


@interact(x = 5, y = 5)
def sum_squares(x,y):
    return x**2 + y**2


# We don't just have to return values - we can also create plots:

# In[ ]:


@interact(k = [1/4, 1/3, 1/2, 1, 2, 3, 4])
def plot_power_function(k):
    xs = range(50)
    dynamic_ys = [x ** k for x in xs]
    plt.plot(xs, dynamic_ys)


# In[ ]:





# **Exercise:** Make the generate_map function interactive, allowing the user to select the year they want to plot.

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution 
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_303.py')


# In[ ]:





# This is pretty good, but notice how the colorscale changes across years. To make it easier to compare, it would be useful to have a fixed colorscale. To accomplish this, we can add the vmin and vmax arguments. Also, we need to redefine our choose_color function to accomodate the expanded colorscale.

# In[ ]:


def choose_color_scaled(num_homes, vmin, vmax):
    if num_homes < (vmin + vmax) / 2: return "black"
    return "white"


# In[ ]:


vmin = council_districts[['num_homes_2009', 'num_homes_2013', 'num_homes_2017']].values.min()
vmax = council_districts[['num_homes_2009', 'num_homes_2013', 'num_homes_2017']].values.max()

@interact(year = ['2009', '2013', '2017'])
def generate_map(year):
    fig, ax = plt.subplots(figsize = (10,10))
    column = 'num_homes_' + year
    

    council_districts.plot(ax = ax, column = column, cmap = 'YlOrRd', edgecolor = 'grey', vmin = vmin, vmax = vmax)

    xlims = plt.xlim()
    ylims = plt.ylim()
    interstates.plot(color = 'black', ax = ax)
    plt.xlim(xlims)
    plt.ylim(ylims)
    
    for idx, row in council_districts.iterrows():
        plt.annotate(s=row['district'], xy=row['coords'],
                     horizontalalignment='center', fontweight = 'bold', color = choose_color_scaled(row[column], vmin, vmax))

    plt.title(f'Number of Single-Family Homes by Council District, {year}', fontweight = 'bold', fontsize = 14)
    plt.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cmap = cm.ScalarMappable(
          norm = Normalize(vmin, vmax), 
          cmap = 'YlOrRd')
    cmap.set_array([])    
    fig.colorbar(mappable=cmap, cax = cax);   


# In[ ]:





# ## Analyzing changes in single-family housing supply

# **Exercise:** Create two new columns in the council_districts DataFrame, calculating the absolute and relative change in number of single-family homes from 2009 to 2017. Call these columns `absolute_change` and `relative_change`.

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_304.py')


# We can find the districts which had the largest and smallest change in housing supply by using the `sort_values()` method.

# In[ ]:


council_districts[['district', 'absolute_change']].sort_values('absolute_change').head()


# In[ ]:


council_districts[['district', 'absolute_change']].sort_values('absolute_change', ascending = False).head()


# In[ ]:


council_districts[['district', 'relative_change']].sort_values('relative_change', ascending = False).head()


# **Exercise:** Create an interactive plot showing absolute and relative change in the number of single-family homes by district.

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution 
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_305.py')


# In[ ]:





# ### Part 6: Appraisal Values

# In[ ]:


houses_2017.TOTALAPPR.describe()


# Where are the 5 most expensive houses? To answer this, we can use the `.nlargest()` method.

# In[ ]:


houses_2017.nlargest(n=5, columns='TOTALAPPR')


# In[ ]:





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
# **Workforce Housing:** Won't cost-burden households making between 60% and 120% of AMI.

# Let's classify these according to whether they are affordable to someone making 30%, 60%, 90%, or 120% of AMI. We will need to estimate the total yearly cost of each house. We can make this estimate based on its appraised value.

# In[ ]:


def find_mortgage_payment(TOTALAPPR,years = 30, rate = 4, down_payment = 20):
    P = TOTALAPPR * (1 - (down_payment / 100))
    n = 12 * years
    r = rate / (100 * 12)
    M = P * (r * (1 + r)**n) / ((1 + r)**n - 1)
    return M


# In[ ]:


houses_2009['est_mortgage_cost'] = houses_2009.TOTALAPPR.apply(lambda x: 12*find_mortgage_payment(x))
houses_2013['est_mortgage_cost'] = houses_2013.TOTALAPPR.apply(lambda x: 12*find_mortgage_payment(x))
houses_2017['est_mortgage_cost'] = houses_2017.TOTALAPPR.apply(lambda x: 12*find_mortgage_payment(x))


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


houses_2009['est_property_tax'] = houses_2009.apply(calculate_property_taxes, axis = 1)


# Oops! We get an error. It seems that some of the values in the DistrictCode have some extra white space. The string method `.strip()` removes any white space at the beginning or end of a sting. We can apply this to our DistrictCode column to correct the problem.

# In[ ]:


houses_2009.DistrictCode = houses_2009.DistrictCode.str.strip()
houses_2013.DistrictCode = houses_2013.DistrictCode.str.strip()
houses_2017.DistrictCode = houses_2017.DistrictCode.str.strip()


# In[ ]:


houses_2009['est_property_tax'] = houses_2009.apply(calculate_property_taxes, axis = 1)
houses_2013['est_property_tax'] = houses_2013.apply(calculate_property_taxes, axis = 1)
houses_2017['est_property_tax'] = houses_2017.apply(calculate_property_taxes, axis = 1)


# We also need to factor in insurance cost. We'll use \$60/month, or \$720/year as our estimate for homeowner's insurance.

# In[ ]:


houses_2009['est_yearly_cost'] = houses_2009.est_mortgage_cost + houses_2009.est_property_tax + 720
houses_2013['est_yearly_cost'] = houses_2013.est_mortgage_cost + houses_2013.est_property_tax + 720
houses_2017['est_yearly_cost'] = houses_2017.est_mortgage_cost + houses_2017.est_property_tax + 720


# Now that we have an estimated yearly cost, we can put each house into a category. We'll use 5 categories:
#  * __AFF_1:__ not cost-burdening to those making 30% of AMI
#  * __AFF_2:__ not cost-burdening to those making 60% of AMI
#  * __WF_1:__ not cost-burdening to those making 90% of AMI
#  * __WF_2:__ not cost-burdening to those making 120% of AMI
#  * __AWF:__ Requires more than 120% of AMI

# In[ ]:


def classify_house(value, AMI):
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


houses_2009['category'] = houses_2009.est_yearly_cost.apply(lambda x: classify_house(x, 64900))
houses_2013['category'] = houses_2013.est_yearly_cost.apply(lambda x: classify_house(x, 62300))
houses_2017['category'] = houses_2017.est_yearly_cost.apply(lambda x: classify_house(x, 68000))


# In[ ]:


plt.figure(figsize = (10,6))
houses_2017.category.value_counts()[['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']].plot.bar(rot = 0)
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

# First, we should combine our three DataFrames into one, using the pandas `concat` function. Start by adding a column to each DataFrame to record the year.

# In[ ]:


houses_2009['year'] = 2009
houses_2013['year'] = 2013
houses_2017['year'] = 2017


# To concatenate, we pass a list containing the DataFrames we want to combine to the function `pd.concat()`. The default (which is what we need) is to stack the DataFrames vertically, but we can also combine them horizontally by specifying `axis = 1`. 

# In[ ]:


houses = pd.concat([houses_2009, houses_2013, houses_2017])


# In[ ]:


houses.head()


# Need: A DataFrame with 5 rows (categories) and 3 columns (years).
# 
# **Exercise:** Group the `houses` DataFrame by category and by year and then count the number of homes per group. Save the result as a DataFrame called `category_count`.

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the soution 
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_402.py')


# This is close to what we need except that here all of the years are contained in one column; whereas, we need to have one column per year.
# 
# To show how we can accomplish this, let's first get our chess players DataFrame in the same format as our category_count DataFrame.

# In[ ]:


melted_chess = chess_players.reset_index().melt(id_vars=['player'], var_name = 'outcome')
melted_chess


# To get `melted_chess` back to the correct form, we can use the `.pivot()` method. To use this method, we have to specify which column will become our index, which column we want to split into our new columns, and which column will be used to assign values.

# In[ ]:


melted_chess.pivot(index='player', columns='outcome', values='value')


# **Exercise:** Use `.pivot` to get `category_count` into the right form to create a bar plot. Save the result back to `pivot_df`.

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_403.py')


# We can reorder our DataFrame using `.loc`:

# In[ ]:


pivot_df = pivot_df.loc[['AFF_1', 'AFF_2', 'WF_1', 'WF_2', 'AWF']]
pivot_df


# **Exercise:** Create a bar chart showing the number of single-family homes by affordability category for 2009, 2013, and 2017.

# In[ ]:


# Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_404.py')


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
# **Exercise:** By pivoting the `category_count` DataFrame, create one called `pivot_df_stack` which meets the above requirements. 

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_405.py')


# **Exercise:** Create a stacked bar chart showing the number of homes by category with year on the x-axis, including annotations.

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_406.py')


# **Exercise:** Create an interactive plot allowing the user to see the affordable housing profile for a chosen district.

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_407.py')


# In[ ]:





# ## Changes by District
# 
# Let's look at how much appraisal values increased from 2013 to 2017. First, let's calculate the percent change in median house value overall.

# In[ ]:


overall_pct_change = 100*(houses.loc[houses.year == 2017].TOTALAPPR.median() - houses.loc[houses.year==2013].TOTALAPPR.median()) / houses.loc[houses.year == 2013].TOTALAPPR.median()
overall_pct_change


# Now, let's look by council district.
# 
# **Exercise:** Create a DataFrame `median_appr` which contains the median appraisal value by district and by year.

# In[ ]:


#Your Code Here


# In[ ]:


# Or load the solution
get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_408.py')


# Now we can pivot this DataFrame to make the years into columns.

# In[ ]:


median_appr = median_appr.pivot(index = 'CouncilDistrict', columns='year', values='TOTALAPPR')
median_appr.head()


# In[ ]:


median_appr.columns


# Notice that our column index consists of integers.

# In[ ]:


median_appr['pct_change'] = 100*(median_appr[2017] - median_appr[2013]) / median_appr[2013]


# In[ ]:


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


get_ipython().run_line_magic('load', '../input/exercisesolutions/soln_409.py')


# In[ ]:





# # Bonus Material

# ### Part 7: Seaborn

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
    cd.loc[cd.district == district, 'chosen_district'] = 1
    
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





# ### Part 8: Plotly

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

