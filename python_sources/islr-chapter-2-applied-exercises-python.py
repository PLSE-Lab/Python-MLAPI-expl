#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Load the standard Python data science packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# # Applied Exercise 1
# 
# 
# **This exercise relates to the `College` data set, which can be found in the file `College.csv`. It contains a number of variables for 777 different universities and colleges in the US. The variables are**
# 
# - **`Private`: Public/private indicator**
# - **`Apps`: Number of applications received**
# - **`Accept`: Number of applicants accepted**
# - **`Enroll`: Number of new students enrolled**
# - **`Top10perc`: New students from top 10% of high school class**
# - **`Top25perc`: New students from top 25% of high school class**
# - **`F.Undergrad`: Number of full-time undergraduates**
# - **`P.Undergrad`: Number of part-time undergraduates**
# - **`Outstate`: Out-of-state tuition**
# - **`Room.Board`: Room and board costs**
# - **`Books`: Estimated book costs**
# - **`Personal`: Estimated personal spending**
# - **`PhD`: Percent of faculty with Ph.D.'s**
# - **`Terminal`: Percent of faculty with terminal degree**
# - **`S.F.Ratio`: Student/faculty ratio**
# - **`perc.alumni`: Percent of alumni who donate**
# - **`Expend`: Instructional expenditure per student**
# - **`Grad.Rate`: Graduation rate**
# 
# **Before reading the data into Pandas, it can be viewed in Excel or a text editor.**

# ## Part 1
# **Use the `pd.read_csv()` function to read the data into a Pandas dataframe. Call the loaded data `college`. Make sure that you have the directory set to the correct location for the data.**

# In[ ]:


college_filepath = "../input/ISLR-Auto/College.csv"
college = pd.read_csv(college_filepath)


# ## Part 2 
# **Look at the first few rows data using the `head()` function. You should notice that the first column is just the name of each university. We don't really want Pandas to treat this as data. However, it may be handy to have these names for later. Try the following commands:**
# 
# ```
# > college.set_index("Unnamed: 0", inplace = True)
# > college.head()
# ```

# In[ ]:


college.head()


# In[ ]:


college.set_index("Unnamed: 0", inplace = True)
college.head()


# **Now we see that each row of the dataframe is indexed by the name of the college. Note that the index column didn't have a header name in the original CSV file, so by default is given the name `Unnamed: N`, where `N` is determined by the number of other unnamed columns if we don't specify column names using the `names` argument of `read_csv()`. To give it a more descriptive name, we'll use the `rename_axis()` function.**

# In[ ]:


#Rename the index to be more descriptive
college.rename_axis(index = "College", inplace = True)
college.head()


# **Now you should see that the first data column is `Private`. Note that we could have also taken care of this pre-processing when reading the original CSV file using the `index_col` argument in the `read_csv()` function.**

# In[ ]:


# Have Pandas treat the zeroth column in the CSV file as the index
# Then give the index a more descriptive name.
pd.read_csv(college_filepath, index_col = 0).rename_axis(index = "College").head()


# Before moving on, do the good practice of checking the data for any null values.

# In[ ]:


college.isnull().any()


# It looks like there aren't any missing or null values, so we are good to continue to the remaining parts of this exercise.

# ## Part 3.1
# **Use the `describe()` function to produce a numerical summary of the variables in the data set.**

# In[ ]:


# Generate numerical summary of all of the numerical variables in the college data set
college.describe()


# In[ ]:


# Since the private column is a categorical variable, it's more readable to describe it separately using groupby and count
college["Private"].groupby(by = college["Private"]).count()


# ## Part 3.2
# **Use the `pairplot()` function from Seaborn to produce a scatterplot matrix of the first ten numeric columns or variables of the data (`Apps` through `Books`). Recall that you can reference the first ten columns of a dataframe `A` using `A.loc[:, "Col_1":"Col_10"]` where `Col_i` is the name of the ith column. If we want to use purely integer-based indexing, then we would use `A.iloc[:, 1:11]`. Don't forget that label-based indexing includes the stop column, while integer-based indexing does not.**

# In[ ]:


# Generating the scatterplot matrix using label-based indexing
sns.pairplot(college.loc[:, "Apps":"Books"])


# In[ ]:


# Generating the scatterplot matrix using integer-based indexing
sns.pairplot(college.iloc[:, 1:11])


# ## Part 3.3
# **Use the `catplot()` function to produce side-by-side boxplots of `Outstate` versus `Private`.**

# In[ ]:


ax = sns.catplot(x = "Private", y = "Outstate", kind = "box", order = ["Yes", "No"], data = college)
# Seaborn returns an axis object, so we can set the label for the y-axis to be more descriptive
ax.set(ylabel = "Out-of-state tuition (dollars)")
plt.show()


# ## Part 3.4
# **Create a new qualitative variable, called `Elite`, by *binning* the `Top10perc` variable. We are going to divide universities into two groups based on whether or not the proportion of students coming from the top 10% of their high school classes exceeds 50%.**

# In[ ]:


# Create a new column called Elite and set the default value as "No"
college["Elite"] = "No"
# Select all rows (i.e. schools) with over 50% of their students coming from the top 10% of their high school class
# Set the value of the Elite column for those schools to "Yes"
college.loc[college["Top10perc"] > 50, "Elite"] = "Yes"


# **Use the `groupby()` and `count()` functions to see how many elite universities there are. Now use the `catplot()` function to produce side-by-side boxplots of `Outstate` versus `Elite`.**

# In[ ]:


# Take the Elite column of the college data set, group by its values, and count the occurrences of each value
college["Elite"].groupby(by = college["Elite"]).count()


# In[ ]:


ax = sns.catplot(x = "Elite", y = "Outstate", kind = "box", order = ["Yes", "No"], data = college)
ax.set(ylabel = "Out-of-state tuition (dollars)")
plt.show()


# ## Part 3.5
# **Use the `distplot()` function to produce some histograms with differing numbers of bins for a few of the quantitative variables. You may find the command `plt.subplots(2, 2)` useful: it will divide the print window into four regions so that four plots can be made simultaneously. Modifying the arguments to this function will divide the figure in other ways.**

# In[ ]:


# Create grid of plots (fig)
# ax will be an array of four Axes objects
# Set the figure size so the plots aren't all squished together
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))

# Create histogram for number of applicants across all colleges
sns.distplot(college["Apps"], kde = False, ax = axes[0, 0])
axes[0, 0].set(xlabel = "", title = "All colleges")

# Create histogram for number of applicants at private colleges
sns.distplot(college.loc[college["Private"] == "Yes", "Apps"], kde = False, ax = axes[0, 1])
axes[0, 1].set(xlabel = "", title = "Private schools")

# Create histogram for number of applicants at elite colleges
sns.distplot(college.loc[college["Elite"] == "Yes", "Apps"], kde = False, ax = axes[1, 0])
axes[1, 0].set(xlabel = "", title = "Elite schools")

# Create histogram for number of applicants at public colleges
sns.distplot(college.loc[college["Private"] == "No", "Apps"], kde = False, ax = axes[1, 1])
axes[1, 1].set(xlabel = "", title = "Public schools")

fig.suptitle("Histograms of number of applicants by school type")


# In[ ]:


# Generate numerical summary of applicants by public vs private school
college["Apps"].groupby(by = college["Private"]).describe()


# In[ ]:


# Generate numerical summary of applicants by elite vs non-elite school
college["Apps"].groupby(by = college["Elite"]).describe()


# In[ ]:


# Create grid of plots (fig)
# ax will be an array of four Axes objects
# Set the figure size so the plots aren't all squished together
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))

# Create histogram for instructional expenditure per student across all colleges
sns.distplot(college["Expend"], kde = False, ax = axes[0, 0])
axes[0, 0].set(xlabel = "", title = "All colleges")

# Create histogram for instructional expenditure per student at private colleges
sns.distplot(college.loc[college["Private"] == "Yes", "Expend"], kde = False, ax = axes[0, 1])
axes[0, 1].set(xlabel = "", title = "Private schools")

# Create histogram for instructional expenditure per student at elite colleges
sns.distplot(college.loc[college["Elite"] == "Yes", "Expend"], kde = False, ax = axes[1, 0])
axes[1, 0].set(xlabel = "", title = "Elite schools")

# Create histogram for instructional expenditure per student at public colleges
sns.distplot(college.loc[college["Private"] == "No", "Expend"], kde = False, ax = axes[1, 1])
axes[1, 1].set(xlabel = "", title = "Public schools")

fig.suptitle("Histograms of instructional expenditure (USD) per student by school type")


# In[ ]:


# Generate numerical summary of instructional expenditure per student by public vs private schools
college["Expend"].groupby(by = college["Private"]).describe()


# In[ ]:


# Generate numerical summary of instructional expenditure per student by elite vs non-elite schools
college["Expend"].groupby(by = college["Elite"]).describe()


# In[ ]:


# Create grid of plots (fig)
# ax will be an array of four Axes objects
# Set the figure size so the plots aren't all squished together
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))

# Create histogram for student-faculty ratio across all colleges
sns.distplot(college["S.F.Ratio"], kde = False, ax = axes[0, 0])
axes[0, 0].set(xlabel = "", title = "All colleges")

# Create histogram for student-faculty ratio at private colleges
sns.distplot(college.loc[college["Private"] == "Yes", "S.F.Ratio"], kde = False, ax = axes[0, 1])
axes[0, 1].set(xlabel = "", title = "Private schools")

# Create histogram for student-faculty ratio at elite colleges
sns.distplot(college.loc[college["Elite"] == "Yes", "S.F.Ratio"], kde = False, ax = axes[1, 0])
axes[1, 0].set(xlabel = "", title = "Elite schools")

# Create histogram for student-faculty ratio at public colleges
sns.distplot(college.loc[college["Private"] == "No", "S.F.Ratio"], kde = False, ax = axes[1, 1])
axes[1, 1].set(xlabel = "", title = "Public schools")

fig.suptitle("Histograms of student-faculty ratio by school type")


# In[ ]:


# Generate numerical summary of student-faculty ratio by public vs private schools
college["S.F.Ratio"].groupby(by = college["Private"]).describe()


# In[ ]:


# Generate numerical summary of student-faculty ratio by elite vs non-elite schools
college["S.F.Ratio"].groupby(by = college["Elite"]).describe()


# ## Part 3.6
# **Continue exploring the data, and provide a brief summary of what you discover.**

# In[ ]:


# Make a column for non-tuition costs (room and board, books, and personal)
college["NonTuitionCosts"] = college["Room.Board"] + college["Books"] + college["Personal"]


# In[ ]:


# Side-by-side boxplots for public vs private schools
ax = sns.catplot(x = "Private", y = "NonTuitionCosts", kind = "box", order = ["Yes", "No"],data = college)
ax.set(ylabel = "Total non-tuition costs per year (dollars)")
plt.show()


# In[ ]:


# Generate numerical summary of non-tuition costs by public vs private schools
college["NonTuitionCosts"].groupby(by = college["Private"]).describe()


# In[ ]:


# Side-by-side boxplots for elite vs non-elite schools
ax = sns.catplot(x = "Elite", y = "NonTuitionCosts", kind = "box", order = ["Yes", "No"], data = college)
ax.set(ylabel = "Total non-tuition costs per year (dollars)")
plt.show()


# In[ ]:


# Generate numerical summary of non-tuition costs by elite vs non-elite schools
college["NonTuitionCosts"].groupby(by = college["Elite"]).describe()


# Based on the above box plots, it looks like that, aside from some outlier schools with very high costs, there isn't a wide gap for the median non-tution costs between private schools and public schools. The box plots do show, though, that there is a distinct difference in median non-tuition costs between elite and non-elite schools, with elite schools having higher costs.

# In[ ]:


# Make a column for the acceptance rate of each school
college["AcceptPerc"] = college["Accept"] / college["Apps"] * 100


# In[ ]:


# Side-by-side boxplots for public vs private schools
ax = sns.catplot(x = "Private", y = "AcceptPerc", kind = "box", order = ["Yes", "No"], data = college)
ax.set(ylabel = "Percent of applicants accepted")
plt.show()


# In[ ]:


# Generate numerical summary of acceptance rates by public vs private schools
college["AcceptPerc"].groupby(by = college["Private"]).describe()


# In[ ]:


# Side-by-side boxplots for elite vs non-elite schools
ax = sns.catplot(x = "Elite", y = "AcceptPerc", kind = "box", order = ["Yes", "No"], data = college)
ax.set(ylabel = "Percent of applicants accepted")
plt.show()


# In[ ]:


# Generate numerical summary of acceptance rates by elite vs non-elite schools
college["AcceptPerc"].groupby(by = college["Elite"]).describe()


# The boxplots show that while the median acceptance rates for both private and public schools are pretty close at around 75-80%, private schools have a much wider range of acceptance rates (going down to a minimum of 15.45%). When we distinguish between elite and non-elite schools, elite schools have a much lower median acceptance rate compared to non-elite ones.

# In[ ]:


# Create grid of plots (fig)
# ax will be an array of four Axes objects
# Set the figure size so the plots aren't all squished together
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 10))

# Create histogram for percent of alumni who donate across all colleges
sns.distplot(college["perc.alumni"], kde = False, ax = axes[0, 0])
axes[0, 0].set(xlabel = "", title = "All colleges")

# Create histogram for percent of alumni who donate at private colleges
sns.distplot(college.loc[college["Private"] == "Yes", "perc.alumni"], kde = False, ax = axes[0, 1])
axes[0, 1].set(xlabel = "", title = "Private schools")

# Create histogram for percent of alumni who donate at elite colleges
sns.distplot(college.loc[college["Elite"] == "Yes", "perc.alumni"], kde = False, ax = axes[1, 0])
axes[1, 0].set(xlabel = "", title = "Elite schools")

# Create histogram for percent of alumni who donate at public colleges
sns.distplot(college.loc[college["Private"] == "No", "perc.alumni"], kde = False, ax = axes[1, 1])
axes[1, 1].set(xlabel = "", title = "Public schools")

fig.suptitle("Histograms of percent of alumni who donate by school type")


# In[ ]:


# Generate numerical summary of percent of alumni who donate by public vs private schools
college["perc.alumni"].groupby(by = college["Private"]).describe()


# In[ ]:


# Generate numerical summary of percent of alumni who donate by elite vs non-elite schools
college["perc.alumni"].groupby(by = college["Elite"]).describe()


# Based on the above histograms, private schools and elite schools tend to have a higher percent of alumni who donate.

# Lastly, I explore some factors which might be related to graduation rates.

# In[ ]:


# Side-by-side boxplots for public vs private schools
ax = sns.catplot(x = "Private", y = "Grad.Rate", kind = "box", order = ["Yes", "No"], data = college)
ax.set(ylabel = "Graduation rate")
plt.show()


# In[ ]:


# Generate numerical summary of graduation rate by public vs private schools
college["Grad.Rate"].groupby(by = college["Private"]).describe()


# First, we note that while the range of graduation rates is about the same between public and private schools (I think the point with a value of 118 is a typo, since it is impossible to have a graduation rate that is above 100%), private schools generally have higher graduation rates than public ones.

# In[ ]:


# Side-by-side boxplots for elite vs non-elite schools
ax = sns.catplot(x = "Elite", y = "Grad.Rate", kind = "box", order = ["Yes", "No"], data = college)
ax.set(ylabel = "Graduation rate")
plt.show()


# In[ ]:


# Generate numerical summary of graduation rate by elite vs non-elite schools
college["Grad.Rate"].groupby(by = college["Elite"]).describe()


# The difference is even more striking when comparing elite and non-elite schools. Th minimum graduation rate among elite schools is almost exactly equal to the first quartile for non-elite schools.

# In[ ]:


# Create pair of scatter plots analyzing the relationship between number of faculty with PhDs and graduation rates
# Include least squares regression lines to help distinguish between different facets of the data
# Use columns to distinguish between elite and non-elite schools
# Use hue to distinguish between public and private schools
g = sns.lmplot(x = "PhD", y = "Grad.Rate", hue = "Private", col = "Elite", col_order = ["Yes", "No"],
               markers = ["o", "x"], data = college)
g.set(xlabel = "Number of faculty with PhDs", ylim = (0, 120))
plt.show()


# Next, we look at the the relationship between number of faculty with PhDs and graduation rates. There is a pretty clear positive relationship between the two for elite schools, and while the relationship is weaker for non-elite ones, there still appears to be one. Also, there seems to be a sharper increase in graduation rates per additional faculty member with a PhD for private schools when compared to public schools.

# In[ ]:


# Create pair of scatter plots analyzing the relationship between number of faculty with terminal degrees and graduation rates
# Use columns to distinguish between elite and non-elite schools
# Include least squares regression lines to help distinguish between different facets of the data
# Use hue to distinguish between public and private schools
g = sns.lmplot(x = "Terminal", y = "Grad.Rate", hue = "Private", col = "Elite", col_order = ["Yes", "No"],
                markers = ["o", "x"], data = college)
g.set(xlabel = "Number of faculty with terminal degrees", ylim = (0, 120))
plt.show()


# The trend is similar for the relationship between number of faculty with terminal degrees at elite schools, though for non-elite schools the relationship appears to be even weaker. This especially appears to be the case for non-elite public schools.

# In[ ]:


# Create pair of scatter plots analyzing the relationship between student-faculty ratio and graduation rates
# Include least squares regression lines to help distinguish between different facets of the data
# Use columns to distinguish between elite and non-elite schools
# Use hue to distinguish between public and private schools
g = sns.lmplot(x = "S.F.Ratio", y = "Grad.Rate", hue = "Private", col = "Elite", col_order = ["Yes", "No"],
                markers = ["o", "x"], data = college)
g.set(xlabel = "Student-faculty ratio", ylim = (0, 120))
plt.show()


# Next we move on to looking at the relationship between student-facult ratio and graduation rate. We can see generally negative relationships between the two variables for both elite and non-elite schools. In other words, as the ratio increases, graduation rates tend to decrease. The relationship seems to have a steeper slope for non-elite private schools, while there doesn't appear to be much of a relationship at all for non-elite public schools.

# In[ ]:


# Create pair of scatter plots analyzing the relationship between instructional expenditure per student and graduation rates
# Include least squares regression lines to help distinguish between different facets of the data
# Use columns to distinguish between elite and non-elite schools
# Use hue to distinguish between public and private schools
g = sns.lmplot(x = "Expend", y = "Grad.Rate", hue = "Private", col = "Elite", col_order = ["Yes", "No"], 
               markers = ["o", "x"], data = college)
g.set(xlabel = "Instructional expenditure per student (USD)", ylim = (0, 120))
plt.show()


# Lastly, we look at the relationship between instructional expenditure per student. For both elite and non-elite schools, there is a clear positive relationship between instructional expenditure per student and graduation rate, though the effect does flatten out after a certain point. The slope seems to be steeper for non-elite public schools compared to non-elite private schools. Also, there is a clear outlier and high leverage point: a non-elite private school with a very low graduation rate compared to its instructional expenditure per student.

# # Applied Exercise 2
# 
# **This exercise involves the `Auto` data set studied in the lab. Make sure that the missing values have been removed from the data.**

# In[ ]:


# Create variable for the name of the file containing the Auto data set
auto_filename = "../input/ISLR-Auto/Auto.csv"
# Load the Auto data set into a Pandas dataframe, treating question marks as na values
auto = pd.read_csv(auto_filename, na_values = ["?"])
# Drop the rows which contain missing values (safe to do since we've worked with this data in a previous lab)
auto.dropna(inplace = True)
# Check the dimensions of the dataframe
auto.shape


# ## Part 1
# **Which of the predictors are quantitative, and which are qualitative?**

# In[ ]:


auto.head()


# The quantitative variables are `mpg`, `displacement`, `horsepower`, `weight`, and `acceleration`. Depending on the context, we may want to treat `cylinders` and `year` as quantitative predictors or qualitative ones. Lastly, `origin` and `name` are qualitative predictors. `origin` is a quantitative encoding of a car's country of origin, where 1 being American, 2 being European, and 3 being Japanese.

# ## Part 2
# **What is the *range* of each quantitative predictor? You can answer this using the `max()` and `min()` functions.**

# In[ ]:


# Range = max - min
# Use the max() and min() functions on just the numeric data
# The argument axis = 0 means that we compute the max/min along each index
auto_max = auto.loc[:, "mpg":"year"].max(axis = 0)
auto_min = auto.loc[:, "mpg":"year"].min(axis = 0)
auto_range = auto_max - auto_min
# Generate a dataframe with the max, min, and range for each quantitative variable
pd.DataFrame({"max":auto_max, "min":auto_min, "range":auto_range})


# We have the following ranges for each quantitative predictor:
# 
# - `mpg` = 37.6
# - `cylinders` = 5
# - `displacement` = 387
# - `horsepower` = 184
# - `weight` = 3527
# - `acceleration` = 16.8
# - `year` = 12

# ## Part 3
# **What is the mean and standard deviation of each quantitative predictor?**

# In[ ]:


# Compute mean of each quantitative variable
auto_mean = auto.loc[:, "mpg":"year"].mean(axis = 0)
# Compute standard deviation of each quantitative variable
auto_sd = auto.loc[:, "mpg":"year"].std(axis = 0)
# Generate a dataframe with the mean and standard deviation of each quantitative predictor
# Note that I also could have used the describe() function as well
pd.DataFrame({"mean":auto_mean, "std dev":auto_sd})


# We have the following mean and standard deviation for each quantitative predictor:
# 
# - `mpg`: mean = 23.45, standard deviation = 7.81
# - `cylinders`: mean = 5.47, standard deviation = 1.71
# - `displacement`: mean = 194.41, standard deviation = 104.64
# - `horsepower`: mean = 104.47, standard deviation = 38.49
# - `weight`: mean = 2977.58, standard deviation = 849.40
# - `acceleration`: mean = 15.54, standard deviation = 2.76
# - `year`: mean = 75.98, standard deviation = 3.68

# ## Part 4
# **Now remove the 10th through 85th observations. What is the range, mean, and standard deviation of each predictor in the subset of the data that remains?**

# In[ ]:


# Reset the index of the auto data frame
auto.reset_index(drop = True, inplace = True)
# Create dataframe in which the 10th through 85th observations are dropped
# Don't forget that Pandas dataframes are zero-indexed
auto_dropped = auto.drop(index = list(range(9, 85)))
# Compute max, min, range, mean, and standard deviation for each quantitative variable
dropped_max = auto_dropped.loc[:, "mpg":"year"].max(axis = 0)
dropped_min = auto_dropped.loc[:, "mpg":"year"].min(axis = 0)
dropped_range = dropped_max - dropped_min
dropped_mean = auto_dropped.loc[:, "mpg":"year"].mean(axis = 0)
dropped_sd = auto_dropped.loc[:, "mpg":"year"].std(axis = 0)
# Generate a dataframe with the max, min, range, mean, and standard deviation for each quantitative variable
# Again note that the describe() function would provide all of these values except for the range
pd.DataFrame({"max":dropped_max, "min":dropped_min, "range":dropped_range, "mean":dropped_mean, "std dev":dropped_sd})


# We have the following range, mean,standard deviation for each quantitative predictor after the 10th through 85th rows have been removed:
# 
# - `mpg`: range = 35.6, mean = 24.40, standard deviation = 7.87
# - `cylinders`: range = 5, mean = 5.37, standard deviation = 1.65
# - `displacement`: range = 387, mean = 187.24, standard deviation = 99.68
# - `horsepower`: range = 184, mean = 100.72, standard deviation = 35.71
# - `weight`: range = 3348, mean = 2935.97, standard deviation = 811.30
# - `acceleration`: range = 16.3, mean = 15.73, standard deviation = 2.69
# - `year`: mean = 77.15, standard deviation = 3.11

# ## Part 5
# **Using the full data set, investigate the predictors graphically, using scatterplots or other tools of your choice. Create some plots highlighting the relationships among the predictors. Comment on your findings.**

# In[ ]:


# Convert the origin column from numerical codes to the meanings of each code
# 1 = American, 2 = European, 3 = Japanese
origin_dict = {1: "American", 2: "European", 3: "Japanese"}
auto["origin"] = auto["origin"].transform(lambda x: origin_dict[x]).astype("category")


# In[ ]:


# Create scatter plot for the relationship between engine displacement and mpg
# Use hue to highlight the origin of each car
g = sns.relplot(x = "displacement", y = "mpg", hue = "origin", data = auto)
g.set(xlabel = "Engine displacement (cubic inches)")
plt.show()


# In[ ]:


# Create scatter plot for the relationship between horsepower and mpg
# Use hue to highlight the origin of each car
g = sns.relplot(x = "horsepower", y = "mpg", hue = "origin", data = auto)
plt.show()


# In[ ]:


# Create scatter plot for the relationship between car weight and mpg
# Use hue to highlight the origin of each car
g = sns.relplot(x = "weight", y = "mpg", hue = "origin", data = auto)
g.set(xlabel = "Car weight (pounds)")
plt.show()


# In[ ]:


# Create scatter plot for the relationship between model year and mpg
# Use hue to highlight the origin of each car
g = sns.relplot(x = "year", y = "mpg", hue = "origin", data = auto)
g.set(xlabel = "Model year")
plt.show()


# In[ ]:


# Alternatively use pairplot to create scatterplots relating mpg to engine displacement, horsepower,
# car weight, and car manufacture year
# Use hue to highlight the origin of each car
g = sns.pairplot(auto, hue = "origin", y_vars = ["mpg"], x_vars = ["displacement", "horsepower", "weight", "year"],
                height = 5)


# See discussion in Part 6 below.

# In[ ]:


# Create scatter plot for the relationship between model year and acceleration
# Use hue to highlight the origin of each car
g = sns.relplot(x = "year", y = "acceleration", hue = "origin", data = auto)
g.set(xlabel = "Model year", ylabel = "0 to 60mph time (seconds)")
plt.show()


# In[ ]:


# Create scatter plot for the relationship between model year and engine displacement
# Use hue to highlight the origin of each car
g = sns.relplot(x = "year", y = "displacement", hue = "origin", data = auto)
g.set(xlabel = "Model year", ylabel = "Engine displacement (cubic inches)")
plt.show()


# In[ ]:


# Create scatter plot for the relationship between model year and car weight
# Use hue to highlight the origin of each car
g = sns.relplot(x = "year", y = "weight", hue = "origin", data = auto)
g.set(xlabel = "Model year", ylabel = "Car weight (pounds)")
plt.show()


# In[ ]:


# Create scatter plot for the relationship between model year and horsepower
# Use hue to highlight the origin of each car
g = sns.relplot(x = "year", y = "horsepower", hue = "origin", data = auto)
g.set(xlabel = "Model year", ylabel = "horsepower")
plt.show()


# Looking at how various car characteristics change with model year, we see that there aren't any strong relationships. There are still some weak relationships, such as max engine displacement, car weight, and horsepower generally decreasing from 1970 to 1982. From a historical perspective, these changes could be in response to the 1973 and 1979 oil crises, in which spikes in oil prices pushed auto manufacturers to take measures to improve the efficiency of their cars.

# In[ ]:


# Create scatter plot for the relationship between car weight and acceleration
# Use hue to highlight the origin of each car
g = sns.relplot(x = "weight", y = "acceleration", hue = "origin", data = auto)
g.set(xlabel = "Car weight (pounds)", ylabel = "0 to 60mph time (seconds)")
plt.show()


# In[ ]:


# Create scatter plot for the relationship between engine displacement and acceleration
# Use hue to highlight the origin of each car
g = sns.relplot(x = "displacement", y = "acceleration", hue = "origin", data = auto)
g.set(xlabel = "Engine displacement (cubic inches)", ylabel = "0 to 60mph time (seconds)")
plt.show()


# In[ ]:


# Create scatter plot for the relationship between horsepower and acceleration
# Use hue to highlight the origin of each car
g = sns.relplot(x = "horsepower", y = "acceleration", hue = "origin", data = auto)
g.set(xlabel = "Horsepower", ylabel = "0 to 60mph time (seconds)")
plt.show()


# In[ ]:


# Create swarm plot for the relationship between number of engine cylinders and acceleration
# Use hue to highlight the origin of each car
g = sns.catplot(x = "cylinders", y = "acceleration", hue = "origin", data = auto, kind = "swarm")
g.set(xlabel = "Number of engine cylinders", ylabel = "0 to 60mph time (seconds)")
plt.show()


# In[ ]:


# Alternatively use pairplot to create scatterplots relating acceleration to engine displacement, horsepower,
# car weight
# Use hue to highlight the origin of each car
g = sns.pairplot(auto, hue = "origin", y_vars = ["acceleration"], x_vars = ["displacement", "horsepower", "weight"],
                height = 5)


# Next, I explored the relationship between the number of seconds it takes a car to accelerate from 0 to 60 miles per hour and a number of different factors. As expected, the 0-to-60 time clearly decreases with increased engine displacement and increased horsepower. There is also a weak relationship that as the number of engine cylinders increases the 0-to-60 time tends to decrease. While it may seem counter-intuitive at first, the 0-to-60 time also tends to decrease with car weight. This makes more sense in the context of the two scatterplots below, which shows that the higher weight is correlated with higher horsepower and higher engine displacement.

# In[ ]:


# Create scatter plot for the relationship between car weight and horsepower
# Use hue to highlight the origin of each car
g = sns.relplot(x = "weight", y = "horsepower", hue = "origin", data = auto)
g.set(xlabel = "Car weight (pounds)", ylabel = "Horsepower")
plt.show()


# In[ ]:


# Create scatter plot for the relationship between car weight and engine displacement
# Use hue to highlight the origin of each car
g = sns.relplot(x = "weight", y = "displacement", hue = "origin", data = auto)
g.set(xlabel = "Car weight (pounds)", ylabel = "Engine displacement (cubic inches)")
plt.show()


# ## Part 6
# **Suppose we wish to predict gas mileage (`mpg`) on the basis of the other variables. Do your plots suggest that any of the other variables might be useful in predicting `mpg`? Justify your answer.**

# Based on the scatter plots I made in part 5 which relate miles per gallon to the predictors engine displacement, horsepower, car weight, and model year, it seems as if the first three factors would be most helpful in predicting `mpg`, with model year still potentially being helpful but less so. There are clear relationships that increasing engine displacement/horsepower/car weight results in decreased fuel efficiency. There is also a weak relationship that fuel efficiency generally increased going from 1970 to 1982.

# In[ ]:


# Create box plot comparing the fuel effiency of American, European, and Japanese cars
g = sns.catplot(x = "origin", y = "mpg", data = auto, kind = "box")


# Looking at the above box plot, we can also see that there is a relationship between a car's country of origin and fuel efficiency, where on average Japanese cars are the most efficient, followed by European cars and then by American cars. Looking at the numerical summary below also confirms this and provides some additional insight. As we might have noticed in the scatter plots relating mpg to various other factors, Japanese cars tend to have engines with lower displacement that produce less power. The numerical summary below indicates that the average displacement and average horsepower for Japanese cars are quite close to the corresponding values for European cars. Moreover, Japanese cars are on average about 200 pounds lighter than European cars, and about 1150 pounds lighter than American cars. Especially when combined with the effects engine displacement, horsepower, and car weight appear to have on fuel efficiency, as noted above, it seems that the fact that the Japanese cars in this data set are lightweight cars with small and low-powered engines is the reason why those cars are generally more fuel efficient than American and European ones.

# In[ ]:


auto.loc[:, "mpg":"acceleration"].groupby(auto["origin"]).agg(["mean", "std", "min", "median", "max"]).T


# # Applied Exercise 3
# 
# **This exercise involves the `Boston` housing data set.**

# ## Part 1
# **To begin, load the `Boston` data set. The `Boston` data set is part of the `MASS` *library* in `R`.**

# **Note** Instead of using the `Boston` data set found in the `MASS` library in `R`, I will instead be using the corrected Boston data set, which can be downloaded [here](http://lib.stat.cmu.edu/datasets/boston_corrected.txt).

# In[ ]:


# Create variable for corrected Boston dataset file name
boston_filename = "../input/corrected-boston-housing/boston_corrected.csv"
# Load the data into a Pandas dataframe
# Create a multi-index on the TOWN and TRACT columns
boston = pd.read_csv(boston_filename, index_col= ["TOWN", "TRACT"])
boston.head()


# In[ ]:


boston.shape


# **How many rows are in this data set? How many columns? What do the rows and columns represent?**

# The corrected Boston data set has 506 rows and 20 columns. Each row represents a particular tract of land within the city of Boston. The dataset has the following columns.
# 
# - `TOWN`: Name of the town in which the tract is located
# - `TOWNNO`: Numeric code corresponding to the town
# - `TRACT`: ID number of the tract of land
# - `LON`: Longitude of the tract in decimal degrees
# - `LAT`: Latitude of the tract in decimal degrees
# - `MEDV`: Median value of owner-occupied housing in \\$1000 for the tract
# - `CMEDV`: Corrected median value of owner occupied housing in \\$1000 for the tract, since the original values in MEDV were censored in the sense that all median values at or over \\$50000 are set to \\$50000
# - `CRIM`: Per capita crime rate for the tract
# - `ZN`: Percent of residential land zoned for lots over 25000 square feet per town (constant for all tracts within the same town)
# - `INDUS`: Percent of non-retail business acres per town (constant for all tracts within the same town)
# - `CHAS`: Dummy variable to indicate whether or not the tract borders the Charles River (1 = Borders Charles River, 0 = Otherwise)
# - `NOX`: Nitric oxides concentration (in parts per 10 million) per town (constant for all tracts within the same town)
# - `RM`: Average number of rooms per dwelling in the tract
# - `AGE`: Percent of owner-occupied units in the tract built prior to 1940 
# - `DIS`: Weighted distance from the tract to five Boston employment centers
# - `RAD`: Index of accessibility to radial highways per town (constant for all tracts within the same town)
# - `TAX`: Full-value property tax rate per \\$10000 per town (constant for all tracts within the same town)
# - `PTRATIO`: Pupil-teacher ratio per town (constant for all tracts within the same town)
# - `B`: $1000(B - 0.63)^2$, where $B$ is the proportion of black residents in the tract
# - `LSTAT`: Percent of tract population designated as lower status

# In[ ]:


# Since I won't be using them, I'll drop the TOWNNO, LON, LAT, and MEDV columns
boston.drop(columns = ["TOWNNO", "LON", "LAT", "MEDV"], inplace = True)
boston.head()


# ## Part 2
# **Make some pairwise scatterplots of the predictors (columns) in the data set. Describe your findings.**

# In[ ]:


# Use pairplot to create a trio of scatterplots relating median home value with
# percent of home built prior to 1940, percent of lower socioeconomic status residents, and pupil-teacher ratio
g = sns.pairplot(boston, x_vars = ["AGE", "LSTAT", "PTRATIO"], y_vars = ["CMEDV"], height = 4)


# In[ ]:


# Use catplot to create a boxplot comparing the median home values between tracts
# bordering the Charles River and those which do not
g = sns.catplot(x = "CHAS", y = "CMEDV", kind = "box", order = [0, 1], data = boston)


# First, I generated some plots to explore the relationship between median home value and a number of non-crime factors. There aren't any especially clear patterns I can discern from thes plots aside from the expected result that as a tracts with higher median home values have a greater proportion of lower-status residence. Also, it appears as if tracts that border the Charles river are a high a slightly higher median home value on average.

# In[ ]:


# Use pairplot to create a pair of scatter plots to relate the concentration of nitric oxides with median home value
# and percent of non-retail business acres
g = sns.pairplot(boston, x_vars = ["CMEDV", "INDUS"], y_vars = ["NOX"], height = 4)


# These two scatter plots in this next group explore factors that might relate to the concentration of nitric oxides. While there isn't a strong relationship, it appears that tracts with higher median home value also weakly tend to have lower concentrations of nitric oxides. There is a much clearer relationship with the percentage of non-retail business acres -- tracts with a higher proportion of non-retail business acres tend to have higher concentrations of nitric oxides. The next two plots look at some more factors which might be related to the median home value of a tract. 

# In[ ]:


# Use pairplot to create a pair of scatter plots to relate the median home value to the proportion of black residents
# and the proximity to Boston employment centers
g = sns.pairplot(boston, x_vars = ["B", "DIS"], y_vars = ["CMEDV"], height = 4)


# The left-hand plot seems to indicate that there is a relationship between the value of `B` and `CMEDV`, where as `B` increases,`CMEDV` increases. If I am interpreting this correctly, this means that tracts with high median home values have a very low (close to 0%) proportion of Black residents, while tracts with low median home values have a much higher proportion (close to 63%). The right-hand plot appears to indicate that there is also a relationship between proximity to Boston employment centers and median home value, with home values generally increasing as one gets further away from the employment centers.

# ## Part 3
# **Are any of the predictors associated with per capita crime rate? If so, explain the relationship.**

# In[ ]:


# Use pairplot to create a quartet of scatter plots to relate the per capita crime rate with proportion of Black residents,
# proportion of lower-status residents, median home value, and proximity to Boston employment centers
g = sns.pairplot(boston, x_vars = ["B", "LSTAT", "CMEDV", "DIS"], y_vars = ["CRIM"], height = 4)


# Based on the above four scatter plots, it appears that there are pretty clear relationships between crime rate and median home value, percent of lower status residents, and proximity to Boston employment centers. Tracts with lower home values tend to have higher crime rates, as do tracts which are closer to Boston employment centers. In addiion, tracts with higher proportion of lower status residents tend to have higher crime rates. I was also curious if there would be a relationship between crime rate and `B`, which serves as some kind of measurement for the proportion of Black residents. Based on the scatter plot between those two variables, there doesn't appear to be a clear relationship.

# ## Part 4
# **Do any of the suburbs of Boston appear to have particularly high crime rates? Tax rates? Pupil-teacher ratios? Comment on the range of each predictor.**

# In[ ]:


# Create grid of plots (fig)
# ax will be an array of three Axes objects
# Set the figure size so the plots aren't all squished together
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 5))

# Create histogram for Boston crime rates
sns.distplot(boston["CRIM"], kde = False, ax = axes[0])
axes[0].set(xlabel = "", title = "Histogram of per capita crime rate")

# Create histogram for Boston tax rates
# Use more bins than the default given by the Freedman-Diaconis rule to see more of the shape of the distribution
sns.distplot(boston["TAX"], bins = 20, kde = False, ax = axes[1])
axes[1].set(xlabel = "", title = "Histogram of tax rate")

# Create histogram for Boston pupil-teacher ratios
sns.distplot(boston["PTRATIO"], kde = False, ax = axes[2])
axes[2].set(xlabel = "", title = "Histogram of pupil-teacher ratio")


# In[ ]:


boston.loc[:, ["CRIM", "TAX", "PTRATIO"]].describe()


# Based on the histograms and the numerical summary, there do appear to be tracts within Boston which have particularly high crime rates, tax rates, or pupil-teacher ratios. The minimum crime rate is 0.00632, while the maximum is 88.97620, with a median of 0.25651. The minimum tax rate is \\$187 per \\$10000, while the maximum is \\$711, with a median of \\$330. The minimum pupil-teacher ratio is 12.60 pupils per teacher, while the maximum is 22, with a median of 19.05. Given the median value, the maximum pupil-teacher ratio in the data set isn't outrageously high, since about half of the tracts have a ratio of 19 or more.

# ## Part 5
# **How many of the suburbs in this data set bound the Charles river?**

# In[ ]:


# Use the fact that in the data set, 1 = Borders the Charles and 0 = otherwise to count the number
# of tracts which borders the Charles by summing along that column
boston["CHAS"].sum()


# In this data set, 35 tracts neighbor the Charles river.

# In[ ]:


# We can use a boolean mask to only take the rows for tracts which border the Charles
# Then look at the 0-level of the multi-index to access the town names and return the unique ones
# We could then check the Index.size attribute to get the number of unique towns which border the Charles
boston[boston["CHAS"] == 1].index.unique(level = 0)


# In[ ]:


# Alternatively, we can get the level values for the 0-level of the multi-index and use the
# Index.nunique() function to return the number of unique towns which border the Charles
boston[boston["CHAS"] == 1].index.get_level_values(0).nunique()


# From the computations above, we can see that 11 distinct towns border the Charles river: Allston-Brighton, Back Bay, Beacon Hill, Cambridge, Dedham, Dover, Newton, Needham, Waltham, Watertown, and Wellesley.

# ## Part 6
# **What is the median pupil-teacher ratio among towns in this data set?**

# In[ ]:


boston["PTRATIO"].describe()


# The median pupil-teacher ratio among towns in this data set is 19.05 pupils per teacher.

# ## Part 7
# **Which suburb of Boston has the lowest median value of owner-occupied homes? What are the values of the other predictors for that suburb, and how do those values compare to the overall ranges for those predictors? Comment on your findings.**

# In[ ]:


min_medv = boston["CMEDV"].min()
boston[boston["CMEDV"] == min_medv]


# In[ ]:


# Round to four decimal places so I don't have to scroll horizontally to view entire
# set of summary statistics
boston.describe().round(4)


# Two of the tracts of South Boston have the lowest median value of owner-occupied homes, at $5000. Both of these tracts have very high crime rates compared to the overall range for that variable, with values 38.3518 and 67.9208 putting them far into the upper quartile and into the range of being outliers. These tracts have no land zoned for residential lots of 25000 sq. ft., though this is in line with at least half of the tracts in the overall set given the median for `ZN` is 0. The two tracts do have a relatively high proportion of non-retail business acres, with values of 18.1 being right at the third quartile. Similarly, the tracts also have concentrations of nitric oxides in the upper quartile of the overall set with a value of 0.693 parts per ten million. The average number of rooms per dwelling for these two tracts is at the low end, with values of 5.453 and 5.683 putting them at the bottom quartile. Next, these two tracts are among those with the highest proportion of owner-occupied homes built prior to 1940, with a value of 100. The tracts are also quite close Boston employment centers with `DIS` values of 1.4896 and 1.4254 putting them at the bottom quartile. The tracts also are very close to radial highways with the maximum value of `RAD` at 24. Next, the tracts have above average property tax rates, with a value of \\$666 per \\$10000, putting them at the third quartile. The pupil-teacher ratio of 20.2 also puts these tracts at the third quartile. The tracts have relatively high values for `B`, though one tract has a maximum value while the other, with a value of 384.97, is in between the first and second quartiles. Lastly, the tracts have a high proportion of lower status residents (values of 30.59 and 22.98), putting them in the top quartile of the data.
# 
# In summary, these two tracts with the lowest median value of owner-occupied homes have predictors generally at the extreme ends of their respective ranges.

# ## Part 8
# **In this data set, how many of the suburbs average more than seven rooms per dwelling? More than eight rooms per dwelling? Comment on the suburbs that average more than eight rooms per dwelling.**

# In[ ]:


# Use the fact that when summing, False has an integer value of 0 and True has an integer value of 1
(boston["RM"] > 7).sum()


# In[ ]:


# Use the fact that when summing, False has an integer value of 0 and True has an integer value of 1
(boston["RM"] > 8).sum()


# In this data set, there are 64 tracts which average more than seven rooms per dwelling, and 13 of those tracts which average more than 8 rooms per dwelling.

# In[ ]:


boston.loc[boston["RM"] > 8]


# In[ ]:


boston.loc[boston["RM"] > 8].describe().round(4)


# From the numerical summary, one thing that stands out is that the tracts which average at least eight rooms per dwelling have low crime rates, low concentrations of nitric oxides, low proportions of Black residents (high values of `B`), and low proportions of lower status residents compared to the overall data set.
