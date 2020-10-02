#!/usr/bin/env python
# coding: utf-8

# # Exploring and Visualizing Exoplanet Data
# Author: Justin Wilbourne
# 
# Disclaimer: I'm studying data science. The focus of this Jupyter Notebook is data exploration and visualization. The statistical tests toward the end were attempts, and may not be proper examples.

# # Abstract
# Exoplanets are worlds beyond our solar system. Over the past decade, thousands have been discovered through various indirect detecction methods. The team at [The Extrasolar Planets Encyclopaedia](exoplanet.eu) maintains information about the expolanets. A dataset of over 3,000 confirmed exoplanets from the team can also be found at the [Exoplanets Database at Kaggle](https://www.kaggle.com/eduardowoj/exoplanets-database/home).
# 
# This notebook examines correlation of exoplanetary mass and radius, correlation of size [mass] and orbital periods. Little -- if any -- correlation is found. The notebook ends with a brief discussion of detection methods.
# 
# 
# ## Problem
# The goal of this notebook is to directly address questions posed in the Overview section of the [Kaggle site](https://www.kaggle.com/eduardowoj/exoplanets-database/home).
# 
# > * How does the mass of the planets correlates with their radius? What does this tells us about the planets' compositions? (Hint: There are only two kinds of planets: Gas planets and rocky planets, with low and high densities, respectively)
# 
# > * How does the size of the planet correlates with the orbital period? And what correlations are there with the spectral type of the host stars?
# 
# > * Which are the best detection methods? What are their limitations? Why are there no planets with large orbital periods detected by transit and low mass planets detected by radial velocity?
# 
# Additionally, the overview mentions a challenge with regard to data quality.
# > The data is well structured in a CSV file, but since it comes from several different sources, some parameters aren't well formated, and require some cleaning and filtering, an additional challenge!
# 
# 
# 
# 
# ## Conclusion
# In short, some small but questionable relationship was found between planetary mass and radius for hte Primary Transit detection type. No relationship was noted between mass and orbital period for the two detection types, Primary Transit and Radial Velocity.
# 
# In retrospect, the subject domain of the exoplanet data presented was a challenge by itself. There weren't many resources to define columns or group relationships. My major breakthrough was the discovery that detection types played a large role on the presence/availability of data.
# 
# _But how did we get here?_
# 
# Let's dig in!

# # <a id="contents">Contents</a>
# 1. [Exploratory Visualizations](#exploratoryVisualizations)
#   1. [Exoplanets by Year Discovered](#byYearDiscovered) - barplot
#   2. [Exoplanets by Detection Type](#detectionType) - barplot and table
#   3. [Subject Domain: Detection Type](#subjectDomain_detectionType) - important domain knowledge to understanding dataset
#   4. [Exoplanets by Mass Detection Type](#massDetectionType) - barplot
#   5. [Mass, Radius, and Orbital Period](#massRadiusPeriod) - boxplots
# 2. [Preparation and Cleanup](#prep)
#   1. [Error Measures and Sparse Data](#errorMeasures_sparseData)
#   2. [Z-Score Normalization](#zscoreNorm)
# 3. [Stratified Visualizations](#stratifiedVisualizations)
#   1. [Molecules Observed](#moleculesObserved) - barplot
#   2. [Plots: Primary Transit](#plots_pt) - many histograms and barplots
#   3. [Plots: Radial Velocity](#plots_rv) - many histograms and barplots
# 4. [Linear Regression](#linearRegression)  
#   1. [Primary Transit: Mass and Radius](#pt_massRadius)  
#   2. [Primary Transit and Radial Velocity: Mass and Orbital Period](#pt_rv_massOrbitalPeriod)
# 5. [Definitions](#definitions) - definition of terms used in the dataset

# In[ ]:


# Load necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn import linear_model
import statsmodels.formula.api as sm


# Location of dataset
file = "../input/kepler.csv"

# Create a dataframe from csv data
df = pd.read_csv(file)


# <a id="exploratoryVisualizations"></a>

# # Exploratory Visualizations
# ## <a id="byYearDiscovered">Exopanets by Year Discovered</a>
# [back to contents](#contents)  
# 
# The `discovered` column indicates the year when an exoplanet was discovered.

# In[ ]:


discoveries_by_year = df["discovered"].value_counts()

plt.figure(figsize=(12, 4))
plt.bar(discoveries_by_year.index, discoveries_by_year.values, align='center', alpha=0.5)
plt.xticks(discoveries_by_year.index, rotation='vertical')
plt.ylabel('Planets Discovered')
plt.title('Planets by Year of Updated Date')
plt.show()

print("(Data made available Feb 2018)")
discoveries_by_year.sort_index()


# 2014 and 2016 were big years for exoplanet discoveries. Why? Continue onward and find out!
# 
# ## <a id="detectionType">Exoplanets by Detection Type</a>
# [back to contents](#contents)  

# In[ ]:


column = 'detection_type'
plt.figure(figsize=(12, 4))
df[column].value_counts().plot.bar()
plt.title(f"Bar Plot of {column.capitalize()} ({df[column].dtype})")
plt.ylabel("Exoplanets Discovered")
plt.show()
print(df[column].value_counts())


# By far, more exoplanets have been discovered via the Primary Transit `detection_type`. 
# 
# As a bit of trivia, two exoplanets seems to have been discovered via two detection types: Primary Transit and TTV.
# 
# Which methods are used when?

# In[ ]:


print(f"Exoplanets discovered, 1988 - 2007: {df['discovered'].where(df.discovered < 2008).count()}")
print(f"Exoplanets discovered, 2008 - 2018: {df['discovered'].where(df.discovered >= 2008).count()}")

# Add a count column to make counts easy
df["count"] = 1

# more on as_index @
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
grouped = df[["count", "detection_type", "discovered"]].groupby(["discovered", "detection_type"], as_index=False).agg('count')

grouped.pivot(index="discovered", columns="detection_type").fillna(0)["count"]


# Exoplanet discovery has exploded. In the past ten years, 2008-2018, 3,452 exoplanets were discovered! Compare this to the twenty year period, between 1988 and 2007, when only 279 exoplanets were discovered. 
# 
# While the Primary Transit `detection_type` is the cause for the spike, Radial Velocity consistently discovered many planets since the year 2000.
# 
# ## <a id="subjectDomain_detectionType">Subject Domain: Detection Type</a>
# [back to contents](#contents)  
# 
# #### Primary Transit
# The above table showing `detection_type` and `discovered` reveals that Primary Transit as the spikes in discovereis in 2015 and 2016. <a href="https://en.wikipedia.org/wiki/Kepler_(spacecraft)">NASA's Kepler spacecraft</a> is behind the majority of the Primary Transit findings. 
# 
# [How does Kepler discover exoplanets?](https://kcts9.pbslearningmedia.org/resource/nvap-sci-keplerwork/how-does-kepler-work/) The Kepler spacecraft continually monitors the brightness of thousands of stars in a patch of sky. When a planet transits a star -- or moves in front of a star -- it dims the overall brightness of the star. Over time, transits creates patterns in the data and this pattern is how astronomers can infer that a planet orbits a star. This is an _indirect_ method of discovering planets.
# 
# * The bigger the planet, the more starlight will be blocked. 
# * The closer the planet is to a star, the more often a transit will be recorded in a given amount of time.
# 
# #### Radial Velocity
# In contrast, the Radial Velocity `detection_type` measures a star's light as the star _moves_ away or toward Earth. When  "wobbles" in the light are detected, those wobbles can be attributed to the gravitational effects of planets orbiting the star. Another name for this type of detection is Doppler spectroscopy. It is also an indirect method.
# 
# 
# Understanding this aspect of the domain is important because it reveals an important insight of the dataset: **Some columns may be intrinsically connected to the `detection_type`.**

# ## <a id="massDetectionType">Exoplanets by Mass Detection Type</a>
# [back to contents](#contents)  
# We have mass measurements for a fraction of our data.

# In[ ]:


# Purpose: Returns an index array that identifies rows which do not have np.inf or NaN values
# Parameters: Pandas dataframe and name of column in dataframe
def get_usable_indexes(dataframe, column):
    return ~((dataframe[column] == np.inf) | (pd.isnull(dataframe[column])))

# Bar plot
column = "mass_detection_type"
plt.figure(figsize=(12, 4))
df[column].value_counts().plot.bar()
plt.title(f"Bar Plot of {column.capitalize()} ({df[column].dtype})")
plt.ylabel("Exoplanets Discovered")
plt.show()
# Show values behind the barplot
print(df[column].value_counts())


# Define handy function to print info 
def report_sparseness(column):
    usable_indexes = get_usable_indexes(df, column)
    usable = df.loc[usable_indexes, column].count()
    total = df[column].shape[0]
    percent = str(round((usable / total) * 100, 1)) + "%"
    print(f"Rows with {column}: {usable} / {total} ({percent})")

print()
print()
[report_sparseness(column) for column in ["mass", "mass_detection_type"]]
print()


# In[ ]:


grouped_detection_types = df[[
    "detection_type", 
    "mass_detection_type", 
    "mass"]
].groupby(["detection_type", "mass_detection_type"])

grouped_detection_types["mass"].count()


# There is no mass identified for over half the data. 
# 
# The subject domain discussion explains why Radial Velocity is a value in `mass_detection_type` and Primary Transit is not.

# ## <a id="massRadiusPeriod">Mass, Radius, and Orbital Period</a>
# [back to contents](#contents)  
# 
# Since we are interested in exoplanet mass, radius, and Orbital Periods, let's see the spread of values via boxplots. 
# 
# Both mass and radius are recorded in terms of [the planet Jupiter](https://solarsystem.nasa.gov/planets/jupiter/by-the-numbers/), the largest planet in our solar system.
# 
# ### Mass by Detection Type

# In[ ]:


# Show boxplots of mass per detection_type
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
df.loc[:,["detection_type", "mass"]].boxplot(by = ["detection_type"], ax = ax)
ax.set_title("Exoplanet Mass by Detection Type")
ax.set_ylabel("Mass (units of Planet Jupiter)")
fig.suptitle("") # Hide outer title
ax.set_xlabel("") # Auto-show x-labels
plt.show()

# Print counts per detection_type
grouped_detection_types = df[[
    "detection_type", 
    "mass"]
].groupby(["detection_type"])

grouped_detection_types["mass"].count()


# ### Radius by Detection Type

# In[ ]:


# Show boxplots of radius per detection_type
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
df.loc[:,["detection_type", "radius"]].boxplot(by = ["detection_type"], ax = ax)
ax.set_title("Exoplanet Radius by Detection Type")
ax.set_ylabel("Radius (units of Planet Jupiter)")
fig.suptitle("") # Hide outer title
ax.set_xlabel("") # Auto-show x-labels
plt.show()

# Print counts per detection_type
grouped_detection_types = df[[
    "detection_type", 
    "radius"]
].groupby(["detection_type"])

grouped_detection_types["radius"].count()


# From the pair of boxplots, we see we may need to scope our attention to the Primary Transit `detection_type` for finding a correlation between mass and radius.
# 
# ### Orbital Period
# Orbital Period is measured in days.

# In[ ]:


# There are two insane outliers from the imaging type that hamper this boxplot.
# Let's keep orbital periods under ~41 years so it at least somewhat matches our 30 years of data.
outliers = df["orbital_period"] > 15000
print(f"Ignoring {outliers.sum()} outliers")

# Show boxplots of radius per detection_type
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
df.loc[~outliers,["detection_type", "orbital_period"]].boxplot(by = ["detection_type"], ax = ax)
ax.set_title("Exoplanet Orbital Period by Detection Type")
ax.set_ylabel("Orbital Period (days)")
fig.suptitle("") # Hide outer title
ax.set_xlabel("") # Auto-show x-labels
plt.show()

# Print counts per detection_type
grouped_detection_types = df.loc[~outliers, [
    "detection_type", 
    "orbital_period"]
].groupby(["detection_type"])

grouped_detection_types["orbital_period"].count()


# # <a id="prep">Preparation and Cleanup</a>
# [back to contents](#contents)  
# The number of rows and columns are below.

# In[ ]:


# drop the count column added in visualization section
df = df.drop(columns=["count"])

print(f"Rows: {df.shape[0]} Columns: {df.shape[1]}")


# After initial investigation, some columns contain the same value, or no values at all. These are removed.

# In[ ]:


missing_values = df["hot_point_lon"].isnull().sum()
print(f"Missing 'hot_point_lon' values: {missing_values} / {df.shape[0]}")
print()

print("All 'planet_status' values are the same. All are confirmed.")
print(df["planet_status"].unique())
print()

df = df.drop(["hot_point_lon", "planet_status"], axis=1)
print("Dropped hot_point_lon and planet_status")


# Every row has a nan (missing value) or inf value. So throwing out rows with missing values is not an option. 

# In[ ]:


df_with_rows_without_missing_values = df.dropna()
print(f"Rows without missing values: {df_with_rows_without_missing_values.shape[0]}")


# Note: This notebook ignores NaN (not a number) and `np.inf` (infinity) values. 
# 
# Below I replace `np.inf` with `float("nan")`. However, the inf values may tell their own story (say, about the data collection from multiple sources). That story may be important, I dunno. Either way, they are ignored here and as such are equivalent to the NaN values.

# In[ ]:


for column in df:
    inf_indexes = (df[column] == np.inf)
    if inf_indexes.sum() > 0:
        df.loc[inf_indexes, column] = float("nan")
        print(F"{column}: {inf_indexes.sum()} inf -> nan")


# ## <a id="errorMeasures_sparseData">Error Measueres and Sparse Data</a>
# [back to contents](#contents)  
# To manage uncertainty in measurements, the dataset includes minimum and maximum errors. Meanwhile, the dataset itself contains many unusable values (NaN and inf).
# 
# First, let's examine how sparse some of these measures are (due to missing or NaN values).

# In[ ]:


# Suffixes for min and max errors
min_suffix = "_error_min"
max_suffix = "_error_max"

# List of columns having min and max errors
trio_columns = [
    "mass",
    "mass_sini",
    "radius",
    "orbital_period",
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "omega",
    "tperi",
    "tconj",
    "tzero_tr",
    "tzero_tr_sec",
    "lambda_angle",
    "impact_parameter",
    "tzero_vr",
    "k",
    "temp_calculated",
    "geometric_albedo",
    "star_distance",
    "star_metallicity",
    "star_mass",
    "star_radius",
    "star_age",
    "star_teff"
]

# Purpose: Return a summary of usable data as 
#          both a numerical percent value and display string equivalent
def get_usable_row_counts(dataframe, column):
    # Find indexes of NaN or inf values
    usable_indexes = get_usable_indexes(dataframe, column)
    percent = round((dataframe.loc[usable_indexes, column].count() 
                     / dataframe[column].shape[0])*100, 1)
    return percent, str(percent) + "%"

# Relies on outside variables
def make_trio_dataframe(dataframe):
    trio_df = pd.DataFrame()
    for column in trio_columns:
        measure_raw, measure_pct = get_usable_row_counts(dataframe, column)
        min_raw, min_pct = get_usable_row_counts(dataframe, column+min_suffix)
        max_raw, max_pct = get_usable_row_counts(dataframe, column+max_suffix)
        trio_df = trio_df.append({
            "column": column,
            "measure": measure_pct, 
            "measure_raw": measure_raw, 
            "min_error": min_pct, 
            "min_error_raw": min_raw, 
            "max_error": max_pct,
            "max_error_raw": max_raw
        }, ignore_index=True)
    
    return trio_df

trio_df = make_trio_dataframe(df)
trio_df[["column", "measure", "min_error", "max_error"]]


# As shown above, some measures are just barely present. Some may be inherit to the `detection_type` so it would be a mistake to drop these low frequency columns outright.
# 
# Let's stratify the data by `detection_type` for the two largest types: Primary Transit and Radial Velocity.

# In[ ]:


# Create Primary Transit dataframe
pt_df = df.loc[df["detection_type"] == "Primary Transit"]
print(f"Shape of Primary Transit dataframe")
print(f"  Rows: {pt_df.shape[0]} Columns: {pt_df.shape[1]}")
print()

# Create Radial Velocity dataframe
rv_df = df.loc[df["detection_type"] == "Radial Velocity"]
print(f"Shape of Radial Velocity dataframe")
print(f"  Rows: {rv_df.shape[0]} Columns: {rv_df.shape[1]}")


# Now with our major `detection_type`s broken apart, how do the data look?  
# 
# **Primary Transit**

# In[ ]:


trio_pt_df = make_trio_dataframe(pt_df)
trio_pt_df[["column", "measure", "min_error", "max_error"]]


# Let's drop columns that fall below a percentage threshold.

# In[ ]:


# Drop columns from Primary Transit dataframe where utilization is below 20%
columns_to_drop = trio_pt_df.loc[trio_pt_df["measure_raw"] < 20.0, "column"]
for column in columns_to_drop:
    pt_df = pt_df.drop(columns=[column + min_suffix, column + max_suffix, column] )

# See description of new, slimmer Primary Transit dataframe
trio_pt_df[["column", "measure", "min_error", "max_error"]].where(trio_pt_df.measure_raw > 20).dropna()


# **Radial Velocity**

# In[ ]:


trio_rv_df = make_trio_dataframe(rv_df)
trio_rv_df[["column", "measure", "min_error", "max_error"]]


# Just like before, let's drop columns that fall below a percentage threshold.

# In[ ]:


# Drop columns from Primary Transit dataframe where utilization is below 20%
columns_to_drop = trio_rv_df.loc[trio_rv_df["measure_raw"] < 20.0, "column"]
for column in columns_to_drop:
    rv_df = rv_df.drop(columns=[column + min_suffix, column + max_suffix, column] )

# See description of new, slimmer Primary Transit dataframe
trio_rv_df[["column", "measure", "min_error", "max_error"]].where(trio_rv_df.measure_raw > 20).dropna()


# Phew! With some of the more frequent/important columns highlighted, other columns that are descriptive (like `# name` and `star_name`) can also be removed.

# In[ ]:


# These columns contain too many unique string or date values to plot
columns_to_skip = [
    "# name", 
    "updated", 
    "alternate_names", 
    "star_name", 
    "star_sp_type", 
    "temp_measured",
    "log_g",
    "star_alternate_names", 
    "publication_status", 
    "detection_type",
    "star_detected_disc", 
    "star_magnetic_field"]
pt_df = pt_df.drop(columns=columns_to_skip)
rv_df = rv_df.drop(columns=columns_to_skip)

print(f"Dropped unused columns")


# ## <a id="zscoreNorm">Z-Score Normalization</a>
# [back to contents](#contents)  

# In[ ]:


def zscore_norm(dataframe):
    for column in dataframe.columns:
        if df[column].dtypes == np.int64 or df[column].dtypes == np.float:
            # Copy the numpy array, then only normalize non-nan values
            normalized = dataframe[column].copy()
            usable_indexes = get_usable_indexes(dataframe, column)
            normalized[usable_indexes] = stats.zscore(dataframe.loc[usable_indexes, column])
            
            # Add new zscored values to dataframe
            dataframe[column] = normalized
            
zscore_norm(pt_df)
zscore_norm(rv_df)


# # <a id="stratifiedVisualizations">Stratified Visualizations</a>
# ## <a id="moleculesObserved">Molecules Observed</a>
# [back to contents](#contents)  
# 
# First, which molecules are observed from each `detection_type`?  
# (Due to the heavy elements listed, I safely assume `molecules` are observed against the planet, not the star.)
# 
# ### Primary Transit

# In[ ]:


column = "molecules"
plt.figure(figsize=(12, 4))
pt_df[column].value_counts().plot.bar()
plt.title(f"Bar Plot of {column.capitalize()} ({pt_df[column].dtype})")
plt.show()
print(pt_df[column].value_counts())


# ### Radial Velocity

# In[ ]:


column = "molecules"
plt.figure(figsize=(12, 4))
rv_df[column].value_counts().plot.bar()
plt.title(f"Bar Plot of {column.capitalize()} ({rv_df[column].dtype})")
plt.show()
print(rv_df[column].value_counts())

plt.show()


# We see a stark difference of molecules observed between Primary Transit and Radial Velocity `detection_types` from these two barplots. Primary Transit observed far more mixtures of molecules than Radial Velocity.
# 
# While there are more Primary Transit records, I think the reason behind this disparity is the detection method itself. A planet crossing in front of a star probably yields more clues to the planet's composition than measuring a doppler effect.

# In[ ]:


def make_histogram(dataframe, column):
    usable_rows = get_usable_indexes(dataframe, column)
    mean = np.mean(dataframe.loc[usable_rows, column])
    std = np.std(dataframe.loc[usable_rows, column])
    dataframe.loc[usable_rows, column].plot.hist(bins=10)
    plt.axvline(mean, color = 'red', alpha=.8)
    plt.axvline(np.mean(mean + 2*std), color = 'red', alpha=.6, linestyle='--')
    plt.axvline(np.mean(mean - 2*std), color = 'red', alpha=.6, linestyle='--')
    plt.title(f"Histogram of {column.capitalize()} ({dataframe[column].dtype})")
    return plt

def run_plots(dataframe):
    columns_remaining = list(dataframe.columns)
    
    # Plot trio columns
    for column in dataframe.columns:
        if column in trio_columns:
            plt.figure(figsize=(20, 5))
            plt.tight_layout()
            plt.subplot(1, 3, 1)
            make_histogram(dataframe, column)
            columns_remaining.remove(column)

            plt.subplot(1, 3, 2)
            make_histogram(dataframe, column + min_suffix)
            columns_remaining.remove(column + min_suffix)

            plt.subplot(1, 3, 3)
            make_histogram(dataframe, column + max_suffix)
            columns_remaining.remove(column + max_suffix)
            plt.show()
    
    for column in columns_remaining:
        # Plot categorical data
        try:
            if (dataframe[column].dtype == object) and column != "molecules":
                dataframe[column].value_counts().plot.bar()
                plt.title(f"Bar Plot of {column.capitalize()} ({dataframe[column].dtype})")
                plt.show()
                print(dataframe[column].value_counts())
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"No plot for {column}?")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Plot numerical data not in trio columns
        try:
            if dataframe[column].dtypes == np.int64 or dataframe[column].dtypes == np.float:
                usable_rows = get_usable_indexes(dataframe, column)
                mean = np.mean(dataframe.loc[usable_rows, column])
                std = np.std(dataframe.loc[usable_rows, column])

                dataframe.loc[usable_rows, column].plot.hist(bins=20)
                plt.axvline(mean, color = 'red', alpha=.8)
                plt.axvline(np.mean(mean + 2*std), color = 'red', alpha=.6, linestyle='--')
                plt.axvline(np.mean(mean - 2*std), color = 'red', alpha=.6, linestyle='--')
                plt.title(f"Histogram of {column.capitalize()} ({dataframe[column].dtype})")
                plt.show()
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"No plot for {column}?")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!")


# ## <a id="plots_pt">Plots: Primary Transit</a>
# [back to contents](#contents)  

# In[ ]:


run_plots(pt_df)


# ## <a id="plots_rv">Plots: Radial Velocity</a>
# [back to contents](#contents)  

# In[ ]:


run_plots(rv_df)


# # <a id="linearRegression">Linear Regression</a>
# [back to contents](#contents)  
# 
# This section assesses the correlation of 
# * Exoplanetary `mass` and `radius` for the Primary Transit `detection_type` only.
#   * Primary Transit has data for `radius` -- Radial Velocity does not
# * Exoplanetary `mass` and `orbital_period` for both Primary Transit and Radial Velocity

# In[ ]:


# Purpose: Create linear model and plot points
def present_linear_model(x_axis_values, y_axis_values, labels=["x_label", "y_label"]):
    from sklearn import linear_model
    n_points = len(x_axis_values)
    
    # First initialize the model.
    linear_model = linear_model.LinearRegression()
    
    # Fit the model to the data
    x_input = x_axis_values.values.reshape(n_points, 1)
    y_output = y_axis_values.values.reshape(n_points, 1)
    linear_model.fit(x_input, y_output)
    
    # Get predictions
    y_pred = linear_model.predict(x_input)
    
    # Plot output
    plt.scatter(x_input, y_output, alpha=.1)
    plt.plot(x_input, y_pred, linewidth=2, color="black")
    plt.grid(True)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(f"{labels[0]} vs {labels[1]}")
    plt.show()
    
    # Return model parameters
    # slope (m) and y-intercept (b)
    intercept = linear_model.intercept_[0] #'Intercept: {0:.5f}'.format(linear_model.intercept_[0])
    slope = linear_model.coef_[0][0] #'Slope : {0:.5f}'.format(linear_model.coef_[0][0])
    return intercept, slope


# ## <a id="pt_massRadius">Primary Transit: Mass and Radius</a>
# [back to contents](#contents)  
# What is the relationship between a planet's mass and radius?

# In[ ]:


# Create Mass-Radius dataframe
mr_df = pd.DataFrame()
mr_df["mass"] = pt_df["mass"]
mr_df["radius"] = pt_df["radius"]
mr_df = mr_df.dropna()

# Show plot of mass and radius
intercept, slope = present_linear_model(mr_df["mass"], mr_df["radius"], ["mass", "radius"])
print("Intercept: {0:.5f}".format(intercept))
print("Slope : {0:.5f}".format(slope))

# Show Regression results
ols_model = sm.ols(formula = 'radius ~ mass', data=mr_df)
results = ols_model.fit()

print('\nSSE, SST, SSR, and RMSE:')
sst = np.sum((mr_df["radius"] - np.mean(mr_df["radius"]))**2)
sse = sst - results.ssr
print('SSE: {}'.format(sse))
print('SST: {}'.format(sst))
print('SSR: {}'.format(results.ssr))
print('RMSE: {}'.format(np.sqrt(results.mse_model)))

# Get most of the linear regression statistics we are interested in:
print(results.summary())

# Plot a histogram of the residuals
sns.distplot(results.resid, hist=True)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Histogram')
plt.show()


# **Interpretation:** Primary Transit: Mass and Radius  
# It would seem as though the more massive a planet, the greater the radius. However, the more massive something is, the greater its gravity -- and so a planet's radius is unlikely to be a simple relationship.
# 
# While the confidence intervals do not encompass zero and the p-value appears significant, the plot looks odd. There is also significant error, as seen by R and R^2.
# 
# 
# ## <a id="pt_rv_massOrbitalPeriod">Primary Transit and Radial Velocity: Mass and Orbital Period</a>
# [back to contents](#contents)  
# What is the relationship between a planet's mass and orbital period?

# In[ ]:


# Create Mass-Orbital Period dataframe
mop_df = pd.DataFrame()
mop_df["mass"] = np.concatenate((pt_df["mass"], rv_df["mass"]))
mop_df["orbital_period"] = np.concatenate((pt_df["orbital_period"], rv_df["orbital_period"]))
mop_df = mop_df.dropna()

# Show plot of mass and radius
intercept, slope = present_linear_model(mop_df["mass"], mop_df["orbital_period"], ["mass", "orbital_period"])
print("Intercept: {0:.5f}".format(intercept))
print("Slope : {0:.5f}".format(slope))

# Show Regression results
ols_model = sm.ols(formula = 'mass ~ orbital_period', data=mop_df)
results = ols_model.fit()

print('\nSSE, SST, SSR, and RMSE:')
sst = np.sum((mop_df["mass"] - np.mean(mop_df["mass"]))**2)
sse = sst - results.ssr
print('SSE: {}'.format(sse))
print('SST: {}'.format(sst))
print('SSR: {}'.format(results.ssr))
print('RMSE: {}'.format(np.sqrt(results.mse_model)))

# Get most of the linear regression statistics we are interested in:
print(results.summary())

# Plot a histogram of the residuals
sns.distplot(results.resid, hist=True)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Histogram')
plt.show()


# **Interpretation:** Primary Transit and Radial Velocity: Mass and Orbital Period  
# The interaction between mass and orbital period does not seem significant due to the high p-value and because zero is stradled in the confidence intervals. There is also significant error, as seen by R and R^2.
# 
# I was not able to identify a column related to spectral type of the star. The relative brightness of a star should play a part in the indirect planet detection. For example, a large planet will have a greater impact on a small, dim star compared to an impact on a larger, brighter star.
# 
# 
# #### Which are the best detection methods?
# Each detection method surely has its place -- its own advantages and disadvantages. Certainly, Primary Transit has found the most exoplanets thus far.
# 
# 
# #### What are their limitations? Why are there no planets with large orbital periods detected by transit and low mass planets detected by radial velocity?
# Primary Transit relies on extended observation of planets to look for dimming of starlight. Dimming only occurs when a planet passes a line of sight. So if a planet takes many, many years to orbit a star, it will take many, many, **many** years to gather enough data to show a pattern indicating a planet.
# 
# The "wobble" detected by Radial Velocity is driven by the size of the planet. Low mass stars may have a very small effect on the wobble and may be undetectable.
# 
# The major detection types are described under [Subject Domain: Detection Type](#subjectDomain_detectionType). 

# # <a id="definitions">Definitions</a>
# [back to contents](#contents)  
# 
# 
# **mass**
# > Planetary mass in units of Jupiter (comparative measurement to the Jupiter in our solar system)  
# > Source: Column headers at http://exoplanet.eu/catalog/
# 
# **mass_sini**
# > A measurement to find minimum mass  
# > Mass * sin(i); "Minimum mass of a planet as measured by radial velocity [...]"  
# > Source: https://exoplanetarchive.ipac.caltech.edu/docs/API_exomultpars_columns.html  
# > See also: https://en.wikipedia.org/wiki/Minimum_mass
# 
# **radius**
# > Planetary radius in units of Jupiter (comparative measurement to the Jupiter in our solar system)  
# > Source: Column headers at http://exoplanet.eu/catalog/
# 
# **orbital_period**
# > Period of time (in years) needed to fully orbit another body (like a star)  
# > Source: https://www.novac.com/wp/fp/resources/glossary/  
# > Source: Column headers at http://exoplanet.eu/catalog/
# 
# **semi_major_axis**
# > Half of the major axis of an ellipse. Also equal to the average distance from the focus of a body moving on an elliptical orbit  
# > Source: https://www.novac.com/wp/fp/resources/glossary/
# 
# **eccentricity**
# > A measure of the extent to which an orbit departs from circularity. Eccentricity ranges from 0.0 for a circle to 1.0 for a parabola  
# > Source: https://www.novac.com/wp/fp/resources/glossary/
# 
# **inclination**
# > The tilt of the rotation axis or orbital plane of a body  
# > Source: https://www.novac.com/wp/fp/resources/glossary/
# 
# **geometric_albedo**
# > The geometric albedo of a celestial body is the ratio of its actual brightness as seen from the light source (i.e. at zero phase angle) to that of an idealized flat, fully reflecting, diffusively scattering (Lambertian) disk with the same cross-section.  
# > Source: https://en.wikipedia.org/wiki/Geometric_albedo
