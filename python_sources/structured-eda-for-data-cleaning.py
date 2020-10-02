#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# Cleaning and validating a dataset is one of the most important steps in building a machine learning model. This kernel provides: 
# 
# 1. A practice dataset that we know has some issues. Cleaning data is a very open ended problem, so practice is really the single most valuable thing we can offer you. We'll use the credit card dataset from [the R package AER](https://cran.r-project.org/web/packages/AER/AER.pdf).
# - A basic exploratory data analysis (EDA) process for error detection that you can apply to other datasets. There's no one true way, but this is a good place to start.
# 
# To get the most out of this tutorial you should fork this notebook and implement each EDA step on your own. Then, make a list of potential problems with the data. I'll provide a link to my version of the completed exercises at the end. 
# 
# Most of the exercises can be done with single line pandas commands, so if you find yourself writing much code I recommend pausing to look at the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/basics.html) or answer key.

# ### 1) Read the Manual
# A basic understanding of what the dataset is supposed to contain can help you identify discrepancies.
# 
# I've listed this step first, but feel free to change the order. You might find that you are better able to detect issues if you look at the data without any preconceptions, read the manual, then go back to the data.
# 
# 
# ```Format
# A data frame containing 1,319 observations on 12 variables.
# 
# card: Factor. Was the application for a credit card accepted?  
# reports: Number of major derogatory reports.   
# age: Age in years plus twelfths of a year.   
# income: Yearly income (in USD 10,000).   
# share: Ratio of monthly credit card expenditure to yearly income.   
# expenditure: Average monthly credit card expenditure.  
# owner: Factor. Does the individual own their home?  
# selfemp: Factor. Is the individual self-employed?  
# dependents: Number of dependents.  
# months: Months living at current address.  
# majorcards: Number of major credit cards held.  
# active: Number of active credit accounts.
# 
# Details
# According to Greene (2003, p. 952) dependents equals 1 + number of dependents, our calculations
# suggest that it equals number of dependents.
# Greene (2003) provides this data set twice in Table F21.4 and F9.1, respectively. Table F9.1 has just
# the observations, rounded to two digits. Note that age has some suspiciously low values (below one year) for some applicants.```
# 
# 
# ### 2) Review the Data Types
# Do the data types line up with the manual? Do they make sense for kinds of data we have?

# ### 3) Print Sample Rows
# Print a random slice of rows and review the values. Be sure to preserve the row order in case the data happens to be sorted or has other inter-row structure. Do the orders of magnitude of the data make sense? Are there any nested datatypes that need to be unpacked? Is each column expressed in useful units?

# ### 4) Summary Statistics
# Basic descriptive statistics are a good basis for sanity checks and to get a rough sense of how extreme any outliers are. In this case, for example, we know that none of the values in this dataset can be negative and should see that reflected in the minimums.

# ### 5) Plotting
# Simple scatterplots or histograms can reveal deep issues with the data. Be on the lookout for spikes, gaps in the data, or clusters of outliers; those are often worth digging into if you have time.

# Once you're done, head on over to [the answer key to compare notes](https://www.kaggle.com/sohier/structured-eda-for-data-cleaning-results/).
# 
# If you're looking for extra practice, you can also check out:
# - [NBA Shot Logs](https://www.kaggle.com/anthonypino/melbourne-housing-market)
# - [Melbourne Housing Market](https://www.kaggle.com/anthonypino/melbourne-housing-market)
# - Datasets that were scraped, rather than downloaded from prepared files, are often worth a look.

# 
