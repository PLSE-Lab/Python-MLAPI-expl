#!/usr/bin/env python
# coding: utf-8

# # Sweetviz: two lines in Python to analyze & compare datasets
# Exploratory data analysis (EDA) is an essential early step in most data science projects and it often consists of taking the same steps to characterize a dataset (e.g. find out data types, missing information, distribution of values, correlations, etc.). Given the repetitiveness and similarity of such tasks, there are a few libraries that automate and help kickstart the process. 
# 
# One of the latest is an open-source Python library I wrote called Sweetviz ([GitHub](https://github.com/fbdesignpro/sweetviz)). It takes pandas dataframes and creates a self-contained HTML report (similar to libraries such as pandas-profiling). The goal was that, in addition to creating the most informative and beautiful visualizations possible, it would also enable an immediate visualization of:
# * Comparison of 2 datasets (e.g. Train vs Test)
# * Relationship of the target value with all other variables (e.g. "What was the survival rate of male vs female" etc.)
# 
# [You can find the Sweetviz report from the Titanic dataset used in this notebook HERE](http://cooltiming.com/SWEETVIZ_REPORT.html)
# 
# ## EDA made fun!
# Perhaps I'm too much of a data geek, but I have to say I am really happy with the result, to the point where I'm looking forward to analyzing more datasets! Being able to get so much information about the target value and compare different areas of the dataset almost instantly transform this initial step from being out all about tedium to being quicker, fun and interesting (to me at least). :) Of course EDA is a much longer process but at least that first step is a lot smoother. Let's see how it works out with the Titanic dataset!

# # Analyzing the Titanic dataset
# After installation (using `pip install sweetviz`), simply load pandas dataframes and call either `analyze()`, `compare()` or `compare_intra()` depending on your need (more on that below). For now, let's start with the case at hand, loading it as so:
# ```
# import pandas as pd
# train = pd.load_csv("input/titanic/train.csv")
# test = pd.load_csv("input/titanic/test.csv")
# ```
# So we now have 2 dataframes (train and test), and we would like to analyze the target value "Survived". I want to point out in this case we know the name of the target column in advance, but it is always optional to specify a target column. We can now generate a report with this line of code:
# ```
# my_report = sweetviz.compare([train, "Train"], [test, "Test"], "Survived")
# ```
# **Note: since sweetviz is brand-new, it is not yet available to be run in Kaggle kernels. All following output are screenshots taken from the actual result.**
# Running this command will perform the analysis and create the report object. To get the output, simply use the `show_html()` command:
# ```
# my_report.show_html("Report.html") # Not providing a filename will default to SWEETVIZ_REPORT.html
# ```
# 
# After generating the file, it will open it through your default browser and should look something like this:
# 
# ![image.png](attachment:image.png)
# 
# There's a lot to unpack, so let's take it one step at a time!

# # Summary display
# ![image.png](attachment:image.png)
# 
# The summary shows us the characteristics of both dataframes side-by-side. We can immediately identify that the testing set is roughly half the size of the training set, but that it contains the same features. That legend at the bottom shows us that the training set does contain the "Survive" target variable, but that the testing set does not.
# 
# Note that Sweetviz will do a best guess at determining the data type of each column, between numerical, category/boolean and text. These can be overridden, more on that below.

# # Associations
# Hovering your mouse over the "Associations" button in the summary will make the Associations graph appear on the right-hand side:
# 
# ![image.png](attachment:image.png)
# 
# This graph is a composite of the visuals from [Drazen Zaric: Better Heatmaps and Correlation Matrix Plots in Python](https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec) and concepts from [Shaked Zychlinski: The Search for Categorical Correlation](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9).
# 
# Basically, in addition to showing the traditional numerical correlations, it unifies in a single graph both numerical correlation but also the uncertainty coefficient (for categorical-categorical) and correlation ratio (for categorical-numerical). Squares represent categorical-featured-related variables and circles represent numerical-numerical correlations. Note that the trivial diagonal was left empty, for clarity (as I found it added useless noise to typical graphs).
# 
# IMPORTANT: categorical-categorical associations (provided by the uncertainty coefficient) are ASSYMMETRICAL, meaning that each row represents how much the row title (on the left) gives information on each column. For example, "Sex", "Pclass" and "Fare" are the elements that give the most information on "Survived". For the Titanic dataset, this information is rather symmetrical but it is not always the case.
# 
# Finally, it is worth noting these correlation/association methods shouldn't be taken as gospel as they make some assumptions on the underlying distribution of data and relationships. However they can be a very useful starting point.

# # Target variable
# ![image.png](attachment:image.png)
# 
# When a target variable is specified, it will show up first, in a special black box.
# 
# **IMPORTANT: only numerical and boolean features can be targets currently.**
# 
# We can gather from this summary that "Survived" has no missing data in the training set (891, 100%), that there are 2 distinct possible values (accounting for less than 1% of all values), and from the graph it can be estimated that roughly 60% did not survive.
# 

# # Detail area (categorical/boolean)
# When You move the mouse to hover over any of the variables, an area to the right will showcase the details. The content of the details depends on the type of variable being analyzed. In the case of a categorical (or boolean) variable, as is the case with the target, the analysis is as follows.
# 
# **IMPORTANT: a "widescreen" monitor is required to see the full detail area for the moment.**
# ![image.png](attachment:image.png)
# 
# Here, we can see the exact statistics for each class, where 62% did not survive and 38% survived. You also get the detail of the associations for each of the other features.

# # Numerical data
# ![image.png](attachment:image.png)
# 
# Numerical data shows more information on its summary. Here, we can see that in this case about 20% of data is missing (21% in the test data, which is very consistent). Interestingly, we can see from the graph on the right that the survival rate is pretty consistent across all ages, except for the youngest which have a higher survival rate. It would look like "women and children first" was not just talk.

# # Detail area (numerical)
# ![image.png](attachment:image.png)
# 
# As with the categorical data type, the numerical data type shows some extra information in its detail area. Noteworthy here are the buttons on top of the graph. These buttons change how many "bins" are shown in the graph. You can select the following:
# * Auto
# * 5
# * 15
# * 30
# 
# **Note: to get to these buttons, you need to "lock in place" the current feature by clicking on it. The feature then has a READ OUTLINE to show it is locked in place and you can access the detail area.** 
# 
# For example selecting "30" yields a much more granular graph:

# ![image.png](attachment:image.png)

# # Text data
# For now, anything that the system does not consider numerical or categorical will be deemed as "text". Text features currently only show count (percentage) as stats. I am hoping to change this in the future but for now it still yields useful information as far as seeing what the data is made of.
# 
# ![image.png](attachment:image.png)

# # FeatureConfig: forcing data types, skipping columns
# In many cases, there are "label" columns that you may not want to analyze (although target analysis can provide insights on the distribution of target values based on labeling). In other cases, you may want to force some values to be marked as categorical even though they are numerical in nature.
# 
# To do all this, simply create a FeatureConfig object and pass it in to the analyze/compare function. You can specify either a string or a list to kwargs `skip`, `force_cat` and `force_text`:
# ```
# feature_config = sweetviz.FeatureConfig(skip="PassengerId", force_cat=["Ticket"])
# 
# my_report = sweetviz.compare([train, "Train"], [test, "Test"], "Survived", feature_config)
# ```

# # Comparing sub-populations (e.g. Male vs Female)
# Even if you are only looking at a single dataset, it can be very useful to study the characteristics of different subpopulations within that dataset. To do so, Sweetviz provides the `compare_intra()` function. To use it, you provide a boolean test that splits the population, and given name to each subpopulation. For example:
# ```
# my_report = sweetviz.compare_intra(train, train["Sex"] == 'male', ["Male", "Female"], 'Survived')
# 
# my_report.show_html() # Not providing a filename will default to SWEETVIZ_REPORT.html
# ```        
# 
# Yields the following analysis: (here I used feature_config to skip showing the analysis of the "Sex" feature, as it is redundant)
# 
# ![image.png](attachment:image.png)
# 

# # Putting it all together
# EDA is a fluid, artistic process that must be uniquely adapted to each set of data and situation. However, a tool like Sweetviz can help kickstart the process and get rid of a lot of the initial minutiae of characterizing datasets to provide insights right off the bat. 
# ## Individual fields
# * **PassengerId**
# ![image.png](attachment:image.png)
#   * The distribution of ID's and survivability is even and ordered as you would hope/expect, so no surprises here.
#   * No missing data

# * **Sex**
# ![image.png](attachment:image.png)
#   * About twice as many males as females, but...
#   * Females were much more likely to survive than males
#   * Looking at the correlations, Sex is correlated with Fare which is and isn't surprising...
#   * Similar distribution between Train and Test
#   * No missing data

# * **Age**
# ![image.png](attachment:image.png)
#   * 20% missing data, consistent missing data and distribution between Train and Test
#   * Young-adult-centric population, but ages 0-70 well-represented
#   * Surprisingly evenly distributed survivability, except for a spike at the youngest age
#   * Using 30 bins in the histogram in the detail window, you can see that this survivability spike is really for the youngest (about <= 5 years old), as at about 10 years old survivability is really low.

# ![image.png](attachment:image.png)
# * Age seems related to Siblings, Pclass and Fare, and a bit more surprisingly to Embarked

# * **Name**
# ![image.png](attachment:image.png)
#   * No missing data, data seems pretty clean
#   * All names are distinct, which is not surprising

# * **Pclass**
# ![image.png](attachment:image.png)
#   * Survivability closely follows class (first class most likely to survive, third class least likely)
#   * Similar distribution between Train and Test
#   * No missing data

# * **SibSp**
# ![image.png](attachment:image.png)
#   * There seems to be a survival spike at 1 and to some degree 2, but (looking at the detail pane not shown here) there is a sharp drop-off at 3 and greater
#   * Similar distribution between Train and Test
#   * No missing data

# **(Kaggle notebook keeps disconnecting every time I had images so I will stop adding images here)**
# 
# * **Parch**
#   * Similar distribution between Train and Test
#   * No missing data
# * **Ticket**
#   * ~80% distinct values, so about 1 in 5 shared tickets on average
#   * The highest frequency ticket was 7, which is generally consistent with the maximum number of siblings (8)
#   * No missing data, data seems pretty clean
# * **Fare**
#   * As expected, and as with Pclass, the higher fares survived better (although sample size gets pretty thin at higher levels)
#   * A Correlation Ratio of 0.26 for "Survived" is relatively high so it would tend to support this theory
#   * About 30% distinct values feels a bit high as you would expect fewer set prices  but looks like there is a lot of granularity so that's ok
#   * Only 1 missing recordu in the Test set, data pretty consistent between Train and Test
# * **Cabin **
#   * A lot of missing data (up to 78%), but consistent between Train and Test
#   * Maximum frequency is 4, which would make sense to have 4 people maximum in a cabin
# * **Embarked**
#   * 3 distinct values (S, C, Q)
#   * Only 2 missing rows, in Train data. Data seems pretty consistent between Train and Test
#   * Survivability somewhat higher at C; could this be a location with richer people?
#   * Either way, "Embarked" shows a Uncertainty Coefficient of only 0.03 for "Survived", so it may not be very significant

# ## General analysis
# * Overall, most data is present and seems consistent and make sense; no major outliers or huge surprises
# * **Test versus Training data**
#   * Test has about 50% fewer rows
#   * Train and Test are very closely matched in the distribution of missing data
#   * Train and Test data values are very consistent across the board
# * **Association/correlation analysis**
#   * Sex, Fare and Pclass give the most information on Survived
#   * As expected, Fare and Pclass are highly correlated
#   * Age seems to tell us a good amount regarding Pclass, siblings and to some degree Fare, which would be somewhat expected. It seems to tell us a lot about "Embarked" which is a bit more surprising.
# * **Missing data**
#   * There is no significant missing data except for Age (~20%) and Cabin (~77%) (and an odd one here and there on other features)
# 

# # Conclusion
# All this information from just two lines of code!
# 
# I hope you will find Sweetviz as fun and interesting to use as I do! If this quick overview made you want to try it out, you can check out the project on [GitHub](https://github.com/fbdesignpro/sweetviz). I look forward to hearing your thoughts and comments. Thank you.
# 
