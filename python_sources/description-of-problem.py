#!/usr/bin/env python
# coding: utf-8

# **The Basics**
# 
# We have a very large continuous csv of data.  There are 629 million rows of data.  Test data consists of snippets of 150,000 data points.  That's about 4194 full segments.  Based on the EDA  [here](http://https://www.kaggle.com/artgor/seismic-data-eda-and-baseline) it looks like there are 16 different failures in the data.  
# 
# > The data within each test file is continuous, but the test files do not represent a continuous segment of the experiment; thus, the predictions cannot be assumed to follow the same regular pattern seen in the training file. 
# 
# The test data, according to the description, is made up of random segments cut from a strip of data.  Basically, it's a warning that if you feed all the training data into the model straight, it may well pick up on some underlying pattern.
# 
# 

# **First Questions**
# 
# The first question we need to answer is how to handle the occurance of a failure.  Looking at the EDA failures seem to reset the clock on time to failure (by letting stress out of the rock).  However they may also change to rate of time until the next failure.  If we were simply to iterate through the time series in batches of 150,000 approximately 0.4% of the data would have a fault occure during it?  Is this significant?  Probably not.
# 
# The next question, that would be helpful to answer, is what is a labratory earthquake?  Is this simulated data?  Earthquakes (very small) caused by fracking or some other form of explosion?
# 
# Is all data in the test file from the same signal ID?  How many signal IDs are in each test file?

# **First Challenges**
# We need to reduce the dimensionality of this data and create synthetic data poins.  
# 
# 1.  We need to be able to randomly pull a continous series of 150,000 data points (one entry) from the file.  Records the final time to failure datapoint.  Trash the rest of the time to failure data points, transpose the time rows to columns, and then append on the target (time to failure) datapoint.
# 
# 2.  This is a big data problem as the file to too large to load into memory all by itself.
# 
# 3.  We need to be able to reduce the dimensionality of the data.  While some forms of dimensionality reduction only require one row of data (for example computing its mean), others requires multiple rows.  This means we would need to to pull a statistically representative sample -- probably at least 500 of the new rows (1500 or everything would be better).  Create the model, export it and reimport it (so that we only need to crunch the numbers once).  Basically, create a PCA (or what have you) model once and resuse it every time a kernel is submitted.
# 
# 4.  The challenges with the above are big data (handling these very large rows), as well as how to export, save and import models.
# 
# 5.  Once we have a method for reducing the dimensionality of the data, we need to create a new data file with about 60,000 dimensionality reduced rows by randomly pulling batches of 150,000 data points and performing transforms on them.
# 

# **Modeling Challenges**
# 
# Once we have a new, dimensionality reduced, data file, we need to apply modeling.  The answer is almost, certainly, a DNN, though possibly in an ensemble with a statistical approach (SVM).  
# 
# This suggests that we might be looking at a two languages challenge.  Preprosessing data in R, then modeling it in Python/Keras.

# **Resources**
# 
# The following resources might be helpful.  
# 
# [R TSrepr package](https://petolau.github.io/TSrepr-clustering-time-series-representations/)
# 
# [Basic Info on Big Data with R](http://http://www.columbia.edu/~sjm2186/EPIC_R/EPIC_R_BigData.pdf)

# In[ ]:




