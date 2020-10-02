#!/usr/bin/env python
# coding: utf-8

# # American Bellwethers
# 
# A 'bellwether', defined by Wikipedia, is one that indicates or leads trends.  My goal with this project was to identify [PUMAs](https://en.wikipedia.org/wiki/Public_Use_Microdata_Area) (public use microdata areas) that act as the best bellwethers for the rest of America.  In other words, I wanted to see which PUMAs could generate the best predictive models for all of the other PUMAs in the country.  
# 
# For this analysis I chose to build models to predict occupations of surveyed individuals, as this was one feature of the person dataset.  In a more complex analysis, one would ideally use secondary datasets (perhaps primary election information or purchasing information) to test whether the 'bellwether PUMAs' identified here are actually any better than the rest of the country in predicting trends and outcomes.
# 
# My secondary goals for this project, which were definitely more succesful, were simply to get more accustomed to the machine learning Python library [sklearn](http://scikit-learn.org/stable/) and with Python programming in general.  I'm definitely a Python/programming novice, so I hope readers will excuse any obviously flawed practices or code.  Any comments or suggestions are more than welcome!
# 
# ### general analysis pipeline:
# 1. Load the full datasets from the CSV files provided by Kaggle [puma_load.py](https://gitlab.com/atrexler/svm_pumas/blob/0cb16af4be2f7bf25b3e6de82bcf5179a04a75ac/puma_alt_load.py).  I removed the replicate weights columns and replaced NaN values with -1 to simplify analysis later.  Sparse features in the dataset were not scaled but all other features except occupation code (SOCP) were scaled with sklearn preprocessing scale function.
# 2. Reduce the dimensionality of the data with PCA.  For some of my analysis I used the full dataset as well but the majority was performed using a 20 component PCA-generated dataset.  I actually found that the PCA-reduced dataset was better at building models than the full ~200 feature dataset.  Implementation of the PCA was done with just the standard sklearn PCA module in [puma_load.py](https://gitlab.com/atrexler/svm_pumas/blob/0cb16af4be2f7bf25b3e6de82bcf5179a04a75ac/puma_alt_load.py)
# 3. [puma_parallel.py](https://gitlab.com/atrexler/svm_pumas/blob/0cb16af4be2f7bf25b3e6de82bcf5179a04a75ac/puma_parallel_pub.py) builds models with sklearn support vector machine (SVM) algorithm.  I chose SVM after a bit of initial fiddling and following the [algorithm selection guide](http://scikit-learn.org/stable/tutorial/machine_learning_map/) provided by sklearn.  In most rounds of the analysis I built a model per-PUMA so I used joblib.Parallel to achieve some parallelization to speed this process up.
# 4. Use each model to fit a subset of other PUMAs and predict occupations.  Calculate mean scores to evaluate how effective the model was.  I then ranked PUMAs by their mean scores to evaluate how well each PUMA was at predicting behavior (occupations) in the other PUMAs.
# 
# 
# [puma_parallel.py](https://gitlab.com/atrexler/svm_pumas/blob/0cb16af4be2f7bf25b3e6de82bcf5179a04a75ac/puma_parallel_pub.py) is the real workhorse code of my analysis.  At the top is the test_puma function, which takes an input PUMA identifier and trains a model using the full dataset from that PUMA.  I use the SVM machine learning algorithm from sklearn with simple inputs, a linear kernel and only 10000 max iterations.  The function then tests that model against 100 randomly selected PUMA (the subpuma variable) and computes scores for how well the model does.
# 
# test_puma is called using Parallel from joblib to achieve some parallelization.  Parallel creates a number of workers that each carry out the test_puma call using a different input PUMA.
# 
# **TL;DR: More data is always better.**
# 
# 
# # American Bellwether Results
# 
# Initially I tried to preselect PUMAs based on factors I thought might affect how good a model they produce, but then I realized that wasn't working very well so I just tested every PUMA.  Here is a simple plot of sample number in each PUMA versus the mean score of all the predictions that PUMA made.
# 
# ![figure1](https://gitlab.com/atrexler/svm_pumas/raw/master/readme_figure1.png)
# 
# **So, lesson number one, data is king!**  PUMAs with more datapoints clearly produced better models that more accurately predicted occupations in other PUMAs.  No surprise there, really.  Moving on from that, I wanted to try to put the PUMAs on a more level playing field, so I extracted a random subset of datapoints (1000) from each PUMA with at least that many datapoints, and performed the model building and analysis again with this smaller dataset.  Here is the same style plot as Figure 1, showing the performance of each PUMA, using a normalized dataset of 1000 samples, at predicting occupations.
# 
# ![figure2](https://gitlab.com/atrexler/svm_pumas/raw/master/readme_figure2.png)
# 
# **Lesson two, data is still king!**  Somewhat more surprisingly here was that even when I normalized the number of samples used to create each model (1000 datapoints), the PUMAs with more samples still performed better, though the correlation is clearly lower.  This suggests that better sampling leads to better datasets, which of course makes plenty of sense.  
# 
# Next, I wanted to get some insight into what makes the top PUMAs better than the bottom PUMAs so I extracted the top and bottom 100 and pulled out demographic data on them.  The [full dataset](http://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict13.txt) is dozens of features, ranging from income and ethnicity to things like how many vehicles the person owns and when they leave for work.  Here are some of the more interesting things I found.  
# 
# 
# First, what are the top and bottom PUMAs?
# ### Highest average predictor scores:  
# Muskegon County (MI)  
# Stamford & Greenwich Towns (CT)  
# Apalachee Region (FL)  
# Panhandle Regional Planning Commission (TX)  
# Cook County (North)--Niles & Evanston Townships (IL)  
# Central Shenandoah Planning District Commission (VA)  
# East Central Wisconsin (WI)  
# Weld County (South Central) (CO)  
# Walton, Washington, Holmes & Bay Counties (FL)  
# District of Columbia (Northeast) (DC)  
# 
# ### Lowest average predictor scores:
# Los Angeles County (Central)--East Los Angeles  
# NYC-Brooklyn Community District 3--Bedford-Stuyvesant  
# Prince George's County (Central)--Seat Pleasant City, Capitol Heights Town & Landover  
# St. Louis County (Inner Ring North)  
# Riley, Geary & Pottawatomie Counties--Manhattan City  
# Maricopa County--Avondale (Central) & Litchfield Park (Central) Cities  
# Coordinating & Development Corporation 1--Shreveport City (North) (LA)  
# Collin County (Southwest) (TX))  
# Los Angeles County (Central)--Bell Gardens, Bell, Maywood, Cudahy & Commerce Cities  
# Chicago City (South)--Auburn Gresham, Roseland, Chatham, Avalon Park & Burnside 
# 
# Geographically these are fairly widely distributed, though you'll notice right away that PUMAs within cities tended to score lower.  I initially thought that city PUMAs would be better predictors overall because they would tend to be more diverse in a variety of feature spaces.
# 
# ### relative best and worst bellwethers
# 
# ![figure5](https://gitlab.com/atrexler/svm_pumas/raw/master/readme_fig5.png)
# 
# **Tulare County, California ()** had the lowest number of samples amongst the highest scoring PUMAs, making it the relative best predictor PUMA.  On the other hand, **Mower, Steele, Freeborn, and Dodge Counties, Minnesota ()** had the largest number of samples in the worst predictor class, making it the worst relative predictor PUMA.
# 
# ### what else can we learn?
# 
# **Diversity in occupation (occu diver) and how many weeks worked in the past year (WKW) were either negatively correlated or only mildly correlated with average predictor score!**  This was surprising.  You'd expect things like a diverse spread of occupations and an assessment of how much people worked in the past year to correlate with occupation prediction ability.
# 
# In the dataset, about 500 different occupations or categories of occupation are coded with 6 digit codes.  To quickly assess the diversity of the occupations within a given PUMA, I created occu_diver.  Occu_diver takes a dict of occupations with # of samples with that occupation as the value.  
# ```
# occu_diver= ( np.sum(dict.values) / np.max(dict.values) ) -1
# ```
# So a 0 value means everyone in a PUMA has the same occupation-- no diversity.  Higher values indicate greater diversity of occupation types.  How many weeks worked in last year is self-explanatory, and I plot below the fraction of people in the PUMA that worked at least 27 weeks in the past year.
# 
# ![figure 3](https://gitlab.com/atrexler/svm_pumas/raw/master/readme_figure3.png)
# 
# **Lesson three, stay in school, kids.**  Weirdly, the number of persons under 16 correlated reasonably well with average predictor scores, while, not-so-weirdly the fraction of people in the PUMA that finished high school or equivalent degree programs correlated more strongly.  Number of persons with college degrees didn't correlate as strongly as the high school level.  
# 
# ![kids and school](https://gitlab.com/atrexler/svm_pumas/raw/master/readme_figure4.png)
# 
# **Finally, apparently with age and money come PUMA predictor wisdom**.  Average age and income were also slightly correlated with average PUMA predictor.  I suspect the reason for this is that the average incomes and ages in the better predictor PUMAs cluster with the national averages of these values, calculated from the full Kaggle dataset and plotted in the big red dots.  
# 
# 
# ![money and age](https://gitlab.com/atrexler/svm_pumas/raw/master/readme_figure6.png)
# 
# As a summary, here's a correlation matrix of everything I looked at.  Its by no means comprehensive, and to really figure out what makes the good predictor PUMAs good and the bad ones bad, a much deeper analysis is necessary.  There are well over 100 features that could be relevant in there!  The correlation matrix shows some interesting trends that make some sense as well:
# 1.  Income correlates well with e ducation level and how much people worked in the previous year.
# 2.  Number of weeks worked  in the past year correlates with income, occupation, and negatively correlates with how many kids less than 16 are in the PUMA.  
# 
# 
# ![heatmap all features](https://gitlab.com/atrexler/svm_pumas/raw/master/readme_all.png)
