#!/usr/bin/env python
# coding: utf-8

# 
# # **Hypothesis test and p-values - Exploring evolution data**
# 
# 
# **Introduction**
# 
# Statistics is usually a weak spot for many starting on Data Science, and I don't need to reinforce how fundamental this discipline is on the particular field. The content you will go through on this Kernel comes originally from the "Statistical Thinking II" chapter on  [**Data Camp**](http://www.datacamp.com/home), a fantastic platform leading Data Science Education and where I found myself devoted on personal studies to aquire my python skills as well as the "data logical thinking". [**Data Camp**](http://www.datacamp.com/home) solves the Data Science curriculum problem for beginner data enthusiasts, in case you are in this position, go check them out!
# 
# The working dataset here comes from [Dryad](http://datadryad.org/resource/doi:10.5061/dryad.g6g3h) repository. The reaserch conduced by Peter and Rosemary Grant has shown that these changes in populations can happen very quickly. The British couple, both evolutionary biologists at Princeton University, have spent six months every year capturing, tagging, and taking blood samples from finches (small birds) on the island **since 1973**. On this kernel we will use part of the data compiled to assess hypothesis tests while practicing the best practices of exploratory data analysis.
# 
# Original publication:
# 
# *         Grant, PR, Grant, BR (2014) 40 years of evolution: Darwin's finches on Daphne Major Island. Princeton: Princeton University Press. http://www.worldcat.org/oclc/854285415
# 
# **This notebook is structured as follows:**
# 
#     1. Loading libraries and data
#     2. Exploratory data analysis (EDA)
#     3. Statistical analysis
#     4. Conclusions and references on correlation and p-value analysis
#     
#    **Questions to be answered**
# 
#     1. Estimate the difference of the mean beak depth of the G. scandens samples from 1975 and 2012 and report a 95% confidence interval.
#     2. Hypothesis test: Are beaks deeper in 2012?
#     3. Hypothesis test: Are lenght beaks length and depth correlated?

# ## 1. Loading libraries and data:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # advanced data visualization

sns.set() # Using sns graphing style

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading data

# Compiled data
data_1975 = pd.read_csv('../input/beaks-1975/finch_beaks_1975.csv')
data_2012 = pd.read_csv('../input/beaks-2012/finch_beaks_2012.csv')

# Summarized data
df = pd.read_csv('../input/dryad-darwins-finches/Data for Dryad.txt', sep='\t')


# In[ ]:


# Checking df.head from summarized data
df.head()


# In[ ]:


# Preparing working vectors on subsets selecting scandens species from 1975 and 2012

data_2012.columns = data_1975.columns
data_2012['year'] = 2012
data_1975['year'] = 1975
concat_df = pd.concat([data_1975, data_2012])

# Generating subsets
sc_1975 = concat_df[(concat_df['species'] == 'scandens') & (concat_df['year'] == 1975)]
sc_2012 = concat_df[(concat_df['species'] == 'scandens') & (concat_df['year'] == 2012)]


# ## 2. Exploratory data analysis (EDA):

# In[ ]:


# Plotting scandens distributions

concat_sc = concat_df[concat_df['species'] == 'scandens']
_ = plt.figure(figsize=(8,6))
_ = sns.swarmplot(x='year', y='Beak depth, mm', data=concat_sc, size=4);
_ = plt.ylabel('Beak depth (mm)')
_ = plt.xlabel('Year')
plt.tight_layout()


# In[ ]:


# Plotting fortis distributions
concat_sc = concat_df[concat_df['species'] == 'fortis']
_ = plt.figure(figsize=(8,6))
_ = sns.swarmplot(x='year', y='Beak depth, mm', data=concat_sc, size=4);
_ = plt.ylabel('Beak depth (mm)')
_ = plt.xlabel('Year')
plt.tight_layout()


# In[ ]:


# Defining our functions from Data Camp's power toolbox

# ECDF stands for empirical cumulative distribution function.  
def ecdf(data):
    """
    Compute ECDF for a one-dimensional array of measurements.
    
    It assigns a probability of to each datum (x axis), orders the data from smallest to largest in value, 
    and calculates the sum of the assigned probabilities up to and including each datum (x axis).
    """
    
    # Number of data points: n
    n = len(data)
    
    # x-data for the ECDF: x
    x = np.sort(data)
    
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    
    return x, y

def bootstrap_replicate_1d(data, func):
    """
    Compute and return a bootstrap replicate, which is a statistical value according to parameter 'func'
    on a randomized numpy array based on the given 'data'
    """
    return func(np.random.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1):
    """
    Draw 'size' numbers of bootstrap replicates.
    """
    
    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


# In[ ]:


# Plotting an ECDF for years 1975 and 2012
x_1975, y_1975 = ecdf(data_1975['Beak depth, mm'])
x_2012, y_2012 = ecdf(data_2012['Beak depth, mm'])
_ = plt.figure(figsize=(10,8))
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margings
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('Beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('2012', '1975'), loc='lower right')


# The Beak depth differece is clear after plotting the ECDF. Initial conclusions are that over time the average beak depth increased.

# ## 3. Statistical analysis (EDA):
# 
# #### Question 1 - Estimate the difference of the mean beak depth of the G. scandens samples from 1975 and 2012 and report a 95% confidence interval.

# In[ ]:


# Making aliases
bd_1975 = np.array(sc_1975['Beak depth, mm'])
bd_2012 = np.array(sc_2012['Beak depth, mm'])

# Computing confidence intervals

"""if we repeated the measurements over and over again, 
95% of the observed values would lie withing the 95% confidence interval"""

# Compute the observed difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, size=10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])

# Print the results
print('difference of means =', mean_diff, 'mm')
print('95% confidence interval =', conf_int, 'mm')


# As **Confidence Interval** definition we have: A range of values so defined that there is a specified probability that the value of a parameter lies within it.
# 
# Therefore, since we're estimating the mean value, our 95% confidence interval (estimated by the np.percentile function) tells us that in case we could gather new samples for this use case there is a 95% probability that the new mean value will lie between 0.061 and 0.391.

# #### Question 2 - Hypothesis test: Are beaks deeper in 2012?
# 
# For a well conduced Hypothesis Test you need to follow some steps, which I will explain with our practical case:
#     - Null Hypothesis (Ho): The beak depth means are equal from 1975 and 2012, the different values observed are due to a random chance;
#     - Test statistics: Different between means;
#     - P-value: The probability of finding values as similar to the observed mean difference or higher;

# In[ ]:


# Compute mean of combined data set: combined_mean
combined_mean = np.mean(np.concatenate((bd_1975, bd_2012)))

# Shift the samples
# This is done because we are assuming in our Ho that means are equal!
bd_1975_shifted = bd_1975 - np.mean(bd_1975) + combined_mean
bd_2012_shifted = bd_2012 - np.mean(bd_2012) + combined_mean

# Get bootstrap replicates of shifted data sets
bs_replicates_1975 = draw_bs_reps(bd_1975_shifted, np.mean, size=10000)
bs_replicates_2012 = draw_bs_reps(bd_2012_shifted, np.mean, size=10000)

# Compute replicates of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute the p-value: p
p = np.sum(bs_diff_replicates >= mean_diff) / len(bs_diff_replicates)

# Print p-value
print('p-value = {0:.4f}'.format(p))


# Having a low p-value leads us against our initial null hypothesis. In other words the probability of finding the same difference between means as observed or higher values is so low that it's more likely to assume that means are not equal at all. This conclusion contributes to our previous data visualization stating that beak depths increased over time (mean shifted to the right on ECDF plot).

# #### Question 3 - Hypothesis test: Are lenght beaks length and depth correlated?
# 
# Following the same steps for a well conduced Hypothesis Test:
#     - Null Hypothesis (Ho): The beak lenght and depth are not correlated at all in 1975;
#     - Test statistics: Pearson correlation;
#     - P-value: The probability of finding values as similar to the observed Pearson correlation or higher;

# #### Starting, as always, with visualization
# According to the data distribution in a scatter plot we could get an initial sense of the data guessing possible correlations.
# 
# As expected, it is possible to guess some sort of linear positive correlation for the data.

# In[ ]:


_ = plt.figure(figsize=(10,8))
_ = plt.scatter(x=sc_1975['Beak length, mm'], y=sc_1975['Beak depth, mm']);
_ = plt.xlabel('Beak length (mm)')
_ = plt.ylabel('Beak depth (mm)')

# Compute observed correlation: obs_corr_1975
obs_corr_1975 = sc_1975[['Beak length, mm', 'Beak depth, mm']].corr().iloc[0, 1]
print("Pearson correlation =", obs_corr_1975)


# In[ ]:


# The bootstrap test will be done by permuting Beak depth attribute while keeping Beak lenght the same, good practice for correlation bootstrap
# Note that the .iloc[] is used to extract the correlation value ignoring the identity values from the correlation matrix

# Initialize permutation replicates: perm_replicates
perm_replicates = np.empty(10000)

# Draw replicates
for i in range(10000):
    # Permute illiteracy measurments: illiteracy_permuted
    beak_depth_permuted = np.random.permutation(sc_1975['Beak depth, mm'].as_matrix())

    # Compute Pearson correlation
    # Note that here np.corrcoef is used since we're working with arrays instead with a Data Frame
    # Therefore we use [0, 1] to select the correct correlation value from the correlation matrix
    perm_replicates[i] = np.corrcoef(beak_depth_permuted, sc_1975['Beak length, mm'].as_matrix())[1, 0]

# Compute p-value: p
p = np.sum(perm_replicates >= obs_corr_1975)/len(perm_replicates)
print('p-val =', p)


# The p-value = 0 means that in 10000 replicates we could not see a single value that is similar or higher than the observed correlation, which goes against our Null Hypothesis (Ho). 
# 
# Therefore, it is possible to conclude that the correlation observed is not random and should be taken into consideration.

# ## 4. Conclusions and references on correlation and p-value analysis:
# 
# As I mentioned, this was one of my favorite topics so far, and even though most people go directly to machine learning and deep learning the statistical concepts here explored are a crucial part of the foundation to build good models and optimize existing ones.
# 
# #### P-value:
# Initially, I thought that p-value analysis was a boolean thing, it's either less than 0.01 or "statistically significant" or higher meaning "it happens due to randomness". This is stressed more than one time by [**Data Camp**](http://www.datacamp.com/home), such analysis should not be carried in a True or False scenario, the p-value is way more complex than most people think. 
# 
# #### Correlation:
# After the course, correlation techniques started being part of my Exploratory Data Analysis (EDA) toolbox. Correlations address the necessity of increasing modeling efficience while enhancing the big picture of data understanding when used together with visualizations. Phenomenon such as multicollinearity effects were completely new to me while being simple and powerful concepts.
# 
# #### References:
# During my studies I found very good references regarding both topics that I would like to share as a further reading:
# 
# * [Science Isn't Broken](https://fivethirtyeight.com/features/science-isnt-broken/#part1) article by [FiveThirtyEight](http://fivethirtyeight.com) shown a very complicated scenario where p-values are so determinant for publishing a scientific paper nowadays that scientists are hacking statistical analysis in order to achieve significance. They provide an interactive dashboard where you can try it yourself "hacking data for being published". The "Science's magic number" is also discussed in the same aspect by [PBS](http://www.pbs.org/wgbh/nova/next/body/rethinking-sciences-magic-number/?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com).
# * The "Introduction to Correlation" blog from [datascience.com](http://https://www.datascience.com/learn-data-science/fundamentals/introduction-to-correlation-python-data-science) was an outstanding source for going deeper from covariance passing by Pearson and Spearman correlations with simple examples. Special note to their explanation regarding "Correlation and Causation" where according to data there was a high correlation for babies in Germany being delivered by storks. 
# The same line of reasoning, and critic joke behind, is explored by [Spurious Correlations](http://www.tylervigen.com/spurious-correlations), in case you haven't heard of them, be mindful of their last analysis where there is a 99.76% correlation between US spendings on science, space, and technology and Suicides by hanging, strangulation, and suffocation. Correlation implies causation? Not always!  
# 
