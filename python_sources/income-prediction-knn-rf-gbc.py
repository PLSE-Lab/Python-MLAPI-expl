#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author: Caleb Woy

import numpy as np # linear algebra
import pandas as pd # data processing
from scipy.stats import kurtosis, skew # checking distributions
import scipy.stats as stat # plotting, mostly
import scipy.spatial.distance as sp # Computing distances in kNN
import matplotlib.pyplot as pl # plotting
import seaborn as sb # plotting
import sklearn as sk # Regression modelling
import os # Reading data
import sklearn.model_selection # train and test splitting
import matplotlib.pylab as plt # plotting hyperparamter cost curves
import time # timing custom knn model
from sklearn.model_selection import RandomizedSearchCV # tuning hyperparams for complex models
from sklearn.metrics.scorer import make_scorer # defining custom model evaluation function
from sklearn.ensemble import GradientBoostingClassifier as gb # modelling
from sklearn.neighbors import KNeighborsClassifier # modelling
from sklearn.model_selection import GridSearchCV # tuning hyperparams for simple models
from sklearn.ensemble import RandomForestClassifier as rf # modelling
from sklearn.svm import SVC as sv # modelling


# In[ ]:


# So we can see some interesting output without truncation
pd.options.display.max_rows = 1000

path_to_data = "/kaggle/input/"

# Loading the training and test data sets into pandas
train_original = pd.read_csv(path_to_data + "/adult.data", names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                                                  'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                                                  'cap-gain', 'cap-loss', 'hrsperwk', 'native', 'label'])
test_original = pd.read_csv(path_to_data + "/adult.test", skiprows=1, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                                                             'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                                                             'cap-gain', 'cap-loss', 'hrsperwk', 'native', 'label'])

# Combining the training and test sets
frames = [train_original, test_original]
data_original = pd.concat(frames)

# print the head
data_original.head()


# # ***Business Understanding***
# 
# Recorded originally by the US census bureau for the purpose of determining the correct number of House representatives per state via the 1990 census survey. Extracted by from the 1994 census database by Barry Becker under the following conditions: "((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))." Each record represents a single American Citizen that is > 16 years of age, has a difference between their Total income and Adjustments to income of > 100, has a fnlwgt attribute > 1, and works more than 0 hours per week.
# 
# The data set has been curated as a sample of the working population in the United States, for the purpose of predicting whether an individual makes > 50K per year. 
# 
# The feature "fnlwgt" was added by the dataset authors as a controlled estimate of certain socio-economic effects that take individual state population distributions into account. The controls accounted for are:
# 
#     | 1.  A single cell estimate of the population 16+ for each state.
#     | 2.  Controls for Hispanic Origin by age and sex.
#     | 3.  Controls by Race, age and sex.
#     
# Original data converted, by the data set authors, as follows:
# 
#     | 1. Discretized agrossincome into two ranges with threshold 50,000.
#     | 2. Convert U.S. to US to avoid periods.
#     | 3. Convert Unknown to "?"
#     | 4. Run MLC++ GenCVFiles to generate data,test.
#     
# Here, numerical data (gross income) was simplified into binary categories of > 50K or <= 50K. MLC++ GenCVFiles is used to randomly split the data into training and tesing set for the purpose of ML application.
# 
# I have edited the files adult.data and adult.test to include column names.

# # ***__Data Understanding & Data Processing__***

# #######################################################################################################################################################
# 
# ### **label**:     /Categorical. Factor levels include: > 50K, <= 50K
# 
# Meaning: whether the indiviual makes more or less than 50K per year

# In[ ]:


feature_name = 'label'

# Checking counts per designation
data_original.groupby(feature_name).count()


# In[ ]:


# the output here is erroneously grouped into 4 rows, I need to remove the period from every label in the test set to get an accurate count.
data_original[[feature_name]] = data_original[[feature_name]].replace([" <=50K.", " >50K."], [" <=50K", " >50K"])
data_original.groupby(feature_name).count()


# In[ ]:


# better, now we can view a summary
data_original[[feature_name]].describe()


# In[ ]:


# So, there are 48842 values in the label column. There are 2 factor levels for the column. The most common label is '<= 50K' and it occurs 37155 times, 
# roughly 3/4 of the individuals.
# Now we'll check for missing values.
boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# No missing values for our label, that's good. I'll move onto the next feature.


# #######################################################################################################################################################
# 
# ### **age**    /Continuous. 
# 
# Meaning: the integer value age of the individual

# In[ ]:


feature_name = 'age'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


#checking for missing values
boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# No missing values on age, let's check skewness and kurtosis
print(f'Skewness: {skew(data_original.age)}')
print(f'Kurtosis: {kurtosis(data_original.age)}')


# In[ ]:


# The sample distribution of ages appears to be slightly right skewed with very slight negative kurtosis. This may need transformed for future modelling.
# Let's visualize this one to confirm the skewness.
x = data_original.age
pl.hist(x, bins=80)
pl.xlabel('Age')
pl.ylabel('Frequency')


# In[ ]:


# The values at the end of the right tail are definitely outliers however they're meaningful in our analysis (the elderly are important too). There don't appear to be any obvious
# errors caused by typos (like 500 or 0) 

# I'll create a new feature by taking the log 
# I'll create a new feature by centering with the z score
# I'll create a new feature by taking the log and centering with the z score

logage = np.log(data_original['age'])
data_original['log_age'] = logage

mean = np.mean(data_original['age'])
stdev = np.std(data_original['age'])
data_original['age_ZCentered'] = (data_original['age'] - mean) / stdev

mean = np.mean(logage)
stdev = np.std(logage)
data_original['log_age_ZCentered'] = (logage - mean) / stdev

x = data_original['log_age_ZCentered']
pl.hist(x, bins=80)
pl.xlabel('log_age_ZCentered')
pl.ylabel('Frequency')

# checking for success
data_original.head()

#all good


# #######################################################################################################################################################
# 
# ### **workclass**    /Categorical. Factor levels include: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#                   
# Meaning: This feature explains the general category of the economy the individual works within.

# In[ ]:


feature_name = 'workclass'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# Roughly 3/4 of our individuals appear to be working in the private sector. Describe returned that there are 9 factor levels in this feature when we know there are actually 
# only 8. so there must be missing values in this feature. Let's check.

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# There are 2799 ? values currently. None of them are Null or NaN values, so that's good. We have a few options here. The first is to impute the mode level (Private). 
# The second is to check if there are any other features here that might explain variation in with workclass, then if so, predict the missing workclass values. The third is 
# leave the ? value in as a placeholder unkown value and predict based on the effect of the level as we would any other feature.

# I'll make some boxplots to see if there's any explainable variation.

sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["workclass"] )
sb.boxplot( x=data_original["fnlwgt"], y=data_original["workclass"] )
sb.boxplot( x=data_original["education-num"], y=data_original["workclass"] )
sb.boxplot( x=data_original["cap-gain"], y=data_original["workclass"] )
sb.boxplot( x=data_original["cap-loss"], y=data_original["workclass"] )
sb.boxplot( x=data_original["hrsperwk"], y=data_original["workclass"] )


# In[ ]:


# None of these give off the appearance of explainatory variation that I'm looking to test with ANOVA so I'll impute the mode (Private) for the missing values. This can
# always be undone later during the modelling fase should we like to check how well we can predict with an unkown value effect.

# Checking the original counts at each factor level
data_original.groupby(feature_name).count()

# Making the replacement and recalculating the values
data_original[[feature_name]] = data_original[[feature_name]].replace([" ?"], [" Private"])
data_original.groupby(feature_name).count()

# All good.


# #######################################################################################################################################################
# 
# ### **fnlwgt**    /Continuous. 
# 
# Meaning: The feature "fnlwgt" was added by the dataset authors as a controlled estimate of certain socio-economic effects that take individual state population distributions into account.

# In[ ]:


feature_name = 'fnlwgt'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# These are large numbers, any predictive model we apply on this data set would befit from some regularization here in the future. The max is exponetially larger than the mean.
# High values in fnlwgt will need investigated.

# Let's check for missing values.
boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# None, the data set authors created this feature so that should have been expected. Thanks authors!

# Checking skewness and kurtosis.
print(f'Skewness: {skew(data_original[feature_name])}')
print(f'Kurtosis: {kurtosis(data_original[feature_name])}')


# In[ ]:


# The fnlwgt column has some strong right skew and high positive kurtosis. It should look like a big spike on the left side of the distribution.

# Let's visualize to confirm.
x = data_original[feature_name]
pl.hist(x, bins=100)
pl.xlabel('fnlwgt')
pl.ylabel('Frequency')


# In[ ]:


# Yup. This feature would benefit from a log transformation. 

# Creating new features. I'll take the log transform then standardize that using the z-score.
logfnlwgt = np.log(data_original['fnlwgt'])
data_original['log_fnlwgt'] = logfnlwgt

mean = np.mean(data_original['fnlwgt'])
stdev = np.std(data_original['fnlwgt'])
data_original['fnl_wgt_ZCentered'] = (data_original['fnlwgt'] - mean) / stdev

mean = np.mean(logfnlwgt)
stdev = np.std(logfnlwgt)
data_original['log_fnl_wgt_ZCentered'] = (logfnlwgt - mean) / stdev

x = data_original['log_fnl_wgt_ZCentered']
pl.hist(x, bins=100)
pl.xlabel('log_fnl_wgt_ZCentered')
pl.ylabel('Frequency')

# checking for success
data_original.head()

# all good


# In[ ]:


#Let's view the largest values of the distribution. 
data_original.nlargest(10, ['log_fnl_wgt_ZCentered']) 


# In[ ]:


# Regarding the outliers at the tail of fnlwgt, none of these appear to be abnormal. We can't know forsure 
# without knowing how fnlwgt was calulated, yet the consistent increasing of the feature values up to the max appears systematic and not erroneous. I won't do anything
# special about them.


# #######################################################################################################################################################
# 
# ### **Education**    /Categorical. Factor levels include: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#                   
# Meaning: How much schooling the individual has completed.

# In[ ]:


feature_name = 'education'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# Seeing the correct number of unique factor levels here so likely no missing values. HS-grad is the most common level
# Let's confirm:

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# Yup. Looking good here.


# #######################################################################################################################################################
# 
# ### **education-num**    /continuous. 
# 
# Meaning: The number of years of education completed by and individual.

# In[ ]:


feature_name = 'education-num'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# Mean value of ~10. Max and min are prestty evenly spread.
# Checking for missing values:

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# Let's look at the skewness and kurtosis:

print(f'Skewness: {skew(data_original[feature_name])}')
print(f'Kurtosis: {kurtosis(data_original[feature_name])}')


# In[ ]:


# Slight positive kurtosis, slight left skew. Let's visualize:

x = data_original[feature_name]
pl.hist(x, bins=16)
pl.xlabel('education-num')
pl.ylabel('Frequency')


# In[ ]:


# This distribution appears bimodal. Likely due to the effect of college. This might make the categorical feature (education) more useful to us.

# I'll scale this by transforming it with the Z-score.

mean = np.mean(data_original[feature_name])
stdev = np.std(data_original[feature_name])
education_num_ZCentered = (data_original[feature_name] - mean) / stdev

# Visualizing:
x = education_num_ZCentered
pl.hist(x, bins=16)
pl.xlabel('education_num_ZCentered')
pl.ylabel('Frequency')


# In[ ]:


# Now to replace the original feature with the transformed version.

data_original['education_num_ZCentered'] = education_num_ZCentered

# Checking:
data_original.head()


# #######################################################################################################################################################
# 
# ### **marital-status**    /Categorical. Factor-levels include: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
#                        
# Meaning: Whether the individual is married or divorced, etc. Interesting because houses with two income sources will be different than one.

# In[ ]:


feature_name = 'marital-status'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# There are 7 unique factor levels present in our distribution. So, likely no missing values. We can confirm.

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# #######################################################################################################################################################
# 
# ### **occupation**    /Categorical. Factor-levels include: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
#                        
# Meaning: The field in which the individual works.

# In[ ]:


feature_name = 'occupation'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# There are only supposed to be 14 factor levels so they're definitely some missing values here.

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# 2809 missing values. I'll impute the mode value (Prof-specialty)

data_original.groupby(feature_name).count()

# Making the replacement and recalculating the values
data_original[[feature_name]] = data_original[[feature_name]].replace([" ?"], [" Prof-specialty"])
data_original.groupby(feature_name).count()

# Worked fine.


# #######################################################################################################################################################
# 
# ### **relationship**    /Categorical. Factor Values: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#                      
# Meaning: The type of relationship the individual is in.

# In[ ]:


feature_name = 'relationship'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# Seeing 6 unique factor levels as expected. Most common level is Husband.
# Confirming no missing values

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# Good.


# #######################################################################################################################################################
# 
# ### **race**    /Categorical. Factor levels include: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#              
# Meaning: The race of the individual.

# In[ ]:


feature_name = 'race'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# Seeing 5 unique factor levels as expected. Most common level is White.
# Confirming no missing values

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# Good.


# #######################################################################################################################################################
# 
# ### **sex**    /Categorical. Factor levels include: Female, Male. 
# 
# Meaning: The sex of the individual.

# In[ ]:


feature_name = 'sex'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# Seeing 2 unique factor levels as expected. Most common level is Male.
# Confirming no missing values

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# Good.


# #######################################################################################################################################################
# 
# ### **cap-gain**    /Continuous. 
# 
# Meaning: Dollars gained by the individual's investments during the year.

# In[ ]:


feature_name = 'cap-gain'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# The summary here tells us the mean gain is about a thousand dollars. The distribution appears to be dramatically right skewed and is likely mostly (0) values.
# Let's comfirm by checking skew and kurtosis.

print(f'Skewness: {skew(data_original[feature_name])}')
print(f'Kurtosis: {kurtosis(data_original[feature_name])}')


# In[ ]:


# Yeah . . . we'll be transforming this one. First, Let's check for missing values.

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# None. That's helpful. Time to visualize this one.

x = data_original[feature_name]
pl.hist(x, bins=100)
pl.xlabel('cap_gain')
pl.ylabel('Frequency')


# In[ ]:


# Before I transform this, I want to investigate the outlier here that's near 100K.

data_original.nlargest(10, [feature_name]) 

# I've actually viewed the top 60 largest values here but I set the code back to outputting the top 10 to keep this report cleaner.
# The trend shown in the top 10 is the same as the top 60, all are labeled > 50k. 


# In[ ]:


# I think I want to create a new feature here. A simple binary feature specifying whether the individual made > 50K in capital gains alone.

data_original['cap-gains50k'] = data_original.apply(lambda x: True if x[feature_name] > 50000 else False, axis=1).astype('category')

# Checking that it worked:
data_original.nlargest(10, [feature_name]) 


# In[ ]:


# That should be a really significant factor in whatever model we might use to predict our label.
# Now I'll transform the original cap-gain feature. Taking a log of 0 will produce NaNs so I'll transform the feature to log(cap-gains + 1) and then I'll scale It with the z-score.

log_cap_gain = np.log(data_original[feature_name] + 1)
data_original['log_cap_gain'] = log_cap_gain

# Visualizing:
x = log_cap_gain
pl.hist(x, bins=100)
pl.xlabel('log_cap_gain')
pl.ylabel('Frequency')


# In[ ]:


# Now scaling by Z-score:

mean = np.mean(data_original[feature_name])
stdev = np.std(data_original[feature_name])
data_original['cap_gain_ZCentered'] = (data_original[feature_name] - mean) / stdev

mean = np.mean(log_cap_gain)
stdev = np.std(log_cap_gain)
data_original['log_cap_gain_ZCentered'] = (log_cap_gain - mean) / stdev

# Visualizing:
x = data_original['log_cap_gain_ZCentered']
pl.hist(x, bins=100)
pl.xlabel('log_cap_gain_ZCentered')
pl.ylabel('Frequency')


# In[ ]:


# Checking:
data_original.head()


# In[ ]:


# Good.


# #######################################################################################################################################################
# 
# ### **capital-loss**    /Continuous. 
# 
# Meaning: Dollars lost by the individual's investments during the year.

# In[ ]:


feature_name = 'cap-loss'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# This distribution appears that it'll be similar to cap-gain. However, the max loss is far less than 50K so I don't think I'll be making a new feature representing this one.
# Let's check for missing values:

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# Good. Now skewness and kurtosis:

print(f'Skewness: {skew(data_original[feature_name])}')
print(f'Kurtosis: {kurtosis(data_original[feature_name])}')


# In[ ]:


# Yeah pretty bad. Time to visualize:

x = data_original[feature_name]
pl.hist(x, bins=100)
pl.xlabel('log_cap_gain')
pl.ylabel('Frequency')


# In[ ]:


# I know the max value, it's definitly an outlier. Let's invetigate for others.

data_original.nlargest(10, [feature_name]) 


# In[ ]:


# These all look proper. I'll apply the same transformation to this as I did on cap-gain to keep it consistently scaled with the rest of our features.

log_cap_loss = np.log(data_original[feature_name] + 1)
data_original['log_cap_loss'] = log_cap_loss

mean = np.mean(data_original[feature_name])
stdev = np.std(data_original[feature_name])
cap_loss_ZCentered = (data_original[feature_name] - mean) / stdev

mean = np.mean(log_cap_loss)
stdev = np.std(log_cap_loss)
log_cap_loss_ZCentered = (log_cap_loss - mean) / stdev

# Visualizing:
x = log_cap_loss_ZCentered
pl.hist(x, bins=100)
pl.xlabel('log_cap_loss_ZCentered')
pl.ylabel('Frequency')


# In[ ]:


# Now to replace the original feature with the transformed version.

data_original['cap_loss_ZCentered'] = cap_loss_ZCentered
data_original['log_cap_loss_ZCentered'] = log_cap_loss_ZCentered

# Checking:
data_original.head()


# In[ ]:


# Good.


# #######################################################################################################################################################
# 
# ### **hrsperwk**    /Continuous. 
# 
# Meaning: The number of hours an individual works per week.

# In[ ]:


feature_name = 'hrsperwk'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# Mean is about 40 hours, as expected. The first and third quartile are pretty tight to the mean so we'll likely see high kurtosis here. Probably some minor right skew too.
# I'll check skewness and kurtosis, as well as for missing values:

print(f'Skewness: {skew(data_original[feature_name])}')
print(f'Kurtosis: {kurtosis(data_original[feature_name])}')

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# Yup. Glad there aren't missing values.
# Let's visualize:

x = data_original[feature_name]
pl.hist(x, bins=100)
pl.xlabel('hrsperwk')
pl.ylabel('Frequency')


# In[ ]:


# I'll just Z-center this one to scale it properly. The skewness here isn't that extreme.

mean = np.mean(data_original[feature_name])
stdev = np.std(data_original[feature_name])
hrs_per_wk_ZCentered = (data_original[feature_name] - mean) / stdev

# Now to add the transformed version.

data_original['hrs_per_wk_ZCentered'] = hrs_per_wk_ZCentered

# Checking:
data_original.head()


# #######################################################################################################################################################
# 
# ### **native**    /Categorical. Factor Levels include: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# 
# Meaning: The country the individual was born in.

# In[ ]:


feature_name = 'native'

# viewing a summary
data_original[[feature_name]].describe()


# In[ ]:


# Our summary tells us there are 42 unique factor levels here. However, there are only 41 listed in the description, so we have missing values. Most common value is United-states
# Confirming:

boolseries = data_original.apply(lambda x: True if x[feature_name] == ' ?' else False, axis=1)
print(f'Number of missing values (?): {len(boolseries[boolseries == True].index)}')
print(f'Number of null values: {data_original[[feature_name]].isnull().sum()}')


# In[ ]:


# I'll impute the mode (United-States) for the missing values.

data_original.groupby(feature_name).count()

# Making the replacement and recalculating the values
data_original[[feature_name]] = data_original[[feature_name]].replace([" ?"], [" United-States"])
data_original.groupby(feature_name).count()

# Worked fine.


# #######################################################################################################################################################
# 
# ### **Checking for numeric feature correlation**

# In[ ]:


# Checking correlation between all numeric features.

correlation_matrix = data_original.corr().round(2)
pl.figure(figsize=(10,8))
sb.heatmap(data=correlation_matrix, annot=True, center=0.0, cmap='coolwarm')


# In[ ]:


# Our numeric features have very weak correlation to eachother, despite the strong correlations between features and their transformed features that I created. 
# This is good for us if we're looking to predict our label because it means we won't have to worry about the preoblem of multicollinearity among them. 


# #######################################################################################################################################################
# 
# ### **Investigating explainable variation between non-label categorical features**

# Features to look at here [workclass, education, marital-status, occupation, relationship, race, sex, cap-gains50k]
# 
# Pairs of categorical variables don't have a pearson correlation coefficient.
# 
# To explore whether one categorical group effects the distribution of another, I'll be making interaction plots. The plots will consist of lines marking the change in
# frequency from one categorical factor level to another. There will be a set of lines per factor level of the second categorical varaible.
# Generally when the lines between the factor levels controlling the hue are approximatly paralell this means that the second variable is not really effecting the distribution 
# of the first. Otherwise, there may be an interesting effect. In such cases I'll try to explain the interaction or raise questions regarding it.
# 
# I'll be skipping interactions with the feature (native) because of the high volume of factor levels. This type of plot will just be confusing with too many levels.
# 
# I'll also omit any plots that appear to exhibit little interaction for the sake of brevity.

# In[ ]:


# Each graph will be created with the same code, I'll just switch the names of the variables hue_lab and x_lab
hue_lab = 'workclass'
x_lab = 'education'

# Grouping by the hue_group, then counting by the x_lab group
hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)

# Creating the percentage vector to measure the frequency of each type
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]

# Creating and plotting the new dataframe 
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# The interaction between education and workclass shows a higher percentage of 'HS-grad' individuals in the occupation 'Without-pay' than all the other 'HS-grad' individuals.
# There also appear to be more individuals than average who have completed HS or less that are 'Without-pay' or 'Never-worked' than other education groups.

# In[ ]:


hue_lab = 'workclass'
x_lab = 'occupation'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# There is a lot of variation here within these groups. This may be explained by some occupations being available at different frequencies within different economic sectors. One of which, that is rather extreme, is the armed forces only being a part of the federal government.

# In[ ]:


hue_lab = 'cap-gains50k'
x_lab = 'workclass'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Here's something interesting, some variation in frequency occuring in the rates of 'cap-gains50k' based on economic sector. Seeing greater frequency of 'cap-gains50k' for individuals that are self employed. This could result from these individuals making less consistent pay than the other factor levels of workclass, thus taking more risk in equities. Or it could also be the result of these individuals making more money than others, due to having a greater share of profit. Higher principle risked on average provides higher return in markets.

# In[ ]:


hue_lab = 'cap-gains50k'
x_lab = 'education'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# There's a lot of variation here. In advanced degrees such as prof-school, bachelors, masters, and doctorate individuals have a higher frequency of cap-gains50k being True than False. All other education levels are the opposite. This could be because these degree holders have a hgiher access to capital than the others, due to having better job oppotunities. 

# In[ ]:


hue_lab = 'marital-status'
x_lab = 'occupation'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# This graph is somewhat interesting. Occupations on the right side of the x-axis appear very paralell, where occupations on the left side of the x-axis appear much more mixed. I don't really have any idea why this is occurring.

# In[ ]:


hue_lab = 'relationship'
x_lab = 'marital-status'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# This interaction isn't all that interesting. I only left it in because of one small detail. The levels of relationship 'Husband' and 'wife' are almost perfectly matched to the factor level 'Married-civ-spouse' in marital-status. These levels essentially describe the same thing and we may want to exclude one the two features from our final model.

# In[ ]:


hue_lab = 'sex'
x_lab = 'marital-status'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Here, our data seems to imply that American men are much more likely to be married than american women. Also, women have a higher percentage of diverce than men. Perhaps the two are connected? Perhaps not? Women are also more liekly to have never been married which may also contribute.

# In[ ]:


hue_lab = 'cap-gains50k'
x_lab = 'marital-status'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# This interaction exhibits something a lot of people might expect. Married individuals have a significantly hgiher percentage of cap-gains50k being True than False. Most likely due to two people being able to contribute more capital than one. Also, individuals never-married have significantly lower percentage of cap-gains50k being True than False. For the same reason as previous.

# In[ ]:


hue_lab = 'relationship'
x_lab = 'occupation'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# This interaction shows a lot of the same variation that we saw in the occupation ~ marital-status interation. I'm beleiving more and more that marital-status and relationship are a bit redundant.

# In[ ]:


hue_lab = 'sex'
x_lab = 'occupation'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# This interactions displays the idea that men and women often tend to work in different fields. Especially so in the 'adm-clerical' and 'craft-repair' factor levels. The court is still out on whether this is due to socially constructed entry barriers or that men and women tend to prefer focusing on different thing on average.

# In[ ]:


hue_lab = 'cap-gains50k'
x_lab = 'occupation'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Here we see that occupation likely has a lot of influence over whether a person makes capital gains greater than 50k in a year. Occupations with the largest disparity between cap-gains50k being True or False are 'Exec-managerial' and 'Prof-specialty'.

# In[ ]:


hue_lab = 'race'
x_lab = 'relationship'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Something interesting is definitly happening here. Race appears to effect whether an individual is a husband or Unmarried. The 'wife' designation doesn't show this effect much. All of the wife observations are very close to eachother racially. Yet, 'husband' and 'unmarried' have wider variation and an almost preserved ordering, suggesting a lack of 'black' 'husband' individuals may influence a gain in 'black' 'unmarried' individuals.

# In[ ]:


hue_lab = 'sex'
x_lab = 'relationship'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Disregard the disparities here in the 'Husband' and 'wife' relationship designations, these are dependent on sex. The most interesting effect here is that women appear more likely to be an 'only child' or 'unmarried'.

# In[ ]:


hue_lab = 'cap-gains50k'
x_lab = 'relationship'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Here we see that 'husband' and 'wife' individuals are more likely to have capital gains greater than 50k. Likely, a repeat of the effect we saw earlier that I attributed to married individuals having reater access to capital on average.

# In[ ]:


hue_lab = 'cap-gains50k'
x_lab = 'sex'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Why does it appear that men are more likely to have capital gains greater than 50k than women? Are men more aggressive with investments on average? Is it due to men being more likely to be married? Other socio-economic factors?

# #######################################################################################################################################################
# 
# ### **Investigating explainable variation between non-label categorical features and numerical features**

# Features to look at here: Continuous[log_age_ZCentered, log_fnl_wgt_ZCentered, education_num_ZCentered, log_cap_gain_ZCentered, log_cap_loss_ZCentered, hrs_per_wk_ZCentered] ~ Categorical[workclass, education, marital-status, occupation, relationship, race, sex, cap-gains50k]
# 
# I'll be skipping interactions with the feature (native) because of the high volume of factor levels. Plots will just be confusing with too many levels.
# 
# There will be 48 total pairs here to represent with plots so, again, I'll also omit any plots that appear to exhibit little interaction for the sake of brevity.

# In[ ]:


sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["workclass"] )


# Here we see younger than average individuals mostly make up the never-worked factor level.

# In[ ]:


sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["education"] )


# Age effects the degree of education an individual has completed. Somewhat a given, but worth noting.

# In[ ]:


sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["marital-status"] )


# Age also effects marital-status. We see younger individuals making up the majority of 'never-married' individuals. We see older individuals making up the mojority of the 'Widowed' factor level.

# In[ ]:


sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["cap-gains50k"] )


# The average age on individuals making less than 50k in capital gains is lower than the average age of those making more than 50k in capital gains. The variance of the less than 50k sample is also much wider than the other sample. This makes sense because older individuals have more access to capital and can achieve higher dollar value gains with the same percentage increase in portfolio value.

# In[ ]:


sb.boxplot( x=data_original["log_fnl_wgt_ZCentered"], y=data_original["race"] )


# Here we can get a bit of a clue into how racial descriminators effect the value of the final weight feature. American-Indian-Exkimos have a slightly lower average than the other racial groups, but a much lower first quartile than the other racial groups. I still don't know how the authors calculated this statistic so I don't know why this is.

# In[ ]:


sb.boxplot( x=data_original["education_num_ZCentered"], y=data_original["workclass"] )


# Mostly less educated individuals appear to make up the 'never-worked' factor level. As we have already attributed younger individuals to the never worked category and to the lower education category, we're likely seeing the same effect here.

# In[ ]:


sb.boxplot( x=data_original["education_num_ZCentered"], y=data_original["occupation"] )


# Years of education completed has a noticable effect on the variation of the 'occupation' feature. Some jobs have higher barriers to entry than others.

# In[ ]:


sb.boxplot( x=data_original["education_num_ZCentered"], y=data_original["race"] )


# The distribution of education years completed appears to change noticably when factored by race. All of the education distributions have similar a mean (likely around graduating from HS) but the thrid quartiles for whites and asians are higher than the remaining races. The other category has a lower first quartile than the remaining distributions. There are outliers in all categories. This effecty is likely due to some socio-economic factors within these communities.

# In[ ]:


sb.boxplot( x=data_original["education_num_ZCentered"], y=data_original["cap-gains50k"] )


# Here we see that more educated individuals are much more likely than less educated individuals to achieve greater than 50k in capital gains. Probably due to more educated individuals on average being more specialized and paid more.

# In[ ]:


sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["workclass"] )


# Here we see a bit of variation between the hours worked per week within different economic sectors. Self employed individuals work more than average and 'without-pay' + 'Never-worked' individuals work much less than average. The remaining groups have similar distributions with many outliers. 

# In[ ]:


sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["education"] )


# College educated individuals and Prof-school individuals appear to work more hours than the average when compared to all other groups. Is this because these groups are on average tackling more complex problems? Is this because these groups are in positions requiring more responsibility, and thus, more time?

# In[ ]:


sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["marital-status"] )


# Hours worked per week also avaries noticable based on the marital-status of an individual. While all of these factor levels show a similar mean, the first and third quartiles are where we see the effect. Widowed individuals work less than the mean of all groups, likely due to being older on average and perhaps retired. Never married individuals also work less than the shared mean. (perhaps this contributes to why they were never married. Married individuals work slightly more hours than divorced inidividuals for unkown reasons.

# In[ ]:


sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["occupation"] )


# Here we see how some occupations simply require more time than others, especially farming for example. We all know the common trope of the farmer waking up at the crack of dawn to manage the fields. Is this what the data is showing us?

# In[ ]:


sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["relationship"] )


# The most interesting variation between these 'relationship' groups is that husband work on average more hours per week than wives. Is this due to the effect of child rearing? The 'own-child' factor level shows lower hours worked per week than the everage of the rest of the groups. Is this because most of these individuals are younger?

# In[ ]:


sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["sex"] )


# Again, we're seeing that men on average work more hours than women. Again, is this because of the effect of child rearing on the average?

# In[ ]:


sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["cap-gains50k"] )


# Those who work more hours per week appear more likely to achieve grater than 50k of capital gains. Is this because longer hours worked achieves higher pay and more access to capital?

# #######################################################################################################################################################
# 
# ### **Investigating explainable variation between the label and predictor features**
# 
# Here, we'll be investigating the relationships between the label and [log_age_ZCentered, log_fnl_wgt_ZCentered, education_num_ZCentered, log_cap_gain_ZCentered, log_cap_loss_ZCentered, hrs_per_wk_ZCentered, workclass, education, marital-status, occupation, relationship, race, sex, cap-gains50k]
# 
# I'll be using a combination of the same plots from the last two sections.
# 
# I won't omit any plot from this section because there are only 14 of them.

# In[ ]:


sb.boxplot( x=data_original["log_age_ZCentered"], y=data_original["label"] )


# It appears older individuals making > 50k are on average older than the average individual in the sample. Just as we've seen with the cap-gains50k distribution. This is likely due to older individuals being in more senior positions that pay more.

# In[ ]:


sb.boxplot( x=data_original["education_num_ZCentered"], y=data_original["label"] )


# Individuals making > 50k have higher than average years of education completed. The mean of the <= 50k group appears almost and entire standard deviation lower and the variances of the two groups appear significantly different.

# In[ ]:


sb.boxplot( x=data_original["log_fnl_wgt_ZCentered"], y=data_original["label"] )


# The distributions of the two final weight groups here are very similar. It's likely I wouldn't use this in a predictive model given that thte label was our target. 

# In[ ]:


sb.boxplot( x=data_original["log_cap_gain_ZCentered"], y=data_original["label"] )


# The average, first & third quartiles, and tails of the two capital gains groups here are all basically 0. This is why I made a categorical feature off of this numeric feature. The distribution of the > 50k outlier appears more concentrated higher than the <= 50k outliers. 

# In[ ]:


sb.boxplot( x=data_original["log_cap_loss_ZCentered"], y=data_original["label"] )


# Very similar distributions in capital loss as with capital gain. We see higher magnitude gain/loss in the >50k column because of greater dollar value gain on the same percent increase of a likely larger thasn average pool of capital. 

# In[ ]:


sb.boxplot( x=data_original["hrs_per_wk_ZCentered"], y=data_original["label"] )


# The > 50k group here shows a higher average number of hours worked per week than the <= 50k group. Both distributions have pretty tight quartiles and have many outliers. 

# In[ ]:


hue_lab = 'label'
x_lab = 'workclass'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# There's some interaction here. The percentage of individuals making > 50k is higher in all factor levels except for 'Private' and 'never-worked'. Could be useful. I would have expected private workers to have the advantage here. I don't know how this works . . .

# In[ ]:


hue_lab = 'label'
x_lab = 'education'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Again, similar to the interaction between education ~ cap-gains50k, we see advanced degrees such as 'Prof-school', 'Bachelors', 'Masters', & 'Doctorate' having the advantage here. Pretty significant difference in the proportions at these levels.

# In[ ]:


hue_lab = 'label'
x_lab = 'marital-status'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Again, we see greater financial succes with married individuals. Joint income is powerful. We also see a higher frequency of <= 50k individuals under the 'Never-married' factor level.

# In[ ]:


hue_lab = 'label'
x_lab = 'occupation'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Occupation levels provide a lot of variation in frequency regarding our label. The two best off occupation levels are 'Prof-specialty' and 'Exec-managerial'. Likely due to higher than average pay.

# In[ ]:


hue_lab = 'label'
x_lab = 'relationship'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# Individuals that're Husband or wives, again, appear better off. Liekly another redundant effec that we've seen under the 'marital-status ~ label' interaction.

# In[ ]:


hue_lab = 'label'
x_lab = 'race'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# This interaction doesn't appear to be especially interesting for all races except black individuals, whom unfortunatly have a higher frequency of making less than 50k. The remaining racial groups are all very similar.

# In[ ]:


hue_lab = 'label'
x_lab = 'sex'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# This graph is almost identical to the relationship 'sex ~ cap-gains50k'. We, again, see men being more likely to make more than 50k than women. Likely, attributable to the effect of child rearing on average.

# In[ ]:


hue_lab = 'cap-gains50k'
x_lab = 'label'

hue_group = data_original.groupby([hue_lab], sort=False)
counts = hue_group[x_lab].value_counts(normalize=True, sort=False)
data = [
    {hue_lab: hue, x_lab: x, 'percentage': percentage*100} for 
    (hue, x), percentage in dict(counts).items()
]
df = pd.DataFrame(data)
p = sb.catplot(x=x_lab, y='percentage', hue=hue_lab, kind="point", data=df);
p.set_xticklabels(rotation=90)
p.fig.suptitle(f'Interaction of {x_lab} ~ {hue_lab}')


# All individuals making capital gains greater than 50k end the year with the label >50k. (This is why I made this feature) Roughly 20% of individuals making less than 50k in capital gains still make >50k, so this won't be the only feature we need to predict the label.

# #######################################################################################################################################################
# 
# # ***Summary of findings after EDA***

# In[ ]:


# Finally, dropping any duplicate rows
first_len = len(data_original)
data_original.drop_duplicates()
print(f'Dropped {first_len - len(data_original)} records.')

# Final data set
data_original.head()


# I've kept all of the features of the original data set in order to preserve the information available when we begin modelling the data.
# 
# For each numeric feature I created a new transformed feature using either the log, the z-score, and/or the s-zcore and log. Some models fail (especially those using gradient descent) if some features have much larger numbers than others. The features: 'age', 'fnlwgt', 'cap_gain', and 'cap_loss' were all transformed using the log function before they were scaled by the Z-score.
# 
# I created one new feature, 'cap-gain50k' which is a binary categorical feature. Its value is True if the individual made more than 50k in capital gains and False otherwise.
# 
# None of the outliers that I've observed appeared erroneous so I've left them as is. 
# 
# Missing values within a categorical feature were replaced with the mode value. Missing values within a numeric feature were replaced with the mean value.

# In[ ]:


# splitting off the test set
print(len(train_original), end = '\t')
print(len(test_original))

holdout = data_original.tail(16281)
data_original = data_original.head(32561)


# #######################################################################################################################################################
# 
# # ***Defining model evaluation measures***

# Our label distributions are pretty unbalanced. So, accuracy alone won't be a good measure of how our model is performing.
# I think it'll best best to utilize a Score function, Confusion matrix, accuracy, and our type 1 and 2 error rates.
# S.T.:

# CONFUSION MATRIX: 
# *****************************************
# | TP = TP_count   	| FP = FP_count 	|
# *****************************************
# | FN = FN_count   	| TN = TN_count 	|
# *****************************************
# SCORE ON SAMPLE: TP_count - FP_count 
# ACCURACY: (TP_count + TN_count) / all_count	
# ALPHA: (FP_count) / all_count	
# BETA: (FN_count) / all_count

# I believe these measures will be best because we'll be able to emphasize amount of correct classifications, false positives, and false negatives.
# 
# Based on the EDA I conducted above, I believe the best model to use for predicting our label will be based on the following variables:
# 
# cap-gains50k, sex, relationship, occupation, education, workclass, age, and hrsperwk
# 
# These variables have stronger interaction with the label than the remaining variables. I won't be training any models witht he independent features assumption so I'm not terribly worried about the effect of multipcollinearity, but, you may notice I'm not using both the education and education-num features. I'll test a variation of the model for each kind of numeric transformation I created in the pre-processing phase to see which data works best. 

# In[ ]:


"""
Definition of custom scoring function that utilizes our scoring metric.

y: array, actual test labels
y_pred: array, predicted labels
"""
def score(y, y_pred):
    score = 0
    for x1, x2 in zip(y, y_pred):
        # increase score by 1 for every true positive
        if x2 == 1 and x1 == x2:
            score += 1
        elif x2 == 1 and x1 != x2:
            # decrease score by 1 for every false positive
            score -= 1
    return score


# # Fitting and testing sklearn ~ knn
# 

# In[ ]:


"""
Utilizing GridSearchCV's parallel processing to speed up the process of finding the optimal k value.
Maximum number of available processors will be used.

data: pd dataframe, the features and labels
cat_columns: array, names of categorical columns to create dummy encodings for
distance: str, metric to be used in distance calculation

"""
def fit_sklearn_knn_hyperparams(data, cat_columns, score_function, distance = 'euclidean'):

    # initialize scorer function, compatible with GridSearchCV
    my_scorer = make_scorer(score_function, greater_is_better=True)

    data_knn_full = data
    # create dummy encodings of categorical features
    data_knn_full = pd.get_dummies(data_knn_full, columns=cat_columns)

    # define the values of k to test
    k = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    grid_param = {'n_neighbors': k}

    # initialize model with given distance metric
    model = KNeighborsClassifier(metric = distance)
    # initialize grid search with custom scoring function, using default number of folds (5)
    KNN_random = GridSearchCV(estimator = model, 
                                 param_grid = grid_param,
                                 scoring = my_scorer,
                                 verbose = 2,
                                 cv = 5,
                                 n_jobs = -1)
    # begin tuning
    KNN_random.fit(data_knn_full.drop(columns=['label']), data_knn_full['label'].replace([" <=50K", " >50K"], [0, 1]))
    # print results
    print(KNN_random.best_params_)
    print(f'score: {KNN_random.best_score_}')


# In[ ]:


"""
Compute and print the cost matrix for a single k value. Utilizes sklearn kNN.

train: pd dataframe, training data
test: pd dataframe, testing data
cat_columns: array, names of categorical columns to create dummy encodings for
distance: str, metric to be used in distance calculation
k: int, the number of neighbors to tally a vote with

"""
def test_sklearn_knn(train, test, cat_columns, distance='euclidean', k = 1):
    # define number of expiriments to perform and initialize confusion matrix
    result_avgs = [0, 0, 0, 0]

    # create dummy encodings of categorical features
    train = pd.get_dummies(train, columns=cat_columns)
    test = pd.get_dummies(test, columns=cat_columns)

    # seperate features from labels, replace labels with 0s and 1s
    xtrain = train.drop(columns=['label'])
    xtest = test.drop(columns=['label'])
    ytrain = train['label'].replace([" <=50K", " >50K"], [0, 1])
    ytest = test['label'].replace([" <=50K", " >50K"], [0, 1])

    # predict result matrix
    knn = KNeighborsClassifier(n_neighbors = k, metric = distance)
    knn.fit(xtrain, ytrain)
    result = knn.predict(xtest)

    # create simple list of test labels
    y = ytest.values.tolist()

    # iterate over the results for fixed k value and increment counts for each metric of the confusion matrix
    count_true_pos, count_false_pos, count_true_neg, count_false_neg = 0, 0, 0, 0
    for j in range(len(result)):
        if y[j] == 1:
            if result[j] == y[j]:
                count_true_pos += 1
            else:
                count_false_neg += 1
        else:
            if result[j] == y[j]:
                count_true_neg += 1
            else:
                count_false_pos += 1
                
    # bin the counts
    result_avgs[0] += count_true_pos
    result_avgs[1] += count_false_pos
    result_avgs[2] += count_true_neg
    result_avgs[3] += count_false_neg

    # print confusion matrix
    print()     
    print(f'CONFUSION MATRIX FOR K = {k}: ')
    print(f'*********************************')
    print(f'| TP = {round(result_avgs[0], 2)} \t| FP = {round(result_avgs[1], 2)} \t|')
    print(f'*********************************')
    print(f'| FN = {round(result_avgs[3], 2)} \t| TN = {round(result_avgs[2], 2)} \t|')
    print(f'*********************************')
    print(f'SCORE ON SAMPLE: {result_avgs[0] - result_avgs[1]}')
    print(f'ACCURACY: {(result_avgs[0] + result_avgs[2])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')
    print(f'ALPHA: {(result_avgs[1])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')
    print(f'BETA: {(result_avgs[3])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}')
    print()


# In[ ]:


#Fitting model with raw data
fit_sklearn_knn_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], score)


# In[ ]:


#Fitting model with log transformed data
fit_sklearn_knn_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], score)


# In[ ]:


#Fitting model with z-score centered data
fit_sklearn_knn_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], score)


# In[ ]:


#Fitting model with log and z-score centered data
fit_sklearn_knn_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], score)


# In[ ]:


# I'll test both the z-score only model and the log + z-score model because they're so close in cross validation.


# In[ ]:


#Testing model with z-score centered data, k = 40
test_sklearn_knn(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']], 
                 holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']],
                 ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], distance = 'euclidean', k = 40)


# In[ ]:


#Testing model with log and z-score centered data, k = 50
test_sklearn_knn(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']], 
                 holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']],
                 ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], distance = 'euclidean', k = 50)


# ## Best kNN Euclidiean distance metric model:
# 
# Best performance is for z-score of log transformed data:

# CONFUSION MATRIX FOR K = 50: 
# *********************************
# | TP = 2201 	| FP = 1059 	|
# *********************************
# | FN = 1645 	| TN = 11376 	|
# *********************************
# SCORE ON SAMPLE: 1142
# ACCURACY: 0.8339168355752103	ALPHA: 0.06504514464713469	BETA: 0.10103801977765493

# ## Now testing jaccard distance metric

# In[ ]:


#Fitting model with just categorical features
fit_sklearn_knn_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], score, 'jaccard')


# In[ ]:


#Testing model with just categorical features, k = 70, distance = 'jaccard'
test_sklearn_knn(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass']], 
                 holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass']],
                 ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 'jaccard', k = 70)


# # Fitting and testing sklearn ~ Random Forest
# I'm choosing to test a random forest here because of the lack of assumptions for the model and it's general robustness to the imbalanced class problem. I think it will perform better than kNN.

# In[ ]:


"""
Utilizing RandomizedSearchCV's parallel processing to speed up the process of finding the optimal values of n_estimators, min_samples_split, and min_samples_leaf.
Fitting a Random Forest.
Maximum number of available processors will be used.

data: pd dataframe, the features and labels
cat_columns: array, names of categorical columns to create dummy encodings for
random_seed_adder: int, value to be used in calculating random seed of train-test split

"""
def fit_sklearn_rf_hyperparams(data, cat_columns, random_seed_adder, score_function):
    # initialize scorer function, compatible with RandomizedSearchCV
    my_scorer = make_scorer(score_function, greater_is_better=True)

    data_rf_full = data
    # create dummy encodings of categorical features
    data_rf_full = pd.get_dummies(data_rf_full, columns=cat_columns)

    # define the values of n_estimators, min_samples_split, min_samples_leaf to test
    # n_estimators effects the bias of the model
    # min_samples_split and min_samples_leaf mainly effect model variance
    n_estimators = [100, 200, 500]
    min_samples_split = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    min_samples_leaf = [1, 2, 5, 10, 15, 20, 25, 30]
    grid_param = {'n_estimators': n_estimators,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    
    # initialize model
    model = rf(random_state=1)
    # initialize randomized search with custom scoring function, using default number of folds (5)
    RFC_random = RandomizedSearchCV(estimator = model, 
                                 param_distributions = grid_param,
                                 n_iter = 100,
                                 scoring = my_scorer,
                                 verbose=2,
                                 cv = 5,
                                 random_state = random_seed_adder,
                                 n_jobs = -1)
    # begin tuning
    RFC_random.fit(data_rf_full.drop(columns=['label']), data_rf_full['label'].replace([" <=50K", " >50K"], [0, 1]))
    # print results
    print(RFC_random.best_params_)
    print(f'score: {RFC_random.best_score_}')


# In[ ]:


"""
Compute and print the cost matrix for a single hyperparameter setting. Utilizes sklearn Random Forest.

train: pd dataframe, training data
test: pd dataframe, testing data
cat_columns: array, names of categorical columns to create dummy encodings for
n_trees: int, number of trees to generate
m_min: int, min number of samples required to split a node
m_leave: int, min number of samples required to be at each leaf 

"""
def test_sklearn_rf(train, test, cat_columns, n_trees, m_min, m_leave):
    # define number of expiriments to perform and initialize confusion matrix
    result_avgs = [0, 0, 0, 0]

    # create dummy encodings of categorical features
    train = pd.get_dummies(train, columns=cat_columns)
    test = pd.get_dummies(test, columns=cat_columns)

    # seperate features from labels, replace labels with 0s and 1s
    xtrain = train.drop(columns=['label'])
    xtest = test.drop(columns=['label'])
    ytrain = train['label'].replace([" <=50K", " >50K"], [0, 1])
    ytest = test['label'].replace([" <=50K", " >50K"], [0, 1])

    # initialize model and predict result matrix
    rf_model = rf(n_estimators = n_trees, min_samples_split = m_min, min_samples_leaf = m_leave)
    rf_model.fit(xtrain, ytrain)
    result = rf_model.predict(xtest)

    # create simple list of test labels
    y = ytest.values.tolist()

    # iterate over the results for fixed k value and increment counts for each metric of the confusion matrix
    count_true_pos, count_false_pos, count_true_neg, count_false_neg = 0, 0, 0, 0
    for j in range(len(result)):
        if y[j] == 1:
            if result[j] == y[j]:
                count_true_pos += 1
            else:
                count_false_neg += 1
        else:
            if result[j] == y[j]:
                count_true_neg += 1
            else:
                count_false_pos += 1
                
    # bin the counts
    result_avgs[0] += count_true_pos
    result_avgs[1] += count_false_pos
    result_avgs[2] += count_true_neg
    result_avgs[3] += count_false_neg

    # print confusion matrix
    print()     
    print(f'CONFUSION MATRIX FOR n_estimators = {n_trees}, min_samples_split = {m_min}, min_samples_leaf = {m_leave}: ')
    print(f'*********************************')
    print(f'| TP = {round(result_avgs[0], 2)} \t| FP = {round(result_avgs[1], 2)} \t|')
    print(f'*********************************')
    print(f'| FN = {round(result_avgs[3], 2)} \t| TN = {round(result_avgs[2], 2)} \t|')
    print(f'*********************************')
    print(f'SCORE ON SAMPLE: {result_avgs[0] - result_avgs[1]}')
    print(f'ACCURACY: {(result_avgs[0] + result_avgs[2])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')
    print(f'ALPHA: {(result_avgs[1])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')
    print(f'BETA: {(result_avgs[3])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}')
    print()


# In[ ]:


#Fitting model with raw data
fit_sklearn_rf_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 654, score)


# In[ ]:


#Fitting model with log transformed data
fit_sklearn_rf_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 96846, score)


# In[ ]:


#Fitting model with z-score centered data
fit_sklearn_rf_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 496874, score)


# In[ ]:


#Fitting model with log and z-score centered data
fit_sklearn_rf_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 564, score)


# In[ ]:


# Testing all because they're very close


# In[ ]:


#Testing model with raw data, n_estimators = 200, min_samples_split = 35, min_samples_leaf = 2
test_sklearn_rf(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']], 
                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']],
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 200, 35, 2)


# In[ ]:


#Testing model with log transformed data, n_estimators = 200, min_samples_split = 35, min_samples_leaf = 2
test_sklearn_rf(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']], 
                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']],
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 200, 35, 2)


# In[ ]:


#Testing model with z-score centered data, n_estimators = 500, min_samples_split = 35, min_samples_leaf = 2
test_sklearn_rf(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']], 
                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']],
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 500, 35, 2)


# In[ ]:


#Testing model with log and z-score centered data, n_estimators = 500, min_samples_split = 35, min_samples_leaf = 2
test_sklearn_rf(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']], 
                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']],
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 500, 35, 2)


# ## Comments on Random Forest model performance :
# 
# Best performance is for z-score of log transformed data: 

# CONFUSION MATRIX FOR n_estimators = 500, min_samples_split = 35, min_samples_leaf = 2: 
# *********************************
# | TP = 2186 	| FP = 927 	|
# *********************************
# | FN = 1660 	| TN = 11508 	|
# *********************************
# SCORE ON SAMPLE: 1259
# ACCURACY: 0.8411031263435906	ALPHA: 0.056937534549474846	BETA: 0.10195933910693446

# # Fitting and testing sklearn ~ Gradient Boosting Classifier
# I chose to test a Gradient Boosting Classifier because they're generally even better than random forests for unbalanced class problems. 

# In[ ]:


"""
Utilizing RandomizedSearchCV's parallel processing to speed up the process of finding the optimal values of n_estimators, min_samples_split, and min_samples_leaf.
Fitting a Gradient Boosting Classifier.
Maximum number of available processors will be used.

data: pd dataframe, the features and labels
cat_columns: array, names of categorical columns to create dummy encodings for
random_seed_adder: int, value to be used in calculating random seed of train-test split

"""
def fit_sklearn_GB_hyperparams(data, cat_columns, random_seed_adder, score_function):
    # initialize scorer function, compatible with RandomizedSearchCV
    my_scorer = make_scorer(score_function, greater_is_better=True)

    # define the values of n_estimators, learning_rate, min_samples_split, min_samples_leaf to test
    # n_estimators effects the bias of the model
    # min_samples_split and min_samples_leaf mainly effect model variance
    n_estimators = [100, 250, 500, 750, 1000, 1250]
    learning_rate = [0.01, 0.05, 0.1, 0.2, 0.3]
    min_samples_split = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    min_samples_leaf = [1, 2, 5, 10, 15, 20, 25, 30]
    grid_param = {'n_estimators': n_estimators,
                  'learning_rate': learning_rate,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}
    
    data_gb_full = data
    # create dummy encodings of categorical features
    data_gb_full = pd.get_dummies(data_gb_full, columns=cat_columns)

    # initialize model
    model = gb(random_state=1)
    # initialize randomized search with custom scoring function, using default number of folds (5)
    GB_random = RandomizedSearchCV(estimator = model, 
                                 param_distributions = grid_param,
                                 scoring = my_scorer,
                                 verbose=5,
                                 cv = 5,
                                 random_state = random_seed_adder,
                                 n_jobs = -1)
    # begin tuning
    GB_random.fit(data_gb_full.drop(columns=['label']), data_gb_full['label'].replace([" <=50K", " >50K"], [0, 1]))
    # print result
    print(GB_random.best_params_)
    print(f'score: {GB_random.best_score_}')


# In[ ]:


"""
Compute and print the cost matrix for a single hyperparameter setting. Utilizes sklearn Gradient Boosting Classifier.

train: pd dataframe, training data
test: pd dataframe, testing data
cat_columns: array, names of categorical columns to create dummy encodings for
n_trees: int, number of trees to generate
lr: float, the learning rate
m_min: int, min number of samples required to split a node
m_leave: int, min number of samples required to be at each leaf 

"""
def test_sklearn_gb(train, test, cat_columns, n_trees, lr, m_min, m_leave):
    # define number of expiriments to perform and initialize confusion matrix
    result_avgs = [0, 0, 0, 0]

    # create dummy encodings of categorical features
    train = pd.get_dummies(train, columns=cat_columns)
    test = pd.get_dummies(test, columns=cat_columns)

    # seperate features from labels, replace labels with 0s and 1s
    xtrain = train.drop(columns=['label'])
    xtest = test.drop(columns=['label'])
    ytrain = train['label'].replace([" <=50K", " >50K"], [0, 1])
    ytest = test['label'].replace([" <=50K", " >50K"], [0, 1])

    # initialize model and predict result matrix
    gb_model = gb(n_estimators = n_trees, learning_rate = lr, min_samples_split = m_min, min_samples_leaf = m_leave)
    gb_model.fit(xtrain, ytrain)
    result = gb_model.predict(xtest)

    # create simple list of test labels
    y = ytest.values.tolist()

    # iterate over the results for fixed k value and increment counts for each metric of the confusion matrix
    count_true_pos, count_false_pos, count_true_neg, count_false_neg = 0, 0, 0, 0
    for j in range(len(result)):
        if y[j] == 1:
            if result[j] == y[j]:
                count_true_pos += 1
            else:
                count_false_neg += 1
        else:
            if result[j] == y[j]:
                count_true_neg += 1
            else:
                count_false_pos += 1
                
    # bin the counts
    result_avgs[0] += count_true_pos
    result_avgs[1] += count_false_pos
    result_avgs[2] += count_true_neg
    result_avgs[3] += count_false_neg

    # print confusion matrix
    print()     
    print(f'CONFUSION MATRIX FOR n_estimators = {n_trees}, learning_rate = {lr}, min_samples_split = {m_min}, min_samples_leaf = {m_leave}: ')
    print(f'*********************************')
    print(f'| TP = {round(result_avgs[0], 2)} \t| FP = {round(result_avgs[1], 2)} \t|')
    print(f'*********************************')
    print(f'| FN = {round(result_avgs[3], 2)} \t| TN = {round(result_avgs[2], 2)} \t|')
    print(f'*********************************')
    print(f'SCORE ON SAMPLE: {result_avgs[0] - result_avgs[1]}')
    print(f'ACCURACY: {(result_avgs[0] + result_avgs[2])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')
    print(f'ALPHA: {(result_avgs[1])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}', end='\t')
    print(f'BETA: {(result_avgs[3])/(result_avgs[0] + result_avgs[1] + result_avgs[3] + result_avgs[2])}')
    print()


# In[ ]:


#Fitting model with raw data
fit_sklearn_GB_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 5416, score)


# In[ ]:


#Fitting model with log transformed data
fit_sklearn_GB_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 6748, score)


# In[ ]:


#Fitting model with z-score centered data
fit_sklearn_GB_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 86784, score)


# In[ ]:


#Fitting model with log and z-score centered data
fit_sklearn_GB_hyperparams(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']], 
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 6874384, score)


# In[ ]:


# Testing all because they're very close


# In[ ]:


#Testing model with raw data, n_estimators = 750, learning_rate = 0.05, min_samples_split = 50, min_samples_leaf = 20
test_sklearn_gb(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']],
                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age', 'hrsperwk']],
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 750, 0.05, 50, 20)


# In[ ]:


#Testing model with log transformed data, n_estimators = 500, learning_rate = 0.1, min_samples_split = 2, min_samples_leaf = 25
test_sklearn_gb(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']],
                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age', 'hrsperwk']],
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 500, 0.1, 2, 25)


# In[ ]:


#Testing model with z-score centered data, n_estimators = 1250, learning_rate = 0.05, min_samples_split = 40, min_samples_leaf = 3
test_sklearn_gb(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']],
                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'age_ZCentered', 'hrs_per_wk_ZCentered']],
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 1250, 0.05, 40, 2)


# In[ ]:


#Testing model with log and z-score centered data, n_estimators = 500, learning_rate = 0.1, min_samples_split = 55, min_samples_leaf = 20
test_sklearn_gb(data_original[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']],
                holdout[['label', 'cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass', 'log_age_ZCentered', 'hrs_per_wk_ZCentered']],
                           ['cap-gains50k', 'sex', 'relationship', 'occupation', 'education', 'workclass'], 750, 0.05, 20, 30)


# ## Comments on GBC performance:
# 
# Best performance is for raw data:

# CONFUSION MATRIX FOR n_estimators = 750, learning_rate = 0.05, min_samples_split = 50, min_samples_leaf = 20: 
# *********************************
# | TP = 2226 	| FP = 922 	|
# *********************************
# | FN = 1620 	| TN = 11513 	|
# *********************************
# SCORE ON SAMPLE: 1304
# ACCURACY: 0.8438670843314293	ALPHA: 0.05663042810638167	BETA: 0.09950248756218906

# # ***Ranking of modelling results based on best score achieved***

# ## 1. Gradient Boosting Classifier best performance for raw data: 
# ### label ~ cap-gains50k + sex + relationship + occupation + education + workclass + age + hrsperwk

# CONFUSION MATRIX FOR n_estimators = 750, learning_rate = 0.05, min_samples_split = 50, min_samples_leaf = 20: 
# *********************************
# | TP = 2226 	| FP = 922 	|
# *********************************
# | FN = 1620 	| TN = 11513 	|
# *********************************
# SCORE ON SAMPLE: 1304
# 
# ACCURACY: 0.8438670843314293
# 
# ALPHA: 0.05663042810638167	
# 
# BETA: 0.09950248756218906

# ## 2. Random Forest for z-score of log transformed data: 
# ### label ~ cap-gains50k + sex + relationship + occupation + education + workclass + log_age_ZCentered + hrs_per_wk_ZCentered

# CONFUSION MATRIX FOR n_estimators = 500, min_samples_split = 35, min_samples_leaf = 2: 
# *********************************
# | TP = 2186 	| FP = 927 	|
# *********************************
# | FN = 1660 	| TN = 11508 	|
# *********************************
# SCORE ON SAMPLE: 1259
# 
# ACCURACY: 0.8411031263435906
# 
# ALPHA: 0.056937534549474846
# 
# BETA: 0.10195933910693446

# ## 3. sklearn KNN (Euclidean) for z-score of log transformed data:
# ### label ~ cap-gains50k + sex + relationship + occupation + education + workclass + log_age_ZCentered + hrs_per_wk_ZCentered

# CONFUSION MATRIX FOR K = 50: 
# *********************************
# | TP = 2201 	| FP = 1059 	|
# *********************************
# | FN = 1645 	| TN = 11376 	|
# *********************************
# SCORE ON SAMPLE: 1142
# 
# ACCURACY: 0.8339168355752103
# 
# ALPHA: 0.06504514464713469
# 
# BETA: 0.10103801977765493

# ## 4. sklearn KNN (Jaccard) only categorical model features: 
# ### label ~ cap-gains50k + sex + relationship + occupation + education + workclass

# CONFUSION MATRIX FOR K = 70: 
# *********************************
# | TP = 2040 	| FP = 971 	|
# *********************************
# | FN = 1806 	| TN = 11464 	|
# *********************************
# SCORE ON SAMPLE: 1069
# 
# ACCURACY: 0.82943308150605
# 
# ALPHA: 0.0596400712486948
# 
# BETA: 0.1109268472452552
