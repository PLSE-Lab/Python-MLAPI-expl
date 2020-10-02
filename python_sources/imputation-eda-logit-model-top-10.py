#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from plotly.offline import init_notebook_mode, iplot
#import plotly.graph_objs as go
#import plotly.plotly as py
#from plotly import tools
#from datetime import date
import pandas as pd
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
#import random 
#import warnings
#import operator
#warnings.filterwarnings("ignore")
#init_notebook_mode(connected=True)


# First we will load in the data and take a quick look at the head.

# In[ ]:


test = pd.read_csv("../input/test.csv")
poverty_train = pd.read_csv("../input/train.csv")
display(poverty_train.head(), poverty_train.shape)


# 
# Here is the test data. "Id" is the unique ID attached to a person. "idhogar" is the ID attached to a house (multiple people can live in a house). And "parentesco1" is a binary code indicating whether a person is the head of the household or not. We mostly care about predictions on the head of the household, as per the competition rules of evaluating only *those* predictions.

# In[ ]:


display(test.head(), test.shape, test[["Id", "idhogar", "parentesco1"]].head(10))


# We want to see what percentage of data is missing. This counts the number of missing values and divides by the length of the training set so that we get a percentage. Note that the first three columns here are missing quite a bit of values. A rule of thumb is to drop data with more than a third of the data missing, but since we don't know too much of the data yet, we can't (shouldn't) do anything yet.

# In[ ]:


print ("Top Columns having missing values in the train set.")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()


# In[ ]:


print ("Top Columns having missing values in the test set.")
missmap = test.isnull().sum().to_frame().sort_values(0, ascending = False) / len(test)
missmap.head()


# 
# So we have the percentage of data missing in each feature for the train and test sets. Lucky for us, the train and test sets have missing 
# values in the same features.
# 
# According to the documentation for this data set, "rez_esc" refers to "years behind in school", "v18q1" refers to the number of tablets in the household, and "v2a1" refers to the monthly rent payment. "meaneduc" is average number of years education, and "SQBmeaned" is the square of the education levels. This is available in the documentation. These are the only columns that are missing values.
# 
# We could drop these values, or we could impute them. For our purposes, it can be better to impute these data points. It is good practice, in general, to impute the values in the columns with the smallest number of missing variables, and work your way up to the columns with larger amounts of missing variables. This is especially true if you are using an algorithm to impute missing data, as the imputation of a variable will depend on what's already been imputed.
# 
# To this end, we'll start with "meaneduc" and "SQBmeaned". Let's take a quick look at the rows that are missing values in "educ". 

# In[ ]:


poverty_train[poverty_train.meaneduc.isnull()]


# 
# Note that the exact same rows are also missing values in SQBmeaned. 

# One thing we can do quickly is impute the mean education level with the overall mean in the data set (meaneduc). Only 5 values are missing, and we are using a mean, so this is not a bad method. We can then square those values and fill in the SQBmeaned, so that the rows are consistent with each other.
# 
# NB: imputing with means is not appropriate when a large chunk of your data is missing, or you have a large amount amount of skew in the data. Since this is not the case with only 5 values missing, we can proceed.

# In[ ]:


value1 = poverty_train["meaneduc"].mean()
value2 = value1*value1
display(value1, value2)


# In[ ]:


poverty_train["meaneduc"].fillna(value1, inplace = True)
poverty_train["SQBmeaned"].fillna(value2, inplace = True)

print ("Top Columns having missing values")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()


# Success.

# Now, we want to spend some time exploring the data to get a better understanding of what we are working with. Understanding the data a little better will give us some insight on how to impute what's missing. Let's have a look. We want to come up with a reasonable imputation of the next feature with the smallest number of NaN's. In this case, it will be the monthly rent payment, "v2a1".

# In[ ]:


poverty_train.describe()


# From the documentation, and the discussion here: https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403
# 
# we can see that there might be a way of dealing with "v2a1". The monthly rent payment is related to the following columns 
# 
# "tipovivi1, =1 own and fully paid house
# 
# tipovivi2, "=1 own,  paying in installments"
# 
# tipovivi3, =1 rented
# 
# tipovivi4, =1 precarious
# 
# tipovivi5, "=1 other(assigned,  borrowed)"
# 
# which are from the dataset documentation. We will filter by these and impute the rent payment accordingly. We will assume that if tipovivi1 is 1 for a particular person, then their rent payment is 0, as they own the house and it's paid off.

# In[ ]:


display(poverty_train.loc[( poverty_train["tipovivi1" ] == 1, "v2a1")].isna().sum(), 

        len(poverty_train.loc[( poverty_train["tipovivi1" ] == 1, "v2a1")]))


# Okay, so we can replace all of these with a 0, as this subset of people actually own their house. Note that the number of NaN's is the same as the number of people in this subset, so it's fine to continue as follows:

# In[ ]:


poverty_train.loc[(poverty_train["tipovivi1" ] == 1, "v2a1")] = 0

print ("Top Columns having missing values")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()


# This drops the number of NaN's in the monthly rent payment considerably. Let's keep going.

# In[ ]:


display(poverty_train.loc[( poverty_train["tipovivi2" ] == 1, "v2a1")].isna().sum(), 
        len(poverty_train.loc[( poverty_train["tipovivi2" ] == 1, "v2a1")]))


# In[ ]:


display(poverty_train.loc[( poverty_train["tipovivi3" ] == 1, "v2a1")].isna().sum(), 
        len(poverty_train.loc[( poverty_train["tipovivi3" ] == 1, "v2a1")]))


# These are clean of NaNs. What about tipovivi4? This is a group of people who have "precarious" housing. tipovivi5 is a group of people that have assigned, or borrowed housing.

# In[ ]:


display(poverty_train.loc[( poverty_train["tipovivi4" ] == 1, "v2a1")].isna().sum(), 
        len(poverty_train.loc[( poverty_train["tipovivi4" ] == 1, "v2a1")]))


# In[ ]:


display(poverty_train.loc[( poverty_train["tipovivi5" ] == 1, "v2a1")].isna().sum(), 
        len(poverty_train.loc[( poverty_train["tipovivi5" ] == 1, "v2a1")]))


# 
# For the moment, we are going to do a quick fix. We'll just assume that the rest of these groups don't pay anything and just fill with 
# zeroes.

# In[ ]:


poverty_train["v2a1"].fillna(0, inplace = True)

print ("Top Columns having missing values (%)")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()


# We're almost there. Let's look at "v18a1" next. According to the documentation, this variable is related to the number of tablets that belong to each household. What do we know about this value?

# In[ ]:


poverty_train["v18q1"].describe()


# 
# The median number of tablets in each house is 1, but also note that the *minimum* number of tablets is also 1. Is it really possible that every family in this data set has at least one tablet at home? Note that approximately 77% of the data for this feature is actually missing as well.  We have a feature related to this value though. It's "v18q" which is a binary value for "owns a tablet". Let's see if the the ownership of tablets lines up with the missing values. First, lets take a look at it

# In[ ]:


poverty_train["v18q"].describe()


# Looks like a binary indicator variable. The documentation doesn't say this, but it is probably safe to assume that a "1" means "owns a tablet" and a "0" means "doesn't own a tablet". Let's see if these line up with our NaNs

# In[ ]:


display(poverty_train.loc[( poverty_train["v18q" ] == 0, "v18q1")].isna().sum(),
        len(poverty_train[poverty_train["v18q"] == 0]))


# 
# Yup, the number of missing values correspond exactly to the number of 0's in the tablet ownership feature. We can simply replace the NaNs with a 0 and see if we imputed everything

# In[ ]:


poverty_train.loc[(poverty_train["v18q" ] == 0, "v18q1")] = 0

print ("Top Columns having missing values (%)")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()


# We're almost done imputing! We finally take a look at one of the last columns, which is "rez_esc". This feature denotes the number of years behind in schooling. Let's have a look here as well

# In[ ]:


poverty_train["rez_esc"].describe()


# The means and medians are pretty close to 0. It is probably "safe" to simply fill these all in with a 0, but we'll take a look at the dataset documentation first. We can find the "instlevel" variable which has the following description:
# 
# instlevel1, =1 no level of education
# 
# instlevel2, =1 incomplete primary
# 
# instlevel3, =1 complete primary
# 
# instlevel4, =1 incomplete academic secondary level
# 
# instlevel5, =1 complete academic secondary level
# 
# instlevel6, =1 incomplete technical secondary level
# 
# instlevel7, =1 complete technical secondary level
# 
# instlevel8, =1 undergraduate and higher education
# 
# instlevel9, =1 postgraduate higher education
# 
# 
# It is probably safe to assume that anyone with a complete high school education (or higher) is not behind in school). We can also say that someone with no level of education is the maximum years behind in school. For this data set, that is listed as a 5. Let's look at the number of NaNs in this area. If there are any to replace, we will replace them with our assumed values.  For the rest, we won't really be able to know what the proper treatment is, so in that case, we can replace them with the median (which is 0). If spending more time imputing this helps our model we can come back to it.

# In[ ]:


poverty_train["rez_esc"].median()


# In[ ]:


poverty_train['rez_esc'] = poverty_train.apply(
    lambda row: 0 if (row['instlevel1'] == 1) else row['rez_esc'],
    axis=1
)

poverty_train['rez_esc'] = poverty_train.apply(
    lambda row: poverty_train["rez_esc"].median() if np.isnan(row["rez_esc"]) else row['rez_esc'],
    axis=1
)

print ("Top Columns having missing values (%)")
missmap = poverty_train.isnull().sum().to_frame().sort_values(0, ascending = False) / len(poverty_train)
missmap.head()


# We're done imputing the NaNs. Here is a short pipeline to do it fast.

# 
# 
# We'll take a moment and discuss three other variables. "edjefe" and "edjefa" and "dependency", which is related to our job of imputation. We use a snippet of code from the kernal https://www.kaggle.com/mlisovyi/categorical-variables-encoding-function to make a plot.

# In[ ]:


def plot_value_counts(series, title=None):
    '''
    Plot distribution of values counts in a pd.Series
    '''
    _ = plt.figure(figsize=(12,6))
    z = series.value_counts()
    sns.barplot(x=z, y=z.index)
    _ = plt.title(title)
    
plot_value_counts(poverty_train['edjefe'], 'Value counts of edjefe')
plot_value_counts(poverty_train['edjefa'], 'Value counts of edjefa')
plot_value_counts(poverty_train['dependency'], 'Value counts of dependency')


# Note that the values in the three columns are mostly categorical, with some random yes's or no's. From the discussion here: https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631
# 
# we can see the following quote from the competition host: "Also a clarification on edjefe and edjefa. These variables are just the interaction of escolari (years of education) head of husehold and gender. Some labels were generated whenever continuous variables have 1 or 0. The rule is to have yes being 1 yes=1 and no=0". 
# 
# Also: "The dependency variable is one of the variables created from the data. the formula is:
# 
# dependency=(number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)"
# 
# 
# According to the same discussion, this mapping of $yes \mapsto 1$, and $no \mapsto 0$ applies to all three columns. So, we'll go ahead and apply this mapping.

# In[ ]:


poverty_train["dependency"].replace('yes', 1, inplace = True)
poverty_train["dependency"].replace('no', 0, inplace = True)
poverty_train["edjefa"].replace('yes', 1, inplace = True)
poverty_train["edjefa"].replace('no', 0, inplace = True)
poverty_train["edjefe"].replace('yes', 1, inplace = True)
poverty_train["edjefe"].replace('no', 0, inplace = True)

#check if our solution worked
display(poverty_train["dependency"].value_counts(),
        poverty_train["edjefa"].value_counts(),
        poverty_train["edjefe"].value_counts())


# In[ ]:


poverty_train["edjefe"] = poverty_train["edjefe"].astype(float)
poverty_train["edjefa"] = poverty_train["edjefa"].astype(float)
poverty_train["dependency"] = poverty_train["dependency"].astype(float)


# Now, according to this kernal https://www.kaggle.com/katacs/data-cleaning-and-random-forest we need to adjust one other thing. There are apparently discrepancies between the Target values for individuals and their households. We'll use some of katacs' code and take a look.

# In[ ]:


d={}
weird=[]
for row in poverty_train.iterrows():
    idhogar=row[1]['idhogar']
    target=row[1]['Target']
    if idhogar in d:
        if d[idhogar]!=target:
            weird.append(idhogar)
    else:
        d[idhogar]=target

len(set(weird))


# Okay, we need to fix this. His other code cell does the job.

# In[ ]:


for i in set(weird):
    hhold=poverty_train[poverty_train['idhogar']==i][['idhogar', 'parentesco1', 'Target']]
    target=hhold[hhold['parentesco1']==1]['Target'].tolist()[0]
    for row in hhold.iterrows():
        idx=row[0]
        if row[1]['parentesco1']!=1:
            poverty_train.at[idx, 'Target']=target


# In[ ]:


poverty_train[poverty_train['idhogar']==weird[1]][['idhogar','parentesco1', 'Target']]


# Looks fixed. Let's take a minute and look at some histograms.

# In[ ]:


#poverty_train.hist(bins=30, figsize=(20,15))
#plt.title( "Histogram Plots")
#plt.show()


# This is sort of hard to see, but we can just get an idea of what distributions might be worth deeper inspection. "Target" is naturally of interest, but so would "meaneduc", "v2a1", "age", "agesq", "SQBmeaned", "SQBovercrowding", "overcrowding", "rooms", "SQBage", and some others. We look at these plots a little closer below. We want to make a note of what features are continuous and which ones are categorical, which will help us process the data later. Let's take a closer look at all the data that looks continuous. The notation "SQB" in front of some of these variables means that it is actually the *square* of another variable. For example, "SQBage" is the square of the age. It seems that the competition hosts have done some feature engineering for us already.

# In[ ]:


#poverty_train[["meaneduc", "v2a1", "age", "agesq", "SQBmeaned",
              #"SQBovercrowding", "overcrowding", "SQBage","escolari", "SQBescolari",
               #"SQBdependency", "SQBhogar_total" ]].hist(bins=30, figsize=(20,15))
#plt.title( "Histogram Plots")
#plt.show()


# Some of these are pretty obviously continuous variables, but it's not clear what escolari is. The documentation  (https://www.kaggle.com/c/costa-rican-household-poverty-prediction/data) says that "escolari" is the number of years in school. We can probably treat this as continuous. It might be useful to aggregate this value later to create a house-hold level feature.
# 
# Understanding of some of these columns can help us engineer more features later. For example, if we see that some columns are interesting or may be important, we can make note of these columns, then use them to define *interactions*. Essentially, multiplying columns with other columns. That may be useful, but for now, let's explore the data a little more.

# 
# One thing that is important to note about this data set, is that we are interested in predicting the Target variable, which is a classification of 1 to 4. Values of 1 indicate that a person, or household, is subjected to 
# "extreme poverty", while values of 4 indicate that person or household is "non-vulnerable". This is useful to know, and may help in explaining some of these features. 

# In[ ]:


poverty_train[["v18q1", "v2a1", "rez_esc"]].describe()


# 
# We'll divide the data up based on the level of poorness. A value of 1 is the most poor, while a value of 4 is a "safe" level of wealth. Ie, the household is not vulnerable to poverty.

# In[ ]:


very_poor = poverty_train[poverty_train["Target"] == 1]
poor = poverty_train[poverty_train["Target"] == 2]
vulnerable = poverty_train[poverty_train["Target"] == 3]
safe = poverty_train[poverty_train["Target"] == 4]


# Now let's have a look at the number of tablets, the monthly rent, and average years behind in school between these groups. But first, a look at the shape of these different groups.

# In[ ]:


display(very_poor.shape,
       poor.shape,
       vulnerable.shape,
       safe.shape)


# In[ ]:


display(
    very_poor[["v18q1", "v2a1", "rez_esc"]].describe(),
    poor[["v18q1", "v2a1", "rez_esc"]].describe(),
    vulnerable[["v18q1", "v2a1", "rez_esc"]].describe(),
    safe[["v18q1", "v2a1", "rez_esc"]].describe())


# Unsurprisingly, the mean monthly payment for each group increases as the group acquires more wealth. Strangely, the median housing payment is zero for most houses. Is this an error? Not really. Nearly half this data set actually owns a paid off house, as we determined earlier, and so we imputed a 0 for the missing valules in the monthly payment, whenever a house was owned. This might seem strange, but makes sense in that the mean monthly payment is going up for each demographic as they move out of poverty.
# 
# "v18q1" indicates the number of tablets at home, which has a higher max value for those who are not vulnerable to poverty. The median is 1, for each group, but so is the minimum. Which means that we dont have anyone who put down a zero for tablets at home. This may be unlikely, as we are talking about families in poverty, and not everyone can have tablet. Perhaps the missing values are actually 0s?
# 
# "rez_esc" is the number of years behind in schooling. Unsurprisingly, this number has a higher mean in poor groups, but is lower in the wealthier group. This likely will not be encoded or anything, as having a mean number of years behind in school actually makes sense inuitively.

# Let's look at some of these distributions a little closer, between the very poor group and the non-vulnerable group.

# In[ ]:


very_poor[["Target", "rez_esc", "meaneduc", "v2a1", "age","overcrowding",
           "rooms", "v18q1"]].hist(bins=25, figsize=(20,15))
plt.title( "Histogram Plots")
plt.show()


# In[ ]:


safe[["Target", "rez_esc", "meaneduc", "v2a1", "age","overcrowding",
           "rooms", "v18q1"]].hist(bins=25, figsize=(20,15))
plt.title( "Histogram Plots")
plt.show()


# Right away, we can see a difference between the distributions in the age of the two groups, and in their monthly payments. The safer crowd has a much more uniform age, while the very poor group seems to be on the young side. Also, the safe group has a more skewed distribution of monthly payments. (Note that the safe group's histogram of monthly payments (v2a1) is actually an order of magnitude higher on the x-axis than the very poor group. This is because there is more skew in that distribution, there are some people who pay a lot of money for rent, yet aren't in danger of poverty. As expected, this means the safe group has quite a bit more money to spend. However, the majority of people in both demographics actually own their own house).
# 
# Age seems interesting. Let's take a closer look

# In[ ]:


plt.figure(figsize=(12,7))
plt.title("Distributions of Age")
sns.kdeplot(very_poor["age"], label = "Very Poor Group", shade = True)
sns.kdeplot(poor["age"], label = "Poor Group", shade = True)
sns.kdeplot(vulnerable["age"], label = "Vulnerable Group", shade = True)
sns.kdeplot(safe["age"], label="Safe Group", shade = True)
plt.legend();


# In[ ]:


import numpy as np
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


# In[ ]:


display(mean_confidence_interval(very_poor["age"], confidence = 0.95),
mean_confidence_interval(safe["age"], confidence = 0.95))


# The above output gives first the mean age, and the 95% confidence interval between the very_poor, and safe groups respectively. Note that the confidence intervals are actually disjoint, which implies there is a statistically significant difference in age between the two groups. The safe group is, on average, older. Perhaps age will be an interesting feature to use in our model (when we build it!).
# 
# Here are some ideas for feature interactions, which we can test later. 
# 
# $age \times meaneduc$ (age times the mean education level in the house)
# 
# $SQBage \times meaneduc$ (Square of the age times the mean education level)
# 
# $age \times escolari$ (age times the years of schooling)

# **Preparing data for the model**
# 
# Now that our missing values are filled in,  we can process our data and get it ready to be fit into a model.  Let's make a copy of the data for now and have another look at the columns. We want to see which columns should be standardized or encoded and which can be left along.

# We'll prepare the data in a bunch of different ways to try and compare models with different data preparations. It was shown in some discussions and other kernals that the majority of the target class is "4". So we will try to balance that with sampling as well.
# 
# We'll use some slight variations on the feature engineering made in this kernal: https://www.kaggle.com/mlisovyi/feature-engineering-lighgbm-with-f1-macro
# 
# We actually didn't use a lot of this stuff, but some of the pipelines were useful. If you liked this part of the notebook, make sure to go give them an upvoat as well.
# 
# 

# In[ ]:


def do_features(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
                 ('rent_per_person', 'v2a1', 'r4t3'),
                 ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
                 ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
                 ('tablet_adult_density', 'v18q1', 'r4t2'),
                 #('', '', ''),
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean'],
                'escolari': ['min', 'max', 'mean']
               }
    aggs_cat = {'dis': ['mean']}
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean', 'count']
    # aggregation over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg
    # do something advanced above...
    
    # Drop SQB variables, as they are just squres of other vars 
    #df.drop([f_ for f_ in df.columns if f_.startswith('SQB') or f_ == 'agesq'], axis=1, inplace=True)
    # Drop id's
    df.drop(['Id'], axis=1, inplace=True)
    # Drop repeated columns
    df.drop(['hhsize', 'female', 'area2'], axis=1, inplace=True)
    return df

def convert_OHE2LE(df):
    print_check = True
    
    tmp_df = df.copy(deep=True)
    for s_ in ['pared', 'piso', 'techo', 'abastagua', 'sanitario', 'energcocinar', 'elimbasu', 
               'epared', 'etecho', 'eviv', 'estadocivil', 'parentesco', 
               'instlevel', 'lugar', 'tipovivi',
               'manual_elec']:
        if 'manual_' not in s_:
            cols_s_ = [f_ for f_ in df.columns if f_.startswith(s_)]
        elif 'elec' in s_:
            cols_s_ = ['public', 'planpri', 'noelec', 'coopele']
        if print_check:
            sum_ohe = df[cols_s_].sum(axis=1).unique()
            if sum_ohe.shape[0]>1:
                print(s_)
                print(df[cols_s_].sum(axis=1).value_counts())
                #print(df[list(cols_s_+['Id'])].loc[df[cols_s_].sum(axis=1) == 0])
        tmp_cat = df[cols_s_].idxmax(axis=1)
        tmp_df[s_ + '_LE'] = LabelEncoder().fit_transform(tmp_cat).astype(np.int16)
        if 'parentesco1' in cols_s_:
            cols_s_.remove('parentesco1')
        tmp_df.drop(cols_s_, axis=1, inplace=True)
    return tmp_df


# In[ ]:


from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    '''
    The function does not return, but transforms the input pd.DataFrame
    
    Encodes the Costa Rican Household Poverty Level data 
    following studies in https://www.kaggle.com/mlisovyi/categorical-variables-in-the-data
    and the insight from https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631
    
    The following columns get transformed: edjefe, edjefa, dependency, idhogar
    The user most likely will simply drop idhogar completely (after calculating houshold-level aggregates)
    '''
    
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])


# In[ ]:


def process_df(df_):
    # fix categorical features
    encode_data(df_)
    #fill in missing values based on https://www.kaggle.com/mlisovyi/missing-values-in-the-data
    for f_ in ['v2a1', 'v18q1', 'meaneduc', 'SQBmeaned']:
        df_[f_] = df_[f_].fillna(0)
    df_['rez_esc'] = df_['rez_esc'].fillna(0)
    # do feature engineering and drop useless columns
    return do_features(df_)

#train = process_df(train)
test = process_df(test)


# In[ ]:


def train_test_apply_func(train_, test_, func_):
    test_['Target'] = 0
    xx = pd.concat([train_, test_])

    xx_func = func_(xx)
    train_ = xx_func.iloc[:train_.shape[0], :]
    test_  = xx_func.iloc[train_.shape[0]:, :].drop('Target', axis=1)

    del xx, xx_func
    return train_, test_


# In[ ]:


train = do_features(poverty_train)


# In[ ]:


#train, test = train_test_apply_func(train, test, convert_OHE2LE)


# In[ ]:


X = train.query('parentesco1==1')
#X = train

# pull out the target variable
y = X['Target'] - 1
X = X.drop(['Target'], axis=1)


# In[ ]:


cols_2_drop = ['agg18_estadocivil1_MEAN', 'agg18_estadocivil3_COUNT', 'agg18_estadocivil4_COUNT', 
               'agg18_estadocivil5_COUNT', 'agg18_estadocivil6_COUNT', 'agg18_estadocivil7_COUNT', 
               'agg18_instlevel1_COUNT', 'agg18_instlevel2_COUNT', 'agg18_instlevel3_COUNT', 
               'agg18_instlevel4_COUNT', 'agg18_instlevel5_COUNT', 'agg18_instlevel6_COUNT', 
               'agg18_instlevel7_COUNT', 'agg18_instlevel8_COUNT', 'agg18_instlevel9_COUNT', 
               'agg18_parentesco10_COUNT', 'agg18_parentesco10_MEAN', 'agg18_parentesco11_COUNT', 
               'agg18_parentesco11_MEAN', 'agg18_parentesco12_COUNT', 'agg18_parentesco12_MEAN', 
               'agg18_parentesco1_COUNT', 'agg18_parentesco2_COUNT', 'agg18_parentesco3_COUNT', 
               'agg18_parentesco4_COUNT', 'agg18_parentesco4_MEAN', 'agg18_parentesco5_COUNT', 
               'agg18_parentesco6_COUNT', 'agg18_parentesco6_MEAN', 'agg18_parentesco7_COUNT', 
               'agg18_parentesco7_MEAN', 'agg18_parentesco8_COUNT', 'agg18_parentesco8_MEAN', 
               'agg18_parentesco9_COUNT', 'fe_people_weird_stat', 'hacapo', 'hacdor', 'mobilephone',
               'parentesco1', 'rez_esc', 'v14a', 'v18q', # here
                'agg18_age_MIN', 'agg18_age_MAX' , 'agg18_age_MEAN', 'agg18_escolari_MIN',
                'agg18_escolari_MAX', 'agg18_escolari_MEAN', 'agg18_dis_MEAN', 
                'agg18_estadocivil1_COUNT', 'agg18_estadocivil2_MEAN', 'agg18_estadocivil2_COUNT',
                'agg18_estadocivil3_MEAN', 'agg18_estadocivil4_MEAN', 'agg18_estadocivil5_MEAN',
                'agg18_estadocivil6_MEAN','agg18_estadocivil7_MEAN','agg18_parentesco1_MEAN',
                'agg18_parentesco2_MEAN','agg18_parentesco3_MEAN','agg18_parentesco5_MEAN','agg18_parentesco9_MEAN',
                'agg18_instlevel1_MEAN','agg18_instlevel2_MEAN','agg18_instlevel3_MEAN','agg18_instlevel4_MEAN',
                'agg18_instlevel5_MEAN','agg18_instlevel6_MEAN','agg18_instlevel7_MEAN','agg18_instlevel8_MEAN',
                'agg18_instlevel9_MEAN']

X.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)
test.drop((cols_2_drop+['idhogar']), axis=1, inplace=True)


# In[ ]:


#use the following to test model generalization
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=314, stratify=y)


# In[ ]:


#display(X_train.shape,test.shape)


# In[ ]:


#num_cols = poverty_train.columns[poverty_train.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
#num_cols = train.select_dtypes(include=[np.float32, np.int])
#num_cols.head()


# In[ ]:


#cols_to_norm = ['v2a1','meaneduc', "overcrowding", "SQBovercrowding", "SBQdependency", 
                #"SQBmeaned", "age", "SQBescolari", "escolari",
                #"SQBage", "SQBhogar_total", "SQBedjefe", "SQBhogar_nin",
                #"agesq"]
#train[cols_to_norm] = survey_data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# In[ ]:


#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#train[num_cols] = scaler.fit_transform(train[num_cols])


# In[ ]:


#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()

#X_train = X
#y_train = train["Target"].copy()
#X_train_scaled = scaler.fit_transform(X_train)


# Now we fit a regularized Logistic Regression to use as a baseline model.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from scipy.stats import expon, reciprocal

param_distribs = { 
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 0.25, 0.50, 0.75, 1, 5,
              10, 500, 750, 1000, 1250, 1500, 2000, 10000],
        #'multi_class': ['ovr', 'multinomial'],
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced']
        
    }

logit = LogisticRegression(random_state=42, solver = "liblinear", multi_class = 'ovr',
                             n_jobs = 4, max_iter = 200)
rnd_search = GridSearchCV(logit, param_grid=param_distribs,
                                cv=5, scoring='f1_macro',
                                verbose=2, n_jobs=4)
rnd_search.fit(X, y)


# In[ ]:


cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


# In[ ]:


print(rnd_search.best_score_, rnd_search.best_estimator_, rnd_search.best_params_)


# In[ ]:


#to test model generalization
#from sklearn.metrics import f1_score
#y_pred = rnd_search.predict(X_test)
#f1_score(y_test, y_pred, average='macro')


# In[ ]:


#fit the entire model with selected parameters.


# In[ ]:


y_subm = pd.read_csv('../input/sample_submission.csv')
y_subm['Target'] = rnd_search.predict(test) + 1
y_subm.to_csv('submission.csv', index=False)


# In[ ]:


#from sklearn.model_selection import RandomizedSearchCV
#from scipy.stats import expon, reciprocal
#from sklearn.svm import SVC

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
#param_distribs = {
        #'kernel': ['linear', 'rbf', 'poly'],
        #'C': reciprocal(20, 200000),
        #'gamma': expon(scale=1.0),
#}

#svm = SVC()
#rnd_search = RandomizedSearchCV(svm, param_distributions=param_distribs,
                                #n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                #averbose=2, n_jobs=4, random_state=42)
#rnd_search.fit(X_train_scaled, y_train)


# In[ ]:


#cvres = rnd_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #print(mean_score, params)

