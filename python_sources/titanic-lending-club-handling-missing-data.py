#!/usr/bin/env python
# coding: utf-8

# **Hello Everyone , I am Adarsh**
# 
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR6q30jOVrk9-VKiq3cGYC5gAd36D0Heip8W3O2ocMfIOya1_sS&usqp=CAU)
# 
# This notebook acts as a tutorial which summarizes the main techniques for Handling Missing data with their short descriptions and implementation.
# 
# **If you like my work and presentaion, please give a Up-vote to the work book as it will motiate me to share more of my learnings !!**

# ## Missing Values
# 
# Missing data, or missing values, occur when __no data__ / __no value__ is stored for certain observations within a variable. 
# 
# Incomplete data is an unavoidable problem in dealing with most data sources. Missing data is a common occurrence in both data science competitions and business datasets, and may have a significant impact on the conclusions that can be derived from the data. 
# 
# ### Why is data missing?
# 
# The source of missing data can be very different. These are just a few examples:
# 
# - A value is missing because it was forgotten, lost or not stored properly
# - For a certain observation, the value of the variable does not exist
# - The value can't be known or identified
# 
# In many organisations, information is collected manually into a form by a person talking with a client on the phone, or alternatively, by customers filling forms online. Often, the person entering the data does not complete all the fields in the form. Many of the fields are not compulsory, and therefore, those values will be missing. The reasons for omitting the information in those fields can vary: perhaps the client does not want to disclose some information, for example income, or they do not know the answer, or the answer is not applicable for a certain circumstance, or on the contrary, the person in the organisation wants to spare the client some time, and therefore omits asking questions they think are not so relevant.
# 
# There are other cases where the value for a certain variable does not exist. For example, in the variable 'total debt as percentage of total income' (very common in financial data), if the person has no income, then the total percentage of 0 does not exist, and therefore it will be a missing value.
# 
# It is important to understand **how the missing data are introduced in the dataset**, or in other words, the **mechanisms** by which missing information is introduced in a dataset. Depending on the mechanism, we may choose to process the missing values differently. In addition, by knowing the source of missing data, we may choose to take action to control that source, and decrease the number of missing data looking forward during data collection.
# 
# 
# ### Missing Data Mechanisms
# 
# There are 3 mechanisms that lead to missing data, 2 of them involve missing data randomly or almost-randomly, and the third one involves a systematic loss of data.
# 
# #### Missing Completely at Random, MCAR:
# 
# A variable is missing completely at random (MCAR) if the probability of being missing is the same for all the observations. 
# When data is MCAR, there is absolutely no relationship between the data missing and any other values, observed or missing, within the dataset. In other words, those missing data points are a random subset of the data. There is nothing systematic going on that makes some data more likely to be missing than other. If values for observations are missing completely at random, then disregarding those cases would not bias the inferences made.
# 
# 
# #### Missing at Random, MAR: 
# 
# MAR occurs when there is a relationship between the propensity of missing values and the observed data. In other words, the probability of an observation being missing depends on available information (i.e., other variables in the dataset). For example, if men are more likely to disclose their weight than women, weight is MAR. The weight information will be missing at random for those men and women who do not disclose their weight, but as men are more prone to disclose it, there will be more missing values for women than for men.
# 
# In a situation like the above, if we decide to proceed with the variable with missing values (in this case weight), we might benefit from including gender to control the bias in weight for the missing observations.
# 
# 
# #### Missing Not at Random, MNAR: 
# 
# Missing data is not at random (MNAR) when there is a mechanism or a reason why missing values are introduced in the dataset. For example, MNAR would occur if people failed to fill in a depression survey because of their level of depression. Here, the missing of data is related to the outcome, depression. Similarly, when a financial company asks for bank and identity documents from customers in order to prevent identity fraud, typically, fraudsters impersonating someone else will not upload documents, because they don't have them, because they are fraudsters. Therefore, there is a systematic relationship between the missing documents and the target we want to predict: fraud.
# 
# Understanding the mechanism by which data is missing is important to decide which methods to use to impute the missing values.
# 
# 
# ## Real Life example: 
# 
# ### Predicting Survival on the Titanic: understanding society behaviour and beliefs
# 
# Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time.
# 
# ### Peer to peer lending: Finance
# 
# Lending Club is a peer-to-peer Lending company based in the US. They match people looking to invest money with people looking to borrow money. When investors invest their money through Lending Club, this money is passed onto borrowers, and when borrowers pay their loans back, the capital plus the interest passes on back to the investors. It is a win for everybody as they can get typically lower loan rates and higher investor returns.
# 
# If you want to learn more about Lending Club follow this [link](https://www.lendingclub.com/)
# 
# The Lending Club dataset contains complete loan data for all loans issued through the 2007-2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information. Features (aka variables) include credit scores, number of finance inquiries, address including zip codes and state, and collections among others. Collections indicates whether the customer has missed one or more payments and the team is trying to recover their money. The file is a matrix of about 890 thousand observations and 75 variables. More detail on this dataset can be found in [Kaggle's website](https://www.kaggle.com/wendykan/lending-club-loan-data)
# 
# ====================================================================================================
# 
# To download the Titanic data, go to the [Kaggle website](https://www.kaggle.com/c/titanic/data)
# 
# Click on the link 'train.csv', and then click the 'download' blue button towards the right of the screen, to download the dataset. Save it in a folder of your choice. Rename the file to 'titanic.csv'.
# 
# For the Lending Club dataset. go to this [website](https://www.kaggle.com/wendykan/lending-club-loan-data)
# 
# Scroll down to the bottom of the page, and click on the link 'loan.csv', and then click the 'download' blue button towards the right of the screen, to download the dataset. Unzip it, and save it to a directory of your choice.
# 
# **Note that you need to be logged in to Kaggle in order to download the datasets**.
# 
# ====================================================================================================

# ## In this Demo:
# 
# In the following cells we will:
# 
# - Learn how to detect and quantify missing values
# 
# - Try to identify the 3 different mechanisms of missing data introduction

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# let's load the titanic dataset
data = pd.read_csv('../input/titanic/train.csv')

# let's inspect the first 5 rows
data.head()


# In python, the missing values are stored as NaN, see for example the first row for the variable Cabin.

# In[ ]:


# we can quantify the total number of missing values using
# the isnull method plus the sum method on the dataframe

data.isnull().sum()


# There are 177 missing values for Age, 687 for Cabin and 2 for Embarked.

# In[ ]:


# alternatively, we can use the mean method after isnull
# to visualise the percentage of
# missing values for each variable

data.isnull().mean()


# There are missing data in the variables Age (19% missing), Cabin -in which the passenger was traveling- (77% missing), and Embarked -the port from which the passenger got into the Titanic- (0.2%  missing).

# ## Mechanisms of Missing Data
# 
# ### Missing data Not At Random (MNAR): Systematic missing values
# 
# In the Titanic dataset, both the missing values of the variables Cabin and Age, were introduced systematically. For many of the people who did not survive, the **age** they had or the **cabin** they were traveling in, could not be established. The people who survived could be otherwise asked for that information.
# 
# Can we infer this by looking at the data?
# 
# In a situation like this, we could expect a greater number of missing values for people who did not survive.
# 
# Let's have a look.

# In[ ]:


# let's create a binary variable that indicates 
# whether the value of cabin is missing

data['cabin_null'] = np.where(data.Cabin.isnull(), 1, 0)


# In[ ]:


# let's evaluate the percentage of missing values in
# cabin for the people who survived vs the non-survivors.

# the variable Survived takes the value 1 if the passenger
# survived, or 0 otherwise

# group data by Survived vs Non-Survived
# and find the percentage of nulls for cabin
data.groupby(['Survived'])['cabin_null'].mean()


# In[ ]:


# another way of doing the above, with less lines
# of code :)

data['Cabin'].isnull().groupby(data['Survived']).mean()


# We observe that the percentage of missing values is higher for people who did not survive (87%), respect to people who survived (60%). This finding is aligned with our hypothesis that the data is missing because after people died, the information could not be retrieved.
# 
# **Note**: Having said this, to truly underpin whether the data is missing not at random, we would need to get extremely familiar with the way data was collected. Analysing datasets, can only point us in the right direction or help us build assumptions.

# In[ ]:


# Let's do the same for the variable age:

# First we create a binary variable to indicates
# whether the value of Age is missing

data['age_null'] = np.where(data.Age.isnull(), 1, 0)

# and then look at the mean in the different survival groups:
data.groupby(['Survived'])['age_null'].mean()


# In[ ]:


# or the same with simpler code :)

data['Age'].isnull().groupby(data['Survived']).mean()


# Again, we observe a higher number of missing data for the people who did not survive the tragedy. The analysis therefore suggests that there is a systematic loss of data: people who did not survive tend to have more missing information. Presumably, the method chosen to gather the information, contributes to the generation of these missing data.

# ### Missing data Completely At Random (MCAR)

# In[ ]:


# In the titanic dataset, there are also missing values
# for the variable Embarked.
# Let's have a look.

# Let's slice the dataframe to show only the observations
# with missing values for Embarked

data[data.Embarked.isnull()]


# These 2 women were traveling together, Miss Icard was the maid of Mrs Stone.
# 
# A priori, there does not seem to be an indication that the missing information in the variable Embarked is depending on any other variable, and the fact that these women survived, means that they could have been asked for this information.
# 
# Very likely the values were lost at the time of building the dataset.
# 
# If these values are MCAR, the probability of data being missing for these 2 women is the same as the probability for values to missing for any other person on the titanic. Of course this will be hard, if possible at all, to prove. But I hope this serves as a demonstration.

# ### Missing data at Random (MAR)
# 
# For this example, I will use the Lending Club loan book. I will look at the variables employer name (emp_title) and years in employment (emp_length), both declared by the borrowers at the time of applying for a loan. emp_title refers to the name of the company for which the borrower works. emp_length refers to how many years the borrower has worked for the company mentioned in emp_title. In this example, data missing in emp_title is associated with undeclared length of work in emp_length.

# In[ ]:


# let's load the columns of interest from the
# Lending Club loan book dataset

##########################################
# Note: newer versions of pandas automatically cast strings as NA,
# so to follow along with the notebook load the data as below if using
# the latest pandas version. Loading method may need to be adjusted if
# using older versions of pandas
##########################################

data = pd.read_csv('../input/lending-club-loan-data/loan.csv',
                   usecols=['emp_title', 'emp_length'],
                   na_values='',
                   keep_default_na=False)
data.head()


# In[ ]:


# let's check the percentage of missing data
data.isnull().mean()


# Around 6% of the observations contain missing data for emp_title. No values are missing for emp_length.

# In[ ]:


# let's insptect the different employer names

# number of different employers names
print('Number of different employer names: {}'.format(
    len(data.emp_title.unique())))

# a few examples of employers names
data.emp_title.unique()[0:20]


# We observe the missing information (nan), and several different employer names.

# In[ ]:


# let's inspect the variable emp_length
data.emp_length.unique()


# The value 'n/a', "not applicable" is the one we are interested in. The customer can't enter an employment length, perhaps because they are not employed. They could be students, retired, self-employed, or work in the house.

# In[ ]:


# let's look at the percentage of borrowers within
# each label / category of emp_length variable

# value counts counts the observations per category
# if we divide by the number of observations (len(data))
# we obtain the percentages of observations per category

data.emp_length.value_counts() / len(data)


# 5 % of the borrowers in the dataset have disclosed 'n/a' for emp_lenght. From previous cells, we know that ~5% of the borrowers present missing data for emp_title. Could there be a relationship between missing values in emp_title and 'n/a' in emp_length? Let's have a look.

# In[ ]:


# the variable emp_length has many categories.
# I will summarise it into 3 for simplicity:
# '0-10 years' or '10+ years' or 'n/a'

# let's build a dictionary to re-map emp_length to just 3 categories:

length_dict = {k: '0-10 years' for k in data.emp_length.unique()}
length_dict['10+ years'] = '10+ years'
length_dict['n/a'] = 'n/a'

# let's look at the dictionary
length_dict


# In[ ]:


# let's re-map the emp_length variable

data['emp_length_redefined'] = data.emp_length.map(length_dict)

# let's see if it worked
data.emp_length_redefined.unique()
data.head()


# It worked, our new variable has only 3 different categories.

# In[ ]:


# let's calculate the proportion of working years
# with the same employer for those who miss data on emp_title

# data[data.emp_title.isnull()] represents the observations
# with missing data in emp_title. I use this below:

# Calculations:
# number of borrowers for whom employer name is missing
# aka, not employed people
not_employed = len(data[data.emp_title.isnull()])

# % of borrowers for whom employer name is missing
# within each category of employment length

data[data.emp_title.isnull()].groupby(
    ['emp_length_redefined'])['emp_length'].count().sort_values() / not_employed


# The above output tells us the following:
# From all the borrowers who show missing information in emp_title, so those who are not employed:
# - 5.4% declared more than 10 years in emp_length (maybe they are self-employed)
# - 8.4% declared between 0-10 years in emp_length (same as above, potentially self-employed)
# - 86.3 % declared n/a in emp_length (maybe they are students, or work at home, or retired)
# 
# The majority of the missing values in emp_title coincides with the label 'n/a' of emp_length (86%). This supports the idea that the 2 variables are related. Therefore, the missing data in emp_title, is MAR.

# In[ ]:


# let's do the same for those bororwers who reported
# the employer name

# number of borrowers for whom employer name is present:
# employed people
employed = len(data.dropna(subset=['emp_title']))

# % of borrowers within each category
data.dropna(subset=['emp_title']).groupby(
    ['emp_length_redefined'])['emp_length'].count().sort_values() / employed


# The number of borrowers who have reported an employer name (emp_title) and yet indicate 'n/a' as emp_length is minimal. This further supports that the missing values in emp_title are related to 'n/a' in emp_length.
# 
# 'n/a' in emp_length could be supplied by people who are retired, or students, or self-employed. In those cases there would not be a number of years in employment to provide, therefore the customer would enter 'n/a' in emp_length and leave the form empty for 'emp_title'.
# 
# A missing value in the variable emp_title depends on, or is related to, the 'n/a' label in the variable emp_length. This is an example of MAR. The value in emp_title is missing at random for those customer who are not employed, but emp_length being missing is influenced by emp_title being missing.

# **That is all for this demonstration. I hope you enjoyed the notebook, and see you in the next one.**
