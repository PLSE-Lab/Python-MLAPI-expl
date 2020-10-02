#!/usr/bin/env python
# coding: utf-8

# <div style="color: green; font-weight:bold;">Hey there, this is a school project that I've done and I thought I would just upload it. Do leave any comments or criticism so that I can improve. 
# 
# NOTE: I commented away a huge part (starting from GridSearch) towards the end as I did not wanted to waste too much time for it. Feel free to fork this notebook and run it yourself. </div>

# # <u>Adult Census Income</u>
# 
# The dataset (https://archive.ics.uci.edu/ml/datasets/Adult) belongs to and is managed by the University of California Irvine. It was donated by Ronny Kohavi and Barry Becker in 1994.
# 
# Problem statement: Predict whether the income of a specific group of people exceeds $50K/yr or not based on census (survey) data.
# This problem is a binary classification problem.
# 
# <b>Table of contents:</b>
# <br>
# 1) Data Preparation
# <br>
# 2) Exploratory Data Analysis
# <br>
# 3) Feature Engineering
# <br>
# 4) Modelling
# <br>
# 5) Conclusion
# <br>
# 6) References

# ### Load libraries

# In[ ]:


# Loading necessary libraries 

# Data analysis and wrangling
print("Data Analysis and Wrangling Packages:")
import pandas as pd # Collection of functions for data processing and analysis 
# modeled after R dataframes with SQL like features.
print("- pandas version: {}". format(pd.__version__))

import numpy as np # Foundational package for scientific computing.
print("- NumPy version: {}". format(np.__version__))

import scipy as sp # Collection of functions for scientific computing and advance mathematics.
print("- SciPy version: {}". format(sp.__version__)) 
print('-'*40)

# Visualization (Exploratory Data Analysis)
print("Visualization Packages:")
import matplotlib # Collection of functions for scientific and publication-ready visualization.
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
print("- matplotlib version: {}". format(matplotlib.__version__))
print('-'*40)

# Modelling
print("Modelling Packages:")
import sklearn # Collection of machine learning algorithms.
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_score, 
StratifiedKFold, learning_curve, train_test_split, KFold)
from sklearn.externals import joblib # Save model.
print("- scikit-learn version: {}". format(sklearn.__version__))
print('-'*40)

# Misc libaries
import re # To use RegEx functions.
import warnings
import random as rnd
warnings.filterwarnings('ignore') # Ignore warnings (e.g. depreciation warnings).


# # 1) <u>Data Preparation<u/>

# ### Importing data

# In[ ]:


# Aquiring data
data = pd.read_csv('../input/adult-census-income/adult.csv') # Reading a comma-separated values (csv) file into DataFrame.


# ### See available features in dataset

# In[ ]:


print(data.columns.values) # Prints what available features are in the dataset.


# <div style="color:green;">
#     The feature "fnlwgt" is an abbreviation for "Final Weight" (datatype: cardinal number). This feature refers to an estimated number of people each row represents.
#     For example, a specific row may have fnlwgt=2500, age=50, race=White, sex=female ...etc.
#     This would mean 2500 people fall into the category of being 50 years old, White, female etc. Logically speaking, since this feature basically represents the count of a specific group of people represented, it should not affect the label. I will be conducting some analysis to see if my judgement is correct or not. If it indeed does not affect the label, then I will drop it.
# <br><br>
# The feature "education.num" is the mapped version of the "education" feature. For example, education=HS-grad -> education.num=9. Since they both represent the same thing, I will be removing one of them. Machine works better with numbers over strings/text, thus I will be removing the feature "education" in Feature Engineering. </div> 

# ### Renaming Features

# In[ ]:


data = data.rename(columns={ 
    # Renaming the features based on standard conventions and my preference so that it is more pleasing and easier to understand.
    # The original names of some features will also affect/make it difficult when calling methods as they have '.' in between (e.g. marital.status).
    'workclass': 'work_class', 
    'fnlwgt': 'num_of_ppl_rep', 
    'education.num': 'education_num', 
    'marital.status': 'marital_status', 
    'capital.gain': 'capital_gain', 
    'capital.loss': 'capital_loss',
    'hours.per.week': 'hours_per_week', 
    'native.country': 'native_country'
})


# ### Basic peeks and checks to get a feel of the dataset and features

# In[ ]:


data.info() # Appears to not have any null values


# <div style="color: green;">
#     Based on the DataFrame.info() peek, it seems that the dataset does not have any missing values (Null, NaN, etc.). When missing values are present, there will be a mismatch between the entries (total rows vs available data in each features). In this case, all feature has the same value as the number rows/entries.
#     </div>

# In[ ]:


data.head() # Preview of the data (first 5 rows).


# In[ ]:


data.tail() # Preview of the data (last 5 rows).


# <div style="color: green;">
#     The DataFrame.head() and DataFrame.tail() peek gives me a preview of the dataset and allows me to see the features' data type.
#     <br>
#     Findings: 
#     <br>
#     1) Although no missing values are present, there are values of '?' present. This will require correction as soon as possible.
#     <br>
#     2) Label "income" and the features "work_class", "education", "marital_status", "occupation", "relationship", "race", "sex" and "native_country" are alphabetic. I will likely have to convert/bin/map them into numeric values so that the machine learning algorithm works better.
#     <br>
#     3) Features "age", "education_num" and "hours_per_week" can be binned into categories.
#     <br>
#     4) Label "income" and the feature "sex" can be converted to binary number. 
#     </div>

# ### Check for features with the value '?'

# In[ ]:


print("Number of '?' in:\n")
print("Column", 11*' ', "Count")
print("-"*25)
for i in data.columns: # i refers to features.
    t = data[i].value_counts() # t refers to count of each feature.
    index = list(t.index)
    print(i, end="")
    x = 20 - len(i) # For styling purposes
    for j in index:
        temp = 0
        if j == '?':
            print (x * ' ', t['?']) # Once a '?' is found, print the number of '?' in the feature.
            temp = 1
            break
    if temp == 0:
        print (x * ' ', "0") # '?' is absent from all rows of a specific feature.


# <div style="color: green;">
#     Based on this simple nested for-loop, we can see that features "work_class", "occupation" and "native_country" have '?' in some rows. This requires correction and to deal with this, we have 3 possible methods. Firstly, I can remove the feature. Secondly, I can fill the '?' data with the mode of the feature (*Note: only mode is available because all 3 features are alphabetic [mean and median cannot be used]). Lastly, I can drop rows that have '?' as a value for any of their feature.
#     <br>
#     <br>
#     Since the number of '?' present in the 3 features is not high (~6% for "work_class" and "occupation", ~2% for "native_country"), there is no need to drop the entire feature.
#     </div>

# ### Analysing "work_class" for data correction

# In[ ]:


data.work_class.value_counts() # See the spread of "work_class".


# <div style="color: green;">
#     Based on the DataFrame.Series.value_counts() peek, we can see that the mode of "work_class", "Private" is ahead by a large margin, accounting for ~70% of the feature. As such, I will be using option number 2 which is to fill the '?' values with the mode of the feature.
#     </div>

# ### Correcting "work_class" by filling '?' values with mode

# In[ ]:


data.work_class.replace('?', np.nan, inplace=True) # Replace all '?' to null values (NaN).
data.work_class.fillna(data.work_class.mode()[0], inplace=True) # Fill all null values with mode.


# ### Analysing "occupation" for data correction

# In[ ]:


data.occupation.value_counts() # See the spread of "occupation".


# <div style="color: green;">
#     Based on the peek, we can see that the data in "occupation" are quite evenly spread. The mode ("Prof-specialty") is only slightly more as compared to other values and it only accounts to about ~10% of the feature's data. Hence, instead of filling '?' with the mode which will mess up the spread, I will drop the rows where "occupation"='?'.
#     </div>

# ### Correcting "occupation" by dropping rows where "occupation"='?'

# In[ ]:


data = data[data.occupation != '?'] 


# ### Analysing "native_country" for data correction

# In[ ]:


data.native_country.value_counts() # See the spread of "native_country".


# <div style="color: green;">
#     Similar to the "work_class" feature, the mode of "occupation" ("United-States") is also ahead by a large margin (accounts for about ~85% of the feature's value). Therefore I will be using option number 2 which is to fill '?' values with the mode of the feature. 
#     </div>

# ### Correcting "native_country" by filling '?' values with mode

# In[ ]:


data.native_country.replace('?', np.nan, inplace=True) # Replace all '?' to null values (NaN).
data.native_country.fillna(data.native_country.mode()[0], inplace=True) # Fill all null values with mode.


# ### Check

# In[ ]:


# Prior to the data correction, DataFrame.head() had some '?' in some fields. Thus to check, I will use DataFrame.head() again.
data.head() # Check the first 5 rows to see if the data has been corrected successfully.


# <div style="color: green;">
#     Data has been cleaned, corrected and prepared. We could move onto Data Exploration straight away, however, the label ("income_above_50k") is string based. This will make it extremely hard or even impossible to perform any sorts of visualization and comparison. Thus, I will be performing a minor feature engineering first.
#     </div>

# ### Engineering the label

# In[ ]:


data.income.value_counts() # See the spread and the unique values of the label.


# In[ ]:


# Changing the label to binary number.
data['income_above_50k'] =data['income'].apply(lambda x: 1 if x=='>50K' else 0) # "income"='>50k' -> 1, else (<=50k) -> 0.
data.drop(['income'], axis=1, inplace=True) # Since a new label will be created, the original label is redundant.
# Since the label has been mapped to a binary value column (previously string), I can now perform multiple analysis through visualizations (graphs etc).
# and see the correlations between features and the label.


# In[ ]:


data.income_above_50k.value_counts() # Checking if the engineering is successful.


# <div style="color: green;">
#     *Note: Although this is feature engineering, this section is placed in Data Preparation as this engineering is required for Exploratory Data Analysis to work (correlate label and features). The main bulk of feature engineering will be below Exploratory Data Analysis, in Feature Engineering.
#     </div>

# # 2) <u>Exploratory Data Analysis</u>

# ### Count plot: "income_above_50k" (label)

# In[ ]:


sns.countplot(data['income_above_50k'],label="Count") # See the spread of the label.


# <div style="color: green;">
#     Judging from the count plot, we can see that majority (75%) of the rows have an income of below 50k per year. This would mean that for every "income_above_50k"=1, there will be around 3 "income_above_50k"=0.
#     </div>

# ### Correlation matrix diagram

# In[ ]:


corr = data.corr() # Gets the correlation matrix of all numerical columns.
corr.style.background_gradient(cmap='coolwarm') # Styles the matrix diagram.


# <div style="color: green;">
#     From this correlation matrix diagram, we can see the "num_of_ppl_rep" does not correlate much with other columns, especially the label. This mean that it will not affect the model much, so I will be removing this feature at the Feature Engineering section. This verified my judgement at the early stages of Data Preparation. <br><br>
#     Since the other features correlate fairly well with the label ("income_above_50k"), I will likely be keeping them.
#     </div>

# # 3) <u>Feature Engineering</u>

# In[ ]:


data.nunique() # Check the number of distinct values of all the features.


# <div style="color: green;">
#     Knowing the number of unique value of each feature will help in the decision making in the Feature Engineering section.
#     </div>

# ### Creating a Binning function 

# In[ ]:


def binning_func(dataframe, feature, bins, group_names):
    bin_value = bins
    group = group_names
    dataframe[feature + '_bin'] = pd.cut(dataframe[feature], bin_value, labels=group) # pd.cut function to bin
# This binning function will create a new feature named as the old feature + "_bin".


# <div style="color: green;">
#     This binning function will come in handy for some features. Features that I already know can be binned are "age", "education_num" and "hours_per_week" (Based on the prior peeks in the Data Preparation section).
#     </div>

# <div style="color: green;">
#     Binning steps:<br>
#     1 - Check the min and max of the feature to know the range.<br>
#     2 - Bin (separate/categorize) them logically (domain knowledge, norms and common sense).<br>
#     3 - Check the unique count.<br>
#     </div>

# ### Feature engineering (binning) "education_num":

# - Checking the min and max of "education_num"

# In[ ]:


print("Min:", data.education_num.min()) # Get the lowest value of the feature (numeric).
print("Max:", data.education_num.max()) # Get the highest value of the feature (numeric).


# - Binning "education_num"

# In[ ]:


# Range of data: 1 ~ 16.
binning_func(data, 'education_num', [0,4,8,12,20], [0, 1, 2, 3]) 
# "education_num" -> "education_num_bin".
# 0~3 = 0 (Lowly education)
# 4~7 = 1 (Decently educated)
# 8~11 = 2 (Highly educated)
# 12~20 = 3 (Very highly educated)
# Went slightly out of range just in case new data falls outside of the current min and max.


# - Checking newly binned feature "education_num_bin"

# In[ ]:


data.education_num_bin.value_counts() # Verifying the newly created bin feature "education_num_bin" and also checking the spread. 


# ### Feature engineering (binning) "age":

# - Checking the min and max of "age"

# In[ ]:


print("Min:", data.age.min()) # Get the lowest value of the feature (numeric).
print("Max:", data.age.max()) # Get the highest value of the feature (numeric).


# - Binning "age"

# In[ ]:


# Range of data: 17 ~ 90.
binning_func(data, 'age', [13, 20, 35, 55, 100], [0, 1, 2, 3])
# "age" -> "age_bin".
# 13~19 = 0 (Young/Teen)
# 20~34 = 1 (Young adult)
# 35~54 = 2 (Middle age)
# 55~99 = 3 (Old)
# Went slightly out of range just in case new data falls outside of the current min and max.


# - Checking newly binned feature "education_num_bin"

# In[ ]:


data.age_bin.value_counts() # Verifying the newly created bin feature "age_bin" and also checking the spread. 


# ### Feature engineering (binning) "hours_per_week":

# - Checking the min and max of "hours_per_week"

# In[ ]:


print("Min:", data.hours_per_week.min()) # Get the lowest value of the feature (numeric).
print("Max:", data.hours_per_week.max()) # Get the highest value of the feature (numeric).


# - Binning "hours_per_week"

# In[ ]:


# Range of data: 1 ~ 99
binning_func(data, 'hours_per_week', [0,30,45,60,100], [0, 1, 2, 3]) # 0-low, 1-medium, 2-high, 3-vhigh
# "hours_per_week" -> "hours_per_week_bin".
# 0~29 = 0 (Low)
# 30~44 = 1 (Medium)
# 45~59 = 2 (High)
# 60~99 = 3 (Very high)


# - Checking the newly binned feature "hours_per_week_bin"

# In[ ]:


data.hours_per_week_bin.value_counts() # Verifying the newly created bin feature "hours_per_week_bin" and also checking the spread. 


# <div style="color: green;">
#     Although the binning feature was useful, its usage has come to an end. The remaining features (excluding "capital_gain" & "capital_loss") cannot be binned as they are alphabetical values. Thus, instead of binning, I will be mapping these alphabetic features to numerical values. As for the features "capital_gain" & "capital_loss", I do not think that they need to be feature engineered.
#     <br>
#     <br>
#     Target for mapping:<br>
#     - "sex" <br>
#     - "work_class"<br>
#     - "relationship"<br>
#     - "native_country"<br>
#     - "occupation"<br>
#     - "race"<br>
#     - "marital_status"
#     </div>

# ### Feature engineering (mapping) "sex"

# - Check the spread of "sex"

# In[ ]:


data.sex.value_counts() # Checking the spread and all the possible values of the feature.


# - Mapping "sex"

# In[ ]:


data["sex_cat"] = data["sex"].map({"Male": 0, "Female":1}) # Map "sex"="Male" -> 0, "sex"="Female" -> 1.


# - Verifying and checking the newly created feature "sex_cat"

# In[ ]:


data.sex_cat.value_counts() # Verifying by comparing the values with the original feature's DataFrame.Series.value_counts() peek.


# ### Feature engineering (mapping) "work_class"

# - Possible values of "work_class"

# In[ ]:


data.work_class.value_counts() # Checking the spread and all the possible values of the feature.


# <div style="color: green;">
#     Based on the DataFrame.Series.value_counts() peek, we can see that the 7 different values can be further categorized into "Private", "Self-Employed", "Public" and "Others".
#     <br>
#     - "Private" will consist of "Private" from the original values.
#     <br>
#     - "Self-Employed will consist of "Self-emp-not-inc" and "Self-emp-inc" from the original values.
#     <br>
#     - "Public" will consist of "Local-gov", "State-gov" and "Federal-gov" from the original values.
#     <br>
#     - "Others" will consist of "Without-pay" from the original values.
#     </div>

# - Further categorizing "work_class"

# In[ ]:


def map_workclass(x): # Simple function using RegEx function.
    if re.search('Private', x):
        return 'Private'
    elif re.search('Self', x):
        return 'Self-Employed'
    elif re.search('gov', x):
        return 'Public'
    else:
        return 'Others'
data['work_class_cat'] = data.work_class.apply(lambda x: x.strip()).apply(lambda x: map_workclass(x)) # Applying the function.


# - Mapping the new categories to numerical values

# In[ ]:


data["work_class_cat"] = data["work_class_cat"].map({"Others": 0, "Self-Employed": 1, "Public": 2, "Private": 3}) # Mapping the new categories to numerical values.


# - Checking the spread and verifying the feature engineering is successful

# In[ ]:


data.work_class_cat.value_counts() # Verifying the new feature. 


# ### Feature engineering (mapping) "relationship"

# - Possible values of "relationship"

# In[ ]:


data.relationship.value_counts() # Checking the spread and all the possible values of the feature.


# <div style="color: green;">
#     Based on the DataFrame.Series.value_counts() peek, we can see that the 6 different values can be further categorized into "Spouse" and "Others". The spread of the 2 new category will have relatively similar count.
#     <br>
#     - "Spouse" will consist of "Husband" and "Wife" from the original values.
#     <br>
#     - "Others" will consist of "Not-in-family", "Own-child", "Unmarried" and "Other-relative" from the original values.
#     </div>

# - Further categorizing "relationship"

# In[ ]:


def map_relationship(x): # Simple function using RegEx function.
    if re.search('Husband', x):
        return 'Spouse'
    elif re.search('Wife', x):
        return 'Spouse'
    else:
        return 'Others'
data['relationship_cat'] = data.relationship.apply(lambda x: x.strip()).apply(lambda x: map_relationship(x)) # Applying the function.


# - Mapping the new categories to numerical values

# In[ ]:


data["relationship_cat"] = data["relationship_cat"].map({"Others": 0, "Spouse": 1}) # Mapping the new categories to numerical values.


# - Checking the spread and verifying the feature engineering is successful

# In[ ]:


data.relationship_cat.value_counts() # Verifying the new feature. 


# ### Feature engineering (mapping) "native_country"

# - Possible values of "native_country"

# In[ ]:


data.native_country.value_counts() # Checking the spread and all the possible values of the feature.


# <div style="color: green;">
#     Judging from the spread, it is obvious that a large majority of the data in "native_country" are "United-States" (>85%). The next highest count only accounts for ~2% of the feature. As such, I will be categorizing all other values as "Others". 
#     </div>

# - Further categorizing "native_country"

# In[ ]:


def map_native_country(x): # Simple function using RegEx function.
    if re.search('United-States', x):
        return 'US'
    else:
        return 'Others'
data['native_country_cat'] = data.native_country.apply(lambda x: x.strip()).apply(lambda x: map_native_country(x)) # Applying the function.


# - Mapping the new categories to numerical values

# In[ ]:


data["native_country_cat"] = data["native_country_cat"].map({"Others": 0, "US": 1}) # Mapping the new categories to numerical values.


# - Checking the spread and verifying the feature engineering is successful

# In[ ]:


data.native_country_cat.value_counts() # Verifying the new feature. 


# ### Feature engineering (mapping) "occupation"

# - Possible values of "occupation"

# In[ ]:


data.occupation.value_counts() # Checking the spread and all the possible values of the feature.


# <div style="color: green;">
#     Unlike the previous features, "occupation" is more subjective, complex and requires more analysis. In the end, I further categorized the 14 different values into "Technical", "Service", "Support", "High-level" and "Low-level" based on my own understanding of the job scope of each jobs.
#     <br>
#     - "Technical" will consist of "Craft-repair" and "Machine-op-inspc" from the original values.
#     <br>
#     - "Service" will consist of "Sales", "Other-service", "Protective-serv" and "Priv-house-serv" from the original values.
#     <br>
#     - "Support" will consist of "Adm-clerical" and "Tech-support" from the original values.
#     <br>
#     - "High-level" will consist of "Prof-specialty" and "Exec-managerial" from the original values.
#     <br>
#     - "Low-level" will consist of "Transport-moving", "Farming-fishing", "Handlers-cleaners" and "Armed-Forces" from the original values.
#     </div>

# - Further categorizing "occupation"

# In[ ]:


# Since the names of the occupations are so unique, I will have to use the replace() function as the RegEx's search function will not work.
data["occupation_cat"] = data["occupation"].replace(['Craft-repair', 'Machine-op-inspct'], 'Technical')
data["occupation_cat"] = data["occupation_cat"].replace(['Sales', 'Other-service', 'Protective-serv', 'Priv-house-serv'], 'Service')
data["occupation_cat"] = data["occupation_cat"].replace(['Adm-clerical', 'Tech-support'], 'Support')
data["occupation_cat"] = data["occupation_cat"].replace(['Prof-specialty', 'Exec-managerial'], 'High-level')
data["occupation_cat"] = data["occupation_cat"].replace(['Transport-moving', 'Farming-fishing', 'Handlers-cleaners', 'Armed-Forces'], 'Low-level')


# - Mapping the new categories to numerical values

# In[ ]:


# Mapping the new categories to numerical values.
data["occupation_cat"] = data["occupation_cat"].map({"Low-level": 0, "Support": 1, "Technical": 2, "Service": 3, "High-level": 4}) 


# - Checking the spread and verifying the feature engineering is successful

# In[ ]:


data['occupation_cat'].value_counts() # Verifying the new feature.


# ### Feature engineering (mapping) "race"

# - Possible values of "race"

# In[ ]:


data.race.value_counts()  # Checking the spread and all the possible values of the feature.


# <div style="color: green;">
#     Similar to the previous feature ("occupation"), the majority of the "race" feature's values are "White" (~80%). As such, I will be engineering this feature to be binary (1 = "White", 0 = all other values).
#     </div>

# - Mapping the feature as binary

# In[ ]:


data['race_cat'] = data['race'].apply(lambda x: 1 if x=='White' else 0) # Simple function to map "White" as 1 and all other values as "0".


# - Checking the spread and verifying the feature engineering is successful.

# In[ ]:


data.race_cat.value_counts() # Verifying the new feature.


# ### Feature engineering (mapping) "marital_status"

# - Possible values of "marital_status"

# In[ ]:


data.marital_status.value_counts() # Checking the spread and all the possible values of the feature.


# <div style="color: green;">
#     This feature can be logically categorized into "Single" and "Married".
#     </div>

# - Further categorizing "marital_status"

# In[ ]:


# Since the names of the occupations are so unique, I will have to use the replace() function as the RegEx's search function will not work.
data["marital_status_cat"] = data["marital_status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
data["marital_status_cat"] = data["marital_status_cat"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')


# - Mapping the new categories to numerical values

# In[ ]:


data["marital_status_cat"] = data["marital_status_cat"].map({"Married":1, "Single":0}) # Mapping the new categories to binary.


# - Checking the spread and verifying the feature engineering is successful

# In[ ]:


data.marital_status_cat.value_counts() # Verifying the new feature.


# ### Peek to view at the Dataset after all the feature engineering

# In[ ]:


data.head()


# <div style="color: green;">
#     Since the new features looks good and ready to be used, the old features can be dropped. Features that does not affect the feature or repeated features can also be dropped.
#     </div>

# ### Dropping of features

# In[ ]:


data = data.drop(['age', # Taken over by newly engineered feature.
                  'work_class', # Taken over by newly engineered feature.
                  'num_of_ppl_rep', # Does not affect the label.
                  'education', # Same as "education_num", but in alphabetical form. Thus, it is redundant.
                  'education_num', # Taken over by newly engineered feature.
                  'marital_status', # Taken over by newly engineered feature.
                  'occupation', # Taken over by newly engineered feature.
                  'relationship', # Taken over by newly engineered feature.
                  'race', # Taken over by newly engineered feature.
                  'sex', # Taken over by newly engineered feature.
                  'hours_per_week', # Taken over by newly engineered feature.
                  'native_country'# Taken over by newly engineered feature.
                 ], axis=1)


# ### Last checking of all the available features before moving on to Modelling

# In[ ]:


print(data.columns.values) # Prints what available features are in the dataset.


# # 4) <u>Modelling</u>

# ### Preparation

# In[ ]:


X = data.drop(['income_above_50k'], axis=1) # X consists of all the features (label is removed).
y = data['income_above_50k'] # y consist of the label ("income_above_50k").

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 8) # Splitting the data (8/2).


# ### Feature scaling

# In[ ]:


scaler = StandardScaler() # Similar to nominalizing.
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# <div style="color: green;">
#     Using StandardScaler may improve the speed and results of the model training. Some alortihms also require feature scaling for it to work, and even if my models do not need it, it is still a good idea to standardize the data.
#     </div>

# ### Model training

# <div style="color: green;">
#     To ensure optimal results, I will be training with various different algorithms. I've selected 9 different machine algorithms that are compatible with my problem statement (binary classification). As much as possible, all 9 algorithms will be used with their default values/parameters.
#     </div>

# In[ ]:


models = [] 
# 9 different machine learning algorithms
models.append(("Logistic Regression", LogisticRegression()))
models.append(("Support Vector Classifier (SVC)", SVC()))
models.append(("K-Nearest Neighbour (KNN)",  KNeighborsClassifier(n_neighbors=3)))
models.append(("Gaussian Naive Bayes", GaussianNB()))
models.append(("Perceptron", Perceptron()))
models.append(("Linear Support Vector", LinearSVC()))
models.append(("Stochastic Gradient Descent", SGDClassifier()))
models.append(("Decision Tree", DecisionTreeClassifier()))
models.append(("Random Forest", RandomForestClassifier(n_estimators=100)))
results = [] # Used to store the result of each algorithm.
model_names =[] # Used to store the name of each algorithm for visualization purposes.
for name, model in models:
    kfold = KFold(n_splits=10, random_state=8) # Cross-validation. Using 10 folds is an overkill but it doesn't matter as this training will not take too long.
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy') # Standard of measure: 'accuracy' (suitable for my classification problem).
    results.append(round(cv_results.mean() * 100, 2)) # Appending the results (Using the mean of the 10 results).
    model_names.append(name) # Appending the name of the algorithms
print("Training complete.") # To give an indication when the training is done.


# ### Results

# In[ ]:


compare_models = pd.DataFrame({
    'Model': model_names,
    'Score': results})
compare_models.sort_values(by='Score', ascending=False)


# <div style="color: green;">
#     Based on the comparison, we can see that the Random Forest (RF) algorithm performed the best followed by the Decision Tree (DT) algorithm. Since RF performed the best and it is less likely to overfit unlike the DT algorithm, I will be using RF as my main model.
#     <br>
#     <br>
#     Now that I know which model I will be using, I can proceed to performing Grid Search to further improve the result.
#     </div>

# ## I will be commenting away my GridSearch and RandomizedSearch since it'll take too long. Besides, I was just testing it out in my original notebook.

# ### Grid Search

# <div style="color: green;">
#     I will be using 2 types of algorithms, RandomizedSearchCV & the more popular GridSearchCV on the Random Forest machine learning algorithm.
#     <br><br>
#     *Note: Using these 2 algorithms on all 9 machine learning algorithms will be too resource and time consuming. Thus, I will only be performing it on the best performing machine learning algorithm. 
#     </div>

# - RandomizedSearchCV

# In[ ]:


"""
# Number of trees in random forest.
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, Y_train)
"""


# In[ ]:


"""
rf_random.best_params_ # Get the best parameters from the RandomizedSearchCV
"""


# In[ ]:


"""
# Train a new Random Forest model with the best parameters from RandomizedSearchCV.
forestRS = RandomForestClassifier(random_state = 1, max_depth = 100, n_estimators = 800, min_samples_split = 10, min_samples_leaf = 4, bootstrap = True, max_features = 'sqrt')
forestRS.fit(X_train, Y_train)
Y_predRS = forestRS.predict(X_test)
acc_rf_rs = round(forestRS.score(X_train, Y_train) * 100, 2)
acc_rf_rs
"""


# <div style="color: green;">
#     The result for the RF model after using RandomSearchCV is 86.66. There is an improvement of 1.67
#     when compared to the RF model with the default parameters (84.99).
#     </div>

# - GridSearchCV

# In[ ]:


"""
rf = RandomForestClassifier(random_state = 1)

# Parameters and their range.
n_estimators = [100, 300, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 100]
min_samples_leaf = [1, 2, 5, 10] 
# There will be 500 permutations.

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(rf, hyperF, cv = 3, verbose = 1, n_jobs = -1)
rf_Grid = gridF.fit(X_train, Y_train)
"""


# In[ ]:


"""
rf_Grid.best_params_ # Get the best parameters from the GridSearchCV.
"""


# In[ ]:


"""
# Train a new Random Forest model with the best parameters from GridSearchCV.
forestGS = RandomForestClassifier(random_state = 1, max_depth = 15, n_estimators = 800, min_samples_split = 5, min_samples_leaf = 2)
forestGS.fit(X_train, Y_train)
Y_predGS = forestGS.predict(X_test)
acc_rf_gs = round(forestGS.score(X_train, Y_train) * 100, 2)
acc_rf_gs
"""


# <div style="color: green;">
#     The result for the RF model after using GridSearchCV is 86.89. There is an improvement of 1.9 when compared to the RF model with the default parameters (84.99).
#     <br>
#     It also performed better than the RF model that used RandomizedSearchCV (improvement of 0.23). However, it is not really a fair comparison as the GridSearchCV had more fits, but it does not matter as my main aim is to get a good parameters for the Random Forest algorithm and not compare hyper-parameter tuning algorithms.
# </div>

# # 5) <u>Conclusion</u>

# With an accuracy of 86.89, the algorithm that worked the best for this problem is the Random Forest Classifier machine learning algorithm coupled with the parameters generated from the GridSearchCV algorithm.

# ### Feature importance

# In[ ]:


"""
feature_importance_data = data.drop(['income_above_50k'], axis=1) 
feature_importance_data.columns.values
"""


# <div style="color: green;">
#     Get all the features name to copy and paste to an array.
#     </div>

# In[ ]:


"""
# All the features in the correct order.
feature_labels = ['capital_gain', 'capital_loss', 'education_num_bin', 'age_bin',
       'hours_per_week_bin', 'sex_cat', 'work_class_cat',
       'relationship_cat', 'native_country_cat', 'occupation_cat',
       'race_cat', 'marital_status_cat']
"""       


# In[ ]:


"""
importance = forestGS.feature_importances_
feature_indexes_by_importance=importance.argsort()
# Print each feature label, from most importance to least important(reverse order)
for index in feature_indexes_by_importance:
   print("{} - {:.2f}".format(feature_labels[index], (importance[index]*100.0)))
"""


# ### Saving model

# In[ ]:


"""
joblib.dump(forestGS, 'models//trained_adult_income_classifier.pkl') # Save the trained model to a file for external usage.
"""


# ### Sample prediction

# - Making a fake data point

# In[ ]:


"""
sample_data = [
    3025, #capital_gain
    0, #capital_loss
    10, #education_num_bin
    2, # age_bin
    3, # hours_per_week_bin
    1, # sex_cat
    3, # work_class_cat
    1, # relationship_cat
    1, # native_country_cat
    3, # occupation_cat
    0, # race_cat
    0 # marital_status_cat
]

sample_data = [sample_data] # Need 2d array to work (scikit assumes that we are predicting for multiple data).
"""


# - Loading the trained model

# In[ ]:


"""
model = joblib.load('models//trained_adult_income_classifier.pkl') # Load model
"""


# - Predict

# In[ ]:


"""
results = model.predict(sample_data)
"""


# - Result

# In[ ]:


"""
prediction = results[0]
print(prediction)
"""


# <div style="color: green;">
#     1 -> income is above 50k.
#     </div>

# # 6) <u>References</u>
# 
# - https://archive.ics.uci.edu/ml/datasets/Adult
# - https://www.kaggle.com/uciml/adult-census-income/discussion/32698
# 
# - https://www.kaggle.com/prashant111/eda-logistic-regression-pca
# - https://www.kaggle.com/startupsci/titanic-data-science-solutions
# - https://www.kaggle.com/sumitm004/eda-and-income-predictions-86-75-accuracy
# - https://www.kaggle.com/ipbyrne/income-prediction-84-369-accuracy
# - https://www.kaggle.com/kost13/us-income-logistic-regression
# - https://www.kaggle.com/marksman/us-adult-income-salary-prediction
# 
# - https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
# - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
# - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
# 
# - https://stats.stackexchange.com/questions/244507/what-algorithms-need-feature-scaling-beside-from-svm
# - https://towardsdatascience.com/visualizing-your-exploratory-data-analysis-d2d6c2e3b30e
# - https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6
# - https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
