#!/usr/bin/env python
# coding: utf-8

# # Food choices and preferences of college students
# 
# This dataset includes information on food choices, nutrition, preferences, childhood favorites, and other information from college students. There are 125 responses from students. Data is raw and uncleaned. Data comes from [kaggle](https://www.kaggle.com/borapajo/food-choices/data).<br>
# Dataset was created to take a look at student's nutrition habits:<br>
#     - What does influence on students cooking frequency?
#     - Are students who are active in sports put more attention to nutritional check than the others?
#     - Are kids and parents have similar cooking habits?
#     - Is there any correlation between types of world cuisines that students like or do not like? 
# In this notebook we will visualize some of these features and try find any connections.<br>
# <br>
# **The main goals:**<br>
#     - Data exploration/some cleaning.
#     - Data analysis.
#     - Visualization.
# 
# <br>
# *Dataset can be used in natural language processing.

# # Imported libraries

# In[ ]:


# Data engineering.
import pandas as pd
import numpy as np

# Regular expressions module.
import re

# Data visualization and frame's visualization options.
import missingno as msno # Copyright (c) 2016 Aleksey Bilogur
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
sns.set_style("whitegrid")


# In[ ]:


# Load data
df = pd.read_csv('../input/food_coded.csv')


# # Data exploration

# In[ ]:


# Check data frame's shape
df.shape


# In[ ]:


df.head() 


# In[ ]:


# Data features
df.columns


# In[ ]:


# Explore the dataset
df.describe(include='all').T


# ## Features dtypes
# 
# 
# In the table above we have some features which need to get converted. Two columns - unique and top, gives us some help. Values of these columns, which are not missing can be interpreted as values of object features. Very high percentage of their values is unique what is reasonable considering the method it was gathered (questionnaire). Most of them have already prepared numeric features with some sort of weighting. However two of them looks suspicious to me. 'GPA' and 'weight' features have many unique values and the highest values of 'top' column are numbers. It looks like their dtype might be wrong and they can contain **blended values**. Other features have float types with '1-2' type values or some sort of range of numbers. For the second kind we need convert ranges into roughly the same scale.

# In[ ]:


from collections import Counter
# Explore datatypes
col_dtypes = np.array([df[x].dtype for x in df.columns])

# Quick features dtype counter
for i, j in Counter(col_dtypes).items():
    print('dtype: ', i, ' --- ', 'value: ', j)


# As we mentioned above, we will explore object features in search of blended values. It's necessary to fix it for further correlation investigation and results visualization. Now let's create a series with object dtype features.

# In[ ]:


# Quick look at features list with object datatype
df_obj = pd.DataFrame({'dtype_': col_dtypes}, index=df.columns)


# In[ ]:


# Slice the dtype
df_obj = df_obj[df_obj['dtype_'] == object]
df_obj


# Create a data frame with more details about possible blended values or missing data.

# In[ ]:


types = {}

for feature in df_obj.index.values:
    feat_dict = {}
    
    for value in df[feature].values:
        # Take out dtype from a string with regex
        dtype = str(type(value))
        match = re.search("int|float|str", dtype)

        # Create a dict with number of dtypes for particular feature
        if match.group() not in feat_dict.keys():
            feat_dict[match.group()] = 1
        else:
            feat_dict[match.group()] += 1
    types[feature] = feat_dict
    # Clean up the dict before next iteration
    feat_dict = {}


# In[ ]:


# Create transposed data frame with dtypes counter for each object feature
df_type = pd.DataFrame.from_dict(types).T
# Fill missing data with zeros
df_type.fillna(value=0)


# We can assume that float dtype represents missing values. 'GPA' and 'weight' features have mostly string values with a few NaN's. It is a good news for us, because it will be easier to isolate the digits with regex. We will handle this problem in other subsection.

# ## Missing data
# 
# 

# First thing we can look at is count column. We can notice that there is some range of values. It means we have to deal with NaN values. Let's explore a data frame and visualize the distribution of missing values of inputs. We are going to use missingno module created by [Aleksey Bilogur](https://github.com/ResidentMario/missingno).

# In[ ]:


# Check how many features have missing data
df.isnull().any().value_counts()


# We are going to create more detailed data frame with informations about missing values.

# In[ ]:


# Amount of NaN values for each feature
total = df.isnull().sum().sort_values(ascending=False)
# Percentage part of total
percent = (df.isnull().sum()/df.isnull().count()*100).round(1).sort_values(ascending=False)
# Merge series
nan_data = pd.concat({"# of NaN's": total, '% of Total': percent}, axis=1)
nan_data.head(10)


# Let's plot missing values distribution. We are going to use these 34 True values above as indices in dataframe.columns .<br>
# **Note:** Author of this module recommends to use max. 50 featuers at once in visualization.

# In[ ]:


# Use missingno module for NaN's distribution
msno.matrix(df[df.columns[df.isnull().any()]])


# Distribution of missing values across the dataset is pretty good with one exception. Feature called 'comfort_food_reasons_coded' has a significant gap at the end. I think it might be dangerous. Let's explore the importance of this feature.<br>
# One more thing about this plot - on the right side we have specified rows with min. and max. amount of features that does not contain NaN's.<br>
# <br>
# When we look back at data frame we can notice a few related with each other features - **'comfort_food_reasons', 'comfort_food_reasons_coded', 'comfort_food_reasons_coded.1'.**<br>
# It looks like comfort_food_reasons feature was converted to numerical values before. The third feature seems to be a copy of that we mentioned above. It does not appear on graph with missing values. Let's check if values of both do overlap.

# In[ ]:


(df['comfort_food_reasons_coded'] == df['comfort_food_reasons_coded.1']).value_counts()


# We can quickly check if the last 20 values are reasonable to the rest. Reasons like stress, boredom and sadness had 1-3 values. If there is  more unique reason then value goes higher. It looks ok for me.

# In[ ]:


df[['comfort_food_reasons', 'comfort_food_reasons_coded.1']].tail(20)


# That is confirmation of our assumption. We can drop both features -  **'comfort_food_reasons'** and **'comfort_food_reasons_coded'**.<br> The one we leave has no missing values. Let's rename it.

# In[ ]:


df.drop(['comfort_food_reasons', 'comfort_food_reasons_coded'], axis=1, inplace=True)
df.rename(columns={'comfort_food_reasons_coded.1': 'comfort_food_reasons'}, inplace=True)


# ## Correcting erroneous values
# 
# For some columns, there are some values that do not match obviously with others. This can be some random words in cells where there should be only digits. In previous subsections we assumed that two features can contain such erroneous values. We will investigate it in this subsection.

# ###  'GPA' feature
# 
# Let's try to replace non-numeric values to NaN and fix blended data.

# In[ ]:


df['GPA'].unique()


# As we can see above, there is blended value and some cells without digits. Here we got 2 float values which are NaN's what we could see in features dtype table in one of previous subsections. We will fix that and replace incorrect and missing values with the most common feature's value. It should be fine since the values distribution does not emphasize one of the unique values in significant way.

# In[ ]:


# Take the most common value to fill. Assume that it will not oversimplify model (<5% of columns data to replace).
df['GPA'].value_counts().head()


# In[ ]:


# Use regex to clean blended data, fill missing values and set up dtype
df['GPA'] = df['GPA'].str.replace(r'[^\d\.\d+]', '').replace((np.nan, ''), '3.5').astype(float).round(2)


# In[ ]:


# Values after changes
df['GPA'].unique()


# In[ ]:


# Boxplot to visualize and check results
fig, ax = plt.subplots(figsize=[12,6])
sns.boxplot(df['GPA'])
ax.set_title("'GPA' distribution")


# ### 'weight' feature
# 
# We encounter problem of the same type as above in this case. We will fix incorrect cells and fill missing values dependent on gender.

# In[ ]:


# Some strings, blended and missing values
df['weight'].unique()


# In[ ]:


# Clean blended values, non-numeric become NaN
df['weight'] = df['weight'].str.replace(r'[^\d\d\d]', '').replace('', np.nan).astype(float)


# In[ ]:


df['weight'].unique()


# Let's make weight's distribution for each gender. To do this we can use seaborn's module. We have to drop missing values manually, because distplot function does not offer built-in parameter. Pandas dropna function returns a copy so it will not affect on our dataset.

# In[ ]:


# Dict contains mean values of weights for both genders
weight_mean = {}
# Dict contains std values of weights for both genders
weight_std = {}

# Set plot size
fig, ax = plt.subplots(figsize=[16,4])

# Create two distributions for both genders
for gen, frame in df[['Gender', 'weight']].dropna().groupby('Gender'):
    weight_mean[gen] = frame['weight'].values.mean()
    weight_std[gen] = frame['weight'].values.std()
    sex_dict = {1: 0, 2: 1}
    sns.distplot(frame['weight'], ax=ax, label=['Female', 'Male'][sex_dict[gen]])

ax.set_title('weight distribution, hue=Gender')
ax.legend()


# Different distributions between sets of values associated with Female/Male are good reason to consider while filling missing values.<br>
# <br>
# A deeper view on missing values of this feature.

# In[ ]:


# Let's check rows with NaN weight values
df[df['weight'].isnull()]


# In[ ]:


# Let's check how many NaN's these 3 rows have
inv_data = {} # key - observation index; values - (actual value, # of row's NaNs, gender)

for index, row in df[df['weight'].isnull()].iterrows():
    # Variable for printing results and getting # of NaNs
    temp = row.isnull().value_counts()
    print('Index: ', temp.name, " --- # of NaN's: ", temp.values[1])
    
    # Adds to dict 3 values by data frame's index - (actual value, # of row's NaNs, gender)
    inv_data[str(index)] = (row['weight'], temp.values[1], row['Gender'])        


# We've created a dict with set of informations about particular rows. One of the observations contains 11 NaN's. Reffering to missing values distribution matrix, this observation has the largest amount of missing data. I was considering dropping that row, but most of absent values have premade numerical weights. The other thing is that we have small number of observations and we should not decrease that if it is not necessary.<br>
# <br>
# Missing values will be filled with random number based on mean value in regards to the standard deviation. I will leave a condition for dropping a row.

# In[ ]:


# Condition for dropping a row
drop_cond = df.shape[1]/2

for df_index, tuple_ in inv_data.items():
    # Row with # of NaNs > 'drop_cond' will be dropped.
    if tuple_[1] > drop_cond:
        df.drop(int(df_index), inplace=True)
    # Weight's NaN will be replaced with random number based on mean value 
    # in regards to the standard deviation.
    else:
        # Mean value of weights set in regards to gender
        mean_val = weight_mean[tuple_[2]]
        # The standard deviation value of weights set in regards to gender
        std_val = weight_std[tuple_[2]]
                
        # Random value creator in range defined by mean and the standard deviation 
        # value in regards to gender
        rand_val = np.random.randint(mean_val - std_val, mean_val + std_val)
        
        # Replacing NaN's with prepared value
        df['weight'].values[int(df_index)] = rand_val


# In[ ]:


# Finally we can change column dtype.
df['weight'] = df['weight'].astype(int)


# # Data analysis

# ## Binary features
# 
# Plot represents distribution of dichotomous variables. In this dataset *1* is binary *0* (blue colour) and *2* is binary *1* (red colour). We can notice that some of features are imbalanced. If we decide to feed them into classifier without any preprocessing, it might result in worse predictions. In this scenario, classifiers are more sensitive to detecting the majority class and less sensitive to the minority class. In many cases model will be predicting the majority class of target input.

# In[ ]:


# Slice binary features and their values amount without NaNs
x = df.describe().T
y = pd.Series(x[x['max'] == 2]['count'], index=x[x['max'] == 2].index)

# Percent values of 0/1 for each feature
zero_list = []
one_list = []

# Convert into percentages
for ind, col in y.iteritems():
    zero_list.append(((df[ind]==1).sum()*100)/col)
    one_list.append(((df[ind]==2).sum()*100)/col)


# In[ ]:


# Plot preparation
plt.figure()
fig, ax = plt.subplots(figsize=(6,6))

# Create barplots
sns.barplot(ax=ax, x=x[x['max'] == 2].index, y=zero_list, color="blue")
sns.barplot(ax=ax, x=x[x['max'] == 2].index, y=one_list, bottom= zero_list, color="red")

# Plot labels
plt.ylabel('Percent of zero/one [%]', fontsize=16)
plt.xlabel('Binary features', fontsize=16)

# Plot's font settings
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.tick_params(axis='both', which='major', labelsize=16)


# ## Categorical features
# 
# In this subsection we will explore habits represented by distribution on categorical data. To do this we will use barplots. For data preparation we will define a function called cat_bin(). It returns number of 0/1 values converted into percentages.

# In[ ]:


def cat_bin(data, x_var, hue_var, hue_class, x_names=[], hue_names=[]):
    # Prepare dichotomous and/or ordinal variable/s to graphic representation
    
    # Create data frame
    df = pd.DataFrame(index=x_names)
    
    # Prepare converted values for each hue
    for i, j in [(name, ind+1) for ind, name in enumerate(hue_names)]:
        df[i] = data[x_var][data[hue_var] == j].value_counts().sort_index().values
        df[i] = ((df[i]/df[i].sum())*100).round(1)
        
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'categories'}, inplace=True)
    
    return pd.melt(df, id_vars="categories", var_name=hue_class, value_name="percent")


# In[ ]:


# New category names
objects = ['daily', '3 times/w', 'rarely','holidays', 'never']
# Data preparation for plotting
df_cook_sex = cat_bin(df, 'cook', 'Gender', 'sex', objects, hue_names=['Female', 'Male'])

with sns.plotting_context(context='notebook', font_scale=1.5):
    # Seaborn's barplot creator
    sns.factorplot(x='categories', y='percent', hue='sex', data=df_cook_sex, 
                   kind='bar', palette="muted", size=8)
    # Plot labels
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Percent of Female/Male [%]', fontsize=18)
    plt.title('Cooking frequency', y=1.01, size=25)


# Distribution differences between both genders are significant. We can see that almost half of females cooks rarely. Cooking tendention is more frequent on the all levels in compare to males. 'Holidays' and 'never' categories are rather opposite to cooking in the academic year where males have fully dominance over females.<br>
# <br>
# Let's try to explore more features related to cooking frequency.

# In[ ]:


# New category names
objects = ['very low', 'low', 'average','above-aver.', 'high', 'very high']
# Data preparation for plotting
df_inc = cat_bin(df, 'income', 'Gender', 'sex', objects, hue_names=['Female', 'Male'])

with sns.plotting_context(context='notebook', font_scale=1.5):
    # Seaborn's barplot creator
    sns.factorplot(x='categories', y='percent', hue='sex', data=df_inc, 
                   kind='bar', palette='inferno', size=8)
    # Plot labels
    plt.xlabel('Income', fontsize=18)
    plt.ylabel('Percent of Female/Male [%]', fontsize=18)
    plt.title('Income distribution', y=1.01, size=25) 


# Income distribution can in some way explain why males tends to cook less. More than a half people defined their family income as high or very high.

# In[ ]:


# New category names
objects = ['daily', '3 times/w', 'rarely','holidays', 'never']
# Create data frame
df_camp = pd.DataFrame(index=objects)

# Add new columns in df_camp with data
df_camp['on_campus'] = df['cook'][df['on_off_campus'] == 1].value_counts().sort_index().values
# Values != 1 are different types of accommodation outside the campus
df_camp['off_campus'] = df['cook'][df['on_off_campus'] > 1].value_counts().sort_index().values

# Prepare converted values for each hue
df_camp['on_campus'] = ((df_camp['on_campus']/df_camp['on_campus'].sum())*100).round(1)
df_camp['off_campus'] = ((df_camp['off_campus']/df_camp['off_campus'].sum())*100).round(1)

df_camp.reset_index(inplace=True)
df_camp.rename(columns={'index': 'categories'}, inplace=True)

# Reshaping data frame
df_camp = pd.melt(df_camp, id_vars="categories", var_name='on/off campus', value_name="percent")  


# In[ ]:


with sns.plotting_context(context='notebook', font_scale=1.5):
    # Seaborn's barplot creator
    sns.factorplot(x='categories', y='percent', hue='on/off campus', data=df_camp, 
                   kind='bar', palette="dark", size=8)
    # Plot labels
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Percent of on/off campus [%]', fontsize=18)
    plt.title('Cooking frequency', y=1.01, size=25)  


# The next feature we can look at is on_off_campus. It shows us distribution of cooking frequency of people living on or off campus. Clearly people who lives off the campus tends to cook more often.<br>
# <br>
# Let's check correlation between features 'parents_cook' and 'cook'. Both are categorical and the first one try to measure influence of parents habits when we talk about cooking. Features are correlated with each other. We used Spearman's rank correlation method.

# In[ ]:


df_parent = df[['parents_cook', 'cook']]
df_parent.corr(method = 'spearman')


# In this part of subsection we will check some other features related with food.

# In[ ]:


# New category names
frequency = ['never', 'rare', 'sometimes', 'often', 'always']
# Data preparation for plotting
df_sport = cat_bin(df, 'nutritional_check', 'sports', 'activity', 
                   frequency, hue_names=['no_sport', 'sport'])

with sns.plotting_context(context='notebook', font_scale=1.5):
    # Seaborn's barplot creator
    sns.factorplot(x='percent', y="categories", hue="activity", data=df_sport, 
                   kind='bar', palette="Greens_d", size=8)
    # Plot labels
    plt.xlabel('Percent dependent of activity [%]', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title('Nutritional check frequency', y=1.01, size=25)


# In[ ]:


sport_corr = df[['nutritional_check', 'sports']]
sport_corr.corr(method = 'kendall')


# The plot above tries to explore, how nutritional check frequency is connected with sport activity. Perhaps people who has some sort of sport activity, put more attention on more frequent nutritional check. In this case we also checked correlation between dichotomous and ordinal variable. There is no real correlation.

# In[ ]:


# New category names
food = ['greek_food', 'indian_food', 'italian_food', 'thai_food', 'persian_food', 'ethnic_food']
# Set diverging palette
cmap = sns.diverging_palette(50, 10, as_cmap=True)
# Plot preparation
plt.figure(figsize=(12,12))
plt.title('Correlation between food preferences', y=1.01, size=20)

with sns.plotting_context(context='notebook', font_scale=1.5):
    # Seaborn's heatmap creator
    g = sns.heatmap(df[food].corr(method='spearman'),linewidths=0.5,vmax=1.0, square=True, center=0,
                    cmap=cmap, annot=True, cbar_kws={"shrink": .75, "orientation": "horizontal"})
    # Plot labels
    loc, labels = plt.xticks()
    g.set_xticklabels(labels, rotation=45)
    g.set_yticklabels(labels, rotation=45)


# Heatmap above shows us correlation between different types of world cuisines. Surveyed people weighted their preferences in 1-5 point scale. The correlation between these variables is very strong. We can conclude that people have similar taste when it comes to types of world cuisines. The only one feature that is not dominant is 'italian_food' which seems to be the best choice if we want apply more than one of these to our prediction model.

# ## Categorical and continuous features
# 
# Plotting experiment of 2 kinds of features. I've tried to implement features that seems like measurement of veggies/fruit days in their nutrition and check possible impact of these on weight. I was wondering if it is justifed to use ANOVA model here.

# In[ ]:


# Slice of data frame
df_weight = df[['veggies_day', 'fruit_day', 'weight']]

# Data preparation for plotting
df_weight = pd.melt(df_weight, id_vars='weight', var_name='day', value_name="frequency")

# Seaborn's barplot creator
g = sns.factorplot(x="frequency", y="weight", col="day", data=df_weight, 
                   kind='box', palette="YlGnBu_d", size=8, aspect=.75)
g.despine(left=True)
sns.factorplot(x="frequency", y="weight", col="day", data=df_weight, 
               kind='swarm', palette="YlGnBu_d", size=8, aspect=.75)
g


# # Quick summary

# Raw data made us a lot of effort in exploring dataset and taking a deep view of a few features. Fixing a blended values had to be done before visualization plots could happen. That was one of the goals we focused on. The whole preprocessing part of the data is not finished. Categorical features need to be encoded before any implementation. For this we can use one hot encoder. Amount of features force us to choose a set of variables for a classifier. Splitting up whole set of categorical features would end up with huge amount of columns. Please remember that initial dataset has approx. 2:1 ratio of observations/features. Another thing worth mentioning is grouping continuous values into categories if it is possible and handling the rest missing values. None of the classification models were created, because that was not the goal of this kernel. Small number of observations was enough to play and find some connections between features.<br>
# <br>
# Variety of features gives a lot of space for data exploring finding different correlations. I am open to any suggestions as well as comments pointing out substantive errors.

# # References
# 
# [1] Porto Seguro Exploratory Analysis and Prediction, Kaggle Kernel, [link](https://www.kaggle.com/gpreda/porto-seguro-exploratory-analysis-and-prediction) <br>
# [2] End to End Project with Python, Kaggle Kernel, [link](https://www.kaggle.com/niklasdonges/end-to-end-project-with-python) <br>
# [3] Some basic stats: t-test, spearman's r, Kaggle Kernel, [link](https://www.kaggle.com/borapajo/some-basic-stats-t-test-spearman-s-r)
