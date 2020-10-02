#!/usr/bin/env python
# coding: utf-8

# # <center> Analyzing the "Horses for Courses" Horse Racing Dataset from Kaggle </center>

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, ttest_ind, zscore

get_ipython().run_line_magic('matplotlib', 'inline')

#Supresses scientific notation
pd.set_option('display.float_format', lambda x: '%.2f' % x)

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#There are a lot of issues with the 'position_two' column, so I left it out.
#Furthermore, the 'position_again' column is much more consistent and has all relevant win/place information  

fields = ["position_again","bf_odds","venue_name","date","market_name","condition",
          "barrier","handicap_weight","last_twenty_starts","prize_money","sex",
          "age","jockey_sex","days_since_last_run","overall_starts","overall_wins",
          "overall_places","track_starts","track_wins","track_places","firm_starts",
          "firm_wins","firm_places","good_starts","good_wins","good_places",
          "slow_starts","slow_wins","slow_places","soft_starts","soft_wins",
          "soft_places","heavy_starts","heavy_wins","heavy_places","distance_starts",
          "distance_wins","distance_places"]

df = pd.read_csv('../input/horses 2.csv', usecols=fields, skipinitialspace=True, low_memory=False)

df.head()


# ## I decided not to use any man-made metrics (e.g. odds, field strength, etc.) because these are relative and subject to change.
# 
# ## Also, jockey and trainer win percentages are not included with this dataset.
# 
# ## Fixing the format of some features:

# In[ ]:


df.date = pd.to_datetime(df.date, format='%Y'+'-'+'%m'+'-'+'%d')

#removes numbers from end of 'condition' strings
df.condition = df.condition.str.replace('\d+', '')

#renaming condition values so that they're uniform
df.condition = df.condition.replace(['HVY','AWT'], ['HEAVY','GOOD']) 
#AWT equates to a Good surface under some weather conditions

#reverses 'last_five_starts' (originally written right-to-left) 
#so that it's easier to read in the future
df.last_twenty_starts = df.last_twenty_starts.str[::-1]


# ## Useful Cleaning Functions:

# In[ ]:


def column_cleaner(cleaned_df, grouped_df, column_name):
    non_null_indices = grouped_df[column_name].apply(lambda x: all(x.notnull()))
    
    non_null_df = cleaned_df[non_null_indices]
    
    non_null_grouped = non_null_df.groupby(['date','venue_name','market_name'])
    
    clean_indices = non_null_grouped[column_name].value_counts(normalize=True,dropna=False).        where(lambda x:x != 1).dropna().index.droplevel(column_name)
    
    new_cleaned_df = non_null_df.loc[clean_indices].drop_duplicates()
    return new_cleaned_df

def cleaned_win_df(cleaned_df):
    win_indices = cleaned_df.position_again.apply(lambda x:x == 1)
    
    df_cleaned_win = cleaned_df[win_indices]
    return df_cleaned_win


# # Creating new features and dropping others in order to relate horses in each race to one another while allowing the general input of the dataset into a machine learning model:

# ## Creating a distance column from market_name:

# In[ ]:


new = df.market_name.str.split(expand=True)

df['distance'] = new[1].str.rstrip('m')

df.distance = df.distance.astype(np.int64)

df.distance.head()


# ## Creating general and track, distance, condition-specific 'win_percent' and 'place_percent' columns:

# In[ ]:


#creates overall, track, and distance win_percent and place_percent columns
#and drops existing wins and places columns

columns_list = ["overall","track","distance"]

for x in columns_list:
    df[x+"_win_percent"] = df[x+"_wins"]/df[x+"_starts"]
    
    df[x+"_place_percent"] = df[x+"_places"]/df[x+"_starts"]

    # dropping various columns, though 'starts' columns will be used later
    df.drop([x+'_wins', x+'_places'], axis=1, inplace=True)


# In[ ]:


#creates a condition_starts ,condition_win_percent, and condition_place_percent column
#for each horse according to the condition of the track for that race

df.loc[df.condition.isna(), "condition_win_percent"] = np.nan

condition_list = ["firm","good","slow","soft","heavy"]

for x in condition_list: 
    df.loc[df.condition.str.lower() == x, "condition_starts"] = df[x+"_starts"]
    
    df.loc[df.condition.str.lower() == x, "condition_win_percent"] = df[x+"_wins"]/df[x+"_starts"]
    
    df.loc[df.condition.str.lower() == x, "condition_place_percent"] = df[x+"_places"]/df[x+"_starts"]
    
    df.drop([x+'_starts', x+'_wins', x+'_places'], axis=1, inplace=True)

# Replaces infinity (zero division) with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)


# ## Find and drop features that are primarily NaN:

# In[ ]:


df.isnull().sum()
#The position_again is primarily nan values because it only shows first and place
#However, track_starts is primarily zeros, so the track_win/place_percent columns are nan


# In[ ]:


df.drop(['track_win_percent','track_place_percent'],axis=1,inplace=True)


# In[ ]:


#position_again unique values
df.position_again.unique()


# ## Splits last_twenty_starts column into 20 separate columns, replaces values, then drops last_twenty_starts:

# In[ ]:


new = pd.DataFrame()

for i in range(20):
    new[i] = df.last_twenty_starts.str[i:i+1]

for i in range(20):
    df['last_start'+str(i+1)] = new[i].replace(['0','','x','f'],['ten+','none','scratch','fell'])    

df.drop('last_twenty_starts',axis=1,inplace=True)


# ## Cleaning data by removing races with missing win and/or place values in 'position_again' column:

# In[ ]:


#Used groupby to create indices by which to sort the re-indexed dataframes below, like df_indexed and df_cleaned
df_grouped = df.groupby(['date','venue_name','market_name'])

#Drops all groups/races in 'position_again' column where sum of values [1st, 2nd, 3rd] don't add to 3 or 6
#i.e. 1+2 and 1+2+3
index_list1 = df_grouped.position_again.sum(dropna=False).where(lambda x:(x == 3) | (x == 6)).dropna().index

df_indexed = df.set_index(['date','venue_name','market_name'])

df_cleaned = df_indexed.loc[index_list1].drop_duplicates()

df_grouped = df_cleaned.groupby(['date','venue_name','market_name'])

#Eliminates remaining errors in 'position_again' column by making sure that there isn't a single 3rd-place finish
index_list2 = df_grouped.position_again.value_counts(normalize=True,dropna=False)    .where(lambda x:x != 1).dropna().index.droplevel('position_again')

df_cleaned = df_cleaned.loc[index_list2].drop_duplicates()

df_grouped = df_cleaned.groupby(['date','venue_name','market_name'])


# # Normalizing each group (race) using z-scores is a good and straightforward way to compare horses across races.
# 
# ## Here, I am creating several normalized columns in this way.

# ## Creating a weight_z column:

# In[ ]:


df_cleaned['weight_z'] = df_grouped['handicap_weight'].transform(lambda x: zscore(x,ddof=1))

df_cleaned.drop('handicap_weight',axis=1,inplace=True)

df_grouped = df_cleaned.groupby(['date','venue_name','market_name'])


# ## Creating a prize_money_per_start_z column:
# 
# ### This may be one of the best indicators, as prize money is also an indicator of the difficulty of past races. Therefore, the value (meaningfulness) of past wins is taken into consideration.

# In[ ]:


#creates prize_money_per_start column
df_cleaned['prize_money_per_start'] = df_cleaned.prize_money/df_cleaned.overall_starts

df_cleaned['prize_money_per_start_z'] = df_grouped['prize_money_per_start']    .transform(lambda x: zscore(x,ddof=1))

df_cleaned.drop(['prize_money','prize_money_per_start'],axis=1,inplace=True)

df_grouped = df_cleaned.groupby(['date','venue_name','market_name'])


# ## Creating a horse age_z column:

# In[ ]:


df_cleaned['age_z'] = df_grouped['age'].transform(lambda x: zscore(x,ddof=1))

df_cleaned.drop('age',axis=1,inplace=True)

df_grouped = df_cleaned.groupby(['date','venue_name','market_name'])


# ## Creating race-relative z-scores for the remaining continuous features:

# In[ ]:


z_score_cols = ['days_since_last_run','overall_win_percent','overall_place_percent',
                'distance_win_percent','distance_place_percent','condition_win_percent',
                'condition_place_percent','overall_starts','distance_starts','condition_starts',
                'track_starts']

for col in z_score_cols:
    df_cleaned[col+'_z'] = df_grouped[col].transform(lambda x: zscore(x,ddof=1))


# ## I decided to keep the original "overall_starts," "distance_starts," "condition_starts," and "track_starts" columns because they may have a meaning irrespective of other races (unnormalized).

# In[ ]:


#Replaces infinity (zero division) with NaN
df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

df_grouped = df_cleaned.groupby(['date','venue_name','market_name'])

df_cleaned.head()


# ## Cleaned Dataframe Details:

# In[ ]:


df_cleaned.shape


# In[ ]:


len(df_grouped) #Number of remaining races


# # Testing and graphing the significance of certain features:

# ## For horse gender:

# In[ ]:


#Removes races where only one horse gender is represented
sex_pop_cleaned = column_cleaner(df_cleaned, df_grouped, 'sex')

sex_pop_cleaned_win = cleaned_win_df(sex_pop_cleaned)


# In[ ]:


#General percentage of horse genders for races where multiple genders are represented  
sex_pop_cleaned.sex.value_counts(dropna=False,normalize=True).sort_values(ascending=False)    .drop('Unknown')


# In[ ]:


sex_pop_cleaned_win.sex.value_counts(dropna=False,normalize=True).sort_values(ascending=False)    .drop('Unknown')


# In[ ]:


horse_sex_pop = sex_pop_cleaned.sex.value_counts(dropna=False,normalize=True)    .sort_values(ascending=False).drop('Unknown')

horse_sex_win = sex_pop_cleaned_win.sex.value_counts(dropna=False,normalize=True)    .sort_values(ascending=False).drop('Unknown')

horse_sex_percent_difference = (horse_sex_win - horse_sex_pop)/horse_sex_pop

horse_sex_percent_difference


# In[ ]:


index1 = ['Gelding', 'Mare', 'Filly','Colt', 'Horse']

df1 = pd.DataFrame({'Total Proportion': horse_sex_pop,'Win Proportion': horse_sex_win ,
                    'Percent Difference': horse_sex_percent_difference}, index=index1)

ax = df1.plot.bar(rot=0,title='The Significance of Horse Gender')


# ### Using the Pearson's chi-squared, I find horse gender is significant:

# In[ ]:


observed1 = sex_pop_cleaned_win.sex.value_counts().sort_values(ascending=False)    .drop('Unknown').values

expected_percentages1 = horse_sex_pop.values
expected1 = [x*observed1.sum() for x in expected_percentages1]

test_stat1, p_value1 = chisquare(observed1, expected1)

test_stat1, p_value1


# ## For horse age_z (z-scores):

# In[ ]:


#Removing races where there is only one age
age_pop_cleaned = column_cleaner(df_cleaned, df_grouped, 'age_z')

age_pop_cleaned_win = cleaned_win_df(age_pop_cleaned)


# In[ ]:


age_pop_cleaned.age_z.describe()


# In[ ]:


age_pop_cleaned_win.age_z.describe()


# In[ ]:


data2a = age_pop_cleaned.age_z.dropna().values
data2b = age_pop_cleaned_win.age_z.dropna().values

plt.title("Winner and Race Distributions of Age Z-scores", fontsize=15)

plt.hist(data2a, density=True, bins=24, range=(-3,3), label='Race Average', 
         color='b', alpha=.5, edgecolor='k')

plt.hist(data2b, density=True, bins=24, range=(-3,3), label='Winner Average',
         color='r', alpha=.5, edgecolor='k')

plt.legend(loc='upper right')
plt.xlabel('Age Z-scores')
plt.ylabel('Probability');


# ### Using a 2-sample T-test, I find that Age Z-scores is significant:

# In[ ]:


test_stat2, p_value2 = ttest_ind(data2a, data2b)

test_stat2, p_value2


# ### Does condition affect the win distribution of age? Specifically, do older horses perform worse in bad conditions?

# In[ ]:


condit_age_pop = age_pop_cleaned[age_pop_cleaned.condition == 'HEAVY']


# In[ ]:


condit_age_pop_win = cleaned_win_df(condit_age_pop)


# In[ ]:


data2c = condit_age_pop.age_z.dropna().values
data2d = condit_age_pop_win.age_z.dropna().values

plt.title("Winner and Race Distributions of Age Z-scores \n (Condition Specific)", fontsize=15)

plt.hist(data2c, density=True, bins=24, range=(-3,3), label='Race Average', 
         color='b', alpha=.5, edgecolor='k')

plt.hist(data2d, density=True, bins=24, range=(-3,3), label='Winner Average',
         color='r', alpha=.5, edgecolor='k')

plt.legend(loc='upper right')
plt.xlabel('Age Z-scores')
plt.ylabel('Probability');


# ### Condition depended T-test for Age Z-scores: 

# In[ ]:


test_stat2, p_value2 = ttest_ind(data2c, data2d)

test_stat2, p_value2


# ### It appears as though bad track conditions actually level out the age discrepancies, maybe because they have more experience with those bad conditions.

# ## For horse handicap weight_z (z-scores):

# In[ ]:


#Removing races where there is only one age
weight_pop_cleaned = column_cleaner(df_cleaned, df_grouped, 'weight_z')

weight_pop_cleaned_win = cleaned_win_df(weight_pop_cleaned)


# In[ ]:


weight_pop_cleaned.weight_z.describe()


# In[ ]:


weight_pop_cleaned_win.weight_z.describe()


# In[ ]:


data3a = weight_pop_cleaned.weight_z.dropna().values
data3b = weight_pop_cleaned_win.weight_z.dropna().values

plt.title("Winner and Race Distributions of Weight Z-scores", fontsize=15)

plt.hist(data3a, density=True, bins=24, range=(-3,3), label='Race Average', 
         color='b', alpha=.5, edgecolor='k')

plt.hist(data3b, density=True, bins=24, range=(-3,3), label='Winner Average',
         color='r', alpha=.5, edgecolor='k')

plt.legend(loc='upper right')
plt.xlabel('Weight Z-scores')
plt.ylabel('Probability');


# ### Using a 2-sample T-test, again I find that Weight Z-scores is significant:

# In[ ]:


test_stat3, p_value3 = ttest_ind(data3a, data3b)

test_stat3, p_value3


# ## For prize money, using prize_money_per_start_z (z-scores):

# In[ ]:


money_pop_cleaned = column_cleaner(df_cleaned, df_grouped, 'prize_money_per_start_z')

money_pop_cleaned_win = cleaned_win_df(weight_pop_cleaned)


# In[ ]:


money_pop_cleaned.prize_money_per_start_z.describe()


# In[ ]:


#Winner prize money 
money_pop_cleaned_win.prize_money_per_start_z.describe()


# In[ ]:


data4a = money_pop_cleaned.prize_money_per_start_z.dropna().values
data4b = money_pop_cleaned_win.prize_money_per_start_z.dropna().values

plt.title("Winner and Race Distributions of Prize Money per Start Z-scores",
          fontsize=15)

plt.hist(data4a, density=True, bins=24, range=(-3,3), label='Race Average',
         color='b', alpha=.6, edgecolor='k')

plt.hist(data4b, density=True, bins=24, range=(-3,3), label='Winner Average',
         color='r', alpha=.5, edgecolor='k')

plt.legend(loc='upper right')
plt.xlabel('Prize Money per Start Z-scores')
plt.ylabel('Probability');


# ### Using a 2-sample T-test, I find that Prize Money per Start Z-scores is significant:
# 

# In[ ]:


test_stat4, p_value4 = ttest_ind(data4a, data4b)

test_stat4, p_value4


# ## For overall wins:

# In[ ]:


overall_win_pop_cleaned = column_cleaner(df_cleaned, df_grouped, 'overall_win_percent_z')

overall_win_pop_cleaned_win = cleaned_win_df(overall_win_pop_cleaned)


# In[ ]:


overall_win_pop_cleaned.overall_win_percent_z.describe()


# In[ ]:


overall_win_pop_cleaned_win.overall_win_percent_z.describe()


# In[ ]:


data5a = overall_win_pop_cleaned.overall_win_percent_z.dropna().values
data5b = overall_win_pop_cleaned_win.overall_win_percent_z.dropna().values

plt.title("Winner and Race Distributions of Overall Win Percent Z-scores",
          fontsize=15)

plt.hist(data5a, density=True, bins=24, range=(-3,3), label='Race Average',
         color='b', alpha=.6, edgecolor='k')

plt.hist(data5b, density=True, bins=24, range=(-3,3), label='Winner Average',
         color='r', alpha=.5, edgecolor='k')

plt.legend(loc='upper right')
plt.xlabel('Overall Win Percent Z-scores')
plt.ylabel('Probability');


# ### Using a 2-sample T-test, I find that Overall Win Percent Z-scores is significant:

# In[ ]:


test_stat5, p_value5 = ttest_ind(data5a, data5b)

test_stat5, p_value5


# ### There is high variance in the 100% column (aka beginner's luck). How many races before the 100% column is properly represented? That is, how many races is considered statistically significant?
# 
# ### It seems that a minimum of 5 races for all horses in the race gives the percent difference bar graph an exponential appearance.

# In[ ]:


overall_win_pop_grouped = overall_win_pop_cleaned.groupby(['date','venue_name',
                                                           'market_name'])

overall_starts_indices = overall_win_pop_grouped.overall_starts.agg('min')    .where(lambda x:x >= 5).dropna().index

overall_starts_cleaned = overall_win_pop_cleaned.loc[overall_starts_indices].drop_duplicates()

overall_starts_cleaned_win = cleaned_win_df(overall_starts_cleaned)


# In[ ]:


overall_starts_cleaned.overall_win_percent_z.describe()


# In[ ]:


overall_starts_cleaned_win.overall_win_percent_z.describe()


# In[ ]:


data6a = overall_starts_cleaned.overall_win_percent_z.dropna().values
data6b = overall_starts_cleaned_win.overall_win_percent_z.dropna().values

plt.title("Winner and Race Distributions of Overall Win Percent Z-scores \n (with horses over 5 total races)",
          fontsize=15)

plt.hist(data6a, density=True, bins=24, range=(-3,3), label='Race Average',
         color='b', alpha=.6, edgecolor='k')

plt.hist(data6b, density=True, bins=24, range=(-3,3), label='Winner Average',
         color='r', alpha=.5, edgecolor='k')

plt.legend(loc='upper right')
plt.xlabel('Overall Win Percent Z-scores \n (with horses over 5 total races)')
plt.ylabel('Probability');


# ### Using a 2-sample T-test, I find that Overall Win Percent Z-scores (with horses over 5 total races) is significant:

# In[ ]:


test_stat6, p_value6 = ttest_ind(data6a, data6b)

test_stat6, p_value6


# ## For Barrier:

# In[ ]:


barrier_pop_cleaned = column_cleaner(df_cleaned, df_grouped, 'barrier')

barrier_grouped = barrier_pop_cleaned.groupby(['date','venue_name','market_name'])

barrier_indices = barrier_grouped.barrier.value_counts().where(lambda x:x == 1)    .dropna().index.droplevel('barrier')

barrier_pop_cleaned = barrier_pop_cleaned.loc[barrier_indices]

barrier_pop_cleaned_win = cleaned_win_df(barrier_pop_cleaned)


# In[ ]:


barrier_pop = barrier_pop_cleaned.barrier.value_counts(normalize=True).sort_index()    .drop([18.00,19.00,20.00])

barrier_win = barrier_pop_cleaned_win.barrier.value_counts(normalize=True).sort_index()    .drop(18.00)

barrier_percent_difference = (barrier_win - barrier_pop)/barrier_pop

barrier_percent_difference


# In[ ]:


index7 = barrier_percent_difference.index

df7 = pd.DataFrame({'Total Proportion': barrier_pop,'Win Proportion': barrier_win,
                    'Percent Difference': barrier_percent_difference}, index=index7)

ax = df7.plot.bar(rot=0, title='The Significance of Barrier')


# ### Why is barrier 1 so overrepresented? Is there a problem with the data? There doesn't appear to be.
# 
# ### Using the Pearson's chi-squared test, I find that barrier is significant:

# In[ ]:


observed7 = barrier_pop_cleaned_win.barrier.value_counts().sort_index().drop(18.00).values
expected_percentages7 = barrier_pop.values
expected7 = [x*observed7.sum() for x in expected_percentages7]

test_stat7, p_value7 = chisquare(observed7, expected7)

test_stat7, p_value7


# ### Does the length of a race negate or alter the effect of starting barrier?

# In[ ]:


barr_dist_indices = barrier_pop_cleaned.distance.where(lambda x:x>=1800).dropna().index

barr_dist_cleaned = barrier_pop_cleaned.loc[barr_dist_indices]

barr_dist_cleaned_win = cleaned_win_df(barr_dist_cleaned)


# In[ ]:


barr_dist_pop = barr_dist_cleaned.barrier.value_counts(normalize=True).sort_index()    .drop(18.00)

barr_dist_win = barr_dist_cleaned_win.barrier.value_counts(normalize=True).sort_index()    .drop(18.00)

barr_dist_percent_difference = (barr_dist_win - barr_dist_pop)/barr_dist_pop

barr_dist_percent_difference


# In[ ]:


index7a = barr_dist_percent_difference.index

df7a = pd.DataFrame({'Total Proportion': barr_dist_pop,'Win Proportion': barr_dist_win,
                    'Percent Difference': barr_dist_percent_difference}, index=index7a)

ax = df7a.plot.bar(rot=0, title='The Significance of Barrier for Races Longer than 1800m')


# ### It appears that there may be an even bigger distinction with barrier 1 with longer race distances. However, the other barriers seem to even out.

# ## For jockey gender: 

# #### Overall percentage of men and women in races where both are represented:

# In[ ]:


#Drops races where there is only one jockey gender, meaning that the other gender can't win
jockey_sex_cleaned = column_cleaner(df_cleaned, df_grouped, 'jockey_sex')

jockey_sex_cleaned_win = cleaned_win_df(jockey_sex_cleaned)


# #### Finding the total a different way:

# In[ ]:


jockey_sex_cleaned.jockey_sex.value_counts(normalize=True)
#This amount is the sum of all 'male' and 'female' jockeys added together and THEN 'normalized'


# #### Win percentage of those races:

# In[ ]:


#Isolates wins in races with both jockey genders represented
jockey_sex_cleaned_win.jockey_sex.value_counts(normalize=True, dropna=False)


# In[ ]:


#Finding the percent difference between win and total
jockey_sex_pop = jockey_sex_cleaned.jockey_sex.value_counts(normalize=True,
                                                            dropna=False).values

jockey_sex_win = jockey_sex_cleaned_win.jockey_sex.value_counts(normalize=True,
                                                                dropna=False).values

jockey_sex_percent_difference = (jockey_sex_win - jockey_sex_pop)/jockey_sex_pop


# In[ ]:


index8 = ['Men','Women']

df8 = pd.DataFrame({'Total Proportion': jockey_sex_pop,'Win Proportion': jockey_sex_win ,
                    'Percent Difference': jockey_sex_percent_difference}, index=index8)

ax = df8.plot.bar(rot=0, title='The Significance of Jockey Gender')


# ### Using a 2-proportion z-test, I find that jockey gender is significant with a p-value of 2.3E-30
# #### (There is currently a bug with the statsmodels library concering compatibility with scipy, so I used a scientific calculator)

# ## How far back does form (previous finishes) become irrelevant?
# 
# ### The distribution after 1 start:

# In[ ]:


last_start_pop_cleaned = column_cleaner(df_cleaned, df_grouped, 'last_start1')

last_start_pop_cleaned_win = cleaned_win_df(last_start_pop_cleaned)


# In[ ]:


#Finding the percent difference between win and total
last_start_pop = last_start_pop_cleaned.last_start1.value_counts(normalize=True,
                                                                 dropna=False)

last_start_win = last_start_pop_cleaned_win.last_start1.value_counts(normalize=True,
                                                                     dropna=False)

last_start_percent_difference = (last_start_win - last_start_pop)/last_start_pop

last_start_percent_difference


# In[ ]:


index9 = ['1','2','3','4','5','6','7','8','9','ten+','scratch','fell','none']

df9 = pd.DataFrame({'Total Proportion': last_start_pop,'Win Proportion': last_start_win,
                    'Percent Difference': last_start_percent_difference}, index=index9)


# ### After 5 starts:

# In[ ]:


last_start_pop_cleaned = column_cleaner(df_cleaned, df_grouped, 'last_start5')

last_start_pop_cleaned_win = cleaned_win_df(last_start_pop_cleaned)


# In[ ]:


#Finding the percent difference between win and total
last_start_pop = last_start_pop_cleaned.last_start5.value_counts(normalize=True,
                                                                 dropna=False)

last_start_win = last_start_pop_cleaned_win.last_start5.value_counts(normalize=True,
                                                                     dropna=False)

last_start_percent_difference = (last_start_win - last_start_pop)/last_start_pop


# In[ ]:


index10 = ['1','2','3','4','5','6','7','8','9','ten+','scratch','fell','none']

df10 = pd.DataFrame({'Total Proportion': last_start_pop,'Win Proportion': last_start_win,
                    'Percent Difference': last_start_percent_difference}, index=index10)


# ### After 10 starts:

# In[ ]:


last_start_pop_cleaned = column_cleaner(df_cleaned, df_grouped, 'last_start10')

last_start_pop_cleaned_win = cleaned_win_df(last_start_pop_cleaned)


# In[ ]:


#Finding the percent difference between win and total
last_start_pop = last_start_pop_cleaned.last_start10.value_counts(normalize=True,
                                                                 dropna=False)

last_start_win = last_start_pop_cleaned_win.last_start10.value_counts(normalize=True,
                                                                     dropna=False)

last_start_percent_difference = (last_start_win - last_start_pop)/last_start_pop


# In[ ]:


index11 = ['1','2','3','4','5','6','7','8','9','ten+','scratch','fell','none']

df11 = pd.DataFrame({'Total Proportion': last_start_pop,'Win Proportion': last_start_win,
                    'Percent Difference': last_start_percent_difference}, index=index11)


# ### After 13 starts:

# In[ ]:


last_start_pop_cleaned = column_cleaner(df_cleaned, df_grouped, 'last_start13')

last_start_pop_cleaned_win = cleaned_win_df(last_start_pop_cleaned)


# In[ ]:


#Finding the percent difference between win and total
last_start_pop = last_start_pop_cleaned.last_start13.value_counts(normalize=True,
                                                                 dropna=False)

last_start_win = last_start_pop_cleaned_win.last_start13.value_counts(normalize=True,
                                                                     dropna=False)

last_start_percent_difference = (last_start_win - last_start_pop)/last_start_pop

last_start_percent_difference


# In[ ]:


index12 = ['1','2','3','4','5','6','7','8','9','ten+','scratch','fell','none']

df12 = pd.DataFrame({'Total Proportion': last_start_pop,'Win Proportion': last_start_win,
                    'Percent Difference': last_start_percent_difference}, index=index12)


# ### Graphing form data:

# In[ ]:


fig,ax1 = plt.subplots(2, 2)

df9.plot.bar(ax=ax1[0,0],figsize=(20, 10)).set_title('The Significance of Previous Result')
df10.plot.bar(ax=ax1[0,1],figsize=(20, 10)).set_title('The Significance of 5 Results Ago')
df11.plot.bar(ax=ax1[1,0],figsize=(20, 10)).set_title('The Significance of 10 Results Ago')
df12.plot.bar(ax=ax1[1,1],figsize=(20, 10)).set_title('The Significance of 13 Results Ago')


# # How often would you win and what would be your expected return if you always bet on the favorite?

# In[ ]:


odds_cleaned = column_cleaner(df_cleaned, df_grouped, 'bf_odds')


# In[ ]:


#creates dataframe with a unique index
odds_cleaned['uniq_idx'] = range(len(odds_cleaned))
odds_cleaned_uniq_idx = odds_cleaned.set_index('uniq_idx',append=True)
uniq_idx_grouped = odds_cleaned_uniq_idx.groupby(['date','venue_name',
                                                  'market_name'])

odds_cleaned_uniq_idx.head()


# In[ ]:


bf_min_indices = uniq_idx_grouped.bf_odds.idxmin
    
min_odds_cleaned = odds_cleaned_uniq_idx.loc[bf_min_indices].drop_duplicates()

min_odds_win = cleaned_win_df(min_odds_cleaned)

odds_pop = len(min_odds_cleaned)
odds_win = len(min_odds_win)

average = min_odds_win.bf_odds.agg('min').mean()

#Printing total number of favorite horses (equal to the number of races) and the number of times those horses win:
print(len(min_odds_cleaned))
print(len(min_odds_win))


# ### How often the favorite wins:

# In[ ]:


odds_win/odds_pop


# ### The expected return if betting 1 dollar on favorite every race:

# In[ ]:


-1*(1-odds_win/odds_pop) + average*odds_win/odds_pop


# # Beginning the Machine Learning Process:

# ## Dropping null-majority features, creating dummy variables, and replacing null values:

# In[ ]:


df_cleaned.drop(['condition_place_percent_z','condition_win_percent_z',
                 'distance_place_percent_z','distance_win_percent_z'],
                axis=1,inplace=True)

#drops last_start 11 through 20 to match information provided on racing websites
for i in range(10,20):
    df_cleaned.drop('last_start'+str(i+1),axis=1,inplace=True)


# In[ ]:


df_cleaned_test = df_cleaned.copy()

df_cleaned_test.reset_index(drop=True,inplace=True)


# In[ ]:


#Modifying categorical groups
df_cleaned_test.position_again = df_cleaned_test.position_again.replace([2,3,np.nan],
                                                                        [0,0,0])

categorical_list = ['sex','jockey_sex','condition','barrier']
for i in range(10):
    categorical_list.append('last_start'+str(i+1)) 

df_cleaned_test = pd.get_dummies(df_cleaned_test,columns=categorical_list,drop_first=True,dummy_na=1)

nan_list1 = ['days_since_last_run_z','overall_starts','prize_money_per_start_z',
             'overall_starts_z','overall_win_percent_z','overall_place_percent_z',
             'condition_starts_z','distance_starts_z',"track_starts_z","track_starts",
             "distance_starts","condition_starts",'weight_z','age_z','days_since_last_run',
             'overall_win_percent','overall_place_percent','distance_win_percent',
             'distance_place_percent','condition_win_percent','condition_place_percent']                            

for column1 in nan_list1:
    df_cleaned_test[str(column1)].fillna(-99, inplace=True)

df_cleaned_test = df_cleaned_test.convert_objects(convert_numeric=True)


# In[ ]:


df_cleaned_test.head()


# In[ ]:


df_cleaned_test.isnull().sum()


# ## Shuffling and splitting the grouped data:

# In[ ]:


X = df_cleaned_test.drop(['position_again','bf_odds'],axis=1)
y = df_cleaned_test['position_again']


# In[ ]:


#classifiers
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

#for function below
from sklearn.model_selection import StratifiedKFold
from time import time
from sklearn.metrics import make_scorer,confusion_matrix,accuracy_score,    precision_score,recall_score,f1_score,roc_auc_score,matthews_corrcoef


# ### The dataset is imbalanced and this needs to be accounted for.

# In[ ]:


#for xgboost scale_pos_weight
negative = len(df_cleaned_test[df_cleaned_test.position_again ==0])
positive = len(df_cleaned_test[df_cleaned_test.position_again ==1])
xgb_weight = negative/positive

xgb_weight


# In[ ]:


clf_B = LogisticRegression(random_state=0,class_weight='balanced')

clf_D = RandomForestClassifier(random_state=0,max_depth=15,class_weight='balanced')

clf_F = DecisionTreeClassifier(random_state=0,max_depth=5,class_weight='balanced')

clf_J = xgb.XGBClassifier(random_state=0,scale_pos_weight=xgb_weight)


# ### Creates a function to split data and fit, predict, and score models:

# In[ ]:


def metrics_function(target,pred):
    return accuracy_score(target, pred),precision_score(target, pred),        recall_score(target, pred),f1_score(target, pred),        roc_auc_score(target, pred),matthews_corrcoef(target, pred)

def FOLD_TEST(clf,X_all,y_all,folds_num,row_factor):
    start=time()
    
    KFLD=StratifiedKFold(n_splits=folds_num,random_state=0,shuffle=True)
    print ('{}:'.format(clf.__class__.__name__),'\n')
    
    acc_list_train=[]
    acc_list_test=[]
    prc_list_train=[]
    prc_list_test=[]
    rcal_list_train=[]
    rcal_list_test=[]
    f1_list_train=[]
    f1_list_test=[]
    matt_list_train=[]
    matt_list_test=[]
    AUC_list_train=[]
    AUC_list_test=[]
    
    samp_size=X_all.shape[0]//row_factor
    
    true_values = []
    predict_values =[]
    
    for fold,(train_index,target_index) in enumerate(KFLD.split(X_all[:samp_size],
                                                                y_all[:samp_size])):
        X_train=X_all.iloc[train_index].values
        y_train=y_all.iloc[train_index].values

        X_test=X_all.iloc[target_index].values
        y_test=y_all.iloc[target_index].values
        
        clf.fit(X_train,y_train)
        y_pred1=clf.predict(X_train)
        y_pred2=clf.predict(X_test)

        train_acc,train_prc,train_rcal,train_f1,train_auc,train_matt=metrics_function(y_train,y_pred1)
        
        test_acc,test_prc,test_rcal,test_f1,test_auc,test_matt=metrics_function(y_test,y_pred2)
        
        acc_list_train.append(train_acc)
        acc_list_test.append(test_acc)
        prc_list_train.append(train_prc)
        prc_list_test.append(test_prc)
        rcal_list_train.append(train_rcal)
        rcal_list_test.append(test_rcal)
        
        f1_list_train.append(train_f1)
        f1_list_test.append(test_f1)
        matt_list_train.append(train_matt)
        matt_list_test.append(test_matt)
        AUC_list_train.append(train_auc)
        AUC_list_test.append(test_auc)
        
        true_values = true_values + list(zip(target_index,y_test))
        predict_values = predict_values + list(zip(target_index,y_pred2))
        
    print("Averages:"'\n')
    
    print("Train acc: {}, Test acc: {}".format(np.mean(acc_list_train),
                                               np.mean(acc_list_test)))
    print("Train prc: {}, Test prc: {}".format(np.mean(prc_list_train),
                                               np.mean(prc_list_test)))
    print("Train recall: {}, Test recall: {}".format(np.mean(rcal_list_train),
                                                     np.mean(rcal_list_test)),'\n')
    
    print("Train f1: {}, Test f1: {}".format(np.mean(f1_list_train),
                                             np.mean(f1_list_test)))
    print("Train MattCC: {}, Test MattCC: {}".format(np.mean(matt_list_train),
                                                     np.mean(matt_list_test)))
    print("Train AUC: {}, Test AUC: {}".format(np.mean(AUC_list_train),
                                               np.mean(AUC_list_test)),'\n'*2)
        
    print("Sample Size: {}, Folds Num: {}, Time: {}".format(samp_size,folds_num,
                                                            time()-start),'\n'*2)
    
    total_picks = []
    correct_idx = []

    for ((a,b),(c,d)) in list(zip(true_values,predict_values)):
        if (b==1)&(d==1):
            correct_idx.append(a)
        if d==1:
            total_picks.append(c)

    win_odds_list=[]

    for a in correct_idx:
        win_odds_list.append(df_cleaned_test.bf_odds.iloc[a])

    average_win=np.mean(win_odds_list)
    
    print("Total Picks:",len(total_picks),"Average Win Odds:", average_win)
    print("Total Return:",average_win*len(correct_idx)-len(total_picks))
    print("Average Expected Return:",(average_win*len(correct_idx)-len(total_picks))/len(total_picks))


# ## The meaningful values here are:
# ### Test prc (precision), Sample Size, Total Picks, Average Win Odds, Total Return, and Average Expected Return.

# In[ ]:


FOLD_TEST(clf_B, X, y, 5, 2)


# In[ ]:


FOLD_TEST(clf_D, X, y, 5, 2)


# In[ ]:


FOLD_TEST(clf_F, X, y, 5, 2)


# In[ ]:


FOLD_TEST(clf_J, X, y, 5, 2)

