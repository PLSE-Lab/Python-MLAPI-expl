#!/usr/bin/env python
# coding: utf-8

# # Major League Baseball Historical Pitching Data: Pandas Tutorial and EDA
# 
# This data set contains historical MLB pitching statistics for all pitchers from 1871 to 2015. In this Pandas Tutorial and Exploratory Data Analysis, we will calculate different pitching metrics that are not given in the dataset, examine how pitching trends have changed over time, and attempt to identify which pitchers have been most effective in their roles. 
# 
# For the purposes of this tutorial and EDA, we will only be looking at pitchers from 1990 to 2015 who started at least 26 games in a single season and who did not appear as a relief pitcher at any point during the season. 
# 
# Note: Each player is given a unique playerID. The playerID is not the pitcher's name, but rather a code in order to differentiate between pitchers of the same name or similar names. If you wish to know a particular player's full given name, please Google the playerID.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Dataset
# 1. Load the Pitching.csv file into a DataFrame called `pitching`. You can do this by using the `.read_csv()` function within Pandas. For the input of the read_csv function, use the filepath for the Pitching.csv file that was printed above. 
# 
#     In order to get a sense of what the dataset initially looks like, display the first 5 rows. You can do this using the `.head()` function. If we leave the input blank, then it will display the first 5 rows by default. 
# 
# 

# In[ ]:


# 1. Use pd.read_csv('/kaggle/input/baseball-databank/Pitching.csv') to create a DataFrame called pitching.
# 2. Use .head() to display the first 5 rows of the dataset.

pitching = pd.read_csv('/kaggle/input/baseball-databank/Pitching.csv')
pitching.head()


#    We need to reduce the size of this dataset for a number of reasons.
#    * All statistics were not recorded during baseball's early years as you can see from the missing values above (`NaN`).
#    * Some of the columns contain data that is either meaningless or irrelevant. For example, the `GF` column is a statistic that does not apply to pitchers who start games.
#    * We only want to analyze pitching data from 1990 to 2015 as stated above. This dataset contains data from 1871 to 2015.
#     
# 2. Update the dataset to only include rows from 1990 and after (`pitching['yearID'] >= 1990`), that have at least 26 GS, or games started (`pitching['GS'] >= 26`), and that have no relief appearances (`pitching['G'] == pitching['GS']`). 
# 
#     Then, alter the dataset so that only the first 20 columns are kept. 
#    
#     Afterwards, remove the `stint`, `W`, `L`, `CG`, `SHO`, and `SV` columns from the dataset. 
#    
#     Finally, display the first 5 rows of the updated dataset.

# In[ ]:


# 1. Use conditional selecting to only select rows where the `yearID` is greater than or equal to 1990, `GS` is greater than or equal to 26, and `G` is equal to `GS`. 
# 2. Use .iloc[:, :20] to keep every row in which the three conditions are true and keep only the first 20 columns using slicing. The first parameter `:` means every row in which the three conditions are true will be kept. `:20` means that only the first 20 columns are kept.
# 3. Use .drop() to eliminate specific columns from the dataset. Within .drop() input a list of column names that you wish to drop and set `axis=1` to drop columns rather than rows. If we had set `axis=0`, we would have to include index names of the entire rows we wanted to drop.
# 4. Use .head() to display the first 5 rows of the new dataset.
# Remember to save the dataset as a variable name for each step.

pitching = pitching[(pitching['yearID'] >= 1990) & (pitching['GS'] >= 26) & (pitching['G'] == pitching['GS'])].iloc[:, :20]
pitching = pitching.drop(['stint', 'W', 'L', 'CG', 'SHO', 'SV'], axis=1)
pitching.head()


# 3. Set the playerID column as the index and then use the .head() function to display the first 7 rows in the updated dataset.

# In[ ]:


# 1. Use set_index('playerID') to make each `playerID` the index. Save the dataset as a variable.
# 2. Use .head(7) to display the first 7 rows of the updated dataset. In Pandas, you can choose the number of rows to display using .head(). As stated above, the default value is 5.

pitching = pitching.set_index('playerID')
pitching.head(7)


# 4. Display the bottom 6 rows of the dataset using the .tail() function.

# In[ ]:


# Use .tail(6) to show the bottom 6 rows of the dataset. Much like .head(), you can choose the number of bottom rows to display. Also like .head(), the default value for .tail() is 5.
pitching.tail(6)


# 5. How many rows are left in the dataset? 
#     
#     Note: This is not the same as the number of players left in the dataset as it is possible for the same player to pitch across multiple seasons.

# In[ ]:


# Use .shape[0] to determine the number of rows in the dataset. Using .shape[1] would have returned the number of columns and .shape would have returned (rows, columns).

pitching.shape[0]


# This output tells us that from 1990 to 2015, pitchers made at least 26 starts and did not serve as a relief pitchers at any point in the season 1659 times.

# # Calculating Pitching Metrics
# 
# 1. In baseball, the number of innings pitched, or IP, is a way of expressing the number of outs a pitcher records. There are three outs in one complete inning. For example, if a pitcher records 9 outs, then he is credited with 3.00 IP. IP are expressed in thirds. If a pitcher only records 8 outs, then he is credited with 2 2/3 or 2.67 IP. 
# 
#     In this dataset, we are given the number of outs each pitcher has recorded. Please create a new column called `IP` that shows the innings pitched for each pitcher. To calculate `IP`, select the `IPouts` column and divide by 3.
#     
#     After we calculate `IP`, we no longer need the `IPouts` column, so delete it from the dataset.
#     
#     Display the first five rows of the updated dataset.

# In[ ]:


# 1. To create a new column, index the new column name like you would do for a dictionary.
# 2. Perform the calculation of selecting the `IPouts` column and dividing by 3.
# 3. Use .round(2) to round the results to 2 decimal places.
# 4. Use the del function and select the `IPouts` column to delete the column from the dataset.
# 5. Use .head() to display the first 5 rows of the updated dataset.

pitching['IP'] = ((pitching['IPouts'])/3).round(2)
del pitching['IPouts']
pitching.head()


# 2. We now want to calculate each pitcher's strikeout rate in terms of number of strikeouts per 9 IP. We measure this value by 9 IP because there are typically 9 innings in 1 game. 
# 
#     This is a common metric used to evalutate how effective a pitcher is at striking out opposing hitters. This metric is calculated by multiplying the number of strikeouts by 9 and then dividing by the number of IP. Create a new column called `SO9` that displays each pitcher's rate.

# In[ ]:


# 1. To create a new column, index the new column name like you would do for a dictionary.
# 2. Perform the calculation of selecting the `SO` column, multiplying it by 9, and dividing by the `IP` column.
# 3. Use .round(2) to round the results to 2 decimal places.

pitching['SO9'] = (pitching['SO'] * 9 / pitching['IP']).round(2)


# 3. Please create a new column called `BB9` that displays each pitcher's BB (walks) per 9 innings pitched. This is a common metric used to evalutate how effective a pitcher is at preventing opposing hitters from walking.
#     
#     This is calculated the same way as `SO9`. Display the first 5 rows of the updated data set.

# In[ ]:


# 1. To create a new column, index the new column name like you would do for a dictionary.
# 2. Perform the calculation of selecting the `BB` column,  multiplying it by 9, and then dividing by the `IP` column.
# 3. Use .round(2) to round the results to 2 decimal places.
# 4. Use .head() to display the first 5 rows of the updated dataset.

pitching['BB9'] = (pitching['BB'] * 9 / pitching['IP']).round(2)
pitching.head()


# 4. Create a new column called `SOtoBB` that displays ratio of a pitcher's SO to BB. Display the first 5 rows of the updated data set.

# In[ ]:


# 1. To create a new column, index the new column name like you would do for a dictionary.
# 2. Perform the calculation of selecting the `SO` column and dividing by the `BB` column.
# 3. Use .round(2) to round the results to 2 decimal places.
# 4. Use .head() to display the first 5 rows of the updated dataset.

pitching['SOtoBB'] = (pitching['SO'] / pitching['BB']).round(2)
pitching.head()


# # Examining Pitching Trends from 1990 to 2015
# 
# 1. Create a bar chart of the average IP per GS for each season from 1990 to 2015.

# In[ ]:


# 1. Split by the `yearID` column.
# 2. Select the `IP` column.
# 3. Summarize with `.sum()`.
# 4. Divide
# 5. Split by the `yearID` column.
# 6. Select the `GS` column.
# 7. Summarize with `.sum()`.
# 4. Plot bar chart

(pitching.groupby('yearID').IP.sum()/pitching.groupby('yearID').GS.sum()).plot.bar()


# This chart shows us that since 1990, the average IP per GS has been slowly but steadily decreasing. In other words, starting pitchers are pitching less and less innings per game as time progresses.

# 2. Make a bar chart of the number of pitchers for each season in the dataset.

# In[ ]:


# 1. Split by the `yearID` column.
# 2. Use .size() method to count the number of pitchers for each season.
# 3. Plot bar chart

pitching.groupby('yearID').size().plot.bar()


# This chart shows that the number of pitchers that are exclusively starters and that pitched in at least 26 games is slightly inconsistent but has increased overall since 1990.
# 
# Note: the 1994 season was cut short due to the player labor stike. This is why the data for 1994 is an outlier.

# 3. Create a bar chart of the average SO9 for each season from 1990 to 2015.

# In[ ]:


# 1. Split by the `yearID` column.
# 2. Select the `SO9` column.
# 3. Summarize with `.mean()`.
# 4. Plot bar chart

pitching.groupby('yearID').SO9.mean().plot.bar()


# This graph shows that since 1990, the average strikeout rate has steadily increased.

# # Evaluating Pitcher Performance
# 
# 1. What is the playerID for the pitcher that recorded the most strikeouts in a single season from 1990 to 2015?

# In[ ]:


# 1. Select the SO column using pitching.SO
# 2. Use idxmax() to find the index corresponding to the maximum value.

pitching.SO.idxmax()


# 2. What is the playerID for the pitcher that recorded the least strikeouts in a single season from 1990 to 2015?

# In[ ]:


# 1. Select the SO column using pitching.SO
# 2. Use idxmin() to find the index corresponding to the minimum value.

pitching.SO.idxmin()


# 3. From 1990 to 2015, what players had the 10 lowest ERAs (earned run average), and what were those ERAs?

# In[ ]:


# 1. Select the ERA column using pitching.ERA
# 2. Use sort_values() to sort the values from smallest to largest.
# 3. Use .head(10) to display the smallest ERA to the 10th smallest ERA along with the playerIDs.

pitching.ERA.sort_values().head(10)


# The most important aspect of a pitcher's job is preventing the other team from scoring runs. For a pitcher, the less earned runs allowed, the better. These are the top 10 single season performances in terms of preventing earned runs from 1990 to 2015.

# 4. From 1990 to 2015, what players had the 10 highest SO9, and what were those values?

# In[ ]:


# 1. Select the SO9 column using pitching.SO9
# 2. Use sort_values(ascending=False) to sort the values from largest to smallest. The ascending=False parameter makes the values go from largets to smallest. The default value is ascending=True, meaning the values are sorted from smallest to largest.
# 3. Use .head(10) to display the largest SO9 to the 10th largest SO9 along with the playerIDs.

pitching.SO9.sort_values(ascending=False).head(10)


# It is also very important for pitchers to have high strikeout rates because allowing the ball to be put in play can risk fielding errors, more baserunners, and more runs allowed. These are the top 10 single season performances in terms of causing the opposing hitters to strikeout from 1990 to 2015.

# 5. From 1990 to 2015, what players had the 10 lowest BB9, and what were those values?

# In[ ]:


# 1. Select the BB9 column using pitching.BB9
# 2. Use sort_values() to sort the values from smallest to largest.
# 3. Use .head(10) to display the smallest BB9 to the 10th smallest BB9 along with the playerIDs.


pitching.BB9.sort_values().head(10)


# Preventing walks (BB) is also a key aspect when evaluating a pitcher's effectiveness. Walks are bad because they allow the other team to get into a position to score runs. These are the top 10 single-season performances in terms of preventing walks from 1990 to 2015.

# 6. From 1990 to 2015, what players had the 5 highest SOtoBB, and what were those values?

# In[ ]:


# 1. Select the SOtoBB column using pitching.SOtoBB
# 2. Use sort_values(ascending=False) to sort the values from largest to smallest.
# 3. Use .head(10) to display the largest SOtoBB to the 10th largest SOtoBB along with the playerIDs.

pitching.SOtoBB.sort_values(ascending=False).head(10)


# While preventing walks and causing strikeouts are both important when evaluating how effective pitchers are in preventing runs, pitchers should be able to do both at the same time. This is what SOtoBB measures. Because SO is the numerator and BB is the demominator, the higher the SOtoBB, the better. These are the top 10 single-season performances in terms of simultaneously causing strikeouts and preventing walks.
