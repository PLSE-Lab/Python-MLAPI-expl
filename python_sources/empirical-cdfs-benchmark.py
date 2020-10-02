#!/usr/bin/env python
# coding: utf-8

# # Explanation

# I always find it useful to build a strong (but simple) benchmark model to evaluate the performance of my more complex ML models. For this competition, the obvious benchmark to me is the empircal CDF of Yards gained given the distance between the line of scrimmage and the goaline.
# 
# First, let me explain something for the non American Football fans... Let's say the best team in the league, the Saints, are playing the Falcons at the Superdome. And let's suppose the field runs east and west, meaning there's an endzone at the east side and an endzone at the west side of the field. At the beginning of the 1st quarter, perhaps the Saints are trying to reach the east endzone and the Falcons are trying to reach the west endzone. Your mental image should be something like this
# 
# \[west endzone\] | | 10 | | 20 | | 30 | | 40 | | 50 | | 40 | | 30 | | 20 | | 10 | | \[east endzone\]  
# \[======== Saints side of field ========\]\[====== Falcons side of field ========\]  
# Saints try to go this way ============> <============ Falcons try to go this way  
# 
# At the start of the 2nd quarter, the Saints and Falcons will switch sides of the field.
# 
# \[west endzone\] | | 10 | | 20 | | 30 | | 40 | | 50 | | 40 | | 30 | | 20 | | 10 | | \[east endzone\]  
# \[======== Falcons side of field =======\]\[====== Saints side of field =========\]  
# Falcons try to go this way ===========> <============ Saints try to go this way   
# 
# This can also change in the 3rd and 4th quarters.
# 
# Now, if we're in the 1st quarter and I tell you the Saints have the ball is on the 30 yard line, what does that mean? It's an ambiguous statement because there are two 30 yardlines. If I tell you the Saints have the ball on the Falcons's 30 yard line, then we know exactly where the ball is. This is especially important because the nature of the game imposes two constraints on how many yards a rusher can gain, namely
# 
# 1. He can't gain more than 30 yards because there's only 30 yards left to reach the east endzone
# 2. He can't gain fewer than -70 yards because he's 70 yards from the west endzone
# 
# So, my strategy for this benchmark model is to
# 1. Use *PossessionTeam* and *FieldPosition* to transform *YardLine* (which is ambiguous) into *YL* which goes from 0 to 100 where the offensive team is always trying to reach 100. 
# 2. Calculate the empirical CDF given *YL* and use this as my prediction for future samples
# 

# # Setup

# In[ ]:


import numpy as np
import pandas as pd
from kaggle.competitions import nflrush

# run this once!
env = nflrush.make_env()


# # Load Data

# In[ ]:


# read train
train = pd.read_csv(
    filepath_or_buffer = '/kaggle/input/nfl-big-data-bowl-2020/train.csv',
    usecols = ['PlayId', 'NflId', 'NflIdRusher', 'PossessionTeam', 'FieldPosition', 'YardLine', 'Yards'],
    low_memory = False
)


# # Model

# In[ ]:


class BenchmarkCDFs():

    def __init__(self):
        self.cdfs = None

    def transform_yardline(self, x):
        # Create YL := transformed version of YardLine on the scale 1 - 100 such that
        # the possessing team is always heading towards YL 100
        # (Here, x should be one row of a DataFrame. This method is meant to be
        #  called by DataFrame.apply())

        if (x.YardLine == 50 or x.PossessionTeam == x.FieldPosition):
            return x.YardLine
        else:
            return 50 + (50 - x.YardLine)

    def fit(self, train):
        # learns the empirical CDF given the current line of scrimmage position (YL)
        # saves the lookup table as self.cdfs

        # Subset rows where the player is the rusher. This should create a complete set of unique PlayIds
        plays = train.loc[train.NflId == train.NflIdRusher].copy()

        # Insert YL (modified YardLine on scale 1 - 99)
        plays.loc[:, 'YL'] = plays.apply(self.transform_yardline, axis = 1)

        # Build lookup table rowset (cdfs)
        dfList = [None] * 99
        for i in range(1, 100):
            # Build dataframe with current YL and all possible Yards
            dfList[i - 1] = pd.DataFrame({
                'YL': i,
                'Yards': np.arange(start = -99, stop = 100, step = 1)
            })

        # Combine into one dataframe
        cdfs = pd.concat(dfList)
        cdfs.set_index(keys = ['YL', 'Yards'], inplace = True)

        # Calculate empirical stats
        empiricals = plays.groupby(['YL', 'Yards']).size()
        counts = plays.groupby('YL').size()
        pdfs = empiricals / counts

        # Merge to cdfs and calculate CDF
        cdfs = cdfs.merge(pdfs.rename('PDF'), how = 'left', left_index = True, right_index = True)
        cdfs.fillna(0, inplace = True)
        cdfs.loc[:, 'CDF'] = cdfs.groupby(['YL'])['PDF'].cumsum()
        cdfs.loc[:, 'CDF'] = np.minimum(1.0, cdfs.CDF.values)

        # Save table to this object
        self.cdfs = cdfs

    def predict(self, test):
        # make predictions for a dataframe of play attributes
        # test should be all the rows associated with a single PlayId (although we'll only use the 1st row)
        # returns a 1-row DataFrame with columns {Yards-99, Yards-98, ... Yards98, Yards99}

        if(self.cdfs is None):
            raise Exception('Call the fit() method first!')

        if(test.PlayId.nunique() != 1):
            raise Exception('test should have a single PlayId!')

        # Extract one row from the test set and insert YL
        temp = test.iloc[[0]].loc[:, ['PlayId', 'PossessionTeam', 'FieldPosition', 'YardLine']].copy()
        temp.loc[:, 'YL'] = temp.apply(self.transform_yardline, axis = 1)
        temp.set_index('YL', inplace = True)

        # Lookup the CDF for the given YL
        cdf = temp.merge(self.cdfs, how = 'left', left_index = True, right_index = True)

        # Format the output
        result = cdf.reset_index().pivot(index = 'PlayId', columns = 'Yards', values = 'CDF')
        result = result.reset_index(drop = True)
        result.rename_axis(None, axis = 1, inplace = True)
        result = result.add_prefix('Yards')
        result.index = list(result.index)  # Convert range index to int index
        return result


# # Run

# In[ ]:


# Create benchmark model
mymodel = BenchmarkCDFs()
mymodel.fit(train)

# Loop thru test data and make predictions
for (test_df, sample_prediction_df) in env.iter_test():
    predictions_df = mymodel.predict(test_df)
    env.predict(predictions_df)
    
# Write submisison
env.write_submission_file()

