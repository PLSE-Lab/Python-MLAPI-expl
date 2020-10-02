#!/usr/bin/env python
# coding: utf-8

# # Overview
# In this kernel, we will go through everything we need to score our submission ourself (and a bit more).  When we're done, we'll be able to:
# 1. Create a submission file based on a simple seed-based model.
# 2. Efficiently add seed information to tournament games/results and filter out "play-in" games (which are not used in scoring).
# 3.  Create an appropriate Results file.
# 4. Score predictions for every NCAA Tournament from 1985 (as opposed to just the 2014 - 2018 tournaments included in the official Stage 1 scoring).
# 5.  Add a few checks to make sure we are not doing something obviously wrong.
# 6. Analyze scoring results from individual games for the simple seed-based model.
# 
# 
# # I Introduction
# 
# Say we've created a model in Stage 1, and we're generating a file for submission.  In that submission file, we'll have two columns: 'ID' and 'Pred'.  Then we'll want to see how well that submission file scores.  We have two options: 
# 1. We can submit the file to Kaggle for scoring.
#     * Advantage:  It's easy
#     * Disadvantages: You can submit predictions for only the 2014-2018 tournaments; you don't learn which games (or even which tournaments) are contributing the most/least to your score; you're not warned when you've made an obvious mistake (e.g., your probabilities are not between 0 and 1).
# 2.  We can score the predictions ourselves.
#     * Advantage:  We can make it more transparent and get some feedback about individual predictions; we can build in functionality to warn us we've made obvious errors.
#     * Disadvantage:  It takes a little bit of work.  That's where this kernel comes in.    
# 

# # II  Creating a Submission File
# 
# For this competition, the organizers provide us with a file called SampleSubmissionStage1.csv.   In that file is a list of game ID's for the 2014-2018 season (under the heading 'ID') and a list of predictions, all set to the 50/50 benchmark value of 0.5.  If you make a submission where all of your predictions are 0.5, you are guaranteed to get a score of ln(2) = 0.693...  In this section we will create a Submission file from scratch using the seed-based benchmark of years' past.
# 
# The seed-benchmark was a linear model with the formula
# Pr(Team X beats Team Y) = 0.5 + 0.03*(Seed(Team Y) - Seed(Team X))
# 
# It's not a great model, but it will suit our purposes here.  To get started, we grab the file which contains historical seed information

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting (when it can't be done under the pandas hood)

# Input data files are available in the "../input/" directory.
import os
data_directory = '../input/datafiles'

df_seeds = pd.read_csv(os.path.join(data_directory,'NCAATourneySeeds.csv'))
df_seeds.head()


# We want to create a submission file which lists every possible game that could have been played.  We're going to ignore the fact that most of them weren't played.  At this point we are operating on no information of what actually occurred.  We want each game to be listed once and only once.  The following code does this.

# In[ ]:


# Merge the seeds file with itself on Season.  This creates every combination of two teams by season.
df_sub = df_seeds.merge(df_seeds, how='inner', on='Season')

# We want a little less than half the records in this data frame.  
# Every game appears twice, once with the lower id team in the TeamID_x column, and 
# once in the TeamID_y column.  We also have the impossible matchups of teams with themselves.
# To fix this, we keep only the games where the lower team ID is in the TeamID_x columns.
df_sub = df_sub[df_sub['TeamID_x'] < df_sub['TeamID_y']]

df_sub.head()


# We are now able to make our submission file.  The convention is to write the game ID as a 14 character string of the form 'XXXX_YYYY_ZZZZ', where 'XXXX' is the season, 'YYYY' is the lower team ID, and 'ZZZZ' is the greater team ID.  For the prediction column ('Pred'), we'll get the integer value of the seed and use the formula
# 
# Pr(Team X beats Team Y) = 0.5 + 0.03*(Seed(Team Y) - Seed(Team X))

# In[ ]:


df_sub['ID'] = df_sub['Season'].astype(str) + '_'               + df_sub['TeamID_x'].astype(str) + '_'               + df_sub['TeamID_y'].astype(str)

df_sub['SeedInt_x'] = [int(x[1:3]) for x in df_sub['Seed_x']]
df_sub['SeedInt_y'] = [int(x[1:3]) for x in df_sub['Seed_y']]

df_sub['Pred'] = 0.5 + 0.03*(df_sub['SeedInt_y'] - df_sub['SeedInt_x'])

df_sub.head()


# The only thing left is to get rid of the unnecessary columns and pare down to just 'ID' and 'Pred'.  Just before we do that, let's save out the submission for 2014-2018 as a csv file so we can submit it to Kaggle for Stage 1.

# In[ ]:


# save out the 2014-2018 predictions for later submission
df_sub.loc[(df_sub['Season'] >= 2014) & (df_sub['Season'] <= 2018), ['ID', 'Pred']].to_csv('./Submission.csv',index=False)

# now pare down existing df_sub
df_sub = df_sub[['ID','Pred']]
df_sub.head()


# # III Creating a Results File
# 
# Now that we have a submission file that we want to score, we need a results file to score against. We will create one with every relevant tournament game since 1985.  To get this, we will start with the compact tourney results:

# In[ ]:


df_tc = pd.read_csv(os.path.join(data_directory,'NCAATourneyCompactResults.csv'))
df_tc.head()


# We will now generate the column for the game ID and the column with the result.  The convention is to write the game ID as a 14 character string of the form 'XXXX_YYYY_ZZZZ', where 'XXXX' is the season, 'YYYY' is the lower team ID, and 'ZZZZ' is the greater team ID.  For the Result, the convention is to record a result of 1 if the lower team id team wins, and a result of 0 if the greater team id wins.
# 
# A quick way to do this using pandas is the following:

# In[ ]:


df_tc['ID'] = df_tc['Season'].astype(str) + '_'               + (np.minimum(df_tc['WTeamID'],df_tc['LTeamID'])).astype(str) + '_'               + (np.maximum(df_tc['WTeamID'],df_tc['LTeamID'])).astype(str)

df_tc['Result'] = 1*(df_tc['WTeamID'] < df_tc['LTeamID'])
df_tc.head(10)


# Up until 2001, this would have been all that we needed. However, things got more complicated in 2001 with the addition of "play-in" games. Why is this more complicated? In this competition, we are scored only on the games that occur in the official tournament (i.e., Round 1 or later). To get the real results file, we need to get rid of the play-in games that appear in this data frame. To do that, we have to figure out which tournament games were play-in games.  Fortunately, this isn't too bad.  We just first have to add the seed information for the winning and losing teams.

# In[ ]:


df_tc = df_tc.merge(df_seeds.rename(columns={'Seed':'WSeed','TeamID':'WTeamID'}), 
                    how='inner', on=['Season', 'WTeamID'])
df_tc = df_tc.merge(df_seeds.rename(columns={'Seed':'LSeed','TeamID':'LTeamID'}), 
                    how='inner', on=['Season', 'LTeamID'])
df_tc.head()


# The play-in games are the games where the the first three characters of the teams' seeds are identical.  For example, in one play-in game, you might see seed 'X16a' facing seed 'X16b'.  Only in a play-in game will you see both seeds start with 'X16'.

# In[ ]:


df_playin = df_tc[df_tc['WSeed'].str[0:3] == df_tc['LSeed'].str[0:3]]
df_playin.head()


# We can thus get rid of these games from the tourney results dataframe in the following way:

# In[ ]:


df_tc = df_tc[df_tc['WSeed'].str[0:3] != df_tc['LSeed'].str[0:3]]
df_tc.head()


# # IV Scoring Predictions Against Results
# 
# In this section we present a function which scores a submission file against an existing results file.  We use the log loss as defined by Kaggle (including the clipping), but when we score the submission, we will make note of which predictions contained obvious errors.

# In[ ]:


def kaggle_clip_log(x):
    '''
    Calculates the natural logarithm, but with the argument clipped within [1e-15, 1 - 1e-15]
    '''
    return np.log(np.clip(x,1.0e-15, 1.0 - 1.0e-15))

def kaggle_log_loss(pred, result):
    '''
    Calculates the kaggle log loss for prediction pred given result result
    '''
    return -(result*kaggle_clip_log(pred) + (1-result)*kaggle_clip_log(1.0 - pred))
    
def score_submission(df_sub, df_results, on_season=None, return_df_analysis=True):
    '''
    Scores a submission against relevant tournament results
    
    Parameters
    ==========
    df_sub: Pandas dataframe containing predictions to be scored (must contain a column called 'ID' and 
            a column called 'Pred')
            
    df_results: Pandas dataframe containing results to be compared against (must contain a column 
            called 'ID' and a column called 'Result')
            
    on_season: array-like or None.  If array, should contain the seasons for which a score should
            be calculated.  If None, will use all seasons present in df_results
            
    return_df_analysis: Bool.  If True, will return the dataframe used for calculations.  This is useful
            for future analysis
            
    Returns
    =======
    df_score: pandas dataframe containing the average score over predictions that were scorable per season
           as well as the number of obvious errors encountered
    df_analysis:  pandas dataframe containing information about all results used in scoring
                  Only provided if return_df_analysis=True
    '''
    
    df_analysis = df_results.copy()
    
    # this will overwrite if there's already a season column but it should be the same
    df_analysis['Season'] = [int(x.split('_')[0]) for x in df_results['ID']]
    
    if not on_season is None:
        df_analysis = df_analysis[np.in1d(df_analysis['Season'], on_season)]
        
    # left merge with the submission.  This will keep all games for which there
    # are results regardless of whether there is a prediction
    df_analysis = df_analysis.merge(df_sub, how='left', on='ID')
    
    # check to see if there are obvious errors in the predictions:
    # Obvious errors include predictions that are less than 0, greater than 1, or nan
    # You can add more if you like
    df_analysis['ObviousError'] = 1*((df_analysis['Pred'] < 0.0)                                   | (df_analysis['Pred'] > 1.0)                                   | (df_analysis['Pred'].isnull()))
    
    df_analysis['LogLoss'] = kaggle_log_loss(df_analysis['Pred'], df_analysis['Result'])
    
    df_score = df_analysis.groupby('Season').agg({'LogLoss' : 'mean', 'ObviousError': 'sum'})
    
    if return_df_analysis:
        return df_score, df_analysis
    else:
        return df_score
    


# As a test, let's look at the resulting scores from our submission and results files.  Let's look at 2014-2018, because that's what the official Stage 1 scoring does.

# In[ ]:


df_score, df_analysis =     score_submission(df_sub, df_tc, on_season = np.arange(2014,2019), return_df_analysis=True)
df_score


# Yay! No obvious errors encountered.  And the mean score is the same as what Kaggle claims (I've checked using the csv written out earlier):

# In[ ]:


df_score.mean()


# While not great, the simple seed model is not that bad.  There have been a few recent tournaments where it was quite good.  What about if we expand over a few more years?

# In[ ]:


df_score, df_analysis =     score_submission(df_sub, df_tc, on_season = np.arange(2010,2018), return_df_analysis=True)
df_score


# Again, success!  What if we randomly forget to make a few predictions?  Let's submit only half as many predictions.

# In[ ]:


df_score, df_analysis =     score_submission(df_sub.sample(frac=0.5), df_tc, on_season = np.arange(2010,2018), return_df_analysis=True)
df_score


# Now there are lots of errors because we didn't submit predictions for half the games (some of which the scoring function looked for).  Note that the scoring in this function will just ignore missing values for the purposes of calculating the mean score.  Kaggle is not so kind, but I don't know if they give you the max penalty for null values or if they just refuse to accept your submission.
# 
# We can look at which games had an error by entering something like the following:

# In[ ]:


df_analysis[df_analysis['ObviousError']==1].head()


# Scrolling over to the right, we see the reason for the 'obvious' error: there's not prediction for a game that occurred.  Similarly, we should see an obvious problem if the prediction is less than 0 or greater than 1.

# # V Analyzing Results for the Simple Seed Model
# 
# Let's consider all years in the results file.

# In[ ]:


df_score, df_analysis =     score_submission(df_sub, df_tc, on_season = None, return_df_analysis=True)

df_score


# We can look at the mean score over all years:

# In[ ]:


df_score.mean()


# We can plot the score as a function of year 

# In[ ]:


df_score.reset_index().plot('Season','LogLoss')


# We can make a histogram of the yearly score

# In[ ]:


df_score.hist('LogLoss',bins='auto')


# And we can use  df_analysis to look at the log loss from individual games.  Let's sort df_analysis in order of descending log loss and look at the worst offenders.  Let's also make a histogram of the log losses that we have experienced.

# In[ ]:


df_analysis.hist('LogLoss',bins=10)
df_analysis.sort_values('LogLoss',ascending=False).head(20)


# No surprise that the worst scoring prediction we've had is from last year, when a number 16 seed upset a number one seed for the first time in history.  That's especially bad for a model that bases its entire prediction on the seed difference of the teams.
# 
# For models that are not based just on seeds, we could similarly go through the process that we have gone through:  Create a submission, test against the tournament results, and look at which individual games contributed to our losses.  Ideally, we'll be able to spot some pattern in the worst offenders and then tweak our models so that we are not prone to those types of errors in the future.  But, as always beware of overtraining and leakage!
# 
# 

# # Conclusion
# 
# We've gone through how to create a submission file from scratch based on a seed difference model, how to create a results file containing only the relevant tournament games, and how to score that submission file against the relevant results in such a way that you can avoid obvious errors and gain transparency into what individual games are most affecting your results.  I hope that you have found this kernel instructive and I look forward to your feedback.
# 
# Best of luck in the competition.

# In[ ]:




