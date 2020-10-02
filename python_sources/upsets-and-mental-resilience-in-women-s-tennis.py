#!/usr/bin/env python
# coding: utf-8

#  - How likely it is for a player to beat an opponent whose rank is significantly higher?
#  - Are upsets more frequent on different surfaces?
#  - Are upsets more common in small tournaments?
#  - Do older and more experienced players perform better under pressure (when facing breaking points, 3rd sets or when playing in the finals for instance)?

# 
# 
# 
# 
# 
# 
# First Let's Read the Data and do some feature engineering required for the analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
plt.style.use('fivethirtyeight')

path = "../input/"
os.chdir(path)
filenames = os.listdir(path)
df = pd.DataFrame()
for filename in sorted(filenames):
    try:
        read_filename = '../input/' + filename
        temp = pd.read_csv(read_filename,encoding='utf8')
        frame = [df,temp]
        df = pd.concat(frame)
    except UnicodeDecodeError:
        pass
    
df['Year'] = df.tourney_date.apply(lambda x: str(x)[0:4])
df['Sets'] = df.score.apply(lambda x: x.count('-'))
df['Rank_Diff'] =  df['loser_rank'] - df['winner_rank']
df['Rank_Diff_Round'] = df.Rank_Diff.apply(lambda x: 10*round(np.true_divide(x,10)))
df['ind'] = range(len(df))
df = df.set_index('ind')

bins = np.arange(10,200,10)
diff_df = pd.DataFrame()
diff_df['bins'] = bins
diff_df['Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(df[df.Rank_Diff_Round == x]),(len(df[df.Rank_Diff_Round == x]) +len(df[df.Rank_Diff_Round == -x]))))
diff_df['Grass_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Grass')]),(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Grass')]) +len(df[(df.Rank_Diff_Round == -x) & (df.surface == 'Grass')]))))
diff_df['Clay_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Clay')]),(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Clay')]) +len(df[(df.Rank_Diff_Round == -x) & (df.surface == 'Clay')]))))
diff_df['Hard_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Hard')]),(len(df[(df.Rank_Diff_Round == x) & (df.surface == 'Hard')]) +len(df[(df.Rank_Diff_Round == -x) & (df.surface == 'Hard')]))))


# ## Upsets in Women's Tennis
# 
# Now let's see how do the winning probabilities increase for the favorite with the rank difference. I've done something similar with [Men's Tennis][1] in the past. 
# 
# 
#   [1]: https://www.kaggle.com/drgilermo/d/jordangoblet/atp-tour-20002016/competitiveness-and-expertise-in-tennis

# In[ ]:


plt.bar(diff_df.bins,diff_df.Prob,width = 9)
plt.ylim([0.5,0.9])
plt.xlabel('Rank Difference')
plt.ylabel('Winning Probability')
plt.title('How likely are upsets?')


# We can see that quite predictably the chances for the favorite increase with the rank difference.
# 
# Similarly to men, when the rank differences reaches to 100, the plot flattens and the winning chances cease to increase. Playing against someone who's 100 places below the favorite is similar to playing against some who is 200 places down.
# 
# Does this trend change with the surface?

# In[ ]:


plt.plot(diff_df.bins,diff_df.Grass_Prob,'g')
plt.plot(diff_df.bins,diff_df.Hard_Prob,'b')
plt.plot(diff_df.bins,diff_df.Clay_Prob,'r')
plt.legend(['Grass','Hard','Clay'], loc = 2, fontsize = 12)
plt.ylim([0.5,0.9])
plt.xlabel('Rank Difference')
plt.ylabel('Winning Probability')
plt.title('Upsets on Different Surfaces')


# There isn't seem to be any clear difference between the different surfaces - upsets remain as likely (or unlikely) whether the game is played on grass or clay.
# 
# Now let's see if the tournament effects these odds. in the Men's script, it was obvious that upsets are less likely in Grand Slam tournaments. though it was difficult to conclude whether this is merely due to the fact that grand slam matches are comprised of 5 sets (instead of 3), which makes surprising outcomes less likely.
# 
# In women's tennis, the match length is not dependent on the tournament level. so let's see how this affects the probabilities:

# In[ ]:


big_tour_df = df[df.draw_size == 128]
bins = np.arange(10,200,10)
diff_df = pd.DataFrame()
diff_df['bins'] = bins
diff_df['Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[big_tour_df.Rank_Diff_Round == x]),(len(big_tour_df[big_tour_df.Rank_Diff_Round == x]) +len(big_tour_df[big_tour_df.Rank_Diff_Round == -x]))))
diff_df['Grass_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Grass')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Grass')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Grass')]))))
diff_df['Clay_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Clay')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Clay')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Clay')]))))
diff_df['Hard_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Hard')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Hard')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Hard')]))))

plt.bar(diff_df.bins,diff_df.Prob,width = 9, color = 'r')
plt.ylim([0.5,0.9])
plt.xlabel('Rank Difference')
plt.ylabel('Winning Probability')


big_tour_df = df[df.draw_size == 32]
bins = np.arange(10,200,10)
diff_df = pd.DataFrame()
diff_df['bins'] = bins
diff_df['Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[big_tour_df.Rank_Diff_Round == x]),(len(big_tour_df[big_tour_df.Rank_Diff_Round == x]) +len(big_tour_df[big_tour_df.Rank_Diff_Round == -x]))))
diff_df['Grass_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Grass')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Grass')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Grass')]))))
diff_df['Clay_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Clay')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Clay')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Clay')]))))
diff_df['Hard_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Hard')]),(len(big_tour_df[(big_tour_df.Rank_Diff_Round == x) & (big_tour_df.surface == 'Hard')]) +len(big_tour_df[(big_tour_df.Rank_Diff_Round == -x) & (big_tour_df.surface == 'Hard')]))))

plt.bar(diff_df.bins,diff_df.Prob,width = 7)
plt.ylim([0.5,0.9])
plt.xlabel('Rank Difference')
plt.ylabel('Winning Probability')
plt.xlim(0,175)
plt.legend(['Big Tournaments', 'Small Tournaments'], loc = 2, fontsize = 12)
plt.title('Upsets are more likely in small Tournaments')


# The difference is quite noticeable!
# 
# It's more difficult to win a better opponent in big tournaments. This could be due to:
# 
#  - Better mental and physical preparation of the good players - they attend many tournaments during the year but only a few are considered as "Money Time"
#  - The atmosphere is more stressful but less skilled and experienced players  
# 
# Let's see if we can detect a similar effect if we look at single sets:
# Do better players perform better in the last set, maybe waking up after 2 bad sets?
# Do worse players stress out in the last set?
# 
# Or maybe it goes the other way around - when a good player is forced to play the 3rd set against an inferior opponent, he could actually be the one stressing out, not having the match advancing based on the plan. T
# 
# There is of course another possibility - If the game reached the 3rd set, this means that in that very particular date, the rank difference might not be reflective of the real gap between the players, which is actually smaller. and therefore we would see that rank difference doesn't matter as much in the 3rd set.

# In[ ]:


def who_won(score,set_num):
    try:
        sets = score.split()
        set_score = sets[set_num-1]
        w = set_score[0]
        l = set_score[2]
        if int(w)>int(l):
            return 1
        if int(w)<int(l):
            return 0
    except ValueError:
        return -1
    except IndexError:
        return -1
df['1st_set'] = df.score.apply(lambda x: who_won(x,1))
df['2nd_set'] = df.score.apply(lambda x: who_won(x,2))
df['3rd_set'] = df.score.apply(lambda x: who_won(x,3))


# In[ ]:


def winning_per_set(Rank_diff, df, set_num):
    positive_diff_w = len(df[(df.Rank_Diff_Round == Rank_diff) & (df[set_num] == 1)])
    positive_diff_l = len(df[(df.Rank_Diff_Round == Rank_diff) & (df[set_num] == 0)])
    
    negative_diff_w = len(df[(df.Rank_Diff_Round == -Rank_diff) & (df[set_num] == 1)])
    negative_diff_l = len(df[(df.Rank_Diff_Round == -Rank_diff) & (df[set_num] == 0)])
    
    w = positive_diff_w + negative_diff_l
    l = positive_diff_l + negative_diff_w
    return np.true_divide(w, l + w)
 
bins = np.arange(10,200,10)
diff_df = pd.DataFrame()
diff_df['bins'] = bins
diff_df['Prob_1'] = diff_df.bins.apply(lambda x: winning_per_set(x,df,'1st_set'))
diff_df['Prob_2'] = diff_df.bins.apply(lambda x: winning_per_set(x,df,'2nd_set'))
diff_df['Prob_3'] = diff_df.bins.apply(lambda x: winning_per_set(x,df,'3rd_set'))


plt.plot(diff_df.bins,diff_df.Prob_1)
plt.plot(diff_df.bins,diff_df.Prob_2)
plt.plot(diff_df.bins,diff_df.Prob_3)
plt.legend(['Set 1','Set 2','Set 3'], loc = 2, fontsize = 12)
plt.xlabel('Rank Difference')
plt.ylabel('Winning Probability')
plt.title('Upsets are more likely in the last set')


# In[ ]:


last_set = df[df.Sets == 2]
bins = np.arange(10,200,10)
diff_df = pd.DataFrame()
diff_df['bins'] = bins
diff_df['Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[last_set.Rank_Diff_Round == x]),(len(last_set[last_set.Rank_Diff_Round == x]) +len(last_set[last_set.Rank_Diff_Round == -x]))))
diff_df['Grass_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Grass')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Grass')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Grass')]))))
diff_df['Clay_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Clay')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Clay')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Clay')]))))
diff_df['Hard_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Hard')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Hard')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Hard')]))))

plt.bar(diff_df.bins,diff_df.Prob,width = 9)
plt.ylim([0.5,0.9])
plt.xlabel('Rank Difference')
plt.ylabel('Winning Probability')

last_set = df[df.Sets == 3]
bins = np.arange(10,200,10)
diff_df = pd.DataFrame()
diff_df['bins'] = bins
diff_df['Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[last_set.Rank_Diff_Round == x]),(len(last_set[last_set.Rank_Diff_Round == x]) +len(last_set[last_set.Rank_Diff_Round == -x]))))
diff_df['Grass_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Grass')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Grass')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Grass')]))))
diff_df['Clay_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Clay')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Clay')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Clay')]))))
diff_df['Hard_Prob'] = diff_df.bins.apply(lambda x: np.true_divide(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Hard')]),(len(last_set[(last_set.Rank_Diff_Round == x) & (last_set.surface == 'Hard')]) +len(last_set[(last_set.Rank_Diff_Round == -x) & (last_set.surface == 'Hard')]))))

plt.bar(diff_df.bins,diff_df.Prob,width = 7, color = 'r')
plt.ylim([0.5,0.9])
plt.xlabel('Rank Difference')
plt.ylabel('Winning Probability')

plt.legend(['2 Sets','3 Sets'], loc = 2, fontsize = 12)


# We see in the above graph that games that were ended after 2 sets are more often games won by an obvious favorite. this is of course totally unsurprising - if you are about to win a better opponent, it would probably take you 3 sets to do that.

# ## Age and Mental strength?
# 
# Let's see if we can use the data to show how older players perform better under pressure (spoiler - I could not prove that).
# 
# When a player has a bad serving game, she might face a breaking point. Then the mental resilience can kick in - a calm and experienced player should be more likely to save this point. let's see of the breaking point saving rate is higher among older players:

# It does seem like in the 3rd set some of the difference is eliminated, and the players are more neutral to the rank difference, though this is not entirely conclusive.
# 
# 

# In[ ]:


df['bp_saving_rate_l'] = np.true_divide(df.l_bpSaved,df.l_bpFaced)
df['bp_saving_rate_w'] = np.true_divide(df.w_bpSaved,df.w_bpFaced)

plt.plot(df.winner_age,df.bp_saving_rate_w + np.random.normal(0,0.02,len(df)),'o', alpha = 0.1)
plt.plot(df.loser_age,df.bp_saving_rate_l + np.random.normal(0,0.02,len(df)),'o', alpha = 0.1)

x1 = df.winner_age[(np.isnan(df.winner_age) == 0) & (np.isnan(df.bp_saving_rate_w) == 0)]
y1 = df.bp_saving_rate_w[(np.isnan(df.winner_age) == 0) & (np.isnan(df.bp_saving_rate_w) == 0)]

x2 = df.loser_age[(np.isnan(df.loser_age) == 0) & (np.isnan(df.bp_saving_rate_l) == 0)]
y2 = df.bp_saving_rate_l[(np.isnan(df.loser_age) == 0) & (np.isnan(df.bp_saving_rate_l) == 0)]
plt.ylim([0.1,0.9])
plt.xlabel('Age')
plt.ylabel('Breaking Points Saving Rate')
plt.legend(['Winners','Losers'], loc = 4)
plt.title('Saving breaking points and Age')

print('Correlation between age and saving rates, Winners :',np.corrcoef(x1,y1)[1][0])
print('Correlation between age and saving rates, Losers :', np.corrcoef(x2,y2)[1][0])


# 
# Mmm, not really.
# 
# Maybe in the finals? a finals game is obviously more stressful than any other game. do older players outperform younger?
# 
# in general, the average winner is slightly older than the loser. let's see if the histogram is more skewed towards older players in finals or in the last set:

# In[ ]:


df['Age_Diff'] = df.winner_age - df.loser_age
sns.kdeplot(df.Age_Diff)
sns.kdeplot(df.Age_Diff[df.Sets == 3])
plt.xlim([-15,15])
plt.legend(['All Matches', ' 3rd Set'])
plt.xlabel('Age Difference')
plt.title('Does the age difference kick in in the last set?')


# In[ ]:


sns.kdeplot(df.Age_Diff)
sns.kdeplot(df.Age_Diff[df['round'] == 'F'])
plt.xlim([-15,15])
plt.xlabel('Age Difference')
plt.legend(['All Matches','Finals'])
plt.title('Age Difference in the Finals')


# Difficult to say. If anything, the histogram is just not as smooth since our relevant samples are fewer. 
