#!/usr/bin/env python
# coding: utf-8

# # How to increase the probability of fair catch?
# 
# Fair catch eliminates tackling and reduces the need of blocking, which could reduce the likelihood of concussion.  To force a punt returner to fair catch, the gunners have to arrive where the punt returner is before he catches the ball.  Therefore, if there were fewer jammers, the gunners should be able to get to the punt returner faster. To test the idea, the null hypothesis is the number of jammers has no effect on the percentage of fair catch plays.
# 
# Here is the summary of steps to test the hypothesis. First, I determined how each play ended from NGS events.  Second, the number of jammers and gunners were computed from the positions of players for each punt plays.  Lastly, I bootstrapped the trials 100 times to compute the probability of fair catch plays given 2, 3, and 4 jammers, and performed two-sample Kolmogorov-Smirnov Tests to see whether the probabilities were significantly different. They were indeed different and the null hypothesis was rejected.  This result suggested that the fewer the jammers, the higher the likelihood of a fair catch play.  Because fair catch plays reduce the need for collision, the likelihood of a concussion is also decreased.

# ## Step 1. Determine how a play ended by NGS events 
# 
# Based on the NGS events, a punt play could end in the following ways: 'tackle', 'touchdown', 'touchback', 'fair_catch', 'punt_downed', 'out_of_bounds', 'out_of_bounds_direct' (punter punted the ball directly out of bounds),
#     and 'no_play' (e.g., false start, delay of game)
# 
# The first cell includes libraries and functions to be used later. 

# In[ ]:


from os import listdir
from scipy.stats import ks_2samp
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc as pltrc

'''
get_end_event check NGS data to see how the play ends.
- parameters:
    - thisplay, NGS data of this play for all the players
- return:
    a string that could be 'tackle', 'touchdown', 'touchback', 'fair_catch', 'punt_downed', 
    'out_of_bounds', 'out_of_bounds_direct' (punter punted the ball directly out of bounds),
    and 'no_play' (e.g., false start, delay of game)
'''
def get_end_event(thisplay): 
    end_evts = ['tackle', 'touchdown', 'fair_catch', 'punt_downed', 'out_of_bounds', 'touchback']
    evts = thisplay.Event.unique()
    
    match = set(evts).intersection(end_evts)
    if len(match) == 0:
        return 'no_play'
    else:
        match = match.pop()

        if match == 'out_of_bounds':
            if any(evts == 'punt_received'):
                return match
            else:
                return 'out_of_bounds_direct'
        else:
            return match
    
'''
check the play description if it is a no_play
'''    
def print_no_play_description(playsum, gamekey, playid):    
    
    # show play description
    play_cond = (
        (playsum.GameKey==gamekey) &
        (playsum.PlayID==playid))
    pd.set_option('display.max_colwidth', -1)
    display(playsum.loc[play_cond, ['PlayDescription']])


# Check whether the prepared data frame exists. If it does, then load it and skip data preparation steps.

# In[ ]:


derived_summary_file = Path('../working/data_fair_catch.csv')
if derived_summary_file.is_file():
    playsum = pd.read_csv(derived_summary_file)
    prepare_data = False
else:
    prepare_data = True


# Add three columns to the play_information table: 'End_Event' is the event of how a play ended; 'NumV' is the number of jammers; 'NumG' is the number of gunners. 

# In[ ]:


datapath = '../input/'
if prepare_data:
    playsum = pd.read_csv(datapath + 'play_information.csv') # play summary table
    playsum["End_Event"] = "" # how this play ends
    playsum["NumV"] = ""      # number of jammers (V)
    playsum["NumG"] = ""      # number of gunners (G)


# Looping through all the NGS files and plays to extract the event of how a play ended

# In[ ]:


if prepare_data:
    # loop through all the NGS files
    ngsfiles = [filename for filename in listdir(datapath) if filename.startswith("NGS")]
    for playfile in ngsfiles:
        print(playfile)
        ngsplays = pd.read_csv(datapath + playfile, low_memory=False)

        # get a concise list of the plays in this ngs file
        playf = ngsplays.drop_duplicates(subset=['GameKey','PlayID'], keep='first').copy()
        playf.reset_index(drop=True, inplace=True)

        # loop through all the plays
        for play_ind in range(len(playf)):
            # check ngsplays to see how this play ends
            play_cond = (
                (ngsplays.GameKey==playf.loc[play_ind,'GameKey']) &
                (ngsplays.PlayID==playf.loc[play_ind,'PlayID']))

            # get the end event
            thisplay = ngsplays[play_cond]
            endevt = get_end_event(thisplay)

            # if you'd like to print out play descriptions of no_play, uncomment below
#             if endevt == 'no_play':
#                 print_no_play_description(playsum, 
#                                           playf.loc[play_ind,'GameKey'], 
#                                           playf.loc[play_ind,'PlayID'])

            # update play summary
            play_cond = (
                (playsum.GameKey==playf.loc[play_ind,'GameKey']) &
                (playsum.PlayID==playf.loc[play_ind,'PlayID']))
            playsum.loc[play_cond, 'End_Event'] = endevt

        ngsplays = None
        playf = None


# ## Step 2. Computes the number of jammers (V) and gunners (G) from play_player_role_data.csv

# In[ ]:


if prepare_data:
    players = pd.read_csv(datapath + 'play_player_role_data.csv')

    for ind in range(playsum.shape[0]):
        gamekey = playsum.loc[ind, 'GameKey']
        playid = playsum.loc[ind, 'PlayID']

        thisplay = players[(players.GameKey==gamekey)&(players.PlayID==playid)]
        v = set(thisplay['Role']).intersection(['VLi','VLo','VRi','VRo','VR','VL'])
        g = set(thisplay['Role']).intersection(['GLi','GLo','GRi','GRo','GR','GL'])

        playsum.loc[ind,'NumV'] = len(v)
        playsum.loc[ind,'NumG'] = len(g)


# There are some plays that don't have NGS data. So, how those plays ended were unknown.  Those plays are removed here.

# In[ ]:


if prepare_data:
    playsum = playsum.drop(playsum.index[(playsum.End_Event=='')])   
    playsum.reset_index(drop=True, inplace=True)

    # save the data frame
    playsum.to_csv(derived_summary_file)

# print out how plays end
vc = playsum.End_Event.value_counts()
print(vc)


# [[OPTIONAL]] Three end events (no_play, out_of_bounds_direct, and touchback) could be excluded from the analysis because changing the number of V and G won't have an effect on these plays.  Uncomments the code if you want to exclude these plays.

# In[ ]:


# playsum = playsum.drop(playsum.index[(playsum.End_Event=='no_play')])   
# playsum = playsum.drop(playsum.index[(playsum.End_Event=='out_of_bounds_direct')])   
# playsum = playsum.drop(playsum.index[(playsum.End_Event=='touchback')])   
# playsum.reset_index(drop=True, inplace=True)


# Print out the percentage of how each play ended

# In[ ]:


vc = playsum.End_Event.value_counts()
print('percentage of plays')
print(vc/sum(vc))

prob_faircatch = vc['fair_catch']/sum(vc)


# Overall 24.6% of the plays were fair catches.  First, we would like to see whether the number of gunners (G) has an effect on the percentage of fair catch plays. However, below shows that almost all the punt plays had only 2 gunners (99.1%), which makes other cases neglectable.  

# In[ ]:


vc = playsum.NumG.value_counts()
vc/sum(vc)


# ## Step 3. Test whether the number of jammers (V) has an effect on the percentage of fair catch plays
# 
# Print out the percentage of fair catch plays given 2, 3, or 4 jammers

# In[ ]:


nvs = [2,3,4]
fc = []
for nv in nvs:
    pp = playsum.loc[(playsum.NumV==nv) & (playsum.NumG==2)]
    vc = pp.End_Event.value_counts()
    fc.append(vc['fair_catch']/sum(vc))    
    print('{} jammers, {:.2f}% ({}) plays were fair catch. 2 jammers were {:.2f} times more'.format(nv, 
                                                                      100*fc[-1], 
                                                                      vc['fair_catch'],                                                                      
                                                                      fc[0]/fc[-1]))
    


# Bootstrap plays for 100 times to compute the probability of fair catch given a different number of jammers

# In[ ]:


n = playsum.shape[0]
nrun = 100

fair_prob = {'2':[], '3':[], '4':[]}
for i in range(nrun):
    pboot_ind = np.ceil(n * np.random.rand(n))
    pboot = playsum.reindex(pboot_ind)

    for nv in nvs:
        pp = pboot.loc[(pboot.NumV==nv) & (pboot.NumG==2)]
        vc = pp.End_Event.value_counts()
        fair_prob[str(nv)].append(vc['fair_catch']/sum(vc))        


# Perform Kolmogorov-Smirnov two-sample test with Bonferroni Correction

# In[ ]:


for nv1 in [2, 3]:
    for nv2 in range(nv1+1, 5):
        value, pvalue = ks_2samp(fair_prob[str(nv1)], fair_prob[str(nv2)])
        print('{} vs {} jammers, fair catch probabilities are {:.3f} vs {:.3f}, bonferroni-corrected p-value = {:.4f}'.format(nv1,  nv2, 
                                                                             np.median(fair_prob[str(nv1)]), 
                                                                             np.median(fair_prob[str(nv2)]),                                                                              
                                                                             3*pvalue))


# Plot error bar graph

# In[ ]:


fc = []
fnvs = np.array(nvs)
fnvs = fnvs[::-1]

for nv in fnvs:
    fc.append(np.percentile(fair_prob[str(nv)], [0.005, 0.5, 0.995]))

fc = np.array(fc)    

# set font for figures
font = {'weight' : 'bold',
        'size'   : 14}
pltrc('font', **font)

# plot it
plt.bar(fnvs, fc[:,1], yerr=np.diff(fc).T, align='center', alpha=0.3, width=0.35, color='blue')
plt.xlim([1,5])
plt.xticks(nvs, nvs)
plt.ylabel('percentage of fair catch plays')
plt.xlabel('number of jammers')
plt.show()


# Plot the percentage of plays and the risk of injury for fair catch vs other plays.

# In[ ]:


review = pd.read_csv(datapath+'video_review.csv')
video = pd.read_csv(datapath+'video_footage-injury.csv')

fc = pd.merge(playsum, review, left_on=['PlayID','GameKey'], right_on=['PlayID','GameKey'])
v = fc.End_Event.value_counts()
inj_faircatch = v['fair_catch']/sum(v)

# plotting
fig, ax = plt.subplots()
index = np.array([1, 2])
bar_width = 0.2
opacity = 0.5

rects1 = plt.bar(index, [prob_faircatch, inj_faircatch], bar_width,
                alpha=opacity, color='b', label='fair_catch')

rects2 = plt.bar(index+bar_width, [1-prob_faircatch, 1-inj_faircatch], bar_width,
                alpha=opacity, color='g', label='others')

plt.xticks(index+0.5*bar_width, ('all plays', 'injuried plays'))
plt.ylabel('percentage of plays')
plt.legend(loc=2)
plt.xlim([0.5,2.75])


# Show the videos of injuries that occurred in fair catch plays 

# In[ ]:


fc = fc[(fc.End_Event=='fair_catch')]
fc = pd.merge(fc, video, left_on=['PlayID','GameKey'], right_on=['playid','gamekey'])

pd.set_option('display.max_colwidth', -1)
fc['PREVIEW LINK (5000K)']


# ## Conclusion
# This analysis suggests a way to decrease concussion for punt plays, which is to increase the probability of fair catch by allowing no more jammers than gunners.  By having two jammers against two gunners, the probability of a fair catch is 32.3%, which is **1.73 times** (32.3%/18.7%) more likely that the play ends up by a fair catch comparing to having three jammers, and **2.36 times** (32.3%/13.7%) more likely than using four jammers. In addition, note that while fair catch takes up 24.6% among all punt plays, there were only 8.1% (3 occurrences among 37 injuries) concussions that happened in a fair catch play.  From the videos (see above), two of the injuries occurred at the line of scrimmage, which could have happened in any type of plays; the other injury happened when the punt returner changed his mind to return the ball although he signaled fair catch in the first place.  If he had fair catch the ball, his injury could have been avoided.    **In summary, if the rule limits the number of jammers, it is a statistically significant way to increase the probability of fair catch, which would lead to less high-speed collisions, less concussion, and better player safety.**
# 
# 
# 
