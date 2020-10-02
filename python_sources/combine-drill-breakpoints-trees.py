#!/usr/bin/env python
# coding: utf-8

# **Intro**
# 
# I've seen so many mock drafts touting combine numbers and key thresholds for various positions and I wanted to see if there was data to back any of them up. In order to do that, I am using combine data since 2000 to attempt to predict which round a player will end up in given nothing but their drill results and their positional group. I am not expecting great results from these models (especially for some positions), but I think the structure of the trees can be highly informative. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import graphviz
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))
pd.options.mode.chained_assignment = None
# Any results you write to the current directory are saved as output.


# **Data**

# In[ ]:


comb_dat = pd.read_csv("../input/combine_data_since_2000_PROCESSED_2018-04-26.csv")
comb_dat.head()


# The data is from combines since 2000. Unfortunately the 2018 players do not have their round labels so I am excluding them. I am treating all undrafted players as Round 8 picks to allow us to make this a regression problem.

# In[ ]:


comb_dat['Round'].fillna(8.0, inplace = True)
comb_dat['Pick'].fillna(300.0, inplace = True)


# In[ ]:


comb_dat.groupby('Pos').count()


# In[ ]:


comb_dat.mean()


# I grouped some of the positions that had smaller counts into larger groups. This is my best guess for how to categorize these into better groups. I also excluded Kickers, Punters, Fullbacks, and Long Snappers.

# In[ ]:


comb_dat.loc[:,'Pos']=comb_dat.Pos.replace({'C':'IOL','G':'IOL','OG':'IOL','OL':'IOL'
                                            ,'NT':'DT',
                                            'EDGE':'OLB',
                                            'DB':'S','SS':'S','FS':'S',
                                            'LB':'ILB'})


# In[ ]:


comb_dat=comb_dat.loc[~comb_dat.Pos.isin(['K','P','FB','LS']),:]
comb_dat=comb_dat.loc[comb_dat.Year<2018,:]


# In[ ]:


comb_dat.groupby('Pos').count()


# **Model**

#  If any drill had more than 50% of participants not participate, I removed the drill from the model. Otherwise, I filled null values with the median value for that drill. I also recorded the breakdown of the rounds for the players in the group. I then built position-specific decision trees to try show model performance and each drill's importance by position group. I tried to limit the trees from getting to large by putting a low max-depth (5) and high leaf fraction (0.04). These hyperparmaters could definitely be optimized for better performance but I'm more interested in easy to interpret trees. I optimized on MAE since their is considerable skewness for most positions (~40% of combine invitees go undrafted). I performed 10-fold CV and recorded the mean, median, min, and max MAEs for the 10 runs.  The summarized table is shown below.

# In[ ]:


i=0
pos=comb_dat.Pos.unique()

for posi in pos:

    row_dict={'Pos':[np.nan],

              'Count':[np.nan],
              
              'mean_mae':[np.nan],
              'med_mae':[np.nan],
              'max_mae':[np.nan],
              'min_mae':[],

              'Ht_imp':[np.nan],
              'Wt_imp':[np.nan],
              'Forty_imp':[np.nan],
              'Vertical_imp':[np.nan],
              'BenchReps_imp':[np.nan],
              'BroadJump_imp':[np.nan],
              'Cone_imp':[np.nan],
              'Shuttle_imp':[np.nan],

              'Ht_med':[np.nan],
              'Wt_med':[np.nan],
              'Forty_med':[np.nan],
              'Vertical_med':[np.nan],
              'BenchReps_med':[np.nan],
              'BroadJump_med':[np.nan],
              'Cone_med':[np.nan],
              'Shuttle_med':[np.nan],

              'Round1':[np.nan],
              'Round2':[np.nan],
              'Round3':[np.nan],
              'Round4':[np.nan],
              'Round5':[np.nan],
              'Round6':[np.nan],
              'Round7':[np.nan],
              'Undrafted':[np.nan]
             }



    #posi='WR'
    mod_vars=['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps',
           'BroadJump', 'Cone', 'Shuttle']
    pos_dat=comb_dat[comb_dat.Pos==posi]
    na_sum=pos_dat.loc[:,mod_vars].isna().sum()

    droppers=na_sum[na_sum>pos_dat.shape[0]/2].index
    #print(droppers)
    for drop in droppers:
        #print(drop)
        mod_vars.remove(drop)

    na_fill=pos_dat[mod_vars].quantile(.5).to_dict()
    pos_dat=pos_dat.fillna(na_fill)
    #print(pos_dat[mod_vars].isna().sum())
    pos_tree=tree.DecisionTreeRegressor(criterion='mae',max_depth=5,min_weight_fraction_leaf=.04,random_state=214)
    cv_scores=abs(cross_val_score(pos_tree,pos_dat[mod_vars],pos_dat['Round'],cv=10,scoring='neg_median_absolute_error'))

    round_count=pos_dat.groupby('Round').Player.count()
    round_count=round_count/pos_dat.shape[0]
    pos_tree=pos_tree.fit(pos_dat[mod_vars],pos_dat['Round'])

    row_dict['Pos']=[posi]
    row_dict['Count']=[pos_dat.shape[0]]

    row_dict['mean_mae']=[np.mean(cv_scores)]
    row_dict['med_mae']=[np.median(cv_scores)]
    row_dict['min_mae']=[np.min(cv_scores)]
    row_dict['max_mae']=[np.max(cv_scores)]

    row_dict.update(dict(zip([i+'_imp' for i in mod_vars],[[i] for i in pos_tree.feature_importances_])))
    row_dict.update(dict(zip([i+'_med' for i in mod_vars],[[i] for i in list(na_fill.values())])))
    row_dict.update(dict(zip(['Round1','Round2','Round3','Round4','Round5','Round6','Round7','Undrafted'],[[i] for i in round_count.values])))

    if i==0:
        row_frame=pd.DataFrame.from_dict(row_dict)
        i+=1
    else:
        row_frame=row_frame.append(pd.DataFrame.from_dict(row_dict),ignore_index=True)


# **Results**

# The table below is sorted by model performance. The approach performed best on DEs and RBs. Performance declines pretty considerably from there. I was expecting QB model performance to be horrible, but somehow it performs a bit better than the DT model.

# In[ ]:


row_frame.sort_values('mean_mae')


# In[ ]:


pos=row_frame.sort_values('mean_mae',ascending=False).Pos.values


# The function below produces model summary data from above as well as the most over-drafted and under-drafted for each position. It also renders the tree.

# In[ ]:


def render_sum(posi):
    print('{} Model Summary'.format(posi))
    display(row_frame.loc[row_frame['Pos']==posi,:])
    
    mod_vars=['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps',
               'BroadJump', 'Cone', 'Shuttle']
    pos_dat_raw=comb_dat.loc[comb_dat.Pos==posi,:]
    pos_dat=pos_dat_raw.copy()
    na_sum=pos_dat.loc[:,mod_vars].isna().sum()

    droppers=na_sum[na_sum>pos_dat.shape[0]/2].index
        #print(droppers)
    for drop in droppers:
            #print(drop)
        mod_vars.remove(drop)

    na_fill=pos_dat.loc[:,mod_vars].quantile(.5).to_dict()
    pos_dat=pos_dat.fillna(na_fill)


    pos_tree=tree.DecisionTreeRegressor(criterion='mae',max_depth=5,min_weight_fraction_leaf=.04,random_state=214)
    pos_tree.fit(pos_dat.loc[:,mod_vars],pos_dat.loc[:,'Round'])
    preds=pos_tree.predict(pos_dat.loc[:,mod_vars])
    #print(preds)
    pos_dat_raw.loc[:,'pred_Round']=preds
    pos_dat_raw.loc[:,'res']=pos_dat_raw.loc[:,'Round']-pos_dat_raw.loc[:,'pred_Round']

    showvars=['Pos','Player','Year','Team']+mod_vars+['Round','pred_Round']
    print('Top 10 Underdrafted {}:'.format(posi))
    display(pos_dat_raw.sort_values(['res','Year'],ascending=[False,False])[showvars].head(10))
    print('Top 10 Overdrafted {}:'.format(posi))
    display(pos_dat_raw.sort_values(['res','Year'],ascending=[True,False])[showvars].head(10))


    dot_data = tree.export_graphviz(pos_tree, out_file=None, 
                         feature_names=mod_vars,  
                         class_names=['1','2','3','4','5','6','7','8'],
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = graphviz.Source(dot_data)  
    return(graph)


# **Tree Deep Dive**

# **Defensive Tackles**  

# In[ ]:


g=render_sum(pos[0])
g


# None of the underdrafted players jump out, it's not like there are any diamonds in the rough that were missed for this position group.
# Also, it looks like many of the players that the model considered overdrafted are studs in the making (Jonathan Allen, Robert Nkemdiche). The tree itself seems to priortize the forty times and size. Weird that a low vertical can be an indicator of a lower pick? This model performed the worst, so it's unsurprising that the cut points seem a bit illogical. It looks like the top 40 split basically breaks the group into quick strong players and space eaters. Size becomes very important for players with bad 40 times.

# **QB**

# In[ ]:


g=render_sum(pos[1])
g


# It seems absurd to draft a QB based on combine stats. Height and Vertical(???) are the most important variables. Hopefully Kyler gets rid of that quesitonable measuring stick. Also, kinda great that it lumps in Johnny Football with Aaron Rogers as the most overdrafted players in the data set. This model is a disaster (as expected).

# **Safety**

# In[ ]:


g=render_sum(pos[2])
g


# Forty times are huge for safeties, with the top breakpoint basically deciding the odds of a player getting drafted. A sub 4.4 seems to indicate an elite athlete for the position. For those who lack elite speed,weight and the cone drill become very important. Interestingly, Thomas Davis worked out as a Safety. Converting to a linebacker makes sense for someone with his size/speed profile.

# **Offensive Tackles**

# In[ ]:




g=render_sum(pos[3])
g


# This model actually corrected identified two pretty prominent busts, Jeff Otah and Luke Joeckel, but it whiffed on Sam Baker, who was a beast for Atlanta. Interestingly, it still has forty time as a key predictor along with weight. 305 is a big break pont, with players under that threshold having a hard time getting drafted (unless they run a sub 5 forty).

# **Inside Linebacker**

# In[ ]:


g=render_sum(pos[4])
g


# No big steals in the underdrafted department. Also it whiffed on Brandon Spikes and Luke Kuechly(Kuechly seems way faster than 4.58???). Players who ran <4.76 40 and had a >32 vertical typically heard their names called the earliest. This tree seems a bit overfit, showing that the heighest drafted players get under 26 reps on bench??. Could use some pruning

# **Wide Receiver**
# 
# 

# In[ ]:


g=render_sum(pos[5])
g


# Yet again, 40 is king, this time with a way lower break point. There are actually some interesting rules here, like the idea tht you want your burner WRs(sub 4.4) to weight atleast 185. Additionally, 4.4 receivers that are 6'2 and have a 36 inch vertical are very valuable(and also freaks). When you get above 4.5 the agility drills start to get more important. Guys like Miles Austin ended up being huge steals (didn't realize he had a 40 in vertical??).

# **Outside Linebacker**

# In[ ]:


g=render_sum(pos[6])
g


# Cameron Wake turned into a beast but plays mostly DE now. Pretty accurate about Jarvis Jones and Mingo, they seemed like replaceable athletes. The 4.725 breakpoint is a huge key in wheater someone gets drafted as an OLB. Broad Jump is also important for this group. A sub 4.72 40 who weights 250 and had a 4.3 shuttle time is more than likely a Round 1 pick.

# **Tight End**

# In[ ]:


g=render_sum(pos[7])
g


# 40 times are big again, this time with higher thresholds. Additionally, the bench reps are pretty prominent for this group. Hunter Henery and Zach Ertz have surprisingly underwhelming combine numbers. Also I finally understand why the Cowboys drafted James Hanna, his combine numbers were absurd. 

# **Cornerback**

# In[ ]:


g=render_sum(pos[8])
g


# None of these names jump out. Some of the overdrafted players definitely ended up being flops. The 40 is king again, and interestingly the cone drill. Being over 200 is also important. Bigger corners have a bit more wiggle room on the 40 time, but the highest drafted guys are typically 4.4 range.

# **Interior Offensive Linemen**
# 
# 

# In[ ]:


g=render_sum(pos[9])
g


# Cone Drill is huge in evaluation of these bullies, along with weight. Again, 305 is a big break point for a player's draft position. 
# 
# 

# **Running Backs**

# In[ ]:


g=render_sum(pos[10])
g


# Forty time, weight, and broad jump are key evaluators for the runningback group. Burners that are sub 4.45 and over 213 lbs are of high value.  For guys in the 4.5 range, broad jump becomes critical. Mark Ingram and Lesen McCoy have carved out solid careers but would probably not be a Rd. 1 player if they redid their drafts. Cedric Benson was a massive overdraft.

# **Defensive Ends**

# In[ ]:


g=render_sum(pos[11])
g


# Forty and weight are again the main deciders. Justin Tuck and Jared Allen both probably belonged in their respected drafts, but it's hard to say for Tuck based on the combine since he only participated in the 40. Demarcus Lawrence had a failry underwhelming combine but has turned into a monster in the past few years. If you are 6'2"+, 260+, and run under 4.7, congrats, you're probably going in the first round. ~4.7 seems to be the magic number for a lot of these break points.

# **Conclusion**
# 
# Obviously this model has its flaws, but it also provides some interesting cut points for the type of athlete a given player is. If the apple of your eye has similar numbers to guys that end up going undrafted, you may want to wait around to see if they fall. If you see an elite athlete sliding, maybe take a chance on their upside.  

# In[ ]:




