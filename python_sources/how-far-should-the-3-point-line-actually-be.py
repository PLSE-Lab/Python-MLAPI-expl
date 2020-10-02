#!/usr/bin/env python
# coding: utf-8

# **How far should the 3-pt line actually be?**
# 
# I just finished the book SprawlBall by Kirk Goldsberry. In it, the author talks about how the 3 point line has changed the incentive structure in basketball. The points per shot (aka the expected value of the shot) for three pointers and layups are significantly greater there than anywhere else inbetween. This got me thinking, if the NBA were to make a rule change mandating where the 3 point line should actually be, what is the "fairest" distance for them to move it to? Also, if they were to implement a 4 point line, where should that be located? 
# 
# I will also profile how this would impact Steph Curry, the most prolific 3 point shooter ever. 
# 
# The current 3 point line sits at 23.75' from the rim around the greater arc, but at the corners it is at 22'. For simplicity, I will be making it so that the arc will be a stead curve that is equidistant from the rim at all parts in my analysis.
# 
# * It should be noted that this data is from the 2014-2015 season. It is likely that there is an even larger skew towards the 3-point shot in todays basketball. 

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


# In[ ]:


import matplotlib.pyplot as plt 
import seaborn as sns


# In[ ]:


df = pd.read_csv('/kaggle/input/nba-shot-logs/shot_logs.csv')


# In[ ]:


df.head()


# Create a pivot table that shows the make % and points per shot by distance. Another pivot table for sample sizes by shot 

# In[ ]:


make_pct_dist = pd.pivot_table(df, index='SHOT_DIST', values = ['FGM','PTS'] ).reset_index()
make_pct_dist_cnt = pd.pivot_table(df, index='SHOT_DIST', values = ['FGM'], aggfunc='count' ).reset_index()


# Plot showing fgm % and points per shot by distance

# In[ ]:


make_pct_dist.plot.line(x='SHOT_DIST')

#make_pct_dist_cnt.FGM.plot.line()


# The data looks to get fairly sparse after ~ 27' I look into what range has an acceptable sample size

# In[ ]:


print(make_pct_dist_cnt.to_string())


# For shots inside of 27.3' we have > 100 samples. I will use this as the cut-off point. The FGM % attrition appears to be fairly linear outside of layups, so we can use a regression to expand this further if necessary

# In[ ]:


shot_samp = make_pct_dist[make_pct_dist.SHOT_DIST <= 27.3]


# Same plot without the messy data > 27.3

# In[ ]:


#shot_samp.FGM.plot.line()
#shot_samp.PTS.plot.line()
shot_samp.plot.line(x ='SHOT_DIST')


# Here we see the points per shot between the two point types

# In[ ]:


pd.pivot_table(df, index= 'PTS_TYPE', values='PTS')


# plot 2 point expected value vs 3 point expected value on same chart. We see a huge spike in expected value right outside of the three point line. This is disproportional to any other two point area aside from at the rim.

# In[ ]:


two_pts = pd.pivot_table(df[(df.PTS_TYPE == 2) & (df.SHOT_DIST <= 22)],index='SHOT_DIST',values='PTS').reset_index()
three_pts = pd.pivot_table(df[(df.PTS_TYPE == 3) & (df.SHOT_DIST > 22) & (df.SHOT_DIST <= 27.3)],index='SHOT_DIST',values='PTS').reset_index()


# Here I plot the ev of two point shots vs that of three point shots separated out by color

# In[ ]:


plt.plot(three_pts.SHOT_DIST,three_pts.PTS,label = 'Three_Point_EV')
plt.plot(two_pts.SHOT_DIST,two_pts.PTS,label = 'Two_Point_EV')
plt.legend()
plt.xlabel('Distance')
plt.ylabel('Points Per Shot')
plt.show()


# **How Far Should the Line Be?**
# I believe that a fair 3-point line would make it so that the average points per shot for 2-point shots are roughly equal to the points per shot of 3-point shots. When we move the line back further, we get longer and longer twos that slightly decrease the value of 2point shots (fgm % * 2pts). This effect is smaller than the dropoff in shooting percentage of 3-point shots as you go backward. We should still see the 2-point and 3-point points per shot intersect. That will be our magic number 

# The data sample is large enough that I can compare shooting percentages at various distances. In this function I can set the 3 point threshold and see how the expected value of the 2-pt & 3-pt shots compare

# In[ ]:


def three_pt_ev(data, dist):
        data.PTS_TYPE = data.SHOT_DIST.apply(lambda x: 3 if x > dist else 2)
        data.PTS = data.FGM * data.PTS_TYPE
        dout = pd.pivot_table(data,index='PTS_TYPE', values='PTS')
        print([dist,dout.PTS.iloc[0],dout.PTS.iloc[1]])
        return [dist,dout.PTS.iloc[0],dout.PTS.iloc[1]]

Here, we loop through the range of 3-point shots that we have sufficent data for. If we still do not find equilibrium before 27.2 feet, we would need to run a logistic regression to extrapolate further. 
# In[ ]:


def get_evs(df):
    shotlst = []
    for i in np.arange(24.0,27.3,.01):
        shotlst.append(three_pt_ev(df,i))
    return pd.DataFrame(shotlst, columns = ['distance','2pt_ev','3pt_ev'])


# We can see that around 25.2 feet, the balance shifts to 2pt shots being worth more. Based on this logic, this is a fair place to put the new 3 point line. 

# In[ ]:


ev_by_3pt_dist = get_evs(df)


# Here, we plot the cross over point

# In[ ]:


ev_by_3pt_dist.plot.line(x='distance')


# **The Curious Case of Steph Curry**
# My first thought would be, how would this change impact the greatest 3-point shooter ever? 

# In[ ]:


steph_curry = df[df.player_name == 'stephen curry']


# In[ ]:


steph_curry.shape


# In[ ]:


steph_threept = get_evs(steph_curry)


# While the smaple is fairly small, it appears that this change would not impact steph at all. Actually, at 25.2' he improves on his points per shot, which is astronomical. This suggests to me that he could dominate even in a league where the 3-point line was penalized further. I also think this is a legitimate case for moving the line back. If great 3-point shooters are still able to help their team, keeps them relevant without making them completely indespensable to a team. 

# In[ ]:


steph_threept.plot.line(x='distance')


# **A 4 point line????**
# 
# With this in mind. I wanted to see how far out they would have to put a 4 point line. I will be using the assumption that the 3-point line has been changed to 25.2 for this hypothetical. We will also use the same logic. How far out does a 4 point line have to be to make the expected value of the shots equal to that of 2 & 3 point shots? 
# 
# * with this it is important to note that we need to project the number of shots from ranges as well. When we were using actual data, this was not necessary because it was implicitly included. With this new method, we need to incorporate this. 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# I use a logistic regression to extrapolate out make percentages beyond 27.3 feet. 

# I split the model into two parts, one for close to rim shots (<= 4') and one for all other shots. Splitting the model produced better results than a single one #commented out below 

# In[ ]:


log_reg = LogisticRegression()
log_reg.fit(df.SHOT_DIST.values.reshape(-1, 1),df.FGM.values.reshape(-1, 1))
print(log_reg.score(df.SHOT_DIST.values.reshape(-1, 1),df.FGM.values.reshape(-1, 1)))


# In[ ]:


less4 = df[df.SHOT_DIST <= 4]
log_reg_less4 = LogisticRegression()
log_reg_less4.fit(less4.SHOT_DIST.values.reshape(-1, 1),less4.FGM.values.reshape(-1, 1))
print(log_reg_less4.score(less4.SHOT_DIST.values.reshape(-1, 1),less4.FGM.values.reshape(-1, 1)))


# In[ ]:


greater4 = df[df.SHOT_DIST > 4]
log_reg_greater4 = LogisticRegression()
log_reg_greater4.fit(greater4.SHOT_DIST.values.reshape(-1, 1),greater4.FGM.values.reshape(-1, 1))
print(log_reg_greater4.score(greater4.SHOT_DIST.values.reshape(-1, 1),greater4.FGM.values.reshape(-1, 1)))


# Quick test for accuracy

# In[ ]:


log_reg_greater4.predict_proba(np.array(24).reshape(1,-1)) 


# Very small error here, this appears stable enough that we can extrapolate this further from the rim. 

# In[ ]:


from sklearn.metrics import mean_squared_error

makepcts = make_pct_dist[(make_pct_dist.SHOT_DIST >= 23.75) & (make_pct_dist.SHOT_DIST <= 27.3)]
ypred = log_reg_greater4.predict_proba(makepcts.SHOT_DIST.values.reshape(-1,1))
ypred[:,1]
mean_squared_error(makepcts.FGM,ypred[:,1])


# Now we need to use to predict the quantity of shots from each range. We will use another piecewise regression to account for the spike in shots just outside of the 3 point line. 
# 
# This graph is interesting to look at. You can see the volume just outside the 3 point line is significantly greater than any other distance on the court. We also see an agreesive jump around 22' where the 3 point line pinches in on the corners. 
# 
# For 4 point shots, we can expect a similar spike and drop off pattern, although I would expect it to be significantly less severe. What matters here is the proporiton of 3 point shots from each distance outside of the 3 point line, not the raw number. 
# 
# Again, for this analysis, we will be ignoring the 22' 3 point line on corners. 
# 
# + number to all 3 point shots 

# In[ ]:


make_pct_dist_cnt.plot.line(x='SHOT_DIST')


# In[ ]:


make_pct_dist_cnt_3 = make_pct_dist_cnt[make_pct_dist_cnt['SHOT_DIST'] >23.75]
make_pct_dist_cnt_3.plot.line(x='SHOT_DIST')


# In[ ]:


largest_3_volume = make_pct_dist_cnt_3[make_pct_dist_cnt_3.FGM == make_pct_dist_cnt_3.FGM.max()]


# In[ ]:


distribution = make_pct_dist_cnt_3 = make_pct_dist_cnt[make_pct_dist_cnt['SHOT_DIST'] >= 24.4]


# In[ ]:


three_pts = distribution.FGM.sum()
distribution['qantity_pct'] = distribution.FGM / three_pts


# In[ ]:


distribution['cumulative_pct'] = distribution.qantity_pct.cumsum(axis=0)


# In[ ]:


distribution


# In[ ]:





# In[ ]:


total_3s = make_pct_dist_cnt_3.FGM.sum()


# In[ ]:





# In[ ]:


make_pct_dist_cnt_3['quantity_pct'] = make_pct_dist_cnt_3.FGM / total_3s


# In[ ]:


make_pct_dist_cnt_3


# In[ ]:


def Three_point_proj(fit_model,fit_model2,dist):
        for i in arange(23.75,40.0,.01):
            data.PTS = data.FGM * data.PTS_TYPE
            dout = pd.pivot_table(data,index='PTS_TYPE', values='PTS')
        print([dist,dout.PTS.iloc[0],dout.PTS.iloc[1]])
        return [dist,dout.PTS.iloc[0],dout.PTS.iloc[1]]


# In[ ]:


def four_point_proj(fit_model,fit_model2,dist):
            data.PTS_TYPE = data.SHOT_DIST.apply(lambda x: 3 if x > dist else 2)
            data.PTS = data.FGM * data.PTS_TYPE
            dout = pd.pivot_table(data,index='PTS_TYPE', values='PTS')
        print([dist,dout.PTS.iloc[0],dout.PTS.iloc[1]])
        return [dist,dout.PTS.iloc[0],dout.PTS.iloc[1]]

