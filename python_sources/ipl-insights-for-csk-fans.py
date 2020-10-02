#!/usr/bin/env python
# coding: utf-8

# In[2]:


#input the packages required 
import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import random

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#get in matches 
data_path = "../input/"
matches_info_df = pd.read_csv(data_path+"matches.csv") 
deliveries = pd.read_csv(data_path+"deliveries.csv")


# Myself being a die hard Chennai Super Kings fan since its launch , My exploration is more biased on csk players and csk match venues . 

# CSK Players list 
# 
# 1. MS Dhoni 
# 2. Ravindra Jadeja 
# 3. Suresh Raina 
# 4. Kedar Jadhav  
# 5. Dwayne Bravo 
# 6. Karn Sharma 
# 7. Shane Watson
# 8. Shardul Thakur
# 9. Ambati Rayudu 
# 10. Murali Vijay
# 11. Harbhajan Singh
# 12. Faf Du Plessis 
# 13. Mark Wood
# 14. Sam Billings
# 15. Imran Tahir 
# 16. Deepak Chahar
# 17. Mitchell Santner
# 18. Lungisani Ngidi 
# 19. Asif K M
# 20. N Jagadeesan
# 21. Kanishk Seth 
# 22. Monu Singh
# 23. Dhruv Shorey
# 24. Kshitiz Sharma
# 25. Chaitanya Bishnoi
# 
# ~Tentative Fixtures ~~~~
# 
# Match 1 :
# 
# Mumbai Indians vs Chennai Super Kings
# Wankhede Stadium, Mumbai
# 
# Match 2:
# 
# Chennai Super Kings v Kolkata Knight Riders
# MA Chidambaram Stadium, Chennai
# 
# Match 3:
# 
# Kings XI Punjab v Chennai Super Kings
# Holkar Cricket Stadium, Indore
# 
# 
# Match 4:
# 
# Chennai Super Kings v Rajasthan Royals
# MA Chidambaram Stadium, Chennai
# 
# Match 5:
# 
# Sunrisers Hyderabad v Chennai Super Kings
# Rajiv Gandhi Intl. Cricket Stadium, Hyderabad
# 
# Match 6:
# 
# Royal Challengers Bangalore v Chennai Super Kings
# M. Chinnaswamy Stadium, Bengaluru
# 
# Match 7:
# 
# Chennai Super Kings v Mumbai Indians
# M. A. Chidambaram Stadium, Chennai
# 
# Match 8:
# 
# Chennai Super Kings v Delhi Daredevils
# M. A. Chidambaram Stadium, Chennai
# 
# Match 9:
# 
# Kolkata Knight Riders v Chennai Super Kings
# Eden Gardens, Kolkata
# 
# Match 10:
# 
# Chennai Super Kings v Royal Challengers Bangalore
# M. A. Chidambaram Stadium, Chennai
# 
# 
# Match 11:
# 
# Rajasthan Royals v Chennai Super Kings
# Sawai Mansingh Stadium, Jaipur
# 
# Match 12:
# 
# Chennai Super Kings v Sunrisers Hyderabad
# M. A. Chidambaram Stadium, Chennai
# 
# Match 13:
# 
# Delhi Daredevils v Chennai Super Kings
# Feroz Shah Kotla Ground, Delhi
# 
# Match 14:
# 
# Chennai Super Kings v Kings XI Punjab
# MA Chidambaram Stadium, Chennai

# ---Some stats on the stadiums-----
# 
# Mumbai , Wankhade - redsoil , ball may bounce , low spin , batting pitch
# 
# Indore , Neutral , flat pitch  
# 
# Hyderabad - Rajiv G , flat pitch
# 
# Bengaluru - small ground , batting pitch , fast bowlwers advantage
# 
# Kolkata - batting pitch , spin can work out based in moisture
# 
# Jaipur - Nuetral , flat pitch
# 
# Delhi - Mostly batting , but anything can happen !!
# 
# Chennai - 7 - Spinners advantageous , harbajan already has record of 42 wickets 3rd highest(may be tats y he is picked) ,      batting needs technical strong players .
# 
# 8 matches to win out of 14 for qualifying playoffs

# In[4]:


csk_team = ['MS Dhoni', 
'RA Jadeja',
'SK Raina',
'Kedar Jadhav',
'DJ Bravo',
'KV Sharma',
'SR Watson',
'Shardul Thakur',
'AT Rayudu', 
'M Vijay',
'Harbhajan Singh',
'F du Plessis',
'Mark Wood',
'Sam Billings',
'Imran Tahir',
'Deepak Chahar',
'Mitchell Santner',
'Lungisani Ngidi',
'Asif K M',
'N Jagadeesan',
'Kanishk Seth',
'Monu Singh',
'Dhruv Shorey',
'Kshitiz Sharma',
'Chaitanya Bishnoi']


# In[53]:


#csk match venues
csk_venues =  ['Wankhede Stadium','MA Chidambaram Stadium, Chepauk','Holkar Cricket Stadium',"Rajiv Gandhi International Stadium, Uppal",'M Chinnaswamy Stadium','Eden Gardens','Sawai Mansingh Stadium','Feroz Shah Kotla']


# In[5]:



#matches_info_df.season.value_counts()
matches_info_df.drop(['umpire3'],axis = 1,inplace = True)
matches_info_df.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW','RPS'],inplace=True)

chennai_stad_df = matches_info_df[matches_info_df.city == 'Chennai']
chennai_stad_df.reset_index(drop=True,inplace= True)


# In[6]:


chennai_stad_df.groupby('season').count()['id'].plot(kind= 'Bar', figsize=(10,5))
plt.ylabel('No. of matches')
plt.xlabel('Season Years')
plt.title('No of matches played in Chennai per season')
plt.show()


# In[7]:


chennai_stad_df.groupby('player_of_match').count()['id'].plot(kind= 'Bar', figsize=(19,5),color = 'g')
plt.ylabel('No. of matches')
plt.xlabel('Players')
plt.title('Man of the match title winner in chennai')
plt.show()


# In[8]:


win_bins = [0,10,20,30,40,50,60,70,80,90,100]
pd.cut(chennai_stad_df.win_by_runs , bins = win_bins).value_counts(sort = False).plot(color = 'grey',kind = 'Bar', figsize = (7,5))
plt.xlabel("Runs range")
plt.title("Runs differece matches won i.e by first innings batting")
plt.show()
#chennai_stad_df.win_by_runs.describe(


# In[9]:


win_wic_bins = [0,1,2,3,4,5,6,7,8,9,10]
pd.cut(chennai_stad_df.win_by_wickets , bins = win_wic_bins).value_counts(sort = False).plot(color = 'orange',kind = 'Bar', figsize = (7,5))
plt.xlabel("Wickets range")
plt.title("Wickets differece matches won i.e by second innings batting")
plt.show()
#chennai_stad_df.win_by_runs.describe(


# In[196]:


#if toss winner wins Toss-Win-Win(TWW) & Toss-Win-Loss(TWL)
chennai_stad_df['toss_match_win'] = np.where(chennai_stad_df['toss_winner'] == chennai_stad_df['winner'], 'T-W-W','T-W-L')


# In[12]:


toss_df = chennai_stad_df.toss_match_win.value_counts()
toss_dec = chennai_stad_df.toss_decision.value_counts()


# In[13]:


#toss winner wins the match 
labels = (np.array(['Yes','No']))
sizes = (np.array(toss_df/toss_df.sum()*100))
colors = ['lightblue', 'red']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss winnner wins the match in chennai")
plt.show()


# In[14]:


labels = (np.array(toss_dec.index))
sizes = (np.array(toss_dec/toss_dec.sum()*100))
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss decision percentage")
plt.show()


# In[198]:


plt.figure(figsize=(12,6))
sns.countplot(x='toss_winner', hue='toss_match_win', data=chennai_stad_df)
plt.xticks(rotation='vertical')
plt.title('Toss winning team wins the match in chennai venue')
plt.show()


# In[170]:


del_match_bat_report = pd.DataFrame(deliveries.groupby(['match_id','over','ball','batsman']).sum()['total_runs']).reset_index().merge(matches_info_df , how = 'inner' , left_on=['match_id'], right_on=['id'])
del_mt_bt_wickt_report = pd.DataFrame(deliveries.dropna(subset=['dismissal_kind']).groupby(['match_id','batsman','dismissal_kind']).count()).reset_index().merge(matches_info_df , how = 'inner' , left_on=['match_id'], right_on=['id'])


# In[214]:


def get_bats_stats(batsman_name):
    temp_df = del_match_bat_report[del_match_bat_report.batsman == batsman_name]
    temp_df_1 = del_mt_bt_wickt_report[del_mt_bt_wickt_report.batsman == batsman_name]
    #del_match_over_report.merge(matches_info_df , how = 'inner' , left_on=['match_id'], right_on=['id'])
    temp_df['over_lab'] = temp_df.over.apply(lambda x : 'POW-P' if x < 7 else 'NORM-P')
    
    
    colors = ['r','g','b','y','g','orange','grey']

    for ven in csk_venues :
        if temp_df[temp_df.venue == ven].match_id.count() > 1:
            temp_df[temp_df.venue == ven].groupby('over_lab').mean()['total_runs'].plot(kind = 'bar', color = random.sample(colors,1))
            plt.title('Venue played: '+ ven + ', player name: ' + batsman_name +', runs scored ' )
            plt.ylabel('avg. runs per over')
            plt.xlabel('field type power-play and normal field type')
            plt.show()
        else:
            print('NO match played by %s at this venue %s' %(batsman_name,ven))
    
    #plt.subplot()
        if temp_df_1[temp_df_1.venue == ven].match_id.count() > 1:
            # calc OMRWE
            fours = temp_df[(temp_df.venue == ven) & (temp_df.total_runs == 4)].match_id.count()
            sixs = temp_df[(temp_df.venue == ven) & (temp_df.total_runs == 6)].match_id.count()
            sampl = pd.DataFrame(temp_df[(temp_df.venue == ven)].groupby('match_id').sum()[['total_runs','ball']])
        
        
            fifts = sampl[(sampl.total_runs >=50) & (sampl.total_runs <100)].total_runs.count()
            century = sampl[(sampl.total_runs >=100) & (sampl.total_runs <200)].total_runs.count()
            battingavg = sampl.total_runs.sum() / sampl.total_runs.count()
            balls_faced = sampl.ball.sum()
            strike_rate = (100 * sampl.total_runs.sum())/balls_faced
    
            print('\n')
            print('BATTING ANALYSIS FOR AT VENUE %s, Total : fours-%d|sixs-%d|fifty-%d|century-%d|batting avg-%.2f|balls faced-%d|StrikeRate -%.2f' 
                  %(ven,fours,sixs,fifts,century,battingavg,balls_faced,strike_rate))
            print('\n')
            sns.countplot(x='dismissal_kind', data=temp_df_1[temp_df_1.venue == ven])
            plt.title('Venue played: '+ ven + ', player name: ' + batsman_name +', dismissal type')
            plt.xticks(rotation='vertical')
            plt.show()
        else :
            print('NO Dismissals of the batsman at this venue %s' %ven)


# # MS Dhoni batting stats

# In[209]:


get_bats_stats('MS Dhoni')


# # Raina batting stats

# In[210]:


get_bats_stats('SK Raina')


# In[121]:


del_match_over_report = pd.DataFrame(deliveries.groupby(['match_id','over','bowler']).sum()['total_runs']).reset_index().merge(matches_info_df , how = 'inner' , left_on=['match_id'], right_on=['id'])
del_mt_ov_wickt_report = pd.DataFrame(deliveries.dropna(subset=['dismissal_kind']).groupby(['match_id','bowler','dismissal_kind']).count()).reset_index().merge(matches_info_df , how = 'inner' , left_on=['match_id'], right_on=['id'])


# In[211]:


def get_bowler_stats(bowler_name):
    temp_df = del_match_over_report[del_match_over_report.bowler == bowler_name]
    temp_df_1 = del_mt_ov_wickt_report[del_mt_ov_wickt_report.bowler == bowler_name]
    #del_match_over_report.merge(matches_info_df , how = 'inner' , left_on=['match_id'], right_on=['id'])

    colors = ['r','g','b','y','g','orange','grey']

    for ven in csk_venues :
        if temp_df[temp_df.venue == ven].match_id.count() > 1:
            temp_df[temp_df.venue == ven].groupby('over').mean()['total_runs'].plot(kind = 'bar', color = random.sample(colors,1))
            plt.title('Venue played: '+ ven + ', player name: ' + bowler_name +', average runs given' )
            plt.ylabel('avg. runs per over')
            plt.xlabel('over number in the match')
            plt.show()
        else:
            print('NO match played by %s at this venue %s' %(bowler_name,ven))
    
    # calc OMRWE
        
    
    #plt.subplot()
        if temp_df_1[temp_df_1.venue == ven].match_id.count() > 1:
            over = temp_df[temp_df.venue == ven].over.count()
            maiden = temp_df[(temp_df.venue == ven) & (temp_df.total_runs == 0)].over.count()
            runs = temp_df[temp_df.venue == ven].total_runs.sum()
            wickets = temp_df_1[(temp_df_1.venue == ven)].dismissal_kind.count()
            economy = runs/over
    
            print('\n')
            print('BOWLING ANALYSIS FOR AT VENUE %s, Total : Over-%d|Maiden-%d|Runs-%d|Wickets-%d|Economy-%.2f' %(ven,over,maiden,runs,wickets,economy))
            print('\n')
            sns.countplot(x='dismissal_kind', data=temp_df_1[temp_df_1.venue == ven])
            plt.title('Venue played: '+ ven + ', player name: ' + bowler_name +', dismissal type')
            plt.xticks(rotation='vertical')
            plt.show()
        else :
            print('NO Dismissals by the bowler at this venue %s' %ven)
    #plt.show()
    
    


# # Harbajan Singh bowling stats

# In[212]:


get_bowler_stats('Harbhajan Singh')


# # Imran Tahir bowling stats

# In[213]:


get_bowler_stats('Imran Tahir')


# In[ ]:





# In[ ]:




