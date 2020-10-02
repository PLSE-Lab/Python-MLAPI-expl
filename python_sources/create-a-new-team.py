# import modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load data
player_df=pd.read_csv('..\Player.csv')
match_df=pd.read_csv('..\Match.csv')
team_df=pd.read_csv('..\Team.csv')
season_df=pd.read_csv('..\Season.csv')
ball_run_df=pd.read_csv('..\Ball_by_Ball.csv')
played_match_df=pd.read_csv('..\Player_Match.csv')

# merge dataframes
player_df_played_match_df=player_df.merge(played_match_df, on='Player_Id')
Team_and_players_df=team_df.merge(player_df_played_match_df, on='Team_Id')

# players and their respective teams
Team_and_players_df.to_csv(r'..\team and players.csv')

# captains and their respective teamns
team_player_match_count_captain=Team_and_players_df.groupby(['Team_Name','Player_Name'],as_index=False).agg({'Match_Id':'count','Is_Captain':'sum'})
print 'RESULT : Players,match played for their respective teams, number of matches as captain if any?',team_player_match_count_captain

# Players,match played for their respective teams, number of matches as captain if any?-GRAPH
#team_player_match_count_captain.hist(bins=70,orientation='horizontal',figsize=(12, 6))

# selecting only players who were captains
dummy=Team_and_players_df[Team_and_players_df['Is_Captain']==1]
dummy_1=dummy.groupby(['Team_Name', 'Player_Name'],as_index=False).agg({'Match_Id':'count','Is_Captain':('sum')})

# merging two dataframes to get captains and matches played by them as captains and as team player
dummy_2=dummy_1.merge(team_player_match_count_captain,on=['Team_Name','Player_Name'],suffixes=('_As_Captain', '_In_Total'))

# matches played by all the captains as captains and as team players
matches_as_captains_as_players=dummy_2.drop(['Is_Captain_In_Total','Is_Captain_As_Captain'],axis=1)
print 'RESULT : matches played by all the captains as captains and as team players',matches_as_captains_as_players

# matches played by all the captains as captains and as team players-GRAPH
#matches_as_captains_as_players.hist()
 
# result  : total number of players for each team from 9 seasons
total_number_player_per_team=Team_and_players_df.groupby(['Team_Id','Team_Name'])['Player_Id'].nunique()
print 'RESULT : total number of players for each team from 9 seasons', total_number_player_per_team

# result  : total number of captains for each team from 9 seasons
total_number_captains_per_team=dummy.groupby(['Team_Id','Team_Name'])['Player_Id'].nunique()
print 'RESULT : total number of CAPTAINS for each team from 9 seasons', total_number_captains_per_team

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# merging the dataframes 
t1=ball_run_df.merge(match_df)
t2=t1.merge(played_match_df)
t3=t2.merge(season_df)
t4=t3.merge(player_df)
t4.columns

# dropping the unwanted columns
final_merge=t4.drop(['Is_Umpire','Unnamed: 7','DOB','Host_Country','Second_Umpire_Id','First_Umpire_Id','Fielder_Id','City_Name','Season_Id','Match_Date'],axis=1)

# selecting only batters who are at the strikker end
t5=final_merge[final_merge['Player_Id']==final_merge['Striker_Id']]

#making ball number as same values to count them
t5.Ball_Id = t5.Ball_Id=1

# data cleaning
t5['Batsman_Scored']=t5['Batsman_Scored'].replace({" ":'0',0:'0'})
t5=t5[t5['Batsman_Scored']!= 'Do_nothing' ]
t5['Batsman_Scored']=pd.to_numeric(t5['Batsman_Scored'])
final_merge.to_csv(r'..\final_merge.csv')

# grouping the dataframe on Player name and ID to get total runs, matches, seasons and ball faced by the player
t5_grp=t5.groupby(['Player_Id','Player_Name'],as_index=False).agg({'Batsman_Scored':np.sum,'Ball_Id':'count','Match_Id':'nunique','Season_Year':'nunique'})

#calculating stike rate, AVg score, avg matches played for a season and experience factor
t5_grp['Strike_Rate']=t5_grp['Batsman_Scored']/t5_grp['Ball_Id'] * 100
t5_grp['AVG-Score']=t5_grp['Batsman_Scored']/t5_grp['Match_Id']
t5_grp['AVG_Matches_Season']=t5_grp['Match_Id']/t5_grp['Season_Year']
t5_grp['EXP_Factor']=t5_grp['Match_Id']/14

#normalising stike rate, AVg score, avg matches played for a season and experience factor
t5_grp['Strike_Rate_Norm']=(10*t5_grp['Strike_Rate'])/207.5
t5_grp['AVG-Score_Norm']=(10*t5_grp['AVG-Score'])/42.8
t5_grp['EXP_Factor_Norm']=(10*t5_grp['EXP_Factor'])/10.2

# calculating batsman factor value
t5_grp['Batsman_Factor']=(t5_grp['Strike_Rate_Norm']*0.45)+(t5_grp['AVG-Score_Norm']*0.45)+(t5_grp['EXP_Factor_Norm']*0.10)

t5_grp=np.round(t5_grp,2)
t5_grp=t5_grp.sort_values(by='Batsman_Factor',ascending=False)

t5_grp=t5_grp.rename(columns={'EXP_Factor':'Bat_EXP_Factor'})


t5_grp.to_csv(r'..\t5_grp.csv')
Top_Batters=t5_grp.head(20)
Top_Batters=Top_Batters.loc[Top_Batters['Batsman_Scored'] >= 500]
Top_Batters=Top_Batters[Top_Batters['Player_Id'] != 18]
Top_Batters=Top_Batters[Top_Batters['Player_Id'] != 41]
Top_Batters=Top_Batters[Top_Batters['Player_Id'] != 19]
Top_Batters['Nature_of_Player']='Batsman'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# filtering only left arm bowlers
LAB=final_merge.loc[final_merge['Bowling_Skill'].isin(['Left-arm fast-medium','Left-arm medium-fast'])]
LAB.to_csv(r'..\LAB.csv')

#selecting only bowlers
LAB_grp=LAB[LAB['Player_Id']==LAB['Bowler_Id']]

# data cleaning
LAB_grp['Ball_Id']=1
LAB_grp['Dissimal_Type']=LAB_grp['Dissimal_Type'].replace({'caught':1,'bowled':1,'lbw':1,'caught and bowled':1,'hit wicket':1,'stumped':1})
LAB_grp['Dissimal_Type']=LAB_grp['Dissimal_Type'].replace({'run out':0,'retired hurt':0,'obstructing the field':0})
LAB_grp['Dissimal_Type'] = LAB_grp[['Dissimal_Type']].convert_objects(convert_numeric=True)

LAB_grp['Batsman_Scored']=LAB_grp['Batsman_Scored'].replace({'Do_nothing':0})
LAB_grp['Batsman_Scored']=LAB_grp['Batsman_Scored'].astype(int)

# grouping dataframe on player name and ID to get ball bowled, runs conceeded, wkt taken and matchs played
LAB_grp_top=LAB_grp.groupby(['Player_Name','Player_Id'],as_index=False).agg({'Ball_Id':'count','Batsman_Scored':np.sum,'Dissimal_Type':'count','Match_Id':'nunique'})
LAB_grp_top=LAB_grp_top.loc[LAB_grp_top['Ball_Id'] >= 60]

# calculating Runs per ball, wickets taken per runs and experience factor of the bowler            
LAB_grp_top['WKT_Avg_Runs']=LAB_grp_top['Batsman_Scored']/LAB_grp_top['Dissimal_Type']
LAB_grp_top['Runs_Per_Ball']=LAB_grp_top['Batsman_Scored']/LAB_grp_top['Ball_Id']
LAB_grp_top['EXP_Factor']=LAB_grp_top['Match_Id']/14

# Normalising Runs per ball, wickets taken per runs and experience factor of the bowler
LAB_grp_top['WKT_Avg_Runs_Norm']=(10*10.25)/LAB_grp_top['WKT_Avg_Runs']
LAB_grp_top['Runs_Per_Ball_Norm']=(10*0.928)/LAB_grp_top['Runs_Per_Ball']
LAB_grp_top['Dissimal_Type_Norm']=(10*LAB_grp_top['Dissimal_Type'])/92
LAB_grp_top['EXP_Factor_Norm']=(10*LAB_grp_top['EXP_Factor'])/7.14

# # calculating Left arm bowler factor value
LAB_grp_top['LAB_X_Factor']=(LAB_grp_top['WKT_Avg_Runs_Norm']*0.30)+(LAB_grp_top['Runs_Per_Ball_Norm']*0.30)+(LAB_grp_top['Dissimal_Type_Norm']*0.30) +(LAB_grp_top['EXP_Factor_Norm']*0.10)

LAB_grp_top=np.round(LAB_grp_top,2)
LAB_grp_top=LAB_grp_top.sort_values(by='LAB_X_Factor',ascending=False)
LAB_grp_top=LAB_grp_top.rename(columns={'Batsman_Scored':'Runs_Conceeded','Ball_Id':'Total_Balls_Bowled','EXP_Factor':'Bowlers_EXP_Factor','EXP_Factor_Norm':'Bowlers_EXP_Factor_Norm','LAB_X_Factor':'Bowlers_Factor'})

                
LAB_grp_top.to_csv(r'..\LAB_grp_top.csv')
Top_Left_arm_Fast=LAB_grp_top.head(15)
Top_Left_arm_Fast=Top_Left_arm_Fast[Top_Left_arm_Fast['Player_Id'] != 90]
Top_Left_arm_Fast=Top_Left_arm_Fast[Top_Left_arm_Fast['Player_Id'] != 73]
Top_Left_arm_Fast=Top_Left_arm_Fast[Top_Left_arm_Fast['Player_Id'] != 15]
Top_Left_arm_Fast=Top_Left_arm_Fast[Top_Left_arm_Fast['Player_Id'] != 102]
Top_Left_arm_Fast=Top_Left_arm_Fast[Top_Left_arm_Fast['Player_Id'] != 60]
Top_Left_arm_Fast['Nature_of_Player']='Left Arm Fast Bowler'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# filtering only RIGHT arm bowlers
RAB=final_merge.loc[final_merge['Bowling_Skill'].isin(['Right-arm fast','Right-arm fast-medium','Right-arm medium','Right-arm medium-fast'])]
RAB.to_csv(r'..\RAB.csv')

#selecting only bowlers
RAB_grp=RAB[RAB['Player_Id']==RAB['Bowler_Id']]

# data cleaning
RAB_grp['Ball_Id']=1
RAB_grp['Dissimal_Type']=RAB_grp['Dissimal_Type'].replace({'caught':1,'bowled':1,'lbw':1,'caught and bowled':1,'hit wicket':1,'stumped':1})
RAB_grp['Dissimal_Type']=RAB_grp['Dissimal_Type'].replace({'run out':0,'retired hurt':0,'obstructing the field':0})
RAB_grp['Dissimal_Type'] = RAB_grp[['Dissimal_Type']].convert_objects(convert_numeric=True)

RAB_grp['Batsman_Scored']=RAB_grp['Batsman_Scored'].replace({'Do_nothing':0})
RAB_grp['Batsman_Scored'] = RAB_grp[['Batsman_Scored']].convert_objects(convert_numeric=True)
#RAB_grp['Batsman_Scored']=RAB_grp['Batsman_Scored'].astype(int)

# grouping dataframe on player name and ID to get ball bowled, runs conceeded, wkt taken and matchs played
RAB_grp_top=RAB_grp.groupby(['Player_Name','Player_Id'],as_index=False).agg({'Ball_Id':'count','Batsman_Scored':np.sum,'Dissimal_Type':'count','Match_Id':'nunique'})
RAB_grp_top=RAB_grp_top.loc[RAB_grp_top['Ball_Id'] >= 60]

# calculating Runs per ball, wickets taken per runs and experience factor of the bowler            
RAB_grp_top['WKT_Avg_Runs']=RAB_grp_top['Batsman_Scored']/RAB_grp_top['Dissimal_Type']
RAB_grp_top['Runs_Per_Ball']=RAB_grp_top['Batsman_Scored']/RAB_grp_top['Ball_Id']
RAB_grp_top['EXP_Factor']=RAB_grp_top['Match_Id']/14

# Normalising Runs per ball, wickets taken per runs and experience factor of the bowler
RAB_grp_top['WKT_Avg_Runs_Norm']=(10*11.86)/RAB_grp_top['WKT_Avg_Runs']
RAB_grp_top['Runs_Per_Ball_Norm']=(10*0.91)/RAB_grp_top['Runs_Per_Ball']
RAB_grp_top['Dissimal_Type_Norm']=(10*RAB_grp_top['Dissimal_Type'])/134
RAB_grp_top['EXP_Factor_Norm']=(10*RAB_grp_top['EXP_Factor'])/8.07

# # calculating Left arm bowler factor value
RAB_grp_top['RAB_X_Factor']=(RAB_grp_top['WKT_Avg_Runs_Norm']*0.30)+(RAB_grp_top['Runs_Per_Ball_Norm']*0.30)+(RAB_grp_top['Dissimal_Type_Norm']*0.30) +(RAB_grp_top['EXP_Factor_Norm']*0.10)

RAB_grp_top=np.round(RAB_grp_top,2)
RAB_grp_top=RAB_grp_top.sort_values(by='RAB_X_Factor',ascending=False)
RAB_grp_top=RAB_grp_top.rename(columns={'Batsman_Scored':'Runs_Conceeded','Ball_Id':'Total_Balls_Bowled','EXP_Factor':'Bowlers_EXP_Factor','EXP_Factor_Norm':'Bowlers_EXP_Factor_Norm','RAB_X_Factor':'Bowlers_Factor'})
                
RAB_grp_top.to_csv(r'..\RAB_grp_top.csv')
Top_Right_arm_Fast=RAB_grp_top.head(15)
Top_Right_arm_Fast=Top_Right_arm_Fast[Top_Right_arm_Fast['Player_Id'] != 105]
Top_Right_arm_Fast=Top_Right_arm_Fast[Top_Right_arm_Fast['Player_Id'] != 90]
Top_Right_arm_Fast=Top_Right_arm_Fast[Top_Right_arm_Fast['Player_Id'] != 151]
Top_Right_arm_Fast['Nature_of_Player']='Right Arm Fast Bowler'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# filtering only RIGHT arm bowlers
LASB=final_merge.loc[final_merge['Bowling_Skill'].isin(['Slow left-arm orthodox'])]

#selecting only bowlers
LASB_grp=LASB[LASB['Player_Id']==LASB['Bowler_Id']]

# data cleaning
LASB_grp['Ball_Id']=1
LASB_grp['Dissimal_Type']=LASB_grp['Dissimal_Type'].replace({'caught':1,'bowled':1,'lbw':1,'caught and bowled':1,'hit wicket':1,'stumped':1})
LASB_grp['Dissimal_Type']=LASB_grp['Dissimal_Type'].replace({'run out':0,'retired hurt':0,'obstructing the field':0})
LASB_grp['Dissimal_Type'] = LASB_grp[['Dissimal_Type']].convert_objects(convert_numeric=True)

LASB_grp['Batsman_Scored']=LASB_grp['Batsman_Scored'].replace({'Do_nothing':0})
LASB_grp['Batsman_Scored'] = LASB_grp[['Batsman_Scored']].convert_objects(convert_numeric=True)
#RAB_grp['Batsman_Scored']=RAB_grp['Batsman_Scored'].astype(int)

# grouping dataframe on player name and ID to get ball bowled, runs conceeded, wkt taken and matchs played
LASB_grp_top=LASB_grp.groupby(['Player_Name','Player_Id'],as_index=False).agg({'Ball_Id':'count','Batsman_Scored':np.sum,'Dissimal_Type':'count','Match_Id':'nunique'})
LASB_grp_top=LASB_grp_top.loc[LASB_grp_top['Ball_Id'] >= 60]

# calculating Runs per ball, wickets taken per runs and experience factor of the bowler            
LASB_grp_top['WKT_Avg_Runs']=LASB_grp_top['Batsman_Scored']/LASB_grp_top['Dissimal_Type']
LASB_grp_top['Runs_Per_Ball']=LASB_grp_top['Batsman_Scored']/LASB_grp_top['Ball_Id']
LASB_grp_top['EXP_Factor']=LASB_grp_top['Match_Id']/14

# Normalising Runs per ball, wickets taken per runs and experience factor of the bowler
LASB_grp_top['WKT_Avg_Runs_Norm']=(10*21.13)/LASB_grp_top['WKT_Avg_Runs']
LASB_grp_top['Runs_Per_Ball_Norm']=(10*1.015)/LASB_grp_top['Runs_Per_Ball']
LASB_grp_top['Dissimal_Type_Norm']=(10*LASB_grp_top['Dissimal_Type'])/80
LASB_grp_top['EXP_Factor_Norm']=(10*LASB_grp_top['EXP_Factor'])/7.142

# # calculating Left arm bowler factor value
LASB_grp_top['RAB_X_Factor']=(LASB_grp_top['WKT_Avg_Runs_Norm']*0.30)+(LASB_grp_top['Runs_Per_Ball_Norm']*0.30)+(LASB_grp_top['Dissimal_Type_Norm']*0.30) +(LASB_grp_top['EXP_Factor_Norm']*0.10)

LASB_grp_top=np.round(LASB_grp_top,2)
LASB_grp_top=LASB_grp_top.sort_values(by='RAB_X_Factor',ascending=False)
LASB_grp_top=LASB_grp_top.rename(columns={'Batsman_Scored':'Runs_Conceeded','Ball_Id':'Total_Balls_Bowled','EXP_Factor':'Bowlers_EXP_Factor','EXP_Factor_Norm':'Bowlers_EXP_Factor_Norm','RAB_X_Factor':'Bowlers_Factor'})
                
LASB_grp_top.to_csv(r'..\LASB_grp_top.csv')
Top_Left_arm_spin=LASB_grp_top.head(10)
Top_Left_arm_spin=Top_Left_arm_spin[Top_Left_arm_spin['Player_Id'] != 175]
Top_Left_arm_spin['Nature_of_Player']='Left Arm Orthodox Spin Bowler'


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# filtering only RIGHT arm bowlers
RASB=final_merge.loc[final_merge['Bowling_Skill'].isin(['Right-arm offbreak'])]

#selecting only bowlers
RASB_grp=RASB[RASB['Player_Id']==RASB['Bowler_Id']]

# data cleaning
RASB_grp['Ball_Id']=1
RASB_grp['Dissimal_Type']=RASB_grp['Dissimal_Type'].replace({'caught':1,'bowled':1,'lbw':1,'caught and bowled':1,'hit wicket':1,'stumped':1})
RASB_grp['Dissimal_Type']=RASB_grp['Dissimal_Type'].replace({'run out':0,'retired hurt':0,'obstructing the field':0})
RASB_grp['Dissimal_Type'] = RASB_grp[['Dissimal_Type']].convert_objects(convert_numeric=True)

RASB_grp['Batsman_Scored']=RASB_grp['Batsman_Scored'].replace({'Do_nothing':0})
RASB_grp['Batsman_Scored'] = RASB_grp[['Batsman_Scored']].convert_objects(convert_numeric=True)
#RAB_grp['Batsman_Scored']=RAB_grp['Batsman_Scored'].astype(int)

# grouping dataframe on player name and ID to get ball bowled, runs conceeded, wkt taken and matchs played
RASB_grp_top=RASB_grp.groupby(['Player_Name','Player_Id'],as_index=False).agg({'Ball_Id':'count','Batsman_Scored':np.sum,'Dissimal_Type':'count','Match_Id':'nunique'})
RASB_grp_top=RASB_grp_top.loc[RASB_grp_top['Ball_Id'] >= 60]

# calculating Runs per ball, wickets taken per runs and experience factor of the bowler            
RASB_grp_top['WKT_Avg_Runs']=RASB_grp_top['Batsman_Scored']/RASB_grp_top['Dissimal_Type']
RASB_grp_top['Runs_Per_Ball']=RASB_grp_top['Batsman_Scored']/RASB_grp_top['Ball_Id']
RASB_grp_top['EXP_Factor']=RASB_grp_top['Match_Id']/14

# Normalising Runs per ball, wickets taken per runs and experience factor of the bowler
RASB_grp_top['WKT_Avg_Runs_Norm']=(10*17.29)/RASB_grp_top['WKT_Avg_Runs']
RASB_grp_top['Runs_Per_Ball_Norm']=(10*0.970)/RASB_grp_top['Runs_Per_Ball']
RASB_grp_top['Dissimal_Type_Norm']=(10*RASB_grp_top['Dissimal_Type'])/116
RASB_grp_top['EXP_Factor_Norm']=(10*RASB_grp_top['EXP_Factor'])/8.786

# # calculating Left arm bowler factor value
RASB_grp_top['RAB_X_Factor']=(RASB_grp_top['WKT_Avg_Runs_Norm']*0.30)+(RASB_grp_top['Runs_Per_Ball_Norm']*0.30)+(RASB_grp_top['Dissimal_Type_Norm']*0.30) +(RASB_grp_top['EXP_Factor_Norm']*0.10)

RASB_grp_top=np.round(RASB_grp_top,2)
RASB_grp_top=RASB_grp_top.sort_values(by='RAB_X_Factor',ascending=False)
RASB_grp_top=RASB_grp_top.rename(columns={'Batsman_Scored':'Runs_Conceeded','Ball_Id':'Total_Balls_Bowled','EXP_Factor':'Bowlers_EXP_Factor','EXP_Factor_Norm':'Bowlers_EXP_Factor_Norm','RAB_X_Factor':'Bowlers_Factor'})
                
RASB_grp_top.to_csv(r'..\RASB_grp_top.csv')
Top_Righ_arm_spin=RASB_grp_top.head(15)
Top_Righ_arm_spin=Top_Righ_arm_spin[Top_Righ_arm_spin['Player_Id'] != 121]
Top_Righ_arm_spin=Top_Righ_arm_spin[Top_Righ_arm_spin['Player_Id'] != 104]
Top_Righ_arm_spin=Top_Righ_arm_spin[Top_Righ_arm_spin['Player_Id'] != 56]
Top_Righ_arm_spin=Top_Righ_arm_spin[Top_Righ_arm_spin['Player_Id'] != 5]
Top_Righ_arm_spin=Top_Righ_arm_spin[Top_Righ_arm_spin['Player_Id'] != 153]
Top_Righ_arm_spin=Top_Righ_arm_spin[Top_Righ_arm_spin['Player_Id'] != 104]
Top_Righ_arm_spin=Top_Righ_arm_spin[Top_Righ_arm_spin['Player_Id'] != 57]
Top_Righ_arm_spin=Top_Righ_arm_spin[Top_Righ_arm_spin['Player_Id'] != 158]
Top_Righ_arm_spin['Nature_of_Player']='Right Arm Orthodox Spin Bowler'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# filtering only RIGHT arm bowlers
LSB=final_merge.loc[final_merge['Bowling_Skill'].isin(['Legbreak','Legbreak googly','Slow left-arm chinaman'])]

#selecting only bowlers
LSB_grp=LSB[LSB['Player_Id']==LSB['Bowler_Id']]

# data cleaning
LSB_grp['Ball_Id']=1
LSB_grp['Dissimal_Type']=LSB_grp['Dissimal_Type'].replace({'caught':1,'bowled':1,'lbw':1,'caught and bowled':1,'hit wicket':1,'stumped':1})
LSB_grp['Dissimal_Type']=LSB_grp['Dissimal_Type'].replace({'run out':0,'retired hurt':0,'obstructing the field':0})
LSB_grp['Dissimal_Type'] = LSB_grp[['Dissimal_Type']].convert_objects(convert_numeric=True)

LSB_grp['Batsman_Scored']=LSB_grp['Batsman_Scored'].replace({'Do_nothing':0})
LSB_grp['Batsman_Scored'] = LSB_grp[['Batsman_Scored']].convert_objects(convert_numeric=True)
#RAB_grp['Batsman_Scored']=RAB_grp['Batsman_Scored'].astype(int)

# grouping dataframe on player name and ID to get ball bowled, runs conceeded, wkt taken and matchs played
LSB_grp_top=LSB_grp.groupby(['Player_Name','Player_Id'],as_index=False).agg({'Ball_Id':'count','Batsman_Scored':np.sum,'Dissimal_Type':'count','Match_Id':'nunique'})
LSB_grp_top=LSB_grp_top.loc[LSB_grp_top['Ball_Id'] >= 60]

# calculating Runs per ball, wickets taken per runs and experience factor of the bowler            
LSB_grp_top['WKT_Avg_Runs']=LSB_grp_top['Batsman_Scored']/LSB_grp_top['Dissimal_Type']
LSB_grp_top['Runs_Per_Ball']=LSB_grp_top['Batsman_Scored']/LSB_grp_top['Ball_Id']
LSB_grp_top['EXP_Factor']=LSB_grp_top['Match_Id']/14

# Normalising Runs per ball, wickets taken per runs and experience factor of the bowler
LSB_grp_top['WKT_Avg_Runs_Norm']=(10*9.167)/LSB_grp_top['WKT_Avg_Runs']
LSB_grp_top['Runs_Per_Ball_Norm']=(10*1.045)/LSB_grp_top['Runs_Per_Ball']
LSB_grp_top['Dissimal_Type_Norm']=(10*LSB_grp_top['Dissimal_Type'])/107
LSB_grp_top['EXP_Factor_Norm']=(10*LSB_grp_top['EXP_Factor'])/8.714

# # calculating Left arm bowler factor value
LSB_grp_top['RAB_X_Factor']=(LSB_grp_top['WKT_Avg_Runs_Norm']*0.30)+(LSB_grp_top['Runs_Per_Ball_Norm']*0.30)+(LSB_grp_top['Dissimal_Type_Norm']*0.30) +(LSB_grp_top['EXP_Factor_Norm']*0.10)

LSB_grp_top=np.round(LSB_grp_top,2)
LSB_grp_top=LSB_grp_top.sort_values(by='RAB_X_Factor',ascending=False)
LSB_grp_top=LSB_grp_top.rename(columns={'Batsman_Scored':'Runs_Conceeded','Ball_Id':'Total_Balls_Bowled','EXP_Factor':'Bowlers_EXP_Factor','EXP_Factor_Norm':'Bowlers_EXP_Factor_Norm','RAB_X_Factor':'Bowlers_Factor'})          
               
LSB_grp_top.to_csv(r'..\LSB_grp_top.csv')
#LSB_grp_top=LSB_grp_top.loc[LSB_grp_top['Dissimal_Type'] >= 20]
Top_leg_spin=LSB_grp_top.head(10)
Top_leg_spin=Top_leg_spin[Top_leg_spin['Player_Id'] != 38]
Top_leg_spin=Top_leg_spin[Top_leg_spin['Player_Id'] != 124]
Top_leg_spin['Nature_of_Player']='Leg Spin Bowler'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A1=pd.concat([LSB_grp_top,RASB_grp_top])


A2=pd.concat([A1,LASB_grp_top])


A3=pd.concat([A2,RAB_grp_top])


A4=pd.concat([A3,LAB_grp_top])
A4=A4.reset_index(drop=True)

All_rounder=A4.merge(t5_grp,on='Player_Id')
All_rounder['All_Rounder_Factor']=All_rounder['Bowlers_Factor']+All_rounder['Batsman_Factor']
All_rounder=All_rounder.sort_values(by='All_Rounder_Factor',ascending=False)
All_rounder=All_rounder.loc[All_rounder['Dissimal_Type'] >= 25.00 ]
All_rounder=All_rounder.loc[ All_rounder['Strike_Rate'] >= 130.00 ]

All_rounder.to_csv(r'..\All_rounder.csv')
All_rounder_Top=All_rounder.head(10)
All_rounder_Top=All_rounder_Top[All_rounder_Top['Player_Id'] != 56]
All_rounder_Top=All_rounder_Top[All_rounder_Top['Player_Id'] != 90]
All_rounder_Top['Nature_of_Player']='All Rounder'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
temp_Top_leg_spin=Top_leg_spin[~Top_leg_spin['Player_Id'].isin (All_rounder_Top['Player_Id'])].head(7)
temp_Top_Righ_arm_spin=Top_Righ_arm_spin[~Top_Righ_arm_spin['Player_Id'].isin (All_rounder_Top['Player_Id'])].head(5)
temp_Top_Left_arm_spin=Top_Left_arm_spin[~Top_Left_arm_spin['Player_Id'].isin (All_rounder_Top['Player_Id'])].head(5)                               
temp_Top_Batters=Top_Batters[~Top_Batters['Player_Id'].isin (All_rounder_Top['Player_Id'])].head(15)                               
temp_Top_Left_arm_Fast=Top_Left_arm_Fast[~Top_Left_arm_Fast['Player_Id'].isin (All_rounder_Top['Player_Id'])].head(7)
temp_Top_Right_arm_Fast=Top_Right_arm_Fast[~Top_Right_arm_Fast['Player_Id'].isin (All_rounder_Top['Player_Id'])].head(7)                               
                               
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All_rounder[['Bowlers_Factor','Batsman_Factor','All_Rounder_Factor']].plot.bar(x=All_rounder['Player_Name_x'],title='ALL ROUNDERS',align='center')

Top_leg_spin[['Bowlers_Factor','WKT_Avg_Runs_Norm','Runs_Per_Ball_Norm']].plot.barh(x=Top_leg_spin['Player_Name'],title='WRIST SPIN BOWLER')

Top_Righ_arm_spin[['Bowlers_Factor','WKT_Avg_Runs_Norm','Runs_Per_Ball_Norm']].plot.bar(x=Top_Righ_arm_spin['Player_Name'],title='RIGHT ARM SPIN BOWLER')

Top_Left_arm_spin[['Bowlers_Factor','WKT_Avg_Runs_Norm','Runs_Per_Ball_Norm']].plot.bar(x=Top_Left_arm_spin['Player_Name'],title='LEFT ARM SPIN BOWLER')

Top_Right_arm_Fast[['Bowlers_Factor','WKT_Avg_Runs_Norm','Runs_Per_Ball_Norm']].plot.bar(x=Top_Right_arm_Fast['Player_Name'],title='RIGHT ARM FAST BOWLER')

Top_Left_arm_Fast[['Bowlers_Factor','WKT_Avg_Runs_Norm','Runs_Per_Ball_Norm']].plot.bar(x=Top_Left_arm_Fast['Player_Name'],title='LEFT ARM FAST BOWLER')

Top_Batters[['Strike_Rate_Norm','AVG-Score_Norm','Batsman_Factor']].plot.bar(x=Top_Batters['Player_Name'],title='TOP BATSMEN')

player_df.columns

plr_cntry=player_df[['Player_Id','Player_Name','Country']]


import random

def team(AR,BAT,LF,LS,WS,RF,RS):
    ALLR_ID=All_rounder['Player_Id']
    BTTR_ID=Top_Batters['Player_Id']
    LFAST_ID=Top_Left_arm_Fast['Player_Id']
    LSPIN_ID=Top_Left_arm_spin['Player_Id']
    LEGS_ID=Top_leg_spin['Player_Id']
    RFAST_ID=Top_Right_arm_Fast['Player_Id']    
    RSPIN_ID=Top_Righ_arm_spin['Player_Id']
    
          
    te1=random.sample(ALLR_ID, AR)
    BTTR_ID=[x for x in BTTR_ID if x not in te1] 

    te2=random.sample(BTTR_ID, BAT)
    Whole_Team=te1 + te2
    LFAST_ID=[x for x in LFAST_ID if x not in Whole_Team]

    te3=random.sample(LFAST_ID,LF)
    Whole_Team=Whole_Team+te3
    LSPIN_ID=[x for x in LSPIN_ID if x not in Whole_Team]

    te4=random.sample(LSPIN_ID,LS)
    Whole_Team=Whole_Team+te4
    LEGS_ID=[x for x in LEGS_ID if x not in Whole_Team]

    te5=random.sample(LEGS_ID,WS)
    Whole_Team=Whole_Team+te5
    RFAST_ID=[x for x in RFAST_ID if x not in Whole_Team]

    te6=random.sample(RFAST_ID,RF)
    Whole_Team=Whole_Team+te6
    RSPIN_ID=[x for x in RSPIN_ID if x not in Whole_Team]

    te7=random.sample(RSPIN_ID,RF)
    Whole_Team=Whole_Team+te7
        
    return Whole_Team

temp=team(3,5,2,1,1,4,2)    
    
temp=pd.DataFrame({'Player_Id':temp})

Your_Team=plr_cntry.merge(temp,on='Player_Id')