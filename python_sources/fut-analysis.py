# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

#importing data set from local storage
col_name = ['player_ID','player_name','player_extended_name','quality','revision','origin','overall','club','league','nationality','position','age','date_of_birth','height','weight','intl_rep','added_date','added_date','pace_acceleration','pace_sprint_speed','dribbling','drib_agility','drib_balance','drib_reactions','drib_ball_control','drib_dribbling','drib_composure','shooting','shoot_positioning','shoot_finishing','shoot_shot_power','shoot_long_shots','shoot_volleys','shoot_penalties','passing','pass_vision','pass_crossing','pass_free_kick','pass_short','pass_long','pass_curve','defending','def_interceptions','def_heading','def_marking','def_stand_tackle','def_slid_tackle','physicality','phys_jumping','phys_stamina','phys_strength','phys_aggression','gk_diving','gk_reflexes','gk_handling','gk_speed','gk_kicking','gk_positoning','pref_foot','att_workrate','def_workrate','weak_foot','skill_moves','cb','rb','lb','rwb','lwb','cdm','cm','rm','lm','cam','cf','rf','lf','rw','lw','st','price_ps4','price_xbox','price_pc','traits','specialties']
df = pd.read_csv(r'C:/FIFA18 - Ultimate Team players.csv',names = col_name,header=0,engine='python')

#dropping duplicate entries in rows
df = df.drop_duplicates(subset ='player_name',keep= 'first')

#setting index for unique identification of a row
df = df.set_index('player_name') 


df.info()
#printing 5 entries at the top
print(df.head())

#dropping unwanted columns 
df.drop(['player_ID','player_extended_name','quality','revision', 'origin','date_of_birth','added_date', 'added_date.1','drib_dribbling', 'drib_composure','cb', 'rb', 'lb', 'rwb',
       'lwb', 'cdm', 'cm', 'rm', 'lm', 'cam', 'cf', 'rf', 'lf', 'rw', 'lw',
       'st', 'price_ps4', 'price_xbox', 'price_pc'],axis=1,inplace=True)

print(df.columns)

#limiting the data if needed. Sometimes things go bit lengthy so its better to take into account only limited entries
#df = df.iloc[:100]       #to limit the data to 100 rows

df1 = df.sort_values('dribbling',ascending = False) #to sort out best dribblers in history

#print(df1.head())


x = df.loc['Messi','dribbling':'shooting']    #comparing any two or more players on criteria
y = df.loc['Ronaldo','dribbling':'shooting']
z = df.loc['Maradona','dribbling':'shooting']
p = df.loc['Ronaldinho','dribbling':'shooting']

plt.plot(x,label = 'Messi')
plt.plot(y,label = 'Ronaldo De Lima')
plt.plot(z,label = 'Maradona')
plt.plot(p,label = 'Ronaldinho')
plt.title("Attr vs. Rating")
plt.xlabel("Attr")
plt.ylabel("Rating")
plt.ylim(80,100)
plt.legend() 
plt.show()         ##plotting a line plot

grp = df.groupby('club').groups

#print(grp['Arsenal'])    #Viewing players playing for a particular team

A  = grp['Arsenal']

#print(A[:11])  #finding out Best 11 players at a club

is_ars = []

for club in df.club:
    if club == 'Arsenal':
        is_ars.append(True)
    else:
        is_ars.append(False)
        
Gunners = df[is_ars] 

#print(Gunners.head())    

Cules = df[df.club == 'FC Barcelona']   
    
#print(Cules.head())

X = Gunners[Gunners.position == 'ST'].max()    #popping out Arsenal's best striker
Y = Cules[Cules.position == 'ST'].max()    #popping out Barca's best striker

#print(x)
#print(y)

fig = plt.figure()

a = X.loc['pace_acceleration':'shooting']    #comparing respective players
b = Y.loc['pace_acceleration':'shooting']

plt.subplot(2, 2, 1)
plt.title("Pace vs. Rating")
plt.plot(a,label= 'Gunner')
plt.plot(b,label='Cule')

c = X.loc['shooting':'passing']    #comparing any two or more players on criteria
d = Y.loc['shooting':'passing']

plt.subplot(2, 2, 2)
plt.title("shooting vs. Rating")
plt.plot(c,label= 'Gunner' )
plt.plot(d, label='Cule')

plt.legend()
plt.show()


#some more visualization stuff on the dataset
sns.lineplot(x=Cules.head().index,y=Cules['passing'].head())
sns.scatterplot(x=Cules.head().index,y=Cules['passing'].head(),hue=Cules['overall'].head(),palette='winter_r')
plt.title("Passing Skills")
plt.show()

sns.barplot(x=Cules.head().index,y=Cules['passing'].head(),hue=Cules['position'].head())
plt.title("How position affects passing?")
plt.show()


#Now lets assemble the best team ever with best players on respective positions

def AssemblingGreatest(positions):
    dfx = df.copy()
    best11 = []
    for i in positions:
    
        best11.append([
            i,
            dfx.loc[[dfx[dfx['position'] == i]['overall'].idxmax()]].index,
            dfx[dfx['position'] == i]['overall'].max(),
            dfx.loc[[dfx[dfx['position'] == i]['overall'].idxmax()]]['nationality'].to_string(index = False),
            dfx.loc[[dfx[dfx['position'] == i]['overall'].idxmax()]]['club'].to_string(index = False),
            dfx.loc[[dfx[dfx['position'] == i]['overall'].idxmax()]]['pace_acceleration'].to_string(index = False),
            #dfx.loc[[dfx[dfx['position'] == i]['overall'].idxmax()]]['specialties'],
            dfx.loc[[dfx[dfx['position'] == i]['overall'].idxmax()]]['passing'].to_string(index = False)
 
        ])
                      
    dfx.drop(dfx[dfx['position'] == i]['overall'].idxmax(), inplace = True)
    
    return pd.DataFrame(np.array(best11), columns = ['position', 'player_name', 'overall', 'nationality', 'club', 'pace_acceleration','passing'])


posn = ['GK', 'RB', 'CB', 'LB', 'CDM', 'CM', 'CAM', 'RW', 'ST', 'LW','CF']
goats = AssemblingGreatest(posn)
print('The Greatest team with best players in respective postions')
print (goats)

#taking out a row 
goat = goats[goats.position == 'RW']
print(goat.player_name)



#This is a starter's code, if you have found any mistake or anything that could have been done more efficiently, then comment the same.
