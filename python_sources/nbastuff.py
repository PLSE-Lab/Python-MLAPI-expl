# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

shot_data = pd.DataFrame()

for filename in os.listdir('../input/NBA shot log 16-17-regular season/'):
    if 'shot log ' in filename:
        shot_data = shot_data.append(pd.read_csv('../input/NBA shot log 16-17-regular season/'+filename))

# add shot distance from nearest baseline mid column    
shot_data['distance'] = pd.concat([np.sqrt( \
                        np.square(shot_data['location x']) +\
                        np.square(shot_data['location y']-250)) , 
                        np.sqrt( \
                        np.square(shot_data['location x']-940) +\
                        np.square(shot_data['location y']-250))], axis = 1
                        ).min(axis = 1)
                        

print(shot_data['distance'].describe() )
print(shot_data[['distance', 'location x', 'location y']].head(10) )

#plot
ax = shot_data['distance'].hist(bins=60)
fig = ax.get_figure()
fig.savefig('hist_of_shot_distance.pdf')



print(shot_data.columns)
    
pg_data = shot_data[shot_data['player position']=='PG']

# print( 'TOP 20 FGA - PGs ONLY' ) 
# print( pg_data['shoot player'].value_counts()[:20] )

pg_analysis = pg_data[['shoot player']].drop_duplicates()


# print ( pg_data[['location x', 'location y']].describe() )

# pg_analysis = pg_data['shoot player'].value_counts().to_frame()

def distance(x, y):
    # calc distance from mid-baseline (0,250)
    return np.sqrt(np.square(x) + np.square(y-250))
    
for index, row in pg_analysis.iterrows():
    player = row['shoot player']
    
    
    #LAYUP ATTEMPTS
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'LAYUP ATTEMPTS' ] = \
    shot_data[ (shot_data['shoot player']==player) &\
    ( shot_data['shot type'].str.contains('Dunk') ) |\
    ( shot_data['shot type'].str.contains('Layup') )]['shoot player'].count()

    #LAYUPS MADE
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'LAYUPS MADE' ] = \
    shot_data[(shot_data['shoot player']==player) &\
    (shot_data['current shot outcome']=='SCORED') &\
    ( shot_data['shot type'].str.contains('Dunk') ) |\
    ( shot_data['shot type'].str.contains('Layup') )]['shoot player'].count()
    
    #LAYUP%
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'LAYUP %' ] = \
    float( pg_analysis[pg_analysis['shoot player']==player]['LAYUPS MADE']/\
    pg_analysis[pg_analysis['shoot player']==player]['LAYUP ATTEMPTS'] )
    
    
    #FGA
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'FGA' ] = \
    shot_data[shot_data['shoot player']==player]['shoot player'].count()

    #FGM
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'FGM' ] = \
    shot_data[(shot_data['shoot player']==player) &\
    (shot_data['current shot outcome']=='SCORED')]['shoot player'].count()

    #FG%
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'FG%' ] = \
    float( pg_analysis[pg_analysis['shoot player']==player]['FGM']/\
    pg_analysis[pg_analysis['shoot player']==player]['FGA'] )
    
    #3FGA
    pg_analysis.loc[ pg_analysis['shoot player'] == player, '3FGA' ] = \
    shot_data[ (shot_data['shoot player']==player) \
    & (shot_data['points']==3 )]['shoot player'].count()

    #3FGM
    pg_analysis.loc[ pg_analysis['shoot player'] == player, '3FGM' ] = \
    shot_data[(shot_data['shoot player']==player) &\
    (shot_data['current shot outcome']=='SCORED') &\
    (shot_data['points']==3 )]['shoot player'].count()

    #3FG%
    pg_analysis.loc[ pg_analysis['shoot player'] == player, '3FG%' ] = \
    float( pg_analysis[pg_analysis['shoot player']==player]['3FGM']/\
    pg_analysis[pg_analysis['shoot player']==player]['3FGA'] )
    
    #%LAYUP/FGA
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'LAYUP/FGA' ] = \
    float( pg_analysis[pg_analysis['shoot player']==player]['LAYUP ATTEMPTS']/\
    pg_analysis[pg_analysis['shoot player']==player]['FGA'] )
    
    #3FGA/FGA
    pg_analysis.loc[ pg_analysis['shoot player'] == player, '3FGA/FGA' ] = \
    float( pg_analysis[pg_analysis['shoot player']==player]['3FGA']/\
    pg_analysis[pg_analysis['shoot player']==player]['FGA'] )
    
    
    #MIDRANGE ATTEMPTS
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'MIDA' ] = \
    shot_data[ (shot_data['shoot player']==player) &\
    ( shot_data['distance'] <= 240 ) &\
    ( shot_data['distance'] >= 200 )]['shoot player'].count()


    #MIDRANGE MADE
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'MIDM' ] = \
    shot_data[(shot_data['shoot player']==player) &\
    (shot_data['current shot outcome']=='SCORED') &\
    ( shot_data['distance'] <= 240 ) &\
    ( shot_data['distance'] >= 200 )]['shoot player'].count()

    #MIDRANGE %
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'MID%' ] = \
    float( pg_analysis[pg_analysis['shoot player']==player]['MIDM']/\
    pg_analysis[pg_analysis['shoot player']==player]['MIDA'] )
    
    #MIDRANGE ATTEMPTS/FGA
    pg_analysis.loc[ pg_analysis['shoot player'] == player, 'MIDA/FGA' ] = \
    float( pg_analysis[pg_analysis['shoot player']==player]['MIDA']/\
    pg_analysis[pg_analysis['shoot player']==player]['FGA'] )
    
    
    
pg_analysis = pg_analysis.sort_values('FG%',ascending=False)
print( pg_analysis.loc[pg_analysis['FGA']>500, \
        ['shoot player','LAYUP/FGA','MIDA/FGA','3FGA/FGA','FG%'] ])


print( pg_analysis['LAYUP %'].describe() )














