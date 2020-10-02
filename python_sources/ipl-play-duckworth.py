import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns





delivdf  = pd.read_csv('../input/deliveries.csv')

df = delivdf[['match_id','inning','total_runs','over','ball','player_dismissed']]

def wicket_test(row):
    if isinstance(row['player_dismissed'], str) == True:
    #if np.isnan(row['player_dismissed']) == True:
        return 1
    else:
        return 0

df['wicket'] = df.apply(wicket_test, axis=1)
df['cum_ball'] = df.groupby(['match_id','inning'])['ball'].cumcount()
df['cumsum'] = df.groupby(['match_id','inning'])['total_runs'].cumsum()
df['cum_wickets'] = df.groupby(['match_id','inning'])['wicket'].cumsum()
#maxdf = df.groupby(['match_id','inning','over','ball'], as_index=False )['cumsum'].max()
maxdf = df.groupby(['match_id','inning'], as_index=False )['cumsum'].max()

df1 = pd.merge(df, maxdf, on=['match_id','inning'], how='outer')
df1['delta']  = df1['cumsum_y'] - df1['cumsum_x']
#print df1
df2 = pd.pivot_table(df1, values='delta', columns=['cum_wickets'], index=['over'], aggfunc=np.sum)
df3 = pd.pivot_table(df1, values='delta', columns=['cum_wickets'], index=['over'], aggfunc=np.average)
plt.interactive(False)
#print df2
df2.plot()
df3.plot()
plt.show()

#wtf won't this plot...



