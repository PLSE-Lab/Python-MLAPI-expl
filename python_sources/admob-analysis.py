# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

df=pd.read_csv("../input/admob.csv")
df['Date']=pd.to_datetime(df['Date'])
df['Day']=df['Date'].dt.dayofweek
df=df.sort_values(by="Date", ascending=True)
days={0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thruesday',4:'Friday',5:'satureday',6:'sunday'}
df['Day']=df['Day'].apply(lambda x:days[x])
#rpm=df.groupby('Day',axis=0).mean().reset_index()[['Day','RPM']].sort_values(by="RPM", ascending=False)
#sns.barplot(x=RPM['RPM'],y=RPM['Day'], orient='h')
#plt.title('Impression vs days')
#print(df['Impressions'].describe())

#impressions=df.groupby('Day',axis=0).mean().reset_index()[['Day','Impressions']].sort_values(by="Impressions", ascending=False)
#sns.barplot(x=impressions['Impressions'],y=impressions['Day'], orient='h')
#plt.title('Impression vs days')

earning=df.groupby('Day',axis=0).mean().reset_index()[['Day','Earnings']].sort_values(by="Earnings", ascending=False)
sns.barplot(x=earning['Earnings'],y=earning['Day'], orient='h')
plt.title('Earnings vs days')
corr=df.corr()
# Any results you write to the current directory are saved as output.