# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('all.csv')
#print(df.head())
#print(len(df))
#x=df['State'].fillna(value=0)
#y=df['Post..telegraph.and.telephone.facility'].fillna(value=0)

#df['Post..telegraph.and.telephone.facility'].plot(kind="bar", colormap='jet')
#plt.bar(x,y, color="green")
#plt.title("amenities by region")
#plt.show()


sma=df[['State', 'Persons','Females','Males','X15...59.years']].groupby(df['State']).sum()
sma['Perc_M']= sma.Males*100/sma.Persons
sma['Perc_F']= sma.Females*100/sma.Persons
sma['Youth Percent']= sma['X15...59.years']*100/sma['Persons']

ax = sma[['Youth Percent']].plot(kind='bar',title="Youth percentage",legend=True)
plt.show()
ax = sma[['Perc_F','Perc_M']].plot(kind='bar',title="sex ratio percentage",legend=True)
plt.show()

label=sma.index.tolist()
p1=plt.bar(np.arange(len(sma)), sma.Perc_F, color='#dd2525', alpha=0.6)
p2=plt.bar(np.arange(len(sma)), sma.Perc_M, bottom=sma.Perc_F,color='blue', alpha=0.6)
plt.ylabel('percentage population')
plt.title('sex ratio percentage')
plt.xticks(rotation=90)
plt.xticks(np.arange(len(sma))+0.3,label)
plt.yticks(np.arange(0,100,10))
plt.legend((p1[0],p2[0]),('Females','Males'))
plt.plot(np.ones(len(sma))*50 , color='black')
plt.show()

print(df.columns.values)

smm=df[['State', 'Persons..literacy.rate','Males..Literatacy.Rate', 'Females..Literacy.Rate']].groupby(df['State']).mean()

ax = smm[['Persons..literacy.rate','Males..Literatacy.Rate', 'Females..Literacy.Rate']].plot(kind='bar',title="Literacy percentage",legend=True)
plt.show()

#Have a happy day