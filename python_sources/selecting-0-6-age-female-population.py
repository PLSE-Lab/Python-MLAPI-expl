import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as df
import math
e=pd.read_csv('../input/cities_r2.csv')
data=df(e)
pt=data['population_total']
pm=data['population_male']
pf=data['population_female']
zt=data['0-6_population_total']
zm=data['0-6_population_male']
zf=data['0-6_population_female']
spt=sum(pt[:])
spm=sum(pm[:])
spf=sum(pf[:])
szt=sum(zt[:])
szm=sum(zm[:])
szf=sum(zf[:])
noz=spt-szt
nozm=spm-szm
nozf=spf-szf
labels='population_male','population_femle','0-6 male population','0-6 female population'
p1=float(nozm*100)/spt
p2=float(nozf*100)/spt
p3=float(szm*100)/spt
p4=float(szf*100)/spt
sizes=[p1,p2,p3,p4]
explode=(0.0,0.1,0.0,0.2)
plt.pie(sizes,explode=explode,labels=labels,colors=['b','r','g','b'],autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()