import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import pylab

print("A bar plot displaying income vs. method of transportation to work. The mean income for each method is the red box and is higher than the median in all cases because of the looooong rich-people tail. That people who use the railroad are making more than those who take the bus isn't too suprising, but I certainly didn't expect people who took the ferry to work to be so well off.")

#housing_a = '../input/pums/ss13husa.csv'
#housing_b = '../inputpums/ss13husb.csv'
population_a = '../input/pums/ss13pusa.csv'
population_b = '../input/pums/ss13pusb.csv'

#husa = pd.read_csv(housing_a)
#husb = pd.read_csv(housing_b)
#housing = DataFrame(pd.concat([husa,husb], axis=0))
popa = pd.read_csv(population_a,usecols=['ST','PUMA','SERIALNO','SPORDER','JWTR','PINCP'])
popb = pd.read_csv(population_b,usecols=['ST','PUMA','SERIALNO','SPORDER','JWTR','PINCP'])
pop_subset = DataFrame(pd.concat([popa,popb],axis = 0))

pop_subset_pivot = pop_subset.pivot_table('PINCP',columns='JWTR',index='SERIALNO',aggfunc='mean')

ax = pop_subset_pivot.plot(kind='box',showfliers=False,showmeans=True)
xlabels = ['Car','Bus','Streetcar','Subway','Railroad','Ferry','Taxi','Motorcycle','Bicycle','Shoes\n(Walking)','Slippers\n(From Home)','Other']
ax.set_xticklabels(xlabels,rotation=90)
ax.set_xlabel('Means of Transportation to Work')
ax.set_ylabel('Yearly Income(2013 $)')
plt.ylim(-20000,270000)
pylab.gcf().subplots_adjust(bottom=0.25)
plt.savefig('Income_v_WorkTransportation',dpi=200)