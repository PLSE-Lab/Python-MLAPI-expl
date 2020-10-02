import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap


#ST: state; HINCP: Household income; NRC: Number of children; NPF: number of people
housa=pd.read_csv("../input/pums/ss13husa.csv", usecols=['HINCP', 'NRC', 'NPF'])
housb=pd.read_csv("../input/pums/ss13husb.csv", usecols=['HINCP', 'NRC', 'NPF'])

data=pd.concat([housa,housb])

data=data.dropna(axis=0)

#calculated income per persons living in the household
data['INPP']=data['HINCP']/data['NPF']

#group by 1-10 and > 10
data['NRC']=np.where(data['NRC']>10, 11, data['NRC'])
grouped=data.groupby('NRC')

###The effect of number of children to income
plt.figure(1)
#calculate mean and standard deviation
income=grouped["INPP"].agg([np.mean,np.std])
income=income.reset_index()
labels=['%i'%j for j in range(11)]
labels.append(">10")
plt.xticks([j for j in range(12)],labels)
plt.axis([-0.5,11.5,-5000.0,85000.0])
plt.title("Mean income per person in households with different number of children")
plt.xlabel("Number of children")
plt.ylabel("Mean yearly income per person")
for i in range(len(income)):
    x=income["NRC"][i]
    y=income["mean"][i]
    yerror=income["std"][i]
    plt.scatter(x,y, color="black", s=20)
    plt.errorbar(x,y,yerr=yerror, color="black")
    
plt.show()
plt.savefig("income_per_person_w_children.png")

### Distribution of income per person
plt.figure(2)

bp=data.boxplot(column="INPP",by="NRC")
bp.get_figure().suptitle("")
plt.xticks([j for j in range(1,13)],labels)
plt.title("Distribution of income among housholds with/without children")
plt.yticks([j for j in range(0,1100000,200000)],[j for j in range(0,12,2)])
plt.xlabel("Number of children")
plt.ylabel("Yearly income per person (100.000)")
plt.axis([0.5,12.5,-10000.0,1100000.0])
plt.show()
plt.savefig("distribution_of_income.png")
