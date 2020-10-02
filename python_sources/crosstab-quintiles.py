# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
"""
#CrossTab
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
        'company': ['infantry', 'infantry', 'cavalry', 'cavalry', 'infantry', 'infantry', 'cavalry', 'cavalry','infantry', 'infantry', 'cavalry', 'cavalry'], 
        'experience': ['veteran', 'rookie', 'veteran', 'rookie', 'veteran', 'rookie', 'veteran', 'rookie','veteran', 'rookie', 'veteran', 'rookie'],
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
        'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'experience', 'name', 'preTestScore', 'postTestScore'])

print(df)

print(pd.crosstab(df.regiment, df.company, margins=False))  #Margins give subtotals when True
print(pd.crosstab([df.company, df.experience], df.regiment,  margins=True)) #Crosstab done with two columns in rows
print(pd.crosstab(df.company,df.experience,values=df['preTestScore'],aggfunc=[np.mean,np.sum])) #finding mean and sum of pretestscore

#crosstab
data=pd.read_csv('../input/quintiles.csv')
print(data)

print(pd.crosstab(data.House,data.Age,values=data['Weight'],aggfunc=[np.mean]))

#Quintile
print(data.describe(percentiles=[.2, .4, .6,.8])) #stats for df

data['categories'] = pd.qcut(data['Math'], [0,.2,0.4,0.6,0.8,1],labels=[0,1,2,3,4]) #quintile cut for maths
datamath=data[['Math','categories']]
print(datamath)

#adding 18 beyond the 20th percentile
data.iloc[21:26, 6]=18     #adding 18 beyond the 20th row to see how quintiles are taken
print(data.head(30))
data['QCUTcategories'] = pd.qcut(data['Math'], [0,.2,0.4,0.6,0.8,1],labels=[0,1,2,3,4])  #qcut does not account for ties
datamath=data[['Math','QCUTcategories']]
print(datamath)

print(data.groupby('QCUTcategories')['QCUTcategories'].count()) #uneven distribution in qcut

#trying same with pd.cut
data['CUTcategories'] = pd.cut(data['Math'], [0,25,50,75,100], labels=[0,1,2,3]) #absolute cuts, you can specify the bins however u want here
print(data.groupby('CUTcategories')['CUTcategories'].count())

#pd.cut with no specific range
data['CUTcategories1'] = pd.cut(data['Math'], 5) #when u do not specify bins and u just write, say 5, it will divide the data into 5 equal bins based on values
print(data.groupby('CUTcategories1')['CUTcategories1'].count()) #same as qcut

#Rank with options
import scipy
from scipy import stats

#calculates the percentile of a given score
print(scipy.stats.percentileofscore(data.Math, 18, kind='rank'))
print(scipy.stats.percentileofscore(data.Math, 18, kind='weak'))   
print(scipy.stats.percentileofscore(data.Math, 18, kind='strict'))
print(scipy.stats.percentileofscore(data.Math, 18, kind='mean'))
    
#calculates the score under a given percentile
print(scipy.stats.scoreatpercentile(data.Math, 20, interpolation_method='fraction', axis=None))
print(scipy.stats.scoreatpercentile(data.Math, 20, interpolation_method='lower', axis=None))
print(scipy.stats.scoreatpercentile(data.Math, 20, interpolation_method='higher', axis=None))

#For each score in data.Math, find the percentile it lies under
percentile=[]
for row in data['Math']:
    percentile.append(scipy.stats.percentileofscore(data.Math, row, kind='rank'))   #can be rank/weak/strict/mean since we have ties with 18 repeating
 
data['percentile']=percentile

print(data[['Math','percentile']])
"""

#Dealing with duplicates in data
data=pd.read_csv('../input/statewithduplicates.csv')
print(data)    

print(data.duplicated()) #All false, it means no two rows are entirely same

print(data.duplicated(['State'])) #We already see that there are only two country data, we can check for duplicates in combinations for state and city
data['DupliOfStateCity'] = data.duplicated(['State', 'City'])
print(data)

print(data['DupliOfStateCity'].sum()) #Two data is repeating in state+city level, that can either mean one city has two states of the same name, or just an error
#to do this we can check the population and literacy data provided to see if even they are same/close (indicating an error)

print(data.loc[data.DupliOfStateCity == True]) #Diu and ampara repeat, so we get only those rows

print(data.loc[data['City'].isin(['Diu','Ampara'])]) 
#The population and Literacy rate in both the Diu's are very differnt (so we going to assume there are actually two Diu's in MP

#The population and Literacy rate in both the Ampara's are very close, this might be an error or a mistake of having two different
#data source for same city,so we going to combine them, avg
print(data.groupby(['City','State']).mean().reset_index())

"""Ampara         Eastern                 2.995             62 """ #Use this avg and drop the extra Ampara column
data=data.drop(data.index[9])
data=data.reset_index()
print(data)


#Change Ampara pop and lit rate details to the average we found previously
print(data.iloc[8, 4:6])
data.iloc[8,4]=2.995
data.iloc[8,5]=62
print(data)

#Remove Duplicates at state level (to observe how python removes them)

#1. At entire row level
##print(data.duplicated()) #All are false, so at entire row level,w e have unique data
##print(data)
#At state column level (run the codes in turns, else the first code (a) run together with second code (b) won't make any difference as (a) will remove the duplicates)
#a: 
##data=data.drop_duplicates(['State'])     #By default, the first one is kept, any repetitions after that is removed, also sorting not needed here
##print(data)
#b:
##data=data.drop_duplicates(['State'], keep='last') #you can specify if you want to keep the first or last in a repetition
##print(data)

#Taking average of the two repeating columns (for the numerical part) and hence removing duplicates that way
#drop the duplicates column
del data['DupliOfStateCity']
##print(data)
#Only repeating column in --> country, state, city level is --> India-MP-->Diu, with different values of pop and literacy, we will make it one row with avg
##data=data.groupby(['Country','State','City']).mean()
print(data)


#Remove country and state columns, you have data in city level, how will python help you distignuish each repeating cities uniquely from one another
main=data.drop(['index','Population(millions)','Literacy Rate','Country'],axis=1)
main.iloc[9,1]='Batticaloa'
data= data.drop(['index','Country'],axis=1)
##print(data)

#We have number of cities repeating: Hyderabad, Diu, Batticaloa
##data.City = data.City.where(~data.City.duplicated(), data.City + '_2')
##print(data)

#If we have more than two cities of the same name, how to make them unique
data.iloc[9,1]='Batticaloa' #Change to have an example with three repeating cities
print(data)
main=main.drop_duplicates()
print(main)

#merge (one to many) left side is 'main', right side is 'data'
final=pd.merge(main, data, on=['State','City'],how='left')
print(final)





"""
#using for loop to individualise the city section
#create a col concat two col
###final["temp"] = final["State"].map(str) + main["City"].map(str)    #THIS DOES NOT WORK WHEN DUPLICATE PRESENT IN JOINING COLUMNS

final['temp'] = final[['State', 'City']].apply(lambda x: ''.join(x), axis=1)
print(final)


i=0
for row in final.City:
    if final.duplicated('temp') == True:
        j=i+1
        final.City = final.City + j.map(str)
        break
    else:
        final.City=final.City
print(final)

j=0
for index, row in final.iterrows():
    if (row['State'], row['City']).duplicated()==True:
        j=j+1
        final.iloc[index,1]=final.iloc[index,1]+ "_"+ j
    else:
        final.iloc[index,1]=final.iloc[index,1]

             State        City  Population(millions)  Literacy Rate
0   Andhra Pradesh   Hyderabad                 9.500             36
1   Andhra Pradesh      Guntur                 5.000             47
2            Assam    Guwahati                 4.500             31
3            Assam      Jorhat                 2.500             27
4     Chhattisgarh    Bilaspur                 5.200             31
5   Madhya Pradesh         Diu                 1.500             31
6   Madhya Pradesh         Diu                 3.000             55
7   Madhya Pradesh   Hyderabad                 6.500             44
8          Eastern      Ampara                 2.995             62
9    North Central  Batticaloa                 1.000             56
10         Western  Batticaloa                 2.000             50
11        Southern  Batticaloa                 3.000             63


"""










