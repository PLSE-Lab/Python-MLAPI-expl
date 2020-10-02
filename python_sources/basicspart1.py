# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Import and Read my csv
data=pd.read_csv("../input/Iris.csv")
print(data.head())

print(data.info())
print(data.describe())

#I want to know the unique elements in col "Species"
uniquespecies= data['Species'].unique()
print(uniquespecies)

#I want to make sperate datasets for each kind of species

setosa= data[(data['Species'] == 'Iris-setosa')] #same as setosa= data.loc[data['Species'] == 'Iris-setosa']
print(setosa)

#Now I can drop the species column from the setosa sheet
del setosa['Species']
print(setosa.head())

#Do the same for another species
versicolor= data.loc[data['Species'] == 'Iris-versicolor']
del versicolor['Species']
print(versicolor.head())
#Give versicolor multiple conditions, slice so that only SepalWidthCm more than 5 and less than 15 are 
#accepted and PetalWidthCm between 1.3 to 1.6, call the data versicolor2
versicolor2=versicolor[(versicolor['SepalWidthCm'] > 3) & (versicolor['SepalWidthCm'] < 3.5) & (versicolor['PetalWidthCm'] > 1.3) & (versicolor['PetalWidthCm'] < 1.6)]
print(versicolor2.describe())

#Delete the rows if the value in SepalLengthCm is more than 6.5 and store the new data in data1 (work on data here)
data1=data[(data['SepalLengthCm'] <= 6.5)]
print(data1.describe())
#within data1, delete the rows where the species is 'Iris-Setosa'
data1=data1[(data1['Species'] != "Iris-setosa")]
un=data1['Species'].unique()
print(un)

#I want to combine both the datasets--> setosa and versicolor into one (concat) and call it combined
#first add a species col to each
setosa['Species'] = "Setosa"
print(setosa.head())
versicolor['Species']="Versicolor"
combined=pd.concat([setosa,versicolor],ignore_index=True)
print(combined)


print(setosa) 
setosa1=setosa[['PetalLengthCm','PetalWidthCm','Species','Id']]
#setosa1['Species'][0] = 'S'
#setosa1['Id'][4] = 89
print(setosa1.head())
setosa2=setosa.drop(['PetalLengthCm','PetalWidthCm'],axis=1)
print(setosa2.head())

#now combine setosa 1 and setosa2 on the col 'Species'
combined2=pd.merge(setosa1,setosa2,on='Id')
print(combined2.head())
print(combined2.info())
"""


#from setosa, make two seperate dataframe, 
#first one having (PetalLengthCm,  PetalWidthCm, Species)
#and second having (Id,SepalLengthCm,SepalWidthCm,Species)
print(setosa) 
setosa1=setosa[['PetalLengthCm','PetalWidthCm','Species']]
setosa11=setosa1.iloc[:5,:]
setosa11['Species'][2] = 'S'
print(setosa11)
setosa2=setosa.drop(['PetalLengthCm','PetalWidthCm'],axis=1)
setosa22=setosa2.iloc[:5,:]
setosa22['Species'][3] = 'S'
print(setosa22)


#now combine setosa 1 and setosa2 on the col 'Species'
combined2=pd.concat([setosa11,setosa22])
print(combined2)
print(combined2.info())
"""

"""#rename the combined2 columns
combined2.columns=['petallen','petalwidth','species','id','sepallen','sepalwid']
print(combined2.columns)

#add a formula: a = (petallen+petalwidth)/(sepallen+sepalwid)
combined2['a']= (combined2['petallen'] + combined2['petalwidth']) / (combined2['sepallen'] + combined2['sepalwid'])
print(combined2.head())

#What happpens when we try concat two datasets with unequal number of columns --> #gives nan for missing col dataset
a=data[(data['Species']=='Iris-versicolor')]
del a['Id']
b=data[(data['Species']=='Iris-setosa')]

ab=pd.concat([a,b],ignore_index=True)
print(ab)

#remove the missing values
ab1=ab
ab1=ab1.dropna()
ab1=ab1.reset_index(drop=True)
print(ab1.head())

#Filling missing value with mean of the column
ab2=ab
ab2.loc[(ab2.Id.isnull()),'Id'] = ab2.Id.mean()
ab2['Id']=ab2['Id'].astype(int)
print(ab2.head())"""

