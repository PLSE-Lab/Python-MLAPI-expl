# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Import the Tempclean csv
temp=pd.read_csv('../input/Tempclean.csv')
print(temp)

#col'Type' to be changed to two columns --> Min and Max
tempnew=temp.pivot(index='City',columns='Type',values='Temp')
print(tempnew.head())
#Note that the index (the column we fix, in this case 'City') becomes the index of the new dataset


#Renaming columns
print(list(tempnew))
tempnew.columns = ['Tempmin', 'Tempmax']

#Same things done again for rainfall(cm)
tempnew1=temp.pivot(index='City',columns='Type',values='Rainfall (cm)')
print(tempnew1.head())
tempnew1.columns = ['Rainmin', 'Rainmax']

print(tempnew)
print(tempnew1)

#merge both the data
final=pd.merge(tempnew, tempnew1, right_index=True, left_index=True)
print(final)

#To check retrieve values for only Delhi and Chennai
print(final.loc['Chennai':'Delhi',:])

#convert City column from index to a column in the dataset
final.reset_index(level=0, inplace=True)
print(final)
print(list(final))

#To add back city as the index
final = final.set_index(['City'])
print(final)
print(list(final))



# Import the Tempwrong csv
tempw=pd.read_csv('../input/Tempwrong.csv')
print(tempw)

#col'Type' to be changed to two columns --> Min and Max (here we are having a missing value)

##tempnew1=tempw.pivot(index='City',columns='Type',values='Temp')
##print(tempnew1.head())

#Note that the index (the column we fix, in this case 'City') becomes the index of the new dataset
#pivot won't work here since we have duplicate entries (another Jaipur Min)
#Where duplicates are present, we use pivot_table

tempnew1=tempw.pivot_table(index='City',columns='Type',values='Temp',aggfunc=np.mean) #choose what to do for the duplicate, here we are taking avg
#over here since we have missing value in the duplcate column, our average won't be different from the Jaipur Min temperature given in index [4]
print(tempnew1)


#------------------------------------------------------------------------------------------------------------------------------------------------------

#Import Gender.csv
gender=pd.read_csv('../input/Gender.csv')
print(gender)

#Melt the columns M and F to one , how to convert n columns to 1 column
final=pd.melt(gender,id_vars='Person ID',value_vars=['M','F'])
print(final)

#To make a new column gender where M=1 and F=0 (basically same as M col of the dataset)
del gender['F']
print(gender)
#Rename M to gender
gender=gender.rename(columns = {'M':'Gend'})
print(gender)

#Find number of M and F in the set
print(gender[['Gend']].groupby('Gend')['Gend'].count())

#Import Genderwrong.csv
genderw=pd.read_csv('../input/Genderwrong.csv')
print(genderw) #missing 0 value in M col, index 7

#Melt the columns M and F to one , how to convert n columns to 1 column
final1=pd.melt(genderw,id_vars='Person ID',value_vars=['M','F'])
print(final1) #melt function not affected by  missing values

#fill the data with 0
genderw['M'][7] = 0
print(genderw)


#------------------------------------------------------------------------------------------------------------------------------------------------------

#Globbing, concatanates a lot of datasets(csv files) to a single dataframe
import glob 
path ='../input/' # use your path
Files = glob.glob(path + "/A*.csv")
final = pd.DataFrame()
list1 = []
for file in Files:
    df = pd.read_csv(file,index_col=None, header=0)
    list1.append(df)
final = pd.concat(list1)

print(final)



#------------------------------------------------------------------------------------------------------------------------------------------------------

#Hierarchial Index

# Create dataframe
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'], 
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
        'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])
print(df)

#Set regiment and company as the indexes
df=df.set_index(['regiment', 'company'])
print(df)

#change the levels of index, in there, have the regiment as the inner one
df= df.swaplevel('regiment','company')
print(df)

#Summarize the data by level
print(df.sum(level='regiment'))

#slicing in hierarchial indexes
#2nd company's dragoons and nighthawks
print(df.loc[('2nd',['Nighthawks','Dragoons']),:])

#keep only regiment as the index, get company back as a column
df.reset_index(level='company', inplace=True)
print(df)
print(list(df))


#------------------------------------------------------------------------------------------------------------------------------------------------------

import sqlite3