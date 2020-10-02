# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Importing the three relevant csv files for this data set
data=pd.read_csv('../input/Book1.csv')

#We will first analyse the salary dataset
print(data.head())
print(data.info())
#Gender has one missing column
#Duplicate Name present

#Rename date of join column to remove spaces
data=data.rename(columns = {'Date of join':'Date_Join'})
print(data.columns)

#Figure out a way to fill the gender issue in the dataset
print(data[data.isnull().any(axis=1)]) # First we will get all the missing values present in the data
#The name sounds like a guy's, so we are going to assign M as the missing value (23 from index number)
data['gender'][23] = 'M'
print(data.iloc[23,:])

#Deal with duplicate names
#First we will see how many repeated/unique names we got
print(data.Name.unique())
print(data.groupby('Name')['Name'].count()) #It is seen that one name is repeated twice, we will open just these two rows and see further

print(data[(data['Name'] == 'Aaron')])
#It looks like we have two employees in the name of Aaron, we are going to change the second one's name to Aaron2 to avoid the confusion
data['Name'][2] = 'Aaron2'
print(data.iloc[1:3,:])
#It is also an option to delete the entire Aaron row(2nd index one) assuming it is a data entry error
data.drop(2, inplace=True)
print(data.head()) #here you can see than the index 2 is gone missing after deleting the row, we can reset the index as well

#Reset the index
data=data.reset_index(drop=True)
print(data.head())

#We can map gender column to M:1 and F:0
data['gender']=data['gender'].map({'M':1,'F':0}).astype(int)

#Check the range of salaries, check for outliers etc
print(data.Salary.describe())
#add bands to see
data['SalaryB']=pd.cut(data['Salary'],5)
print(data[['Salary', 'SalaryB']].groupby('SalaryB')['Salary'].count()) #count to see how many employer's getting that salary within each band
#We see that there are two employer's salaries that are above the rest, we want to replace this with the average of the rest
#(128423.12, 156516.65] -->  2

#first step is to replace those two values with 0, to get the mean of the rest
print(data[(data['Salary'] > 128423)]) #11 and 24 rows
data['Salary'][11] = 0
data['Salary'][24] = 0
print(data[(data['Salary'] > 128423)])

#now find the mean
print(data.Salary.mean()) #52925.814411764695 is the mean

#replace the salary of index row 11 and 24 with this
data['Salary'][11] = 52925.8144
data['Salary'][24] = 52925.8144


#Now we can split this dataframe into three separate datasets to practice merge/join etc
#Making three sperate data sets with a common column as ID
Salary=data[['ID','Name','Salary']]
Dept=data[['ID','Dept','Date_Join']]
Gender=data[['ID','gender']]

#Reading the first few rows of each
print(Salary.head())
print(Dept.head())
print(Gender.head())

#we will get all these three datasests into one with merging
#We will first merge Salary and Gender on the ID column (Merge two dataframes along the Gender value)
a= (pd.merge(Salary,Gender, on='ID'))
print(a)
#We will then merge a and Dept on the ID column
b=pd.merge(a,Dept, on='ID')
print(b)

#We an just the first 10 rows on the Salary dataframe in a separate dataset called a, the rest in a dataset called b
a=data.iloc[:11,:]
print(a)
b=data.iloc[11:,:]
print(b)

#Now we can concatnate a and b to get the original dataframe back
print(pd.concat([a,b],ignore_index=True)) #we are ignoring the index of the a and b dataframes

#I am going to add a new column where I will rank the employees  based on the date of joining, to see if salary has any thing to do with the period of stay
#First convert the Date_Join column type to date
data['Date_Join'] =  pd.to_datetime(data['Date_Join'])
print(data.info())
print(data.head())

print(data.Date_Join.describe())
data['DateBand']=pd.cut(data['Date_Join'],3)
print(data[['DateBand', 'Salary']].groupby('DateBand')['Salary'].mean()) #It is observed that the longer the employer stayed, higher the average pay




















