# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import sqlite3
conn = sqlite3.connect('../input/database.sqlite')

#Allows Python code to execute PostgreSQL command in a database session. Cursors are created by the connection.cursor() method: they are bound 
#to the connection for the entire lifetime and all the commands are executed in the context of the database session wrapped by the connection.
c = conn.cursor()

#store country data in a df (Call it country) with a sql query
country=pd.read_sql("""SELECT * FROM Country""",con=conn)
print(country.head())

#read first two rows only, like .head(2)
print(pd.read_sql("""SELECT * FROM Country LIMIT 2""",con=conn))

#read rows which has country col in AFG
print(pd.read_sql("""SELECT * FROM Country WHERE CountryCode='AFG'""",con=conn))

#the count of each region under the region column in the country dataset
print(pd.read_sql("""SELECT Region, COUNT(*) AS [Count] FROM Country GROUP BY Region  ORDER BY 2 DESC""",con=conn))

#left join
print(pd.read_sql("""SELECT A.CountryCode, B.LatestPopulationCensus, B.SourceOfMostRecentIncomeAndExpenditureData, B.ShortName 
FROM (SELECT CountryCode 
      FROM Country Where CountryCode IN ('AFG','ALB','ASM','BEL')) AS A
LEFT JOIN(SELECT CountryCode,LatestPopulationCensus ,SourceOfMostRecentIncomeAndExpenditureData,ShortName 
      FROM Country Where CountryCode IN ('AFG','ARM','URY','BEL')) AS B
ON A.CountryCode = B.CountryCode""",con=conn))

#UNION ALL KEEPS DUPLICATES WHEN JOINING TWO DF WHILE UNION REMOVES THE DUPLICATES

print(pd.read_sql("""SELECT CountryCode,LatestPopulationCensus ,SourceOfMostRecentIncomeAndExpenditureData,ShortName FROM Country 
WHERE CountryCode IN ('AFG','ALB','ASM','BEL') 
UNION
SELECT CountryCode,LatestPopulationCensus ,SourceOfMostRecentIncomeAndExpenditureData,ShortName FROM Country 
WHERE CountryCode IN ('AFG','ARM','URY','BEL')""",con=conn))

print(pd.read_sql("""SELECT CountryCode,LatestPopulationCensus ,SourceOfMostRecentIncomeAndExpenditureData,ShortName FROM Country 
WHERE CountryCode IN ('AFG','ALB','ASM','BEL') 
UNION ALL
SELECT CountryCode,LatestPopulationCensus ,SourceOfMostRecentIncomeAndExpenditureData,ShortName FROM Country 
WHERE CountryCode IN ('AFG','ARM','URY','BEL')""",con=conn))

#INTERSECT RETURNS ROWS COMMON TO BOTH THE DF and EXCEPT RETURNS ROWS FROM TOP DF (MINUS THE ROWS FROM THE BOTTOM DF)
print(pd.read_sql("""SELECT CountryCode,LatestPopulationCensus ,SourceOfMostRecentIncomeAndExpenditureData,ShortName FROM Country 
WHERE CountryCode IN ('AFG','ALB','ASM','BEL') 
INTERSECT
SELECT CountryCode,LatestPopulationCensus ,SourceOfMostRecentIncomeAndExpenditureData,ShortName FROM Country 
WHERE CountryCode IN ('AFG','ARM','URY','BEL')""",con=conn))

print(pd.read_sql("""SELECT CountryCode,LatestPopulationCensus ,SourceOfMostRecentIncomeAndExpenditureData,ShortName FROM Country 
WHERE CountryCode IN ('AFG','ALB','ASM','BEL') 
EXCEPT
SELECT CountryCode,LatestPopulationCensus ,SourceOfMostRecentIncomeAndExpenditureData,ShortName FROM Country 
WHERE CountryCode IN ('AFG','ARM','URY','BEL')""",con=conn))

#---------------------------------------------------------------------------------------------------------------------------------------------

#SORTING
import pandas as pd
data = {'name': ['Ap', 'Ar', 'Na', 'Iy', 'Er'], 
        'year': [1990, 1991, 1992, 1993, 1994], 
        'Reports': [1, 2, 4, 2, 3],
        'Rating': [2, 2, 3, 3, 4]}
df = pd.DataFrame(data)
print(df)

#Sort by reports written and then ratings
print(df.sort_values(by=['Reports','Rating'])) #sorting by default is ascending #Order of the sort matters, first one given priority

#Sort by reports written and then ratings (decending)
print(df.sort_values(by=['Reports','Rating'],ascending=[False,False]))

#Sort by reports written and then ratings (reports decending, ratings ascending)
print(df.sort_values(by=['Reports','Rating'],ascending=[False,True]))


#---------------------------------------------------------------------------------------------------------------------------------------------


#TRANSPOSE
print(df)

#Add a column
df['Gender'] =['M','M','F','F','F']
df=df.set_index('name') #So that name appears on top when you transpose

#Transpose df
df=df.T
print(df)

