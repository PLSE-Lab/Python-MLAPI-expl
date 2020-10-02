# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
Read_File = pd.read_csv("../input/arrests.csv")
print (Read_File.head(3))
#result= df.pivot(index= 'Border','Sector','State/Territory', columns='Product', values='Sales')
import matplotlib.pyplot as plt
#Plots in matplotlib reside within a figure object, use plt.figure to create new figure 
#fig=plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.scatter(Read_File['2000 (All Illegal Immigrants)'],Read_File['2000 (Mexicans Only)'])
#plt.title('2000 All Illegal Immigrants and Mexicans Only ')
#plt.xlabel('All Illegal Immigrants ')
#plt.ylabel('Mexicans Only')
#plt.show()
df_usa = Read_File.loc[Read_File['Border']=="United States"]
with pd.option_context('display.max_rows', None, 'display.max_columns', 5):
    print (df_usa)
#print(df_usa)
df_usa_all_immigrants= df_usa.filter(regex=r'All')
#print (df_usa_all_immigrants)
df_usa_all_immigrants.index = list(['All immigrants'])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (df_usa_all_immigrants)
df_usa_all_immigrants.columns = [int(line[0:4]) for line in df_usa_all_immigrants.columns]
with pd.option_context('display.max_rows', None, 'display.max_columns', 5):
    print (df_usa_all_immigrants)
#print (df_usa_all_immigrants.columns )
#df_usa_all_immigrants(1)
df_mexican_immigrants= df_usa.filter(regex=r'Mexican')
df_mexican_immigrants.index = list(['Mexican immigrants'])
df_mexican_immigrants.columns = [int(line[0:4]) for line in df_mexican_immigrants.columns]

#pylab.plot(df_mexican_immigrants.columns, df_usa_all_immigrants.columns).show
#pd.concat([df_mexican_immigrants,df_usa_all_immigrants]).T.plot(title='Illegal immigrants arrests')
data = {'Column 1'     : [1., 2., 3., 4.],
'Index Title'  : ["Apples", "Oranges", "Puppies", "Ducks"],
'Index Title2'  : ["Apples2", "Oranges2", "Puppies2", "Ducks2"]}
        
for kv in data.items():
    print (kv[0],'\n')      
#print(data)
#print(data.items(1))
frame_1 = pd.DataFrame({('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
('a', 'a'): {('A', 'C','D'): 3, ('A', 'B'): 4},
('a', 'c'): {('A', 'B','D'): 5, ('A', 'C'): 6},
('b', 'a'): {('A', 'C','D'): 7, ('A', 'B'): 8},
('b', 'd'): {('A', 'C','D'): 2, ('A', 'D'): 5},
('b', 'b'): {('A', 'D','D'): 9, ('A', 'B'): 10}})
with pd.option_context('display.max_rows', None, 'display.max_columns', 7):
    print (frame_1)
#series1 =pd.Series(frame_1)
#series1
#print(series1)
#for kv1 in frame_1:
   # if kv1 == ('a','a')
   # print (kv1.value())  
data_test1 = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014],
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
df_test = pd.DataFrame(data_test1, index= ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
df_test
     
with pd.option_context('display.max_rows', None, 'display.max_columns', 7):
    print (df_test)    
    
x = 'blue,red,green'
a,b,c = x.split(',')
print(a)
print(b)
print(c)

     