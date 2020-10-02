# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input/train.csv"]).decode("utf8"))

df=pd.read_csv("../input/train.csv", sep=',', lineterminator='\n')
print("hellozz")

# Create violinplot
sns.violinplot(x = df.Id, data=df.MSSubClass)

# Show the plot
plt.show()

#dynamic_list = ['MSZoning','LotFrontage','LotArea','Street','Alley','LotShape','LandContour','SalePrice']

dynamic_list=df.columns

#print(df.corr())

def  helloworld(a):
    print(a)
helloworld('melaba')    


def applyHASH(q,k):
    print(q[k].apply(hash).unique())

applyHASH(df,'MSZoning')

def printList(a):
    for i in range(len(dynamic_list)):
        if a==dynamic_list[i]:
            print(a)

printList('MSZoning')



def printList2(d,a):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    new_value=''
    for i in range(len(a)):
            print(i)
            if d[d[a].columns[i]].dtype not in numerics:
                print(d[a].columns[i])
                #print(d[d[a].columns[i]].unique())
                print(d[d[a].columns[i]].apply(hash).unique())
                new_value=d[d[a].columns[i]].apply(hash)+''+new_value
                
    df[new_column]=new_value            



printList2(df,dynamic_list)

#df['MSZoning_h']=df.LandContour.apply(hash)






print(df.corr())

'''
OverallQual
YearBuilt
YearRemodAdd
MasVnrArea
TotalBsmtSF
1stFlrSF
GrLivArea
FullBath
TotRmsAbvGrd
Fireplaces
GarageYrBlt
GarageCars
GarageArea

'''

'''
my_submission = pd.DataFrame(df.corr())
# you could use any filename. We choose submission here
my_submission.to_csv('sample_submission.csv', index=False)
print("Writing complete")
'''


