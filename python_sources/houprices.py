# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
#df_train.info()
#print(df_train)
#print('hello')
#df_train.describe()
#df_train.columns
salelist = []
for i in range(df_train.OverallQual.min(),df_train.OverallQual.max() + 1):
    salelist.append(df_train.SalePrice[df_train.OverallQual == i].mean())
    
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  

df_train = pd.read_csv('../input/train.csv')
year = list(set(df_train['YearBuilt']))
saleprice = []
for i in year:
    saleprice.append(df_train.SalePrice[df_train['YearBuilt'] == i].mean())
width=0.3  
print(len(year))
print(len(saleprice))

#x = np.array([1,2,3,4,5,6,7,8,9,10])
#width=0.3  
#plt.bar(x,salelist,width,color='r')  
#plt.title('OverallQual and mean saleprice')  
#plt.xlabel('OverallQual')  
#plt.ylabel('mean saleprice')  
#plt.xticks([1,2,3])  
#plt.yticks(np.arange(0.0, 1.1, 0.1))  
#plt.grid(True,linestyle='-',color='0.7')  
#plt.show()  