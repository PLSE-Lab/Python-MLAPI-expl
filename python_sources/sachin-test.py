# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

ty = pd.read_csv("../input/train.csv")


ty.head()
ty['AgeRange'] = ty['Age'] - ty['Age']%10
ty['CabinRange'] = ty['Cabin'].str[:1]

number=LabelEncoder()
ty['Sex']=number.fit_transform(ty['Sex'].astype('str'))
ty['CabinRange']=number.fit_transform(ty['CabinRange'].astype('str'))
print((ty['Cabin']))
print(ty.head())
print(ty['Cabin'])

#group1 = ty[['Survived','Sex','Pclass','AgeRange','CabinRange' ]].groupby(['Sex','Pclass','AgeRange','CabinRange']).sum()
#print(group1[group1['Survived']>3])

#group2 = ty[['CabinRange','Survived']].groupby(['CabinRange']).sum()
#print(group1[group1['Survived']>10])

#Survival = A + A1*AgeRange+ A2*Pclass + A3*Sex + A4*CabinRange
#0 = A +A1*1+A2*3+A3*20 +A4*0
#1 = A +A1*2+A2*1+A3*30+A4*3
#1=A+A1*2+a2*3+A3*20+A4*0
#1=A+A1*2+a2*1+A3*30+A4*3

#print(ty[['Survived','Sex','Pclass','AgeRange','CabinRange']].head())
#print(ty.corr())

#model1=smf.ols(formula='Survived~Age',data =ty).fit()
#print(model1.summary())
#model2=smf.ols(formula='Survived~Age+Sex',data =ty).fit()
#print(model2.summary())
#model3=smf.ols(formula='Survived~Age+CabinRange',data =ty).fit()
#print(model3.summary())





