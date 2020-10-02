#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Students enrolled in different classes and also in different categories is given by this dataset. This dataset gives the data from 1950 to 2011. But the data from 1950 to 1985 is collected in 5 year term.So here the data is omitted to follow uniformity in the data.The categories in the data are joined and then the students enrollment is predicted for 2019 (the year when prediction is done) and also in 2020 (people might see this notebook in 2020).
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


file = pd.read_csv("/kaggle/input/education-in-india/Statement_SES_2011-12-Enrlment.csv")
file


# # Data Preprocessing
# Data in the initial stages are recorded once in every five years. So we can eliminate the data upto 1986. 
# And the rest of data is categorized to 3 groups such as
#    - class 1 to 5
#    - class 6 to 8
#    - class 9 to 10
# 

# In[ ]:


data = file.iloc[8:,0:]
data


# In[ ]:


for col in data.columns:
    print(col)


# In[ ]:


res = data.drop(columns=['All Categories - Class I-V - Boys',
                        'All Categories - Class I-V - Girls',
                        'All Categories - Class VI-VIII - Boys',
                        'All Categories - Class VI-VIII - Girls',
                        'All Categories - Class IX-X - Boys',
                        'All Categories - Class IX-X - Girls',
                        'All Categories - Class XI-XII - Boys',
                        'All Categories - Class XI-XII - Girls',
                        'Scheduled Caste Category - Class I-V - Boys',
                        'Scheduled Caste Category - Class I-V - Girls',
                        'Scheduled Caste Category - Class VI-VIII - Boys',
                        'Scheduled Caste Category - Class VI-VIII - Girls',
                        'Scheduled Caste Category - Class IX-X - Boys',
                        'Scheduled Caste Category - Class IX-X - Girls',
                        'Scheduled Caste Category - Class XI-XII - Boys',
                        'Scheduled Caste Category - Class XI-XII - Girls',
                        'Scheduled Tribe Category - Class I-V - Boys',
                        'Scheduled Tribe Category - Class I-V - Girls',
                        'Scheduled Tribe Category - Class VI-VIII - Boys',
                        'Scheduled Tribe Category - Class VI-VIII - Girls',
                        'Scheduled Tribe Category - Class IX-X - Boys',
                        'Scheduled Tribe Category - Class IX-X - Girls',
                        'Scheduled Tribe Category - Class XI-XII - Boys',
                        'Scheduled Tribe Category - Class XI-XII - Girls',
                        ])


# In[ ]:


for col in res.columns:
    print(col)


# In[ ]:


temp=res


# In[ ]:


temp.iloc[:,1] += temp.iloc[:,5]+temp.iloc[:,9]
temp.iloc[:,2] += temp.iloc[:,6]+temp.iloc[:,10]
temp.iloc[:,3] += temp.iloc[:,7]+temp.iloc[:,11]
temp.iloc[:,4] += temp.iloc[:,8]+temp.iloc[:,12]
temp


# In[ ]:


temp = temp.drop(columns=['Scheduled Caste Category - Class I-V - Total',
                          'Scheduled Caste Category - Class VI-VIII - Total',
                          'Scheduled Caste Category - Class IX-X - Total',
                          'Scheduled Caste Category - Class XI-XII - Total',
                          'Scheduled Tribe Category - Class I-V - Total',
                          'Scheduled Tribe Category - Class VI-VIII - Total',
                          'Scheduled Tribe Category - Class IX-X - Total',
                          'Scheduled Tribe Category - Class XI-XII - Total',])
temp


# In[ ]:


temp = temp.rename(columns={"All Categories - Class I-V - Total":"1-5",
                    "All Categories - Class VI-VIII - Total":"5-8",
                    "All Categories - Class IX-X - Total":"9-10",
                    "All Categories - Class XI-XII - Total":"11-12"})
temp


# In[ ]:


year=[]
for i in temp.iloc[:,0]:
    i = i[:4]
    year.append(int(i))

temp = pd.DataFrame(temp)
temp.insert(0,"year",year)
temp = temp.drop(columns=['Year'])
temp


# In[ ]:


data = temp.to_numpy()
data


# In[ ]:


data=data.T
data


# # Visualization
# The categorized data is plotted in the same graph

# In[ ]:


import matplotlib.pyplot as plt
plt.plot(data[0],data[1],'.')
plt.plot(data[0],data[2],'.')
plt.plot(data[0],data[3],'.')
plt.show()


# # Model
# Here for prediction Linear regression is used.

# In[ ]:


from sklearn.linear_model import LinearRegression
reg1to5 = LinearRegression()
reg6to8 = LinearRegression()
reg9to10 = LinearRegression()


# The model predicts the students who are getting graded in India.

# In[ ]:


reg1to5.fit(data[0].reshape(-1,1),data[1].reshape(-1,1))
print("Students who are learning from 1 to 5 in 2019 : ",round(reg1to5.predict([[2019]])[0][0],2),"millions")
print("Students who are learning from 1 to 5 in 2020 : ",round(reg1to5.predict([[2020]])[0][0],2),"millions")
reg6to8.fit(data[0].reshape(-1,1),data[2].reshape(-1,1))
print("\nStudents who are learning from 6 to 8 in 2019 : ",round(reg6to8.predict([[2019]])[0][0],2),"millions")
print("Students who are learning from 6 to 8 in 2020 : ",round(reg6to8.predict([[2020]])[0][0],2),"millions")
reg9to10.fit(data[0].reshape(-1,1),data[3].reshape(-1,1))
print("\nStudents who are learning from 9 to 10 in 2019 : ",round(reg9to10.predict([[2019]])[0][0],2),"millions")
print("Students who are learning from 9 to 10 in 2020 : ",round(reg9to10.predict([[2020]])[0][0],2),"millions")


# 1. If you like my work, give a thumbs up to dataset. Comments are always welcome!.

# In[ ]:




