#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load required libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load Kiva_loans.csv file
dataset_kiva_loans = pd.read_csv('../input/kiva_loans.csv')


# In[ ]:


#View Top 10 rows
dataset_kiva_loans.head(5)


# In[ ]:


#View Top 10 rows
dataset_kiva_loans.head(3)


# In[ ]:


#view file data size
dataset_kiva_loans.shape


# In[ ]:


dataset_kiva_loans.describe()


# In[ ]:


#view Columns types
dataset_kiva_loans.dtypes


# In[ ]:


#get Distinct country names from dataset
CountryList = dataset_kiva_loans.country.unique()
CountryList.shape


# In[ ]:


#print all 87 Countries' Name 
CountryList


# In[ ]:


dataset_kiva_loans.groupby("country").count()


# In[ ]:


test = dataset_kiva_loans.groupby(['country'])
test.describe()


# In[ ]:


subset_Coutry = dataset_kiva_loans['country']
subset_Coutry.head(3)


# In[ ]:


subset2_Coutry = dataset_kiva_loans.iloc[:, 7].values


# In[ ]:


counts_by_country = dataset_kiva_loans['country'].value_counts()


# In[ ]:


type(counts_by_country )


# In[ ]:


counts_by_country = pd.DataFrame(dataset_kiva_loans['country'].value_counts())


# In[ ]:


type(counts_by_country )


# In[ ]:


#View Result
counts_by_country


# In[ ]:


counts_by_country.axes


# In[ ]:


counts_by_country = counts_by_country.rename(columns = {"country":"Count"})
counts_by_country


# In[ ]:


#Get Total Lean_Amount By Country
NewDataset = dataset_kiva_loans[['country','loan_amount']]


# In[ ]:


Loan_By_Country = NewDataset.groupby('country').sum()


# In[ ]:


Loan_By_Country


# In[ ]:


Loan_By_Country.reset_index(level=0, inplace=True)


# In[ ]:


Loan_By_Country


# In[ ]:


Loan_By_Country = Loan_By_Country.sort_values(['loan_amount'], ascending=[0])


# In[ ]:


Loan_By_Country


# In[ ]:


plt.figure()
Loan_By_Country.plot(kind='bar')
plt.title('Amount distribution')
plt.xlabel('Country')
plt.ylabel('Amount')


# In[ ]:


Loan_By_Country.iloc[:,0]


# In[ ]:


plt.figure()
r = range(len(Loan_By_Country))
plt.barh(r,Loan_By_Country.iloc[:,1],height = 0.3)
plt.title('Amount distribution')
plt.ylabel('Country')
plt.xlabel('Amount')


# In[ ]:


Loan_By_Country.


# In[ ]:


Loan_By_Country10 = Loan_By_Country.iloc[0:10,:]
Loan_By_Country10


# In[ ]:


xtickNames = Loan_By_Country10['country'].values
type(xtickNames)


# In[ ]:


xtickNames


# In[ ]:



plt.figure()
Loan_By_Country10.plot(kind='bar')
plt.xticks(range(10), xtickNames,rotation = 45)

plt.title('Amount distribution')
plt.xlabel('Country')
plt.ylabel('Amount')



# In[ ]:


Loan_By_Country['loan_amount']


# In[ ]:


plt.hist(Loan_By_Country['loan_amount'],bins = 10) 
plt.title('Amount distribution')
plt.xlabel('Amount')
plt.ylabel('#Country')
plt.show()

