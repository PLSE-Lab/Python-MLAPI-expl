#!/usr/bin/env python
# coding: utf-8

# # Python Pandas Series Introduction
# This is my fourth kernel. If you like it, Please upvote! I prepared this kernel while i was studying Udemy course
# https://www.udemy.com/course/veri-bilimi-ve-makine-ogrenmesi-icin-python/

# # Downloading and saving various File Types

# # 1. Csv Files

# In[ ]:


import pandas as pd


# In[ ]:


d =pd.read_csv("../input/Death_United_States.csv")
d


# In[ ]:


d["State"].tolist()


# In[ ]:


state=d["State"].to_frame()


# In[ ]:


state.to_csv("state.csv",index=False)


# In[ ]:


pd.read_csv("state.csv")


# # 2. Excel files

# In[ ]:


c = pd.read_excel("../input/excel.xlsx",sheet_name=["2006","2010"])


# In[ ]:


c["2010"]


# In[ ]:


four=pd.read_csv("../input/2014 wc.csv")


# In[ ]:


four.to_excel("2014.xlsx",sheet_name="2014")


# # SERIES - Methods, parameters and arguments

# In[ ]:


years = [2015,2016,2017,2018,2019]
income = [10000,20000,30000,40000,50000]

pd.Series(data = income ,index = years)


# In[ ]:


number = [1,2,4,6,8,3,34]
number=pd.Series(number)


# In[ ]:


number


# In[ ]:


number.sum()


# In[ ]:


number.max()


# In[ ]:


number.min()


# In[ ]:


number.product()


# In[ ]:


number.mean()


# # Producing pandas-series (from lists and dictionaries)

# In[ ]:


import pandas as pd


# In[ ]:


x=[1,2,3,5,6]


# In[ ]:


x=pd.Series(x)


# In[ ]:


country=["usa","china","turkey","france"]


# In[ ]:


country=pd.Series(country)


# In[ ]:


d=[True,True,False]


# In[ ]:


pd.Series(d)


# In[ ]:


income={"usa":80000,"germany":50000,"turkey":15000}


# In[ ]:


pd.Series(income)


# In[ ]:


country.values


# In[ ]:


country.index


# In[ ]:


country.dtype


# In[ ]:


x.shape


# In[ ]:


country.name="Countries"


# In[ ]:


country.head()


# # sort_values () method, inplace parameter and "in"

# In[ ]:


country=pd.read_csv("../input/ulke.csv",squeeze=True)
income=pd.read_csv("../input/milligelir.csv",squeeze=True)
country


# In[ ]:


country.sort_values().head()


# In[ ]:


income.sort_values(ascending=False).tail()


# In[ ]:


income.sort_values(ascending=True,inplace=True)


# In[ ]:


income


# In[ ]:


income.sort_index()


# In[ ]:


1 in [1,2,3,4]


# In[ ]:


"USA" in country.values


# # Built-in Functions

# In[ ]:


country=pd.read_csv("../input/ulke.csv",squeeze=True)
income=pd.read_csv("../input/milligelir.csv",squeeze=True)


# In[ ]:


len(income)


# In[ ]:


type(country)


# In[ ]:


sorted(income)


# In[ ]:


list(income)


# In[ ]:


dict(income)


# In[ ]:


max(income)


# In[ ]:


min(income)


# # Indexing and math operations

# In[ ]:


country=pd.read_csv("../input/ulke.csv",squeeze=True)
country


# In[ ]:


country[:10]


# # math operations

# In[ ]:


income=pd.read_csv("../input/milligelir.csv",squeeze=True)
income


# In[ ]:


income.count()


# In[ ]:


len(income)


# In[ ]:


income.sum()


# In[ ]:


income.mean()


# In[ ]:


income.std()


# In[ ]:


income.max()


# In[ ]:


income.min()


# In[ ]:


income.median()


# In[ ]:


income.describe()


# # Read_csv ( ) , head( ) and tail ( ) methods

# In[ ]:


country=pd.read_csv("../input/ulke.csv")
country


# In[ ]:


country=pd.read_csv("../input/ulke.csv",squeeze=True)
country


# In[ ]:


income=pd.read_csv("../input/milligelir.csv")
income


# In[ ]:


income.head(10)


# In[ ]:


income.tail(10)


# # value_counts( ),idxmax( ),idxmin( )  ve apply( ) methods

# In[ ]:


continent=pd.read_csv("../input/kta.csv",squeeze=True)


# In[ ]:


continent.value_counts()


# In[ ]:


continent.value_counts(ascending=True)


# In[ ]:


nat_income=pd.read_csv("../input/milligelir.csv",squeeze=True)


# In[ ]:


nat_income.idxmax()


# In[ ]:


nat_income[0]


# In[ ]:


nat_income.idxmin()


# In[ ]:


nat_income[19]


# In[ ]:


def classs(gel):
    if gel < 2000000:
        return "medium"
    elif gel >=2000000 and gel <=5000000:
        return "high"
    else:
        return "too high"


# In[ ]:


nat_income.apply(classs)


# In[ ]:


nat_income.apply(lambda nat_income:nat_income*2)


# In[ ]:




