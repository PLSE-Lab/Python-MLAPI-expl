#!/usr/bin/env python
# coding: utf-8

# [](https://upload.wikimedia.org/wikipedia/commons/b/b3/Five_Tesla_Model_S_electric_cars_in_Norway.jpg)

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# read train datasets
norway = pd.read_csv('../input/norway_new_car_sales_by_model.csv',encoding="latin-1")


# In[ ]:


norway.head()


# In[ ]:


# Display all informations
norway.info()


# In[ ]:


#heatmap for train dataset

f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(norway.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[ ]:


#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(norway.isnull(),cbar=False)


# As we see above, the data is clean. 

# In[ ]:



norway["Make"]=norway["Make"].str.replace('\xa0Mercedes-Benz ','Mercedes-Benz ') #fixing rows that have some characters in front of Mercedes-Benz
norway.Make=norway.Make.str.lower()
norway.Model=norway.Model.str.lower()
monthly_total_sales=norway.pivot_table("Quantity",index="Month",columns="Year",aggfunc="sum")
print(monthly_total_sales.mean(axis=1))
#Visualize
monthly_total_sales.mean(axis=1).plot.line()


# In[ ]:


#Calculate total amount of the sales for each manufacturer from 2007 to 2017. Find the top-10 manufacturers based on the total sale.
make_total = norway.pivot_table("Quantity",index=['Make'],aggfunc='sum')
top10make=make_total.sort_values(by='Quantity',ascending=False)[:10]
#print(top10make)
top10make.plot.bar()


# In[ ]:


maketotal_1 = norway.pivot_table(values='Quantity',index=['Month','Model','Make'],aggfunc=np.std)
df1 = maketotal_1.reset_index().dropna(subset=['Quantity'])
#df2 = df1.loc[df1.groupby(['Month','Make'])['Quantity'].idxmax()]
#print (df2.head(n=5))
df3 = df1.loc[df1.groupby('Make')['Quantity'].idxmax()]
for index,row in df3.iterrows():
    print("For Manufacturer",row['Make'],"model",row['Model'],"has the highest yearly fluncation.")


# In[ ]:


car_list = list(norway['Make'].unique())

car_selling_quantity = []

for i in car_list:
    x = norway[norway['Make']==i]
    area_car_rate = sum(x.Pct)/len(x)
    car_selling_quantity.append(area_car_rate)

data = pd.DataFrame({'car_list': car_list,'car_selling_quantity':car_selling_quantity})

new_index = (data['car_selling_quantity'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)

plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data['car_list'], y=sorted_data['car_selling_quantity'])
plt.xticks(rotation= 90)
plt.xlabel('Car Models')
plt.ylabel('Percentage share')
plt.title('Percentage share in Norway')

