#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Let us first read our data and try to analyse it

# In[ ]:


df = pd.read_csv("../input/car data.csv", )
df.head()


# #### On looking at the data we need to question ourself as to how we can explore this data. Lets start by analyzing every column individually

# #### On looking at the data it is noted that we have both the present price and the selling price of the car. Although their might be a difference in purhcase prices depending upon which year the car was purchased, howevere that can be neglibible as no extreme variation is ever seen in car purchase price.

# ### Now lets create a seprate column giving us the depreciation of each car and add the column to our main data frame

# In[ ]:


car_depreciation = df['Present_Price'] - df['Selling_Price']
df['depreciation'] = car_depreciation
car_depreciation


# #### Great! Now we know exactly how much depreciation has happened for every car. Also we need to represent this by car's name and not by serial number. So lets make a seprate dataset and include the variables we are going to use to visualise our data which would help us see which car has the best and worst re-sale values

# In[ ]:


depr = df[['Car_Name', 'depreciation']]
depr.head()


# #### Using group by and then sorting it in descending order gives us the car with the maximum depreciation

# In[ ]:


grouped = depr.groupby('Car_Name').mean()
grouped.sort_values('depreciation', ascending=False)


# In[ ]:


df['Car_Name'].value_counts().head(30)


# #### We can see that Cars like Land Cruiser, Camry, Corolla does not have much samples hence we would not be discussing them further in detail

# #### Lets analyse why other cars have such high depreciation and analyze some cars indvidually

# In[ ]:


temp = df.loc[df['Car_Name'] == 'fortuner']
temp.head(11)# since fortuner has 11 samples


# #### Now lets plot the same on a bar graph using matplotlib which will give us an better understanding of depreciation of fortuner for its make year

# In[ ]:


plt.bar(x = temp['Year'], height=temp['depreciation'])
plt.xlabel('Year')
plt.ylabel('Depreciation in lacs')


# #### The above bar graph shows us that depreciation of fortuner does not completely depends on its make year

# #### Now lets do the same for some other cars

# In[ ]:


temp_1 = df.loc[df['Car_Name'] == 'city']
temp_1.head()


# In[ ]:


plt.bar(x = temp_1['Year'], height=temp_1['depreciation'])
plt.xlabel('Year')
plt.ylabel('Depreciation in lacs')


# #### The above graph gives us an idea about depreciation of city over its make year, which is also not constant

# In[ ]:


temp_2 = df.loc[df['Car_Name'] == 'corolla altis']
temp_2.head()


# In[ ]:


plt.bar(x = temp_2['Year'], height=temp_2['depreciation'])
plt.xlabel('Year')
plt.ylabel('Depreciation in lacs')


# In[ ]:


temp_3 = df.loc[df['Car_Name'] == 'verna']
temp_3.head()


# In[ ]:


plt.bar(x = temp_3['Year'], height=temp_3['depreciation'])
plt.xlabel('Year')
plt.ylabel('Depreciation in lacs')


# #### The bar graph for verna and corolla altis gives us a more constant depreciation than city and fortuner, hence leaves us with a mixed idea of make year being the most important factor in depreciation

# ### Now let us analyze some more variables

# In[ ]:


kms = df[['Car_Name', 'Kms_Driven']]
kms.head()


# In[ ]:


by_kms_driven = kms.groupby('Car_Name').mean().tail(30)
by_kms_driven.sort_values('Kms_Driven', ascending=False)


# #### Now lets study the bar graphs for depreciation of different cars based on the kilometeres they are driven

# In[ ]:


temp_4 = df.loc[df['Car_Name'] == 'city']
temp_4.head()


# In[ ]:


plt.bar(x = temp_4['depreciation'], height=temp_4['Kms_Driven'])
plt.xlabel('Depreciation in lacs')
plt.ylabel('Kms Driven')


# In[ ]:


temp_5 = df.loc[df['Car_Name'] == 'fortuner']
temp_5.head()


# In[ ]:


plt.bar(x = temp_5['depreciation'], height=temp_5['Kms_Driven'])
plt.xlabel('Depreciation in lacs')
plt.ylabel('Kms Driven')


# In[ ]:


temp_6 = df.loc[df['Car_Name'] == 'corolla altis']
temp_6.head()


# In[ ]:


plt.bar(x = temp_6['depreciation'], height=temp_6['Kms_Driven'])
plt.xlabel('Depreciation in lacs')
plt.ylabel('Kms Driven')


# In[ ]:


temp_7 = df.loc[df['Car_Name'] == 'verna']
temp_7.head()


# In[ ]:


plt.bar(x = temp_7['depreciation'], height=temp_7['Kms_Driven'])
plt.xlabel('Depreciation in lacs')
plt.ylabel('Kms Driven')


# #### The above graph shows somewhat constant behaviour for depreciation agaist kms driven, howevere even kms driven cannot be considered as sole factor affecting depreciation

# #### for studying other variables we need to transform them to get a better understanding of their importance in depreciation of car

# #### Let us now transform our categoricals values into numerical 

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()


# In[ ]:


le.fit(df['Fuel_Type'])


# In[ ]:


le.classes_


# In[ ]:


df['Fuel_Type'] = le.transform(df['Fuel_Type'])


# In[ ]:


df.head()


# In[ ]:


le.fit(df['Seller_Type'])


# In[ ]:


df['Seller_Type'] = le.transform(df['Seller_Type'])


# In[ ]:


le.classes_


# In[ ]:


df.head()


# In[ ]:


le.fit(df['Transmission'])


# In[ ]:


df['Transmission'] = le.transform(df['Transmission'])


# In[ ]:


df.head()


# In[ ]:


by_fuel_type = df[['Car_Name', 'Fuel_Type', 'depreciation']]
by_fuel_type.head()


# In[ ]:


fuel_t = by_fuel_type.groupby('Fuel_Type').mean()

fuel_t.head()


# #### The Fuel type '0'is CNG, '1' is diesel and '2' is petrol, hence we know from this that diesel cars have the max depreciation, however this is not concerte as other factors also have an important say than just fuel type

# In[ ]:


by_seller_t = df[['Car_Name', 'Seller_Type', 'depreciation']]
by_seller_t.head()


# In[ ]:


seller_t = by_seller_t.groupby('Seller_Type').mean()
# by_kms_driven.sort_values('Kms_Driven', ascending=False)
seller_t.head()


# #### From this we know that '0' is automatic and '1' is manual, howevere even this does not helps our purpose of getting accurate relationship.

# #### Let us plot some more graphs to get some better insights

# #### Let us plot a pair plot to get better understanding of how the variables Kms_Driven and Year affects depreciation, we will be using seaborn for this

# In[ ]:


new_df = df[['Car_Name', 'Year', 'Kms_Driven', 'depreciation']]
new_df.head()


# In[ ]:


sns.pairplot(new_df)


# #### The above graph gives us a more clear picture, we can see from it that both Year and Kms Driven have an important say on depreciation

# #### Let us analyze all the values that can have a say on cars depreciation**

# In[ ]:


final_val = df[['Car_Name', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'depreciation','Year', 'Transmission']]
final_val.head()


# In[ ]:


sns.pairplot(final_val)


# #### The above graph shows that the variales such as transmission and seller type does not have much impact on depreciation as compared to Kms Driven and Year, hence from this we can conclude that kms driven and year has a bigger impact on depreciation than other factors.
