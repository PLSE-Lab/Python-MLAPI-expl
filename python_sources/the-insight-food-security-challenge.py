#!/usr/bin/env python
# coding: utf-8

# # Food Security Challenge

# ### Import libraries and Read Datasets

# In[ ]:


# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math
from datetime import datetime
from copy import deepcopy
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading the input directory files
import os
print(os.listdir("../input/"))


# ### Opening esoko prices file

# In[ ]:


df = pd.read_csv('../input/esoko.csv')
df.shape
# Deleting duplicate rows
df =df.dropna()
# Finding the number of rows and columns in the esoko dataset
df.shape


# ### Defining function to convert dates to datetimes

# In[ ]:


# Convert_to_date helper function
def convert_to_date(x):
    date = None
    try:
        parts = x.split("/")
        m = parts[0]
        d = parts[1]
        y = parts[2][:4]
        if len(m)<2:
            m = "0" + m
        if len(d)<2:
            d = "0" + d
        date_str = m + "/" + d + "/"+ y
        date = datetime.strptime(date_str,"%m/%d/%Y")
    except Exception as e:
        print(x)
        date = "badDate"
    return date


# In[ ]:


#Applying the function to the dataframe and creating new column name datetime
df["datetime"] = df.apply(lambda x: convert_to_date(x["Date"]),axis=1)


# In[ ]:


#Finding the number of rows and columsn for the dataset
df.shape


# In[ ]:


#Getting the data type for each columns to see if each one corresponded to the expected type
df.dtypes


# In[ ]:


# Displaying the null values if any
#df.datetime.tail()
print(" The number of null values is: " , df.isnull().values.sum())
print(df.isnull().sum())


# In[ ]:


# Getting the summary statitcs for each column in the esoko dataframe
df.describe(include="all")


# In[ ]:


# Making a copy of the esoko dataframe 
df1 = deepcopy(df)


# In[ ]:


# Adding a new column year to the esoko dataframe
df1['year'] = pd.to_datetime(df1['datetime']).dt.year
df1.datetime = pd.to_datetime(df.datetime)
df1.head() # displaying the first rows
#df1.datetime.head()


# In[ ]:


df5 = deepcopy(df1)
#df5.head()


# In[ ]:


df5.set_index('year', inplace=True)
df5=df5[['Province','District','Market', 'Commodity','Average Price','Date']]
#df5.head()


# #### Grouping our data into Provinces, Districts,Commodity and year and get the mean prices for each commodity for the grouped values

# In[ ]:





# In[ ]:


df7=df5.groupby(['Province', 'District', 'Commodity', 'year'])['Average Price'].mean().reset_index()
#df7.head()


# In[ ]:


df8=deepcopy(df7)


# #### Creating a list of commodity categories

# In[ ]:


#'Cereals and products'
Cereals=['Ifu-ibigori','Ifu-imyumbati','Ifu-amasaka','Ifu-ibigori-yo-mumahanga','Ifu-ingano','Ibigori','Umuceli-Asia','Umuceli-Tz','Umuceli-RW','Amasaka','Ingano']
#'Fruits and products'
Fruits =['Pome','Avoka','Imineke','Amapera','Igitoki','Ibitoki-inkashi','Indimu','Mandarine','Imyembe','Ipapayi','Inanasi','Inkeri','Ibinyomoro','Amatunda','Amacunga']
#'Vegetables'
Vegetables=['Amashu','Shufureri','Karoti','Isombe','Konkombure','Ibiringanya','Tungurusumu','Imiteja','Leti','Ibitunguru-umweru','ibitunguru-bitukura','Urusenda','Sereli','Ibihaza','Epinari','Inyanya','Puwaro','Puwavuro','Dodo','Perisili']
#'Treenuts' 
Treenuts= ['Ubunyobwa' ,'Soya']
#'Roots and tubers (products)'
Roots =['Beterave' , 'Imyumbati' , 'Ibirayi' , 'Ibijumba']
#'Pulses'
Pulses =[ 'Ibitonore', 'Ibishyimbo', 'Urunyogwe' , 'Amashaza']
#'Eggs'
Eggs= 'igi'
#'Meat'
Meat= ['Inkoko' , 'Inyama-zinka', 'Inyama-ihene', 'Inyama-ingurube', 'Inyama-intama']
#'Milk and Cheese'
Milk ='inshyushyu'


# In[ ]:


#df8.head()


# ### Creating Category for Commodity and apply the new column

# In[ ]:


def category(column_data):
    if column_data in Cereals:
        return 'Cereals and products'
    elif column_data in Fruits:
        return 'Fruits and products'
    elif column_data in Vegetables:
        return 'Vegetables'
    elif column_data in Treenuts:
        return 'Treenuts' 
    elif column_data in Roots :
        return 'Roots and tubers (products)'
    elif column_data in Pulses:
        return 'Pulses'
    elif column_data in Eggs:
        return 'Eggs'
    elif column_data in Meat:
        return 'Meat'
    elif column_data in Milk:
        return 'Milk and Cheese'
    else:
        return 'Others'


# In[ ]:


df8['Category'] = df8['Commodity'].apply(category)


# ### New Category Column with Data grouped by District, Commodity & Year 

# In[ ]:





# In[ ]:


df8.head()


# ### Chosing only the first main commodities

# In[ ]:


# Defining an array of the first main commodities
array = ["Ingano","Ibigori","Amasaka","Umuceli-RW","Ibijumba","Ibirayi","Imyumbati","Igitoki","Inyanya","Amashu","bitoki-inkashi","Imineke","Amacunga","Inanasi","Ibitonore","Ibishyimbo","Amashaza","Soya","Ubunyobwa"]

#Obtain esoko data that belongs to the first main commodities
df9 = df8.loc[df8['Commodity'].isin(array)]


# In[ ]:


df9.head()


# In[ ]:


df10=deepcopy(df9)
df10.head()


# ### Creating an helper function commodity_eng that will add a new column of commodity names translated in english to match the Food security data names

# In[ ]:


# Define the helper function
def commodity_eng(column_data):
    if column_data == "Ingano":
        return "Wheat"
    elif column_data == "Ibigori":
        return "Maize"
    elif column_data == "Amasaka":
        return "Sorghum"
    elif column_data == "Umuceli-RW":
        return "Rice"
    elif column_data == "Ibijumba":
        return "White fleshed sweet potatoes"
    elif column_data == "Ibirayi":
        return "Irish potato"
    elif column_data == "Imyumbati":
        return "Cassava"
    elif column_data == "Igitoki":
        return "Banana cooking"
    elif column_data == "Inyanya":
        return "Tomato"
    elif column_data == "Amashu":
            return "Cabbage"
    elif column_data == "bitoki-inkashi":
            return "Banana (wine)"
    elif column_data == "Imineke":
            return "Banana fruit"
    elif column_data == "Amacunga":
            return "Passion fruit"
    elif column_data == "Inanasi":
            return "Pineapple"
    elif column_data == "Ibitonore":
            return "Iron fortified beans"
    elif column_data == "Ibishyimbo":
            return "Beans"
    elif column_data == "Amashaza":
            return "Peas"
    elif column_data == "Soya":
            return "Soya"
    elif column_data == "Ubunyobwa":
            return "Ground nuts"
    else:
        return 'Others'


# In[ ]:


# create the new column commodity_english with commodity names in english
df10['Commodity_english'] = df10['Commodity'].apply(commodity_eng)


# In[ ]:


# Changing the disr=trict anmes to lower cases adn capitalize the first letter
df10['District']=df10['District'].str.lower().str.capitalize()
df10.head()


# ## Select Food Prices Data for 2018 Year Only

# ### We decided to use data for 2018 for our prototype and generalize it for the execrices. Later  we will use different years to obtain a time series analysis

# In[ ]:


Selected_Food_Prices_2018=df10.loc[df10['year']==2018]
Selected_Food_Prices_2018.head()


# In[ ]:


Selected_Food_Prices_2018.shape


# In[ ]:


# Selected_Food_Prices_2018.to_csv('Food_prices_2018_20190821.csv')


# # End of Processing Esoko Food Prices
# 

# > ## Creating graphs

# In[ ]:


#Food_Prices_2018.describe(include="all")


# In[ ]:


Food_Prices_2018_Fruits=Selected_Food_Prices_2018.loc[Selected_Food_Prices_2018['Category']== 'Fruits and products']


# In[ ]:


#Food_Prices_2018_Fruits.head()


# In[ ]:


Food_Prices_2018_Fruits2=Selected_Food_Prices_2018.loc[(Selected_Food_Prices_2018['Category']== 'Fruits and products') & (Selected_Food_Prices_2018['District']== 'Bugesera')]
#Food_Prices_2018_Fruits2.head()


# #### Plot the fruits and product commodity Average price for Bugesera District

# In[ ]:



Food_Prices_2018_Fruits2.plot(x="Commodity_english", y="Average Price", figsize=(20,10), kind="bar")


# In[ ]:


KIGALI_N_Food_Prices_2018_Fruits=Selected_Food_Prices_2018.loc[(Selected_Food_Prices_2018['Category']== 'Fruits and products') & (Selected_Food_Prices_2018['District']== 'Nyarugenge')]
KIGALI_G_Food_Prices_2018_Fruits=Selected_Food_Prices_2018.loc[(Selected_Food_Prices_2018['Category']== 'Fruits and products') & (Selected_Food_Prices_2018['District']== 'Gasabo')]
KIGALI_K_Food_Prices_2018_Fruits=Selected_Food_Prices_2018.loc[(Selected_Food_Prices_2018['Category']== 'Fruits and products') & (Selected_Food_Prices_2018['District']== 'Kicukiro')]


# In[ ]:


Maize_Food_Prices_2018=Selected_Food_Prices_2018.loc[(Selected_Food_Prices_2018['Commodity_english']== 'Maize')]


# In[ ]:


Maize_Food_Prices_2018.plot(x="District", y='Average Price', figsize=(20,10), kind="bar")


# # Food Security

# ### 1. Crops production Quantity 

# In[ ]:


file_name = '../input/CFSVA_2018_AnnexTables_1.xlsx'
Agriculture_sheet = 'Annex 04 Agriculture'

df_Agriculture = pd.read_excel(file_name, sheet_name = Agriculture_sheet, header = 1,index_col =False)
Crop_product=df_Agriculture.iloc[9:39, 1:34]
#Crop_product
#Crop_product.columns
Crops_production=Crop_product.drop(columns =['Other cereals', 'Other roots and tubers specify:_','Other vegetables','Other fruit specify:______','Other pulses, specify:_____','Other cash crops specify:___']) 
#Crops_production


# In[ ]:


Crops_production_new=Crops_production.drop(columns =['Orange fleshed Sweet potatoes','Taro', 'Yam','Tea', 'Coffee', 'Tobacco', 'Sugar cane']) 
#Crops_production_new


# ### Reshape data in long format

# In[ ]:


df_cro=pd.melt(Crops_production_new, id_vars='District', value_vars=['Wheat', 'Maize', 'Sorghum', 'Rice',
       'White fleshed sweet potatoes', 'Irish potato', 'Cassava',
       'Banana cooking', 'Tomato', 'Cabbage', 'Banana (wine)', 'Banana fruit',
       'Passion fruit', 'Pineapple', 'Iron fortified beans', 'Beans', 'Peas',
       'Soya', 'Ground nuts'])
#df_cro
Crops_production = df_cro.rename(columns={'variable': 'Crops', 'value': 'Production'})
Crops_production


# In[ ]:


Crops_production.shape


# In[ ]:


df_Crops_Prices = pd.merge(Selected_Food_Prices_2018, Crops_production, right_on = ["District","Crops"], left_on = ["District","Commodity_english"], how="inner").reset_index(drop=True)


# In[ ]:


df_Crops_Prices


# In[ ]:


Crops_Prices=df_Crops_Prices.drop(columns =['Commodity_english'])
Crops_Prices.to_csv('Crops_prices_2018_20190821.csv')
Crops_Prices


# ###  2. Food Security Diet Diversity and Food Consumption

# In[ ]:


#file_name = 'CFSVA_2018_AnnexTables_1.xlsx'
Food_Sec_sheet = 'Annex 06 Food Security'
df_food_sec = pd.read_excel(file_name, sheet_name = Food_Sec_sheet, header = 1,index_col =False)
#df_food_sec
Food_Sec=df_food_sec.iloc[9:39,[1,21,22,23,34,35,36,39,40,41,42]]
Food_Sec


# In[ ]:


Food_Security_2=deepcopy(Food_Sec)
Food_Security_last=Food_Security_2.iloc[:,[0,3,6,7]]
Food_Security_last


# ### 3. CFSVA Food access

# In[ ]:


#file_name = 'CFSVA_2018_AnnexTables_1.xlsx'
Food_Access_sheet = 'Annex 07 Food Access Issues'
df_Food_Access = pd.read_excel(file_name, sheet_name = Food_Access_sheet, header = 1,index_col =False)
#df_Food_Access
Food_Access_Issue=df_Food_Access.iloc[9:39,[1,2,3]]
#Food_Access_Issue


# In[ ]:


Food_Access = Food_Access_Issue.rename(columns={'No': 'Have Access', 'Yes': 'Do Not Have Access'})
#Food_Access


# ### Concatenate all CFSVA data into one table

# In[ ]:


All_CFSVA = pd.concat([Food_Security_last, Food_Access], axis = 1)

All_CFSVA2=All_CFSVA.iloc[:,[0,1,2,3,5]]
All_CFSVA2
All_CFSVA2 = All_CFSVA2.rename(columns={'Have Access': 'Food Access'})
All_CFSVA2


# In[ ]:


All_CFSVA2.to_csv('Food_security_Access_20190821.csv')


# # Conclusion

# #### Based on the below Graph analysis and the results from the dashboard Analysis, 
# #### we found that,there is a positive relationship between the Access to Food 
# #### and the food security level.
# #### The more people have access to food the more food secure they are

# In[ ]:


import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
food_security =  deepcopy(All_CFSVA2)
fig, ax = plt.subplots(figsize = (15, 10) )
sns.pointplot(x='District', y='Food secure', data=food_security, ax=ax, label='Food secure',  color='b')
sns.pointplot(x='District', y='Food Access', data=food_security, ax=ax, label= 'Food Access',color='r')
labels = ax.get_xticklabels() 
ax.set_xticklabels(labels, rotation=-90)
plt.title("Food Security vs Food Access")
plt.legend()
plt.show()


# ## Hypothesis testing ##

# To confirm that there is a relationship between food security and independent variables (food consumption, food diet diversity,food access); we are going to perform a 2 sampled T-test.

# Food security and food access

# In[ ]:


from scipy.stats import ttest_ind
import numpy as np
food_access_yes = Food_Access['Have Access']
food_sec = Food_Security_last['Household food security situation']
food_access_mean = np.mean(food_access_yes)
food_sec_mean = np.mean(food_sec)
print("food access mean:",food_access_mean)
print("food security mean:",food_sec_mean)
food_access_std = np.std(food_access_yes)
food_sec_std = np.std(food_sec)
print("food access std value:",food_access_std)
print("food security std value:",food_sec_std)
ttest,pval = ttest_ind(food_access_yes,food_sec)
print("p-value",pval)
if pval <0.05:
  print("we reject null hypothesis, there is a link between food security and food access")
else:
  print("we accept null hypothesis")


# In[ ]:




