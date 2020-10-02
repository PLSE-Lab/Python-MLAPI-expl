#!/usr/bin/env python
# coding: utf-8

# # Best time to buy or sell a car
# 
# To own a car costs money.
# 
# There are many factors which have an influence on the buying and selling price of a car. Further there are a lot of additional costs depending on the model, the millage, the fuel consumption etc.
# 
# Usually the older the car is, the less it will be worth. For example, a new car might lose 20-30% of its value in only 1-2 years, but later the diminution might be slower. A car loses less or more value every year depending of its type. Sometimes it can even increase in value.
# 
# <strong>In the analysis we will focus on the best time to buy and to sell a car to minimize the value lost. </strong>
# 

# <img src="https://i.imgur.com/K5zuZGG.png"/> 

# ## Dataset: "Used Cars Dataset"
# 
# The data comes from Craigslist in the USA and provides information on car sales. It contains more 500.000 vehicles and has 25 columns.
# 
# - <strong>identry:</strong> ID
# - <strong>url:</strong> listing URL
# - <strong>region:</strong> craigslist region
# - <strong>region_url:</strong> region URL
# - <strong>price:</strong> entry price
# - <strong>year:</strong> entry year
# - <strong>manufacturer:</strong> manufacturer of vehicle
# - <strong>model:</strong> model of vehicle
# - <strong>condition:</strong> condition of vehicle
# - <strong>cylinders:</strong> number of cylinders
# - <strong>fuel:</strong> fuel type
# - <strong>odometer:</strong> miles traveled by vehicle
# - <strong>title_status:</strong> title status of vehicle
# - <strong>transmission:</strong> transmission of vehicle
# - <strong>vin:</strong> vehicle identification number
# - <strong>drive:</strong> type of drive
# - <strong>size:</strong> size of vehicle
# - <strong>type:</strong> generic type of vehicle
# - <strong>paint_color:</strong> color of vehicle
# - <strong>image_url:</strong> image URL
# - <strong>description:</strong> listed description of vehicle
# - <strong>county:</strong> useless column left in by mistake
# - <strong>state:</strong> state of listing
# - <strong>lat:</strong> latitude of listing
# - <strong>long:</strong> longitude of listing
# 
# 

# # Descriptive analysis

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../input/craigslist-carstrucks-data/vehicles.csv")


# In[ ]:


df.head()


# In[ ]:


df.describe()


# <strong>Was the oldest car a carriage and was built was the romans? <br/></strong>
# <img src="https://i.imgur.com/IjJBbT2.png" align="left">
# 

# We can see that the oldest vehicle was built in the year 0. Maybe this is the carriage of Jules Cesar? No, because he was already dead by that time.

# In[ ]:


df.info()


# In[ ]:


# Display the missing values
plt.figure(figsize=(12,12))
plt.title("Missing values for each column")
sns.heatmap(df.isnull())
plt.show()


# In[ ]:


# Count the values by year avoiding value under 1900, 
df[df.year >= 1900].year.value_counts().sort_index().plot(lw = 4)
plt.title("Number of vehicles in the dataset by build year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()


# Most of the vehicles in the dataset are from between 2000 and 2020.

# # Data cleaning

# ## Condition & Status:
# 

# The listed vehicles are in different condition and for the purpose this analysis, we don't want to take in consideration vehicles in very poor condition.

# <img src="https://i.imgur.com/UENEQXf.png" align = "left">

# <strong>VS</strong>

# <img src="https://i.imgur.com/lwTcFMl.png" align = "left">

# There are two different columns showing the condition and the status of the cars with some overlapping:

# In[ ]:


df.condition.value_counts().plot.bar()
plt.title("Condition of the vehicles")
plt.show()


# In[ ]:


df.title_status.value_counts().plot.bar()
plt.title("Status of the vehicles")
plt.show()


# For our analysis, we don't want to consider cars which have the condition/status:
# - salvage
# - lien
# - missing
# - parts only 
# 
# Therefore, we will delete those rows

# In[ ]:


idx1 = df[df["condition"] == "salvage"].index

for w in ["salvage","lien","missing","parts only"]:
    idx2 = df[df["title_status"] == w].index
    idx1 = idx1.union(idx2)
    
df.drop(idx1, axis = 0, inplace = True)


# ## Price

# <img src="https://i.imgur.com/0VmCdN7.png" align = "left">

# In[ ]:


print(f"Maximum price: {df.price.max()} $\nMinimum price: {df.price.min()} $")


# The maximum price might be a mistake (over one billion US dollars). We want to ignore the outliers and what interesse us, are cars, which can be bought by "average" people, so we will delete all the rows with a price over 100.000 US dollars.
# 
# 
# We will also delete the cars with a price less than 100$, because it might be a mistake if the price is so low.

# In[ ]:


df = df[(df["price"] >= 100) & (df["price"] <= 100000)]


# In[ ]:


sns.boxplot(df.price)
plt.title("Repartition of the price after deleting price over 100.000$")
plt.show()


# We can see that the price is usually between 100 and 40.000\$ with a lot of outliers.

# ## Build year of the vehicle

# In[ ]:


print(f"Higher year: {df.year.max()}\nLowest year: {df.year.min()}")


# The years 0 and 2021 are obviously wrong. Therefore, we will delete them.
# 
# To facilitate our analysis, we will create a new column with the age of the vehicle (age = 2020-year_build). We will keep vehicle with an age between 0 and 30 because we are more interested in "recent" vehicles.

# In[ ]:


df = df[df.year.notnull()]
df["age"] = df.year.apply(lambda x: int(2020-x))
df = df[(df.age >= 0) & (df.age <= 30)]


# In[ ]:


sns.distplot(df.age, hist = False)
plt.title("Distribution of the age of the vehicles")
plt.show()


# As we can see most of the vehicle are between 0 and 20 years old.

# ## Vehicule type

# In[ ]:


df.type.value_counts(dropna=False).plot(kind = "bar")
plt.title("Number of each type of vehicle:")
plt.show()


# Further, we will use the type of vehicles, therefore, we will drop the types: 
# - NaN
# - other
# - bus (not enough data)
# - offroad (not enough data)

# In[ ]:


# Delete the NaN
df = df[df["type"].notnull()]

# Delete "other","bus" and "offroad"
for v in ["other","bus", "offroad"]:
    df = df[df["type"] != v]


# ## Correlation

# In[ ]:


cols = ["price","age", "odometer"]
sns.heatmap(df[cols].corr(), annot = True)
plt.title("Correlation:")
plt.show()


# The highest correlation is between age and price.

# # Find the best time to buy or sell a car

# Finally we come to the most interesting part, the price by age of the vehicle depending on the vehicle type. To calculate the price, the median has been choosen, because as we've seen in the boxplot with the prices, there are quite a lot of outliers, even limiting the prices between 100 and 100.000$. This way it will be more stable. 

# In[ ]:


# Images of cars for the graphics
images = {'all':'https://i.imgur.com/1vNeS3S.png',
         'SUV':'https://i.imgur.com/hDAAIQ1.png', 
         'wagon':'https://i.imgur.com/AScvovW.png', 
         'sedan':'https://i.imgur.com/geFnoDw.png',
         'convertible':'https://i.imgur.com/OJyUNkl.png',
         'pickup':'https://i.imgur.com/RZI2aBP.png',
         'hatchback':'https://i.imgur.com/I6nKBgU.png',
         'truck':'https://i.imgur.com/d5ImbCK.png',
         'coupe':'https://i.imgur.com/zf6cHos.png',
         'van':'https://i.imgur.com/ly3Fg5V.png',
         'mini-van':'https://i.imgur.com/CfmLXIG.png'}

def display_price(df, age = (0,12), price = (100,100000), vehicle_type = "all", state = "all"):
    # Display the median price of vehicles depending on its type and its state.
    
    if state != "all":
        df = df[df["state"] == state]
    
    if vehicle_type != "all":
        df = df[df["type"] == vehicle_type]
        
    df = df[(df["age"] <= age[1]) & (df["age"] >= age[0])]
    
    df = df[(df["price"] >= price[0]) & (df["price"] <= price[1])]
    
    price_age = pd.pivot_table(df, values = "price", index = "age", aggfunc= np.median)
    price_age.columns = ["Median Price"]
    
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_axes([0,0,1,1])
    ax2 = fig.add_axes([0.6,0.47,.35,.35])
    
    ax.plot(price_age["Median Price"], lw = 5)
    
    ax2.imshow(plt.imread(images[vehicle_type]))
    ax2.set_title(f"Vehicle type: {vehicle_type}\nNumber of vehicles: {df.shape[0]}\nCountry: USA\nUS-State: {state}", fontsize = 15)
    ax2.axis('off')
    
    ax.set_title(f"Median price by age of the vehicles",fontsize=25)
    ax.set_ylim(0,price_age["Median Price"].max()+1000)
    ax.set_xlabel("Age", fontsize = 15)
    ax.set_ylabel("Median price in $", fontsize = 15)
    
    ax.tick_params(axis='both', which='major', labelsize=15) 

    plt.show()


# In[ ]:


display_price(df, vehicle_type="all")


# In[ ]:


for t in df.type.unique()[:3]:
    display_price(df, vehicle_type=t)


# In[ ]:


for t in df.type.unique()[3:6]:
    display_price(df, vehicle_type=t)


# In[ ]:


for t in df.type.unique()[6:9]:
    display_price(df, vehicle_type=t)


# In[ ]:


for t in df.type.unique()[9:]:
    display_price(df, vehicle_type=t)


# ## Conclusion
# 
# As we have seen, the price depends on the age and the type of vehicle. Planning strategically when a car is bought and sold makes it possible to lose as little money as possible. In general, to buy a 2 years old car and to sell it when it is 4-5 years old seem to be a good choice.
# 
# The analysis could even be made more specific with more data. For example: exploring the graphic for a car model in a specific place.
# 
