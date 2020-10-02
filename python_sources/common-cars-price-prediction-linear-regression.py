#!/usr/bin/env python
# coding: utf-8

# # Predict prices of commonly used cars on eBay using linear regression
# 
# ### Structure
# - import libraries
# - load and prepare the dataset
# - explore data 
# - linear regression
# - coefficient of determination
# - conclusion

# ### import Libaries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ### Load and prepare dataset
# 
# - remove columns that I do not want to fit the model with
# - replace or remove 0 and NaN values
# - calculate and create new column "registration" with year and month of registration in year.month form starting with year.0 from january
# - filter the DataFrame with common car prices/ps and remove rows with faulty registration dates that are in the future (over 2019)

# In[ ]:


df = pd.read_csv("../input/autos.csv", encoding = "iso8859-1")

df = df.drop(["dateCrawled", "abtest", "dateCreated", "nrOfPictures", "lastSeen", "postalCode", "seller", "offerType", "model"], axis = 1)


df["monthOfRegistration"] = np.where(df["monthOfRegistration"] == 0, 6, df["monthOfRegistration"])
df["registration"] = df["yearOfRegistration"] + (df["monthOfRegistration"] -1) /12
df = df.drop(["yearOfRegistration", "monthOfRegistration"], axis = 1)

df = df.drop(df[(df["powerPS"] == 0) | (df["price"] == 0)].index)

df["notRepairedDamage"] = np.where(df["notRepairedDamage"] == "ja", 1, df["notRepairedDamage"])
df["notRepairedDamage"] = np.where(df["notRepairedDamage"] == "nein", 0, df["notRepairedDamage"])
df = df[df["notRepairedDamage"].notnull()]
#convert values to integer so I can work with them / visualize them more easiliy
df["notRepairedDamage"] = pd.to_numeric(df["notRepairedDamage"])
 
df = df[(df["price"] < 100000) & (df["powerPS"] < 2000) & (df["registration"] <= 2019)]


# ### Explore data
# 
# **Pairplot to get a distribution overview and detect first correlations between columns**
# 
# - reduce sample volume to 300 to get a clearer visualization
# - use seaborn to create a pairplot with all columns that contain metric data
# 
# Insights:
# - The higher the PS, the higher the price
# - The higher the kilometer count, the lower the price
# - The higher the registration date/year, the higher the price

# In[ ]:


g = sns.pairplot(df.sample(300))
plt.show()


# **Not repaired Damages**
# 
# Due to the fact that I have a lot less data on cars that have not repaired Damages than cars that don't have unrepaired damages the graph is not meaningful enough for that column. The reason is that around 80% of the data contain cars that have no unrepaired damage.
# 
# One way to solve that would be to compare identical cars with exactly the same parameters (kilometer, registration, ps, etc.) and see if there are any changes in the price if they have unrepaired damages or not.
# 
# Sadly the data does not contain enough resources to do that.

# In[ ]:


print("cars with unrepaired damage: " + str(len(df[df["notRepairedDamage"] == 1])))
print("cars without unrepaired damage: " + str(len(df[df["notRepairedDamage"] == 0])))
print("total cars: " + str(245541 + 29962))
print("percentage with unrepaired damage: " + str(100 / 275503 * 29962) + "%")
print("percentage without unrepaired damage: " + str(100 / 275503 * 245541) + "%")
labels = 'no unrepaired damage', 'unrepaired damage'
sizes = [89.12461933263884, 10.875380667361155]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Distribution of unrepaired damaged cars in data", fontweight="bold")

plt.show()


# So do I still fit my model with this column?
# 
# Yes, because it is obvious that cars that have unrepaired damages lose value and thus are meaningful for my model to predict prices for common cars. Also, if I cut the number of cars that have no unrepaired damage to the total number of cars that have unrepaired damages and then compare the prices, the mean price is a lot lower on cars with unrepaired damages.

# In[ ]:


print(df[df["notRepairedDamage"]==1]["price"].mean())
print(df[df["notRepairedDamage"]==0]["price"][:29962].mean())


# ### Linear Regression
# 
# * one hot encoding
# * filter common cars and faulty prices
# * train-test-split
# * linear regression
# * r2 score

# **One hot encoding**
# - create new columns based on the nomial values in "vehicleType", "gearbox", "fuelType", "brand" with 1 and 0 values (depending on if the values/parameters are given in the car) to be able to fit them into the linear regression.

# In[ ]:


df2 = pd.get_dummies(df, columns = ["vehicleType", "gearbox", "fuelType", "brand"])
df2.head()


# **filter common cars and faulty prices**
# 
# The prices in the dataset are based on the amount the private seller has set in the specific field on eBay. There are people that set the amount to an extremely low amount, such as 1 euro and then write the real price down in the description, which we can't recognize. 
# 
# To filter *common* cars and remove the faults I mentioned, I will filter the price once again and set the scale between 500 and 20000 for a *common* used cars. This means that the model will only work for those!

# In[ ]:


df2 = df2[(df2["price"] > 500) & (df2["price"] < 20000)]


# In[ ]:


df2.head()


# **train-test-split**
# - split data into train and test data so I will be able to test the accuracy of my model with data it has not seen yet.

# In[ ]:


y = df2[["price"]].values
X = df2.drop(["price", "name"], axis = 1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)


# **train model and calculate r2 score**

# In[ ]:


model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))


# ### Conclusion:
# 
# We managed to achieve a coefficient of determination of around 62% with the model, which is okay but not very good. Sadly, as I mentioned the data has a lot of flaws, such as that the prices are not based on the market, but based on what private sellers type in the specific field on eBay. I tried to filter the data and narrow it down the remove faulty values and only include cars that can be categorized as *common*.
# 
# There can be still done some things, such as include the postal Code and find out whether the locations have an impact on the price. Maybe people have more money in one town and pay more for the same used car? This, for example, can be done through clustering.

# In[ ]:




