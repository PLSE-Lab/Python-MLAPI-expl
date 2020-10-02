#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

import os
print(os.listdir("../input"))


# # Topics covered here:
# * Does money makes people happier?<br>
# * [Predicting Life Satisfaction Score using GDP and Scikit-Learn](#Predicting-Life-Satisfaction-Score-using-GDP-and-Scikit-Learn)

# Importing GDP(PPP) data.

# In[ ]:


gdp = pd.read_excel("../input/gdp-per-capita-19902017/GDP(PPP) Per capita 1990-2017.xlsx")
gdp.head()


# Importing Life Satisfaction Score data.

# In[ ]:


lss = pd.read_excel("../input/life-satisfaction-score-20152017/Life Satisfaction (2015-2017).xlsx")
lss.columns = lss.iloc[0]
lss = lss.iloc[1:].reset_index(drop=True)
lss.head()


# Our Life Satisfaction Score (LSS) data contains information from 2015 to 2017. So let's get GDP corresponding to that space of time.

# In[ ]:


# Filtering  Data from 2015-2017
gdp = gdp[["Country Name", 2015, 2016, 2017]]
# Getting average value
gdp["2015-2017"] = gdp[[2015, 2016, 2017]].mean(axis=1)
# Deleting rows: 2015, 2016, 2017
gdp = gdp[["Country Name", "2015-2017"]]

gdp.head()


# As we can notice not all the countries from GDP data has their LSS data available, so we will take the countries with both data available.

# In[ ]:


gdp_for_lss = []
for country in lss["Country"]:
    gdp_for_lss.append(list(gdp[gdp["Country Name"] == country]["2015-2017"]))

# Empty values -> None (to avoid future problems in DataFrames)
for i in range(len(gdp_for_lss)):
    if gdp_for_lss[i] == []:
        gdp_for_lss[i] = None
    else:
        gdp_for_lss[i] = round(gdp_for_lss[i][0], 2)

# First 5 values
gdp_for_lss[:5]


# Let's collect everything we need.

# In[ ]:


gdp_lss = pd.DataFrame({"Country": lss["Country"], 
                        "LSS": lss["Life Satisfaction AVG Score"],
                       "GDP(PPP)": gdp_for_lss})

# Deleting rows containing empty values in GDP(PPP) column
gdp_lss = gdp_lss[np.isfinite(gdp_lss["GDP(PPP)"])].reset_index(drop=True)

# Saving this dataframe to share with other kaggle users
gdp_lss.to_csv("gdp_lss.csv")

# First 5 values of our dataframe
gdp_lss.head()


# Now let's plot our dataframe.

# In[ ]:


plt.figure(dpi = 200)
plt.scatter(x = gdp_lss["GDP(PPP)"], y = gdp_lss["LSS"])
plt.xlabel("GDP(PPP) per capita")
plt.ylabel("Life Satisfaction Score")


# As we can see our data points ("dots") are tend to go to the right corner. It means that money can make people a little bit happier;)

# # Predicting Life Satisfaction Score using GDP and Scikit-Learn

# To check that our predicting model is working efficiently let's remove one countrie's LSS from our dataframe and try to predict it.<br>
# For example let's delete France's data.

# In[ ]:


# France's index in our dataframe (df) is 8, GDP is 42269.59, LSS is 6.5
gdp_lss = gdp_lss.drop(8).reset_index(drop=True)
# Printing first 10 values
gdp_lss[:10]


# Preparing the data.

# In[ ]:


x = pd.DataFrame(gdp_lss["GDP(PPP)"])
y = pd.DataFrame(gdp_lss["LSS"])


# Selecting a linear model

# In[ ]:


model = sklearn.linear_model.LinearRegression()


# Training the model

# In[ ]:


model.fit(x, y)


# Making a prediction for France

# In[ ]:


x_new = [[42269.59]] # France's GDP Per capita 
print("Predicted LSS: ", model.predict(x_new))


# France's actual LSS is **6.5**, so our predicting model is **98.5%** accurate.
