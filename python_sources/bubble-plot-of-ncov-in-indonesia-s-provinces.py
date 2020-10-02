#!/usr/bin/env python
# coding: utf-8

# In this project, we are going to visualize our data by using scatter plot. We try to correlate between total confirmed cases against total victims died due to Covid-19 in some Indonesian provinces. However, I am also adding population of each province as my third variable, just to make them like bubble plot. So firstly, let's import our library to be used in this project.

# In[ ]:


import pandas as pd                      #For importing our datasets
import matplotlib.pyplot as plt          #To visualize our data
import numpy as np                       #To do an array operation
from sklearn import linear_model as lm   #To build the linear regression model 
plt.style.use("ggplot")                  #To set the style as "ggplot" in R


# Firstly, we are going to import our dataset by using Pandas. 
# The dataset was obtained from these resources: 
# 
# 1.) For Confirmed_cases, Recovered_cases, and Death_cases (7 July 2020): https://data.humdata.org/dataset/indonesia-covid-19-cases-recoveries-and-deaths-per-province
# 
# 2.) For Populations: https://www.bps.go.id/statictable/2014/02/18/1274/proyeksi-penduduk-menurut-provinsi-2010---2035.html

# In[ ]:


df = pd.read_excel("../input/indonesian-covid19-data-7-july-2020/data covid.xlsx")

df.head()


# However, we only choose some variables in our experiment. I only filtered some columns such as Province_name, Confirmed_cases, Recovered_cases, Death_cases, and Populations.

# In[ ]:


dacov = df.loc[:, "Province_name" : "Populations"]
dacov.head()


# Next, we need to check whether there are some missing values in our datasets. By using notnull(), it would return "True", if the value was not missing. From the results we found that the number of True for each column is 33, meaning that all columns have no missing values.

# In[ ]:


print("The number of data sets :", df.shape[0])

for i in range(5):
    
    x = dacov.iloc[:,i].notnull().value_counts()   
    
    print(x)


# We will also conduct the descriptive statistics to see the mean, median, minimum, and maximum value. 

# In[ ]:


dacov.describe()


# We will also fetch another dataset for the top 5 provinces with the high death cases due to Covid-19 in Indonesia

# In[ ]:


dacov.sort_values(by = "Death_cases", ascending = False).head()


# In[ ]:


#dacov = np.log(dacov.loc[:, "Confirmed_cases":"Populations"])


# We are going to visualize our data by using scatter plots. X-Axis will be labelled as total confirmed cases, Y-Axis will be labelled as total patients died. Besides, we also use the population size to resize the plot. We also include the dataset for the top 5 provinces with the high death rate. As we can see, those five provinces namely Jawa timur (1020 death cases), DKI Jakarta (649 death cases), Kalimantan Selatan (200 death cases), Jawa Tengah (200 death cases), and Sulawesi Selatan (199 death cases).

# In[ ]:


plt.figure(figsize = (30, 20))
plt.grid(linestyle = '--', linewidth = 2)

#To visualize the whole countries

plt.scatter(x = dacov["Confirmed_cases"], y = dacov["Death_cases"], s = dacov["Populations"]/5, 
            alpha = 0.6, c = 'b')

#To only visualize top five-country

top_five = dacov.sort_values(by = "Death_cases", ascending = False).head()

plt.scatter(x = top_five["Confirmed_cases"], y = top_five["Death_cases"], 
            s = top_five["Populations"]/5, alpha = 0.6)

plt.title("Total Confirmed Cases against Total Patients Died in Indonesia", size = 45, pad =35) #pad works as the space between the title and graph
plt.xlabel("Total Confirmed Cases", size = 35, labelpad = 30) #labelpad works as the space between the label and graph
plt.ylabel("Total Patients Died", size = 35, labelpad = 30) 

plt.xticks(size = 23)
plt.yticks(size = 23)


plt.annotate("Jawa Timur", xy = (12000, 1020), size = 40, color = 'black')
plt.annotate("DKI Jakarta", xy = (12435, 649), size = 40)
plt.annotate("Kalimantan \nSelantan", xy = (2200, 200), size = 35)
plt.annotate("Jawa Tengah", xy = (4611, 230), size = 35)
plt.annotate("Sulawesi Selatan", xy = (5890, 150), size = 35)


# Although I am not going to build regression model in this notebook. However, I am going to add the regression line for this scatter plot. Firstly, I need to fetch three values to visualize my regression line namely, intercept, Y-prediction, and coefficient values.

# In[ ]:


regress = lm.LinearRegression()

x_var = np.array(dacov["Confirmed_cases"]) 
y_var = np.array(dacov["Death_cases"])

x_var = x_var.reshape(-1, 1)
y_var = y_var.reshape(-1, 1)

regress.fit(x_var, y_var)

print("Intercept = ", regress.intercept_)
print("Coefficient = ", regress.coef_)

y_predict = regress.predict(x_var)


# After we obtained the intercept and coefficient values, we will draw the regression line in our scatter plots

# In[ ]:


plt.figure(figsize = (30, 20))
plt.grid(linestyle = '--', linewidth = 2)

#To visualize the whole countries

plt.scatter(x = dacov["Confirmed_cases"], y = dacov["Death_cases"], s = dacov["Populations"]/5, 
            alpha = 0.6, c = 'b')

#To only visualize top five-country

top_five = dacov.sort_values(by = "Death_cases", ascending = False).head()

plt.scatter(x = top_five["Confirmed_cases"], y = top_five["Death_cases"], 
            s = top_five["Populations"]/5, alpha = 0.6)

plt.title("Total Confirmed Cases against Total Patients Died in Indonesia", size = 45, pad =35)  #pad works as the space between the title and graph
plt.xlabel("Total Confirmed Cases", size = 35, labelpad = 30) #labelpad works as the space between the label and graph
plt.ylabel("Total Patients Died", size = 35, labelpad = 30)   

plt.xticks(size = 23)
plt.yticks(size = 23)


plt.annotate("Jawa Timur", xy = (12000, 1020), size = 40, color = 'black')
plt.annotate("DKI Jakarta", xy = (12435, 649), size = 40)
plt.annotate("Kalimantan \nSelantan", xy = (2200, 200), size = 35)
plt.annotate("Jawa Tengah", xy = (4611, 230), size = 35)
plt.annotate("Sulawesi Selatan", xy = (5890, 150), size = 35)

plt.annotate('y = -22.004+0.061X', xy = (2000, 800), size = 40)

plt.plot(x_var, y_predict, linewidth = 3)


# Since I visualized the data due to my boredom in quarantine, kindly let me know some suggestions or revisions regarding to this visualization in this nice forum, and I am looking forward to your favorable feedback. Thank you. 

# In[ ]:




