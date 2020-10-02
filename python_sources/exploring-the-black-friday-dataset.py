#!/usr/bin/env python
# coding: utf-8

# I will be doing an EDA on this dataset.
# 
# It would be interesting to find out more about the customer purchase behaviour from questions like
# 
# Do women spend more than men
# Which is the top selling product among customers?
# Do people from a certain city category spend more?
# Which occupation spends more?
# 
# # Table of Contents
# 1. <a href="#Data Cleaning">Data Cleaning</a>
# 2. <a href="#Exploratory Data Analysis (EDA)"> Exploratory Data Analysis (EDA)</a>
# 3. <a href="#Analysis"> Analysis</a>

# In[58]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Import relevant packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[59]:


df = pd.read_csv("../input/BlackFriday.csv")


# In[60]:


df.head()


# In[61]:


df.info()


# <a id="Data Cleaning">
# # 1. Data Cleaning
# </a>

# From the sample of the table above, we can see that there are NaN values. Lets check with columns have 'NaN' values'

# In[62]:


# To check which columns contain null values first
df.isna().any()


# In[63]:


# To check the unique values in Product_Category_2 and Product_Category_3 so that no null value would be missed out
print("Before")
print(df["Product_Category_1"].unique())
print(df["Product_Category_2"].unique())
print(df["Product_Category_3"].unique())

# To replace the only 1 null value which is nan values in Product_Category_2 and Product_Category_3 with 0

df["Product_Category_2"].fillna(0, inplace=True)
df["Product_Category_3"].fillna(0, inplace=True)

print("")

print("After")
print(df["Product_Category_1"].unique())
print(df["Product_Category_2"].unique())
print(df["Product_Category_3"].unique())


# Product Categories 2 & 3 are floats, while Product Category 1 is an int. As all the numbers in 2 and 3 have no decimal places, change them to int.

# In[64]:


df = df.astype({"Product_Category_2": int, "Product_Category_3": int})
df.head()


# In[65]:


#checking the results
print(sorted(df["Age"].unique()))
print(sorted(df["Occupation"].unique()))
print(sorted(df["City_Category"].unique()))

# creating a dict file  
#Age_range = {'0-17': 0,'18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}

#Changing the age range to a numerical value
print(sorted(df["Marital_Status"].unique()))


# <a href="Exploratory Data Analysis (EDA)"> 
# # 2. Exploratory Data Analysis (EDA)
# </a>

# In[66]:


df.head()


# In[67]:


#Removing all User_ID duplicates to get an accurate count of gender and age groups
gender_unique_df = df.copy()
gender_unique_df = gender_unique_df.drop_duplicates(subset='User_ID', keep="first")
gender_unique_df.head()


# In[68]:


gender_count_df = gender_unique_df.groupby(["Gender"]).size().reset_index(name="Number of buyers")
gender_count_df


# There are a lot more male buyers

# In[69]:


#create a dataframe to see the breakdown by gender and age
age = gender_unique_df.groupby(["Age"]).size().reset_index(name="counts")
age


# Most of the shoppers are in the 26-35 and 36-45 age group

# In[70]:


customers_city = gender_unique_df.groupby("City_Category").size().reset_index(name = "No. of Customers")
customers_city.head()


# Most of the buyers live in city category C

# In[71]:


occupation_count = gender_unique_df.groupby("Occupation").size().reset_index(name = "Occupation count")
occupation_count.head()


# Most of the buyers either have an occupation of 4 or 0

# In[72]:


marital_count = gender_unique_df.groupby("Marital_Status").size().reset_index(name = "Marital count")
marital_count.head()


# Most buyers are unmarried

# In[73]:


print(sorted (df["Product_Category_1"].unique()))
print(sorted (df["Product_Category_2"].unique()))
print(sorted (df["Product_Category_3"].unique()))


# Product category 1 has no 0 values, so I'll assume that it is the mandatory category

# In[74]:


check = df.copy()
category = df["Product_Category_1"] == 1
check = check[category]
check.head()


# <a href="Analysis">  
# # 3. Analysis 
# </a>

# <a href="Product Category"> 
# ## 3.1 Product Category
# </a>
# I'll first analyse the product category. My assumption here is that the product categories are linked to one another.

# In[75]:


#Creating a combined category field
products = df.copy()
products["Full Category"] = products["Product_Category_1"].apply(str) + ", " + products["Product_Category_2"].apply(str) + ", " + products["Product_Category_3"].apply(str)
products.head()


# In[76]:


# Start by creating the figure and add the subplot
fig1 = plt.figure(figsize=(10,6))
ax1 = fig1.add_subplot(111)

# Find out the total selling quantity of each product category
sales_total_cat1 = df[df["Product_Category_1"] != 0]["Product_Category_1"].count()
sales_total_cat2 = df[df["Product_Category_2"] != 0]["Product_Category_2"].count()
sales_total_cat3 = df[df["Product_Category_3"] != 0]["Product_Category_3"].count()

# Convert the total selling quantity of each product category into a DataFame
df_sales_cat = pd.DataFrame({"Product Category": ["Product_Category_1", "Product_Category_2", "Product_Category_3"],
                            "Selling Quantity": [sales_total_cat1, sales_total_cat2, sales_total_cat3]})

# Plot the bar graph here
df_sales_cat.plot(kind="bar", x="Product Category" ,y="Selling Quantity", ax=ax1, color="skyblue")

# Set the title, label of y-axis of the bar graph
ax1.set_title("Total Selling Quantity of each product category")
ax1.set_ylabel("Selling Quantity")

labels1 = [sales_total_cat1, sales_total_cat2, sales_total_cat3]

for rect, label in zip( ax1.patches, labels1):
    height = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    plt.text(x_value, height + 5, label,
            ha='center', va='bottom')

plt.show()


# From the bar graph, we know the top selling product category is Product_Category_1. We would like to explore further on the sales performance of the sub-categories within each category.

# In[77]:


# Find out the selling quantity of the sub-catagories within each product category
sales_cat1 = products.groupby("Product_Category_1").size().reset_index(name="Selling Quantity")
sales_cat2 = products.groupby("Product_Category_2").size().reset_index(name="Selling Quantity")
sales_cat3 = products.groupby("Product_Category_3").size().reset_index(name="Selling Quantity")
sales_combined = products.groupby("Full Category").size().reset_index(name="Selling Quantity")

# Drop the row where sub-category is 0
sales_cat2 = sales_cat2.drop([0])
sales_cat3 = sales_cat3.drop([0])


# In[78]:


#only taking the top 10 combined category products as the resultant plot is too large
sales_combined.sort_values(by=['Selling Quantity'], inplace=True,  ascending=False)
top15_sales = sales_combined.head(15)


# Based on the EDA, I can see that products with the same product categories across 1, 2 and 3 do not have the same price. I'll check the data to see if any of them have the same price.

# In[79]:


purchase_fullcat = products.groupby(["Full Category",  "Purchase"])["Product_ID"].count().reset_index(name = "cost count")
purchase_fullcat.head()


# From the dataframe, items may have the same classification but have different purchase amounts.

# In[80]:


price = products.copy()
comb_cat_price = price.groupby(["Full Category"])["Purchase"].mean().reset_index(name="Cost")
cat1_price = products.groupby("Product_Category_1")["Purchase"].mean().reset_index(name="Cost")
cat2_price = products.groupby("Product_Category_2")["Purchase"].mean().reset_index(name="Cost")
cat3_price = products.groupby("Product_Category_3")["Purchase"].mean().reset_index(name="Cost")

# Drop the row where sub-category is 0
cat2_price = cat2_price.drop([0])
cat3_price = cat3_price.drop([0])

#only taking the top 10 combined category products as the resultant plot is too large
comb_cat_price.sort_values(by=['Cost'], inplace=True,  ascending=False)
top15_price = comb_cat_price.head(15)

#Retrieving the price of the top selling products
cond_top_15_Sold = comb_cat_price["Full Category"].isin(top15_sales["Full Category"])
top_15_sold_price = comb_cat_price[cond_top_15_Sold]


# In[81]:


#Combining the selling quantity and cost together
cat1 = pd.merge(cat1_price, sales_cat1,  on='Product_Category_1')
cat1 = cat1.sort_values('Cost', ascending=False)

cat2 = pd.merge(cat2_price, sales_cat2,  on='Product_Category_2')
cat2 = cat2.sort_values('Cost', ascending=False)

cat3 = pd.merge(cat3_price, sales_cat3,  on='Product_Category_3')
cat3 = cat3.sort_values('Cost', ascending=False)

top15 = pd.merge(top_15_sold_price, top15_sales, on='Full Category')
top15 = top15.sort_values('Cost', ascending=False)


# In[82]:


# Amt earned
comb_cat_earn = price.groupby(["Full Category"])["Purchase"].sum().reset_index(name="Earn")
cat1_earn = products.groupby("Product_Category_1")["Purchase"].sum().reset_index(name="Earn")
cat2_earn = products.groupby("Product_Category_2")["Purchase"].sum().reset_index(name="Earn")
cat3_earn = products.groupby("Product_Category_3")["Purchase"].sum().reset_index(name="Earn")

# Drop the row where sub-category is 0
cat2_earn = cat2_earn.drop([0])
cat3_earn = cat3_earn.drop([0])

#only taking the top 10 combined category products as the resultant plot is too large
comb_cat_earn.sort_values(by=['Earn'], inplace=True,  ascending=False)
top15_earn = comb_cat_earn.head(15)


# In[83]:


# Create the figure and add the subplot
fig1 = plt.figure(figsize=(10,20))
ax1_0 = fig1.add_subplot(511)
ax1_1 = fig1.add_subplot(512)
"""ax2_0 = fig1.add_subplot(513)
ax2_1 = fig1.add_subplot(514)
ax3_0 = fig1.add_subplot(515)
ax3_1 = fig1.add_subplot(516)
ax4_0 = fig1.add_subplot(517)"""

width=0.20

# Plot the sorted bar graph (Cat 1) according to cost
cat1.plot( 
    kind="bar", x="Product_Category_1" ,y="Cost", ax=ax1_0, position=2, width=width, color='orange')

# Plot the sorted bar graph (category 1) according to qty sold
cat1.plot(
    kind="bar", x="Product_Category_1" ,y="Selling Quantity", ax=ax1_0, position=1, width=width, color='purple')

ax1_0.set_title("Cost of sub-categories in category 1 compared to quantity sold")
ax1_0.set_xlabel("Cost/Selling Qty")
ax1_0.set_ylabel("sub-categories in category 1")

#plot earnings
cat1_earn.sort_values(by = "Earn", ascending = False).plot(
    kind="bar", x="Product_Category_1" ,y="Earn", ax=ax1_1, color='tomato')

ax1_1.set_title("Earnings of sub-categories in category 1")
ax1_1.set_xlabel("Earnings")
ax1_1.set_ylabel("sub-categories in category 1")

fig1.tight_layout()

plt.show()


# Based on the graphs, we can see that the top selling sub-category in Product_Category_1 is sub-category 5, the top selling sub-category in Product_Category_2 is sub-category 8 and the top selling sub-category in Product_Category_3 is sub-category 16. Specifically, items classified as Product_Category_1 sub-category 5, Product_Category_2 sub-category 0, Product_Category_3 sub-category 0 has the highest selling quantity.
# 
# 

# <a href="Gender"> 
# ## 3.2 Gender
# </a>
# Analysing the gender and purchases made. From the EDA above, we can see that there are more male than female buyers

# In[84]:


gender_df = df.copy()
gender_df = gender_df.groupby(["Gender"])["Purchase"].sum().reset_index(name = "Total Purchase ($)")
gender_df


# In[85]:


# Create the figure and add the subplot
fig2 = plt.figure(figsize=(8,5))
ax2 = fig2.add_subplot(121)
ax3 = fig2.add_subplot(122)

# Create a pie chart
ax2.pie(
    gender_count_df['Number of buyers'],
    explode = None,
    labels=gender_count_df['Gender'],
    #colors = colors,
    autopct='%1.1f%%', 
    shadow=False,
    )

# Create the line chart
gender_df.plot(kind='bar', x='Gender', y='Total Purchase ($)', ax=ax3, legend=None)

# Set the title here
#
ax3.set_title("Total Purchase by Gender")

# Set the y axis label
#
ax3.set_ylabel("Total Purchase ($) Billion")

labels1 = ["1164624021", "3853044357"]

# To add the number label on top of the bars
for rect, label in zip( ax3.patches, labels1):
    height = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    plt.text(x_value, height + 5, label,
            ha='center', va='bottom')

# View the plot
plt.tight_layout()
plt.show()


# The bar graph shows that the total purchase amount by male customers is more than that by female customers. It would be interesting to find out the difference in purchasing between male and female customers by comparing the median purchase amount of each gender through a boxplot.

# In[162]:


# Lets create a datafram that sums everything up by customer
total_by_cust = df.groupby(["User_ID", "Gender", "Age", "Occupation", "City_Category","Marital_Status", "Stay_In_Current_City_Years"])["Purchase"].sum().reset_index(name="Total Purchase")
total_by_cust.head()


# In[87]:


# Create a box plot of the spending of females vs males
# For a more accurate comparison
fig4 = plt.figure(figsize=(6,14))
ax4 = fig4.add_subplot(111)

sns.boxplot(x = 'Gender', y = 'Total Purchase', data = total_by_cust, ax=ax4)
ax4.set_title("Boxplot of cutomer's individual purchase amount by gender")
plt.show()


# In[88]:


#mean, median and mode of the 2 purchasing powers of the gender
print(total_by_cust.groupby("Gender").mean())
print()
print(total_by_cust.groupby("Gender").median())
print()
print("Females spend about " + str((911963.16-699054.03)/(911963.16 + 699054.03)*100) + "% less than males")
#We can see that Males spend more than females.


# In[89]:


#calculate the number of females/males that spend above the IQR


# In[90]:


#are females buying more of a certain category that is more expensive?
#fig5 = plt.figure(figsize=(6,14))


# Lets look into the breakdown by age group as well

# In[100]:


#create a dataframe to see the breakdown by gender and age
gender_age = gender_unique_df.groupby(["Age", "Gender"]).size().reset_index(name="counts")
gender_age_purchase = gender_unique_df.groupby(["Age", "Gender"])["Purchase"].sum().reset_index(name="Purchase Total")
gender_age_ave_purchase = gender_unique_df.groupby(["Age", "Gender"])["Purchase"].median().reset_index(name="Purchase Total")


# In[92]:


gender_age_pivot = pd.pivot_table(
    gender_age,
    index="Age",
    columns="Gender",
    values="counts",
    aggfunc=sum
)

gender_age_pivot.columns = ["F", "M"]
gender_age_pivot = gender_age_pivot.reset_index()
gender_age_pivot


# In[93]:


gender_age_purchase_pivot = pd.pivot_table(
    gender_age_purchase,
    index="Age",
    columns="Gender",
    values="Purchase Total",
    aggfunc=sum
)

gender_age_purchase_pivot.columns = ["F", "M"]
gender_age_purchase_pivot = gender_age_purchase_pivot.reset_index()
gender_age_purchase_pivot


# In[101]:


gender_age_ave_purchase_pivot = pd.pivot_table(
    gender_age_ave_purchase,
    index="Age",
    columns="Gender",
    values="Purchase Total",
    aggfunc=sum
)

gender_age_ave_purchase_pivot.columns = ["F", "M"]
gender_age_ave_purchase_pivot = gender_age_ave_purchase_pivot.reset_index()
gender_age_ave_purchase_pivot


# In[109]:


width=0.20

#Create the figure
fig6 = plt.figure(figsize=(15, 15))

#Add the subplot
ax6 = fig6.add_subplot(221)
ax7 = fig6.add_subplot(222)
ax8 = fig6.add_subplot(223)

#Plot the number of buyers by gender and age group
gender_age_pivot.plot(kind='bar', x='Age', y='F', 
                            ax=ax6, position=1, width=width, color='purple')
gender_age_pivot.plot(kind='bar', x='Age', y='M', 
                            ax=ax6, position=2, width=width, color='tomato')

# Add the title of the plot
ax6.set_title("No. of Buyers by age group and gender")
ax6.set_ylabel("Number of Buyers")

#Plot the purchase by gender and age group
gender_age_purchase_pivot.plot(kind='bar', x='Age', y='F', 
                            ax=ax7, position=1, width=width, color='purple')
gender_age_purchase_pivot.plot(kind='bar', x='Age', y='M', 
                            ax=ax7, position=2, width=width, color='tomato')

ax7.set_title("Purchase by age group and gender")
ax7.set_ylabel("Total Purchase (Billions)")

#Plot the ave purchase by gender and age group
gender_age_ave_purchase_pivot.plot(kind='bar', x='Age', y='F', 
                            ax=ax8, position=1, width=width, color='purple')
gender_age_ave_purchase_pivot.plot(kind='bar', x='Age', y='M', 
                            ax=ax8, position=2, width=width, color='tomato')

ax8.set_title("Median Purchase by age group and gender")
ax8.set_ylabel("Median Purchase (Billions)")

#Finally, show the plot
fig6.tight_layout()

plt.show()


# Based on the 2 graphs, the total purchase made by each gender and age group is proportionate to the number of buyers in each gender and age group. Most of the buyers are from the 26-35 age group for both genders and they also spent the most. However, it is interesting to note that the median purchase made by each age group and gender doesn't differ by much.

# In[95]:


#Look at amount spend by each age group (bar and box plots)
#fig9 = plt.figure(figsize=(18, 15))


# In[ ]:


#Look at the breakdown of the categories for each age group/gender
#fig10 = plt.figure(figsize=(18, 15))


# <a href="Occupation"> 
# ## 3.3 Occupation
# </a>
# Analysing the occupation and purchases made. From the EDA above, we can see that most of the buyers have an Occupation of 4.

# In[114]:


occupation_count.sort_values(by=['Occupation count'], inplace=True,  ascending=False)
occupation_count.head()


# In[115]:


# Find out the total purchase amount of each occupation
total_purchase_occupation = df.groupby("Occupation")["Purchase"].sum().reset_index(name = "Total Amount")
total_purchase_occupation.head()


# In[127]:


occupation = pd.merge(total_purchase_occupation, occupation_count,  on='Occupation')
occupation.head()


# In[118]:


fig11 = plt.figure(figsize=(12,6))
ax11 = fig11.add_subplot(111)

sns.boxplot(x = 'Occupation', y = 'Total Purchase', data = total_by_cust, ax=ax11)
ax11.set_title("Boxplot of purchase amount by occupation")
plt.show()


# In[121]:


#In general, there seem to be quite a lot of outliers. Therefore, the median will be used
#Median spending by occupation
median_purchase_occupation = df.groupby("Occupation")["Purchase"].median().reset_index(name = "Median Amount")
median_purchase_occupation.head()


# In[135]:


occupation.head()


# In[147]:


# Create the figure and add the subplot
fig12 = plt.figure(figsize=(12,8))
ax12 = fig12.add_subplot(221)
ax13 = fig12.add_subplot(222)
ax14 = fig12.add_subplot(223)

colors = {0: 'gold', 1: 'navy', 2: 'orange', 3: 'g', 4: 'r', 5: 'purple', 6:'chocolate', 7:'m', 8:'slategrey', 
          9:'darkolivegreen', 10:'teal', 11:'deepskyblue', 12:'darkseagreen', 13:'lightcoral',
          14:'tan', 15:'bisque', 16:'mediumaquamarine', 17:'violet', 18:'thistle', 19:'pink', 20:'steelblue'}

# Plot the sorted bar graph here (Total Purchase Amount)
occupation.sort_values(by = "Total Amount", ascending = False).plot(
    kind="bar", x="Occupation" ,y="Total Amount", ax=ax12, legend = False, 
    color=[colors[i] for i in occupation['Occupation']])

ax12.set_title("Total purchase amount by occupation")
ax12.set_ylabel("Purchase amount")

# Plot the sorted bar graph here (Median Purchase Amount)
median_purchase_occupation.sort_values(by = "Median Amount", ascending = False).plot(
    kind="bar", x="Occupation" ,y="Median Amount", ax=ax13, legend = False,
    color=[colors[i] for i in occupation['Occupation']])
ax13.set_title("Median purchase amount by occupation")
ax13.set_ylabel("Purchase amount")

# Plot number of buyers
occupation.sort_values(by = "Occupation count", ascending = False).plot(
    kind='bar', x='Occupation', y='Occupation count', ax=ax14, 
    color=[colors[i] for i in occupation['Occupation']])

ax14.set_title("Buyers by occupation")
ax14.set_ylabel("Number of buyers")

fig12.tight_layout()

plt.show()

#I have no idea why the colours are not showing up by group


# Based on the graphs 'Buyers by occupation' and 'Total purchase amount by occupation', many buyers have the occupation 4, 0 and 7 and they spend the most money. Most of the total amounts are in proportion to the number of buyers. It is interesting to note that the median purchase doesn't differ much by occupation

# <a href="City"> 
# ## 3.4 City
# </a>
# Analysing the city and purchases made. From the EDA above, we can see that most of the buyers are from City category C

# In[107]:


customers_city.head()


# Does this mean that city C generates the most purchases?

# In[156]:


# Find out the total purchase amount of each city category
total_purchase_city = total_by_cust.groupby("City_Category")["Total Purchase"].sum().reset_index()
total_purchase_city.head()

# Find out the median purchase amount of each city category
median_purchase_city = total_by_cust.groupby("City_Category")["Total Purchase"].median().reset_index(name = "Median Amount")
median_purchase_city.head()


# In[157]:


# Create the figure and add the subplot
fig15 = plt.figure(figsize=(16,6))
ax15 = fig15.add_subplot(121)
ax16 = fig15.add_subplot(122)

# Plot the bar graph here
total_purchase_city.plot(kind="bar", x="City_Category" ,y="Total Purchase", ax=ax15)
ax15.set_title("Total purchase amount by city category")
ax15.set_ylabel("Total amount")

median_purchase_city.plot(kind="bar", x="City_Category" ,y="Median Amount", ax=ax16)
ax16.set_title("Median purchase amount by city category")
ax16.set_ylabel("Median amount")

plt.show()


# 17.7% of the buyers come from category A city, 29% of them come from cat-B city and 53.3% of them come from cat-C city.
# 
# Even though more than half of the customers who visited this retail store come from category C city, category B city had the greatest total purchase amount and mean purchase amount. Another observation is customers from category A city spend more on average than the category C city customers even though the number of customers coming from category A city is the least.
# 
# Are people in A or B buying many items or just a select few?

# In[158]:


item_count_city = df.groupby("User_ID")["Product_ID"].count().reset_index(name = "Item Count")
item_count_city.head()


# In[163]:


cust_city = total_by_cust.drop(['Gender', 'Age', "Occupation",  
                                "Stay_In_Current_City_Years", "Marital_Status", "Total Purchase"], axis=1)
cust_city.head()


# In[164]:


def get_city(x): 
    user = total_by_cust["User_ID"] == x
    line = total_by_cust[user]
    city_letter = line["City_Category"].iloc[0]
    return city_letter


# In[ ]:


item_count_city["City"] = df["User_ID"].apply(get_city)
item_count_city.head()


# In[ ]:


#Are people in city a/b buying more items?

# Create the figure and add the subplot
fig16 = plt.figure(figsize=(16,6))
ax17 = fig16.add_subplot(121)
ax18 = fig16.add_subplot(122)

# Find out the mean purchase amount of each city category
mean_count_items = item_count_city.groupby("City")["Item Count"].mean().reset_index(name = "Mean Amount")
#print(mean_count_items)

median_count_items = item_count_city.groupby("City")["Item Count"].median().reset_index(name = "Median Amount")
#print(mean_count_items)

# Plot the mean bar graph
mean_count_items.plot(kind="bar", x="City" ,y="Mean Amount", ax=ax17)
ax17.set_title("Mean number of items purchased by city category")
ax17.set_ylabel("Mean number of items purchased")

# Plot the median bar graph
median_count_items.plot(kind="bar", x="City" ,y="Median Amount", ax=ax18)
ax18.set_title("Median number of items purchased by city category")
ax18.set_ylabel("Median number of items purchased")

plt.show()


# With the graph above, buyers in all 3 cities are buying a similar number of items for each purchase.
# 
# Even though most customers come from city category C and buy a similar number of items as the other 2 city categories, they still spend the least. This means that customers from category C are probably buying cheaper goods. 

# <a href="Marriage"> 
# ## 3.5 Marriage
# </a>
# From the EDA above, we can see that most of the customers are married

# In[165]:


marital_count


# In[166]:


total_by_cust.head()


# In[178]:


# Find out the total purchase amount of married or single customers with different gender
purchase_marital_gender = total_by_cust.groupby(
    ["Marital_Status", "Gender"])["Total Purchase"].sum().reset_index()

# Convert the purchase_marital_gender dataframe into a pivot table
purchase_pivot2 = purchase_marital_gender.pivot_table(index = "Marital_Status",
                                                      columns = "Gender",
                                                     values = "Total Purchase",
                                                     aggfunc=np.sum)

purchase_pivot2.head()


# In[179]:


# Create the figure and add the subplot
fig19 = plt.figure(figsize=(8,8))
ax19 = fig19.add_subplot(111)

# Plot the stacked bar chart here
purchase_pivot2.plot(kind="bar", stacked = True, ax=ax19)
ax19.set_title("Total purchase amount of married and single customers by gender")
ax19.set_ylabel("Total amount")

plt.show()


# From the stacked bar chart above, single customers generally spend more than married customers on Black Friday and single male customers purchase the most among the 4 groups of customers.

# In[192]:


# Find out the total purchase amount of married or single customers with age groups
purchase_marital_gender = total_by_cust.groupby(
    ["Marital_Status", "Age"])["Total Purchase"].sum().reset_index()

purchase_marital_gender_pivot = pd.pivot_table(
    purchase_marital_gender,
    index="Age",
    columns="Marital_Status",
    values="Total Purchase",
    aggfunc=sum
)

purchase_marital_gender_pivot.columns = ["0", "1"]
purchase_marital_gender_pivot = purchase_marital_gender_pivot.reset_index()

purchase_marital_gender_pivot["1"].fillna(0, inplace=True)
purchase_marital_gender_pivot = purchase_marital_gender_pivot.astype({"0": int, "1": int})


purchase_marital_gender_pivot


# In[209]:


# Find out the total purchase amount of married or single customers with age groups
median_purchase_marital_gender = total_by_cust.groupby(
    ["Marital_Status", "Age"])["Total Purchase"].median().reset_index(name="Median Purchase")

median_purchase_marital_gender_pivot = pd.pivot_table(
    median_purchase_marital_gender,
    index="Age",
    columns="Marital_Status",
    values="Median Purchase",
    aggfunc=np.median,
    fill_value=0
)

median_purchase_marital_gender_pivot.columns = ["0", "1"]
median_purchase_marital_gender_pivot = median_purchase_marital_gender_pivot.reset_index()

median_purchase_marital_gender_pivot


# In[267]:


median_purchase_marital_gender_pivot["total"] = median_purchase_marital_gender_pivot["0"] + median_purchase_marital_gender_pivot["1"]
    
median_perc = median_purchase_marital_gender_pivot.copy()

median_perc['0 %'] = round(median_perc['0'] / median_perc['total'] * 100,2)
median_perc['1 %'] = round(median_perc['1'] / median_perc['total'] * 100,2)
#print(median_perc)

perc = []
for index, row in median_perc.iterrows():
    perc.append(str(row["0 %"]) + "%")

for index, row in median_perc.iterrows():
    perc.append(str(row["1 %"]) + "%")

perc


# In[272]:


width=0.20

#Create the figure
fig20 = plt.figure(figsize=(10, 9))

#Add the subplot
ax20 = fig20.add_subplot(211)
ax21 = fig20.add_subplot(212)

#Plot the values (Total Purchase)
purchase_marital_gender_pivot.plot(kind='bar', x='Age', y='0', 
                            ax=ax20, position=2, width=width, color='navy')
purchase_marital_gender_pivot.plot(kind='bar', x='Age', y='1', 
                            ax=ax20, position=1, width=width, color='limegreen')

#Add the title of the plot
ax20.set_title("Total Purchase by age group and maritial status")
ax20.set_ylabel("Total Purchase (Billion)")

#Plot the values (Median Purchase)
median_purchase_marital_gender_pivot.plot(kind='bar', x='Age', y='0', 
                            ax=ax21, position=2, width=width, color='navy')
median_purchase_marital_gender_pivot.plot(kind='bar', x='Age', y='1', 
                            ax=ax21, position=1, width=width, color='limegreen')

#Add the title of the plot
ax21.set_title("Median Purchase by age group and maritial status")
ax21.set_ylabel("Median Purchase $")

# To add the number label on top of the bars
for rect, label in zip( ax21.patches, perc):
    height = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    ax21.text(x_value, height + 5, label,
            ha='left', va='bottom')

#Finally, show the plot
plt.tight_layout()
plt.show()


# Based on the first graph above, we can see that singles aged 0-45 spend more than their married counterparts. However, this changes for ages 46 and above.
# 
# Based on the second graph above, aside from customers aged 0-17, the median spending for each age group is similar. The unmarried cutomers in the age group of 18-25 and 55+ have about a 8-10% higher median purchase amount compared to the unmarried customers in the same age group.
# 
# In general, unmarried customers from 0-35 and 55+ have a higher median purchase amount than their married counterparts. However, the married customers from ages 36-55 spend slightly more than their unmarried counterparts.

# In[ ]:




