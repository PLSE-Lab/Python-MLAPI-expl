#!/usr/bin/env python
# coding: utf-8

# # ** Hey!  It's Black Friday!!!**

# ![](https://media4.s-nbcnews.com/i/newscms/2015_48/871586/black-friday-shoppers-tease-151123_c4900e348d71aae38b400e04421681b2.jpg)

# **Note:**  
# Kindly upvote the kernel if you find it useful. Suggestions are always welome. Let me know your thoughts in the comment if any.

# # Overview of the Dataset
# The dataset here is a sample of the transactions made in a retail store. The store wants to know better the customer purchase behaviour against different products. Specifically, here the problem is a regression problem where we are trying to predict the dependent variable (the amount of purchase) with the help of the information contained in the other variables.
# 
# Classification problem can also be settled in this dataset since several variables are categorical, and some other approaches could be "Predicting the age of the consumer" or even "Predict the category of goods bought". This dataset is also particularly convenient for clustering and maybe find different clusters of consumers within it.
# 
# # Acknowledgements
# The dataset comes from a competition hosted by Analytics Vidhya.

# **0. Global Options**

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# **1. Reading the Olympic Dataset**

# In[ ]:


import pandas as pd
bl_fri = pd.read_csv("../input/black-friday/BlackFriday.csv", header = 'infer')


# **2. Investigating the datasets**

# In[ ]:


print(bl_fri.shape)


# In[ ]:


print(bl_fri.info())


# In[ ]:


bl_fri.head()


# **3. Function to calculate the missing values**

# In[ ]:


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[ ]:


missing_values_table(bl_fri)


# As the missing values are both in category column, I don't want to replace these NaN with imputation. So, I am replacing all the NaN with 0 as the user didn't buy any product in that category.

# In[ ]:


#Replacing the NaN with 0 in columns Product_Category_3 and Product_Category_2
bl_fri['Product_Category_2'] = bl_fri['Product_Category_2'].fillna(0)
bl_fri['Product_Category_3'] = bl_fri['Product_Category_3'].fillna(0)


# **4. Analysis Incoming**  
# We will start the analysis from each of the columns that are available in the dataset and will point out the inferences.

# **5.  Gender Proportion**

# In[ ]:


from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

gen_q = """
select Gender, count(distinct User_ID) as cnt
From bl_fri
GROUP BY Gender;
"""

gen_df = pysqldf(gen_q)


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

fig = {
  "data": [
    {
      "values": gen_df.cnt,
      "labels": gen_df.Gender,
      "domain": {"x": [0, .5]},
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
 "layout": {
        "title":"Gender Proportion on Black Friday"
    }
}

iplot(fig)


# From the above plot we can clearly say that 72% of the population who bought products on Black friday were Men and the rest 28% are female.

# **6.  Age Proportion**

# In[ ]:


from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

age_q = """
select Age, count(distinct User_ID) as cnt
From bl_fri
GROUP BY Age;
"""

age_df = pysqldf(age_q)


# In[ ]:


fig = {
  "data": [
    {
      "values": age_df.cnt,
      "labels": age_df.Age,
      "domain": {"x": [0, .5]},
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
 "layout": {
        "title":"Age Group Proportion on Black Friday"
    }
}

iplot(fig)


# From the above plot its clearly evident that almost 70% of the black friday sales is between the age group of 18 - 45.

# **7. Marriage Vs Age**

# In[ ]:


am_q = """
select Age,  Marital_Status, count(distinct User_ID) as cnt
From bl_fri
GROUP BY Age,  Marital_Status;
"""

am_df = pysqldf(am_q)

am_df_m = am_df[am_df.Marital_Status == 1]
am_df_nm = am_df[am_df.Marital_Status == 0]


# In[ ]:


trace1 = go.Bar(
                x = am_df_m.Age,
                y = am_df_m.cnt,
                name = "Married")

trace2 = go.Bar(
                x = am_df_nm.Age,
                y = am_df_nm.cnt,
                name = "Single")

data = [trace1, trace2]
layout = go.Layout(barmode = "group", title = "Buying volume - Marriage Vs Age")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# There is an interesting buying pattern in the above chart. From the age 18 to 45 married people tend to buy less compared to single. But, the trend gets reversed after the age to 46 as married people buy more compared to single.

# **8. Product Category Vs Gender**
# 
# I have consider each of the different values in the Product category column as categorical number as in the analytics vidhya website where the original dataset was hosted they have the description as category masked which I understand them as categorical variable.
# 
# Refrence:
# https://datahack.analyticsvidhya.com/contest/black-friday/

# In[ ]:


#Product Category 1
gen_prod_1 = """
select Gender, Product_Category_1, count(Product_Category_1) as cnt
From bl_fri
GROUP BY Gender, Product_Category_1;
"""

gen_prod_1_df = pysqldf(gen_prod_1)

gen_prod_1_m = gen_prod_1_df[gen_prod_1_df.Gender == 'M']
gen_prod_1_f = gen_prod_1_df[gen_prod_1_df.Gender == 'F']


# In[ ]:


fig = {
  "data": [
    {
      "values": gen_prod_1_m.cnt,
      "labels": gen_prod_1_m.Product_Category_1,
      "domain": {"x": [0, .48]},
      "name": "Product Category 1",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": gen_prod_1_f.cnt,
      "labels": gen_prod_1_f.Product_Category_1,
      "domain": {"x": [.52, 1]},
      "name": "Product Category 1",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Product Category 1 Vs Gender",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Male",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Female",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}

iplot(fig)


# With respect to Product category 1, Both Male and Female have the same top 3 category [1, 5 and 8] purchases. The top purchase category of Male is Product category 1 which constitiutes to 28.1% of the overall purchase where as the top purchase category of Female is Product category 5 which constitiutes to 31.2% of the overall purchase.  
# 
# Further, the top 3 category purchase of Male occupies 74.1% of their over all purchase whereas the top 3 category purchase of Female occupies 74.6% of their over all purchase.

# In[ ]:


#Product Category 2
gen_prod_2 = """
select Gender, Product_Category_2, count(Product_Category_2) as cnt
From bl_fri
where Product_Category_2 != 0
GROUP BY Gender, Product_Category_2;
"""

gen_prod_2_df = pysqldf(gen_prod_2)

gen_prod_2_m = gen_prod_2_df[gen_prod_2_df.Gender == 'M']
gen_prod_2_f = gen_prod_2_df[gen_prod_2_df.Gender == 'F']


# In[ ]:


fig = {
  "data": [
    {
      "values": gen_prod_2_m.cnt,
      "labels": gen_prod_2_m.Product_Category_2,
      "domain": {"x": [0, .48]},
      "name": "Product Category 2",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": gen_prod_2_f.cnt,
      "labels": gen_prod_2_f.Product_Category_2,
      "domain": {"x": [.52, 1]},
      "name": "Product Category 2",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Product Category 2 Vs Gender",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Male",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Female",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}

iplot(fig)


# With respect to Product category 2, Males top 3 category are 8, 2 and 14 where as Females top 3 category are 14, 8 and 16. The top purchase category of Male is Product category 8 which constitiutes to 16.7% of the overall purchase where as the top purchase category of Female is Product category 14 which constitiutes to 21.5% of the overall purchase.  
# 
# Further, the top 3 category purchase of Male occupies 43.1% of their over all purchase whereas the top 3 category purchase of Female occupies 49.9% of their over all purchase.

# In[ ]:


#Product Category 3
gen_prod_3 = """
select Gender, Product_Category_3, count(Product_Category_3) as cnt
From bl_fri
where Product_Category_3 != 0
GROUP BY Gender, Product_Category_3;
"""

gen_prod_3_df = pysqldf(gen_prod_3)

gen_prod_3_m = gen_prod_3_df[gen_prod_3_df.Gender == 'M']
gen_prod_3_f = gen_prod_3_df[gen_prod_3_df.Gender == 'F']


# In[ ]:


fig = {
  "data": [
    {
      "values": gen_prod_3_m.cnt,
      "labels": gen_prod_3_m.Product_Category_3,
      "domain": {"x": [0, .48]},
      "name": "Product Category 3",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": gen_prod_3_f.cnt,
      "labels": gen_prod_3_f.Product_Category_3,
      "domain": {"x": [.52, 1]},
      "name": "Product Category 3",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Product Category 3 Vs Gender",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Male",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Female",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}

iplot(fig)


# With respect to Product category 3, Males top 3 category are 16, 15 and 17 where as Females top 3 category are 16, 14 and 15. The top purchase category of Male is Product category 16 which constitiutes to 20% of the overall purchase where as the top purchase category of Female is Product category 16 which constitiutes to 18% of the overall purchase.  
# 
# Further, the top 3 category purchase of Male occupies 48.1% of their over all purchase whereas the top 3 category purchase of Female occupies 45.2% of their over all purchase.

# **9. Occupation Vs Gender**

# In[ ]:


gen_occ = """
select Gender, Occupation, count(Occupation) as cnt
From bl_fri
GROUP BY Gender, Occupation;
"""

gen_occ_df = pysqldf(gen_occ)

gen_occ_df_m = gen_occ_df[gen_occ_df.Gender == 'M']
gen_occ_df_f = gen_occ_df[gen_occ_df.Gender == 'F']


# In[ ]:


fig = {
  "data": [
    {
      "values": gen_occ_df_m.cnt,
      "labels": gen_occ_df_m.Occupation,
      "domain": {"x": [0, .48]},
      "name": "Occupation",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },     
    {
      "values": gen_occ_df_f.cnt,
      "labels": gen_occ_df_f.Occupation,
      "domain": {"x": [.52, 1]},
      "name": "Occupation",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Occupation Vs Gender",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Male",
                "x": 0.20,
                "y": 0.5
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Female",
                "x": 0.8,
                "y": 0.5
            }
        ]
    }
}

iplot(fig)


# With respect to Occupation, Males top 3 category are 4, 0 and 7 where as Females top 3 category are 0, 1 and 4. The top occupation category of Male is 4 which constitiutes to 13.2% of the overall occupation categories where as the top occupation category of Female is 0 which constitiutes to 13.4% of the overall occupation categories.  
# 
# Further, the top 3 category occupation of Male occupies 37.5% of their over all occupation categories whereas the top 3 category occupation of Female occupies 39.7% of their over all occupation categories.

# **10. City Vs Average years of stay in the city**

# In[ ]:


import numpy as np
city = """
select distinct User_ID, City_Category, Stay_In_Current_City_Years, Age
From bl_fri
GROUP BY User_ID, City_Category, Stay_In_Current_City_Years, Age;
"""

city_df = pysqldf(city)

city_df['Years'] = np.where(city_df['Stay_In_Current_City_Years']=='4+', '4', city_df['Stay_In_Current_City_Years'])

city_df['Years'] = city_df['Years'].astype(str).astype(int)


# In[ ]:


city_avg = """
select City_Category, Age, round(avg(Years),2) as Avg_yrs
From city_df
GROUP BY City_Category, Age;
"""

city_avg_df = pysqldf(city_avg)

city_avg_A_df = city_avg_df[city_avg_df.City_Category == 'A']
city_avg_B_df = city_avg_df[city_avg_df.City_Category == 'B']
city_avg_C_df = city_avg_df[city_avg_df.City_Category == 'C']


# In[ ]:


trace_A = go.Bar(
                x = city_avg_A_df.Age,
                y = city_avg_A_df.Avg_yrs,
                name = "City - A")

trace_B = go.Bar(
                x = city_avg_B_df.Age,
                y = city_avg_B_df.Avg_yrs,
                name = "City - B")

trace_C = go.Bar(
                x = city_avg_C_df.Age,
                y = city_avg_C_df.Avg_yrs,
                name = "City - C")

data = [trace_A, trace_B, trace_C]
layout = go.Layout(barmode = "group", title = "Average yrs of stay - City Vs Age")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# From the above plot, we can clearly see that the average years of stay for both 26-35 and 36-35 age groupss are almost the same except the little difference in City - B and C. The maximum average years of stay is on 46-50 age group with 2.21 years in City - A.

# **11. Which Age group purchase the most in Dollars based on Gender**

# In[ ]:


ad_dollar = """
select Age, Gender, sum(Purchase) as Dollars
From bl_fri
GROUP BY Age, Gender;
"""

ad_dollar_df = pysqldf(ad_dollar)

ad_dollar_m = ad_dollar_df[ad_dollar_df.Gender == 'M']
ad_dollar_f = ad_dollar_df[ad_dollar_df.Gender == 'F']


# In[ ]:


trace_M = go.Bar(
                x = ad_dollar_m.Age,
                y = ad_dollar_m.Dollars,
                name = "Male")

trace_F = go.Bar(
                x = ad_dollar_f.Age,
                y = ad_dollar_f.Dollars,
                name = "Female")

data = [trace_M, trace_F]
layout = go.Layout(barmode = "group", title = "Purchase in Dollar - Age Vs Gender")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# From the above plot we can clearly see that Males are dominating with respect to dollar values in all the age groups. In the age group of 26-35, the dolllar value is almost 4 times bigger than the purchase of females.

# # **The End! Will see you all in next Kernel!..........**
