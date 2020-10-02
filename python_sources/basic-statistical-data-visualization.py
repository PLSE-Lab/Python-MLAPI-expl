#!/usr/bin/env python
# coding: utf-8

# ### Some comparsion charts about two popular brands
# * Barplot
# * Countplot
# * Scatterplot
# creating a lightweight Seaborn based barchart, in order to compare the mean price of the uploaded BMWs and AUDIs during the years.
# For to start:
# * SQL connection and query for the brand, price, and upload date (just month data relevant now)
# 

# In[ ]:


import sqlite3
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import datetime

sql_database = '/kaggle/input/secondhand-car-market-data-parsing-dataset-v1/kaggle_sqlite'
conn = sqlite3.connect(sql_database)

test_query_df_v3 = pd.DataFrame()
test_query_df_v3 = pd.read_sql("""
                            SELECT 
                                brand_name as 'brand',
                                strftime('%m', upload_date) as 'upload_month',
                                ad_price as 'price',
                                mileage as 'mileage'
                            FROM advertisements
                            JOIN brand ON advertisements.brand_id = brand.brand_id
                            WHERE brand_name = 'BMW'
                            OR brand_name = 'AUDI';
                            """,
                            conn,
                            parse_dates=['upload_date'])  #parsing the date correctly


# ### Seaborn.barplot
# * creating a barplot with seaborn:
#     * X value is the month
#     * Y value is the price
#     * the two brands had been separated (coloring legend)
# * highlighting the values above all the columns:
#     * creating a grouped version of the original database
#     * iterating over the months
#     * for each iteration setting the x,y coordinates and the value (as text) for both of the brands
# * calibrating the axes of the output chart
# 
# As you can see the average price of BMWs is significantly higher than the mean price of Audis. BMWs popularity, reputation (you can decide if it is positive or negative) and brand's image is unquestionably the number one.

# In[ ]:


plt.figure(figsize=(25,15))
sns.set(font_scale=1.0)  #using the default font size
bc = sns.barplot(x=test_query_df_v3.upload_month,  #creating the barplot
            y=test_query_df_v3.price,
            #estimator = max,
            hue=test_query_df_v3.brand,
            palette = {'AUDI':'#CB4335','BMW':'#2471A3'},
            ci= None)

groupedvalues = test_query_df_v3.groupby(by=['brand','upload_month']).mean().reset_index()  #creating a grouped dataframe (seaborn made it for itself before plotting); sum() method

y_coord_push_const = 100000
for month in np.sort(test_query_df_v3.upload_month.unique()):  #every month from the dataset, iterating over the numpy array
    plt.text(
        int(month)-1.25,  #x-coordinate, slightly psuhed
        int(groupedvalues.loc[(groupedvalues.brand == 'AUDI')&(groupedvalues.upload_month == str(month)), "price"].values[0]) + y_coord_push_const,  #y-ccordinate
        str(groupedvalues.loc[(groupedvalues.brand == 'AUDI')&(groupedvalues.upload_month == str(month)), "price"].values[0])[:7],  #text itself
        rotation = 'vertical',  #text's attributes
        fontsize = 'x-large',
        fontstyle = 'oblique'
        )
    plt.text(
        int(month)-0.85,
        int(groupedvalues.loc[(groupedvalues.brand == 'BMW')&(groupedvalues.upload_month == str(month)), "price"].values[0]) + y_coord_push_const,
        str(groupedvalues.loc[(groupedvalues.brand == 'BMW')&(groupedvalues.upload_month == str(month)), "price"].values[0])[:7],
        rotation = 'vertical',
        fontsize = 'x-large',
        fontstyle = 'oblique'
        )

bc.set_xlabel("Month", fontsize = 20)
bc.set_ylabel("Price (in million HUF)", fontsize = 20)
bc.tick_params(labelsize = 15)

plt.setp(bc.get_legend().get_texts(), fontsize = "20")  #legend text
plt.setp(bc.get_legend().get_title(), fontsize = "20")  #legend title

#plt.ticklabel_format(style='plain', axis='y')  #y-axis scientific notation turned off


# ### Seaborn.Countplot
# 
# Since the average offer price of the cars was greatly variable, it is worth a look at the countplot, which is aggregating the average number of cars had been offered for sale each month. The methodology was the same as in case of the barplot. The only key difference was:
# * only the X-axis needed to be provided for Seaborn, rather than both of the axexs

# In[ ]:


plt.figure(figsize=(25,15))
sns.set(font_scale=1.0)  #using the default font size
cc = sns.countplot(
            x=test_query_df_v3.upload_month,  #creating the barplot
            hue=test_query_df_v3.brand,
            palette = {'AUDI':'#CB4335','BMW':'#2471A3'}
            )
groupedvalues = test_query_df_v3.groupby(by=['brand','upload_month']).count().reset_index()  #creating a grouped dataframe (seaborn made it for itself before plotting); count() method
y_coord_push_const = 10
for month in np.sort(test_query_df_v3.upload_month.unique()):  #every month from the dataset, iterating over the numpy array
    plt.text(
        int(month)-1.25,  #x-coordinate, slightly psuhed
        int(groupedvalues.loc[(groupedvalues.brand == 'AUDI')&(groupedvalues.upload_month == str(month)), "price"].values[0]) + y_coord_push_const,  #y-ccordinate
        str(groupedvalues.loc[(groupedvalues.brand == 'AUDI')&(groupedvalues.upload_month == str(month)), "price"].values[0])[:7],  #text itself
        rotation = 'vertical',  #text's attributes
        fontsize = 'x-large',
        fontstyle = 'oblique'
        )
    plt.text(
        int(month)-0.85,
        int(groupedvalues.loc[(groupedvalues.brand == 'BMW')&(groupedvalues.upload_month == str(month)), "price"].values[0]) + y_coord_push_const,
        str(groupedvalues.loc[(groupedvalues.brand == 'BMW')&(groupedvalues.upload_month == str(month)), "price"].values[0])[:7],
        rotation = 'vertical',
        fontsize = 'x-large',
        fontstyle = 'oblique'
        )

cc.set_xlabel("Month", fontsize = 20)
cc.set_ylabel("Count", fontsize = 20)
cc.tick_params(labelsize = 15)

plt.setp(cc.get_legend().get_texts(), fontsize = "20")  #legend text
plt.setp(cc.get_legend().get_title(), fontsize = "20")  #legend title 


# ### seaborn.scatterplot
# 
# A seaborn based scatterplot, the axes: price and mileage. I set the 'alpha' parameter of the points to the one fifth of the original value. It is easy to identify the "psychological limits" in the pricing, they are appearing as columns on the figure. For this plot I did not distinguished the two brands by colors.

# In[ ]:


test_query_df_v3_trimmed = test_query_df_v3.copy()
test_query_df_v3_trimmed.drop(test_query_df_v3_trimmed.loc[(test_query_df_v3_trimmed.mileage)>1000000].index, inplace = True)  #outlier values must be droped before the plotting
test_query_df_v3_trimmed.drop(test_query_df_v3_trimmed.loc[(test_query_df_v3_trimmed.price<100000)|(test_query_df_v3_trimmed.price>40000000)].index, inplace = True)  #outlier values must be droped before the plotting
plt.figure(figsize=(25,15))
sns.set(font_scale=1.0)  #using the default font size

sc = sns.scatterplot(               #creating the plot
            x=test_query_df_v3_trimmed.price,
            y=test_query_df_v3_trimmed.mileage,  
            #hue=test_query_df_v3_trimmed.brand,
            #palette = {'AUDI':'#CB4335','BMW':'#2471A3'},
            alpha = 0.2,
            #size = test_query_df_v3_trimmed.mileage
            )
sc.set_xlabel("Price", fontsize = 20)
sc.set_ylabel("Mileage (in kms)", fontsize = 20)
sc.tick_params(labelsize = 15)
#plt.xticks(np.arange(min(test_query_df_v3_trimmed.price), max(test_query_df_v3_trimmed.mileage), step=1))

plt.setp(cc.get_legend().get_texts(), fontsize = "20")  #legend text
plt.setp(cc.get_legend().get_title(), fontsize = "20")  #legend title 
plt.ticklabel_format(style='plain', axis='y')  #y-axis scientific notation turned off
plt.ticklabel_format(style='plain', axis='x')  #x-axis scientific notation turned off

