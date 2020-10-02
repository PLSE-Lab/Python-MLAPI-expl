#!/usr/bin/env python
# coding: utf-8

# # Seaborn plotting notebook
# 
# With this notebook I am going to demonstrate some basic capabilities of the **Seaborn Python** module. For the data visualizaiton I've used my own dataset, that I've gathered in a web scraping project. Further details about the web scraping could be found on [GitHub](https://github.com/atttilakiss/Project-HaHU_KA).  
# For the analysis I've queried the some basic details of the "holy trinity" (at least in Hungary) advertisements: Audi, BMW, Mercedes-Benz
# 
# ## Table of contents
# [Modules](#Modules)  
# [Data_query](#Data_query)  
# [Seaborn.Barplot](#Seaborn.Barplot)  
# [Seaborn.Lineplot & Seaborn.Barplot](#Seaborn.Lineplot_&_Seaborn.Barplot)  
# [Seaborn.Scatterplot](#Seaborn.Scatterplot)  
# [Seaborn.Distplot](#Seaborn.Distplot)  
# [Seaborn.Boxplot](#Seaborn.Boxplot)
# 
# 

# # Modules
# 
# The following modules were used in order to achieve the data plotting

# In[ ]:


import sqlite3  #connection with the database

import pandas as pd  #query data stored in pandas
pd.options.display.float_format = "{:.2f}".format  #reformat the pandas dataframe output in order to show the plain values
import numpy as np  #pandas built upon numpy, so it is necessary

import matplotlib.pyplot as plt  #plotting module
import matplotlib.patches as mpatches  #create my own legends and colors on the figures
import matplotlib.ticker as ticker  #provides broader customization options on the axes tickers

import seaborn as sns  #built upon matplotlib and pandas, used for visualization

import datetime  #SQL query date parsing


# # Data_query
# 
# I have only queried basic information about the dataset:
# * brand
# * upload date / only upload month had been loaded into the pandas DF
# * price of the advertisement
# * mileage of the car
# 
# The resulting df has 7597 rows, basic information are being printed out

# In[ ]:


sql_database = '/kaggle/input/secondhand-car-market-data-parsing-dataset-v1/kaggle_sqlite'
conn = sqlite3.connect(sql_database)  #conn object with the SQLite DB

car_data_df = pd.DataFrame()  #empty DF
car_data_df = pd.read_sql("""
                            SELECT 
                                brand_name as 'brand',
                                strftime('%m', upload_date) as 'upload_month',
                                ad_price as 'price',
                                mileage as 'mileage'
                            FROM advertisements
                            JOIN brand ON advertisements.brand_id = brand.brand_id
                            WHERE brand_name IN ('BMW','AUDI', 'MERCEDES-BENZ');
                            """,
                            conn)  #SQL query; used pandas 'read_sql' method
print('Dataframe basic info:\n')
print(car_data_df.info())
print('\n\nDataframe data (top 5 rows):\n')
print(car_data_df.head())
print('\n\nDataframe description:\n')
print(car_data_df[['mileage','price']].describe())


# # Seaborn.Barplot
# 
# The first figure is containing four subplots:
# * Mean price (y-axis) / Month (x-axis); Count of advertisements (secondary y-axis) / Month (x-axis)
#     * subplot 1: all three brands combined
#     * subplot 2: Audi
#     * subplot 3: BMW
#     * subplot 4: Mercedes-Benz
# 
# The challenges in order to create this plot:
# * creating the figure frame and determining the subplot where the chart should be plotted
#     * 'ax' paramter was serving well
#     * pandas' '.loc' method was handy for the data selection
# * creating the secondary y-axis
#     * 'ax.twinx()' had became useful
# * plot the count of the advertisements rather than sum
#     * creating the grouped copy of the dataframe (all brands combined and separately by brands) / 'df.groupby' method
#     * created a lineplot
#     
# 
# The key takeaways:
# * I have run the web scraping more frequently in certain periods of the year
# * The lower amount of observation is resulting higher standard deviation 

# In[ ]:


fig, axes = plt.subplots(2,2, figsize = (25,15))  #initialize the figure and axes
sns.set(style = 'darkgrid')  #modifying the style of the figure

#grouped dataframes
ax00b_grouped_df = car_data_df.groupby(by=['upload_month']).count().reset_index()  #three brands combined
car_data_df_grouped = car_data_df.groupby(by=['upload_month', 'brand']).count().reset_index()  #separated by brands

"""--------------------------------------------------------------------------"""

#creating the barplot for all three brands combined; subplot[0,0]
ax00a = sns.barplot(
                ax=axes[0,0],
                x=car_data_df.upload_month,
                y=car_data_df.price,
                palette = "GnBu_d",
                errcolor = '#FF5511',
                capsize = 0.2)

ax00a.set_xlabel('Month', fontsize=15.0)  #x label
ax00a.set_ylabel('Price (in million HUF)', fontsize=15.0)  #y label

#secondary axis; lineplot
ax00b = ax00a.twinx()  #creating the secondary y-axis for the lineplot; subplot[0,0]
ax00b = sns.lineplot(
        x=ax00b_grouped_df.upload_month,
        y=ax00b_grouped_df.price,
        linewidth = 1.5,
        color = '#FF5511',
        marker = 'x',
        markersize = 15.0,
        markeredgecolor = '#FF5511',
        markeredgewidth = 3.0)

ax00b.set_ylabel('Total number of advertisements (count)', fontsize=15.0)

"""--------------------------------------------------------------------------"""

#creating the barplot for AUDI brand; subplot[0,1]
ax01a = sns.barplot(
        ax=axes[0,1],
        x=car_data_df.upload_month.loc[car_data_df['brand']=='AUDI'],
        y=car_data_df.price.loc[car_data_df['brand']=='AUDI'],
        palette = "Blues",
        errcolor = '#BBC400',
        capsize = 0.2)

ax01a.set_xlabel('Month', fontsize=15.0)
ax01a.set_ylabel('Price (in million HUF)', fontsize=15.0)
ax01a.set_title('AUDI')

#secondary axis; lineplot
ax01b = ax01a.twinx() #creating the secondary y-axis for the lineplot; subplot[0,1]
ax01b = sns.lineplot(
        x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='AUDI'],
        y=car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='AUDI'],
        linewidth = 0,
        color = '#BBC400',
        marker = 'o',
        markersize = 15.0,
        markeredgecolor = '#BBC400',
        markeredgewidth = 3.0)

ax01b.set_ylabel('Total number of advertisements (count)', fontsize=15.0)

"""--------------------------------------------------------------------------"""

#creating the barplot for BMW brand; subplot[1,0]
ax10a = sns.barplot(
        ax=axes[1,0],
        x=car_data_df.upload_month.loc[car_data_df['brand']=='BMW'],
        y=car_data_df.price.loc[car_data_df['brand']=='BMW'],
        palette = "Greys",
        errcolor = '#0068C4',
        capsize = 0.2)

ax10a.set_xlabel('Month', fontsize=15.0)
ax10a.set_ylabel('Price (in million HUF)', fontsize=15.0)
ax10a.set_title('BMW')

#secondary axis; lineplot
ax10b = ax10a.twinx() #creating the secondary y-axis for the lineplot; subplot[0,1]
ax10b = sns.lineplot(
        x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='BMW'],
        y=car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='BMW'],
        linewidth = 0,
        color = '#0068C4',
        marker = 'o',
        markersize = 15.0,
        markeredgecolor = '#0068C4',
        markeredgewidth = 3.0)

ax10b.set_ylabel('Total number of advertisements (count)', fontsize=15.0)

"""--------------------------------------------------------------------------"""

#creating the barplot for MERCEDES-BENZ brand; subplot[1,1]
ax11a = sns.barplot(
        ax=axes[1,1],
        x=car_data_df.upload_month.loc[car_data_df['brand']=='MERCEDES-BENZ'],
        y=car_data_df.price.loc[car_data_df['brand']=='MERCEDES-BENZ'],
        palette = "Greens",
        errcolor = '#000000',
        capsize = 0.2)

ax11a.set_xlabel('Month', fontsize=15.0)
ax11a.set_ylabel('Price (in million HUF)', fontsize=15.0)
ax11a.set_title('MERCEDES-BENZ')

#secondary axis; lineplot
ax11b = ax11a.twinx()  #creating the secondary y-axis for the lineplot; subplot[1,1]
ax11b = sns.lineplot(
        x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='MERCEDES-BENZ'],
        y=car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='MERCEDES-BENZ'],
        linewidth = 0,
        color = '#000000',
        marker = 'o',
        markersize = 15.0,
        markeredgecolor = '#000000',
        markeredgewidth = 3.0)

ax11b.set_ylabel('Total number of advertisements (count)', fontsize=15.0)

"""--------------------------------------------------------------------------"""

fig.savefig('month_price_fig1.png')


# # Seaborn.Lineplot_&_Seaborn.Barplot
# 
# I wanted to visualize the distribuiton of the published advertisements by brands in order to idenitfy if there is a significant difference amongst the brands all in seaborn. For this examination I've used a lineplot and a stacked barchart. For a better chart I've also used the 'sharex' parameter for the subplots, so the x-axis is commonly used by the two charts.  
# The challneges:
# * Lineplot
#     * used my brand separated grouped df, so I had the counts of the advertisements by brand in every month
#     * customizing the legend manually -> I used the patches object of the matplotlib
# * Stacked Barchart
#     * seaborn does not support the 'stacked' parameter (as far as I know), so I had to figure out a different solution
#         * I have created three barcharts for every month and I've displayed them on the top of each other
#         * the heights of the bars are the result of the month-by-month summary of the advertisement, e.g. I wanted to show the Mercedes advertisement on the top, so the bottom of this bar is 'bmw+audi' and the top is 'bmw+audi+mercedes'; rule of thumb:
#             * bmw = bmw
#             * audi = bmw + audi
#             * mercedes = bmw + audi + mercedes
#         * for each month I've created a reindexed (from zero) Series for every brand
#     * Also wanted to show the total value of stacked bars
#         * 'plt.text' is serving for this customization: x and y coordinates had to be provided and the text itself
#         * I have iterated over the unique month values (x coordinate), and for all of them I've selected the corresponding count value (y coordinate and text)
#         * for some reason I had to decrease the month value by 1, I'm guessing the x-axis has zero indexing, with reduced month it was fitting perfectly

# In[ ]:


fig2, axes2 = plt.subplots(2,figsize = (25,20), sharex = True)  #initialize the figure and axes
fig2.subplots_adjust(hspace = 0.01)  #reducing the space between the subplots
sns.set(style = 'darkgrid')  #modifying the style

#visualization parameters
palette = {'AUDI':'#BBC400', 'BMW':'#0068C4', 'MERCEDES-BENZ':'#000000'}
bmw_patch = mpatches.Patch(color='#0068C4', label='BMW')
audi_patch = mpatches.Patch(color='#BBC400', label='AUDI')
merc_patch = mpatches.Patch(color='#000000', label='MERCEDES-BENZ')

"""--------------------------------------------------------------------------"""
#creating the lineplot; subplot[0]
ax00 = sns.lineplot(
                x=car_data_df_grouped.upload_month,
                y=car_data_df_grouped.price,
                hue = car_data_df_grouped.brand,
                palette = palette,
                linewidth = 2.5,
                ax=axes2[0])

ax00.set_ylabel('Count of advertisements', fontsize = 20.0)
ax00.tick_params(axis='y', labelsize=15.0)

ax00.legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0)  #creating the legend manually

"""--------------------------------------------------------------------------"""

#creating the height values of the stacked bars
top_bmw = car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='BMW'].copy()
top_bmw.reset_index(drop=True, inplace=True)

top_audi = car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='AUDI'].copy()
top_audi.reset_index(drop=True, inplace=True)

top_merc = car_data_df_grouped.price.loc[car_data_df_grouped['brand']=='MERCEDES-BENZ'].copy()
top_merc.reset_index(drop=True, inplace=True)

#creating the lineplot; subplot[1]
ax01 = sns.barplot(
                x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='MERCEDES-BENZ'].reset_index(drop=True),
                y=top_bmw + top_audi + top_merc,
                color = '#000000')

ax01 = sns.barplot(
                x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='AUDI'].reset_index(drop=True),
                y=top_bmw + top_audi,
                color = '#BBC400')

ax01 = sns.barplot(
                x=car_data_df_grouped.upload_month.loc[car_data_df_grouped['brand']=='BMW'].reset_index(drop=True),
                y=top_bmw,
                color = '#0068C4')

#creating the data labels for the columns
for month in np.sort(car_data_df_grouped.upload_month.unique()):
    month=int(month)
    plt.text(
        int(month-1)-0.1,
        int(top_bmw.iloc[month-1] + top_audi.iloc[month-1] + top_merc.iloc[month-1]) +10,
        str(top_bmw.iloc[month-1] + top_audi.iloc[month-1] + top_merc.iloc[month-1]),
        fontsize = 'large',
        fontstyle = 'normal')


ax01.set_ylabel('Count of advertisements', fontsize = 20.0)
ax01.tick_params(axis='y', labelsize=15.0)

ax01.set_xlabel('Month', fontsize = 15.0)
ax01.tick_params(axis='x', labelsize=15.0)

ax01.legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0) #creating the legend manually

"""--------------------------------------------------------------------------"""

fig2.savefig("month_count_fig2.png")


# # Seaborn.Scatterplot
# 
# The next chart is a scatterplot, which is showing the relation between the price of a car (x-axis) and the milage  it ran (y-axis). For a better view, I have trimmed the outlier values of the dataset, so all the advertisements, which:  
# * price is below 100 000 HUF (about 350 Euros) 
# * price is higher than 20 000 000 HUFs (about 65 000 Euros)
# * mileage less than 1500 kms
# * mileage higher than 400 000 kms  
# 
# had been eliminated! To achieve this I've used pandas 'drop' and '.loc' methods with multiple conditions.  
# Ploting the dataset was quite easy, all I had to do as an extra-task was to provide the coloring palette, which I've used in the previous charts.  
# It was a little bit more complicated to properly format the tick values on the axes, but 'ticklabel_format' method and it's 'style' parameter did the job! It is also possible to determine the number of digits by the 'scilimits' of the same method. For legend I've reused the previous patchscheme.
# 
# It is easy to identify some 'columns' in the pricing of the cars regardless of it's milage or brand.

# In[ ]:


#trim the extreme values from the dataset
#removing rows with price less than 100 000 HUFs, more than 20 000 000 HUFs and milage less than 1500 kms and higher than 400 000 kms
car_data_df_trimmed = car_data_df.copy().drop(car_data_df.loc[(car_data_df.mileage>400000)|(car_data_df.mileage<1500)|(car_data_df.price<100000)|(car_data_df.price>20000000)].index, inplace = False)

fig3, ax3 = plt.subplots(1, figsize = (25,20))
ax3= sns.scatterplot(
            x=car_data_df_trimmed.price,
            y=car_data_df_trimmed.mileage,
            alpha = 0.2,
            hue = car_data_df_trimmed.brand,
            palette = {'AUDI':'#BBC400','BMW':'#0068C4','MERCEDES-BENZ':'#000000'})

ax3.ticklabel_format(style='plain', axis='y')  #y-axis scientific notation turned off
ax3.tick_params(axis='y', labelsize=20.0)
ax3.set_ylabel('Mileage', fontsize = 25.0)

ax3.ticklabel_format(style='sci', axis='x', scilimits=(6,6))  #x-axis scientific notation turned off
ax3.tick_params(axis='x', labelsize=20.0, labelrotation=45)
ax3.set_xlabel('Price (in million HUF)', fontsize = 25.0)

ax3.legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0)

fig3.savefig('mileage_price_fig3.png')


# # Seaborn.Distplot
# 
# As a next plot I wanted to examine the distribution of the prices, so I've used seaborn's distplot with a 100 bins. I've also created a subplot, with the previously trimmed dataset. I've plotted all the brands on the same charts, so I needed to create three plots on top of each other for both of the charts.

# In[ ]:


fig4, ax4 = plt.subplots(2, figsize = (25,20), sharex=False)
ax00 = sns.distplot(
                a=car_data_df.price.loc[(car_data_df.brand=='AUDI')],
                bins=100,
                color = '#BBC400',
                hist=True,
                kde=False,
                kde_kws={'shade': True, 'linewidth': 3},
                ax=ax4[0])
ax00 = sns.distplot(
                a=car_data_df.price.loc[(car_data_df.brand=='BMW')],
                bins=100,
                color = '#0068C4',
                hist=True,
                kde=False,
                kde_kws={'shade': True, 'linewidth': 3},
                ax=ax4[0])

ax00 = sns.distplot(
                a=car_data_df.price.loc[(car_data_df.brand=='MERCEDES-BENZ')],
                bins=100,
                color = '#000000',
                hist=True,
                kde=False,
                kde_kws={'shade': True, 'linewidth': 3},
                ax=ax4[0])

ax4[0].ticklabel_format(style='plain', axis='y')
ax4[0].tick_params(axis='y', labelsize=20.0)
ax4[0].set_ylabel('Count of advertisements', fontsize = 25.0)

ax4[0].ticklabel_format(style='sci', axis='x', scilimits=(6,6))
ax4[0].tick_params(axis='x', labelsize=20.0, labelrotation=45)
ax4[0].set_xlabel('Price (in million HUF)', fontsize = 25.0)

ax4[0].legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0)

"""--------------------------------------------------------------------------"""

car_data_df_price_trimmed = car_data_df.copy().drop(car_data_df.loc[(car_data_df.price<100000)|(car_data_df.price>20000000)].index, inplace = False)

ax01 = sns.distplot(
                a=car_data_df_price_trimmed.price.loc[(car_data_df_price_trimmed.brand=='AUDI')],
                bins=100,
                color = '#BBC400',
                hist=True,
                kde=False,
                kde_kws={'shade': True, 'linewidth': 3},
                ax=ax4[1])
ax01 = sns.distplot(
                a=car_data_df_price_trimmed.price.loc[(car_data_df_price_trimmed.brand=='BMW')],
                bins=100,
                color = '#0068C4',
                hist=True,
                kde=False,
                kde_kws={'shade': True, 'linewidth': 3},
                ax=ax4[1])

ax01 = sns.distplot(
                a=car_data_df_price_trimmed.price.loc[(car_data_df_price_trimmed.brand=='MERCEDES-BENZ')],
                bins=100,
                color = '#000000',
                hist=True,
                kde=False,
                kde_kws={'shade': True, 'linewidth': 3},
                ax=ax4[1])

ax4[1].ticklabel_format(style='plain', axis='y')  #y-axis scientific notation turned off
ax4[1].tick_params(axis='y', labelsize=20.0)
ax4[1].set_ylabel('Count of advertisements', fontsize = 25.0)

ax4[1].ticklabel_format(style='sci', axis='x', scilimits=(6,6))  #x-axis scientific notation turned off
ax4[1].tick_params(axis='x', labelsize=20.0, labelrotation=45)
ax4[1].set_xlabel('Price (in million HUF); trimmed price: 100 000< price < 20 000 000', fontsize = 25.0)
#ax4[1].xaxis.set_major_locator(ticker.MaxNLocator(30))
#ax4[1].xaxis.set_minor_locator(ticker.MaxNLocator(30))

ax4[1].legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0)

fig4.savefig('price_dist_fig4.png')


# # Seaborn.Boxplot
# 
# The most compact plotting technique, a boxplot is representing many information and it is quite easy to create:  
# * For every month I've created three boxplots for the brands
# * I set the whiskers to represent the data from the 10th to the 90th percentile
# * I've turned off the outlier values
# * I've used a slightly modified color palette and legend patches  
# 
# What should be recognised is the mean value is not varying greatly in any month amongst the brands, so in a certain month the mean price of the advertisement is almost a same regardless if it is a BMW, Audi or Mercedes.

# In[ ]:


fig5, ax5 = plt.subplots(1, figsize = (25,20))

ax00=sns.boxplot(
                x='upload_month',
                y='price',
                data=car_data_df,
                hue = 'brand',
                whis=[10, 90],
                sym="",
                palette = {'AUDI':'#BBC400','BMW':'#0068C4','MERCEDES-BENZ':'#FFFFFF'},
                )

ax5.ticklabel_format(style='sci', axis='y', scilimits=(6,6))  #y-axis scientific notation turned off
ax5.tick_params(axis='y', labelsize=20.0)
ax5.set_ylabel('Price in million HUF (from 10th to 90th percentiles)', fontsize = 25.0)


ax5.tick_params(axis='x', labelsize=20.0)
ax5.set_xlabel('Month', fontsize = 25.0)

merc_patch = mpatches.Patch(color='#FFFFFF', label='MERCEDES-BENZ')

ax5.legend(handles=[bmw_patch, audi_patch, merc_patch], title = 'Brand', fontsize=20.0)

fig5.savefig('boxplot_fig5.png')

