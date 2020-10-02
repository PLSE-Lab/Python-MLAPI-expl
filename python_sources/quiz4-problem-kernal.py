#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# According to the data set I choose, there is large amount of data related to the use of electricity and gas in many areas. As a result, I randomly select one of the data set to do the basic EDA.
# 
# * The data set I select is the "Electricity use of Stedin in 2017"
# * In the following codes, I will use pandas to do the EDA to present the fundamental analysis. 

# 1. Presneting part of table and understand the amount of data. For example, how many rows and columns included? The name of colunms?

# * The code below shows the part of orginal data set.

# In[ ]:


electricity = pd.read_csv("/kaggle/input/dutch-energy/Electricity/stedin_electricity_2017.csv")
electricity.head(5)


# * Number of rows and columns in the data set.

# In[ ]:


electricity.shape


# 2. Since the electricity is purchaed from different area, I want to understand basic supply information in Stedin.

# * List all the areas that electricity is purchased from. Besides, also list the number of streets supplied by each supply point.

# In[ ]:


ele_grp = electricity.groupby("purchase_area")
supply_count = ele_grp.purchase_area.size().sort_values(ascending = False)
supply_count


# * Plot the piechart to show the supply distribution(in streets) in Stedin.

# In[ ]:


supply_count.plot(kind = "pie")


# 3. To understnd how many streets supplied by each supply points is not enough, I still want to understand which supply point supplies the most, which supplies the least and so on.

# * Total amount of electricity supplied by each point.

# In[ ]:


ele_grp.annual_consume.sum().sort_values(ascending = False)


# Basically, the more areas served, the more electricity is supplied as well.

# * Besides, to realize the total amount supplied is not enough, I still want to understand on everage, how much electricity is supplied to each street by each supply point and its standard deviation.

# In[ ]:


each_point_mean = ele_grp.annual_consume.mean().round(2)
each_point_std = ele_grp.annual_consume.std().round(2)
each_point_max = ele_grp.annual_consume.max()
each_point_min = ele_grp.annual_consume.min()
each_point_median = ele_grp.annual_consume.median()
frame = {"average_consume": each_point_mean , "std": each_point_std , "max_consume": each_point_max , "min_consume": each_point_min
        ,"median_consume": each_point_median}
table = pd.DataFrame(frame).reset_index()
table.sort_values("average_consume" , ascending = False)
#each_point_std


# In the table above, we can realize that Stedin Weert on average supplies the most electricity to each street. However, for each supply point, the standard deviation is pretty large compared to mean.

# In[ ]:


each_point_median = ele_grp.annual_consume.median()
each_point_median


# 4. On everage, I dicsover that Stedin Weert supplies most electricuty to each treet, so I want to first select areas supplied by Stedin Weert and do more observations.

# * To show the table only includes electricity purchased from Stedin Weert and present part of result.

# In[ ]:


sw_ele = electricity[electricity.purchase_area == "Stedin Weert"]
sw_ele.head(10)


# * To choose columns we are interested in, such as "delivery_perc", "annual_consume_lowtarif_perc"and calculate the average for all columns

# In[ ]:


sw_ele2 = sw_ele.loc[:,["delivery_perc", "annual_consume_lowtarif_perc"]]
sw_ele2.head(10)


# In[ ]:


sw_ele2.mean()


# * In Stedin Weert, I want to see whether the average of columns I select above will be different with group with annual_consume larger than average and group with annual_consume lower than average.

# ** Show the result with annual_consume smaller than average.

# In[ ]:


average = sw_ele.annual_consume.mean()
sw_ele_low = sw_ele[sw_ele.annual_consume < average]
sw_ele_low.loc[:,["delivery_perc", "annual_consume_lowtarif_perc"]].mean().round(2)


# ** Show the result with annual_consume larger than average.

# In[ ]:


average = sw_ele.annual_consume.mean()
sw_ele_lar = sw_ele[sw_ele.annual_consume > average]
sw_ele_lar.loc[:,["delivery_perc", "annual_consume_lowtarif_perc"]].mean().round(2)


# According to the result above, we could find out that for delivery_perc,the difference is not really big. However, as for annual_consume_lowtarif_perc, the difference is quite large.
