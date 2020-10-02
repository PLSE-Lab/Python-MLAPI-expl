#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


pa = pd.read_csv("../input/Police_activity.csv")


# #### Glimpse of data

# In[ ]:


pa.head(2)


# #### Column details

# In[ ]:


print(pa.columns)
print(pa.columns.size) # or print(pa.shape[1])


# #### Total observations

# In[ ]:


print(pa.index)
print(pa.shape)


# #### See number of NULL values

# In[ ]:


pa.isnull().sum()


# #### drop those columns which is having more than 50 missing values

# In[ ]:


for col in pa.columns:
    if pa[col].isnull().sum() > 50:
        pa.drop(col,axis="columns",inplace=True)


# #### Verify number of NULL values

# In[ ]:


pa.isnull().sum() # small missing values, we are going to replace with mean,median or mode based on need


# In[ ]:


pa.head(2)


# #### Drop those colums which is not required for analysis
# - State :: only one unique value, hence drop
# - location_raw and district are same so drop either of them
# 

# In[ ]:


np.array([pa.location_raw != pa.district]).sum() # checking if location_raw is exactly same as district. if sum = 0 then same


# #### Now drop above mentioned columns

# In[ ]:


pa.drop(["state","location_raw"],axis=1,inplace=True)


# In[ ]:


pa.head(2)


# #### Few columns have very less missing values, so we will fill those missing values based on there data type

# In[ ]:


pa.isnull().sum()


# #### Check data type of those missing values columns

# In[ ]:


for col in pa.columns:
    if pa[col].isnull().sum():
        print("Data type of %s column is %s"%(col,pa[col].dtype))


# #### Since its not a good idea to assume any stop_date and stop_time, hence we will drop those rows where missing values are there for those columns

# In[ ]:


pa.dropna(subset=["stop_date","stop_time"],inplace=True)


# #### It seems oll the missing values from other columns (police department as well as search_conducted ) were falling under same stop_date and stop_time, that is what is shown in next output, now no more missing values in any of the considered column

# In[ ]:


pa.isnull().sum()


# ### In our data, we have start and stop date/time, that can be used as index of the dataframe, lets see how we can use those columns as index. To do this we need to merge them and create standard time format for analysis

# In[ ]:


pa.head(2)


# ###### We will use `str` method to concatenate strings and create new columns

# In[ ]:


pa["start_stop_time"] = pa.stop_date+" "+pa.stop_time


# In[ ]:


pa.head(2)


# In[ ]:


pa.start_stop_time.dtype


# ###### Convert pa.start_stop_time to time series data type

# In[ ]:


pa.start_stop_time = pd.to_datetime(pa.start_stop_time,infer_datetime_format=True)


# In[ ]:


pa.start_stop_time.dtype


# In[ ]:


pa.start_stop_time.head(5) # We can see that dtype is converted as pandas `datetime` dtype


# # Now our data is completely ready for further analysis 

# In[ ]:


pa.head(2)


# #### Lets see in which HOUR most of the time police stopped

# In[ ]:


plt.figure(figsize=(16,4))
pa.start_stop_time.dt.hour.value_counts().plot(kind="bar",color="r")
plt.xlabel("Hour")
plt.ylabel("Hourly_Count")
plt.title("Hourly_Count_Details")
plt.show()


# #### Frequently stopped timing HOUR details using histogram plot

# In[ ]:


plt.figure(figsize=(16,4))
sns.distplot(pa.start_stop_time.dt.hour)
plt.xlabel("Hour")
plt.ylabel("Hourly_freq")
plt.title("Hourly_Count_Details_Histogram")
plt.show()


# ###### Lets see in which YEAR most of the time police stopped

# In[ ]:


plt.figure(figsize=(16,4))
pa.start_stop_time.dt.year.value_counts().plot(kind="bar",color="g")
plt.xlabel("Year")
plt.ylabel("Yearly_Count")
plt.title("Yearly_Count_Details")
plt.show()


# #### Frequently stopped YEAR details using histogram plot

# In[ ]:


plt.figure(figsize=(16,4))
sns.distplot(pa.start_stop_time.dt.year)
plt.xlabel("Year")
plt.ylabel("Yearly_Count")
plt.title("Yearly_Count_Details_Histogram")
plt.show()


# ###### Lets see in which MONTH most of the time police stopped

# In[ ]:


plt.figure(figsize=(16,4))
pa.start_stop_time.dt.month.value_counts().plot(kind="bar",color="c")
plt.xlabel("Month_Number")
plt.ylabel("Monthly_Count")
plt.title("Monthly_Count_Details")
plt.show()


# #### Frequently stopped MONTH details using histogram plot

# In[ ]:


plt.figure(figsize=(16,4))
sns.distplot(pa.start_stop_time.dt.month)
plt.xlabel("Month")
plt.ylabel("Monthly_Count")
plt.title("Monthly_Count_Details_Histogram")
plt.show()


# #### Frequently stopped Day details using histogram plot

# In[ ]:


plt.figure(figsize=(16,4))
sns.distplot(pa.start_stop_time.dt.day)
plt.xlabel("Daily")
plt.ylabel("Daily_Count")
plt.title("Dailly_Count_Details_Histogram")
plt.show()


# In[ ]:


pa.head(2)


# ###### Lets see which Count of drug related stops

# In[ ]:


np.log(pa.drugs_related_stop.value_counts().values)


# In[ ]:


pa.drugs_related_stop.value_counts().plot(kind="bar")
plt.xlabel("True/False")
plt.ylabel("True/False_Count")
plt.title("drugs_related_True/False_Count_Details")
plt.show()


# ##### Log representation, to cut short big values

# In[ ]:


np.log(pa.drugs_related_stop.value_counts()).plot(kind="bar")
plt.xlabel("True/False")
plt.ylabel("True/False_Count")
plt.title("Log: drugs_related_True/False_Count_Details")
plt.show()


# In[ ]:


pa.head(2)


# ###### Total search conducted yearwise/monthwise
# - Need to add year & month column so that grouping can be made

# In[ ]:


pa["year"] = pa.start_stop_time.dt.year
pa["month"] = pa.start_stop_time.dt.month


# In[ ]:


pa.head(2)


# ### Create group for year wise

# In[ ]:


grp1 = pa.groupby(["year"])


# ###### year wise search_conducted

# In[ ]:


plt.figure(figsize=(16,4))
for key in grp1.groups:
    true_val = grp1.get_group(key).search_conducted.value_counts().loc[True]
    false_val = grp1.get_group(key).search_conducted.value_counts().loc[False]
    plt.bar(str(key),false_val,color="#934666",width=.5)
    plt.text(str(key),false_val-11000,false_val,rotation=90)
    plt.bar(str(key),true_val,bottom=false_val,color="#930006",width=.5)
    plt.text(str(key),true_val+false_val+500,true_val)
plt.xticks(rotation=90)
plt.legend(["False","True"])
plt.xlabel("Year")
plt.ylabel("Yearly_Count")
plt.title("Grouping With Yearly search_conducted Count_Details")
plt.show()


# In[ ]:


pa.head(2)


# ###### year wise contraband_found

# In[ ]:


plt.figure(figsize=(16,4))
for key in grp1.groups:
    true_val = grp1.get_group(key).contraband_found.value_counts().loc[True]
    false_val = grp1.get_group(key).contraband_found.value_counts().loc[False]
    plt.bar(str(key),false_val,color="#934666",width=.5)
    plt.text(str(key),false_val-11000,false_val,rotation=90)
    plt.bar(str(key),true_val,bottom=false_val,color="k",width=.5)
    plt.text(str(key),true_val+false_val+500,true_val)
plt.xticks(rotation=90)
plt.legend(["False","True"])
plt.xlabel("Year")
plt.ylabel("Yearly_Count")
plt.title("Grouping With Yearly Contraband_found Count_Details")

plt.show()


# ###### year wise drugs_related_stop

# In[ ]:


plt.figure(figsize=(16,4))
for key in grp1.groups:
    true_val = grp1.get_group(key).drugs_related_stop.value_counts().loc[True]
    false_val = grp1.get_group(key).drugs_related_stop.value_counts().loc[False]
    plt.bar(str(key),false_val,color="#934666",width=.5)
    plt.text(str(key),false_val-11000,false_val,rotation=90)
    plt.bar(str(key),true_val,bottom=false_val,color="b",width=.5)
    plt.text(str(key),true_val+false_val+500,true_val)
plt.xticks(rotation=90)
plt.legend(["False","True"])
plt.xlabel("Year")
plt.ylabel("Yearly_Count")
plt.title("Grouping With Yearly frugs_related Count_Details")
plt.show()


# In[ ]:


pa.head(2)


# ### Create group for month wise

# In[ ]:


grp2 = pa.groupby(["month"])


# ###### month wise search_conducted

# In[ ]:


plt.figure(figsize=(16,4))
for key in grp2.groups:
    true_val = grp2.get_group(key).search_conducted.value_counts().loc[True]
    false_val = grp2.get_group(key).search_conducted.value_counts().loc[False]
    plt.bar(str(key),false_val,color="#934666",width=.5)
    plt.text(str(key),false_val-11000,false_val,rotation=90)
    plt.bar(str(key),true_val,bottom=false_val,color="#930006",width=.5)
    plt.text(str(key),true_val+false_val+500,true_val)
plt.xticks(rotation=90)
plt.legend(["False","True"])
plt.xlabel("Month")
plt.ylabel("Monthly_Count")
plt.title("Grouping With Monthly search_conducted Count_Details")
plt.show()


# ###### month wise contraband_found

# In[ ]:


plt.figure(figsize=(16,4))
for key in grp2.groups:
    true_val = grp2.get_group(key).contraband_found.value_counts().loc[True]
    false_val = grp2.get_group(key).contraband_found.value_counts().loc[False]
    plt.bar(str(key),false_val,color="m",width=.5)
    plt.text(str(key),false_val-11000,false_val,rotation=90)
    plt.bar(str(key),true_val,bottom=false_val,color="k",width=.5)
    plt.text(str(key),true_val+false_val+500,true_val)
plt.xticks(rotation=90)
plt.legend(["False","True"])
plt.xlabel("Month")
plt.ylabel("Monthly_Count")
plt.title("Grouping With Monthly contraband_found Count_Details")
plt.show()


# ###### month wise drugs_related_stop

# In[ ]:


plt.figure(figsize=(16,4))
for key in grp2.groups:
    true_val = grp2.get_group(key).drugs_related_stop.value_counts().loc[True]
    false_val = grp2.get_group(key).drugs_related_stop.value_counts().loc[False]
    plt.bar(str(key),false_val,color="c",width=.5)
    plt.text(str(key),false_val-11000,false_val,rotation=90)
    plt.bar(str(key),true_val,bottom=false_val,color="b",width=.5)
    plt.text(str(key),true_val+false_val+500,true_val)
plt.xticks(rotation=90)
plt.legend(["False","True"])
plt.xlabel("Month")
plt.ylabel("Monthly_Count")
plt.title("Grouping With Monthly drugs_related_stop Count_Details")
plt.show()


# In[ ]:


pa.head(2)


# #### Scatter plot of Stop_date Vs Search _Conducted Vs Hourly_Stop

# In[ ]:


plt.figure(figsize=(16,4))
data_to_display = 5000
sns.scatterplot(pa.stop_date[:data_to_display],pa.search_conducted[:data_to_display],
                hue=pa.district[:data_to_display],
                alpha=.1,
                s=20*pa.start_stop_time.dt.hour[:data_to_display]
               )
plt.xticks(pa.stop_date[:data_to_display:2],rotation=90)
plt.xlabel("Date")
plt.ylabel("Daily_Search_Conducted")
plt.title("Scatter plot Date Vs search Conducted Vs District Vs Hours")
plt.show()

