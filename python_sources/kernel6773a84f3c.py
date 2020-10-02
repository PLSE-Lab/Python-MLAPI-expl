#!/usr/bin/env python
# coding: utf-8

# #                         Data Analysis and Visualization on Suicides 1985-2016

# ## Introduction
# 
# This is data analysis and visualization solution done for Pucho Technoligies LTD. as a part of internship 
# Following notebook can be accessed at: <a href="https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/94e36cb6-7771-491b-ad08-a6d9de0f05a2/view?access_token=79e8092e02ee583467d796fb796b37b2d5a9021f1d3da2cde620e7bf6f81e2fc"><strong> Click Here</strong></a>

# ### 1) Data Ingestion
# 
# First step is to ingest dataset found from the kaggle dataset at this link: https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
# As IBM Watson IOT platform was used easy options are available for ingesting data into spark or pandas dataframe.
# Here I am using pandas dataframe for better visualiztions. Though processing data is done in Spark Dataframe as spark provides a great a performance when dealing with large datasets. 

# In[3]:


import types
import pandas as pd

pandasRAW = pd.read_csv('../input/master.csv')
pandasRAW.head()


# ### 2) Data Cleaning
# 
# According to IBM data analytics nearly 80% of time is spent by Data Scientists on Data cleaning. Data cleaning is an inmportant step in data analysis as further gaining insights of data heavily depends on how far the data is accurate and scalable. Here we can see that column 'HDI_for_year' is not a number column which makes it useless for the analysis part, removing columns which are not necessary for analysis improves effeciency and thus in turn increases latency. The second step is to check for null values if any in any columns. Pandas provides a wide range of mehods to accomplish the repsective null value finding task. The third and the final step is to check for duplicate values if any which can render further insights of data.

# In[4]:


pandasRAW.drop("HDI for year", inplace=True, axis=1)
pandasRAW.rename(columns={'gdp_for_year($)': 'gdp_year', 'gdp_per_capita ($)': 'gdp_capita'}, inplace=True)
pandasRAW.dtypes
pandasRAW.isnull().sum()
pandasRAW[pandasRAW.duplicated(keep=False)]


# ### 3) Data Processing
# 
# Now that the data is cleaned and ready for gaining insights from it. The next step is data processing. I am using Apache Spark services for analysis purposes which makes is for effecient when it comes to handelling larger datasets. Thus we import pyspark packages and register dataframe as a temptable

# In[8]:



import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# ### 4) Data Visualization
# 
# Once the processing of data is done as in the data insights are gained from Spark DF we then convert it back to Pandas DF so that using matpolib visualization of the gained insights can be achieved.

# In[ ]:


sparkRAW=sqlContext.createDataFrame(pandasRAW)
sparkRAW.registerTempTable("suicides")
import matplotlib.pyplot as plt
plt.close('all')


# ### 5) Data Analysis and gaining insights
# 
# ### - Report 1
# 
# First query denotes the general trend of suicide rates from 1985-2016. It shows the suicides percentage over the total population present in that year. The line graph shows the upvalues and the lowvalues for the same. We can see that around 11.5% of the overall population in 1985 committed suicide.
# 

# In[ ]:


query1=spark.sql("select year,round((sum(suicides_no)/sum(population))*100000,1) as suicide_percentage from suicides group by year order by year")
query1.show(n=10)
pandas1=query1.toPandas()
pandas1.cumsum()
plt.figure()
pandas1.plot.line(x='year',y='suicide_percentage')


# ### - Report 2 
# 
# Query 2 shows the suicide percentage in the year of 2004 with respect to country. We can see that Srilanka had the highest suicide rate over population in th e year 2004 followed by Russian federation
# 

# In[ ]:


query2=spark.sql("select country,year,round((sum(suicides_no)/sum(population))*100000,2) sums from suicides where year=2004 group by year,country order by year,sums DESC LIMIT 5")
query2.show(n=10)
pandas2=query2.toPandas()
pandas2.plot.bar(x='country',y='sums')


# ### - Report 3
# 
# Query 3 shows the total number of suicides happened till date. We can see that Russian federation has seen the highest suicide rate till date followed by United States. 

# In[ ]:


query3=spark.sql("select country,sum(suicides_no) total_suicides from suicides group by country order by total_suicides DESC LIMIT 5")
query3.show(n=10)
pandas3=query3.toPandas()
pandas3.plot.bar(x='country',y='total_suicides')


# 
# 
# 
# 
# ### - Report 4 :
#  
# Query 4 show the number of countries which were in high,moderate and low risk according to total suicides happened in 2015. First the total suicides accroding to country happended in 2015 is listed. And then if the suicides no. is greater than 1000 is listed as High, between 500 and 1000 is Moderate and between 0 to 500 is listed as low. If there are no suicides happenedn in a country in 2015 then it is listed as n/a.<br> We can see that there were 30 such countries which were in high risk of suicides in 2015.

# In[ ]:


from pyspark.sql.functions import udf
from pyspark.sql.types import *
query3=spark.sql("select country,sum(suicides_no) total_suicides from suicides where year=2015 group by country order by total_suicides DESC")
def valueToCategory(value):
   if   value >0 and value<=500: 
    return 'Low'
   elif value >500 and value<=1000:
    return 'Moderate'
   elif value >1000:
    return 'High'
   else: 
    return 'n/a'

udfValueToCategory = udf(valueToCategory, StringType())
query4= query3.withColumn("category", udfValueToCategory("total_suicides"))
query4.registerTempTable("temp")
finalQuery=spark.sql("select category,count(*) counts from temp group by category order by counts desc ")
finalQuery.show(n=10)
pandas4=finalQuery.toPandas()
pandas4.head
pandas4.plot(kind='pie',y='counts',labels=pandas4.category,autopct='%.2f')


# ### - Report 5:
# 
# Query 5 shows the number of total suicides happened in the year 2016 according to age. We can see that maximum of suicides have happened in the age-group of 35-54 years.

# In[ ]:


query5=spark.sql("select age,sum(suicides_no) sums from suicides where year=2016 group by age order by sums desc")
query5.show(n=50)
pandas5=query5.toPandas()
pandas5.plot.bar(x='age',y='sums')


# ### - Report 6
# 
# Query 6 shows the total number of suicides happened till date according to age. Maximum of suicides have been happened in 'Male' category. That is around 76.9%
# 

# In[ ]:


query6=spark.sql("select sex,sum(suicides_no) sums from suicides group by sex order by sums desc")
query6.show(n=50)
pandas6=query6.toPandas()
pandas6.plot(kind='pie',labels=pandas6.sex,y='sums',autopct='%.2f')


# 
# ### - Report 7
# 
# Query 7 shows the the age according to sex which have commited suuicides. <br>On summing the values we can find that most of the males in the age group of '35-54' years have committed  most suicides.<br> When dealing with females we can see that females of age-group '35-54 years' have commited most suicides.

# In[ ]:


query7_1=spark.sql("select sex x,age,sum(suicides_no) sums from suicides where sex='male' group by age,sex order by sums desc limit 1")
query7_2=spark.sql("select sex x,age,sum(suicides_no) sums from suicides where sex='female' group by age,sex order by sums desc limit 1")
final=query7_1.union(query7_2)
final.show(n=10)
pandas7=final.toPandas()
pandas7.plot.pie(y='sums',labels=['male_35-54','female_35-54'],autopct='%.2f')


# ### - Report 8
# 
# This query is proposed to check wether the GDP_per capita has any impact on the suicide rates. Per capita income is caluculated as GDP_year/total_population<br>
# To check the same we will use the gdp per capita income of Russian Federation in all years compare it with the suicide rates and gain insights from the data<br>
# For this value categorization is set as follows:<br>
# 1) Values between 0-10000 per capita income is set to LOW GDP<br>
# 2) Values greater than 10000 per capita income is set to HIGH GDP
# 
# __CONCLUSION__
# We can see that when the GDP_per capita income was high the number of suicide rates gradually decreased as compared to when there were LOW GDP_per capita incomes. Russian federation saw a decrease rate of nearly __12.22%__ when the per capita income increased.
# 
# __Thus per capita income has direct or indirect impact on suicide rates happening__

# In[ ]:


query8=spark.sql("select year,country,round(sum(suicides_no)/sum(population)*100000,2) sums,gdp_capita from suicides where country='Russian Federation' group by year,country,gdp_capita order by gdp_capita desc ")
def valueToCategory(value):
   if   value >0 and value<=10000: 
    return 'Low GDP'
   elif value >10000:
    return 'High GDP'
   else: 
    return 'n/a'

udfValueToCategory = udf(valueToCategory, StringType())
query8= query8.withColumn("category", udfValueToCategory("gdp_capita"))
query8.registerTempTable("temp")
final=spark.sql("select category,round(avg(sums),2) avg_suicide_rate from temp group by category")
final.show(n=10)
pandas8=final.toPandas()
pandas8.plot(kind='pie',labels=pandas8.category,y='avg_suicide_rate',autopct='%.2f')
final.registerTempTable("temp1")
decrease=spark.sql("select round(max(avg_suicide_rate)-min(avg_suicide_rate),2) avg_decrease_rate from temp1")
pandas8_result=decrease.toPandas()
pandas8_result.head()


# ### Report 9 
# 
# Query 9 shows the total suicides happened in the year 2000 according to Generation. We can see that maximum of suicides have happened with the people belonging to generation 'BOOMERS'

# In[ ]:


query9=spark.sql("select generation,sum(suicides_no) sums from suicides where year=2000 group by generation order by sums desc")
pandas9=query9.toPandas()
pandas9.plot(kind='pie',labels=pandas9.generation,y='sums',autopct='%.2f')


# ### - Report 10
# Query 10 shows the maximum suicides happened in the category of 'Male' and in the year of 1995

# In[ ]:


query10=spark.sql("select sex,max(suicides_no) from suicides where sex='male' AND year=1995 group by sex")
pandas10=query10.toPandas()
pandas10.head()

