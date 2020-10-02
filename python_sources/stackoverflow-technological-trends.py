#!/usr/bin/env python
# coding: utf-8

# # **Introduction**

# This kernel provides an analysis on the StackOverflow Data loaded as Google's BigQuery Dataset. It aims to find out the trend of various technologies from 2009 to 2018 and predict the future technological trends using Linear Regression.
# 
# These are some of the questions that this notebook aims to answer for now:  
# 1) What is the trend in the technologies from 2009 - 2018?  
# 2) What is the trend in the various categories of each technology?  
# 3) What will be the upcoming trends in the technologies?  
# 4) What will be the upcoming trend in the various categories of each technology?
# 
# These are some of the technologies that this kernel discusses:
#    * **Web Development**
#       * AngularJs
#       * BootStrap
#       * PHP
#       * HTML
#       * JavaScript
#       * CSS
#    * **DataBase Technologies**
#       * MySQL
#       * MongoDB
#       * NoSQL
#       * PostgreSQL
#       * Cassandra
#    * **Big Data**
#       * Hadoop
#       * Hive
#       * Spark
#       * HBase
#       * Kafka
#    * **Data Science**
#       * Pandas
#       * Matplotlib
#       * Regression
#       * Support Vector Machines (SVM)
#       * Kaggle
#    * **Programming Languages**
#       * C++
#       * Ruby
#       * Java
#       * C#
#       * Python

# # **Importing the Packages**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # graphs and charts
import pandas_profiling # generating Profile Report

import bq_helper # accessing bigQuery database

import sklearn
from sklearn.model_selection import train_test_split # data splitting
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LinearRegression # Linear model

import wordcloud


# # **Accessing the Dataset**
#    * Accessing the required dataset from BigQuery
#    
# ### About '*bq_helper*'
#    * <p style='text-align: justify;'> bq_helper package simplifies common read-only tasks in BigQuery by dealing with object references and unpacking result objects into pandas dataframes.</p>
#    * <p style='text-align: justify;'> It currently only works here on Kaggle as it does not have any handling for the BigQuery authorization functions that Kaggle handles behind the scenes. </p>
#    * <p style='text-align: justify;'>bq_helper requires the creation of one BigQueryHelper object per dataset. Let's make one now. We'll need to pass it two arguments: </p>
#       1) The name of the BigQuery project, which on Kaggle should always be bigquery-public-data  
#       2) The name of the dataset, which can be found in the dataset description  

# In[ ]:


stackoverflow = bq_helper.BigQueryHelper("bigquery-public-data","stackoverflow")


# # **Dataset Tables**
#    * Listing the tables in the dataset

# In[ ]:


stackoverflow.list_tables()


# # **Exploratory Data Analysis (EDA)**
# 
# ### **1) 'head' function**
#    * Used to view top n rows of the table - "posts_questions" in the dataset, by default n=5

# In[ ]:


stackoverflow.head("posts_questions")


# ### **2) Table Schema**
#    * Finding the schema of the table queried to get some more details about the table columns

# In[ ]:


stackoverflow.table_schema("posts_questions")


# # **Cleaning the Data**
# ## **Posts Count**
#    * Removing irrelevant data
#    * Querying the year and the number of posts per year from the '*posts_questions*' table

# In[ ]:


queryx = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >= 2009 and extract(year from creation_date) < 2019
        group by year
        order by year
        """

PostsCount = stackoverflow.query_to_pandas(queryx)
print(PostsCount)


# In[ ]:


PostsCount.describe()


# * Taking care of the null/missing values in the dataset

# In[ ]:


# data.isnull.sum()
# data['favorite_count'].fillna(0,inplace=True)
# data.head()


# # **Data Profiling**
#    * Displaying the Profile Report of the dataframe using the '_ProfileReport()_' method of the '*pandas_profiling*' library

# In[ ]:


data = pd.DataFrame(PostsCount)
pandas_profiling.ProfileReport(data)


# ## **PostsCount Basic Look**
#    * Viewing the top five rows of the '_PostsCount_' dataframe to get an idea about the dataframe's structure

# In[ ]:


PostsCount.head()


# # **WordCloud**

# In[ ]:


query4 = """SELECT tags
         FROM `bigquery-public-data.stackoverflow.posts_questions`
         LIMIT 200000;
         """

alltags = stackoverflow.query_to_pandas_safe(query4)
tags = ' '.join(alltags.tags).lower()


# In[ ]:


cloud = wordcloud.WordCloud(background_color='black',
                            max_font_size=200,
                            width=1600,
                            height=800,
                            max_words=300,
                            relative_scaling=.5).generate(tags)
plt.figure(figsize=(20,10))
plt.axis('off')
plt.savefig('stackOverflow.png')
plt.imshow(cloud);


# ## **Reformatting the column type**
#    * Changing the datatype of the column **'_year_'** in the dataframe to **'_numeric_'** type

# In[ ]:


pd.to_numeric(PostsCount['year'])


# ## **Reshaping the columns**
#    * Storing reshaped columns of the dataframe in new variables

# In[ ]:


year=PostsCount['year'].values.reshape(-1,1)
#print (year)
posts=PostsCount['posts'].values.reshape(-1,1)
#print (posts)


# # **Linear Regression**
#    * Performing Linear Regression to predict future values using the past data
#    * Creating the model

# In[ ]:


reg = LinearRegression()


# ## **Train and Test Data**
#    * Splitting the data into train and test using '*train_test_split()*' method

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(year,posts,test_size=0.2,shuffle=False)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


# ## **Fitting and Predicting**
#    * Training the model using the training data and then using it to predict the values for the test data 

# In[ ]:


reg.fit(X_train,y_train)
predictions = reg.predict(X_test)


# In[ ]:


print('Predicted values\n',predictions)


# # **Visualisations**
#    * Visualising the training data and the test data and the predictions for better understanding

# In[ ]:


plt.scatter(X_train,y_train, color = "black")
plt.scatter(X_test, y_test, color = "green")
plt.plot(X_test, predictions, color = "red")
plt.gca().legend(('Y-Predicted','Y-Train', 'Y-Test'))
plt.title('Y-train and Y-test and Y-predicted')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


#    * Visualising only the test values and the predicted values to check the accuracy of the model

# In[ ]:


plt.scatter(X_test, y_test, color = "green")
plt.plot(X_test, predictions, color = "red")
plt.gca().legend(('Y-Train','Y-Test'))
plt.title('Y-test and Y-predicted')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


# # **Model Accuracy Score and Error**
# * Finding the score of the model for this data
# * Finding the mean squared error and root mean squared error

# In[ ]:


reg.score(X_test,y_test)


# In[ ]:


print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))


# In[ ]:


print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# # **TECHNOLOGIES DISCUSSED**:

# # **WEB DEVELOPMENT**
#    * Finding the percentage of Web Development posts with respect to total posts each year

# In[ ]:


#angularjs,bootstrap,php,html,javascript,css
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and (tags like '%bootstrap%' or 
        tags like '%angularjs%' or tags like '%php%' or tags like '%html%' or tags like '%javascript%' or tags like '%css%')
        group by year
        order by year
        """

WebDev_Posts = stackoverflow.query_to_pandas(query)
WebDev_Posts['posts']= WebDev_Posts['posts']*100/PostsCount.posts
WebDev_Posts


# In[ ]:


WebDev_Posts.describe()


# ## **Reformatting the columns and Reshaping**
#    * Changing the datatype of the **'_year_'** column to **'_numeric_'** type
#    * Storing the reshaped columns of the dataframe in new variables

# In[ ]:


pd.to_numeric(WebDev_Posts['year'])


# In[ ]:


WebDevYear=WebDev_Posts['year'].values.reshape(-1,1)
#print (WebDevYear)
WebDevPosts=WebDev_Posts['posts'].values.reshape(-1,1)
#print (WebDevPosts)


# ## **Train data and Test data**
#    * Splitting the data into train and test using '*train_test_split()*' method

# In[ ]:


XWebDev_train, XWebDev_test, yWebDev_train, yWebDev_test = train_test_split(WebDevYear,WebDevPosts,test_size=0.2,shuffle=False)
# print(XWebDev_train)
# print(XWebDev_test)
# print(yWebDev_train)
# print(yWebDev_test)


# ## **Linear Regression Model and  Prediction**

# In[ ]:


WebDevReg=LinearRegression()
WebDevReg.fit(XWebDev_train,yWebDev_train)
WebDevPredictions = WebDevReg.predict(XWebDev_test)
print('Predicted Values:\n',WebDevPredictions)


# ## **Visualisations**
#    * Visualising the training data and the test data and the predictions for better understanding

# In[ ]:


plt.scatter(XWebDev_train,yWebDev_train, color = "black")
plt.scatter(XWebDev_test, yWebDev_test, color = "green")
plt.plot(XWebDev_test, WebDevPredictions, color = "red")
plt.gca().legend(('Y-Predicted','Y-Train', 'Y-Test'))
plt.title('WEB DEVELOPMENT')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


#    * Visualising the test values and the predicted values to check the accuracy of the model

# In[ ]:


plt.scatter(XWebDev_test, yWebDev_test, color = "green")
plt.plot(XWebDev_test, WebDevPredictions, color = "red")
plt.gca().legend(('Y-Train','Y-Test'))
plt.title('Web Development')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


# ## **Model Accuracy Score and Error**

# In[ ]:


WebDevReg.score(XWebDev_test,yWebDev_test)


# In[ ]:


print('Mean Squared Error:',metrics.mean_squared_error(yWebDev_test, WebDevPredictions))


# In[ ]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yWebDev_test, WebDevPredictions)))


# # **1) AngularJS**
#    * Finding the percentage of AngularJS posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%angularjs%'
        group by year
        order by year
        """

AngularJSPosts = stackoverflow.query_to_pandas(query)
AngularJSPosts['posts']= AngularJSPosts['posts']*100/PostsCount.posts
AngularJSPosts


# # **2) BootStrap**
#    * Finding the percentage of BootStrap posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%bootstrap%'
        group by year
        order by year
        """

BootstrapPosts = stackoverflow.query_to_pandas(query)
BootstrapPosts['posts']= BootstrapPosts['posts']*100/PostsCount.posts
pd.to_numeric(BootstrapPosts['year'])
BootstrapPosts


# # **3) PHP**
#    * Finding the percentage of PHP posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%php%'
        group by year
        order by year
        """

PHPPosts = stackoverflow.query_to_pandas(query)
PHPPosts['posts']= PHPPosts['posts']*100/PostsCount.posts
pd.to_numeric(PHPPosts['year'])
PHPPosts


# # **4) HTML**
#    * Finding the percentage of HTML posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%html%'
        group by year
        order by year
        """

htmlPosts = stackoverflow.query_to_pandas(query)
htmlPosts['posts']= htmlPosts['posts']*100/PostsCount.posts
pd.to_numeric(htmlPosts['year'])
htmlPosts


# # **5) JavaScript**
#    * Finding the percentage of JavaScript posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%javascript%'
        group by year
        order by year
        """

JavaScriptPosts = stackoverflow.query_to_pandas(query)
JavaScriptPosts['posts']= JavaScriptPosts['posts']*100/PostsCount.posts
pd.to_numeric(JavaScriptPosts['year'])
JavaScriptPosts


# # **6) CSS**
#    * Finding the percentage of CSS posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%css%'
        group by year
        order by year
        """

CSSPosts = stackoverflow.query_to_pandas(query)
CSSPosts['posts']= CSSPosts['posts']*100/PostsCount.posts
pd.to_numeric(CSSPosts['year'])
CSSPosts


# # **Comparisons: WebDev**
#    * Comparing the popularities of the various categories under Web Development

# In[ ]:


WebDev= pd.merge(PHPPosts, htmlPosts, how='inner', on = 'year')
WebDev=WebDev.set_index('year')
WebDev= pd.merge(WebDev, JavaScriptPosts, how='inner', on = 'year')
WebDev =WebDev.set_index('year')
WebDev=pd.merge(WebDev,AngularJSPosts,how='inner',on='year')
WebDev = WebDev.set_index('year')
WebDev=pd.merge(WebDev,BootstrapPosts,how='inner',on='year')
WebDev = WebDev.set_index('year')
WebDev=pd.merge(WebDev,CSSPosts,how='inner',on='year')
WebDev = WebDev.set_index('year')

WebDev.plot(kind='line')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Posts %', fontsize=15)
y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

plt.xticks(y_pos,fontsize=10)
plt.yticks(fontsize=10)
plt.title('Web Development')
plt.legend(['PHP','HTML','JavaScript','AngularJS','BootStrap','CSS'],loc=[1.0,0.5])
plt.show()


# # **DATABASE TECHNOLOGIES**
#   * Finding the percentage of Database Technologies posts with respect to total posts each year

# In[ ]:


#mysql,mongodb,nosql,postgresql,cassandra
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date)>= 2009 and extract(year from creation_date) < 2019 
        and (tags like '%mysql%' or tags like '%nosql%' or tags like '%mongodb%' 
        or tags like '%postgresql%' or tags like '%cassandra%')
        group by year
        order by year
        """

DataBase_Posts = stackoverflow.query_to_pandas(query)
DataBase_Posts['posts']= DataBase_Posts['posts']*100/PostsCount.posts
DataBase_Posts


# In[ ]:


DataBase_Posts.describe()


# ## **Reformatting the columns and Reshaping**
#   * Changing the datatype of the **'_year_'** column to **'_numeric_'** type
#   * Storing the reshaped columns of the dataframe in new variables

# In[ ]:


pd.to_numeric(DataBase_Posts['year'])


# In[ ]:


DataBaseYear=DataBase_Posts['year'].values.reshape(-1,1)
# print (DataBaseYear)
DataBasePosts=DataBase_Posts['posts'].values.reshape(-1,1)
# print (DataBasePosts)


# ## **Train data and Test data**
#    * Splitting the data into train and test using *'train_test_split()'* method

# In[ ]:


XDataBase_train, XDataBase_test, yDataBase_train, yDataBase_test = train_test_split(DataBaseYear,DataBasePosts,test_size=0.2,shuffle=False)
# print(XDataBase_train)
# print(XDataBase_test)
# print(yDataBase_train)
# print(yDataBase_test)


# ## **Linear Regression Model and Prediction**

# In[ ]:


DataBaseReg=LinearRegression()
DataBaseReg.fit(XDataBase_train,yDataBase_train)
DataBasePredictions = DataBaseReg.predict(XDataBase_test)
print('Predicted Values:\n',DataBasePredictions)


# ## **Visualisations**
#    * Visualising the training data and the test data and the predictions for better understanding

# In[ ]:


plt.scatter(XDataBase_train,yDataBase_train, color = "black")
plt.scatter(XDataBase_test, yDataBase_test, color = "green")
plt.plot(XDataBase_test, DataBasePredictions, color = "red")
plt.gca().legend(('Y-Predicted', 'Y-Train','Y-Test'))
plt.title('Database Technologies')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


#   * Visualising the test values and the predicted values to check the accuracy of the model

# In[ ]:


plt.scatter(XDataBase_test, yDataBase_test, color = "green")
plt.plot(XDataBase_test, DataBasePredictions, color = "red")
plt.gca().legend(('Y-Train','Y-Test'))
plt.title('Database Technologies')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


# ## **Model Accuracy Score and Error**

# In[ ]:


DataBaseReg.score(XDataBase_test, yDataBase_test)


# In[ ]:


print('Mean Squared Error:', metrics.mean_squared_error(yDataBase_test, DataBasePredictions))


# In[ ]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yDataBase_test, DataBasePredictions)))


# # **1) MySQL**
#    * Finding the percentage of MySQL posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%mysql%'
        group by year
        order by year
        """

MySQLPosts = stackoverflow.query_to_pandas(query)
MySQLPosts['posts']= MySQLPosts['posts']*100/PostsCount.posts
pd.to_numeric(MySQLPosts['year'])
MySQLPosts


# # **2) MongoDB**
#    * Finding the percentage of MongoDB posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%mongodb%'
        group by year
        order by year
        """

MongoDBPosts = stackoverflow.query_to_pandas(query)
MongoDBPosts['posts']= MongoDBPosts['posts']*100/PostsCount.posts
pd.to_numeric(MongoDBPosts['year'])
MongoDBPosts


# # **3) NoSQL**
#    * Finding the percentage of NoSQL posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%nosql%'
        group by year
        order by year
        """

NoSQLPosts = stackoverflow.query_to_pandas(query)
NoSQLPosts['posts']= NoSQLPosts['posts']*100/PostsCount.posts
pd.to_numeric(NoSQLPosts['year'])
NoSQLPosts


# # **4) PostgreSQL**
#    * Finding the percentage of PostgreSQL posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%postgresql%'
        group by year
        order by year
        """

PostgreSQLPosts = stackoverflow.query_to_pandas(query)
PostgreSQLPosts['posts']= PostgreSQLPosts['posts']*100/PostsCount.posts
pd.to_numeric(PostgreSQLPosts['year'])
PostgreSQLPosts


# # **5) Cassandra**
#   * Finding the percentage of Cassandra posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 
        and tags like '%cassandra%'
        group by year
        order by year
        """

CassandraPosts = stackoverflow.query_to_pandas(query)
CassandraPosts['posts']= CassandraPosts['posts']*100/PostsCount.posts
pd.to_numeric(CassandraPosts['year'])
CassandraPosts


# # **Comparisons: DataBase**
#    * Comparing the popularities of the various categories under Database Technologies

# In[ ]:


DataBase= pd.merge(MySQLPosts, NoSQLPosts, how='inner', on = 'year')
DataBase=DataBase.set_index('year')
DataBase= pd.merge(DataBase, MongoDBPosts, how='inner', on = 'year')
DataBase=DataBase.set_index('year')
DataBase= pd.merge(DataBase, PostgreSQLPosts, how='inner', on = 'year')
DataBase=DataBase.set_index('year')
DataBase= pd.merge(DataBase, CassandraPosts, how='inner', on = 'year')
DataBase=DataBase.set_index('year')


DataBase.plot(kind='line')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Posts %', fontsize=15)
y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

plt.xticks(y_pos,fontsize=10)
plt.yticks(fontsize=10)
plt.title('Database Technologies')
plt.legend(['MySQL','NoSQL','MongoDB','PostgreSQL','Cassandra'],loc=[1.0,0.5])
plt.show()


# # **BIG DATA**
#   * Finding the percentage of Big Data posts with respect to total posts each year

# In[ ]:


#hadoop,hive,spark,hbase,kafka
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date)>= 2009 and extract(year from creation_date) < 2019 and (tags like '%hadoop%' or 
        tags like '%spark%' or tags like '%hive%' or tags like '%hbase%' or tags like '%kafka%')
        group by year
        order by year
        """

BigData_Posts = stackoverflow.query_to_pandas(query)
BigData_Posts['posts']= BigData_Posts['posts']*100/PostsCount.posts
BigData_Posts


# In[ ]:


BigData_Posts.describe()


# ## **Reformatting the columns and Reshaping**
#    * Changing the datatype of **'_year_'** column to **'_numeric_'** type
#    * Storing the reshaped columns of the dataframe in new variables

# In[ ]:


pd.to_numeric(BigData_Posts['year'])


# In[ ]:


BigDataYear=BigData_Posts['year'].values.reshape(-1,1)
# print (BigDataYear)
BigDataPosts=BigData_Posts['posts'].values.reshape(-1,1)
# print (BigDataPosts)


# ## **Train data and Test data**
#    * Splitting the data into train and test data using *'train_test_split()'* method

# In[ ]:


XBigData_train, XBigData_test, yBigData_train, yBigData_test = train_test_split(BigDataYear,BigDataPosts,test_size=0.2,shuffle=False)
# print(XBigData_train)
# print(XBigData_test)
# print(yBigData_train)
# print(yBigData_test)


# ## **Linear Regression Model and Prediction**

# In[ ]:


BigDataReg=LinearRegression()
BigDataReg.fit(XBigData_train,yBigData_train)
BigDataPredictions = BigDataReg.predict(XBigData_test)
print('Predicted Values:\n',BigDataPredictions)


# ## **Visualisations**
#    * Visualising the training data and the test data and the predictions for better understanding

# In[ ]:


plt.scatter(XBigData_train,yBigData_train, color = "black")
plt.scatter(XBigData_test, yBigData_test, color = "green")
plt.plot(XBigData_test, BigDataPredictions, color = "red")
plt.gca().legend(('Y-Predicted', 'Y-Train','Y-Test'))
plt.title('Big Data')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


#    * Visualising the test values and the predicted values to check the accuracy of the model

# In[ ]:


plt.scatter(XBigData_test, yBigData_test, color = "green")
plt.plot(XBigData_test, BigDataPredictions, color = "red")
plt.gca().legend(('Y-Train','Y-Test'))
plt.title('Big Data')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


# ## **Model Accuracy Score and Error**

# In[ ]:


BigDataReg.score(XBigData_test, yBigData_test)


# In[ ]:


print('Mean Squared Error:', metrics.mean_squared_error(yBigData_test, BigDataPredictions))


# In[ ]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yBigData_test, BigDataPredictions)))


# # **1) Hadoop**
#    * Finding the percentage of Hadoop posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%hadoop%'
        group by year
        order by year
        """

HadoopPosts = stackoverflow.query_to_pandas(query)
HadoopPosts['posts']= HadoopPosts['posts']*100/PostsCount.posts
pd.to_numeric(HadoopPosts['year'])
HadoopPosts


# # **2) Hive**
#    * Finding the percentage of Hive posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%hive%'
        group by year
        order by year
        """

HivePosts = stackoverflow.query_to_pandas(query)
HivePosts['posts']= HivePosts['posts']*100/PostsCount.posts
pd.to_numeric(HivePosts['year'])
HivePosts


# # **3) Spark**
#    * Finding the percentage of Spark posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%spark%'
        group by year
        order by year
        """

SparkPosts = stackoverflow.query_to_pandas(query)
SparkPosts['posts']= SparkPosts['posts']*100/PostsCount.posts
pd.to_numeric(SparkPosts['year'])
SparkPosts


# # **4) HBase**
#    * Finding the percentage of HBase posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%hbase%'
        group by year
        order by year
        """

HBasePosts = stackoverflow.query_to_pandas(query)
HBasePosts['posts']= HBasePosts['posts']*100/PostsCount.posts
pd.to_numeric(HBasePosts['year'])
HBasePosts


# # **5) Kafka**
#    * Finding the percentage of Kafka posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%kafka%'
        group by year
        order by year
        """

KafkaPosts = stackoverflow.query_to_pandas(query)
KafkaPosts['posts']= KafkaPosts['posts']*100/PostsCount.posts
pd.to_numeric(KafkaPosts['year'])
KafkaPosts


# In[ ]:


df = pd.DataFrame({"year":[2009,2010],"posts":[0,0]})
KafkaPosts = KafkaPosts.append(df, ignore_index = True)
KafkaPosts.sort_values("year", axis = 0, ascending = True, inplace = True)
KafkaPosts = KafkaPosts.reset_index(drop=True)
KafkaPosts


# # **Comparisons: BigData**
#    * Comparing the popularities of the various categories under Big Data

# In[ ]:


BigData= pd.merge(HadoopPosts, SparkPosts, how='inner', on = 'year')
BigData=BigData.set_index('year')
BigData= pd.merge(BigData, HivePosts, how='inner', on = 'year')
BigData=BigData.set_index('year')
BigData= pd.merge(BigData, HBasePosts, how='inner', on = 'year')
BigData=BigData.set_index('year')
BigData= pd.merge(BigData, KafkaPosts, how='inner', on = 'year')
BigData=BigData.set_index('year')

BigData.plot(kind='line')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Posts %', fontsize=15)
y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

plt.xticks(y_pos,fontsize=10)
plt.yticks(fontsize=10)
plt.title('Big Data')
plt.legend(['Hadoop','Spark','Hive','HBase','Kafka'],loc=[1.0,0.5])
plt.show()


# # **DATA SCIENCE**
#    * Finding the percentage of Data Science posts with respect to total posts each year

# In[ ]:


#pandas,matplotlib,regression,svm,kaggle
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date)>= 2009 and extract(year from creation_date) < 2019 
        and (tags like '%pandas%' or tags like '%matplotlib%'
        or tags like '%regression%' or tags like '%svm%' or tags like '%kaggle%')
        group by year
        order by year
        """

DataScience_Posts = stackoverflow.query_to_pandas(query)
DataScience_Posts['posts']= DataScience_Posts['posts']*100/PostsCount.posts
DataScience_Posts


# In[ ]:


DataScience_Posts.describe()


# ## **Reformatting the columns and Reshaping**
#    * Changing the datatype of **'_year_'** column to **'_numeric_'** type
#    * Storing the reshaped columns of the dataframe in new variables

# In[ ]:


pd.to_numeric(DataScience_Posts['year'])


# In[ ]:


DataScienceYear=DataScience_Posts['year'].values.reshape(-1,1)
# print (DataScienceYear)
DataSciencePosts=DataScience_Posts['posts'].values.reshape(-1,1)
# print (DataSciencePosts)


# ## **Train data and Test data**
#    * Splitting the data into train and test using *'train_test_split()'* method

# In[ ]:


XDataScience_train, XDataScience_test, yDataScience_train, yDataScience_test = train_test_split(DataScienceYear,DataSciencePosts,test_size=0.2,shuffle=False)
# print(XDataScience_train)
# print(XDataScience_test)
# print(yDataScience_train)
# print(yDataScience_test)


# ## **Linear Regression Model and Prediction**

# In[ ]:


DataScienceReg=LinearRegression()
DataScienceReg.fit(XDataScience_train,yDataScience_train)
DataSciencePredictions = DataScienceReg.predict(XDataScience_test)
print('Predicted Values:\n',DataSciencePredictions)


# ## **Visualisations**
#   * Visualising the training data and the test data and the predictions for better understanding

# In[ ]:


plt.scatter(XDataScience_train,yDataScience_train, color = "black")
plt.scatter(XDataScience_test, yDataScience_test, color = "green")
plt.plot(XDataScience_test, DataSciencePredictions, color = "red")
plt.gca().legend(('Y-Predicted', 'Y-Train','Y-Test'))
plt.title('Data Science')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


#    * Visualising the test values and the predicted values to check the accuracy of the model

# In[ ]:


plt.scatter(XDataScience_test, yDataScience_test, color = "green")
plt.plot(XDataScience_test, DataSciencePredictions, color = "red")
plt.gca().legend(('Y-Train','Y-Test'))
plt.title('Data Science')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


# ## **Model Accuracy Score and Error**

# In[ ]:


DataScienceReg.score(XDataScience_test,yDataScience_test)


# In[ ]:


print('Mean Squared Error:', metrics.mean_squared_error(yDataScience_test, DataSciencePredictions))


# In[ ]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yDataScience_test, DataSciencePredictions)))


# # **1) Pandas**
#    * Finding the percentage of Pandas posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%pandas%'
        group by year
        order by year
        """

PandasPosts = stackoverflow.query_to_pandas(query)
PandasPosts['posts']= PandasPosts['posts']*100/PostsCount.posts
pd.to_numeric(PandasPosts['year'])
PandasPosts


# In[ ]:


df = pd.DataFrame({"year":[2009],"posts":[0]})
PandasPosts = PandasPosts.append(df, ignore_index = True)
PandasPosts.sort_values("year", axis = 0, ascending = True, inplace = True)
PandasPosts = PandasPosts.reset_index(drop=True)
PandasPosts


# # **2) Matplotlib**
#    * Finding the percentage of Matplotlib posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%matplotlib%'
        group by year
        order by year
        """

MatplotlibPosts = stackoverflow.query_to_pandas(query)
MatplotlibPosts['posts']= MatplotlibPosts['posts']*100/PostsCount.posts
pd.to_numeric(MatplotlibPosts['year'])
MatplotlibPosts


# # **3) Regression**
#    * Finding the percentage of Regression posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 
        and tags like '%regression%'
        group by year
        order by year
        """

RegressionPosts = stackoverflow.query_to_pandas(query)
RegressionPosts['posts']= RegressionPosts['posts']*100/PostsCount.posts
pd.to_numeric(RegressionPosts['year'])
RegressionPosts


# # **4) SVM**
#    * Finding the percentage of SVM posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 
        and tags like '%svm%'
        group by year
        order by year
        """

SVMPosts = stackoverflow.query_to_pandas(query)
SVMPosts['posts']= SVMPosts['posts']*100/PostsCount.posts
pd.to_numeric(SVMPosts['year'])
SVMPosts


# # **5) Kaggle**
#    * Finding the percentage of Kaggle posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 
        and tags like '%kaggle%'
        group by year
        order by year
        """

KagglePosts = stackoverflow.query_to_pandas(query)
KagglePosts['posts']= KagglePosts['posts']*100/PostsCount.posts
pd.to_numeric(KagglePosts['year'])
KagglePosts


# In[ ]:


df = pd.DataFrame({"year":[2009,2010],"posts":[0,0]})
KagglePosts = KagglePosts.append(df, ignore_index = True)
KagglePosts.sort_values("year", axis = 0, ascending = True, inplace = True)
KagglePosts = KagglePosts.reset_index(drop=True)
KagglePosts


# # **Comparisons: DataScience**
#    * Comparing the popularities of the various categories under Data Science

# In[ ]:


DataScience= pd.merge(PandasPosts, MatplotlibPosts, how='inner', on = 'year')
DataScience=DataScience.set_index('year')
DataScience= pd.merge(DataScience, RegressionPosts, how='inner', on = 'year')
DataScience=DataScience.set_index('year')
DataScience= pd.merge(DataScience, SVMPosts, how='inner', on = 'year')
DataScience=DataScience.set_index('year')
DataScience= pd.merge(DataScience, KagglePosts, how='inner', on = 'year')
DataScience=DataScience.set_index('year')

DataScience.plot(kind='line')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Posts %', fontsize=15)
y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

plt.xticks(y_pos,fontsize=10)
plt.yticks(fontsize=10)
plt.title('Data Science')
plt.legend(['Pandas','Matplotlib','Regression','SVM','Kaggle'],loc=[1.0,0.5])
plt.show()


# # **PROGRAMMING LANGUAGES**
#    * Finding the percentage of programming languages posts with respect to total posts each year

# In[ ]:


#C++,ruby,java,c#,python
query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >= 2009 and extract(year from creation_date) < 2019 
        and (tags like '%c++%' or tags like '%python%' or tags like '%ruby%' 
        or tags like '%c#%' or (tags like '%java%' and tags not like '%javascript%'))
        group by year
        order by year
        """

ProgLang_Posts = stackoverflow.query_to_pandas(query)
ProgLang_Posts['posts']=ProgLang_Posts['posts']*100/PostsCount.posts
ProgLang_Posts


# In[ ]:


ProgLang_Posts.describe()


# ## **Reformatting the columns and Reshaping**
#    * Changing the datatype of the **'_year_'** column to **'_numeric_'** type
#    * Storing the reshaped columns of the dataframe in new variables

# In[ ]:


pd.to_numeric(ProgLang_Posts['year'])


# In[ ]:


ProgLangYear=ProgLang_Posts['year'].values.reshape(-1,1)
# print (ProgLangYear)
ProgLangPosts=ProgLang_Posts['posts'].values.reshape(-1,1)
# print (ProgLangPosts)


# ## **Train data and Test data**
#    * Splitting the data into train and test using *'train_test_split()* method

# In[ ]:


XProgLang_train, XProgLang_test, yProgLang_train, yProgLang_test = train_test_split(ProgLangYear,ProgLangPosts,test_size=0.2,shuffle=False)
# print(XProgLang_train)
# print(XProgLang_test)
# print(yProgLang_train)
# print(yProgLang_test)


# ## **Linear Regression Model and Prediction**

# In[ ]:


ProgLangReg=LinearRegression()
ProgLangReg.fit(XProgLang_train,yProgLang_train)
ProgLangPredictions = ProgLangReg.predict(XProgLang_test)
print('Predicted Values:\n',ProgLangPredictions)


# ## **Visualisations**
#    * Visualising the training data and the testing data and the predictions for better understanding

# In[ ]:


plt.scatter(XProgLang_train,yProgLang_train, color = "black")
plt.scatter(XProgLang_test, yProgLang_test, color = "green")
plt.plot(XProgLang_test, ProgLangPredictions, color = "red")
plt.gca().legend(('Y-Predicted', 'Y-Train','Y-Test'))
plt.title('Programming Languages')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


#    * Visualising the test values and the predicted values to check the accuracy of the model

# In[ ]:


plt.scatter(XProgLang_test, yProgLang_test, color = "green")
plt.plot(XProgLang_test, ProgLangPredictions, color = "red")
plt.gca().legend(('Y-Train','Y-Test'))
plt.title('Programming Languages')
plt.xlabel('Year')
plt.ylabel('Posts')
plt.show()


# ## **Model Accuracy Score and Error**

# In[ ]:


ProgLangReg.score(XProgLang_test, yProgLang_test)


# In[ ]:


print('Mean Squared Error:', metrics.mean_squared_error(yProgLang_test, ProgLangPredictions))


# In[ ]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yProgLang_test, ProgLangPredictions)))


# # **1) C++**
#    * Finding the percentage of C++ posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%c++%'
        group by year
        order by year
        """

CplusPosts = stackoverflow.query_to_pandas(query)
CplusPosts['posts']= CplusPosts['posts']*100/PostsCount.posts
pd.to_numeric(CplusPosts['year'])
CplusPosts


# # **2) Ruby**
#    * Finding the percentage of Ruby posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%ruby%'
        group by year
        order by year
        """

RubyPosts = stackoverflow.query_to_pandas(query)
RubyPosts['posts']= RubyPosts['posts']*100/PostsCount.posts
pd.to_numeric(RubyPosts['year'])
RubyPosts


# # **3) Java**
#    * Finding the percentage of Java posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%java%' and tags not like '%javascript%'
        group by year
        order by year
        """

JavaPosts = stackoverflow.query_to_pandas(query)
JavaPosts['posts']= JavaPosts['posts']*100/PostsCount.posts
pd.to_numeric(JavaPosts['year'])
JavaPosts


# # **4) C#**
#    * Finding the percentage of C# posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%c#%'
        group by year
        order by year
        """

CHashPosts = stackoverflow.query_to_pandas(query)
CHashPosts['posts']= CHashPosts['posts']*100/PostsCount.posts
pd.to_numeric(CHashPosts['year'])
CHashPosts


# # **5) Python**
#    * Finding the percentage of Python posts with respect to total posts each year

# In[ ]:


query = """select EXTRACT(year FROM creation_date) AS year, sum(id) as posts
        from `bigquery-public-data.stackoverflow.posts_questions`
        where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%python%'
        group by year
        order by year
        """

PythonPosts = stackoverflow.query_to_pandas(query)
PythonPosts['posts']= PythonPosts['posts']*100/PostsCount.posts
pd.to_numeric(PythonPosts['year'])
PythonPosts


# # **Comparisons: ProgLang**
#    * Comparing the popularities of the various categories under Programming Languages

# In[ ]:


ProgLang= pd.merge(RubyPosts, CplusPosts, how='inner', on = 'year')
ProgLang =ProgLang.set_index('year')
ProgLang= pd.merge(ProgLang, PythonPosts, how='inner', on = 'year')
ProgLang =ProgLang.set_index('year')
ProgLang=pd.merge(ProgLang,CHashPosts,how='inner',on='year')
ProgLang = ProgLang.set_index('year')
ProgLang=pd.merge(ProgLang,JavaPosts,how='inner',on='year')
ProgLang = ProgLang.set_index('year')

ProgLang.plot(kind='line')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Posts %', fontsize=15)
y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

plt.xticks(y_pos,fontsize=10)
plt.yticks(fontsize=10)
plt.title('Programming Languages')
plt.legend(['Ruby','C++','Python','C#','Java'],loc=[1.0,0.5])
plt.show()


# # **Past Trends Comparison**
# 

# In[ ]:


PastTrends= pd.merge(WebDev_Posts, DataBase_Posts, how='inner', on = 'year')
PastTrends =PastTrends.set_index('year')
PastTrends= pd.merge(PastTrends, BigData_Posts, how='inner', on = 'year')
PastTrends =PastTrends.set_index('year')
PastTrends=pd.merge(PastTrends,DataScience_Posts,how='inner',on='year')
PastTrends = PastTrends.set_index('year')
PastTrends=pd.merge(PastTrends,ProgLang_Posts,how='inner',on='year')
PastTrends = PastTrends.set_index('year')

PastTrends.plot(kind='line')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Posts %', fontsize=15)
y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

plt.xticks(y_pos,fontsize=10)
plt.yticks(fontsize=10)
plt.title('Past Trends')
plt.legend(['Web Development','DataBase Technologies','Big Data','Data Science','Programming Languages'],
           loc=[1.0,0.5])
plt.show()


# # **Future Trends Comparison**
# <p style='text-align: justify;'>The following function is used to create stacked graphs visualisations for any of the various technologies to compare the trends amongst them. The datasets for the technolgies or their sub categories can be passed as the parameter and these datasets can be variable in number starting from 2.</p> 
# <p style='text-align: justify;'>The labels for the various technologies and the title for the visualisation can also be passed, but these fields are optional and are set to **None** and **'Trends in Technologies in 2019'** by default, respectively. *'Year'* can also be passed as a parameter to the function and the function predicts the trends for the passed year or 2019 as default.</p>

# In[ ]:


def trends(dfall, labels=None, Year = 2019, title="Trends in Technologies in ", **kwargs):

    plt.figure(figsize=(20,10))
   
    predict = []
    for df in dfall :
        year=df['year'].values.reshape(-1,1)
        posts=df['posts'].values.reshape(-1,1)
        reg=LinearRegression()
        X_train = year
        Y_train = posts
        X_test = [[Year]]
        reg.fit(X_train,Y_train)
        predictions = reg.predict(X_test)
        predict.append(predictions)

    trend = pd.DataFrame(columns = ['Technology','Posts %'])
    trend['Technology'] = labels
    trend['Posts %'] = predict
    
    x_pos = np.arange(len(trend['Technology']))
    plt.bar(x_pos,trend['Posts %'])
    plt.xticks(x_pos, trend['Technology'],fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Technologies',fontsize=20)
    plt.ylabel('Posts Percentage',fontsize=20)
    plt.title(title+str(Year),fontsize=30)
    plt.show()


# In[ ]:


trends([WebDev_Posts, DataBase_Posts, BigData_Posts, DataScience_Posts, ProgLang_Posts],
       ["Web Development",'DataBase Technologies','Big Data','Data Science','Programming Languages'])


# # **Generalized evaluator for technologies**
# <p style='text-align: justify;'>This function takes in a list of tags for which the user wants to find the past trends. These tags are queried within the function to get the past data for each tag and these are merged into one datframe with each column as one tag. This dataframe is then used to plot a line graph which shows the past trends in the technologies mentioned as the tags list.</p>
# <p style='text-align: justify;'>The function can also take optional parameters, namely labels and title. *'title'* is set to **'Trends in Technologies in 2019'** as default, while labels is equal to the list of tags if not mentioned explicitly.</p>

# In[ ]:


def PastTrends(dfall, labels = None, title="Past Trends", **kwargs):

    query1 = "select EXTRACT(year FROM creation_date) AS year, sum(id) as posts from `bigquery-public-data.stackoverflow.posts_questions` where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%"
    query3 ="%' group by year order by year"
    df = []
    
    if labels==None:
        labels = dfall
        
    l = len(dfall)
    for i in range(l):
        query2 = dfall[i]
        query = query1+query2+query3
        Posts = stackoverflow.query_to_pandas(query)
        Posts['posts']= Posts['posts']*100/PostsCount.posts
        pd.to_numeric(Posts['year'])
        df.append(Posts)
    
    trend = pd.merge(df[0], df[1], how='inner', on = 'year')
    trend = trend.set_index('year')
    if(l>2):
        for i in range(2,l):
            trend = pd.merge(trend, df[i], how='inner', on = 'year')
            trend = trend.set_index('year')
            
    trend.plot(kind='line')
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Posts %', fontsize=15)
    y_pos=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]
    plt.xticks(y_pos,fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(title)
    plt.legend(labels, loc=[1.0,0.5])
    plt.show()


# In[ ]:


PastTrends(["android","javascript","cassandra"])


# # **Generalized predictor for technologies**
# <p style='text-align: justify;'>This function takes in a list of tags for which the user wants to find the future trends. These tags are queried within the function to get the past data for each tag and a linear regression model is built for each of the tags and future predictions are made using the queried past data and a bar graph is created which shows the predicted future trends in 2019 for the given tags.</p>
# <p style='text-align: justify;'>The function can also take optional parameters, namely labels and title. *'title'* is set to **'Trends in Technologies in 2019'** as default, while labels is equal to the list of tags if not mentioned explicitly. The function can also take *'year'* as one of the arguments, which controls the year for which predictions are made. By default, *'year'* is set to 2019.</p>

# In[ ]:


def FutureTrends(dfall, Year = 2019, labels = None, title="Trends in Technologies in ", **kwargs):

    plt.figure(figsize=(20,10))
    
    query1 = "select EXTRACT(year FROM creation_date) AS year, sum(id) as posts from `bigquery-public-data.stackoverflow.posts_questions` where extract(year from creation_date) >=2009 and extract(year from creation_date) < 2019 and tags like '%"
    query3 ="%' group by year order by year"
    df = []
    l = len(dfall)
    
    if (labels==None):
        labels = dfall
        
    for i in range(l):
        query2 = dfall[i]
        query = query1+query2+query3
        Posts = stackoverflow.query_to_pandas(query)
        Posts['posts']= Posts['posts']*100/PostsCount.posts
        pd.to_numeric(Posts['year'])
        df.append(Posts)
        
    predict = []
    for d in df:
        year=d['year'].values.reshape(-1,1)
        posts=d['posts'].values.reshape(-1,1)
        reg=LinearRegression()
        X_train = year
        Y_train = posts
        X_test = [[Year]]
        reg.fit(X_train,Y_train)
        predictions = reg.predict(X_test)
        predict.append(predictions)
    #print(predict)
    
    trend = pd.DataFrame(columns = ['Technology','Posts %'])
    trend['Technology'] = labels
    trend['Posts %'] = predict
    
    x_pos = np.arange(len(trend['Technology']))
    plt.bar(x_pos,trend['Posts %'])
    plt.xticks(x_pos, trend['Technology'],fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Technologies',fontsize=20)
    plt.ylabel('Posts Percentage',fontsize=20)
    plt.title(title+str(Year),fontsize=30)
    plt.show()


# In[ ]:


FutureTrends(["spark","hive","python"])


# In[ ]:


FutureTrends(["jquery","javascript","html"],2020, ['JQuery','JavaScript','HTML'])

