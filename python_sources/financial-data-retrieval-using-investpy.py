#!/usr/bin/env python
# coding: utf-8

# # investpy - Financial Data Retrieval from Investing.com
# 
# [investpy](https://github.com/alvarobartt/investpy/tree/developer) is an Open Source Python package to retrieve financial data from Investing.com which is a popular finanacial/economic website used by traders, data analysts, etc.
# 
# Along this notebook some of the most used/relevant features that investpy offers will be explained so as to share it with all the Kagglers that visit this Notebook.
# 
# All the source code can be found at https://github.com/alvarobartt/investpy/tree/developer, and you can follow me at: https://github.com/alvarobartt

# ![investpy logo](https://raw.githubusercontent.com/alvarobartt/investpy/master/docs/investpy_logo.png)

# ## Installation
# 
# Currently, investpy is almost on 1.0 release, but the most updated and operative version is being developed under the developer branch.
# 
# So on, along this tutorial the developer version of investpy will be used, until the stable release comes out, which you can easily install using the following command:
# 
# ``$ pip install git+https://github.com/alvarobartt/investpy.git@developer``

# In[ ]:


get_ipython().system('pip install git+https://github.com/alvarobartt/investpy.git@developer')


# In[ ]:


import investpy


# ## Recent/Historical Data
# 
# As some of you may already know, the most used feature is the extraction of both recent and historical data from different financial products, since this data is usually used to analyze market trends, predict the closing price, etc.
# 
# investpy integrates some of the static data provided by Investing.com which is also stored as static among investpy files, this is made due to a listing error on Investing.com, since the provided listings sometimes contain format errors, missing data, wrong labeled data, etc.
# 
# This means that some financial products may not be found using any of the presented functions below, but later we will also explain how to retrieve all the Investing.com that does not seem to be available in investpy.

# ### Recent Data
# 
# To retrieve recent data you can simple use the following function structure `investpy.get_X_recent_data()`, where X can be any financial product type among all the ones available at Investing.com. So on, that means that X can be replaced by any of the following values: `stock, fund, etf, index, bond, currency_cross, crypto, commodity and certificate`. This function contains 2 mandatory parameters which are the financial product identifier and the country (is applicable).
# 
# The example provided below will be the most common one, which is Stocks. Also, note that if you have any doubt about the usage of all the different params of every function you can check the [investpy - API Referece](https://investpy.readthedocs.io/api.html) or use the command `help(investpy.function_name)`.

# In[ ]:


recent_data = investpy.get_stock_recent_data(stock='AAPL', country='United States')

recent_data.head()


# ### Historical Data
# 
# This function is similar to the recent data retrieval function mentioned above, but changing the function name template to `investpy.get_X_historical_data()` and including the parameters `from_date` and `to_date`, so as to specify the date range of the data to retrieve as long as Investing.com contains that data.
# 
# Note that the date is formatted as `dd/mm/YYYY`, so please take it into consideration; also note that the parameters date format does not affect to the resulting `pandas.DataFrame` which returns the Date column as a DateTime object, which is presented using the format `YYYY-mm-dd`.

# In[ ]:


historical_data = investpy.get_stock_historical_data(stock='AAPL', country='United States', from_date='01/01/2019', to_date='01/01/2020')

historical_data.head()


# ## Financial Product Information
# 
# Also there is a function to retrieve any financial product information as presented at Investing.com, which contains values such as the year change, ratio, market cap, etc. Note that some fields may differ among different financial product types, since this information is adapted to each financial product type.
# 
# So on, the function name template is: `investpy.get_X_information()` which also requires both the financial product identifier and the country (if applicable), and X can be replaced by the following financial product types: `stock, fund, etf, index, bond, crypto, commodity and certificate`.

# In[ ]:


stock_information = investpy.get_stock_information(stock='AAPL', country='United States', as_json=True)

stock_information


# ## Search Data
# 
# As already mentioned above, some Investing.com data may not be available using the standard data retrieval functions previously described, but indeed investpy contains the Investing.com Search Engine integrated, which means that any financial product found at Investing is available in investpy.
# 
# This function solves a lot of errors related to missing data, which are not common, but can sometimes happen if the desired financial product is not listed among the investpy static files.
# 
# The search function is `investpy.search_quotes()` which will be explained in detail below, since it may seem a little bit complex the first time you are using it.

# In[ ]:


search_results = investpy.search_quotes(text='Apple', products=['stocks'], countries=['United States'])


# So on, it returns a list of SearchObj class instances which contain all the retrieved entries for the introduced search query, so on, we will print all the search results inside a FOR loop. 

# In[ ]:


for search_result in search_results:
    print(search_result)


# Now we can just select the financial product we were looking for, or taking all the ones that we need to and proceed to retrieve both its OHLC data and its information, using the functions `retrieve_recent_data` & `retrieve_historical_data` and `retrieve_information`, respectively over the `SearchObj` class instance.
# 
# In the example provided below we will just pop out the first result of the `search_results` list as it is our desired financial product (the one we were looking for).

# In[ ]:


search_result = search_results.pop(0)
print(search_result)


# In[ ]:


recent_data = search_result.retrieve_recent_data()

recent_data.head()


# In[ ]:


historical_data = search_result.retrieve_historical_data(from_date='01/01/2019', to_date='01/01/2020')

historical_data.head()


# In[ ]:


stock_information = search_result.retrieve_information()

stock_information


# As you may have noticed, the additional parameters `products` and `countries` have been used, which are some filters to apply to the search query, note that also `n_results` can be specified so as to set a limit of returned products. More information about the Search Engine can be found at [investpy - GitHub Wiki](https://github.com/alvarobartt/investpy/wiki/investpy.search_quotes())

# ## Technical Indicators, Overview, Financial Summary and much more

# __Since a lot of functions are available in investpy, in the upcoming days those functions will also be explained in this Notebook, so make sure to upvote it to get notified from the newer versions that will come!__

# ### Remember to star the repository at GitHub if you found it useful and follow me so as to get notified of all the updates and projects I upload! Thank you!
