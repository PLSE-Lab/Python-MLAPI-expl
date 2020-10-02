#!/usr/bin/env python
# coding: utf-8

# # Analyze open data set of webstore Dream Market's "Cocaine" category
# 
# Determine what are the factors that influence price of cocaine, based on the dataset from  https://www.kaggle.com/everling/cocaine-listings
# The dataset is approximately 1,400 cleaned and standardized product listings from Dream Market's "Cocaine" category. It was collected with web-scraping and text extraction techniques in July 2017.
# 
# As bonus questions, we can also:
# - Compare Australian market with rest of the world 
# - Determine if there is volume discount and how significant it is
# 
# 
# 

# # Table of contents
# The main parts of this notebook are:
# 1. [Load data into DataFrames](#load)
# 1. [Check the data](#check)
# 1. [Data Analysis](#analysis)
#     1. [Data visualisation to identity possible hypothesis](#raw_plots)   
#     1. [Correlation matrix](#corr)
#     1. [Data distribution](#distribution)
#     1. [Summary and hypothesis](#hyp)
#     
# 1. [Data Transformation and features engineering](#eng)
# 1. [Model and model evaluation](#model)
# 1. [Model Result](#model_result)
# 1. [Output Reports based on the model data](#summary)
#     1. [Report 1 Gram of Q75 by Country of shipment](#rep_1)
#     1. [Report 2 Cost per gram of Q90 by Grams Pack in Australia](#rep_2)
#     1. [Report 3  Cost for 1 gram by Quality in AU](#rep_3)
#     1. [Repot 4. Factor plot](#rep_4)
#     
# 1. [Next steps](#next)
# 

# In[3]:


#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, metrics


import statsmodels.api as sm
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings; warnings.simplefilter('ignore')


# <a name="load"></a>
# ## Upload Dreamworld Data set

# In[5]:


from subprocess import check_output
# Any results you write to the current directory are saved as output.
filename = check_output(["ls", "../input"]).decode("utf8").strip()
df_raw = pd.read_csv("../input/" + filename, thousands=",")
df_raw.head()


# <a name="analysis"></a>
# ### Data Exploration  

# <a name="corr"></a>
# ### Correlation matrix
# Let's see how different variable correlated each other and see if we can make some initial conclusion out of the result

# In[44]:


#check correlation
df_corr = df_raw[['grams', 'quality', 'btc_price', 'cost_per_gram', 'cost_per_gram_pure', 'rating']].corr()

print (df_corr)

sns.heatmap(df_corr, cmap="Blues")


# Ther is an expected *Correlation* between:
# * cost per gram and cost per gram pure
# * btc price and grams
# 
# However what is interesting that no correlation between quality and cost per gram pure 
# 
# <a id="distribution"></a>

# Let's dig now in categorical variables and see how different shipment_from countries impact the price of cocaine

# In[10]:


#visual on categorical variable ship from
sns.factorplot(
    x='ships_from',
    y='cost_per_gram',
    data=df_raw, 
    kind='box', 
    size = 3,
     aspect=5

)


# *Result: The price is more or less the same across all countries except Australia, which has noticable difference in comarison to other countries. China has the least expecnsive, but it seems that number of offers is significantly lower*

# In[11]:


fg = sns.FacetGrid(df_raw[['quality', 'cost_per_gram', 'ships_from']], hue='ships_from',size=5, aspect=3)
fg.map(plt.scatter, 'quality', 'cost_per_gram').add_legend()


# *result: surprisingly, quality has no significant impact on the price of the product* 

# <a name="hyp"></a>
# ### Results and hypotesis

# Observations:
# - Quality variable, despite on continues nature, has very strong patterns and has very limited options to select from. Due to the variable is not normal distributed I can assume that we can pack it into classes and analyse as categorical variable. For the sake of simplicity I use all of available quality options.
# - Grams variable also has a very strong packing pattern. Most likely, use it as categorical variable will bring us better quality
# - Quality has no correlation with other variables, even with Price
# - Number of transactions in Europe is greater than in Australia, USA and significantly greater than in ASIA. My assumption is that the most trustful result should be about Europe. Reasonable assumptions may apply for other regions.
# 
# Hypothesis: 
# 
# The most influenced factors on cocaine price are the shipment location and transaction volume, based on the dataset of the dream market web shop.
# 

# In[ ]:





# <a id="eng"></a>
# ## Data Transformation and Feature engineering 

# In[14]:


#Plot data distributions

# Set up the matplotlib figure
f, axes = plt.subplots(2, 3, figsize=(10, 10), sharex=False, sharey=False)

sns.distplot(df_raw['cost_per_gram_pure'], ax=axes[0, 0])
sns.distplot(df_raw['cost_per_gram'], ax=axes[0, 1])
sns.distplot(df_raw['quality'], ax=axes[0, 2])
sns.distplot(df_raw['btc_price'], ax=axes[1,0]) 
sns.distplot(df_raw['grams'], ax=axes[1,1]) 
sns.distplot(df_raw['rating'], ax=axes[1,2]) 


# In[15]:


df_raw[['grams', 'quality', 'btc_price', 'cost_per_gram', 'cost_per_gram_pure', 'rating']].skew()  # Check skewness


# *Result: the data is quite skewed. will see how reasonable the model performs. We may need to do some extra work on it:*
# * dropping some the first peak in *cost per gram*
# * Check if we can drop some variable from the model
# 

# In[16]:


df_raw[['grams', 'quality', 'btc_price', 'cost_per_gram', 'cost_per_gram_pure', 'rating']].kurt()  # Check kurtosis


# *Result: all results has a significant positive kurtosis. Some further actions may required* 

# In[18]:


#Attemp to fix kurtosis and have better normal distribution by appling log function

df_raw['btc_price_log'] = df_raw['btc_price'].apply(np.log)


# In[19]:


f, axes = plt.subplots(1, 2, figsize=(10, 7), sharex=False, sharey=False)

sns.distplot(df_raw['btc_price'], ax=axes[0])
sns.distplot(df_raw['btc_price_log'], ax=axes[1])


# In[20]:


df_raw[['btc_price', 'btc_price_log']].skew() # Check kurtosis


# *result: Log function improves kurtosis quite significanly. btc_price_log will be used as outcome variable *

# ### Add dummy Variables

# In[ ]:





# In[21]:


print ("grams  unique categories")
print (df_raw['grams'].unique())


# In[22]:


#Add dummy categories for grams
df_raw['grams'].unique()
dummy_rank = pd.get_dummies(df_raw['grams'], prefix='grams', prefix_sep='_')
df_combo = df_raw.join(dummy_rank)


# In[23]:


print ("quality unique categories")
print (df_raw['quality'].unique())


# In[24]:


#Add dummy categories for quality
df_raw['quality'].unique()
dummy_rank = pd.get_dummies(df_raw['quality'], prefix='quality', prefix_sep='_')
df_combo = df_combo.join(dummy_rank)
df_combo.info()


# In[25]:


#convert boolean variables (with String elements in it) to numeric to fit the model

ships_to = df_combo.columns
ships_to_t = []
for q in ships_to:
    if 'ships_to_' in q:
        ships_to_t.append(q)
     #   print 1
    elif 'ships_from_' in q:
        ships_to_t.append(q)
ships_to_t.remove('ships_from_to')

for d in ships_to_t:
    df_combo[d] = df_combo[d].apply(lambda x: int(bool(x)))
    


# <a name="model"></a>
# ### Build the model

# In[26]:



#add intercept
df_combo['intercept'] = 1


#collect all names of columns' features in a list
X_columns = df_combo.columns
X_btc_price_combo_col = []
for q in X_columns:
    if 'grams_' in q:
        X_btc_price_combo_col.append(q)
      #  print q

for w in X_columns:
    if 'quality_' in w:
        X_btc_price_combo_col.append(w)
 #       print q

for q in X_columns:
    if 'ships_from_' in q:
        X_btc_price_combo_col.append(q)
    elif 'ships_to_' in q:
        X_btc_price_combo_col.append(q)
        
X_btc_price_combo_col.remove('ships_from_to')
X_btc_price_combo_col.append('escrow')    
X_btc_price_combo_col.append('intercept')
print (X_btc_price_combo_col)


# In[27]:


X_btc_price_combo = df_combo[X_btc_price_combo_col]
y_btc_price_combo = df_combo["btc_price_log"]


#no need in calssical as we use lasso to suppress all potentially multicollinearity features
#model = sm.OLS(y_btc_price_combo, X_btc_price_combo)
#results = model.fit()
#print(results.summary())
#results.resid


# In[28]:



#Use GS to identify the best alpha parameter 
from sklearn import grid_search


alphas = np.logspace(-10, 10, 21)
gs = grid_search.GridSearchCV(
    estimator= linear_model.Lasso(),
    param_grid={'alpha': alphas},
    scoring='mean_squared_error')

gs.fit(X_btc_price_combo, y_btc_price_combo)


# In[29]:


print (gs.best_score_ )


# In[30]:


print (gs.best_estimator_ )


# In[53]:


#Apply the identified alpha and use the model outcome in features analysis 
lm = linear_model.Lasso(alpha=0.0001,positive=True).fit(X_btc_price_combo, y_btc_price_combo)
#print (lm.coef_)
#print (lm.intercept_)
print ("~~~ Lasso ~~~")
print ('Lasso MSE: ', metrics.mean_squared_error(y_btc_price_combo, lm.predict(X_btc_price_combo)))
print ('Lasso R2:', lm.score(X_btc_price_combo, y_btc_price_combo))


# In[32]:


# some quality metrics. Plot true values versus residual
#rss = sum((lm.predict(X_btc_price_combo)-df_combo["btc_price_log"])**2)
#print rss
plt.scatter(lm.predict(X_btc_price_combo), df_combo["btc_price_log"] - lm.predict(X_btc_price_combo), color="g")
plt.xlabel("True Values")
plt.ylabel("Predicted Values resid")
#plt.show()


# <a name='model_result'></a>
# Result:  MSE:  0.285987248101 and  R2: 0.912521649006 show that the model is worth to consider. Many statistical significant variable factors are good for our goal to explore the price. 
# 
# Although, the residual distribution chart shows that the model has noticeable bias error. So some factors are not considered here or data sample isnt represent the full population.
# 

# ### Report 1 Gram of Q75 by Country of shipment 
# <a name="rep_1"></a>

# In[35]:


## rate of BTC to AUD in July 2017
CurrencyRate = 3050


# In[36]:


#create a reporting table based on prediction of 1 gram of quality 75% from different countries
Country = []
Cost = []  
df_predict = pd.DataFrame({'Predict' : 0
                           }, index = X_btc_price_combo_col).T

#df_predict['escrow'] = 1
df_predict['quality_75.0'] = 1
df_predict['grams_1.0'] = 1
df_predict['ships_to_GB'] = 1


for q in X_btc_price_combo_col:
    #go through the loop of ships_from_ prefix predictors 
    if 'ships_from_' in q:
        df_predict[q] = 1
        Country.append(q)
        Cost.append(np.exp(lm.predict(df_predict)) * CurrencyRate)
        df_predict[q] = 0
       # print q, np.exp(lm.predict(df_predict)) * CurrencyRate

df_plt_by_country = pd.DataFrame({ 'Country' : Country,
                       'Cost' : Cost
                         })
df_plt_by_country['Cost_num'] = df_plt_by_country['Cost'].astype(float)


# <a id="(#report_by_country"></a>

# In[37]:


#Report predicted Cost by Country
sns.factorplot(
    y='Country',
    x='Cost_num',
    data=df_plt_by_country.sort_values('Cost_num',ascending=False ), 
    kind='bar', 
    size = 5,
     aspect=1,
orient = 'h',
    color = 'b'
)


# *Result: Australia has the most expecncive cocaine. GB has the cheapest cocaine.*

# ### Report 2 Cost per gram of Q90 by Grams Pack in Australia
# <a name="rep_2"></a>

# In[38]:


Grams = []
Cost = []  


df_predict = pd.DataFrame({'Predict' : 0
                           }, index = X_btc_price_combo_col).T

CurrencyRate = 3050
df_check = pd.DataFrame({ 'btc' : float,
                       'Cost' : float
                         }, index = [1])
df_check['Cost'] = np.exp(lm.predict(df_predict)) * CurrencyRate
df_check['btc'] = np.exp(lm.predict(df_predict))
df_check['log'] = lm.predict(df_predict)
df_check['rate'] = CurrencyRate

for q in X_btc_price_combo_col:
    #go through the loop of ships_from_ prefix predictors 
    if 'grams_' in q:
        df_predict[q] = 1
        Grams.append(q)
        Cost.append(np.exp(lm.predict(df_predict)) * CurrencyRate)
        df_predict[q] = 0
       # print q, np.exp(lm.predict(df_predict)) * CurrencyRate

df_plt_by_grams = pd.DataFrame({ 'Grams' : Grams,
                       'Cost' : Cost
                         })
#Convert to numeric and calcualte cost per gram
df_plt_by_grams['Cost_num'] = df_plt_by_grams['Cost'].astype(float)
df_plt_by_grams[['Cost_num', 'Grams']]
df_plt_by_grams['Grams_num'] = df_plt_by_grams['Grams'].map(lambda x: str(x)[6:]).astype(float)
df_plt_by_grams['Cost_per_gram'] =df_plt_by_grams['Cost_num'] /  df_plt_by_grams['Grams_num'] 

#print df_plt_by_grams


# In[39]:


sns.factorplot(
    y='Grams',
    x='Cost_per_gram',
    data=df_plt_by_grams.sort_values('Grams_num',ascending=True ), 
    kind='bar', 
    size = 15,
     aspect=1,
orient = 'h'
    ,
    color = 'b'
)


# *Result: The relation isnt perfcet, but we can see clear trend that more volume we take, less cost per gram it has*
# *It is quite expecnive to buy packs up to 0.3g. The next level of discounts starts from packs of 10g

# ### Report 3  Cost for 1 gram by Quality in AU
# <a name="rep_3"></a>

# In[40]:


#create a reporting table based on prediction of 1 gram  by quality variables in Australia
Quality = []
Cost = []  
df_predict = pd.DataFrame({'Predict' : 0
                           }, index = X_btc_price_combo_col).T
#df_predict['escrow'] = 1
#df_predict['ships_to_AU'] = 1
#df_predict['ships_from_AU'] = 1
#df_predict['grams_1.0'] = 1

for q in X_btc_price_combo_col:
    #go through the loop of quality_ prefix predictors 
    if 'quality_' in q:
        df_predict[q] = 1
        Quality.append(q)
        Cost.append((lm.predict(df_predict)))
        df_predict[q] = 0
       # print q, np.exp(lm.predict(df_predict)) * CurrencyRate

    
df_plt_by_qa = pd.DataFrame({ 'quality' : Quality,
                       'Cost' : Cost
                         })

df_plt_by_qa['Cost_num'] = df_plt_by_qa['Cost'].astype(float)


# In[41]:


sns.factorplot(
    y='quality',
    x='Cost_num',
    data=df_plt_by_qa.sort_values('quality',ascending=False ), 
    kind='bar', 
    size = 5,
     aspect=1,
orient = 'h'
    ,
    color = 'b'
)


# *Result: as expected, there is no or weak relation between quality and price*

# ### Repot 4. Factor plot
# <a name="rep_4"></a>

# In[42]:


df_coef = pd.DataFrame({'Coef' : lm.coef_,
                         'Inercept' : lm.intercept_,
                        'Name': X_btc_price_combo_col
                           }, index = X_btc_price_combo_col)

sns.factorplot(
    y='Name',
    x='Coef',
    data=df_coef, 
    kind='bar', 
    size = 40,
     aspect=1,
#orient = 'v',
     legend_out=True,
    color = 'b'
 #   palette = 'RdBu'
)


# Overall Result: 
# - The most influence factor set are grams. 
# - Quite noticeable that some countries have a positive coefficient which increases the price. 
# - Australia has the most impactful coefficient. 
# - Quality, also has different corrections.
# - Shipment to has almost no impact on price. (except Australia)
# 
# Overall, I feel that the result is quite reasonable. The biggest advantage of the model in comparison with table aggregation is that we can predict and show the answers to which we have no direct intersections.
# 
# Waterfall chart to visualize the result
# ![Capture.PNG](attachment:Capture.PNG)
# 

# ### Next steps
# <a name="next"></a>
# 
# As a further improvement, the classification of quality categories can be done more properly.
# the exploration within one country by vendor and get from the short description the country of origin and if the product 'right from the brick' can be interesting direction.
# 
# I feel there will be a great check if I the model can be applied for another similar shop
# 
# History of the sales with customer ID (as well as Facebook link) and date reference can significantly improve the model.
# 

# In[ ]:




