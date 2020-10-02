#!/usr/bin/env python
# coding: utf-8

# # Kaggle's M5 forecasting Competition
# 
# Author: `Armando Miguel Trejo Marrufo`

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
import seaborn as sns 
import statsmodels.formula.api as smf
import statsmodels.api         as sm
from sklearn.utils import shuffle
from scipy.stats import yeojohnson, yeojohnson_normplot, probplot, boxcox
from statsmodels.stats.outliers_influence import OLSInfluence
warnings.filterwarnings("ignore")

# Display all columns
pd.set_option('display.max_columns', None)

# Use a ggplot style for graphics
plt.style.use('ggplot')


# ***Notes***
# 
# The results may vary because for Kaggle version I'm considering only a shuffle version of 3,000,000 observations for the data of prices to avoid running out of memory. For the results with the complete data see [link](https://github.com/TremaMiguel/KaggleCompetitions/blob/master/M5_Forecasting/notebooks/M5_Forecasting_CDA.ipynb).
# 
# 1. ```dt_complementary``` merges the original calendar and price data to know the ```date``` when the ```product_id``` of the ```store_id``` was saled at. Additionaly to know if they were particular events like ```SNAP``` purchases or major events, check the documentation.
# 
# 2. ```dt_sales_s```. In case you don't have enough RAM take initial n rows of the original dataframe and shuffle the data, because it is ordered.
# 
# 3. ```dt_sales_melt```. Melt the dataframe so that each sale by product can be seen as row.
# 
# 4. ```dt_work```. Merges ```dt_sales_melt``` with ```dt_complementary``` to know for each day of sale the price and relevant events associated to that day.

# In[ ]:


# Load data
files = ['/kaggle/input/m5-forecasting-accuracy/calendar.csv', 
         '/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv',
         '/kaggle/input/m5-forecasting-accuracy/sell_prices.csv']
data = [pd.read_csv(f) for f in files]
dt_calendar, dt_sales, dt_prices = data

# Merge calendar and prices
dt_prices = shuffle(dt_prices, n_samples = 3000000)
dt_complementary = dt_prices.merge(dt_calendar, how='left', on='wm_yr_wk')
del dt_prices
del dt_calendar

# Shuffle data (it is originally ordered) and take n rows (if you don't have enough RAM)
#dt_complementary = shuffle(dt_complementary, n_samples=10000000, random_state=0)


# ### 1. Preprocess Data

# ***A. Zeros in absolute and percentage terms***
# 
# If we plot the distribution of the `sales_day` variable we're going to notice that most of the observations are zero, in fact, even the median is zero. For this reason, we're going to analyse the number of zeros in `dt_sales`, because they could represent noise for forecasting.

# In[ ]:


# Count the number of zeros in data 
dt_sales['num_zeros'] = (dt_sales == 0).sum(axis=1)
_ = plt.hist(dt_sales.num_zeros)
plt.xlabel('Number of zeros')
plt.ylabel('Frequency')
plt.title("Distribution of number of zeros by observation")


# We see a staggered distribution, that is, as the number of zeros increases so the frequency. But remember that we're not yet considered when does each item started to being sell. For example, a certain product could have zero sell from `d_1` up to `d_1000` because it started to been sold on `d_1001`. Thus, it is unnecessary to consider the information prior to this day for this item. 
# 
# Therefore, we're going to obtain the first day of sale form `dt_complementary` and then measure the percentage of zeros by the total number of sale day = `d_1913` - `first_day_of_sale`.

# In[ ]:


# Transform date variable to datetime
dt_complementary.date = pd.to_datetime(dt_complementary.date)

# Append the first day of sales to each item 
first_date = dt_complementary.groupby(['store_id','item_id']).agg({'date':'min'}).reset_index().rename(columns={'date':'date_first_sale'})
dt_sales = dt_sales.merge(first_date, how='left', on=['store_id','item_id'])

# Delete data to save RAM
del first_date

# Difference in days between date_first_sales and d_1
dt_sales['since_d_1'] = dt_sales.date_first_sale - pd.to_datetime('2011-01-29')
dt_sales['since_d_1'] = dt_sales['since_d_1'].apply(lambda x: x.days)


# In[ ]:


# Percentage of zeros since first day of sale
dt_sales['%_zeros_of_total'] = round(((dt_sales.num_zeros - dt_sales.since_d_1) / (1913 - dt_sales.since_d_1)) * 100, 2)
_ = sns.distplot(dt_sales['%_zeros_of_total'])
#_ = plt.hist(dt_sales['%_zeros_of_total'], bins=10)
plt.xlabel('% of zeros')
plt.ylabel('Frequency')
plt.title("Distribution of % of zeros by item")


# Based on the graph, we're considering the following bins:
# 
# 1. `perc_bin_1` --> 0 to 20% zeros.
# 2. `perc_bin_2` --> 21 to 40% zeros.
# 3. `perc_bin_3` --> 41 to 60% zeros.
# 4. `perc_bin_4` --> 61 to 80% zeros.
# 5. `perc_bin_5` --> 81 to 100% zeros.

# In[ ]:


def perc_bin(num:int):
    if num <= 20:
        output = 'perc_bin_1'
    elif num <= 40:
        output = 'perc_bin_2'
    elif num <= 60:
        output = 'perc_bin_3'
    elif num <= 80:
        output = 'perc_bin_4'
    else:
        output = 'perc_bin_5'
    return output

dt_sales['perc_zeros_bin'] = dt_sales['%_zeros_of_total'].apply(lambda x: perc_bin(x))


# ***B. Melt data and dataframe to work***

# In[ ]:


# Items with less percentage of zeros
dt_sales_bin1 = dt_sales[dt_sales.perc_zeros_bin == 'perc_bin_1'].drop(columns=['id','num_zeros','date_first_sale','since_d_1','%_zeros_of_total','perc_zeros_bin'])
del dt_sales

# Melt sales data
indicators = [f'd_{i}' for i in range(1,1914)]

dt_sales_bin1_melt = pd.melt(dt_sales_bin1, 
                             id_vars = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                             value_vars = indicators, 
                             var_name = 'day_key', 
                             value_name = 'sales_day')
del dt_sales_bin1

# Extract the number of day from the day_key variable
dt_sales_bin1_melt['day'] = dt_sales_bin1_melt['day_key'].apply(lambda x: x[2:]).astype(int)


# In[ ]:


# Data to work with
columns = ['store_id','item_id','sell_price','date','year','d','event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']
dt_work = dt_sales_bin1_melt.merge(dt_complementary[columns], how = 'inner', left_on=['item_id','store_id','day_key'], right_on=['item_id','store_id','d'])
del dt_complementary
print(dt_work.shape)


# ### 2. Data Imputation

# In[ ]:


# If there are null values, print the unique values of the column 
for k,v in dict(dt_work.isnull().sum()).items():
    if v > 0:
        print(f"The unique values for the column {k} are:", dt_work[k].unique(), "\n")


# We notice that we got missing values for the events, and that this is due to the fact that in that particular day was not tagged as an special event day. Thus, we're simply considering this days as `Normal` and the event type associated with them as `Non-Special`.

# In[ ]:


dt_work['event_name_1'] = dt_work['event_name_1'].fillna('Normal')
dt_work['event_name_2'] = dt_work['event_name_2'].fillna('Normal')
dt_work['event_type_1'] = dt_work['event_type_1'].fillna('Non-Special')
dt_work['event_type_2'] = dt_work['event_type_2'].fillna('Non-Special')


# ### 3. Correlation 

# In[ ]:


# Taken from https://datascience.stackexchange.com/questions/10459/calculation-and-visualization-of-correlation-matrix-with-pandas
corr = dt_work.corr()

cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '100px', 'font-size': '10pt'})    .set_caption("Correlation between variables")    .set_precision(2)    .set_table_styles(magnify())


# Assessing **Multicollinearity** is key, because knowing that one variable could be expressed as a linear combination of other(s), let us avoid the sensity of the coefficient of the model to this phenomenon. 
# 
# It is interesting to notice that there is nearly a perfect correlation between `day` and `year`. This is due to the fact that as year increases the day increases, for example, the first 365 observations of the day are for the first year, the next 365 for the second year and so.

# ### 4. `sales_day` distribution

# In[ ]:


print(dt_work.sales_day.describe(),
      f"The 1th percentile is {dt_work.sales_day.quantile(.01)}", "\n",
      f"The 5th percentile is {dt_work.sales_day.quantile(.05)}", "\n",
      f"The 10th percentile is {dt_work.sales_day.quantile(.1)}", "\n",
      f"The 15th percentile is {dt_work.sales_day.quantile(.15)}", "\n",
      f"The 20th percentile is {dt_work.sales_day.quantile(.15)}", "\n",
      f"The 90th percentile is {dt_work.sales_day.quantile(.90)}", "\n",
      f"The 98th percentile is {dt_work.sales_day.quantile(.98)}", "\n",
      f"The 99th percentile is {dt_work.sales_day.quantile(.99)}", "\n",
      f"The 99th percentile is {dt_work.sales_day.quantile(.995)}", "\n",
      f"The 99th percentile is {dt_work.sales_day.quantile(.999)}", "\n")


# In[ ]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

# Yeo-Johson Normality Plot 
lmbd_yj = yeojohnson_normplot(dt_work.sales_day, -10, 10, plot=ax)
sales_transformed, maxlmbd = yeojohnson(dt_work.sales_day)

ax.axvline(maxlmbd, color='r')
plt.show()


# In[ ]:


# QQ-plot of sales transformation 
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
qq_plot = probplot(sales_transformed, dist="norm", plot=ax)
ax.set_title("QQ-plot for normal distribution")
plt.show()


# In[ ]:


plt.figure(figsize=(12, 6))

# Left side figure
plt.subplot(1,2,1)
_ = sns.distplot(dt_work.sales_day)
plt.title("Original Distribution")

# Right side figure
plt.subplot(1,2,2)
_ = sns.distplot(sales_transformed, rug=True)
plt.title("With Yeo-Johnson Transformation")
plt.tight_layout(pad=1)
plt.show()


# In[ ]:


dt_work['sales_day_yj'] = sales_transformed

# Filter by columns of interest
cols_to_drop = ['item_id', 'day_key', 'day', 'date', 'd']
dt_reg = dt_work.drop(columns=cols_to_drop)

# Covert to category type 
for k, v in dict(dt_reg.dtypes).items(): 
    if v == 'object':
        dt_reg[k] = dt_reg[k].astype('category')
 
# Dummy variables    
dt_reg = pd.get_dummies(dt_reg)


# ### 5. Multiple Linear Regression

# #### A. All variables model

# We're not surprise to see that our model achieves to explain only 6% of the variability in the sale. Notwithstanding, we notice the following
# 
# * All Coefficients are statistically significant. The `p-value` is close to zero for all the coefficient associated to each variable. 
# 
# * There is higher sale in `HOBBIES_1` compared to the base category `FOODS_1`. However, when considering categories, the `FOODS` get more sales. 
# 
# * Across stores the sales appears to be the same. In fact, the coefficient associated to each store is very close to two. 
# 
# * There are some events that leverage positively the sale like `NBA finals` or the `SuperBowl`. By contrast, it is surprising to see that `Christmas day` is associated with a reduction in sale, maybe these days the stores are not at all open or not the whole day. 
# 
# * `Cultural_events` tend to foster sales. 
# 
# * It appears that sale decrease with the years and with price increases. This confirms the negative correlation that we saw before, but we should be cautious about affirming this.

# In[ ]:


# Model with Yeo-Johnson transformation
formula_yj="sales_day_yj ~ "

for col in dt_reg.columns[8:]:
    formula_yj+='Q("'+col+'")+'
    
formula_yj = formula_yj + 'sell_price + year'

model_yj = smf.ols(formula = formula_yj, data = dt_reg).fit()
model_yj.summary()


# In[ ]:


# Model with Yeo-Johnson transformation
formula="sales_day ~ "

for col in dt_reg.columns[8:]:
    formula+='Q("'+col+'")+'
    
formula = formula + 'sell_price + year'

model_orig = smf.ols(formula = formula, data = dt_reg).fit()
model_orig.summary()


# ### 6. Model Diagnosis 

# #### A.Evaluate Homokesdasticity (Residuals vs Fitted values)
# 
# The zero values are causing distortions in the regression. However, we see a **handfan** form, that is, as the value to fit increases so the residual, thus `Heterokedasticity`. 

# In[ ]:


# Residuals vs fitted values 
model_fitted = model_yj.fittedvalues
model_residuals = model_yj.resid
fig = plt.figure(figsize = (8, 6))
sns.scatterplot(model_fitted, 
                model_residuals,
                alpha=0.5
                  )
plt.title('Residuals vs Fitted')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')


# #### B. Leverage Points
# 
# The `Cook's distance plot` and the `leverage vs residuals` plot are informative. We see that there are some points with high Cook's distance (left plot) that appear frequently (apparently every 7500 units). Furthermore, in the right plot that there's a clear distinction between normal points and leverage points. When building the forecasting model, we should be cautious about them because they will **highly leverage the model** that we choose.

# In[ ]:


# Cook's distance values
outlierInfluence = OLSInfluence(model_yj)
(c, p) = outlierInfluence.cooks_distance

# Leverage and normalized residuals
model_leverage = model_yj.get_influence().hat_matrix_diag
model_norm_residuals = model_yj.get_influence().resid_studentized_internal
model_cooks = model_yj.get_influence().cooks_distance[0]


# In[ ]:


plt.figure(figsize=(12, 6))

# Cook's distance plot
plt.subplot(1,2,1)
plt.stem(np.arange(20000), c[:20000], markerfmt=",")
plt.title("Cook's distance plot for the residuals",fontsize=16)
plt.grid(False)


# Scatterplot of leverage vs normalized residuals
plt.subplot(1,2,2)
plt.scatter(model_leverage[:200000], 
            model_norm_residuals[:200000], alpha=0.5)
sns.regplot(model_leverage[:200000], 
            model_norm_residuals[:200000],
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
plt.xlim(-0.0005, 0.0010)
plt.title('Residuals vs Leverage')
plt.xlabel('Leverage')
plt.ylabel('Standardized Residuals')

plt.tight_layout(1.0)
plt.show()


# #### C. Variance Inflation Factor
# 
# With the `VIF` factor we try to understand how much the variance is going to increase because of collinearity. It is recommended is that if VIF is greater than 5, then the explanatory variable given is highly collinear with the other explanatory variables, and the parameter estimates will have large standard errors because of this. In our case (we're not considering dummy variables), the values for the choosen variables are near 1. 

# In[ ]:


# Taken from https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python/54857466

def variance_inflation_factors(exog_df):
    '''
    Parameters
    ----------
    exog_df : dataframe, (nobs, k_vars)
        design matrix with all explanatory variables, as for example used in
        regression. One recommendation is that if VIF is greater than 5, then 
        the explanatory variable given by exog_idx is highly collinear with the 
        other explanatory variables, and the parameter estimates will have large 
        standard errors because of this.

    Returns
    -------
    vif : Series
        variance inflation factors
    '''
    exog_df = sm.add_constant(exog_df)
    vifs = pd.Series([1 / (1. - sm.OLS(exog_df[col].values, 
                                       exog_df.loc[:, exog_df.columns != col].values) \
                                   .fit() \
                                   .rsquared
                           ) 
                           for col in exog_df],
        index=exog_df.columns,
        name='VIF'
    )
    return vifs

cols = ['sell_price', 'year', 'snap_CA', 'snap_TX', 'snap_WI', 'sales_day_yj']
variance_inflation_factors(dt_reg[cols])


# ### 7. Interaction Effects
# 
# **A. Is the sale different on SNAP days across the years?**
# 
# We notice a slight negative slope for the three states, in other words, it seems to be a decrease in sales in the recent years. In fact, we see that in the first three year 2011, 2012 and 2013 the maximum values are greater than the other years 2014, 2015 and 2016.
# 
# However, by visual comparison is not at all clear if snap sales leverage sales. For this reason we run an `OLS estimation` and evaluate the coefficients and p-values. 
# 
# 1. **California**. The coefficients are statistically significiant. We read the output as follows
#     * The sale increases by 12.2571% among SNAP days, but, 
#     
#     * for each year of increment, ***the sale should decrease by 0.0315 percent among not SNAP days and decrease by 0.0376 among SNAP days***.
#     
#     
# 2. **Texas**. The interpretation is similar as California, but ***SNAP days have a greater impact on sales by 17.0033%***.
# 
# 
# 3. **Wisconsin**. In this state, the situtation is totally the opposite. The p-values are greater than .05, thus, ***there is no difference in sales between SNAP days and not SNAP days***. Furthermore, even though the sale decreases by 0.0235% each year, there is no effect of SNAP days on it.   
#     

# In[ ]:


iterables = [('state_id_CA','snap_CA'), ('state_id_TX', 'snap_TX'), ('state_id_WI', 'snap_WI')]


for i in iterables:
    state, snap = i
    sns.lmplot(x='year',
               y='sales_day_yj', 
               data=dt_reg[dt_reg[state]==1], 
               hue=snap,
               col=snap,
               height=8, 
               scatter_kws={"s": 10},
               x_jitter=.25,
               y_jitter=.05
              )
    
plt.tight_layout(1)
plt.show()


# In[ ]:


for i in iterables:
    state, snap = i
    formula = (f'sales_day_yj ~ year*C({snap})')
    model = smf.ols(formula=formula, data=dt_reg[dt_reg[state]==1]).fit()
    print(f"Model for {state}",
          "\n",
          "-----------------------------------------------------------",
          "\n",
          model.summary(),
          "\n",
          "-----------------------------------------------------------")


# **B. Are prices higher on SNAP days?**
# 
# We would like to answer the prior question and evaluate this behavior across the years. The conclusion is that there is very little statistical evidence of an interaction between `snap` and `year` variables to explain prices. In other words, the general view is that ***the prices are not higher on SNAP days***. Now, let's evaluate this across the different departments. 

# In[ ]:


dt_reg['log_sell_price'] = np.log(dt_reg.sell_price + 1)

for i in iterables:
    state, snap = i
    sns.lmplot(x='year',
               y='log_sell_price', 
               data=dt_reg[dt_reg[state]==1], 
               hue=snap,
               col=snap,
               height=8, 
               scatter_kws={"s": 10},
               x_jitter=.15,
               y_jitter=.05)
    
plt.tight_layout(1)
plt.show()


# In[ ]:


for i in iterables:
    state, snap = i
    formula = (f'log_sell_price ~ year*C({snap})')
    model = smf.ols(formula=formula, data=dt_reg[dt_reg[state]==1]).fit()
    print(f"Model for {state}",
          "\n",
          "-----------------------------------------------------------",
          "\n",
          model.summary(),
          "\n",
          "-----------------------------------------------------------")


# **C. What about the `sell_price` on SNAP days across different departments?**
# 
# We conclude that there is no statistical significance difference for the prices across different departments on SNAP days.

# In[ ]:


for i in iterables:
    state, snap = i
    formula = (f'log_sell_price ~ dept_id_FOODS_1*C({snap}) + dept_id_FOODS_2*C({snap}) + dept_id_FOODS_3*C({snap}) + dept_id_HOBBIES_1*C({snap}) + dept_id_HOUSEHOLD_1*C({snap}) + dept_id_HOUSEHOLD_2*C({snap})')
    model = smf.ols(formula=formula, data=dt_reg[dt_reg[state]==1]).fit()
    print(f"Model for {state}",
          "\n",
          "-----------------------------------------------------------",
          "\n",
          model.summary(),
          "\n",
          "-----------------------------------------------------------")


# In[ ]:


iterables = [('CA','snap_CA'), ('TX', 'snap_TX'), ('WI', 'snap_WI')]

for i in iterables:
    state, snap = i
    formula = (f'np.log(sell_price+1) ~ C(dept_id)*C({snap})')
    model = smf.ols(formula=formula, data=dt_work[dt_work.state_id==state]).fit()
    print(f"Model for {state}",
          "\n",
          "-----------------------------------------------------------",
          "\n",
          model.summary(),
          "\n",
          "-----------------------------------------------------------")


# ## Conclusions

# The purpose of the Confirmatory Data Analysis was to statistically proof the possible relationships between the variables. We've learned the following:
# 
# 1. The presence of zeros among items vary. This is highly important because we would going to apply different techniques and models to these items, not necessarily solely intermetting models.
# 
# 
# 2. Sales does not follow a normal distribution. For the purposes of fitting a multiple regression we applied a Yeo-Johnson transformation so that the data approximates a normal distribution. However, we saw that zero values present in data makes this difficult. We would need other techniques or feature engineer to deal with this.
# 
# 
# 3. The dependent variables do not inflate the variance. In other words, we can incorporate variables such as prices and snap days to predict the sale. 
# 
# 
# 4. It is possible to divide leverage points from normal. This is highly valuable when forecasting, because leverage points have a high influence in the coefficients of the model. Thus, we could develop an strategy to discard or incorporate these points. 
# 
# 
# 5. Behavior on SNAP days. We saw that SNAP prices increase the sales and that there are not changes in prices on SNAP days. 
