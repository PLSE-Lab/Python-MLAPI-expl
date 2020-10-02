#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Function used in this notebook
def count_unique_values(df) :
    var = list(df.columns)
    print('Count unique values :')
    
    for i in var :
        count = len(df[i].unique())
        print(i,':',count)

def check_missing_values(df) :
    n = len(df)
    var = list(df.columns)
    missing_var = []
    missing_count = []
    print('Variable with missing values :')
    
    for i in var :
        count = np.sum(df[i].isna())
        count_percentage = round(count*100/n, 2)
        if count > 0 :
            print(i,':',count,'//',count_percentage,'%')
            missing_var.append(i)
            missing_count.append(count_percentage)
    
    return missing_var, missing_count

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
 
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.genmod.GLM(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))
                                ,family=sm.genmod.families.Gamma(link=sm.genmod.families.links.log)).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.genmod.GLM(y, sm.add_constant(pd.DataFrame(X[included]))
                            ,family=sm.genmod.families.Gamma(link=sm.genmod.families.links.log)).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def dataset_ready(x_train, y_train) :
    # Make dummy variable for categorical variable
    X = pd.get_dummies(x_train)

    # Make Intercept
    X['Intercept'] = [1]*len(X)

    # Make interaction between 'gdp_per_capita' and 'population'
    X['gdp_pop'] = np.log(X['gdp_per_capita']*X['population'])

    # Scale continuous variable with log function
    cont_var = ['gdp_per_capita','population']
    for i in cont_var :
        X[i] = np.log(X[i])

    # Make interaction between 'continent' and 'gdp'
    col = pd.Series(X.columns)
    var1 = list(X.filter(like='continent').columns)
    for i in var1 :
        string = i+'_gdp'
        X[string] = X[i]*X['gdp_per_capita']   

    # Make interaction between 'continent' and 'population'
    for i in var1 :
        string = i+'_population'
        X[string] = X[i]*X['population']  

    # Target variable
    Y = y_train
    
    return X,Y

# I use stepwise algorith from this link https://datascience.stackexchange.com/questions/24405/how-to-do-stepwise-regression-using-sklearn


# In[ ]:


# Load and configure notebook settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pylab import rcParams
# For every plotting cell use this
rcParams['figure.figsize'] = [10,5]

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

sns.set()
sns.set_style('whitegrid')


# # <font color='#00205b'>Suicide Rates Overview 1985 to 2016</font>
# This compiled dataset pulled from four other datasets linked by time and place, and was built to find signals correlated to increased suicide rates among different cohorts globally, across the socio-economic spectrum.
# 
# __Objective__
# <br> Understand the main reason why people committed suicide country-scale 
# 
# __Variables__
# - (c) __country__
# - (n) __year__ : from 1985 - 2016
# - (c) __sex__  : male or female
# - (c) __age__ : grouped age
# - (n) __suicides_no__ : suicides count
# - (n) __population__ : 
# - (n) __suicides/100k pop__ : suicides count rescale
# - (c) __country-year__ : country with year combine
# - (n) __HDI for year__ : Human Development Index, higher better
# - (n) __gdp_for_year__ : Gross Domestic Product
# - (n) __gdp_per_capita__ : Gross Domestic Product rescale
# - (c) __generation__
# 
# __References__
# <br>United Nations Development Program. (2018). Human development index (HDI). Retrieved from http://hdr.undp.org/en/indicators/137506
# 
# World Bank. (2018). World development indicators: GDP (current US$) by country:1985 to 2016. Retrieved from http://databank.worldbank.org/data/source/world-development-indicators#
# 
# [Szamil]. (2017). Suicide in the Twenty-First Century [dataset]. Retrieved from https://www.kaggle.com/szamil/suicide-in-the-twenty-first-century/notebook
# 
# World Health Organization. (2018). Suicide prevention. Retrieved from http://www.who.int/mental_health/suicide-prevention/en/

# In[ ]:


# Load dataset
df_train = pd.read_csv('../input/master.csv')


# In[ ]:


# Overview of the dataset
df_train.head()


# In[ ]:


# Change some variable name
df_train.rename(columns={' gdp_for_year ($) ':'gdp_for_year', 'gdp_per_capita ($)':'gdp_per_capita'}, inplace=True)
df_train_v2 = df_train.copy()


# In[ ]:


# Summary of the dataset
df_train.describe()


# In[ ]:


# Overview data types of the dataset
print('Data types of the dataset :')
print(df_train.dtypes)


# Variable __'gdp_for_year'__ should have numeric data type, but in here this variable have a object data type. Lets change it

# In[ ]:


# Change 'gdp_for_year' data type
change_var = []
for i in df_train['gdp_for_year'] :
    split = i.split(',')
    val = ''
    for j in split :
        val = val + j
    change_var.append(int(val))
    
df_train['gdp_for_year'] = change_var


# In[ ]:


# See the correlation between the variables
rcParams['figure.figsize'] = [10,5]
sns.heatmap(df_train.corr(), annot=True, linewidths=0.2, cmap='coolwarm' )
plt.title('Correlation heatmap of the dataset', size=15, fontweight='bold') ;
plt.xticks(rotation=45)


# We can see that __'HDI for year'__ and __'gdp_per_capita($)'__ have high correlation and variable __'HDI for year'__ have a NaN value
# <br> __<font color='7f181b'>Conclusion</font>__ : maybe remove variable __'HDI for year'__
# 
# Beside that we see that variable __'gdp_for_year'__ and __'population'__ also have high correlation, but oddly not with __'gdp_per_capita'__. We the two gdp variable has to be correlated and there must be some error in recording the data
# <br> __<font color='7f181b'>Conclusion</font>__ : maybe remove variable __'gdp_for_year'__ 

# # <font color='#00205b'>Data Preprocessing</font>

# In[ ]:


# Check for missing values in the dataset
missing_var, missing_count = check_missing_values(df_train)


# Variable __'HDI for year'__ have too many missing values and we know that this variable have high correlation with variable __'gdp_per_capita'__, so we should remove this variable

# In[ ]:


# Remove unwanted variable
df_train_v2.drop(columns=['country-year','HDI for year','gdp_for_year'], inplace=True)


# In[ ]:


# Check how many unique values in categorical variable
category_var = ['country','year','sex','age','generation']
count_unique_values(df_train[category_var])


# We can see that variable __'country'__ have too many unique values, this will become a problem when making the model (curse of dimensional)
# <br> __<font color='7f181b'>Conclusion</font>__ : Group the category into new variable __'continent'__

# In[ ]:


# Check the distribution of target variable 'suicides_no'
rcParams['figure.figsize'] = [15,5]
gs = gridspec.GridSpec(1,2)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])

# Plot 1 - Distribution of the target variable
sns.distplot(df_train['suicides_no'], color='#7f181b', kde=True, hist=False, ax=ax1) ;
ax1.set_title('Distribution of the suicides_no', size=15, fontweight='bold') ;

# Plot 2 - Distribution of the log(target variable)
sns.distplot(np.log(df_train[df_train['suicides_no']>0]['suicides_no']), color='#7f181b', kde=True, hist=True, ax=ax2) ;
ax2.set_title('Distribution of the log of population', size=15, fontweight='bold') ;


# We can see that the target variable have a gamma / inverse gaussian distribution
# <br> __<font color='7f181b'>Conclusion</font>__ use GLM with gamma / inverse gaussian distribution

# In[ ]:


# Make new variable 'continent' that represent continent of each country
# Based on wikipedia.com
country = df_train_v2['country'].unique()
new_val = ['Europe','Central America','South America','Asia','Central America'
          ,'Australia','Europe','Asia','Central America','Asia'
          ,'Central America','Europe','Europe','Central America'
          ,'Europe','South America','Europe','Africa'
          ,'North America','South America','South America','Central America','Europe','Central America'
          ,'Asia','Europe','Europe','Central America','South America'
          ,'Central America','Europe','Oceania','Europe','Europe','Asia'
          ,'Europe','Europe','Central America','Central America','South America','Europe'
          ,'Europe','Europe','Asia','Europe','Central America','Asia'
          ,'Asia','Oceania','Asia','Asia','Europe'
          ,'Europe','Europe','Asia','Asia','Europe'
          ,'Africa','North America','Asia','Europe','Europe'
          ,'Oceania','Central America','Europe','Asia','Central America','South America'
          ,'Asia','Europe','Europe','Central America','Asia'
          ,'Asia','Europe','Europe'
          ,'Central America','Central America'
          ,'Central America','Europe','Europe'
          ,'Africa','Asia','Europe','Europe','Africa'
          ,'Europe','Asia','South America','Europe','Europe'
          ,'Asia','Central America','Asia','Asia'
          ,'Europe','Asia','Europe'
          ,'North America','South America','Asia']
new_var = []

for i in range(len(country)) :
    n = len(df_train[df_train['country']==country[i]])
    for j in range(n) :
        new_var.append(new_val[i])
        
df_train_v2['continent'] = new_var


# In[ ]:


# Top 10 country with highest suicide median
# We use median because the distribution is skewed to the right
df_check = df_train_v2.groupby(by=['country','continent']).median()[['suicides_no','population','gdp_per_capita']].sort_values('suicides_no',ascending=False).reset_index()
cat = list(df_check.head(10)['country'])
print('Top 10 country with highest suicide median ')
print(df_check.head(10))


# We can see that the top 3 country have a significant difference in suicides count with the the other country. Beside that we can see that generally the more population and gdp, the more suicides count a country have. But theres some country like Russia and Brazil that have high population but low gdp.  

# In[ ]:


# Top 10 country number of suicides growth year by year
# We use median because the distribution is skewed to the right
df_check = df_train.groupby(by=['country','year']).median()['suicides_no'].reset_index()
rcParams['figure.figsize'] = [10,6]
gs = gridspec.GridSpec(2,1)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])

# Plot 1 - Line plot for top 3 country
for i in cat[:3] :
    sns.lineplot(data=df_check[df_check['country']==i], x='year', y='suicides_no', ax=ax1) ;
    
ax1.legend(cat[:3], loc=7,  bbox_to_anchor=(1.3, 0.5)) ;
ax1.set_title('Number of suicide growth of top 3 country', size=15, fontweight='bold') ;

# Plot 2 - Line plot for reminding country
for i in cat[3:] :
    sns.lineplot(data=df_check[df_check['country']==i], x='year', y='suicides_no', ax=ax2) ;
    
ax2.legend(cat[3:], loc=7,  bbox_to_anchor=(1.3, 0.5)) ;
ax2.set_xlabel('Number of suicide growth of reminding country', size=15, fontweight='bold') ;


# We can see that theres a country with a positive growth and negative growth, lets see if we group it by continent

# In[ ]:


# Count how many country in each continent recorded in the dataset
continent = list(df_train_v2['continent'].unique())
new_val = pd.Series(new_val)
count = []

print('Count contry in each continent recorded in the dataset :')
for i in continent :
    n = len(new_val[new_val==i])
    count.append(n)
    print(i,':',n)
 
#  Plot
df = pd.DataFrame({'continent':continent, 'count':count})
df.sort_values(by='count', ascending=False, inplace=True)
sns.catplot(data=df, x='continent', y='count') ;
plt.xticks(rotation=45)
plt.title('How many country in each continent', size=15, fontweight='bold') ;


# We can see from here that the dataset is mainly consist of country in Europe, Asia, and Central America. The other continent only consist of <10 countries

# In[ ]:


# Number of suicides growth in each continent
df_check = df_train_v2.groupby(by=['continent','year']).median()['suicides_no'].reset_index()
rcParams['figure.figsize'] = [10,6]
gs = gridspec.GridSpec(2,1)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])

# Plot 1 - Line plot for North America
sns.lineplot(data=df_check[df_check['continent']=='North America'], x='year', y='suicides_no', ax=ax1 ) ;
ax1.legend(['North America'], loc=7,  bbox_to_anchor=(1.3, 0.5)) ;
ax1.set_title('Number of suicide growth of North America', size=15, fontweight='bold') ;

# Plot 2 - Line plot for reminding continent
continent = pd.Series(continent)
for i in continent[continent!='North America'] :
    sns.lineplot(data=df_check[df_check['continent']==i], x='year', y='suicides_no', ax=ax2) ;
    
ax2.legend(continent[continent!='North America'], loc=7,  bbox_to_anchor=(1.3, 0.5)) ;
ax2.set_xlabel('Number of suicide growth of reminding continent', size=15, fontweight='bold') ;


# We can see that North America have a significant difference of number of suicides with the other continent. But we suspect thats because in this dataset, North America consist of only 3 countries. Which is why the aggregated number of suicides become higher than other continent. We suspect Australia also have the same situation (Australia consist of only Australia).
# 
# From before we get that the 6 out of 10 top country with highest suicide count come from Europe. But in this plot wee see that the aggregated number of Europe is significantly lower than aggregated number from each all 6 country. With this we suspect that there must be some interaction between variable __'continent'__ with other variable that can explain the variability in target variable much better

# In[ ]:


# Check the effect of gender and sex
rcParams['figure.figsize'] = [15,5]
gs = gridspec.GridSpec(1,2)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])

# Plot 1 - based on
sns.barplot(data=df_train_v2, x='sex', y='suicides_no', hue='age', ax=ax1
           ,hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']) ;
ax1.set_title('Disrtibution of suicide count by sex', size=15, fontweight='bold') ;

# Plot 2
sns.barplot(data=df_train_v2, x='sex', y='suicides/100k pop', hue='age', ax=ax2
           ,hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']) ;
ax2.set_title('Disrtibution of suicide count (rescale) by sex ', size=15, fontweight='bold') ;


# From here we can conclude that male more likely to commit suicide than female and people who are in the middle-age have higher chance to commit suicide than any other age category. So this two variable are likely to have significant effect to the model

# In[ ]:


# Check the effect of variable 'generation'
rcParams['figure.figsize'] = [16,5]
gs = gridspec.GridSpec(1,3, width_ratios=[2,8,6])
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[0,2])

# Plot 1 - North America perspective
sns.barplot(data=df_train_v2[df_train_v2['continent']=='North America'], x='continent', y='suicides_no', ax=ax1
           ,hue_order=['G.I. Generation','Silent','Boomers','Generation X','Milennials','Generation Z'], hue='generation',) ;
ax1.get_legend().remove()
ax1.set_title('(NA)', size=15, fontweight='bold') ;

# Plot 2 - Europe, South America, Asia, Australia perspective
sns.barplot(data=df_train_v2[df_train_v2['continent'].isin(['Europe','South America','Asia','Australia'])]
            , x='continent', y='suicides_no'
            ,hue='generation', ax=ax2
           ,hue_order=['G.I. Generation','Silent','Boomers','Generation X','Milennials','Generation Z']) ;
ax2.set_title('Suicide count by generation (E,SA,AS,AUS)', size=15, fontweight='bold') ;

# Plot 3 - Central America, Oceania, Africa perspective
sns.barplot(data=df_train_v2[df_train_v2['continent'].isin(['Central America','Oceania','Africa'])]
            , x='continent', y='suicides_no'
            ,hue='generation', ax=ax3
           ,hue_order=['G.I. Generation','Silent','Boomers','Generation X','Milennials','Generation Z']) ;
ax3.get_legend().remove()
ax3.set_title('(CA,O,AF)', size=15, fontweight='bold') ;


# From the start we can assume that variable __'generation'__ is an interaction variable between __'year'__ and __'age'__, so we want to see wether this interaction will be significant to our model. From this plot we can see clearly that Boomers have a high chance than any other category to commit suicide. This maybe because in the past, people tend to underestimate the effect of depression and there exist very little help/instituion/suicide hotlin not like now. Beside that around 1995-1998, there is an economic crisis happen all around the world with the impact of many people losing their job 

# # <font color='#00205b'>Model Selection</font>

# After preparing the data by using insight we get from our analysis, now its time to do some modelling. We will test some combination using variable __suicide_no__ and __suicides/100k pop__ as target variable and evaluate the best model using AIC
# -	Model 1 ->  Model 0 + Intercation of continent with population + stepwise
# -	Model 2 ->  Model 0 + Interaction of continent with gdp_per_capita   + stepwise
# -	Model 3 -> Model 0 + Interaction of continent with population  and gdp  + stepwise
# -	Model 4 ->  Model 0 + stepwise
# 
# In the end, our best model is Model 3 with __suicides/100k pop__ as the target variable
# 

# In[ ]:


# Split train and validation set
from sklearn.model_selection import train_test_split
dummy = pd.Series(df_train_v2.columns)
df_train_v3 = df_train_v2[df_train_v2>0].dropna()

x = dummy[~dummy.isin(['country','suicides_no','suicides/100k pop'])]
y = 'suicides/100k pop'
                      
x_train, x_valid, y_train, y_valid = train_test_split(df_train_v3[x], df_train_v3[y], test_size=0.2, random_state=11)


# In[ ]:


# Preparing dataset
X, Y = dataset_ready(x_train, y_train)
X2, Y2 = dataset_ready(x_valid, y_valid)

best_var = stepwise_selection(X,Y, list(X.columns))


# In[ ]:


# Generalize Linear Model - Gamma distribution
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
GLM_gamma = sm.genmod.GLM(endog=Y, exog=X[best_var]
                            ,family=sm.genmod.families.Gamma(link=sm.genmod.families.links.log))
GLM_result = GLM_gamma.fit()
print(GLM_result.summary())
print('Model AIC :',GLM_result.aic)
print('Model BIC :',GLM_result.bic)
print('Model deviance :',GLM_result.deviance)
print('Model RMSE :',rmse(GLM_result.predict(X2[best_var]),Y2))


# # <font color='#00205b'>Prediction</font>

# We will be using United States data from 2016 with GDP_per_capita 57588 and in North America continent
# <br> __Option 1__ : Male within age 25-34 with population 21845000 Milennials generation with suicide rates 26.95
# <br> __Option 2__ : Female within age 25-34 with population 21917000 Milennials generation with suicide rates 6.75
# <br> __Option 3__ : Male within age 35-54 with population 40539000 X generation with suicide rates 28.35
# <br> __Option 4__ : Female within age 35-54 with population 42031000 X generation with suicide rates 9.46
# <br> __Option 5__ : Male within age 15-24 with population 21719000 Milennials generation with suicide rates 21.06
# <br> __Option 6__ : Female within age 15-24 with population 21169000 Milennials generation with suicide rates 5.42
# 
# Sources : 
# <br> https://en.wikipedia.org/wiki/Suicide_in_the_United_StatesFrom
# <br> https://www.census.gov/data/tables/2016/demo/age-and-sex/2016-age-sex-composition.html
# <br> https://www.google.com/search?client=opera&q=united+states+gpd+per+capita+2016&sourceid=opera&ie=UTF-8&oe=UTF-8

# In[ ]:


# Preparing prediction
var = pd.Series(df_train_v3.columns)
x = var[~var.isin(['country','suicides_no','suicides/100k pop'])]
y = 'suicides/100k pop'

X3 = df_train_v3[x]
Y3 = df_train_v3[y]

# Input data for prediction
nx = len(X3)
ny = len(Y3)
X3.loc[nx+1] = [2016, 'male', '25-34 years', 21845000, 57588, 'Millenials', 'North America']
X3.loc[nx+2] = [2016, 'female', '25-34 years', 21917000, 57588, 'Millenials', 'North America']
X3.loc[nx+3] = [2016, 'male', '35-54 years', 40539000, 57588, 'Generation X', 'North America']
X3.loc[nx+4] = [2016, 'female', '35-54 years', 42031000, 57588, 'Generation X', 'North America']
X3.loc[nx+5] = [2016, 'male', '15-24 years', 21719000, 57588, 'Millenials', 'North America']
X3.loc[nx+6] = [2016, 'female', '15-24 years', 21169000, 57588, 'Millenials', 'North America']
Y3.loc[ny+1] = 26.95
Y3.loc[ny+2] = 6.75
Y3.loc[ny+3] = 28.35
Y3.loc[ny+4] = 9.46
Y3.loc[ny+5] = 21.06
Y3.loc[ny+6] = 5.42

# Tranform the data to be ready for precition
X3,Y3 = dataset_ready(X3, Y3)
X3 = X3.loc[nx+1:nx+6]
Y3 = Y3.loc[ny+1:nx+6]

# Predict
predict = GLM_result.predict(X3[best_var])
for i in range(len(predict)) :
    print('Option',i+1)
    print('Predicted suicide rates :',round(predict.iloc[i],2))
    print('Actual suicide rates :',Y3.iloc[i])
    print('')


# # <font color='#00205b'>Conclusion</font>

# - Suicide rates in United States is actually increasing just like Central for Disease Control say in their report. But in another contry it usually not the case except for South Korea. More detailed dataset is needed to understand why only in this country the suicide rates is increasing
# - Millenials is not the generation with the highest probability of committing suicide, it is actuallt Boomers generation. From the analysis we can see that Boomers suicide rates peak within 1995 to 1999. As we all know within that year, World economic crisis is happening causing many people to lose their job. This fenomena is most likely trigger the mass suicide
# - The model we use has an AIC value of 117661, a large number indeed indicating that our model is still bad for predicting. We can see from the prediction that our model tend to undervalue the suicide rates. This is bad especially for country like United States that most likel increase their suicide rates overtime. We think that a more detailed dataset is needed to improve our model
