#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Kaggle-Kiva-Crowdfunding-Exploratory-Data-Analysis" data-toc-modified-id="Kaggle-Kiva-Crowdfunding-Exploratory-Data-Analysis-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Kaggle Kiva Crowdfunding Exploratory Data Analysis</a></span><ul class="toc-item"><li><span><a href="#About-Kiva:" data-toc-modified-id="About-Kiva:-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>About Kiva:</a></span></li><li><span><a href="#Dataset-Objective:" data-toc-modified-id="Dataset-Objective:-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Dataset Objective:</a></span></li><li><span><a href="#Objective-of-notebook:" data-toc-modified-id="Objective-of-notebook:-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Objective of notebook:</a></span></li></ul></li><li><span><a href="#Load-Libraries" data-toc-modified-id="Load-Libraries-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Load Libraries</a></span></li><li><span><a href="#Read-the-data" data-toc-modified-id="Read-the-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Read the data</a></span></li><li><span><a href="#Check-for-missing-values-in-the-data" data-toc-modified-id="Check-for-missing-values-in-the-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Check for missing values in the data</a></span></li><li><span><a href="#Data-Exploration" data-toc-modified-id="Data-Exploration-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Data Exploration</a></span><ul class="toc-item"><li><span><a href="#Univariate-Analysis" data-toc-modified-id="Univariate-Analysis-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Univariate Analysis</a></span><ul class="toc-item"><li><span><a href="#Number-of-loans-by-activities" data-toc-modified-id="Number-of-loans-by-activities-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Number of loans by activities</a></span></li><li><span><a href="#Number-of-loans-sectorwise" data-toc-modified-id="Number-of-loans-sectorwise-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Number of loans sectorwise</a></span></li><li><span><a href="#Number-of-loans-funded-in-a-country" data-toc-modified-id="Number-of-loans-funded-in-a-country-5.1.3"><span class="toc-item-num">5.1.3&nbsp;&nbsp;</span>Number of loans funded in a country</a></span></li><li><span><a href="#Distribution-of-loans-by-gender" data-toc-modified-id="Distribution-of-loans-by-gender-5.1.4"><span class="toc-item-num">5.1.4&nbsp;&nbsp;</span>Distribution of loans by gender</a></span></li><li><span><a href="#Distirbution-of-loans-by-use" data-toc-modified-id="Distirbution-of-loans-by-use-5.1.5"><span class="toc-item-num">5.1.5&nbsp;&nbsp;</span>Distirbution of loans by use</a></span></li><li><span><a href="#Distribution-of-loans-by-repayment_interval" data-toc-modified-id="Distribution-of-loans-by-repayment_interval-5.1.6"><span class="toc-item-num">5.1.6&nbsp;&nbsp;</span>Distribution of loans by repayment_interval</a></span></li><li><span><a href="#Kernel-Density-Distribution-of-attributes/features" data-toc-modified-id="Kernel-Density-Distribution-of-attributes/features-5.1.7"><span class="toc-item-num">5.1.7&nbsp;&nbsp;</span>Kernel Density Distribution of attributes/features</a></span><ul class="toc-item"><li><span><a href="#Kernel-Density-Estimation-of-Loan-amount-attribute" data-toc-modified-id="Kernel-Density-Estimation-of-Loan-amount-attribute-5.1.7.1"><span class="toc-item-num">5.1.7.1&nbsp;&nbsp;</span>Kernel Density Estimation of Loan amount attribute</a></span></li><li><span><a href="#Kernel-Density-Estimation-of-term-in-months-attribute" data-toc-modified-id="Kernel-Density-Estimation-of-term-in-months-attribute-5.1.7.2"><span class="toc-item-num">5.1.7.2&nbsp;&nbsp;</span>Kernel Density Estimation of term in months attribute</a></span></li><li><span><a href="#Kernel-Density-Estimation-of-Lender-Count--attribute" data-toc-modified-id="Kernel-Density-Estimation-of-Lender-Count--attribute-5.1.7.3"><span class="toc-item-num">5.1.7.3&nbsp;&nbsp;</span>Kernel Density Estimation of Lender Count  attribute</a></span></li></ul></li><li><span><a href="#Distribution-of-number-of-loans-in-a-country-via-Choropelth-maps" data-toc-modified-id="Distribution-of-number-of-loans-in-a-country-via-Choropelth-maps-5.1.8"><span class="toc-item-num">5.1.8&nbsp;&nbsp;</span>Distribution of number of loans in a country via Choropelth maps</a></span></li><li><span><a href="#Map-of-country-by-average-loan-amount" data-toc-modified-id="Map-of-country-by-average-loan-amount-5.1.9"><span class="toc-item-num">5.1.9&nbsp;&nbsp;</span>Map of country by average loan amount</a></span></li></ul></li><li><span><a href="#Bivariate-Analysis" data-toc-modified-id="Bivariate-Analysis-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Bivariate Analysis</a></span><ul class="toc-item"><li><span><a href="#Repayment-Interval-v/s-Borrowers-Gender" data-toc-modified-id="Repayment-Interval-v/s-Borrowers-Gender-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>Repayment Interval v/s Borrowers Gender</a></span></li></ul></li></ul></li></ul></div>

# ## Kaggle Kiva Crowdfunding Exploratory Data Analysis
# 
# ### About Kiva:
# Kiva.org is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people.
# 
# ### Dataset Objective:
# In order to set investment priorities, help inform lenders, and understand their target communities, knowing the level of poverty of each borrower is critical. However, this requires inference based on a limited set of information for each borrower. Kiva would like to build more localized models to estimate the poverty levels of residents in the regions where Kiva has active loans. 
# 
# ### Objective of notebook:
# This notebook aims to undearstnd and explore the dataset using various univariate, Bivariate and Multivariate analysis techniques.
# 

# ## Load Libraries

# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt # for plotting
import seaborn as sns

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
from plotly.graph_objs import *
import plotly
init_notebook_mode(connected=True)


# ## Read the data
# First, Lets look at the available variables

# In[69]:


kiva_loans_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")


# In[48]:


kiva_mpi_region_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
kiva_theme_ids_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
kiva_themes_region_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")


# In[49]:


lenders_df = pd.read_csv("../input/additional-kiva-snapshot/lenders.csv")
loan_coords =  pd.read_csv("../input/additional-kiva-snapshot/loan_coords.csv")
loan = pd.read_csv("../input/additional-kiva-snapshot/loans.csv")
country_stats = pd.read_csv("../input/additional-kiva-snapshot/country_stats.csv")
loans_lenders = pd.read_csv("../input/additional-kiva-snapshot/loans_lenders.csv")


# In[50]:


print("size of kiva_loans: {}".format(kiva_loans_df.shape))
print("size of kiva_mpi_region: {}".format(kiva_mpi_region_df.shape))
print("size of kiva_themes_region: {}".format(kiva_themes_region_df.shape))
print("size of kiva_theme_ids: {}".format(kiva_theme_ids_df.shape))
print("size of lenders: {}".format(lenders_df.shape))
print("size of loan_coords: {}".format(loan_coords.shape))
print("size of loans: {}".format(loan.shape))
print("size of country_stats: {}".format(country_stats.shape))
print("size of loans_lenders: {}".format(loans_lenders.shape))


# In[51]:


print(kiva_loans_df[100:150])


# In[52]:


print(loan_coords)


# In[53]:


print(kiva_theme_ids_df)


# ## Check for missing values in the data

# In[54]:


def find_missing_values_in_columns(df):
    total = (df.isnull().sum()).sort_values(ascending = False)
    percent = ((df.isnull().sum()/ df.isnull().count())).sort_values(ascending = False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)
    


# In[55]:


find_missing_values_in_columns(kiva_loans_df)


# In[56]:


find_missing_values_in_columns(kiva_mpi_region_df)


# In[57]:


find_missing_values_in_columns(kiva_themes_region_df)


# In[58]:


find_missing_values_in_columns(kiva_theme_ids_df)


# In[59]:


find_missing_values_in_columns(lenders_df)


# In[60]:


find_missing_values_in_columns(loan)


# In[61]:


find_missing_values_in_columns(loan_coords)


# In[62]:


find_missing_values_in_columns(loans_lenders)


# In[63]:


find_missing_values_in_columns(country_stats)


# ## Data Exploration
# ### Univariate Analysis

# In[70]:


#Shows the distribution of loans based on columns 
def loans_groupby_cols(df,colname,option ='horizontal',c=25):
    total_count = df[colname].value_counts().head(c)
    plt.figure(figsize=(15,9))
    if option == 'vertical':
        sns.barplot(total_count.index,total_count.values)
        for i, v in enumerate(total_count.values):
            plt.text(i,4500,v,color='k',fontsize=12,rotation='vertical')
        plt.xticks(rotation='vertical')
        plt.ylabel('Number of loans were given')
        plt.xlabel(colname)
    else:
        sns.barplot(total_count.values,total_count.index)
        for i, v in enumerate(total_count.values):
            plt.text(0.9,i,v,color='k',fontsize=12)
        plt.xticks(rotation='vertical')
        plt.xlabel('Number of loans were given')
        plt.ylabel(colname)
    plt.title("Top {} in which more loans were given".format(colname))
    plt.show()


# #### Number of loans by activities

# In[71]:


loans_groupby_cols(kiva_loans_df,'activity','vertical',50)


# Farming and General Store are the top two activities funded by Kiva loans.

# #### Number of loans sectorwise

# In[66]:


loans_groupby_cols(kiva_loans_df,'sector')


# Agriculture is the top most sector funded by kiva loans.

# #### Number of loans funded in a country

# In[72]:


loans_groupby_cols(kiva_loans_df,'country')


# Philippines and Kenya received maximum number of loans.

# #### Distribution of loans by gender

# In[73]:


def loans_by_borrowers_gender(df):
    df1= df['borrower_genders']
    f_count = 0
    m_count = 0
    for gender_lst in df1:
        if pd.isnull(gender_lst) == False:
            gender_lst = gender_lst.split(',')
            for v in gender_lst:
                if v.strip() == 'female':
                    f_count = f_count + 1
                else:
                    m_count = m_count + 1
    total = f_count + m_count         
    df=pd.DataFrame({'gender':{'female':round((float(f_count)/total*100)),'male':round((float(m_count)/total*100))}}) 
    plt.figure(figsize=(9,6))
    sns.barplot(y=df['gender'].values,x = df.index)
    plt.ylim(0, 100)
    plt.xlabel('Borrowers Gender')
    plt.ylabel('Percentage(%)')
    plt.title("Distirbution of loans by Gender")
    plt.show()    
    
loans_by_borrowers_gender(kiva_loans_df)
#loans_groupby_cols(kiva_loans_df,'borrower_genders')


# Note: The borrowers_gender column needed some preprocessing to be clearly able to divide data into male / female genders.Assuming that the loans may be sometimes given to a group of people instead of single applicant. we can count number of male and female in each groups and consider them as each individual applicant to get correct represntation of loans per gender. The borrowers_gender column has few missing values for the chart these values are dropped.
# 
# Alternative: One can consider the group of people in the borrowers gender as a seperate label like 'group'. Now the borrowers gender column with have (female, male, group,nan) values.
# 
# 
# Observation: 80% of loan borrowers are female and 20% are male.
# 

# #### Distirbution of loans by use

# In[74]:


loans_groupby_cols(kiva_loans_df,'use')


# Note: From the above results we can see that the data needs some cleaning done in order to aggregate similar uses into one group. For eg: The top 3 uses of the laon amount was to buysafe drinking water for the family which can be consolidated into on category say "to buy safe drinking water for the family"

# #### Distribution of loans by repayment_interval

# In[75]:


def loans_by_repayment_interval(df, colname):
    total_count = (df[colname].value_counts() / len(df) * 100)
    plt.figure(figsize=(15,9))
    sns.barplot(total_count.index,total_count.values, palette=sns.color_palette("muted"))
    ax = plt.gca()
    labels = ["%d" % i for i in total_count.values]
    for p,label in zip(ax.patches,labels):
        ax.text(p.get_x() + p.get_width()/2.,p.get_height(),label, 
                fontsize=12, color='black', ha='right', va='bottom')
    plt.ylabel('percentage(%)')
    plt.xlabel(colname)
    plt.title("Distribution of loans based on repayment interval types")
    plt.show()

loans_by_repayment_interval(kiva_loans_df,'repayment_interval')


# 51% of loans were repaid using monthly installments. However, it is noteworthy to see that 38% of borrower made irregular payments.
# 

# #### Kernel Density Distribution of attributes/features

# In[76]:


#Plots KDE based on the column name passed.
def dist_colname(df,colname,remove_outliers = True):
    temp = df[colname]
    
    #formatter = ticker.FormatStrFormatter('$%1.2f')
    plt.figure(figsize=(15,5))
    if remove_outliers == False:
        sns.distplot(temp)
    else:
        sns.distplot(temp[~(((temp - temp.median()).abs()) > 3*temp.std())])
    #plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(500))
    #plt.gca().xaxis.set_major_formatter(formatter)
    plt.ylabel('Density Estimate')
    plt.xlabel(colname)
    plt.title("")
    plt.show()


# ##### Kernel Density Estimation of Loan amount attribute

# In[77]:


dist_colname(kiva_loans_df,'loan_amount',False)


# From the Density Estimation plot we can see that data has very long tail. To take better look at the distribution of the loan amount we remove the outliers that 3 standard deviations away from the median.

# In[78]:


dist_colname(kiva_loans_df,'loan_amount')


# After removing the outliers, we can see that most of the loan amount borrowed range between 200 to 500 dollars and second most loan amount borrowed is $1000.

# ##### Kernel Density Estimation of term in months attribute

# In[79]:


dist_colname(kiva_loans_df,'term_in_months',False)


# In[80]:


dist_colname(kiva_loans_df,'term_in_months')


# Most of the loan were repaid in 8 months followed by  14 months

# ##### Kernel Density Estimation of Lender Count  attribute

# In[81]:


dist_colname(kiva_loans_df,'lender_count')


# Most of the loan were sponosered by a group of 10 - 15 people/lenders

# #### Distribution of number of loans in a country via Choropelth maps

# In[82]:


#@hidden_cell

grouped =  kiva_loans_df.groupby(['country'])['id'].count().sort_values(ascending=False)
#print(grouped)
data = [ dict(
    type = 'choropleth',
    locations = grouped.index,
    locationmode='country names',
    z = grouped.values,
    text = grouped.index,
    colorscale = 'Reds',
    marker = dict(line = dict (width = 0.5)),
    colorbar = dict(title = 'Number of loans'),
    ) ]

layout = dict(
    title = 'Map showing number of loans distributed in the country ',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(type = 'Natural earth')),
    )

fig = dict( data=data, layout=layout )
py.iplot(fig,validate=False)

#loans_by_region(kiva_loans_df)   


# #### Map of country by average loan amount

# In[83]:


df = kiva_loans_df.groupby(['country'])['loan_amount'].mean().sort_values(ascending=False)
#print(df)
data = [ dict(
        type = 'choropleth',
        locations = df.index,
        locationmode='country names',
        z = df.values,
        text = df.index,
        colorscale = 'Reds',
        marker = dict(
            line = dict (
                width = 0.5
            )
        ),
        colorbar = dict(
            tickprefix = '$',
            title = 'Loan Amount US$'
        ),
    ) ]

layout = dict(
    title = 'Map showing average loan amount distributed in the country ',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Natural earth'
        )
    ),
)

fig = dict( data=data, layout=layout )
py.iplot(fig)



# Country Cote D'Ivoire received the highest average loan amount is 50000 US Dollars. It  is interesting to see that average loan amount in Philippines is  345 US Dollars and Kenya is 455 US Dollars.
# 

# ### Bivariate Analysis

# #### Repayment Interval v/s Borrowers Gender

# In[84]:


#Pre Processing to the borrowers gender column to perform bivariate analysis with other variable. The group values in gender 
# are split and only the first item in the list is considered for groups and nan's are kept as it is.

def pre_process_gender(group = False):
    kiva_loans_df.borrower_genders = kiva_loans_df.borrower_genders.astype(str)
    gender_data = pd.DataFrame(kiva_loans_df.borrower_genders.str.split(',').tolist())
    
    if group == False:
        kiva_loans_df['borrowers_gender_new'] = gender_data[0]
        kiva_loans_df.loc[kiva_loans_df.borrowers_gender_new == 'nan', 'borrowers_gender_new'] = np.nan
        #print kiva_loans_df[['borrowers_gender_new']]
    else:
        kiva_loans_df['borrowers_gender_group'] = gender_data[0]
        for i in range(len(gender_data)):
            if gender_data.loc[i,1] != None:
                kiva_loans_df.at[i,'borrowers_gender_group'] = 'group'
        #print kiva_loans_df[['borrowers_gender_group']]
    


# In[85]:


def repayment_by_gender(df,group = False):
    sns.set(style="ticks")
    if group == True:
        pre_process_gender(True)
        temp =  df[['id','repayment_interval','borrowers_gender_group']]
        grouped = temp.groupby(['borrower_genders','repayment_interval'])['id'].count()
        sns.countplot( x="borrower_genders",data=temp, hue='repayment_interval')                    
    else:
        pre_process_gender(False)
        temp =  df[['id','repayment_interval','borrowers_gender_new']]
        grouped = temp.groupby(['borrowers_gender_new','repayment_interval'])['id'].count()
        sns.countplot(x='borrowers_gender_new',data = temp,hue="repayment_interval")
        
    level0 = grouped.groupby(level=0)
    print("Repayment Intreval by Gender in term Count")
    for val in grouped.index.levels[0].values:
        print("\nRepyament by {}".format(val))
        for i,j in zip(grouped[val].index.values,grouped[val].values):
            print("{} :{}".format(i,j))          
    plt.show()
    


# In[86]:


def repayment_by_gender_percent(df,group = False):
    sns.set(style="ticks")
    if group == True:
        pre_process_gender(True)
        temp =  df[['id','repayment_interval','borrowers_gender_group']]
        grouped = temp.groupby(['borrowers_gender_group','repayment_interval'])['id'].count()
    
    else:
        pre_process_gender(False)
        temp =  df[['id','repayment_interval','borrowers_gender_new']]
        grouped = temp.groupby(['borrowers_gender_new','repayment_interval'])['id'].count()

        
    level0 = grouped.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    
    print("Repayment Intreval by Gender in term percent")
    for val in grouped.index.levels[0].values:
        print("\nRepyament by {}".format(val))
        for i,j in zip(level0[val].index,level0[val]):
            print("{} :{}".format(i,j))
            
    df_1 = pd.DataFrame(columns=['gender','interval','percent'])
    for val in grouped.index.levels[0].values:
        for i,j in zip(level0[val].index,level0[val]):
            df_1 = df_1.append({'gender': val,'interval':i,'percent':j}, ignore_index=True)

    #sns.barplot(x='gender',data = df_1,hue="interval")
    sns.barplot(x='gender',y ='percent',hue="interval",data =df_1)
    plt.ylim(0, 100)
    plt.xlabel('Borrowers Gender')
    plt.ylabel('Percentage(%)')
    plt.title("Distirbution of repayment interval by Gender")
    plt.show()    


# In[87]:


repayment_by_gender(kiva_loans_df,False)


# In[88]:


repayment_by_gender_percent(kiva_loans_df,False)


# Using the simple count of repayment_intreval gender-wise does not show correct information as we know that there more female borrowers then males. So I decided that percentage would be a better way to compare repayments amongs genders. From the above chart, we can see that male borrowers made higher monthly repayments than female. Females made more irregular payments than males.
# 
