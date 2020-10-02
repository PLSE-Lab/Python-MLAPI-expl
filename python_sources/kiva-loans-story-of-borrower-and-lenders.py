#!/usr/bin/env python
# coding: utf-8

# Kiva.org is an online crowdfunding platform to extend financial services to poor and financially excluded people around the world. 
# Kiva lenders have provided over $1 billion dollars in loans to over 2 million people. 
# 
# As every other financial transaction, Kiva loan has two sides of story, one from borrower and other is lender.
# The main objective of this notebook is to explore the people who are taking the loan for what reason and why
# lenders are supporting the project?  If you want to reuse the code for your exploration feel free to do so
# 
# Let us start the journey.
# 

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


kiva_loans = pd.read_csv("../input/kiva_loans.csv")
kiva_region_locations = pd.read_csv("../input/kiva_mpi_region_locations.csv")


# In[4]:


kiva_loans.head(3)


# Let us find the datasets information in details. I want to find the datatypes of the columns as well as how many different values are in each columns

# In[5]:


def unique_col_values(df):
    print("Number of rows in the dataset {rows} and cloumns {columns}".format(rows=df.shape[0], columns=df.shape[1]))
    for column in df.columns:
        print("{name} | {number} | {dt}". format(name=df[column].name, number = df[column].nunique(), dt=df[column].dtype))


# In[6]:


unique_col_values(kiva_loans)


# The loan amount has 479 unique values which is interesting given the fact it is a numeric column, where as, borrower_genders have 11298 values which is catergorical field !!! It suggests we need to do some data cleaning . For now lets explore the loan amount.

# In[7]:


print('Average Kiva loan amount US {:.3f}$'.format(kiva_loans['loan_amount'].mean()))


# By exploring the loan amount, it seems loan amount is highly skewed and very large standard deviation. So, if we do not exclude outliers then median would be right statistics, however, if we exclude the extreme values we would be able to get right stats from the data. 
# 
# Let us get the insight what is the maximum loan amount, and has it been funded or not?
# 

# In[8]:


kiva_loans.loc[kiva_loans['loan_amount'].idxmax()]


# This is great that US 100000$ has been funded by supporter of this project. It brings new route to explore about the project activity and sector of loan in which funding was available.

# In[9]:


def agg_count(df, group_field):
    aggregate = df.groupby(group_field).size().sort_values(ascending=False)
    aggregate = pd.DataFrame(aggregate)
    aggregate.columns = ['Count']
    
    return aggregate


# In[10]:


activity_counts = agg_count(kiva_loans, 'activity') 


# What are the **Top10 Activity** for the loan borrower

# In[11]:


top10_activity = activity_counts[:10]
plt.figure(figsize=(12,8))
ax = sns.barplot(data =top10_activity, x='Count', y=top10_activity.index)
ax.set(xlabel='\n Top 10 activity in Kiva loans')
plt.title('Loans count by Activity');


#  Right for micro lending it takes "Agriculture" as  top landing sector. What are other sectors that borrower have loans?

# In[12]:


sector_counts = agg_count(kiva_loans, 'sector')


# In[13]:


plt.figure(figsize=(12,8))
ax = sns.barplot(data = sector_counts, x='Count', y=sector_counts.index)
ax.set(xlabel='\n Number of Kiva loans by sector')
plt.title('Loans count by Sector');


# Agriculture, Food ... 
# some of the sectors have the higher counts of loan. Well how about the total loan amount in these sectors. Let us visualize that

# In[14]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1520668977393' style='position: relative'><noscript><a href='#'><img alt='Loan_Sector ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ki&#47;Kiva_loans&#47;Loan_Sector&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Kiva_loans&#47;Loan_Sector' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ki&#47;Kiva_loans&#47;Loan_Sector&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1520668977393');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[15]:


def agg_avg(df, group_field, value_field):
    grouped = df.groupby(group_field)[value_field].mean().sort_values(ascending=False)
    return grouped


# Is the average loan amount story  same for each sector as its' count and sum?

# In[16]:


sector_avg_loans = agg_avg(kiva_loans, 'sector', 'loan_amount')
plt.figure(figsize=(10,6))
ax = sns.barplot(x=sector_avg_loans.values, y=sector_avg_loans.index)
ax.set(xlabel='\n Mean loan amount USD')
plt.title('Average loan amount by sector')


# It is interesting that Entertainment has the highest average loan amount? So, what are the other sectors, borrower are taking the loans.
# We need to get into the details each sector. 
# 
# That means how the loan was requested by sector and how the project was funded.

# As we have seen loan amout is positivly skewed, so if we want to get the right stats we need to exculde the outliers. 
# 
# We need to clean the datasets as discussed earlier.  I will do it a functional way so that anyone can use these funtion can discover their own story 

# In[17]:


def loan_sector(df, sector):
    """Kiva loans analysis by sector"""
    
    import pandas as pd 
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
    kiva_loans = df.drop(['id', 'region', 'currency', 'posted_time', 'tags'], axis =1)
    kiva_loans['disbursed_time'] = pd.to_datetime(kiva_loans['disbursed_time'])
    kiva_loans['funded_time'] = pd.to_datetime(kiva_loans['funded_time'])
    kiva_loans['borrower_genders'] = kiva_loans['borrower_genders'].apply(clean_gender) 
    
    loan_sector = kiva_loans[kiva_loans['sector']== sector]
    loan_borrower(loan_sector, sector)
    fund_lender(loan_sector, sector)
    gender_avg, loan_country, fund_repayment, fund_country = sector_analysis(loan_sector, sector)
    return gender_avg, loan_country, fund_repayment, fund_country 

def clean_gender(value):
    try:
        if value == 'female' or value == 'male': 
            return value
        else:
            return value.split(',')[0].strip()
    except:
        return 'Unknown'

def loan_borrower(sector_df, sector):
    "Printing the summary of maximum loan amount by sector"""    
    
    import pandas as pd
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
    max_sector = sector_df['loan_amount'].idxmax()
    sector_loan = sector_df['loan_amount'].max()
    loan_use = sector_df.loc[max_sector]['use']
    loan_country = sector_df.loc[max_sector]['country']
    loan_borrower = sector_df.loc[max_sector]['borrower_genders']
    loan_lender = sector_df.loc[max_sector]['lender_count']
    months_term = sector_df.loc[max_sector]['term_in_months']
    print(" *"*24)
    print("{0}{sec} sector largest borrower ".format(' '*8, sec=sector))
    print(" *"*24)
    print()
    print("Maximum loan in the {sec} sector is US {loan}$ ".format(sec=sector,loan =sector_loan))
    print("{country} has the highest {sec} sector loan".format(sec=sector, country = loan_country))
    if loan_use != 'nan':
        print("The loan is use to {use}".format(use=loan_use))
    print()
    if loan_borrower !='Unknown':
        print("Primary gender of the borrower is {gender}".format(gender=loan_borrower))
    if loan_lender != 0:    
        print('{lender} donars supported this project of total term in {term} months '.format(lender=loan_lender, term=months_term))
    print()

def fund_lender(sector_df,sector):    
    "Printing the summary of maximum fund amount by sector"""     
    import pandas as pd
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
    max_lender_sector = sector_df['lender_count'].idxmax()
    max_funded_amount = sector_df.loc[max_lender_sector]['funded_amount']
    total_lenders = sector_df.loc[max_lender_sector]['lender_count']
    lender_country = sector_df.loc[max_lender_sector]['country']
    lender_term = sector_df.loc[max_lender_sector]['term_in_months']
    lender_repayment = sector_df.loc[max_lender_sector]['repayment_interval']
    print(" *"*24)
    print('{0}Summary of highest funded loan'.format(' '*8))
    print(" *"*24)
    print()
    print("The biggest group have {lender} lenders in the {sec} sector".format(lender= total_lenders, sec=sector))
    print("The largest  group people supported US {funded}$ project in the {sec} sector".format(funded =max_funded_amount, sec=sector))
    print()
    print("{country} has got the maximum support for funded loan in {sec} sector".format(country = lender_country,sec=sector))
    print('This project has total term in {term} months and repyment interval is {repayment}'.format(term=lender_term, repayment=lender_repayment))
    print(' -'*24)

def sector_analysis(sector_df,sector):
    """Given the sector dataframe it will  analysis the sector"""
    import pandas as pd
    import numpy as np
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
    sector_IQR = sector_df['loan_amount'].quantile(0.75) - sector_df['loan_amount'].quantile(0.25)
    upper_limit = sector_df['loan_amount'].quantile(0.75) + 1.5*sector_IQR
    sector_loan = sector_df[sector_df['loan_amount']<= upper_limit]
    avg_loan = sector_loan['loan_amount'].mean()
    print('On average US {0:.3f}$  loan was requested in  {sec} sector'.format(avg_loan, sec=sector))
    
    sector_high_loans = sector_loan[sector_loan['loan_amount']> upper_limit]
    high_loan = sector_high_loans['loan_amount'].mean()
    if not np.isnan(high_loan):
        print('However, largest average loan amount was US {high}$  in the {sec} sector '.format(high=high_loan, sec=sector))
    
    sector_funded_IQR = sector_df['funded_amount'].quantile(0.75) - sector_df['funded_amount'].quantile(0.25)
    higher_limit = sector_df['funded_amount'].quantile(0.75) + 1.5*sector_funded_IQR
    funded_sector = sector_df[sector_df['funded_amount']<= higher_limit]
    avg_fund = funded_sector['funded_amount'].mean()
    print('The average funded project had US {0:.3f}$ in {sec} sector'.format(avg_fund, sec=sector))
    sector_gender_avg = sector_loan.groupby('borrower_genders')['loan_amount'].mean()
    top10_country_loan = sector_loan.groupby('country')['loan_amount'].mean().nlargest(10)
    fund_repayment = funded_sector.groupby('repayment_interval')['funded_amount'].mean()
    top10_country_fund = funded_sector.groupby('country')['funded_amount'].mean().nlargest(10)
    
    return sector_gender_avg, top10_country_loan, fund_repayment, top10_country_fund


# In[18]:


sector_gender, top10_country_sector, repayment, top10_funded_country = loan_sector(kiva_loans, 'Entertainment')


# That is very impresive. United State has highest request loan and largest funded project in the entertainment sector. We can visualize **Top 10** countries by average loan amount in the entertainment sector. 

# In[19]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

trace0 = go.Bar( x=top10_country_sector.index,
                y=top10_country_sector.values)
data = [trace0]
layout = go.Layout(
    title='Top 10 countries in the Entertainment sector<br>Average Loan amount in US dollars')
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='entertainment-loans')


# This is interesting that loan for entertainment was requested from different region of the worlds. How about repayment intervel in this sector

# In[20]:


trace0 = go.Bar( x=repayment.index,
                y=repayment.values,width = 0.5,
                marker=dict(color='rgb(244,109,67)'))
data = [trace0]
layout = go.Layout(title='Repayment interval in the Entertainment sector')
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='repayment-entertainment')


# Most of the repayments are monthly in this setor, however, irregular payment is quite high as well. has funding in the entertainment sector got relation with repayments? Let us visualize  the average funding of  **Top10** countries.

# In[21]:


trace0 = go.Bar(x=top10_funded_country.index, y=top10_funded_country.values,
                marker=dict(color='rgba(50, 171, 96, 0.6)',
                line=dict(color='rgba(50, 171, 96, 1.0)',
                width=1)))
data = [trace0]
layout = go.Layout(title='Top 10 countries in the Entertainment sector<br>Average Funded amount in US dollars')
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='entertainment-fund')


# As seen earlier  higher education is in the top activity. So let us get the insight of Education sector

# In[22]:


edu_gen_loan, top10_edu_loan, repayment_edu, top10_funded_edu = loan_sector(kiva_loans, 'Education')


# In[ ]:


trace0 = go.Bar(x=top10_edu_loan.index, y=top10_edu_loan.values,
                name='Mean Loan_amount',
                marker=dict(
                color='rgb(49,130,189)')
               )
trace1 = go.Bar(x=top10_funded_edu.index, y=top10_funded_edu.values,
                name='Mean Fund_amount',
                marker=dict(
                color='rgb(204,204,204)'
                )
                )

data = [trace0, trace1]
layout = go.Layout(xaxis=dict(tickangle=-45),
    barmode='group',)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='edu-country-bar')


# So We can discover so many story for each sector !!! The next thing I will try to do the same for all the country.
# 
# Finally I will finish with the country mean loan visualization which will be next journey of this note book 

# In[1]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1520811364414' style='position: relative'><noscript><a href='#'><img alt='Loan by Country ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;BK&#47;BKYY2ZSMH&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;BKYY2ZSMH' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;BK&#47;BKYY2ZSMH&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1520811364414');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:





# 
