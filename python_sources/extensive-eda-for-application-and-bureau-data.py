#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import missingno as msno

from scipy.stats import gaussian_kde

plt.style.use('seaborn')
sns.set(font_scale=1.5)

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()
import random

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm


# Contents
# - <a href='#1'>1. Read dataset</a>
#     - <a href='#1_1'>1.1. Read dataset</a>
#     - <a href='#1_2'>1.2. Check null data</a>
#     - <a href='#1_2'>1.3. Make meta dataframe</a>
# - <a href='#2'>2. EDA - application train</a>
#     - <a href='#2_1'>2.1. Object feature</a>
#         - <a href='#2_1_1'>2.1.1 Contract type</a>
#         - <a href='#2_1_2'>2.1.2. Gender</a>
#         - <a href='#2_1_3'>2.1.3. Do you have an own car?</a>
#         - <a href='#2_1_4'>2.1.4. Do you have own realty?</a>
#         - <a href='#2_1_5'>2.1.5. Suite type</a>
#         - <a href='#2_1_6'>2.1.6. Income type</a>
#         - <a href='#2_1_7'>2.1.7 Contract type </a>
#         - <a href='#2_1_8'>2.1.8. 2.8 Family status</a>
#         - <a href='#2_1_9'>2.1.9. Housing type</a>
#         - <a href='#2_1_10'>2.1.10. Occupation type</a>
#         - <a href='#2_1_11'>2.1.11. Process start (weekday)</a>
#         - <a href='#2_1_12'>2.1.12. Organization type</a>
#         - <a href='#2_1_13'>2.1.13. FONDKAPREMONT </a>
#         - <a href='#2_1_14'>2.1.14. House type</a>
#         - <a href='#2_1_15'>2.1.15. Wall material</a>
#         - <a href='#2_1_16'>2.1.16. Emergency</a>
#     - <a href='#2_2'>2.2. Int feature</a>
#         - <a href='#2_2_1'>2.2.1 Count of children</a>
#         - <a href='#2_2_2'>2.2.2. Mobil</a>
#         - <a href='#2_2_3'>2.2.3. EMP Phone</a>
#         - <a href='#2_2_4'>2.2.4. Work phone</a>
#         - <a href='#2_2_5'>2.2.5. Cont mobile</a>
#         - <a href='#2_2_6'>2.2.6. Phone</a>
#         - <a href='#2_2_7'>2.2.7 Region Rating Client</a>
#         - <a href='#2_2_8'>2.2.8. Region Rating Client With City</a>
#         - <a href='#2_2_9'>2.2.9. Hour Appr Process Start</a>
#         - <a href='#2_2_10'>2.2.10. Register region and not live region</a>
#         - <a href='#2_2_11'>2.2.11. Register region and not work region</a>
#         - <a href='#2_2_12'>2.2.12. Live region and not work region</a>
#         - <a href='#2_2_13'>2.2.13. Register city and not live city</a>
#         - <a href='#2_2_14'>2.2.14. Register city and not work city</a>
#         - <a href='#2_2_15'>2.2.15. Live city and not work city</a>
#         - <a href='#2_2_16'>2.2.16. Heatmap for int features</a>
#         - <a href='#2_2_17'>2.2.17. More analysis for int features which have correlation with target</a>
#         - <a href='#2_2_18'>2.2.18. linear regression analysis on the high correlated feature combinations</a> 
# - <a href='#3'>3. EDA - Bureau</a>
#     - <a href='#3_1'>3.1. Read and check data</a>
#     - <a href='#3_2'>3.2. Merge with application_train</a>
#     - <a href='#3_3'>3.3. Analysis on object feature</a>
#         - <a href='#3_3_1'>3.3.1. Credit active</a>
#         - <a href='#3_3_2'>3.3.2. Credit currency</a>
#         - <a href='#3_3_3'>3.3.3. Credit type</a>
#     - <a href='#3_4'>3.4. Analysis on int feature</a>
#         - <a href='#3_4_1'>3.4.1. Credit day</a>
#         - <a href='#3_4_2'>3.4.2. Credit day overdue</a>
#         - <a href='#3_4_3'>3.4.3. Credit day prolong</a>
#     - <a href='#3_5'>3.5. Analysis on float feature</a>
#         - <a href='#3_5_1'>3.5.1 Amount credit sum</a>
#         - <a href='#3_5_2'>3.5.2 Amount credit sum debt</a>
#         - <a href='#3_5_3'>3.5.3 Amount credit sum limit</a>
#         - <a href='#3_5_4'>3.5.4 Amount credit sum overdue</a>

# # <a id='1'>1. Read dataset</a>

# ## <a id='1_1'>1.1. Read dataset</a>

# In[ ]:


application_train = pd.read_csv('../input/application_train.csv')
# POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
# installments_payments = pd.read_csv('../input/installments_payments.csv')
# credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
# bureau = pd.read_csv('../input/bureau.csv')
# application_test = pd.read_csv('../input/application_test.csv')


# In[ ]:


print('Size of application_tra data', application_train.shape)
# print('Size of POS_CASH_balance data', POS_CASH_balance.shape)
# print('Size of bureau_balance data', bureau_balance.shape)
# print('Size of previous_application data', previous_application.shape)
# print('Size of installments payments data', installments_payments.shape)
# print('Size of credit_card_balance data', credit_card_balance.shape)
# print('Size of bureau data', bureau.shape)


# In[ ]:


application_train.head()


# ## <a id='1_2'>1.2. Check null data</a>

# - With msno library, we could see the blanks in the dataset. Check null data in application train.

# In[ ]:


msno.matrix(df=application_train, figsize=(10, 8), color=(0, 0.6, 1))


# In[ ]:


# checking missing data
total = application_train.isnull().sum().sort_values(ascending = False)
percent = (application_train.isnull().sum()/application_train.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.head(20)


# ## <a id='1_3'>1.3. Make meta dataframe</a>

# In[ ]:


application_train.info()


# - There are 3 data types(float64, int64, object) in application_train dataframe.

# - Before starting EDA, It would be useful to make meta dataframe which include the information of dtype, level, response rate and role of each features. 

# In[ ]:


def make_meta_dataframe(df):
    data = []
    for col in df.columns:
        if col == 'TARGET':
            role = 'target'
        elif col == 'SK_ID_CURR':
            role = 'id'
        else:
            role = 'input'

        if df[col].dtype == 'float64':
            level = 'interval'
        elif df[col].dtype == 'int64':
            level = 'ordinal'
        elif df[col].dtype == 'object':
            level = 'categorical'

        col_dict = {
            'varname': col,
            'role': role,
            'level': level,
            'dtype': df[col].dtype,
            'response_rate': 100 * df[col].notnull().sum() / df.shape[0]
        }
        data.append(col_dict)

    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'dtype', 'response_rate'])
    meta.set_index('varname', inplace=True)
    return meta


# In[ ]:


meta = make_meta_dataframe(application_train)


# ## <a id='1_4'>1.4. Check imbalance of target</a>

# - Checking the imbalance of dataset is important. If imbalanced, we need to select more technical strategy to make a model.

# In[ ]:


def random_color_generator(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color


# In[ ]:


cnt_srs = application_train['TARGET'].value_counts()
text = ['{:.2f}%'.format(100 * (value / cnt_srs.sum())) for value in cnt_srs.values]

trace = go.Bar(
    x = cnt_srs.index,
    y = (cnt_srs / cnt_srs.sum()) * 100,
    marker = dict(
        color = random_color_generator(2),
        line = dict(color='rgb(8, 48, 107)',
                   width = 1.5
                   )
    ), 
    opacity = 0.7
)

data = [trace]

layout = go.Layout(
    title = 'Target distribution(%)',
    margin = dict(
        l = 100
    ),
    xaxis = dict(
        title = 'Labels (0: repay, 1: not repay)'
    ),
    yaxis = dict(
        title = 'Account(%)'
    ),
    width=800,
    height=500
)
annotations = []
for i in range(2):
    annotations.append(dict(
        x = cnt_srs.index[i],
        y = ((cnt_srs / cnt_srs.sum()) * 100)[i],
        text = text[i],
        font = dict(
            family = 'Arial',
            size = 14,
        ),
        showarrow = True
    ))
    layout['annotations'] = annotations

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# - As you can see, target is imbalanced.
# - This fact makes this competition diffcult to solve. But, no pain, no gain. After this competition, we could learn many things! Enjoy!

# # <a id='2'>2. EDA - application_train </a>

# ## <a id='2_1'>2.1. Object feature</a>

# - I want to draw two count bar plot for each object and int features. One contain the each count of responses and other contain the percent on target.

# In[ ]:


def get_percent(df, temp_col, width=800, height=500):
    cnt_srs = df[[temp_col, 'TARGET']].groupby([temp_col], as_index=False).mean().sort_values(by=temp_col)

    trace = go.Bar(
        x = cnt_srs[temp_col].values[::-1],
        y = cnt_srs['TARGET'].values[::-1],
        text = cnt_srs.values[::-1],
        textposition = 'auto',
        textfont = dict(
            size=12,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
            marker = dict(
                color = random_color_generator(100),
                line=dict(color='rgb(8,48,107)',
                  width=1.5,)
            ),
            opacity = 0.7,
    )    
    return trace
#     fig = go.Figure(data=data, layout=layout)
#     py.iplot(fig)


def get_count(df, temp_col, width=800, height=500):
    cnt_srs = df[temp_col].value_counts().sort_index()

    trace = go.Bar(
        x = cnt_srs.index[::-1],
        y = cnt_srs.values[::-1],
        text = cnt_srs.values[::-1],
        textposition = 'auto',
        textfont = dict(
            size=12,
            color='rgb(0, 0, 0)'
        ),
        name = 'Percent',
        orientation = 'v',
            marker = dict(
                color = random_color_generator(100),
                line=dict(color='rgb(8,48,107)',
                  width=1.5,)
            ),
            opacity = 0.7,
    )    
    return trace
#     fig = go.Figure(data=data, layout=layout)
#     py.iplot(fig)


# In[ ]:


def plot_count_percent_for_object(df, temp_col, height=500):
    trace1 = get_count(df, temp_col)
    trace2 = get_percent(df, temp_col)

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Count', 'Percent'), print_grid=False)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout']['yaxis1'].update(title='Count')
    fig['layout']['yaxis2'].update(range=[0, 1], title='% TARGET')
    fig['layout'].update(title='{} (Response rate: {:.2f}%)'.format(temp_col, meta[(meta.index == temp_col)]['response_rate'].values[0]), margin=dict(l=100), width=800, height=height, showlegend=False)

    py.iplot(fig)


# In[ ]:


features_dtype_object = meta[meta['dtype'] == 'object'].index
features_dtype_int = meta[meta['dtype'] == 'int64'].index
features_dtype_float = meta[meta['dtype'] == 'float64'].index


# - Sometimes, null data itself can be important feature. So, I want to compare the change when using null data as feature and not using nulll data as feature.

# In[ ]:


application_object_na_filled = application_train[features_dtype_object].fillna('null')
application_object_na_filled['TARGET'] = application_train['TARGET']


# ### <a id='2_1_1'>2.1.1. Contract type</a>

# **REMIND:  repay == 0 and not repay == 1**

# In[ ]:


temp_col = features_dtype_object[0]
plot_count_percent_for_object(application_train, temp_col)


# - Most contract type of clients is Cash loans. 
# - Not repayment rate is higher in cash loans (~8%) than in revolving loans(~5%).

# ### <a id='2_1_2'>2.1.2. Gender</a>

# In[ ]:


temp_col = features_dtype_object[1]
plot_count_percent_for_object(application_train, temp_col)


# - The number of female clients is almoist double the number of male clients.
# - Males have a higher chance of not returning their loans (~10%), comparing with women(~7%).

# ### <a id='2_1_3'>2.1.3. Do you have an own car?</a>

# In[ ]:


temp_col = features_dtype_object[2]
plot_count_percent_for_object(application_train, temp_col)


# - The clients that owns a car are higher than no-car clients by a factor of two times. 
# - The Not-repayment percent is similar. (Own: ~7%, Not-own: ~8%)

# ### <a id='2_1_4'>2.1.4. Do you have own realty?</a>

# In[ ]:


temp_col = features_dtype_object[3]
plot_count_percent_for_object(application_train, temp_col)


# - T he clients that owns a realty almost a half of the ones that doesn't own realty. 
# - Both categories have not-repayment rate, about ~8%.

# ### <a id='2_1_5'>2.1.5. Suite type</a>

# In[ ]:


temp_col = features_dtype_object[4]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)


# - Most suite type of clients are 'Unaccompanied', followed by Family, Spouse, children.
# - When considering null data, there is no change the order.
# - Other_B and Other_A have higher not-repayment rate than others.

# ### <a id='2_1_6'>2.1.6. Income type</a>

# In[ ]:


temp_col = features_dtype_object[5]
plot_count_percent_for_object(application_train, temp_col)


# - Most of the clients get income from working. 
# - The number of Student, Unemployed, Bussnessman and Maternity leave are very few.
# - When unemployed and maternity leave, there is  high probability of not-repayment.

# ### <a id='2_1_7'>2.1.7. Education type</a>

# In[ ]:


temp_col = features_dtype_object[6]
plot_count_percent_for_object(application_train, temp_col)


# - Clients with secondary education type are most numerous, followed by higher education, incomplete higher.
# - Clients with Lower secondary have the highest not-repayment rate(~10%).

# ### <a id='2_1_8'>2.1.8. Family status</a>

# In[ ]:


temp_col = features_dtype_object[7]
plot_count_percent_for_object(application_train, temp_col)


# - Most of clients  for loans are married followed by single/not married, civial marriage.
# - Civil marriage have almost 10% ratio of not returning loans followed by single/notmarried(9.9%), separate(8%).

# ### <a id='2_1_9'>2.1.9. Housing type</a>

# In[ ]:


temp_col = features_dtype_object[8]
plot_count_percent_for_object(application_train, temp_col)


# - Clients with house/apartment are most numerous, followed by With parents, Municipal apartment.
# - When Rented apartment and live with parents, clients have somewhat high not-repayment ratio. (~12%)
# 

# ### <a id='2_1_10'>2.1.10. Occupation type</a>

# In[ ]:


temp_col = features_dtype_object[9]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)


# - When not considering null data, Majority of clients are laborers, sales staff, core staff, drivers. But with considering null data, null data(I think it would be 'not want to repond' or 'no job', 'not in category') are most numerous.
# - However, not-repayment rate is low for null data. Low-skill labor is the most high not-repayment rate (~17%) in both plot.

# ### <a id='2_1_11'>2.1.11. Process start (weekday)</a>

# In[ ]:


temp_col = features_dtype_object[10]
plot_count_percent_for_object(application_train, temp_col)


# - The number of process for weekend is less than other days. That's because Weekend is weekend.
# - There are no big changes between not-repayment rate of all days.
# - Day is not important factor for repayment.

# ### <a id='2_1_12'>2.1.12. Organization type</a>

# In[ ]:


temp_col = features_dtype_object[11]
plot_count_percent_for_object(application_train, temp_col)


# - The most frequent case of organization is Bussiness Entity Type 3 followed XNA and self-employ.
# - The Transport: type 3 has the highest not repayment rate(~16%), Industry: type 13(~13.5%).

# ### <a id='2_1_13'>2.1.13. FONDKAPREMONT</a>

# In[ ]:


temp_col = features_dtype_object[12]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)


# - Actually, I don't know exact meaning of this feature FONDKAPREMONT_MODE.
# - Anyway, when considering null data, nul data has the highest count and not-repayment rate.

# ### <a id='2_1_14'>2.1.14. House type</a>

# In[ ]:


temp_col = features_dtype_object[13]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)


# - When considering null data, null data and block of flats are two-top. 
# - But, specific housing and terraced house have higher not-repayment rate than block of flats. 
# - null data has the highest not-repayment rate(~9%).

# ### <a id='2_1_15'>2.1.15. Wall material</a>

# In[ ]:


temp_col = features_dtype_object[14]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)


# - There are over 150,000 null data for WALLSMATERIAL_MODE. 
# - Clients with Wooden have higher than 9% not repayment rate.

# ### <a id='2_1_16'>2.1.16. Emergency</a>

# In[ ]:


temp_col = features_dtype_object[15]
plot_count_percent_for_object(application_train, temp_col)
plot_count_percent_for_object(application_object_na_filled, temp_col)


# - For emergency state, there is also many null data. 
# - If clients is in an emergency state, not-repayment rate(~10%) is higher than not in an emergency state.
# - null is also high not-repayment rate(~-10%).

# ## <a id='2_2'>2.2. Int feature</a>

# - Let's do similar analysis for int features.

# In[ ]:


def plot_count_percent_for_int(df, temp_col, height=500):
    trace1 = get_count(df, temp_col)
    trace2 = get_percent(df, temp_col)

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Count', 'Percent'), print_grid=False)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout']['xaxis1'].update(tickvals=[i for i in range(20)])
    fig['layout']['xaxis2'].update(tickvals=[i for i in range(20)])
    fig['layout']['yaxis1'].update(title='Count')
    fig['layout']['yaxis2'].update(range=[0, 1], title='% TARGET')
    fig['layout'].update(title='{} (Response rate: {:.2f}%)'.format(temp_col, meta[(meta.index == temp_col)]['response_rate'].values[0]), margin=dict(l=100), width=800, height=height, showlegend=False)
    
    py.iplot(fig)


# In[ ]:


application_train_int = application_train[meta[meta['dtype'] == 'int64'].index]
application_train_int['TARGET'] = application_train['TARGET']


# ### <a id='2_2_1'>2.2.1. Count of children</a>

# In[ ]:


features_dtype_int


# In[ ]:


temp_col = features_dtype_int[2]
plot_count_percent_for_int(application_train_int, temp_col)


# - Most clients with no children requested loan. 
# - Clients with 9, 11 have 100% not-repayment rate. the each count of those cases is 2 and 1.
# - Except 9, 11, Clients with 6 children has high not-repayment rate.

# ### <a id='2_2_2'>2.2.2. Mobil</a>

# In[ ]:


temp_col = features_dtype_int[6]
plot_count_percent_for_int(application_train_int, temp_col)


# - There are no clients without mobil(maybe mobile).

# ### <a id='2_2_3'>2.2.3. EMP Phone</a>

# In[ ]:


temp_col = features_dtype_int[7]
plot_count_percent_for_int(application_train, temp_col)


# - Most clients(82%) have EPM Phone.
# - The gap between the not-repayment percent is about 3%.

# ### <a id='2_2_4'>2.2.3. Work Phone</a>

# In[ ]:


temp_col = features_dtype_int[8]
plot_count_percent_for_int(application_train, temp_col)


# - Most clients(80%) don't have work phone.

# ### <a id='2_2_5'>2.2.5. Cont mobile</a>

# In[ ]:


temp_col = features_dtype_int[9]
plot_count_percent_for_int(application_train, temp_col)


# - Clients who chose 'no' for CONT_MOBILE FALG is very few.(574)

# ### <a id='2_2_6'>2.2.6. Phone</a>

# In[ ]:


temp_col = features_dtype_int[10]
plot_count_percent_for_int(application_train, temp_col)


# - Most clients(72%) don't have work phone.

# ### <a id='2_2_7'>2.2.7. Region Rating Client</a>

# In[ ]:


temp_col = features_dtype_int[12]
plot_count_percent_for_int(application_train, temp_col)


# - Clients who chose 2 for REGION_RATING_CLIENT is numerous, followed by 3, 1.
# - For not-repayment, the order is 3, 2, 1.

# ### <a id='2_2_8'>2.2.8. Region Rating Client With City</a>

# In[ ]:


temp_col = features_dtype_int[13]
plot_count_percent_for_int(application_train, temp_col)


# - Clients who chose 2 for REGION_RATING_CLIENT with city is numerous, followed by 3, 1.
# - For not-repayment, the order is 3, 2, 1.

# ### <a id='2_2_9'>2.2.9. Hour Appr Process Start</a>

# In[ ]:


temp_col = features_dtype_int[14]
plot_count_percent_for_int(application_train, temp_col)


# - The most busy hour for Appr Process Start is a range from 10:00 to 13:00.

# ### <a id='2_2_10'>2.2.10. Register region and not live region</a>

# In[ ]:


temp_col = features_dtype_int[15]
plot_count_percent_for_int(application_train, temp_col)


# - 98.5% of clients registered their region but not live in the region.

# ### <a id='2_2_11'>2.2.11. Register region and not work region</a>

# In[ ]:


temp_col = features_dtype_int[16]
plot_count_percent_for_int(application_train, temp_col)


# - 95% of clients registered their region but not work in the region.

# ### <a id='2_2_12'>2.2.12. Live region and not work region</a>

# In[ ]:


temp_col = features_dtype_int[17]
plot_count_percent_for_int(application_train, temp_col)


# - 95.9% of clients lives in their region but don't work in the region.

# - For 3 questions about region(10, 11, 12), the not-repayment percent is similar for each case.

# ### <a id='2_2_13'>2.2.13. Register city and not live city</a>

# In[ ]:


temp_col = features_dtype_int[18]
plot_count_percent_for_int(application_train, temp_col)


# - 92.1% of clients registered city and don't live in the city.
# - Unlike region, city could be good information. Because the difference of the not-repayment percent between 'yes' and 'no' is higher than region case(2.2.10, 2.2.11, 2.2.12)

# ### <a id='2_2_14'>2.2.14. Register city and not work city</a>

# In[ ]:


temp_col = features_dtype_int[19]
plot_count_percent_for_int(application_train, temp_col)


# - 78% of clients registered city and don't work in the city.
# - If client is this case, the not-repayment rate is about 10%.

# ### <a id='2_2_15'>2.2.15. Live city and not work city</a>

# In[ ]:


temp_col = features_dtype_int[20]
plot_count_percent_for_int(application_train, temp_col)


# - 82% of clients registered city and don't work in the city.
# - If client is this case, the not-repayment rate is about 10%.

# ### <a id='2_2_16'>2.2.16. Flag document</a>

# In[ ]:


for i in range(21, 40):
    temp_col = features_dtype_int[i]
    plot_count_percent_for_int(application_train, temp_col)


# - Document 2: 13 clients chose 1 and not-repayment rate is high, about 30%.
# - Document 4: 25 clients chose 1 and not-repayment rate is 0. all the clients who chose 1 repaid.
# - Document 10: 7 clients chose 1 and not-repayment rate is 0. all the clients who chose 1 repaid.
# - Document 12: 2 clients chose 1 and not-repayment rate is 0. all the clients who chose 1 repaid.

# ### <a id='2_2_16'>2.2.16. Heatmap for int features</a>

# - Let's see the correlations between the int features. Heatmap helps us to see this easily.

# In[ ]:


data = [
    go.Heatmap(
        z = application_train_int.corr().values,
        x = application_train_int.columns.values,
        y = application_train_int.columns.values,
        colorscale='Viridis',
        reversescale = False,
#         text = True ,
    )
]
layout = go.Layout(
    title='Pearson Correlation of float-type features',
    xaxis = dict(ticks=''),
    yaxis = dict(ticks='' ),
    width = 900, height = 700,
    margin = dict(
        l = 250
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')


# - There are some combinations with high correlation coefficient.
# - FLAG_DOCUMENT_6 and FLAG_EMP_PHONE
# - DAYS_BIRTH and FLAG_EMP_PHONE
# - DAYS_EMPLOYED and FLAG_EMP_PHONE
# - In follow section, we will look those features more deeply using linear regression plot with seaborn.

# ### <a id='2_2_17'>2.2.17. More analysis for int features which have correlation with target</a>

# - At first, find the int features which have high correlation with target.

# In[ ]:


correlations = application_train_int.corr()['TARGET'].sort_values()
correlations[correlations.abs() > 0.05]


# - DAYS_BIRTH is some high correlation with target.
# - With dividing 365(year) and applying abs(), we can see DAYS_BIRTH in the unit of year(AGE).

# In[ ]:


temp_col = 'DAYS_BIRTH'
sns.kdeplot((application_train_int.loc[application_train_int['TARGET'] == 0, temp_col]/365).abs(), label='repay(0)', color='r')
sns.kdeplot((application_train_int.loc[application_train_int['TARGET'] == 1, temp_col]/365).abs(), label='not repay(1)', color='b')
plt.xlabel('Age(years)')
plt.title('KDE for {} splitted by target'.format(temp_col))
plt.show()


# - As you can see, The younger, The higher not-repayment probability.
# - The older, The lower not-repayment probability.

# ### <a id='2_2_18'>2.2.18. linear regression analysis on the high correlated feature combinations</a>

# - With lmplot from seaborn, we can draw linear regression plot very easily. Thanks!

# In[ ]:


sns.lmplot(x='FLAG_DOCUMENT_6', y='FLAG_EMP_PHONE', data=application_train_int)


# In[ ]:


col1 = 'FLAG_DOCUMENT_6'
col2 = 'FLAG_EMP_PHONE'
xy = np.vstack([application_train[col1].dropna().values[:100000], application_train[col2].dropna().values[:100000]])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
im = ax.scatter(application_train[col1].dropna().values[:100000], application_train[col2].dropna().values[:100000], c=z, s=50, cmap=plt.cm.jet)
ax.set_xlabel(col1)
ax.set_ylabel(col2)
fig.colorbar(im)


# - With gaussian kde density represented by color and linear regression plot, we can see that there are many clients who have EMP Phone and chose document 6.

# In[ ]:


sns.lmplot(x='DAYS_BIRTH', y='FLAG_EMP_PHONE', data=application_train_int)


# In[ ]:


col1 = 'DAYS_BIRTH'
col2 = 'FLAG_EMP_PHONE'
xy = np.vstack([np.abs((application_train[col1].dropna().values[:100000]/365)), application_train[col2].dropna().values[:100000]])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
im = ax.scatter(np.abs((application_train[col1].dropna().values[:100000]/365)), application_train[col2].dropna().values[:100000], c=z, s=50, cmap=plt.cm.jet)
ax.set_xlabel(col1)
ax.set_ylabel(col2)
fig.colorbar(im)


# - With gaussian kde density represented by color and linear regression plot, we can see that the younger people tend to have EMP phone.

# In[ ]:


sns.lmplot(x='DAYS_EMPLOYED', y='FLAG_EMP_PHONE', data=application_train_int.dropna().loc[:100000, :])


# In[ ]:


col1 = 'DAYS_EMPLOYED'
col2 = 'FLAG_EMP_PHONE'
xy = np.vstack([np.abs((application_train[col1].dropna().values[:100000]/365)), application_train[col2].dropna().values[:100000]])
z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
im = ax.scatter(np.abs((application_train[col1].dropna().values[:100000]/365)), application_train[col2].dropna().values[:100000], c=z, s=50, cmap=plt.cm.jet)
ax.set_xlabel(col1)
ax.set_ylabel(col2)
fig.colorbar(im)


# - With gaussian kde density represented by color and linear regression plot, we can see that clients with shorter employed tend to have EMP phone. (simiar result compared to FLAG_EMP_PHONE vs DAYS_BIRTH)

# ## <a id='2_3'>2.3. float feature</a>

# - Let's move on float features.

# ### <a id='2_3_1'>2.3.1. Heatmap for float features</a>

# - Let us draw the heatmap of float features.

# In[ ]:


application_train_float = application_train[meta[meta['dtype'] == 'float64'].index]
application_train_float['TARGET'] = application_train['TARGET']


# In[ ]:


data = [
    go.Heatmap(
        z = application_train_float.corr().values,
        x = application_train_float.columns.values,
        y = application_train_float.columns.values,
        colorscale='Viridis',
        reversescale = False,
        text = True ,
    )
]
layout = go.Layout(
    title='Pearson Correlation of float-type features',
    xaxis = dict(ticks=''),
    yaxis = dict(ticks='' ),
    width = 1200, height = 1200,
    margin = dict(
        l = 250
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')


# - There are some features which have  some high correlation with target. In follow section, we will find them and analyze them.
# - There are many feature combinations which have high correlation value(larger than 0.9).
# - Let's find the combinations.

# ### <a id='2_3_2'>2.3.2. More analysis for int features which have correlation with target</a>

# - Let's find the float features which are highly correlated with target.

# In[ ]:


correlations = application_train_float.corr()['TARGET'].sort_values()
correlations[correlations.abs() > 0.05]


# In[ ]:


temp_col = 'EXT_SOURCE_1'
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 0, temp_col], label='repay(0)', color='r')
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 1, temp_col], label='not repay(1)', color='b')
plt.title('KDE for {} splitted by target'.format(temp_col))
plt.show()


# - The simple kde plot(kernel density estimation plot) shows that the distribution of repay and not-repay is different for EXT_SOURCE_1.
# - EXT_SOURCE_1 can be good feature.

# In[ ]:


temp_col = 'EXT_SOURCE_2'
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 0, temp_col], label='repay(0)', color='r')
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 1, temp_col], label='not repay(1)', color='b')
plt.title('KDE for {} splitted by target'.format(temp_col))
plt.show()


# - Not as much as EXT_SOURCE_1 do, EXT_SOURCE_2 shows different distribution for each repay and not-repay.

# In[ ]:


temp_col = 'EXT_SOURCE_3'
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 0, temp_col], label='repay(0)', color='r')
sns.kdeplot(application_train_float.loc[application_train_float['TARGET'] == 1, temp_col], label='not repay(1)', color='b')
plt.title('KDE for {} splitted by target'.format(temp_col))
plt.show()


# - EXX_SOUCE_3 has similar trend with EXT_SOURCE_1.
# - EXT_SOURCE_3 can be good feature.

# ### <a id='2_3_3'>2.3.3. linear regression analysis on the high correlated feature combinations</a>

# - Using corr() and numpy boolean technique with triu(), we could obtain the correlation matrix without replicates.

# In[ ]:


corr_matrix = application_train_float.corr().abs()
corr_matrix.head()


# In[ ]:


upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()


# In[ ]:


threshold = 0.9
count = 1
combinations = []
for name, column in upper.iteritems():
    if (column > threshold).any():
        for col, value in column[column > 0.9].iteritems():
            print(count, name, col, value)
            combinations.append((name, col, value))
            count += 1


# - There are 60 combinations which have larger correlation values than 0.95.
# - Let's draw the regplot for all combinations with splitting the target.

# In[ ]:


fig, ax = plt.subplots(28, 2, figsize=(20, 400))
count = 0
for i in range(28):
    for j in range(2):
        sns.regplot(x=combinations[count][0], y=combinations[count][1], data=application_train_float[application_train_float['TARGET'] == 0], ax=ax[i][j], color='r')
        sns.regplot(x=combinations[count][0], y=combinations[count][1], data=application_train_float[application_train_float['TARGET'] == 1], ax=ax[i][j], color='b')
        ax[i][j].set_title('{} and {}, corr:{:.2f} '.format(combinations[count][0], combinations[count][1], combinations[count][2]))
        ax[i][j].legend(['repay', 'not repay'], loc=0)
        count += 1


# - After looking these 56 plots, I found som combinations in which the distribution for repay and not-repay is a bit different.
# - Let's see this with single and multi variable kde plot.
# - It is nice to use log-operation on features. With log-operation, we can analyze the distribution more easily.

# In[ ]:


def multi_features_kde_plot(col1, col2):
    fig, ax = plt.subplots(3, 2, figsize=(14, 20))
    g = sns.kdeplot(application_train_float.loc[application_train['TARGET'] == 0, :].dropna().loc[:50000, :][col1], application_train_float.loc[application_train['TARGET'] == 0, :].dropna().loc[:50000, :][col2], ax=ax[0][0], cmap="Reds")
    g = sns.kdeplot(application_train_float.loc[application_train['TARGET'] == 1, :].dropna().loc[:50000, :][col1], application_train_float.loc[application_train['TARGET'] == 1, :].dropna().loc[:50000, :][col2], ax=ax[0][1], cmap='Blues')
    ax[0][0].set_title('mutivariate KDE: target == repay')
    ax[0][1].set_title('mutivariate KDE: target == not repay')

    temp_col = col1
    sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, temp_col].dropna(), label='repay(0)', color='r', ax=ax[1][0])
    sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, temp_col].dropna(), label='not repay(1)', color='b', ax=ax[1][0])
    ax[1][0].set_title('KDE for {}'.format(temp_col))

    sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][1])
    sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][1])
    ax[1][1].set_title('KDE for {} with log'.format(temp_col))

    temp_col = col2
    sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, temp_col].dropna(), label='repay(0)', color='r', ax=ax[2][0])
    sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, temp_col].dropna(), label='not repay(1)', color='b', ax=ax[2][0])
    ax[2][0].set_title('KDE for {}'.format(temp_col))

    sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[2][1])
    sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[2][1])
    ax[2][1].set_title('KDE for {} with log'.format(temp_col))


# In[ ]:


col1 = 'OBS_60_CNT_SOCIAL_CIRCLE'
col2 = 'OBS_30_CNT_SOCIAL_CIRCLE'
multi_features_kde_plot(col1, col2)


# - The mutivariate kde plot of not-repay is broader than one of repay.
# - For both CNT_60_SOCIAL_CIRCLE and OBS_30_CNT_SOCIAL_CIRCLE, the distribution of each repay and not-repay is a bit different. Log-operation helps us to see them easily.

# In[ ]:


col1 = 'NONLIVINGAREA_MEDI'
col2 = 'NONLIVINGAREA_MODE'
multi_features_kde_plot(col1, col2)


# - This case is similar with previous case.
# - The mutivariate kde plot of not-repay is broader than one of repay.
# - For both CNT_60_SOCIAL_CIRCLE and OBS_30_CNT_SOCIAL_CIRCLE, the distribution of each repay and not-repay is a bit different. Log-operation helps us to see them easily.

# # <a id='3'>3. EDA - Bureau </a>

# - Bureau data contains the information of previous loan history of clients from other company.

# ## <a id='3_1'>3.1. Read and check data</a>

# In[ ]:


# Read in bureau
bureau = pd.read_csv('../input/bureau.csv')
bureau.head()


# In[ ]:


msno.matrix(df=bureau, figsize=(10, 8), color=(0, 0.6, 1))


# In[ ]:


bureau.head()


# ## <a id='3_2'>3.2. Merge with application_train</a>

# - A client can have several loans so that merge with bureau data can explode the row of application train.

# In[ ]:


print('Applicatoin train shape before merge: ', application_train.shape)
application_train = application_train.merge(bureau.groupby('SK_ID_CURR').mean().reset_index(), 
                                            left_on='SK_ID_CURR', right_on='SK_ID_CURR', 
                                            how='left', validate='one_to_one')
print('Applicatoin train shape after merge: ', application_train.shape)


# In[ ]:


meta= make_meta_dataframe(application_train)


# In[ ]:


bureau_cat_features = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']


# In[ ]:


bureau_object_df = pd.DataFrame()

for i, group_df in tqdm(enumerate(bureau.groupby('SK_ID_CURR'))):
    bureau_object_df.loc[i, 'SK_ID_CURR'] = group_df[0]
    bureau_object_df.loc[i, bureau_cat_features[0]] = group_df[1][bureau_cat_features[0]].values[0]
    bureau_object_df.loc[i, bureau_cat_features[1]] = group_df[1][bureau_cat_features[1]].values[0]
    bureau_object_df.loc[i, bureau_cat_features[2]] = group_df[1][bureau_cat_features[2]].values[0]


# In[ ]:


application_train = application_train.merge(bureau_object_df, on='SK_ID_CURR')


# ## <a id='3_3'>3.3. Analysis on object feature</a>

# ### <a id='3_3_1'>3.3.1 Credit active</a>

# In[ ]:


len(bureau.columns)


# In[ ]:


temp_col = 'CREDIT_ACTIVE'
plot_count_percent_for_object(application_train, temp_col)


# - Most credit type of clients is 'Closed', 'Active'. 
# - If credit type is finished in the state of bad dept, the not-repayment rate is some high.(20%)

# ### <a id='3_3_2'>3.3.2 Credit currency</a>

# In[ ]:


temp_col = 'CREDIT_CURRENCY'
plot_count_percent_for_object(application_train, temp_col)


# - 99.9% of clients chose currency 1.
# - By the way, the not-repayment rate is high at currency 3.

# ### <a id='3_3_3'>3.3.3 Credit type</a>

# In[ ]:


temp_col = 'CREDIT_TYPE'
plot_count_percent_for_object(application_train, temp_col)


# - Clients with consumer credit is most numerous, followed by credit card.
# - If clients requested loan for the purchase of equipment, the not-repayment rate is high.(23.5%). Next is microloan(20.6%).

# ## <a id='3_4'>3.4. Analysis on int feature</a>

# ### <a id='3_4_1'>3.4.1 Credit day</a>

# In[ ]:


temp_col = 'DAYS_CREDIT'
plt.figure(figsize=(10, 6))
sns.distplot(application_train.loc[(application_train['TARGET'] == 0), temp_col].dropna(), bins=100, label='repay(0)', color='r')
sns.distplot(application_train.loc[(application_train['TARGET'] == 1), temp_col].dropna(), bins=100, label='not repay(1)', color='b')
plt.title('Distplot for {} splitted by target'.format(temp_col))
plt.legend()
plt.show()


# - There are 2 general(not linear) trends we can see.
# - The shorter credit days, the more not-repayment.
# - The larger credit days, the more repayment.

# 
# ### <a id='3_4_2'>3.4.2 Credit day overdue</a>

# In[ ]:


temp_col = 'CREDIT_DAY_OVERDUE'
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, temp_col].dropna(), label='repay(0)', color='r', ax=ax[0])
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0])
ax[0].set_title('KDE for {}'.format(temp_col))

sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1])
sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1])
ax[1].set_title('KDE for {} with log'.format(temp_col))
plt.show()


# - It is hard to see the trend for now. Let's remove the samples. (overdue < 200)

# In[ ]:


temp_col = 'CREDIT_DAY_OVERDUE'
fig, ax = plt.subplots(2, 2, figsize=(16, 16))

sns.kdeplot(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0][0])
ax[0][0].set_title('KDE for {}'.format(temp_col))

application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 0), temp_col].dropna().hist(bins=100, ax=ax[0][1], normed=True, color='r', alpha=0.5)
application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 1), temp_col].dropna().hist(bins=100, ax=ax[0][1], normed=True, color='b', alpha=0.5)


sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][0])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][0])
ax[1][0].set_title('KDE for {} with log'.format(temp_col))


np.log(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001).hist(bins=100, ax=ax[1][1], normed=True, color='r', alpha=0.5)
np.log(application_train.loc[(application_train[temp_col] > 200) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001).hist(bins=100, ax=ax[1][1], normed=True, color='b', alpha=0.5)


# - As you can see, repay have a litter more right-skewed distribution.
# - To see more deeply, Let's divide the overdue feature into several groups. 

# In[ ]:


def overdue(x):
    if x < 30:
        return 'A'
    elif x < 60:
        return 'B'
    elif x < 90:
        return 'C'
    elif x < 180:
        return 'D'
    elif x < 365:
        return 'E'
    else:
        return 'F'


# In[ ]:


application_train['CREDIT_DAY_OVERDUE_cat'] = application_train['CREDIT_DAY_OVERDUE'].apply(overdue)
meta = make_meta_dataframe(application_train)


# In[ ]:


temp_col = 'CREDIT_DAY_OVERDUE_cat'
plot_count_percent_for_object(application_train, temp_col)


# - The clients with short overdue days(<30) is most numerous.
# - B group has the highest not-repayment rate (19%), followed by C, D, E. A group is the lowest.

# In[ ]:


temp_col = 'CREDIT_DAY_OVERDUE'
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

sns.kdeplot(application_train.loc[(application_train['TARGET'] == 0) & (application_train['CREDIT_DAY_OVERDUE'] > 30), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0])
sns.kdeplot(application_train.loc[(application_train['TARGET'] == 1) & (application_train['CREDIT_DAY_OVERDUE'] > 30), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0])
ax[0].set_title('KDE for {} (>30)'.format(temp_col))

sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 0) & (application_train['CREDIT_DAY_OVERDUE'] > 30), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1])
sns.kdeplot(np.log(application_train.loc[(application_train['TARGET'] == 1) & (application_train['CREDIT_DAY_OVERDUE'] > 30), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1])
ax[1].set_title('KDE for {} with log (>30)'.format(temp_col))
plt.show()


# - KDE plot with samples which have overdue larger than 30 shows that the distribution of clients who repaid is larger than that of not-repay clients.

# ### <a id='3_4_3'>3.4.3 Credit day prolong</a>

# In[ ]:


temp_col = 'CNT_CREDIT_PROLONG'
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.kdeplot(application_train.loc[application_train['TARGET'] == 0, temp_col], label='repay(0)', color='r', ax=ax[0])
sns.kdeplot(application_train.loc[application_train['TARGET'] == 1, temp_col], label='not repay(1)', color='b', ax=ax[0])
plt.title('KDE for {} splitted by target'.format(temp_col))

sns.kdeplot(application_train.loc[(application_train['TARGET'] == 0) & (application_train[temp_col] > 2), temp_col], label='repay(0)', color='r', ax=ax[1])
sns.kdeplot(application_train.loc[(application_train['TARGET'] == 1) & (application_train[temp_col] > 2), temp_col], label='not repay(1)', color='b', ax=ax[1])
plt.title('KDE for {} splitted by target (>3)'.format(temp_col))
plt.show()


# - There are no clients who have prolong larger than 3.

# ## <a id='3_5'>3.5. Analysis on float feature</a>

# ### <a id='3_5_1'>3.5.1 Amount credit sum</a>

# In[ ]:


temp_col = 'AMT_CREDIT_SUM'
fig, ax = plt.subplots(2, 2, figsize=(16, 16))
threshold = 2 * 10e6
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0][0])
ax[0][0].set_title('KDE for {} (< {})'.format(temp_col, threshold))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[0][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[0][1])
ax[0][1].set_title('KDE for {} with log (< {})'.format(temp_col, threshold))

sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[1][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[1][0])
ax[1][0].set_title('KDE for {} (> {})'.format(temp_col, threshold))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][1])
ax[1][1].set_title('KDE for {} with log (> {})'.format(temp_col, threshold))
plt.show()


# - As you can see, if credit is lower than 2,000,000, the distribution of each repay and not-repay is similar.
# - But, if credit is larger than 2,000,000, the distribution of each repay and not-repay is different. Many clients who have very high(> 10,000,000) credit repaid.

# ### <a id='3_5_2'>3.5.2 Amount credit sum debt</a>

# In[ ]:


temp_col = 'AMT_CREDIT_SUM_DEBT'
fig, ax = plt.subplots(2, 2, figsize=(16, 16))
threshold = 2 * 10e6
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0][0])
ax[0][0].set_title('KDE for {} (< {})'.format(temp_col, threshold))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[0][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[0][1])
ax[0][1].set_title('KDE for {} with log (< {})'.format(temp_col, threshold))

sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[1][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[1][0])
ax[1][0].set_title('KDE for {} (> {})'.format(temp_col, threshold))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][1])
ax[1][1].set_title('KDE for {} with log (> {})'.format(temp_col, threshold))
plt.show()


# - AMT_CREDIT_SUM_DEBT shows similar trend compared to AMT_CREDIT_SUM.
# - Many clients with high dept(> 50,000,000) repaid.

# ### <a id='3_5_3'>3.5.3 Amount credit sum limit</a>

# In[ ]:


temp_col = 'AMT_CREDIT_SUM_LIMIT'
fig, ax = plt.subplots(3, 2, figsize=(16, 24))
threshold1 = 1e4
threshold2 = 1e6

sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold1) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0][0])
ax[0][0].set_title('KDE for {} (< {})'.format(temp_col, threshold1))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[0][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[0][1])
ax[0][1].set_title('KDE for {} with log (< {})'.format(temp_col, threshold1))


sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[1][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[1][0])
ax[1][0].set_title('KDE for {} ({} < and < {})'.format(temp_col, threshold1, threshold2))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][1])
ax[1][1].set_title('KDE for {} with log ({} < and < {})'.format(temp_col, threshold1, threshold2))


sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[2][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold2) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[2][0])
ax[2][0].set_title('KDE for {} (> {})'.format(temp_col, threshold2))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[2][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[2][1])
ax[2][1].set_title('KDE for {} with log (> {})'.format(temp_col, threshold2))


plt.show()


# - In rough way, the repay clients have high CREDIT_SUM_LIMIT.
# - Is it possible to have minus credit sum limit??

# ### <a id='3_5_4'>3.5.4 Amount credit sum overdue</a>

# In[ ]:


temp_col = 'AMT_CREDIT_SUM_OVERDUE'
fig, ax = plt.subplots(3, 2, figsize=(16, 24))
threshold1 = 1e3
threshold2 = 1e5

sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[0][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] < threshold1) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[0][0])
ax[0][0].set_title('KDE for {} (< {})'.format(temp_col, threshold1))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[0][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] < threshold1) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[0][1])
ax[0][1].set_title('KDE for {} with log (< {})'.format(temp_col, threshold1))


sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[1][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[1][0])
ax[1][0].set_title('KDE for {} ({} < and < {})'.format(temp_col, threshold1, threshold2))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[1][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold1) & (application_train[temp_col] < threshold2) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[1][1])
ax[1][1].set_title('KDE for {} with log ({} < and < {})'.format(temp_col, threshold1, threshold2))


sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 0), temp_col].dropna(), label='repay(0)', color='r', ax=ax[2][0])
sns.kdeplot(application_train.loc[(application_train[temp_col] > threshold2) &(application_train['TARGET'] == 1), temp_col].dropna(), label='not repay(1)', color='b', ax=ax[2][0])
ax[2][0].set_title('KDE for {} (> {})'.format(temp_col, threshold2))

sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 0), temp_col].dropna()+0.00001), label='repay(0)', color='r', ax=ax[2][1])
sns.kdeplot(np.log(application_train.loc[(application_train[temp_col] > threshold2) & (application_train['TARGET'] == 1), temp_col].dropna()+0.00001), label='not repay(1)', color='b', ax=ax[2][1])
ax[2][1].set_title('KDE for {} with log (> {})'.format(temp_col, threshold2))


plt.show()


# - Likewise, there are many repay-clients who have large credit sum overdue.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




