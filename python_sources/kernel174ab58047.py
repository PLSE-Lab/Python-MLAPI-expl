#!/usr/bin/env python
# coding: utf-8

# Project to determine **trends and relationships** in a particular supermarket's daily data collection. You can check [here](http://www.kaggle.com/aungpyaeap/supermarket-sales) to download the data. The data is really interesting so, join me as we explore the data to find useful, interesting and salient information from the dataset.
# 1. First the data file in 'csv format' is parsed into pandas.read_csv() to enable the data to be read into the jupyter notebook. 
# 2. Afterwards we create a form of DataFrame(or a table) to show some properties of the features of the data( this is to know the level of cleaning the data requires).

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sales_data = pd.read_csv( '../input/supermarket-sales/supermarket_sales - Sheet1.csv' , parse_dates=['Date', 'Time'])
det = sales_data.count()
details = pd.DataFrame(det)
details.rename(columns={0: 'counts per column'}, inplace= True)
details['dtypes per column']= sales_data.dtypes
details['unique_values']= sales_data.nunique()
details


# From the table, it can be seen that the data does not have any missing values. One can also learn that the data is well structured. The deuctions show that the data does not need any cleaning so we move on to the Exploratory Data Analysis. To help with the analysis, we will use the data to answer some questions that will help us understand the data better. 
# 

# In[ ]:


sales_data['Product line'].unique()


# In[ ]:


sales_data


# *The first question goes thus*
# 
# 1. what productline is in demand for more amongst all the productline

# In[ ]:


#we print out a list of the various categories in the product line
print(list(sales_data['Product line'].unique()))
#write a function to rename the categories, they are too long for plots
def rename(col):
    if col == 'Health and beauty':
        return 'H & B'
    elif col == 'Electronic accessories':
        return 'E'
    elif col == 'Home and lifestyle':
        return 'H & L'
    elif col == 'Sports and travel':
        return 'S & T'
    elif col == 'Food and beverages':
        return 'F & B'
    elif col == 'Fashion accessories':
        return 'F'
sales_data['Product line(abbr)']= sales_data['Product line'].apply(rename)

#now we use the groupby method to group the dataset by Product line
qdata= sales_data.groupby('Product line')['Quantity'].sum().reset_index()
plt.figure(figsize=(12,8))
sns.barplot(x= 'Product line', y= 'Quantity', data= qdata )


# Now, taking a good look at the Barplot, we can see that the Productline that has the highest demand in terms of quantity bought is the **Electronic Accessories**. Another thing that is noteworthy again is that all the product lines are within the same range in terms of their overall demand. Before we draw a conclusion, let's see how the demand varies in the various branches in the supermarkets. 

# In[ ]:


qdata1 = sales_data.groupby(['Product line(abbr)', 'Branch'])['Quantity'].sum().reset_index()
plt.figure(figsize=(12,8))
sns.catplot(x= 'Product line(abbr)', y= 'Quantity', col = 'Branch', kind = 'bar', data= qdata1 )
sales_data.filter(['Product line', 'Product line(abbr)']).drop_duplicates().reset_index(drop = True)


# I was able to divide the plots into the three branches. From the plots, it is evident that the trends vary according to the branches. If we conclude on the information we could retrieve from the previous bar plot where we ignored the branches and analysed the supermarket as a whole, we would be making a huge mistake. These plots show that to really understand or draw salient information from the dataset we should not neglect the categories and we should try study trends by category and see why the trends go in the direction in which they go.
# 
# **INSIGHT FROM BARPLOT**
# 
# 1. The plot,*home and lifestyle* has the highest demand in branch A, *Health and Beauty* have the highest demand in branch B, and for branch C it's *Food and Beverage* . None of the plots look alike in terms of the levels of the bars. Now the question is that, why the difference for each branch?. The data has to give a clue
# 2. we will notice again that no three product lines are the same(in terms of the quantity of the product line that was bought) for the three branches.
# 
# The varying difference in demand in the various branches indicates difference in the general taste and preference of the people in that perimeter. It further indicates difference in the population of predominant personality across the three branches. 
# 
# In the graph, the only column that talks about personality is the *Gender*. Therefore let's see the difference in population of the 'male' and 'female' for each product line in each branch to know why the trend varies for each branch

# In[ ]:


#The X axis will be to cumbersome so we can use the abbr versions of the product line
def frequency(data):
    output = data.count()
    return output
qdata2 = sales_data.groupby(['Product line(abbr)', 'Branch', 'Gender'])['Quantity'].agg([frequency, sum]).reset_index()
qdata2
plt.figure(figsize=(16,10))
for y in [['sum', 'husl'], ['frequency', 'bright']]:
    sns.catplot(x= 'Product line(abbr)', y= y[0], hue = 'Gender', col = 'Branch', kind = 'bar',palette =y[1], data= qdata2 )
sales_data.filter(['Product line', 'Product line(abbr)']).drop_duplicates().reset_index().drop('index', axis =1)


# The top row is a plot between the **Product line** and the Quantity of the product line bought while the second row is a plot between the **Product line** and the **population** of people that bought them. 
# 
# 1. With very few exceptions, we can see that the Quantity has a form of correlation with the population of people that came to buy ( the exceptions can be understood as quantity isn't just dependent on population alone but on the financial capacitiy of buyers as well(which cannot be controlled).
# 
# 2. The barplot was plotted with a hue(male and female), the Product line with the highest demand for branch A happens to be **home and lifestyle**. Notice carefully that for all the branches, the females contribute more to the demand for **home and lifestyle** product line. It can be inferred that, one of the reasons why **Home and lifestyle** is the highest in demand for branch A is that their populated with more females than the rest branches. 
# 
# 3. The case is the same for the **Health and Beauty** also. The males are more than the females for all three branches. Ofcourse it makes a little bit of sense, cause males buy most of the cosmetics and ladies stuff for their women( this is just an evidence). Even as males contribute more to the quantity bought for Health and beauty for all three branches, it is predominant in branch A indicating tha male customers there are generally more than female customers for all three branches.
# 
# 4. There is a little bit of anomaly in the **Food and Beverage** in the Branch A. The other two branches have their customers populated more with females than males, but the reverse is the case with branch A. This shows that other factors affect the demand for product lines apart from the population buying. It could be the preference of the customers, the type of jobs that majority of the males have in the various branches ( the branches are located in different cities so certain factors can differ from branch to branch).
# 
# One beautiful thing the data gives is that, each branch can know the product line that gives brings in more of the income to the branch if the quantity bought affects the total cash inflow.
# *a quick code is done next to show the product line that is responsible for most of the cash in fow for all three branches.*
# 

# In[ ]:


qdata3= sales_data.groupby(['Product line(abbr)', 'Branch'])['Total'].sum().reset_index()
qdata3['Quantity']=sales_data.groupby(['Product line(abbr)', 'Branch'])['Quantity'].sum().reset_index().Quantity
sns.catplot(x= 'Product line(abbr)', y= 'Total', col = 'Branch', kind = 'bar', data= qdata3 )


# In[ ]:


sns.relplot(x= 'Quantity', y= 'Total', col = 'Branch', kind = 'scatter', data= qdata3 )


# As it is, the product lines with the highest demands for the three branches are the same product lines contributing more to the total cash inflow for the three branches. The scatterplot also shows a strong relationship between Quantity bought and the Total cash inflow for all three branches.

# **IS THERE A RELATIONSHIP THAT EXISTS BETWEEN _UNIT PRICE_ AND _QUANTITY_?**

# In[ ]:


sns.scatterplot(x= 'Unit price', y= 'Quantity', data= sales_data)


# From the scatterplot it is safe to say there is not relationship between the _unit price_ and the _Quantity_

# **_PAYMENT TYPE_ VERSUS _TOTAL_ AMOUNT SPENT FOR EACH BUYER?**

# In[ ]:


sns.boxplot(x= 'Payment', y = 'Total', data=sales_data)
sales_data


# In[ ]:


sales_data2= sales_data.groupby('Payment')['Total'].sum()
payment_type= pd.DataFrame(sales_data2)
payment_type['frequency']= sales_data.groupby('Payment').size()
payment_type ['Quantity']= sales_data.groupby('Payment')['Quantity'].sum()
payment_type


# In[ ]:


payment_type.plot(kind='bar', figsize=(12,6))


# From the table/Dataframe **_payment_type_** it is explicit that the total Quantity for each payment type has an obvious relationship with the payment type fequently used the most. Before we draw conclusion we will see how this works with all three branches.
# We can also see that the credit is least used for payment transactions that is if we generalise it. It will make more sense if we see it through the three branches.

# In[ ]:


payment = sales_data.groupby(['Payment', 'Branch'])['Total'].agg([frequency, sum]).reset_index()
payment['Quantity']=  sales_data.groupby(['Payment', 'Branch'])['Quantity'].sum().reset_index().Quantity
payment
sns.catplot(x='Payment', y= 'sum', col= 'Branch', kind ='bar', data = payment, palette='Accent_r')
sns.set_style('whitegrid')
payment


# In[ ]:


for y in ['sum', 'Quantity']:
    sns.relplot(x='frequency', y=y, kind='line', col='Branch', data= payment )


# Isn't this lovely, from the barplot we can see the predominant payment type for all three branches. For branch A it's the **_E wallet users_**, while for branch B it's the **_credit card_** users finally for brach C the people who pay in **_cash_** contribute more to the overall cash inflow. The three branches have their uniqueness.
# 
# For the relationship between the number of people buying with the Total cash inflow as well as the Quantity bought is linear and upward apart from branch B. We cannot bank on the line graph because we do not have enough data to validate our claim.If the supermarket can provide enough data for the remaining months, then we can get a more confident inference.

# **CUSTOMER TYPE ANALYSIS**

# In[ ]:


CL= sales_data.groupby(['Customer type', 'Payment',])['Total'].sum().reset_index()
CL


# In[ ]:


plt.figure(figsize=(10,6), facecolor= 'w', dpi=100)
sns.barplot(x= 'Customer type', y='Total', hue='Payment', data=CL, capsize = 0.1, palette='inferno_r')


# In[ ]:


CL1= sales_data.groupby(['Customer type', 'Payment','Branch'])['Total'].sum().reset_index()
for hue in ['Payment', None]:
    plt.figure(figsize=(10,6), facecolor= 'w', dpi=100)
    sns.catplot(x= 'Customer type', y='Total', hue=hue, col='Branch', kind='bar', data=CL1, capsize = 0.1)


# For all three branches, each customer type(Member, Normal) seem to have slight diffeence in the overall contribution to the cash inflow. So either of the two are important and worthy of attnetion. Breaking them down into the payment types for each customer type, one can see the payment type that is more predominant for all three branches.

# **EXPLORATORY DATA ANALYSIS USING TIME**

# In[ ]:


#First we create three columns, month, day and hour
sales_data['month']= sales_data['Date'].dt.month
sales_data['day']= sales_data['Date'].dt.day
sales_data['hour']= sales_data['Time'].dt.hour
dates= sales_data.sort_values(by= 'Date')
dates


# In[ ]:


plt.figure(figsize=(16,10), facecolor= 'w', dpi=100)
sns.lineplot(x='Date', y='Total', data=dates)


# The total cash inflow fluctuates throughout the month. It will be reasonable to find out what causes the fluctuation. What variable changes that in turn causes the change in the total cash inflow for the supermarkets across the days and months. 
# 
# First let's  see the linechart for each Branch

# In[ ]:


dates1= dates.groupby(['Date', 'Branch'])['Total'].sum().reset_index()
dates1


# In[ ]:


for branch in [['A','Blues' ], ['B', 'bone_r'], ['C', 'Greens']]:
    datedata = dates1[dates1['Branch']== branch[0]]
    plt.figure(figsize=(7,3), facecolor= 'w', edgecolor = 'r', dpi=100)
    plt.xticks(rotation = -45) 
    sns.lineplot(x='Date', y= 'Total', hue = 'Branch', data= datedata, palette= branch[1], ci= None)
    sns.set_style('darkgrid')
    #plt.plot(color=branch[1])


# In[ ]:



sales1= sales_data.groupby(['month', 'Branch'])['Total'].sum().reset_index()
sales2 = sales_data.groupby(['month', 'Branch'])['Total'].count().reset_index()

sales1['count']= sales2['Total']
sales1


# In[ ]:


time_data=sales_data.groupby(['day', 'month', 'Branch'])['Total'].sum().reset_index()
time_data['People/day']= sales_data.groupby(['day', 'month', 'Branch'])['Total'].count().reset_index().Total
time_data


# In[ ]:


number =[1,2,3]
for month in number:
    data = time_data[time_data['month']== month]
    plt.figure(figsize=(10,3), facecolor= 'w', dpi=100)
    sns.relplot(x='day', y='Total', kind='line', col= 'Branch', col_wrap=3, data=data)
    


# In[ ]:


number =[1,2,3]
for month in number:
    data = time_data[time_data['month']== month]
    plt.figure(figsize=(10,3), facecolor= 'w', dpi=100)
    sns.relplot(x='People/day', y='Total', kind='scatter', col='Branch', col_wrap=3,data=data)


# In[ ]:


time_data['quantity']=sales_data.groupby(['day', 'month', 'Branch'])['Quantity'].sum().reset_index().Quantity
time_data.corr()['Total']


# The Variables that have the strongest relationship with the Total  amount for goods bought are the quantity bought and the People/daty that come to ourchase one product line or the other.
# 
# This indicates that when there is a drop in the time graph, the people that came to buy that day as well as the quantity bought dropped. The quantity brought can be made steady if the number of people that come in per day are steady.
# 
# If the data covers several months and years, we could see how the maximum number of people that buy per day increase across the years and the increment in th total goods bought also. That way, our insight can be validated more.

# In[ ]:




