#!/usr/bin/env python
# coding: utf-8

# ## Overview of this notebook
# -**Understand the problem.** We'll look at each variable and do a philosophical analysis about their meaning and importance for this problem.
# 
# -**Univariable study.** We'll just focus on the dependent variable and try to know a little bit more about it.
# 
# -**Multivariate study.** We'll try to understand how the dependent variable and independent variables relate.

# ## Part-1 : Understanding the problem statement
# 

# ### **Buisness case: ** 
# * A large U.S. Electrical appliance's retailer has many branches. There is no Fixed Price for a product(for various reasons),the SalesPerson has the freedom to choose the price at which they sell the product. There is no cap on the minimum and maximum quantity of sales on the Salesperson.Due to these reasons the average sale size and average quantity for a transaction varies.The company wants to do a **'Sales and Productivity'** analysis.
# * It is for this reason the company wants to implement a system to classify the reports into one of the Three categories,
#   **Suspicious/Not Suspicious/Indeterminate**.                       
# * The company also wants the Salespersons to be grouped into **HighRisk** or **MediumRisk** or **LowRisk** categories based on the report info provided by them. 
# 

# ### **ML Problem Statement:**
# 1. Build and Develop a Machine Learning Model to **Classify** the Reports submitted by the Salesperson into one of three categories Yes/NO/Indeterminate.
#    The dependent column is **'Suspicious'**.
# 2. **Segment** the Salesperson into one of the three categories HighRisk or MediumRisk or LowRisk categories.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Part-0 : Importing all the necessary Libraries and Loading the Data

# In[ ]:


#import all the necessary libraries here
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.offline as pyoff
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[ ]:


def generate_layout_bar(col_name):
    layout_bar = go.Layout(
        autosize=False, # auto size the graph? use False if you are specifying the height and width
        width=1600, # height of the figure in pixels
        height=600, # height of the figure in pixels
        title = "Distribution of {} column".format(col_name), # title of the figure
        # more granular control on the title font 
        titlefont=dict( 
            family='Courier New, monospace', # font family
            size=14, # size of the font
            color='black' # color of the font
        ),
        # granular control on the axes objects 
        xaxis=dict( 
        tickfont=dict(
            family='Courier New, monospace', # font family
            size=14, # size of ticks displayed on the x axis
            color='black'  # color of the font
            )
        ),
        yaxis=dict(
#         range=[0,100],
            title='Percentage',
            titlefont=dict(
                size=14,
                color='black'
            ),
        tickfont=dict(
            family='Courier New, monospace', # font family
            size=14, # size of ticks displayed on the y axis
            color='black', # color of the font
            tickangle=45
            )
        ),
        font = dict(
            family='Courier New, monospace', # font family
            color = "white",# color of the font
            size = 12 # size of the font displayed on the bar
                )  
        )
    return layout_bar


# In[ ]:


init_notebook_mode(connected=True)


# In[ ]:


data=pd.read_excel("../input/Train.xlsx")


# In[ ]:


datacopy_master=data


# snapshot of  the data

# In[ ]:


data.head()


# In[ ]:


Records=data.shape[0]
Attributes=data.shape[1]
print("Number of Records in dataset = ",Records)
print("Number of Attributes in dataset = ",Attributes)


# ### **Part-2 : Univariate Analysis**
# *First things first*:Analysing **'Suspicious'** column 

# In[ ]:


data['Suspicious'].describe()


# * well it seems that there are 3 unique values in this column
# * the top most occuring value is **'indeterminate'** with a Frequency of **39846**.
# * let's check the frequency of the other two values too.

# In[ ]:


value,count=np.unique(data['Suspicious'],return_counts=True)
percent=(count/Records)*100
print(np.asarray([value,count,percent]).T)


# * the above calculation show's that the dataset has 93.57% of the records with indeterminate level,06.00% of the records with No level,00.42% of the records with Yes level.

# In[ ]:


count


# In[ ]:


data1 = [
    go.Bar(
        x=value, # assign x as the dataframe column 'x'
        y=count,
        text=count,
        textposition='auto'
    )
]

layout = go.Layout(
    autosize=True,
    title='Distribution of the Suspicious column',
    xaxis=dict(title='Suspicious'),
    yaxis=dict(title='Count')
)

fig = go.Figure(data=data1, layout=layout)

# IPython notebook
iplot(fig)


# *let us start to group the records based on salesperson and  find the distribution of each salesperson

# In[ ]:


data.head()


# In[ ]:


#sum of the total sales value
summation_of_total_Sales=data['TotalSalesValue'].sum()
summation_of_total_Sales


# In[ ]:





# In[ ]:


#sum of the total quantity
summation_of_total_Quantity=data['Quantity'].sum()
summation_of_total_Quantity


# In[ ]:


temp=data.groupby(['SalesPersonID']).sum()


# In[ ]:


top_ten_quantity_by_SalesPersonID=temp.sort_values(by='Quantity',ascending=False).head(10)
last_ten_quantity_by_SalesPersonID=temp.sort_values(by='Quantity',ascending=False).tail(10)


# In[ ]:


((top_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100)
((last_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100)


# In[ ]:


data1 = [
    go.Bar(
        x=top_ten_quantity_by_SalesPersonID.index, # assign x as the dataframe column 'x'
        y=((top_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100),
        text=((top_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100),
        textposition='auto'
    )
]

layout = go.Layout(
        autosize=True,
        title='Top performers based on Quantity',
        xaxis=dict(title='SalesPersonID'),
        yaxis=dict(title='Total Quantity')
)

fig = go.Figure(data=data1, layout=layout)

# IPython notebook
iplot(fig)


# In[ ]:


data1 = [
    go.Bar(
        x=last_ten_quantity_by_SalesPersonID.index, # assign x as the dataframe column 'x'
        y=((last_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100),
        text=((last_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100),
        textposition='auto'
    )
]

layout = go.Layout(
        autosize=True,
        title='Under performers based on Quantity',
        xaxis=dict(title='SalesPersonID'),
        yaxis=dict(title='Total Quantity')
)

fig = go.Figure(data=data1, layout=layout)

# IPython notebook
iplot(fig)


# In[ ]:


#top ten guys total contribution by quantity
((top_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100).sum()
#top ten guys contributed 13.948894548010024 percent to the total quantity sold.


# In[ ]:


#bottom ten guys total contribution by quantity
((last_ten_quantity_by_SalesPersonID['Quantity']/summation_of_total_Quantity)*100).sum()
#Bottom ten guys contributed 0.00014734402037043685 percent to the total quantity sold.


# In[ ]:


temp=data.groupby(['SalesPersonID']).sum()


# In[ ]:


temp.head()


# In[ ]:


top_ten_TotalSales_by_SalesPersonID=temp.sort_values(by='TotalSalesValue',ascending=False).head(10)
last_ten_TotalSales_by_SalesPersonID=temp.sort_values(by='TotalSalesValue',ascending=False).tail(10)


# In[ ]:


last_ten_TotalSales_by_SalesPersonID


# In[ ]:


data1 = [
    go.Bar(
        x=top_ten_TotalSales_by_SalesPersonID.index, # assign x as the dataframe column 'x'
        y=((top_ten_TotalSales_by_SalesPersonID['TotalSalesValue']/summation_of_total_Sales)*100),
        text=((top_ten_TotalSales_by_SalesPersonID['TotalSalesValue']/summation_of_total_Sales)*100),
        textposition='auto'
    )
]

layout = go.Layout(
        autosize=True,
        title='Top performers based on TotalSales Percentage',
        xaxis=dict(title='SalesPersonID'),
        yaxis=dict(title='Total Sales')
)

fig = go.Figure(data=data1, layout=layout)

# IPython notebook
iplot(fig)


# In[ ]:


data1 = [
    go.Bar(
        x=last_ten_quantity_by_SalesPersonID.index, # assign x as the dataframe column 'x'
        y=((last_ten_TotalSales_by_SalesPersonID['TotalSalesValue']/summation_of_total_Sales)*100),
        text=((last_ten_TotalSales_by_SalesPersonID['TotalSalesValue']/summation_of_total_Sales)*100),
        textposition='auto'
    )
]

layout = go.Layout(
        autosize=True,
        title='least performers based on TotalSales Percentage',
        xaxis=dict(title='SalesPersonID'),
        yaxis=dict(title='Total Sales by percentage')
)

fig = go.Figure(data=data1, layout=layout)

# IPython notebook
iplot(fig)


# In[ ]:


top_ten_quantity_by_SalesPersonID.head()


# In[ ]:


data.head()


# In[ ]:


temp=data.groupby(['ProductID']).sum()


# In[ ]:


top_ten_quantity_by_ProductID=temp.sort_values(by='Quantity',ascending=False).head(10)
last_ten_quantity_by_ProductID=temp.sort_values(by='Quantity',ascending=False).tail(10)


# In[ ]:


top_ten_quantity_by_ProductID.head()


# In[ ]:


((top_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100)
((last_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100)


# In[ ]:


data1 = [
    go.Bar(
        x=top_ten_quantity_by_ProductID.index, # assign x as the dataframe column 'x'
        y=((top_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100),
        text=((top_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100),
        textposition='auto'
    )
]

layout = go.Layout(
        autosize=True,
        title='Top performers based on productID Quantity',
        xaxis=dict(title='ProductID'),
        yaxis=dict(title='Quantity percentage')
)

fig = go.Figure(data=data1, layout=layout)

# IPython notebook
iplot(fig)


# In[ ]:


sum(((top_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100))


# In[ ]:


data1 = [
    go.Bar(
        x=last_ten_quantity_by_ProductID.index, # assign x as the dataframe column 'x'
        y=((last_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100),
        text=((last_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100),
        textposition='auto'
    )
]

layout = go.Layout(
        autosize=True,
        title='last ten performers based on ProductID Quantity',
        xaxis=dict(title='ProductID'),
        yaxis=dict(title='Quantity percentage')
)

fig = go.Figure(data=data1, layout=layout)

# IPython notebook
iplot(fig)


# In[ ]:


sum(((last_ten_quantity_by_ProductID['Quantity']/summation_of_total_Quantity)*100))


# In[ ]:


top_ten_TotalSales_by_ProductID=temp.sort_values(by='TotalSalesValue',ascending=False).head(10)
last_ten_TotalSales_by_ProductID=temp.sort_values(by='TotalSalesValue',ascending=False).tail(10)


# In[ ]:


((top_ten_TotalSales_by_ProductID['TotalSalesValue']/summation_of_total_Sales)*100)
((last_ten_TotalSales_by_ProductID['TotalSalesValue']/summation_of_total_Sales)*100)


# In[ ]:


data1 = [
    go.Bar(
        x=top_ten_TotalSales_by_ProductID.index, # assign x as the dataframe column 'x'
        y=((top_ten_TotalSales_by_ProductID['TotalSalesValue']/summation_of_total_Sales)*100),
        text=((top_ten_TotalSales_by_ProductID['TotalSalesValue']/summation_of_total_Sales)*100),
        textposition='auto'
    )
]

layout = go.Layout(
        autosize=True,
        title='Top products perform based on TotalSales',
        xaxis=dict(title='Product ID'),
        yaxis=dict(title='Total Sales')
)

fig = go.Figure(data=data1, layout=layout)

# IPython notebook
iplot(fig)


# In[ ]:


sum(((top_ten_TotalSales_by_ProductID['TotalSalesValue']/summation_of_total_Sales)*100))


# In[ ]:


dataframe_Yes=data[data['Suspicious']=='Yes']


# In[ ]:


dataframe_No=data[data['Suspicious']=='No']


# In[ ]:


dataframe_Indeterminate=data[data['Suspicious']=='indeterminate']


# In[ ]:


print(dataframe_Yes.shape)
print(dataframe_No.shape)
print(dataframe_Indeterminate.shape)


# #### ***Feature engineering***
# 
# * Part of our quest is to come up with a a rule(or set of rules) to define the properties of 'Yes','NO'and'Indeterminate' class.Since we could not find any such rule we have to create new Features to better classify the reports
# * As there is no buisness intuition defining the condition for FRAUD we start to first apply our knowledge from statistical background to find new features that better explain our problem statement
# * From the above plot we can find that just by considering the RAW data we cannot classifiy the records. I believe the reason for this is that'Since each salesperson has the Freedom to choose whatever price he can sell a product,and whatever the quantity he can sell there is no pattern we could infer from this.We are not comparing apples to apples!'
# *

# 1. First we have divided the two columns TotalSalesValue by Quantity this will give us the PricePerUnit of that particular ProductID for that SalesPersonID for that particular ReportID

# In[ ]:


data['PricePerUnit']=data.TotalSalesValue/data.Quantity


# #### Here we have started to Groupby SalesPersonID and ProductID and extracted the mean of the Quantity,TotalSalesValue,PricePerUnit column

# In[ ]:


temp=data.groupby(['SalesPersonID','ProductID']).mean()['Quantity']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')


# In[ ]:


#renaming the Quantity_x and Quantity_y columns to Quantity and Average_qty_guy_prdID
data=data.rename(index=str, columns={"Quantity_y": "Average_qty_guy_prdID","Quantity_x":"Quantity"})


# marker 1.............................................................................................................

# In[ ]:


temp=data.groupby(['SalesPersonID','ProductID']).mean()['TotalSalesValue']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')


# In[ ]:


#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and Average_TotalSalesValue_guy_prdID
data=data.rename(index=str, columns={"TotalSalesValue_y": "Average_TotalSalesValue_guy_prdID","TotalSalesValue_x":"TotalSalesValue"})


# marker 2...........................................................................................................

# In[ ]:


temp=data.groupby(['SalesPersonID','ProductID']).mean()['PricePerUnit']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')


# In[ ]:


#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and Average_PricePerUnit_guy_prdID
data=data.rename(index=str, columns={"PricePerUnit_y": "Average_PricePerUnit_guy_prdID","PricePerUnit_x":"PricePerUnit"})


# marker 3.................................................................................

# #### Groupby SalesPersonID and extracting the mean of the Quantity,TotalSalesValue,PricePerUnit column

# In[ ]:


temp=data.groupby(['SalesPersonID']).mean()['Quantity']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID'],how='left')


# In[ ]:


#renaming the Quantity_x and Quantity_y columns to Quantity and Average_qty_guy
data=data.rename(index=str, columns={"Quantity_y": "Average_qty_guy","Quantity_x":"Quantity"})


# marker 1.....................................................................................................

# In[ ]:


temp=data.groupby(['SalesPersonID']).mean()['TotalSalesValue']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID'],how='left')


# In[ ]:


#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and Average_TotalSalesValue_guy
data=data.rename(index=str, columns={"TotalSalesValue_y": "Average_TotalSalesValue_guy","TotalSalesValue_x":"TotalSalesValue"})


# marker 2..........................................................................................................

# In[ ]:


temp=data.groupby(['SalesPersonID']).mean()['PricePerUnit']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID'],how='left')


# In[ ]:


#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and Average_PricePerUnit_guy
data=data.rename(index=str, columns={"PricePerUnit_y": "Average_PricePerUnit_guy","PricePerUnit_x":"PricePerUnit"})


# marker 3..........................................................................................................

# #### Groupby ProductID and extracting the mean of the Quantity,TotalSalesValue,PricePerUnit column

# In[ ]:


temp=data.groupby(['ProductID']).mean()['Quantity']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['ProductID'],how='left')


# In[ ]:


#renaming the Quantity_x and Quantity_y columns to Quantity and Average_qty_prdID
data=data.rename(index=str, columns={"Quantity_y": "Average_qty_prdID","Quantity_x":"Quantity"})


# marker 1.............................................................................................................

# In[ ]:


temp=data.groupby(['ProductID']).mean()['TotalSalesValue']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['ProductID'],how='left')


# In[ ]:


#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and Average_TotalSalesValue_guy_prdID
data=data.rename(index=str, columns={"TotalSalesValue_y": "Average_TotalSalesValue_prdID","TotalSalesValue_x":"TotalSalesValue"})


# marker 2..............................................................................................

# In[ ]:


temp=data.groupby(['ProductID']).mean()['PricePerUnit']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['ProductID'],how='left')


# In[ ]:


#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and Average_PricePerUnit_guy_prdID
data=data.rename(index=str, columns={"PricePerUnit_y": "Average_PricePerUnit_prdID","PricePerUnit_x":"PricePerUnit"})


# marker 3...............................................................

# #### Here we have started to Groupby SalesPersonID and ProductID and extracted the StdDEV of the Quantity,TotalSalesValue,PricePerUnit column

# In[ ]:


temp=data.groupby(['SalesPersonID','ProductID']).std()['Quantity']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')


# In[ ]:


#renaming the Quantity_x and Quantity_y columns to Quantity and Average_qty_guy_prdID
data=data.rename(index=str, columns={"Quantity_y": "std_qty_guy_prdID","Quantity_x":"Quantity"})


# marker 1................................................................................................

# In[ ]:


temp=data.groupby(['SalesPersonID','ProductID']).std()['TotalSalesValue']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')


# In[ ]:


#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and std_totalsalesvalue_guy_prdID
data=data.rename(index=str, columns={"TotalSalesValue_y": "std_TotalSalesValue_guy_prdID","TotalSalesValue_x":"TotalSalesValue"})


# marker 2..................................................................

# In[ ]:


temp=data.groupby(['SalesPersonID','ProductID']).std()['PricePerUnit']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID','ProductID'],how='left')


# In[ ]:


#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and std_PricePerUnit_guy_prdID
data=data.rename(index=str, columns={"PricePerUnit_y": "std_PricePerUnit_guy_prdID","PricePerUnit_x":"PricePerUnit"})


# marker 3...............................................................................................

# #### Groupby SalesPersonID and extracting the std of the Quantity,TotalSalesValue,PricePerUnit column

# In[ ]:


temp=data.groupby(['SalesPersonID']).std()['Quantity']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID'],how='left')


# In[ ]:


#renaming the Quantity_x and Quantity_y columns to Quantity and std_qty_guy
data=data.rename(index=str, columns={"Quantity_y": "std_qty_guy","Quantity_x":"Quantity"})


# marker 1.....................................................................................................

# In[ ]:


temp=data.groupby(['SalesPersonID']).std()['TotalSalesValue']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID'],how='left')


# In[ ]:


#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and std_TotalSalesValue_guy
data=data.rename(index=str, columns={"TotalSalesValue_y": "std_TotalSalesValue_guy","TotalSalesValue_x":"TotalSalesValue"})


# marker 2..............................................................................................

# In[ ]:


temp=data.groupby(['SalesPersonID']).std()['PricePerUnit']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['SalesPersonID'],how='left')


# In[ ]:


#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and std_PricePerUnit_guy
data=data.rename(index=str, columns={"PricePerUnit_y": "std_PricePerUnit_guy","PricePerUnit_x":"PricePerUnit"})


# marker 3.............................................................................................

# #### Groupby ProductID and extracting the std of the Quantity,TotalSalesValue,PricePerUnit column

# In[ ]:


temp=data.groupby(['ProductID']).std()['Quantity']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['ProductID'],how='left')


# In[ ]:


#renaming the Quantity_x and Quantity_y columns to Quantity and std_Quantity_prdID
data=data.rename(index=str, columns={"Quantity_y": "std_Quantity_prdID","Quantity_x":"Quantity"})


# marker 1..................................................................................

# In[ ]:


temp=data.groupby(['ProductID']).std()['TotalSalesValue']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['ProductID'],how='left')


# In[ ]:


#renaming the TotalSalesValue_x and TotalSalesValue_y columns to TotalSalesValue and std_TotalSalesValue_prdID
data=data.rename(index=str, columns={"TotalSalesValue_y": "std_TotalSalesValue_prdID","TotalSalesValue_x":"TotalSalesValue"})


# marker 2...................................................................

# In[ ]:


temp=data.groupby(['ProductID']).std()['PricePerUnit']
temp=pd.DataFrame(temp)
temp=temp.reset_index()


# In[ ]:


data=pd.merge(data,temp,on=['ProductID'],how='left')


# In[ ]:


#renaming the PricePerUnit_x and PricePerUnit_y columns to PricePerUnit and std_PricePerUnit_prdID
data=data.rename(index=str, columns={"PricePerUnit_y": "std_PricePerUnit_prdID","PricePerUnit_x":"PricePerUnit"})


# marker 3........................................................................................................

# In[ ]:


data.head()


# In[ ]:


len(data.columns)


# In[ ]:


feature_engineered_dataSet=data.iloc[ : ,0:7]


# In[ ]:


feature_engineered_dataSet['diff_Average_qty_guy_prdID']=data['Quantity']-data['Average_qty_guy_prdID']
feature_engineered_dataSet['diff_Average_TotalSalesValue_guy_prdID']=data['TotalSalesValue']-data['Average_TotalSalesValue_guy_prdID']
feature_engineered_dataSet['diff_Average_PricePerUnit_guy_prdID']=data['PricePerUnit']-data['Average_PricePerUnit_guy_prdID']


# In[ ]:


feature_engineered_dataSet['diff_Average_qty_guy']=data['Quantity']-data['Average_qty_guy']
feature_engineered_dataSet['diff_Average_TotalSalesValue_guy']=data['TotalSalesValue']-data['Average_TotalSalesValue_guy']
feature_engineered_dataSet['diff_Average_PricePerUnit_guy']=data['PricePerUnit']-data['Average_PricePerUnit_guy']


# In[ ]:


feature_engineered_dataSet['diff_Average_qty_prdID']=data['Quantity']-data['Average_qty_prdID']
feature_engineered_dataSet['diff_Average_TotalSalesValue_prdID']=data['TotalSalesValue']-data['Average_TotalSalesValue_prdID']
feature_engineered_dataSet['diff_Average_PricePerUnit_prdID']=data['PricePerUnit']-data['Average_PricePerUnit_prdID']


# In[ ]:





# In[ ]:


data.head()


# In[ ]:


feature_engineered_dataSet.head()


# In[ ]:


feature_engineered_dataSet.columns


# In[ ]:


data.to_csv('basic_feature_engineering.csv',index=False)


# In[ ]:


feature_engineered_dataSet.to_csv('advanced_feature_engineering_using_mean_iteration_1.csv',index=False)


# In[ ]:


data.head()


# In[ ]:


feature_engineered_dataSet.head()

