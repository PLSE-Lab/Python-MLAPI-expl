#!/usr/bin/env python
# coding: utf-8

# # Hi,
# ** in this kernel, I tried to visualize relations between the cryptocurrencies. I hope you like it. Enjoy this kernel and if you have a comment, feel free to do that. **

# ### Let us begin
# At first, I import all our libraries and print our files in **input** folder.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly
from plotly.offline import iplot
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from IPython.display import clear_output
import random
import time
import os
print(os.listdir("../input"))


# I learned that they are all csv(comma seperated values) files. Now I can save this list of file names to **csv_list**. Then cut them untill their last 4 chars which are **.csv** and save them into **data_list** as strings. I planned to point all datas using **data_list** so I am going to copy the **data_list** to **data_str_list** and  keep data names in **data_str_list** list. Otherwise the data names can be hard to reach later. I will create a for loop to read csv and assign **a data_list item** which is actually the file name

# In[ ]:


csv_list = os.listdir("../input")
data_list = [x[:-4] for x in csv_list]
data_str_list = data_list.copy()
for i in range(0,len(csv_list)):
    temp = "../input/" + csv_list[i]
    data_list[i] = pd.read_csv(temp)


# When I anlyzed the file names, I saw that they includes **2 special key:["dataset","price"]**. I am going to create a printer function which takes a string as an argument. Searches for items include that string in **data_str_list**. Print the item and *the columns of data at same order in* **data_list** so I can able to see columns of same datasets sequentially.

# In[ ]:


def printer(string):
    for i in range(0,len(data_list)):
        if data_str_list[i].count(string) > 0:
            print(data_str_list[i],"\n",data_list[i].columns,"\n")

printer("dataset")
printer("price")


# If I add a column keeps data name, I can able to merge the price datasets. Because they all have same columns and work on one dataframe is noticeably easier.
# So I will keep price datas in another list(**price_datas**) and add them a name column that includes their data name but I dont need to keep the **_price** string. As above I am going to use same method, I will only give its string untill the last 6 char. I noticed that a column name has **space**, we can reaplace the spaces with **_**. 

# In[ ]:


price_datas=[]
for i in range(0,len(data_list)):
    if data_str_list[i].count("price") > 0:
        data_list[i]["Name"] = data_str_list[i][:-6]
        data_list[i].columns = [col.replace(' ', '_') for col in data_list[i].columns]
        #data_list[i].Date = pd.to_datetime(price_datas[i].Date, infer_datetime_format=True)
        price_datas.append(data_list[i])

price_datas[0].columns


# We could also fix the date type but i wanted to show what i mean by fixing. So I made it comment line. Now lets look at the type of a sample price datas date columns first item (sorry for this annoying long sentence :D)

# In[ ]:


print("type of dates:\t",type(price_datas[4].Date[0]))
price_datas[4].head()


# As we can see, this is a *string* variable and it is hard to work with it because it is hard to classify it by month,by years,by days etc.
# We can convert it to *timestamp* and working with it can be easier

# In[ ]:


for i in range(0,len(price_datas)):
    price_datas[i].Date = pd.to_datetime(price_datas[i].Date, infer_datetime_format=True)


# In[ ]:


print("type of dates:\t",type(price_datas[4].Date[0]))
price_datas[4].head()


# As can be seen from above we fixed Date type.  I think this format is a lot better. Now it is time to concatenate the price datas. We can name it "**price_comp_list**".

# In[ ]:


price_comp_list = pd.concat(price_datas,ignore_index=True)
price_comp_list.head()


# lets look its info

# In[ ]:


price_comp_list.info()


# the bigger problem makes better performance :D. As we can see, the **Volume** and the **Market_Cap** columns keep **string(object)** values. We should convert them to *integer* or *float* to work with. But before doing this, we have to remove the commas from them. They are only for making reading easier. Otherwise we cant convert it.

# In[ ]:


#price_comp_list.Market_Cap.str.replace(",","").astype(int)


# The above code fails because the data actually contains **NaN** objects but there is a "**-**" not a **Null**. Lets give some try for making it more clear. I will copy the data to another dataframe, because I don't want to distrupt my data. We can name it **ClearData** and drop the **NaN** rows on it

# In[ ]:


ClearData = price_comp_list.copy()
ClearData = ClearData[ClearData.Market_Cap != "-"]
ClearData.Market_Cap = ClearData.Market_Cap.str.replace(",","").astype(int)
ClearData = ClearData[ClearData.Volume != "-"]
ClearData.Volume = ClearData.Volume.str.replace(",","").astype(int)
ClearData.head()


# Now, we can see that whether we are clear in reality or not

# In[ ]:


ClearData.info()


# it seems OK for now. Lets try to see the mean values of every cryptocurrecy but before doing that, I dont like the *scientific notation* so I will use the format that I can understand.

# In[ ]:


pd.options.display.float_format = '{:,.2f}'.format
MeanValues = ClearData.groupby("Name").mean()
MeanValues


# everybody loves piecharts. I love plotly

# In[ ]:


def PlotPieChart(Name,label,value):
    trace = go.Pie(labels=label, values=value)
    
    data = [trace]
    layout = dict(title = str(Name))
    fig = dict(data=data, layout=layout)
    iplot(fig)


# I wrote my piechart as a function. The Market Capitalizations and the Volumes piechart should be as below

# In[ ]:


PlotPieChart("Cryptocurrency MarketCaps",MeanValues.index,MeanValues.Market_Cap)
PlotPieChart("Cryptocurrency Volumes",MeanValues.index,MeanValues.Volume)


# We can also the see cryptocurrencies low and high values separately. I will use *plotly* again. Of course as a function.

# In[ ]:


def traceGraph(CryptocurrencyName):
    ClearCrypto = ClearData[ClearData.Name == CryptocurrencyName].set_index("Date")
    
    trace_high = go.Scatter(x=list(ClearCrypto.index),
                            y=list(ClearCrypto.High),
                            name='High',
                            line=dict(color='#33CFA5'))

    trace_high_avg = go.Scatter(x=list(ClearCrypto.index),
                                y=[ClearCrypto.High.mean()]*len(ClearCrypto.index),
                                name='High Average',
                                visible=False,
                                line=dict(color='#33CFA5', dash='dash'))

    trace_low = go.Scatter(x=list(ClearCrypto.index),
                           y=list(ClearCrypto.Low),
                           name='Low',
                           line=dict(color='#F06A6A'))

    trace_low_avg = go.Scatter(x=list(ClearCrypto.index),
                               y=[ClearCrypto.Low.mean()]*len(ClearCrypto.index),
                               name='Low Average',
                               visible=False,
                               line=dict(color='#F06A6A', dash='dot'))

    data = [trace_high, trace_high_avg, trace_low, trace_low_avg]

    high_annotations=[dict(y=ClearCrypto.High.mean(),
                           text='High Average:<br>'+str(ClearCrypto.High.mean()),
                           ax=0, ay=-50),
                      dict(x=ClearCrypto.High.idxmax(),
                           y=ClearCrypto.High.max(),
                           text='High Max:<br>'+str(ClearCrypto.High.max()),
                           ax=0, ay=-50)]
    low_annotations=[dict(y=ClearCrypto.Low.mean(),
                          text='Low Average:<br>'+str(ClearCrypto.Low.mean()),
                          ax=0, ay=50),
                     dict(x=ClearCrypto.High.idxmin(),
                          y=ClearCrypto.Low.min(),
                          text='Low Min:<br>'+str(ClearCrypto.Low.min()),
                          ax=0, ay=50)]

    updatemenus = list([
        dict(type="buttons",
             active=-1,
             buttons=list([
                dict(label = 'High',
                     method = 'update',
                     args = [{'visible': [True, True, False, False]},
                             {'title': CryptocurrencyName.capitalize() + ' High',
                              'annotations': high_annotations}]),
                dict(label = 'Low',
                     method = 'update',
                     args = [{'visible': [False, False, True, True]},
                             {'title': CryptocurrencyName.capitalize() + ' Low',
                              'annotations': low_annotations}]),
                dict(label = 'Both',
                     method = 'update',
                     args = [{'visible': [True, True, True, True]},
                             {'title': CryptocurrencyName.capitalize() + ' Both',
                              'annotations': high_annotations+low_annotations}]),
                dict(label = 'Reset',
                     method = 'update',
                     args = [{'visible': [True, False, True, False]},
                             {'title': CryptocurrencyName.capitalize(),
                              'annotations': []}])
            ]),
        )
    ])

    layout = dict(title=CryptocurrencyName.capitalize(), showlegend=False,
                  updatemenus=updatemenus)

    fig = dict(data=data, layout=layout)
    iplot(fig)


# In[ ]:


traceGraph("bitcoin")


# Come on, this is  enjoyful. I will show you why I am writing them as functions. If you want to try it, just remove "**#**"s. It gives you different random cryptocurrencies graphs every 10 second.

# In[ ]:


#while True:
#    traceGraph(MeanValues.index[random.randint(0,len(MeanValues.index))])
#    time.sleep(10)
#    clear_output(wait=True)


# As an another aspect we can see the coin market capitalizations in a 3d graph. Again a function:

# In[ ]:


def MarketCapGraph(currencyList):
    gf = ClearData.groupby('Name')
    data = []

    for currency in currencyList[::-1]:
        group = gf.get_group(currency)
        dates = group['Date'].tolist()
        date_count = len(dates)
        marketCap = group['Market_Cap'].tolist()
        zeros = [0] * date_count

        data.append(dict(
            type='scatter3d',
            mode='lines',
            x=dates + dates[::-1] + [dates[0]],  # year loop: in incr. order then in decr. order then years[0]
            y=[currency] * date_count,
            z=marketCap + zeros + [marketCap[0]],
            name=currency,
            line=dict(
                width=4
            ),
        ))

    layout = dict(
        title='Cryptocurrencies Market Capitalizations',
        scene=dict(
            xaxis=dict(title='Dates'),
            yaxis=dict(title='Cryptocurrencies'),
            zaxis=dict(title='Market Capitalizations'),
            camera=dict(
                eye=dict(x=-1.7, y=-1.7, z=0.5)
            )
        )
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)


# Lets see the best 5 currency on market capitalizations graph first
# 
# **Note:** Use right click, left click and mouse wheel to walk around

# In[ ]:


MarketCapGraph(MeanValues.sort_values(by=['Market_Cap'],ascending=False).head(5).index)


# But we can also plot all currencies

# In[ ]:


MarketCapGraph(MeanValues.index)


# # Thank you for reading, hope to see you again.
