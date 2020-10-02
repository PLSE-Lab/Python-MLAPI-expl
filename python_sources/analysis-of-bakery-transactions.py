#!/usr/bin/env python
# coding: utf-8

# # **Analysis of Bakery Transactions**
# 
# ### Hello, In this kernel I try to find best  item combinations. 
# <a id="0"></a> <br>
# #### **Content:**
# 1. [Load and Exploring the Data](#1)
# 2. [Cleaning the Data](#2)
# 3. [Find most popular Items](#3)
# 4. [Find the best item combinations with coffee](#4)
# 5. [Prediction of Coffee Sales Per Day](#5)
# 

# <a id="1"></a> <br>
# ## 1-Exploring the Data [^](#0)
# This dataset contains more than 6000 transactions and the date and time data's of these transactions. There is, of course, the item list that sold. So the list of problems is here that we can maybe find a solution.
# * 1 - Maybe we can find the most popular item combination bought by people.
# * 2 - We have the date information so maybe we can use RNN to predict the item sales per day.
# * 3 - I will add more problem here if I found.
# 
# Okay let's start with load the data.

# In[ ]:


# Import libraries and load data with pandas as a dataframe
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

data = pd.read_csv('../input/BreadBasket_DMS.csv')


# In[ ]:


# .info() is good for first look.
data.info()


# We have 4 column.

# In[ ]:


# .head() and .tail() will show first and last 10 items in dataframe.
data.head(10)


# In[ ]:


data.tail(10)


# We have 21293 entries in nearly 6-month time interval. 9684 are at a different time.
# So let's start with to find unique items in our data.

# In[ ]:


# After use .unique() we use len() to find how many unique items that we have.
items_unique_list = data["Item"].unique()
len(items_unique_list)


# We can clearly see there are 95 items has been found in dataset however there can be some *NaN* values int data that we need to clear. To do this we create a basic *NaN* word list and then clear the data.

# <a id="2"></a> <br>
# ## 2-Cleaning the Data [^](#0)
# Our data include some missing values. I want to use .dropna() function to drop them but their type is object(string) and they are not *NaN* they are *None*. So I found different solution to get rid of this problem. I construct a word list of possible words for missing values. Then check my data with this list. 

# In[ ]:


# Here my little list :D
word_list = ["NaN", "-", "nan", "NAN", "None", "NONE", "none", " ", "_", "."]

# I use the list comprehension to make this code smaller.
found_words = [word_list[i] for i, c in enumerate([w in items_unique_list for w in word_list]) if c == True]

# Found word types is 1 so only one of thing in my list founded in our data ("NONE")
len(found_words)


#  So how many of them are "NONE" ?

# In[ ]:


len(data[data["Item"] == "NONE"])


# In[ ]:


# Data include 786 missing values let's drop them
for f in found_words:
    data = data.replace(to_replace=f, value=np.nan).dropna()


# In[ ]:


# Let's look again unique Item list it must be 94.
items_unique_list = data["Item"].unique()
len(items_unique_list)


# Data is cleared and now the item number is 94. Let's look at the top items that bought most. We will need some visualization.
# I am going to use plotly beacuse I love it <3

# <a id="3"></a> <br>
# ## 3-Find most popular Items  [^](#0)
# Start with import some libraries then we will do some data transform for prepare data for visualizing.

# In[ ]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly

plotly.offline.init_notebook_mode(connected=True)


# Let's get the top ten most bought items. Then sum and add the other items as "Others" in our top list.

# In[ ]:


# Get first 10 items from list
hot_items = data.Item.value_counts()[:10]

# Find and sum the remaining items and label it as "Others".
other_items_count = data.Item.count() - hot_items.sum()

# Add two of them in one series.
item_list = hot_items.append(pd.Series([other_items_count], index=["Others"]))

# Here the item list. Yes I like coffee too.
item_list


# Okay, finally we can start to render some graphs.

# In[ ]:


# Values include the list you see above.
values = item_list.tolist()
# Labels include top ten items name.
labels = item_list.index.values.tolist()

# Pie is suitable I think
fig = {
  "data": [
    {
      "values": values,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Top 10 Items",
      "hoverinfo":"label+percent+name",
      "type": "pie"
    },],
  "layout": {
        "title":"Top 10 Most Popular Items",
    }
}
iplot(fig)


# Yes, coffee is the most popular and the bread is second. So let's try to find our first problems answer. 
# 
# **Do people like them two as a combine or separately?**
# <a id="4"></a> <br>
# ## 4-Find the best item combinations with coffee [^](#0)
# 
# To find this out firstly we need to get only transactions that include coffee.

# In[ ]:


# We get all transaction numbers which contain coffee.
coffee_transaction_list = data[data['Item'] == "Coffee"]["Transaction"].tolist()


# In[ ]:


# Then copy our data to protect it.
data_copy = data.copy(deep=True)


# In[ ]:


# And drop all transactions which not contain coffee.
# I use this method based on a suggestion in comments.(Thanks Sam'ir)
data_copy = data_copy[data_copy['Transaction'].isin(coffee_transaction_list)]
# Old way here ->
'''
# Note: Please comment if is there a more efficient way to do this beacuse I think this is not a good way to do this.
for i in range(max(data["Transaction"])+1):
    if i not in coffee_transaction_list:
        data_copy = data_copy.drop(data_copy[data_copy.Transaction == i].index)
'''


# In[ ]:


# Lastly, we get our precious dataframe
data_copy.head(15)


# Now we have a data frame which only includes transactions that contain coffee. Let's look at our top items again with this data frame. Then we can compare them.

# In[ ]:


# We get top ten items from two different data frame.
hot_items_coffe_combine = data_copy.Item.value_counts()[:10]
hot_items = data.Item.value_counts()[:10]

# We need to drop coffee values beacuse we don't need it when we compare items with coffee or without coffee
hot_items_coffe_combine = hot_items_coffe_combine.drop(labels=["Coffee"])
hot_items = hot_items.drop(labels=["Coffee"])

# Labels are Item names
labels = hot_items_coffe_combine.index.values.tolist()

# And values are just values :/ 
values_coffe_combine = hot_items_coffe_combine.tolist()
values = hot_items.tolist()

# First time when I wrote this kernel I made a critical mistake.
# We need to subtract values_coffe_combine from values to get values_without_coffee.
# I forgot to do this step.
values_without_coffee = [values[i]-v for i,v in enumerate(values_coffe_combine)]

df = pd.DataFrame({'with_coffee':values_coffe_combine, 'without_coffee':values_without_coffee})
df

# We have 9 items (The coffee is dropped)
# Bread           
# Tea             
# Cake            
# Pastry          
# Sandwich        
# Medialuna       
# Hot chocolate   
# Cookies         
# Brownie 


# In[ ]:


# To plot a graph we transform it to list
values_list = df.values.tolist()

# We added ratio (with_coffee / without_coffee) to all labels
for index, value in enumerate(values_list):
    labels[index] = labels[index] + "  %{:.2f}".format(value[0] / value[1])

# Here our values
labels


# This graph code is too long I know but this is the best graph for this job I think.

# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go

top_labels = ["Bought With Coffee", "Bought Without Coffee"]

colors = ['rgba(38, 24, 74, 0.8)', 'rgba(190, 192, 213, 1)']

x_data = values_list

y_data = labels

traces = []

for i in range(0, len(x_data[0])):
    for xd, yd in zip(x_data, y_data):
        traces.append(go.Bar(
            x=[xd[i]],
            y=[yd],
            orientation='h',
            marker=dict(
                color=colors[i],
                line=dict(
                        color='rgb(248, 248, 249)',
                        width=1)
            )
        ))

layout = go.Layout(
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        domain=[0.15, 1]
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
    ),
    barmode='stack',
    title='Sales Information of Top Ten Items With/Without Coffee ',
    paper_bgcolor='rgb(248, 248, 248)',
    plot_bgcolor='rgb(248, 248, 255)',
    margin=dict(
        l=120,
        r=10,
        t=140,
        b=80
    ),
    showlegend=False,
)

annotations = []

for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.14, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=14,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first percentage of each bar (x_axis)
    annotations.append(dict(xref='x', yref='y',
                            x=xd[0] / 2, y=yd,
                            text=str(xd[0]),
                            font=dict(family='Arial', size=14,
                                      color='rgb(248, 248, 255)'),
                            showarrow=False))
    # labeling the first Likert scale (on the top)
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper',
                                x=xd[0] / 2, y=1.1,
                                text=top_labels[0],
                                font=dict(family='Arial', size=16,
                                          color='rgba(38, 24, 74, 0.8)'),
                                showarrow=False))
    space = xd[0]
    for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i]/2), y=yd, 
                                    text=str(xd[i]),
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i]/0.2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=16,
                                                  color='rgba(190, 192, 213, 1)'),
                                        showarrow=False))
            space += xd[i]

layout['annotations'] = annotations

fig = go.Figure(data=traces, layout=layout)
iplot(fig, filename='bar-colorscale')


# We can understand from this graph that people don't like bread + coffee combine. But coffee + toast seems very liked. The medialuna + coffee combine is our second popular combine. This bakery maybe make some discounts for this combines : )

# Okay, let's go for our second problem.
# <a id="5"></a> <br>
# ## 5-Prediction of Coffee Sales Per Day [^](#0)

# In this data, we have time and date information for all transaction so I think we can construct a model that predict the coffee sales per day. Let's try it.

# In[ ]:


# Remeber that I have data_copy data frame which only 
# includes transactions that contain coffee.
data_copy.head(6)


# I will delete all other items except coffee

# In[ ]:


data_copy = data_copy[data_copy['Item'] == "Coffee"].drop(['Transaction'], axis=1)
data_copy.head()


# Now we can count the days in order to found coffee sales per day.

# In[ ]:


# Groupby will find the group Date column by size 
date_count_df = data_copy.groupby(["Date"]).size().reset_index(name="Coffee")
date_count_df.head()


# Okay, let's split our data to train and test parts. The test size will be 0.2 of all data.

# In[ ]:


data_len = len(date_count_df.Coffee)

test = date_count_df.Coffee.loc[data_len*0.8:]
train = date_count_df.Coffee.loc[:data_len*0.8]

print(train.shape, test.shape)


# In[ ]:


train.describe()


# Our train data have 127 day's coffe sales information. Average sales per day = 34.7 cup coffee.

# In[ ]:


# Let's add some visualize I will use matplotlib
import matplotlib.pyplot as plt

# Set graph size
plt.figure(figsize=(16,4))

# Add vertical lines in every 7 days.
for xc in np.arange(0, len(train), step=7):
    plt.axvline(x=xc, color='k', linestyle='--')

# Graph's Y axis will be train(Coffe Sales) and X axis will be (lenght(train) / 7) (Every 7 day)
plt.plot(train)
plt.xticks(np.arange(0, len(train), step=7))
plt.xlabel('Days')
plt.ylabel('Coffee Sale')
plt.show()


# In this graph, we can see there is a peak in every 7 days. I think this because of the weekends.
# If we want we can learn it but we don't need to find it for our model.

# Let's return to build model. We need to normalize our sales to (0,1) range.

# In[ ]:


# We reshape train (127,) to (127, 1)
train_scaled = train.values.reshape(-1,1)

# Add Sklearn MinMaxScaler library
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train_scaled)
train_scaled[:5]


# We create X_train and y_train datas with timesteps = 7 for feed our LSTM model. 
# 
# I will try to show it with an example list. More detail is here -> [link](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
# 
# ![](http://i65.tinypic.com/309hg7b.png)

# In[ ]:


X_train = []
y_train = []
timesteps = 7
for i in range(timesteps, len(train)):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

print(X_train.shape, y_train.shape)


# In[ ]:


# We reshape X_train (122, 7) to (122, 7, 1) 
print(X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape


#  **LSTM model**

# In[ ]:


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(40, input_shape=(timesteps, 1))) # 40 LSTM layer
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=300, batch_size=6, verbose=2)


# In[ ]:


# We must prepare "inputs" to create "X_train". This "inputs" contain "test" and last 7 items of "train".
# So why we add last 7 items of "train" to beginning of the "inputs"?
# Beacuse when we use "train" to create "X_train", we couldn't use last 7 item
# to create new array step.
# So our "inputs" size equal to (lenght of "test" + "timestep")
# When we start to create "X_test" we will see "X_test" size will be ("inputs" size - 7)
# because again we won't use last 7 item of "inputs".

total = pd.concat((train, test), axis = 0)
inputs = total[len(total) - len(test) - timesteps:].values.reshape(-1,1)
# Lastly we normalize our inputs data.
inputs = scaler.transform(inputs) 

print("Inputs shape ->",inputs.shape, "  Test shape ->", test.shape)


# In[ ]:


# And we will create X_test data from inputs data.
# Timestep is same

X_test = []
for i in range(timesteps, len(inputs)):
    X_test.append(inputs[i-timesteps:i, 0])

print(len(X_test)) # As you can see "X_test" size equal to "test" size not to "inputs" size


# In[ ]:


# We turn it list to np.array
X_test = np.array(X_test)

# Then transform it's shape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# We give X_test to our model to predict predicted_coffe_sales
predicted_coffe_sales = model.predict(X_test)
# Inverse transform will transform it from (0,1) to it's original range
predicted_coffe_sales = scaler.inverse_transform(predicted_coffe_sales)

# We will compare Predicted Coffe Sales and Real Coffee Sales
real_coffe_sales = test.values

plt.plot(real_coffe_sales, color = 'red', label = 'Real Coffe Sales')
plt.plot(predicted_coffe_sales, color = 'blue', label = 'Predicted Coffe Sales')
plt.title('Coffee Sales Prediction For Last Month')
plt.xlabel('Days')
plt.ylabel('Coffee Sale per Day')
plt.legend()
plt.show()


# As you can see our model is not good enough. Next update I will try to make it better. Please for comment any suggestion. Thank you.
