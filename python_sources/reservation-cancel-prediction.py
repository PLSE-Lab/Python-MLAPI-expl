#!/usr/bin/env python
# coding: utf-8

# ## Models and Crossvalidation
# Using the follow dataset [Hotel Booking Demand](https://www.kaggle.com/jessemostipak/hotel-booking-demand/tasks), I will try to answer the task, "**Can we predict the possibility of a booking**?", or better, "**Can we predict if a booking will be cancelled or not?**". Starting this notebook with an EDA, Exploratory Data Analysis, and after that, I'm going to test more than one classifier and compare the results. The models will be tested with cross-validation.
# 
# Shall we start? :D

# Starting all the libs and setting warnings...

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


# Load all the dataset

# In[ ]:


hotel_bookings = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
hotel_bookings.head(2)


# ## Exploratory Data Analysis

# As you can see, there is a lot of columns in this dataset and this is the first question, which ones have do I to use?
# I'll ignore the columns with data information such as: `arrival_date_month`, `arrival_date_day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`. I had to keep the `arrival_date_week_number` because this column can preserve the all data information I need about the time of year. During part of the year, tourism is higher than others and this column will help me with that.

# In[ ]:


hotel_bookings.columns


# For this point, I wanna see the value impact in the label, where we correlate, by math with the action 'cancel a book'. Selecting only the numeric value columns, let's plot the matrix.

# In[ ]:


corr = hotel_bookings[
    ['is_repeated_guest', 'previous_cancellations', 
     'previous_bookings_not_canceled', 'booking_changes', 
     'days_in_waiting_list', 'lead_time', 'adults', 
     'children', 'babies','is_canceled']
]

with sns.axes_style("white"):
    table = corr.corr()
    mask = np.zeros_like(table)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(18,7))
    sns.heatmap(table, cmap='Reds', mask=mask, vmax=.3, linewidths=0.5, annot=True,annot_kws={"size": 15})


# The next pairplot with linear regression is another way to see the last graph. And there is some interesting information, for example, we can see who not cancelled a book before has a .42 to be a repeat guest. The lead time is important to a customer not to cancel a book.

# In[ ]:


get_ipython().run_cell_magic('time', '', "sns.pairplot(corr, kind='reg', y_vars='is_canceled', x_vars=['is_repeated_guest', 'previous_cancellations', \n                                                             'previous_bookings_not_canceled', 'booking_changes', \n                                                             'lead_time', 'days_in_waiting_list', 'adults', 'children', \n                                                             'babies'])")


# In this part, I will plot the cancel rate for four columns and see with some categories has a different cancel rate, what means could be a pattern for a specific category.

# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=2, cols=2, shared_yaxes=True)

customer_type = hotel_bookings.groupby(['customer_type']).is_canceled.mean().round(2) * 100
reservation_status = hotel_bookings.groupby(['reservation_status']).is_canceled.mean().round(2) * 100
arrival_date_year = hotel_bookings.groupby(['arrival_date_year']).is_canceled.mean().round(2) * 100
hotel = hotel_bookings.groupby(['hotel']).is_canceled.mean().round(2) * 100

# Plots
fig.add_trace(go.Bar(x=customer_type.index, y=customer_type, text=customer_type, textposition='auto'),1, 1)
fig.add_trace(go.Bar(x=reservation_status.index, y=reservation_status, text=reservation_status, textposition='auto'),1, 2)
fig.add_trace(go.Bar(x=arrival_date_year.index, y=arrival_date_year, text=arrival_date_year, textposition='auto'),2, 1)
fig.add_trace(go.Bar(x=hotel.index, y=hotel, text=hotel, textposition='auto'),2, 2)

fig.update_layout(height=800, width=1000, title_text="Cancel rate by column")

# Update xaxis properties
fig.update_xaxes(title_text="Customer Type", row=1, col=1)
fig.update_xaxes(title_text="Reservation Status", row=1, col=2)
fig.update_xaxes(title_text="Arrival Year", row=2, col=1)
fig.update_xaxes(title_text="Hotel Type", row=2, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Cancel rate in percent (%)", row=1, col=1)
fig.update_yaxes(title_text="Cancel rate in percent (%)", row=1, col=2)
fig.update_yaxes(title_text="Cancel rate in percent (%)", row=2, col=1)
fig.update_yaxes(title_text="Cancel rate in percent (%)", row=2, col=2)

fig.show()


# - Looking at the customer graph, we can see the group costumer has a low probability to cancel if you compare to another client type.
# - The reservation status... to be honest, seems nothing for me.
# - Analysing the years, I can affirm the year is not relevant to cancel, for this dataset guys.
# - The hotel type has a relevant difference rate. A city hotel is higher than a resort hotel. This graph makes sense because the resort hotel is more planned for vacation for example.

# Now let's see with some epoch of this dataset has anomalies.

# In[ ]:


month = {
    'January':'01',
    'February':'02',
    'March':'03',
    'April':'04',
    'May':'05',
    'June':'06',
    'July' :'07',
    'August':'08',
    'September':'09',
    'October' :'10',
    'November':'11',
    'December': '12'
}

def translate(data):
    return month[data]

def plot_groupby(cancel, title, xaxis, yaxis, tt='%{text:.2s}'):
    aux_dfs = []
    for year in [2015,2016,2017]:
        aux_df = pd.DataFrame(cancel.loc[year])
        aux_df.index = [str(year) + '-' + translate(m) for m in aux_df.index]
        aux_dfs.append(aux_df)

    cancel_rate = pd.concat(aux_dfs)
    cancel_rate['epoch'] = cancel_rate.index
    cancel_rate.is_canceled = cancel_rate.is_canceled * 100
    
    fig = px.bar(cancel_rate, y='is_canceled', x='epoch', text='is_canceled')
    
    fig.update_xaxes(title_text=xaxis)
    fig.update_yaxes(title_text=yaxis)
    fig.update_traces(texttemplate=tt, textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(
        height=450,
        title_text=title
    )
    return fig


# In[ ]:


cancel = hotel_bookings.groupby(['arrival_date_year','arrival_date_month']).is_canceled.mean()
plot_groupby(cancel, "Cancel rate per month-year (%)", "Months-Year", "Cancel rate in percent (%)")


# Looking by months-yers, our cancel rate don't show us some `seasonal` event, in November 2015 we have 21% but in November 2016 is 37%. May an isolated event happened in November 2015 and this one not happened in November 2016. Our worst epoch was July 2015, the best was November 2015.

# ## Prediction with 6 models

# As I said before, I will pick this follow columns and encoder the categorical columns. 
# 
# My features are: 
# - hotel 
# - lead_time 
# - arrival_date_week_number 
# - adults
# - children 
# - babies 
# - meal 
# - country 
# - market_segment  
# - distribution_channel
# - is_repeated_guest 
# - previous_cancellations
# - previous_bookings_not_canceled 
# - reserved_room_type
# - assigned_room_type 
# - booking_changes 
# - deposit_type 
# - days_in_waiting_list
# - customer_type 
# - required_car_parking_spaces 
# - total_of_special_requests
# - reservation_status
# 
# My label is:
# - is_canceled

# In[ ]:


dataset = hotel_bookings[[
    'hotel', 
    'lead_time', 
    'arrival_date_week_number', 
    'adults',
    'children', 
    'babies', 
    'meal', 
    'country', 
    'market_segment',  
    'distribution_channel',
    'is_repeated_guest', 
    'previous_cancellations',
    'previous_bookings_not_canceled', 
    'reserved_room_type',
    'assigned_room_type', 
    'booking_changes', 
    'deposit_type', 
    'days_in_waiting_list',
    'customer_type', 
    'required_car_parking_spaces', 
    'total_of_special_requests',
    'reservation_status',
    'is_canceled'
]]


# Let's encoder the categories, and to identify the columns, we can use the `dtypes` and take all the `object` columns.

# In[ ]:


dataset.dtypes


# In[ ]:


dataset.dropna(inplace=True) # We will lose a few rows...
dataset.children = dataset.children.astype('int64') # convert from float to int
types = pd.DataFrame(dataset.dtypes, columns=['type']) # prepare the categorical columns
columns = list(types[types.type == 'object'].index)  # making a list to the 'for' loop

from sklearn.preprocessing import LabelEncoder 

lb_make = LabelEncoder()
for column in columns:
    dataset[column] = lb_make.fit_transform(dataset[column])

dataset.head()


# Our dataset now is ready to be used by predict models.

# In[ ]:


X = dataset[[
    'hotel', 
    'lead_time', 
    'arrival_date_week_number', 
    'adults',
    'children', 
    'babies', 
    'meal', 
    'country', 
    'market_segment',  
    'distribution_channel',
    'is_repeated_guest', 
    'previous_cancellations',
    'previous_bookings_not_canceled', 
    'reserved_room_type',
    'assigned_room_type', 
    'booking_changes', 
    'deposit_type', 
    'days_in_waiting_list',
    'customer_type', 
    'required_car_parking_spaces', 
    'total_of_special_requests',
    'reservation_status',
]]

Y = dataset[['is_canceled']]


# I choose 6 Classifier inside the `sklearn` lib.

# In[ ]:


import time

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

models = [
    # ('SVC', SVC()), # I change my mind, this model is taking too much time
    ('RandomForestClassifier', RandomForestClassifier()),
    ('SGDClassifier', SGDClassifier()),
    ('MLPClassifier', MLPClassifier()),
    ('Tree', DecisionTreeClassifier()),
    ('NearestCentroid', NearestCentroid()),
    ('KNeighborsClassifier', KNeighborsClassifier())
]

def train_test_validation(model, name, X, Y):
    print(f"Starting {name}.") # Debug
    ini = time.time() # Start clock
    scores = cross_val_score(model, X, Y, cv=4) # Cross-validation
    fim = time.time() # Finish clock
    print(f"Finish {name}.") # Debug
    return (name,scores.mean(), scores.max(), scores.min(), fim-ini)


# In[ ]:


get_ipython().run_cell_magic('time', '', "results = [ train_test_validation(model[1], model[0], X, Y) for model in models ] # Testing for all models\nresults = pd.DataFrame(results, columns=['Classifier', 'Mean', 'Max', 'Min', 'TimeSpend (s)']) # Making a data frame")


# Plotting the results and see the best model for this dataset.

# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=2, cols=1, shared_yaxes=True)
x=results['Classifier']
y=round(results['Mean'] * 100,2)
z=round(results['TimeSpend (s)'],2)

# Plots
fig.add_trace(go.Bar(x=x, y=y, text=y, textposition='auto'),1, 1)
fig.add_trace(go.Bar(x=x, y=z, text=z, textposition='auto'),2, 1)

fig.update_layout(height=800, width=1000, title_text="Traing Models for Booking Hotel")

# Update xaxis properties
fig.update_xaxes(title_text="Acurracy by Crossvalidation", row=1, col=1)
fig.update_xaxes(title_text="Time Spended by traing", row=2, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="Accurracy in percent (%)", row=1, col=1)
fig.update_yaxes(title_text="Time in seconds (s)", row=2, col=1)


fig.show()


# The `DecisionTreeClassifier` for me is the best, got 100% accuracy and a sort of time spent to train. `MLPClassifier` is as good as `DecisionTreeClassifier` but spend so much time to train, which makes a bad choice.

# In[ ]:


results


# Now I'll use the `DecisionTreeClassifier` and plot the model.

# In[ ]:


from sklearn.tree import export_graphviz
import graphviz

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20) # Split the dataset

model_t = DecisionTreeClassifier(criterion='entropy')
model_t.fit(X_train, y_train)

y_pred = model_t.predict(X_test)

accuracy = accuracy_score(y_pred, y_test) * 100
print(f'The tree model accuracy was {accuracy} %')

features = X.columns
dot_data = export_graphviz(model_t, out_file=None,
                           filled = True, rounded = True,
                           feature_names = features,
                           class_names = ["no", "yes"])

graphics = graphviz.Source(dot_data)
graphics


# Finishing this notebook answering the question on the task dataset.
# Can we predict the possibility of a booking? Yes, we can.
