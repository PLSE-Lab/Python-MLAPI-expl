#!/usr/bin/env python
# coding: utf-8

# # Hello : )
# ## Welcome to my notebook
# ### By Reading my notebook you will get . . .
# 
#     - The roughly analysis of this dataset
#     - Simple Data Cleaning technique
#     - Some statistic technique to deal with each datatype
#     - Some colorful visualize 
#     - Good start point if you want to play this dataset
#     - My Code (I hope it clean enough to use in other dataset)
#     - My thanks :)
#     

# ## Feel free to <font color=deepskyblue> FORK  </font> this notebook, Please  <font color=deepskyblue> UPVOTE !! </font> if it's helpful to you  <font color=deepskyblue> : ) </font>

# ## Ready ? Let's go !!

# ![imglink](https://flightitineraryforvisa.com/wp-content/uploads/2018/12/Hotel-Booking-1280x720.jpg)
# 
# (Image taken from [imglink](https://flightitineraryforvisa.com/wp-content/uploads/2018/12/Hotel-Booking-1280x720.jpg))

# ## Import some useful libraries

# In[ ]:


import pandas as pd    ## library for playing with dataframe
import numpy as np    ## library for dealing with array & numeric value
import matplotlib.pyplot as plt   ## Visualize library
import seaborn as sns 

## make notebook more clean by not show the warning
import warnings
warnings.filterwarnings("ignore")

## make dataframe show only 2 digits float 
pd.options.display.float_format = '{:.2f}'.format


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read the dataset

# In[ ]:


data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')


# In[ ]:


data.head()


# In[ ]:


data.columns


# # Visualization

# ## Hotel

# In[ ]:


plt.rcParams['figure.figsize'] = 10,10
labels = data['hotel'].value_counts().index.tolist()
sizes = data['hotel'].value_counts().tolist()
explode = (0, 0.2)
colors = ['indianred','khaki']

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=False, startangle=30)
plt.axis('equal')
plt.tight_layout()
plt.title("How many type of hotel in this dataset", fontdict=None, position= [0.48,1], size = 'xx-large')
plt.show()


# ## lead time
# 
#     Number of days that elapsed between the entering date of the booking into the PMS and the arrival date

# In[ ]:


plt.rcParams['figure.figsize'] = 15,6
plt.hist(data['lead_time'].dropna(), bins=30,color = 'paleturquoise' )

plt.ylabel('Count')
plt.xlabel('Time (days)')
plt.title("Lead time distribution ", fontdict=None, position= [0.48,1.05], size = 'xx-large')
plt.show()


# ## is_canceled
# 
#     Value indicating if the booking was canceled (1) or not (0)

# In[ ]:


plt.rcParams['figure.figsize'] = 15,8

height = data['is_canceled'].value_counts().tolist()
bars =  ['Not Cancel','Cancel']
y_pos = np.arange(len(bars))
color = ['lightgreen','salmon']
plt.bar(y_pos, height , width=0.7 ,color= color)
plt.xticks(y_pos, bars)
plt.xticks(rotation=90)
plt.title("How many booking was cancel", fontdict=None, position= [0.48,1.05], size = 'xx-large')
plt.show()


# ## Stays in Week Night vs Weekend Night
# 
#     Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
#     
#     Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel

# In[ ]:


plt.rcParams['figure.figsize'] = 15,6

plt.hist(data['stays_in_week_nights'][data['stays_in_week_nights'] < 10].dropna(), 
         bins=8,alpha = 1,color = 'lemonchiffon',label='Stays in week night' )

plt.hist(data['stays_in_weekend_nights'][data['stays_in_weekend_nights'] < 10].dropna(),
         bins=8, alpha = 0.5,color = 'blueviolet',label='Stays in weekend night' )

plt.ylabel('Count')
plt.xlabel('Time (days)')
plt.title("Stays in Week Night vs Weekend Night ", fontdict=None, position= [0.48,1.05], size = 'xx-large')
plt.legend(loc='upper right')
plt.show()


# ## Agent
# 
#     ID of the travel agency that made the booking

# In[ ]:



plt.rcParams['figure.figsize'] =10,10
sizes = data['agent'].value_counts()[:8].tolist() + [len(data) - sum(data['agent'].value_counts()[:8].tolist())]
labels = ["Agent " + str(string) for string in data['agent'].value_counts()[:8].index.tolist()] + ["Other"]

explode = (0.18,0.11,0.12,0,0,0,0,0,0,0,0)
colors =  ['royalblue','mediumaquamarine','moccasin'] +['linen']*7 + ['oldlace']

plt.pie(sizes, explode = explode, colors = colors ,labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=96)
plt.axis('equal')
plt.tight_layout()
plt.title("Who is the best agent", fontdict=None, position= [0.5,1], size = 'xx-large')

plt.show()


# ## ADR
# 
#     Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights

# In[ ]:


print("The highest value is : ", data['adr'].max())
print("The lowest value is : ", data['adr'].min())


#     Seem like it has large gap between the highest value and the lowest,
#     let ignore outlier first   : )
#     I will use the simple remove outlier technique such as 1.5IQR
# [Dealing with Outlier](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)

# In[ ]:


Q1 = data['adr'].quantile(0.25)
Q3 = data['adr'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = (Q1 - 1.5 * IQR)
upper_bound = (Q3 + 1.5 * IQR)


# In[ ]:


without_outlier = data[(data['adr'] > lower_bound ) & (data['adr'] < upper_bound)]


# In[ ]:


plt.boxplot(without_outlier['adr'],  notch=True,  # notch shape
                         patch_artist=True,
                   boxprops=dict(facecolor="sandybrown", color="black"),)
plt.ylabel('ADR')
plt.title("Box plot for Average Daily Rate ", fontdict=None, position= [0.48,1.05], size = 'xx-large')

plt.show()


# ## Reserve Room Type

#     Code of room type reserved. Code is presented instead of designation for anonymity reasons.

# In[ ]:


plt.rcParams['figure.figsize'] = 15,8

height = data['reserved_room_type'].value_counts().tolist()
bars =  data['reserved_room_type'].value_counts().index.tolist()
y_pos = np.arange(len(bars))
color= ['c']+['paleturquoise']*10
plt.bar(y_pos, height , width=0.7 ,color= color)
plt.xticks(y_pos, bars)
plt.ylabel('Count')
plt.xlabel('Roomtype')
plt.title("How many reserves in each type of room", fontdict=None, position= [0.48,1.05], size = 'xx-large')
plt.show()


# ## Is Cancel ? 

# In[ ]:


data.is_canceled.value_counts()


# In[ ]:


plt.rcParams['figure.figsize'] = 10,10
labels = ['Not','Cancel']
sizes = data['is_canceled'].value_counts().tolist()
explode = (0, 0.2)
colors = ['dodgerblue','tomato']

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=False, startangle=190)
plt.axis('equal')
plt.tight_layout()
plt.title("How many bookings were cancel", fontdict=None, position= [0.48,1], size = 'xx-large')
plt.show()


# ## Select data to feed to the model

# In[ ]:


input_information = data[['hotel','lead_time','stays_in_week_nights','stays_in_weekend_nights','adults','reserved_room_type','adr'
                          ,'is_canceled']]


# In[ ]:


input_information.shape


# In[ ]:


## Binary encoding the categorical data

input_information = pd.get_dummies(data=input_information)


# In[ ]:


input_information.shape


# In[ ]:


Y_train = input_information["is_canceled"]
X_train = input_information.drop(labels = ["is_canceled"],axis = 1)


# ## Random Forest

#     Here I just applied default Random Forest without any tuning 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  cross_val_score,GridSearchCV

Rfclf = RandomForestClassifier(random_state=15)
Rfclf.fit(X_train, Y_train)


# In[ ]:


clf_score = cross_val_score(Rfclf, X_train, Y_train, cv=10)
print(clf_score)


# In[ ]:


clf_score.mean()


# ## Extract Knowledge from model

# In[ ]:


Rfclf_fea = pd.DataFrame(Rfclf.feature_importances_)
Rfclf_fea["Feature"] = list(X_train) 
Rfclf_fea.sort_values(by=0, ascending=False).head()


# In[ ]:


g = sns.barplot(0,"Feature",data = Rfclf_fea.sort_values(by=0, ascending=False)[0:5], palette="Pastel1",orient = "h")
g.set_xlabel("Weight")
g = g.set_title("Random Forest")


# 
# Feel free to folk this kernel.
# 
# I hope my notebook may help you as start point for this dataset.
# 
# You **upvote** will help me a lot :)
# 
# Thanks
