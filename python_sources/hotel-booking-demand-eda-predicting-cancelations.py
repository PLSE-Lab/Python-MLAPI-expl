#!/usr/bin/env python
# coding: utf-8

# # Hotel Booking Demand
# 
# # EDA (Exploratory Data Analysis)

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from timeit import default_timer as timer
from sklearn.model_selection import GridSearchCV


# In[ ]:


dataset = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.isnull().sum()


# In[ ]:


#children: NA -> 0
#country: NA -> 'N.A.'
#agent: NA -> 0
#company: NA -> 0

dataset.fillna({'children' : 0, 'country' : 'N. A.', 'agent' : 0, 'company' : 0}, inplace = True)


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset[['children', 'country', 'agent', 'company']]


# In[ ]:


dataset.children.value_counts()


# In[ ]:


dataset.country.value_counts()


# In[ ]:


dataset.agent.value_counts()


# In[ ]:


dataset.company.value_counts()


# In[ ]:


dataset.describe(include = 'all')


# In[ ]:


#number of reservations for each hotel

def auto_label(ax, container):
    for c in container:
        height = c.get_height()
        
        ax.annotate('{}'.format(height),
                    xy = (c.get_x() + c.get_width() / 2, height),
                    xytext = (0, 3),  #3 points vertical offset
                    textcoords = 'offset points',
                    ha = 'center', va = 'bottom')

def show_number_reservations():
    dict = []
    
    for hotel in dataset.hotel.unique():
        dict.append({'hotel' : hotel, 
                     'reservations' : len(dataset.loc[dataset.hotel == hotel, 'hotel']), 
                     'cancelations' : len(dataset.loc[(dataset.hotel == hotel) & (dataset.is_canceled == 1), 'hotel'])})
    
    data = pd.DataFrame(dict)
    
    fig, ax = plt.subplots()
    
    reserv_bar = ax.bar(data.hotel, data.reservations, label = 'Reservations')
    cancel_bar = ax.bar(data.hotel, data.cancelations, label = 'Cancelations')
            
    plt.title('Number of reservations by Hotel')
    plt.xlabel('Hotel')
    plt.ylabel('Number of reservations')
    plt.legend()

    auto_label(ax, reserv_bar)
    auto_label(ax, cancel_bar)
    
    return data

show_number_reservations()


# In[ ]:


attrs = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 
         'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 
         'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests']

def show_attributes(canceled, attributes):
    dict = []
    
    for attr in attributes:        
        data = dataset.loc[dataset.is_canceled == (1 if canceled else 0), ['hotel', attr]]                      .groupby('hotel').agg(['mean']).reset_index()
        
        for i in range(len(dataset.hotel.unique())):
            dict.append({'hotel' : data.iat[i, 0], 'attribute' : attr, 'value' : data.iat[i, 1]})
    
    result = pd.DataFrame(dict)
    
    for attr in attributes:
        ax = result.loc[result.attribute == attr].plot('hotel', 'value', kind = 'bar', label = attr)
        ax.set_title(attr + (' | canceled reservations' if canceled else ''))
        ax.set_xlabel('Hotel')
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, horizontalalignment = 'center')
        plt.show()
    
    return result

show_attributes(True, attrs)


# In[ ]:


attrs = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 
         'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 
         'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests', 
         'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type',
         'deposit_type', 'agent', 'company', 'customer_type']

def list_attributes(canceled, attributes):
    dict = []

    for attr in attributes:        
        data = dataset.loc[dataset.is_canceled == (1 if canceled else 0), ['hotel', attr]]

        operation = 'count' if data.dtypes[attr] == np.object else 'mean'
            
        data = data.groupby('hotel').agg([operation]).reset_index()

        for i in range(len(dataset.hotel.unique())):
            dict.append({'hotel' : data.iat[i, 0], 'attribute' : attr, 'type': operation, 'value' : data.iat[i, 1]})

    return pd.DataFrame(dict)

list_attributes(True, attrs)


# In[ ]:


attrs = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 
         'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 
         'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests', 
         'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type',
         'deposit_type', 'agent', 'company', 'customer_type']

def freq_attributes(canceled, attributes):
    hotels = dataset.hotel.unique()
        
    for attr in attributes:
        for h in hotels:            
            counts = dataset.loc[(dataset.hotel == h) & (dataset.is_canceled == (1 if canceled else 0)), attr].value_counts()
            
            plt.figure(figsize = (10, 4))
            ax = counts.head(30).plot(kind = 'bar', label = attr)
            ax.set_title(h + ' | ' + attr + (' | canceled reservations' if canceled else ''))
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
            plt.tight_layout()
            plt.show()
        
freq_attributes(True, attrs)


# In[ ]:


#from which countries did the guests come?
def show_countries():
    countries = dataset[dataset.is_canceled == 0].groupby('country').country.count().sort_values(ascending = False)
    
    plt.figure(figsize = (15, 5))
    ax = countries.head(20).plot(kind = 'bar')
    ax.set_title('Number of reservations per Country')
    ax.set_xlabel('Country')
    ax.set_ylabel('Number of reservations')
#     ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    plt.xticks(rotation = 45)
    plt.show()
    
    return countries

show_countries()


# In[ ]:


#from wich country were the guests who canceled their reservations the most?
def show_countries_canceled():
    countries = dataset[dataset.is_canceled == 1].groupby('country').country.count().sort_values(ascending = False)
    
    plt.figure(figsize = (15, 5))
    ax = countries.head(20).plot(kind = 'bar')
    ax.set_title('Number of canceled reservations per Country')
    ax.set_xlabel('Country')
    ax.set_ylabel('Number of canceled reservations')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    plt.show()
    
    return countries

show_countries_canceled()


# In[ ]:


#what are the months with the most guests in each hotel?
def guests_month():
    info = ['arrival_date_month', 'hotel', 'is_canceled', 'adults', 'children', 'babies']
    guests = dataset.loc[dataset.is_canceled == 0, info]
    guests['total_guests'] = guests.adults + guests.children + guests.babies
    guests.total_guests = guests.total_guests.astype(int)
    
    guests = guests.groupby(['arrival_date_month', 'hotel'])                    .aggregate({'total_guests' : sum})                    .reset_index()
    
    guests.columns = ['month', 'hotel', 'guests']
        
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
        
    guests.month = pd.Categorical(guests.month, categories = months, ordered = True)
    
    guests.sort_values('month', inplace = True)
    
    hotels = guests.hotel.unique()
    fig, ax = plt.subplots(figsize = (12, 6))
    bars = []
        
    for i in range(len(hotels)):
        bars.append(ax.bar(guests.loc[guests.hotel == hotels[i], 'month'],                            guests.loc[guests.hotel == hotels[i], 'guests'],                            label = hotels[i]))
        auto_label(ax, bars[i])
        
    anos = dataset.arrival_date_year.unique()
    plt.title(f'Total number of guests per month by Hotel from {min(anos)} to {max(anos)}')
    plt.xlabel('Month')
    plt.ylabel('Number of guests')
    plt.legend()
    plt.show()
    
    return guests

guests_month()


# In[ ]:


#what are the average prices for each person per night by hotel?
def average_daily_rate_person():
    for h in dataset.hotel.unique():
        print(f'Average daily rate per person: {dataset[dataset.hotel == h].adr.mean():.2f} in {h}.')
    return

average_daily_rate_person()


# In[ ]:


#what is the average stay in days?
def average_stay_days():
    stay = dataset.loc[dataset.is_canceled == 0, ['hotel', 'stays_in_week_nights', 'stays_in_weekend_nights']] 
    stay['nights'] = stay.stays_in_week_nights + stay.stays_in_weekend_nights    
    stay = stay.groupby('hotel').agg({'nights' : 'mean'})
    
    return stay

average_stay_days()


# # Predicting cancelations
# ## Determining the most relevant variables

# In[ ]:


dataset.corr()


# In[ ]:


corr = dataset.corr()

plt.figure(figsize = (10, 5))

ax = sns.heatmap(corr, vmin = -1, vmax = 1, center = 0, 
                 cmap = sns.diverging_palette(20, 220, n = 200))

ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment = 'right')


# In[ ]:


dataset.corr()['is_canceled'].abs().sort_values(ascending = False)


# The first 5 variables (lead_time, total_of_special_requests, required_car_parking_spaces, booking_changes, previous_cancellations) will be used for building a model, since they are the ones with correlation > 0.10 for is_canceled.

# In[ ]:


dataset.lead_time.value_counts()


# In[ ]:


dataset.groupby(['is_canceled', 'lead_time']).count()['hotel']


# In[ ]:


dataset.total_of_special_requests.value_counts()


# In[ ]:


dataset.groupby(['is_canceled', 'total_of_special_requests']).agg({'hotel' : 'count'})


# In[ ]:


dataset.required_car_parking_spaces.value_counts()


# In[ ]:


dataset.groupby(['is_canceled', 'required_car_parking_spaces']).agg({'hotel' : 'count'})


# The variable 'required_car_parking_spaces' may not be decisive in determining if a reservation is canceled or not.

# In[ ]:


dataset.booking_changes.value_counts()


# In[ ]:


dataset.groupby(['is_canceled', 'booking_changes']).agg({'hotel' : 'count'})


# In[ ]:


dataset.previous_cancellations.value_counts()


# In[ ]:


dataset.groupby(['is_canceled', 'previous_cancellations']).agg({'hotel' : 'count'})


# ## Model 1: Naive Bayes
# 
# Since the most relevant variables are already numeric, no conversions are needed.
# 
# Using the following variables, the classifier is expected to determine wether a reservation will probably be canceled or not:
# * lead_time
# * total_of_special_requests
# * required_car_parking_spaces
# * booking_changes
# * previous_cancellations

# In[ ]:


variables = ['lead_time', 'total_of_special_requests', 'required_car_parking_spaces', 
             'booking_changes', 'previous_cancellations', 'is_canceled']

dataset[variables]


# In[ ]:


model = GaussianNB()

#using KFold validation to try to increase the accuracy
def kfold_validation(dataset, variables, k, shuffle = False):
    i = 1

    kfold = KFold(n_splits = k, shuffle = shuffle)

    for i_train, i_test in kfold.split(dataset):
        data = dataset.loc[i_train, variables[:-1]]
        target = dataset.loc[i_train, variables[-1]]

        model.fit(data, target)

        data = dataset.loc[i_test, variables[:-1]]
        target = dataset.loc[i_test, variables[-1]]

        predicted = model.predict(data)

        print(53 * '-' + f'\nFold {i} | train: {len(i_train)} | test: {len(i_test)}')
        print(metrics.classification_report(target, predicted))
        print(metrics.confusion_matrix(target, predicted))

        i += 1
        
    return kfold

kfold_validation(dataset, variables, 5)


# In[ ]:


data = dataset[variables[:-1]]
target = dataset[variables[-1]]

predicted = model.predict(data)

print(metrics.classification_report(target, predicted))
print(metrics.confusion_matrix(target, predicted))


# The Naive Bayes classifier had a bad performance. The accuracy was low in almost all folds. Although recall for canceled reservations was high, the precision was low. The classifier can detect most of the real cancelations, but wrongly classifies a great number of valid reservations as canceled.
# 
# ## Model 2: Naive Bayes + new set of variables
# 
# Since using the 5 most correlated variables didn't produce good results, we'll try to use a better set of variables. Of the 5 used variables, required_car_parking_spaces is the only one which has just one value for all canceled reservations. The other variables have more values for both canceled and valid reservations. Let's try to use just the other 4 variables.
# 
# Using the following variables, the classifier is expected to determine wether a reservation will probably be canceled or not:
# * lead_time
# * total_of_special_requests
# * booking_changes
# * previous_cancellations

# In[ ]:


variables = ['lead_time', 'total_of_special_requests',
             'booking_changes', 'previous_cancellations', 'is_canceled']

dataset[variables]


# In[ ]:


model = GaussianNB()

kfold_validation(dataset, variables, 5)


# In[ ]:


data = dataset[variables[:-1]]
target = dataset[variables[-1]]

predicted = model.predict(data)

print(metrics.classification_report(target, predicted))
print(metrics.confusion_matrix(target, predicted))


# Precision and recall for valid cancelations were better than in Model 1. Unfortunately, precision and recall for canceled reservations were worse. The variable required_car_parking_spaces had a greater impact than anticipated.
# 
# ## Model 3: Naive Bayes + all numeric variables

# In[ ]:


variables = list(dataset.select_dtypes(include = np.number).columns.values)
variables


# In[ ]:


temp = variables[0]
variables[0] = variables[-1]
variables[-1] = temp

variables


# In[ ]:


dataset[variables]


# In[ ]:


model = GaussianNB()

kfold_validation(dataset, variables, 5)


# In[ ]:


data = dataset[variables[:-1]]
target = dataset[variables[-1]]

predicted = model.predict(data)

print(metrics.classification_report(target, predicted))
print(metrics.confusion_matrix(target, predicted))


# Using all numeric variables did not produce better results than before.
# 
# ## Model 4: Naive Bayes + PCA (using the 5 variables with higher correlation to is_canceled)

# In[ ]:


variables = ['lead_time', 'total_of_special_requests', 'required_car_parking_spaces', 
             'booking_changes', 'previous_cancellations', 'is_canceled']

dataset[variables[:-1]]


# In[ ]:


pca = PCA()

pca


# In[ ]:


transformed = pca.fit_transform(dataset[variables[:-1]])
transformed


# In[ ]:


len(pca.components_)


# In[ ]:


sum(pca.explained_variance_ratio_[0:1])


# In[ ]:


dataset_pca = pd.DataFrame(transformed)
dataset_pca.drop([1, 2, 3, 4], axis = 1, inplace = True)
dataset_pca


# In[ ]:


dataset_pca['is_canceled'] = dataset['is_canceled']
dataset_pca.is_canceled


# In[ ]:


dataset_pca.columns = ['pc1', 'is_canceled']

dataset_pca


# In[ ]:


len(dataset) == len(dataset_pca)


# In[ ]:


print(f'dataset -> {dataset[variables].shape}\ndataset_pca -> {dataset_pca.shape}')


# In[ ]:


variables = list(dataset_pca.columns)
variables


# In[ ]:


model = GaussianNB()

kfold_validation(dataset_pca, variables, 5, False)


# In[ ]:


data = dataset_pca[variables[:-1]]
target = dataset_pca[variables[-1]]

predicted = model.predict(data)

print(metrics.classification_report(target, predicted))
print(metrics.confusion_matrix(target, predicted))


# Using highly correlated variables with PCA did not produce better results than before.
# 
# ## Model 5: Naive Bayes + PCA (with all numeric variables)

# In[ ]:


variables = list(dataset.select_dtypes(include = np.number).columns.values)
temp = variables[0]
variables[0] = variables[-1]
variables[-1] = temp

variables


# In[ ]:


dataset[variables]


# In[ ]:


pca = PCA()
transformed = pca.fit_transform(dataset[variables])
transformed


# In[ ]:


len(pca.components_)


# In[ ]:


sum(pca.explained_variance_ratio_[0:5])


# In[ ]:


dataset_pca = pd.DataFrame(transformed)
dataset_pca.drop(range(5, 20), axis = 1, inplace = True)
dataset_pca


# In[ ]:


dataset_pca['is_canceled'] = dataset.is_canceled
dataset_pca.columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'is_canceled']
dataset_pca


# In[ ]:


print(f'dataset -> {dataset[variables].shape}\ndataset_pca -> {dataset_pca.shape}')


# In[ ]:


variables = list(dataset_pca.columns)
variables


# In[ ]:


model = GaussianNB()

kfold_validation(dataset_pca, variables, 5)


# In[ ]:


data = dataset_pca[variables[:-1]]
target = dataset_pca[variables[-1]]

predicted = model.predict(data)

print(metrics.classification_report(target, predicted))
print(metrics.confusion_matrix(target, predicted))


# Using all numeric variables with PCA did not produce better results than before.
# 
# ## Model 6: Random Forest + 5 highly correlated variables

# In[ ]:


variables = ['lead_time', 'total_of_special_requests', 'required_car_parking_spaces', 
             'booking_changes', 'previous_cancellations', 'is_canceled']

dataset[variables[:-1]]


# In[ ]:


model = RandomForestClassifier(n_estimators = 1000, max_depth = 20, n_jobs = -1, 
                               random_state = 0, bootstrap = True)

kfold_validation(dataset, variables, 5, True)


# In[ ]:


data = dataset[variables[:-1]]
target = dataset[variables[-1]]

predicted = model.predict(data)

print(metrics.classification_report(target, predicted))
print(metrics.confusion_matrix(target, predicted))


# The Random Forest classifier produced better results than any Naive Bayes run. Maybe it will produce better results if all numeric variables are used.
# 
# ## Model 7: Random Forest + all numeric variables

# In[ ]:


variables = list(dataset.select_dtypes(include = np.number).columns.values)
temp = variables[0]
variables[0] = variables[-1]
variables[-1] = temp

variables


# In[ ]:


model = RandomForestClassifier(n_estimators = 1000, max_depth = 20, n_jobs = -1, 
                               random_state = 0, bootstrap = True)

kfold_validation(dataset, variables, 5, True)


# In[ ]:


data = dataset[variables[:-1]]
target = dataset[variables[-1]]

predicted = model.predict(data)

print(metrics.classification_report(target, predicted))
print(metrics.confusion_matrix(target, predicted))


# Random Forest with all numeric variables, using shuffle during CV, produces the best result so far. High precision and recall for both classes and a high overall accuracy.
