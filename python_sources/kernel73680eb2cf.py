#!/usr/bin/env python
# coding: utf-8

# # Loading the data
# 
# Anyone who travels has to deal with flight delays or, even worse, cancelations. Let's see if we can put an end to this with a kaggle dataset. In this notebook I will load up the data, look at graphs to create new features and then make a model that predicts flight delays. The dataset has about 6 million entries about flights with features like departure time, delay, aircraft type, airline and airport.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings


warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv("../input/flight-delays/flights.csv")


# In[ ]:


df.columns


# This dataset has loads of features, let's narrow it down a bit.

# In[ ]:


features = ["YEAR",
            "MONTH",
            "DAY",
            "SCHEDULED_DEPARTURE",
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "DEPARTURE_DELAY",
            "CANCELLED",
            "DIVERTED",
            "DISTANCE"]
df = df[features]


# # Data sanitization
# We need to check for null values, units of measurements and so on.

# In[ ]:


df.DEPARTURE_DELAY.describe()


# Looking at the **DEPARTURE_DELAY** values - the mean flight delay is 9 minutes, not too bad! The maximum flight delay value is 1988 minutes! That's 33 hours. Even funnier is the minimal flight delay value - negative 82 minutes. So there was a flight that left an hour and a half *before* schedule. I wonder what happened there...
# 
# The 50th percentile is 11 minutes less than the mean, this means that there are extreme outliers among the delayed flights.

# In[ ]:


df.head()


# In[ ]:


df.info()


# Couple of things to note:
# + Columns **CANCELLED** and **DIVERTED** are boolean values, that got misinterpreted as `int64`.
# + **DEPARTURE_DELAY** is an integer value (as said in [a Kaggle competition post](https://www.kaggle.com/usdot/flight-delays/discussion/29308)) that got misinterpreted as `float64`. The `pandas` module doesn't allow `int` datatype to be `NaN` (this has to do with [performance optimisations in numpy](http://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#support-for-integer-na)).
# + **SCHEDULED_TIME** is provided as "%hour%minute" was also misinterpreted to be a float.
# 
# Let's read the dataframe from the file again combining the **YEAR**, **MONTH**, **DAY** and **SCHEDULED_DEPARTURE** columns into a single `datetime` type.

# In[ ]:


df = pd.read_csv("../input/flight-delays/flights.csv",
                 parse_dates=[["YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE"]],
                 dtype={"SCHEDULED_DEPARTURE":str,
                        "DESTINATION_AIRPORT":str,
                        "ORIGIN_AIRPORT":str,
                        "AIRLINE":str},
                 usecols=features)\
        .rename(columns={"YEAR_MONTH_DAY_SCHEDULED_DEPARTURE":"DATETIME"})
df.head()


# In[ ]:


df.info()


# First - convert **CANCELLED** and **DIVERTED** as boolean

# In[ ]:


df.CANCELLED = (df["CANCELLED"] == 1)
df.DIVERTED = (df["DIVERTED"] == 1)


# Now let's inspect why some departure delay values are missing.

# In[ ]:


df[df.DEPARTURE_DELAY.isnull()].head()


# Seems like canceled flights don't have a departure delay - makes sense. I'm going to remove canceled and diverted planes from the dataset, since I only want to analyse delays.

# In[ ]:


df = df[np.invert(df.DIVERTED)&np.invert(df.CANCELLED)]
df.isna().any()


# All na values removed. Let's convert departure delay to int.

# In[ ]:


df.DEPARTURE_DELAY = df.DEPARTURE_DELAY.astype("int")
df.drop(["DIVERTED", "CANCELLED"], axis=1, inplace=True)
df.info()


# That's it then, we're ready to do stuff with the data? Well, not exactly: [one Kaggler found out](https://www.kaggle.com/usdot/flight-delays/discussion/29600) that for the month of October the dataset describes the same airport 3 letter shortcodes with a different internal numerical ID.

# In[ ]:


df.DESTINATION_AIRPORT.sample(10,random_state=42).unique()


# To fix this I went on the department of transportation of US [website](https://www.bts.gov/) and downloaded airport data they had which is the `additional_airport_info.csv` file.

# In[ ]:


id_code_to_letter_code = pd.read_csv("../input/additional-flight-delay-data/additional_airport_info.csv",
                                     dtype={"ORIGIN_AIRPORT_ID":str, "ORIGIN":str})

df.ORIGIN_AIRPORT.replace(id_code_to_letter_code.ORIGIN_AIRPORT_ID.tolist(),
                          id_code_to_letter_code.ORIGIN.tolist(),
                          inplace=True)
df.DESTINATION_AIRPORT.replace(id_code_to_letter_code.ORIGIN_AIRPORT_ID.tolist(),
                          id_code_to_letter_code.ORIGIN.tolist(),
                          inplace=True)
df.DESTINATION_AIRPORT.sample(10, random_state=42).unique()


# In[ ]:


airline_names = pd.read_csv("../input/flight-delays/airlines.csv")                    .set_index("IATA_CODE")                    .AIRLINE                    .to_dict()

airport_names = pd.read_csv("../input/flight-delays/airports.csv")                    .set_index('IATA_CODE')                    .AIRPORT                    .to_dict()


df_with_full_names = df.replace({"AIRLINE": airline_names,
                                 "ORIGIN_AIRPORT": airport_names})


# # Analysis
# 
# The first thing that comes to mind is - airlines. The airlines operate the planes, thus it's their fault the flights are delayed. Let's break down flights by airline and see how the delays compare.

# In[ ]:


plt.rcParams.update({'figure.figsize': (11,6)})
plot_order = df_with_full_names[["AIRLINE","DEPARTURE_DELAY"]]                .groupby("AIRLINE")                .median()                .sort_values(by=["DEPARTURE_DELAY"])                .index

sns.boxplot(x="DEPARTURE_DELAY",
            y="AIRLINE",
            data = df_with_full_names,
            order = plot_order,
            showfliers=False)

# the full dataset is too much to handle for seaborn, we will use 600k datapoints for this graph
small_sample = df_with_full_names.sample(frac=0.1, random_state=42)
sns.stripplot(x="DEPARTURE_DELAY", y="AIRLINE", data = small_sample[small_sample.DEPARTURE_DELAY<400],
              order=plot_order,
              size=1,
              alpha=0.25)


# The results are quite promising, United Air Lines have a history of providing bad service. However, we see a problem here. Lowest flight delay airlines are Alaska, Hawaiian and Atlantic Southeast airlines. These mostly focus on specific destinations and airports - Hawaiian airlines, for example, mostly operate flights to Hawaii. It might be the case that these airlines aren't better, it's just the airports they fly to have smaller delays or Hawaiian weather helps with fast departures.
# 
# Therefore we cannot make any conclusions just from this data.

# So from there, let's have a look if big airports (ones that have a lot of flights coming in and out) have bigger delays than small ones. In other words, is there's a correlation between the size of the airport to the delay of the flights departing from that airport.

# In[ ]:


grouped_by_airport = df.groupby("ORIGIN_AIRPORT")

airport_flight_count = grouped_by_airport.DEPARTURE_DELAY.size()                        .rename("FLIGHT_COUNT")
airport_delay_count = df[df.DEPARTURE_DELAY > 10]                        .groupby("ORIGIN_AIRPORT")                        .DEPARTURE_DELAY                        .size()                        .rename("DELAY_COUNT")
data = pd.concat([airport_delay_count, airport_flight_count], axis=1)

sns.regplot(x="FLIGHT_COUNT", y="DELAY_COUNT", data=data)   .set(xlabel="Number of flights departed from the airport", ylabel="Number of flights delayed by more than 10 minutes")


# This graph shows that bigger airports have more delayed flights, this proves our point, right? Well, not exactly. On the graph we also see that the proportion of delayed flights to total number of flights stays roughly the same, therefore if you go to a big airport you have about the same chance of getting a delayed flight if you went to a smaller airport.
# 
# However, maybe the delays in bigger airports are much longer?

# In[ ]:


avg_flight_delay = grouped_by_airport                    .DEPARTURE_DELAY                    .mean()                    .rename("DELAY_MEAN")
num_of_flights = grouped_by_airport                    .DEPARTURE_DELAY                    .count()                    .rename("NUM_OF_DEPARTURES")
data = pd.concat([num_of_flights, avg_flight_delay], axis=1)

sns.regplot(x="NUM_OF_DEPARTURES", y="DELAY_MEAN", data=data)    .set(xlabel="Number of flights per year by airport", ylabel="Average departure delay")


# The graph shows that flights in bigger airports are delayed longer. Combining the previous graph this means that both small and big airports have the same percantage of delayed flights but delayed flights in big airports are delayed by much longer. It is my speculation that a delayed flight in a big airport will have more trouble departing because there are more planes leaving and arriving to the airport, making the delayed plane wait till there is a gap in the schedule which wouldn't happen in a smaller airport.
# 
# Finally, let's have a look at a grid that shows delays with both the airline and the airport.

# In[ ]:


grouped_by_airport_airline = df.groupby(["AIRLINE","ORIGIN_AIRPORT"]).DEPARTURE_DELAY.mean()
airlines = df.AIRLINE.unique()
airport_size = grouped_by_airport.size().sort_values(ascending=False)
airports = airport_size.index

# reindex so that all possible combinations of airport-airline are present
index = pd.MultiIndex.from_product([airlines,airports], names=["Airline", "Airport"])
airline_airport_delay_mean = grouped_by_airport_airline.reindex(index)                                .to_frame()                                .reset_index()

# rename IATA codes to human friendly names
df_for_plotting = airline_airport_delay_mean.pivot(columns="Airport", index="Airline", values="DEPARTURE_DELAY")
data = df_for_plotting[airports[:15]].rename(index=airline_names, columns=airport_names)
with sns.axes_style("white"):
    sns.heatmap(data, mask = data.isnull())


# On the grid it becomes very clear how airlines perform. There is about a ten minute difference between United Air Lines planes and US airways planes departing from the same Atlanta International Airport. This shows departure delay so the destination airport should not play a role here.
# 
# From my knowledge of flying, I suspect that flights during the holidays are busier and therefore more delayed. This could be a useful feature. To evaluate this, I will pick some American holidays and see if flight delays increase if the flight is closer to a holiday.

# In[ ]:


holidays = pd.to_datetime(["1/1/2015", "4/7/2015", "28/11/2015", "25/12/2015",  "1/1/2016"])

def get_time_to_holiday(time):
    return min(abs(pd.to_timedelta(holidays-time))).days

rounded_departure_dates = df.DATETIME.dt.round("D").unique()
day_to_holiday_dict = {x: get_time_to_holiday(x) for x in rounded_departure_dates}
day_to_holiday_feature = df.DATETIME.dt.round("D")                            .map(day_to_holiday_dict)                            .rename("TIME_TO_HOLIDAY")

df = pd.concat([df, day_to_holiday_feature], axis = 1)


# In[ ]:


sns.barplot(x="TIME_TO_HOLIDAY", y="DEPARTURE_DELAY", data=df[df.TIME_TO_HOLIDAY<20])


# This is interesting. On the day of the holiday there is a smaller delay compared to 3 days later. This could be because not a lot of people travel on the day of the holiday. On the graph there are peaks on 3, 6, 10 and 14 days before the holiday, all of them are somewhat meaningful dates.
# 
# We will add this as an additional feature to our model.

# # Machine learning
# 
# Here we will combine the given features and the features we made ourselves and train a simple linear regression model.

# In[ ]:


numerical_df = pd.concat(
    [df.TIME_TO_HOLIDAY],
    axis=1)

categorical_df = pd.concat(
    [df.DATETIME.dt.hour,
     df.DATETIME.dt.month,
     df.DATETIME.dt.day,
     df.ORIGIN_AIRPORT, 
     df.AIRLINE],
    axis=1
).set_axis(["HOUR", "MONTH", "DAY", "ORIGIN_AIRPORT", "AIRLINE"], axis=1, inplace=False)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
    
encoder = OneHotEncoder()
encoded_categorical_df = encoder.fit_transform(categorical_df)

type(encoded_categorical_df)


# The one hot encoder produces a sparse numpy matrix.

# In[ ]:


from scipy import sparse
from sklearn.model_selection import train_test_split

X = sparse.hstack((sparse.csr_matrix(numerical_df), encoded_categorical_df))

y = df.DEPARTURE_DELAY.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# We will only fit the scaler on the training data to avoid data leak (the maximum value might not be in the training set).

# In[ ]:


from sklearn.preprocessing import StandardScaler

X_train_numerical = X_train[:, 0:3].toarray()
X_test_numerical = X_test[:, 0:3].toarray()
   
scaler = StandardScaler()
scaler.fit(X_train_numerical)
X_train_numerical = sparse.csr_matrix(scaler.transform(X_train_numerical))
X_test_numerical = sparse.csr_matrix(scaler.transform(X_test_numerical))

X_train[:, 0:3] = X_train_numerical
X_test[:, 0:3] = X_test_numerical


# The model of my choice is the scikit-learn SGDregressor. I was considering using the random forest regressor but that model cannot handle the size of this dataset.

# In[ ]:


from sklearn.linear_model import SGDRegressor

model = SGDRegressor(random_state=42)
model.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import mean_absolute_error

y_pred = model.predict(X_test)
print("Our model mean error: ", mean_absolute_error(y_test, y_pred))


# Now, 18 minutes is not great, considering that there will be a 36 minute window in which the real result is likely to lie, however given the data that we got it might not be so bad. Flight delays could be affected by millions of factors such as weather, quality/health of the aircraft and even traffic jams before the flight, we just don't have access to that information.
