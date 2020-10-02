#!/usr/bin/env python
# coding: utf-8

# # 2016 US Bike Share Activity Snapshot
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Posing Questions](#pose_questions)
# - [Data Collection and Wrangling](#wrangling)
#   - [Condensing the Trip Data](#condensing)
# - [Exploratory Data Analysis](#eda)
#   - [Statistics](#statistics)
#   - [Visualizations](#visualizations)
# - [Performing Your Own Analysis](#eda_continued)
# - [Conclusions](#conclusions)
# 
# <a id='intro'></a>
# ## Introduction
# 
# 
# Over the past decade, bicycle-sharing systems have been growing in number and popularity in cities across the world. Bicycle-sharing systems allow users to rent bicycles for short trips, typically 30 minutes or less. Thanks to the rise in information technologies, it is easy for a user of the system to access a dock within the system to unlock or return bicycles. These technologies also provide a wealth of data that can be used to explore how these bike-sharing systems are used.
# 
# In this project, you will perform an exploratory analysis on data provided by [Motivate](https://www.motivateco.com/), a bike-share system provider for many major cities in the United States. You will compare the system usage between three large cities: New York City, Chicago, and Washington, DC. You will also see if there are any differences within each system for those users that are registered, regular users and those users that are short-term, casual users.

# <a id='pose_questions'></a>
# ## Posing Questions
# 
# Before looking at the bike sharing data, you should start by asking questions you might want to understand about the bike share data. Consider, for example, if you were working for Motivate. What kinds of information would you want to know about in order to make smarter business decisions? If you were a user of the bike-share service, what factors might influence how you would want to use the service?
# 
# **Question 1**: Write at least two questions related to bike sharing that you think could be answered by data.
# 
# **Answer**:<br/> 1- Where exactly in these cities have the most bike rentals?  
#             2- What is the average time of renting for each city?

# <a id='wrangling'></a>
# ## Data Collection and Wrangling
# 
# Now it's time to collect and explore our data. In this project, we will focus on the record of individual trips taken in 2016 from our selected cities: New York City, Chicago, and Washington, DC. Each of these cities has a page where we can freely download the trip data.:
# 
# - New York City (Citi Bike): [Link](https://www.citibikenyc.com/system-data)
# - Chicago (Divvy): [Link](https://www.divvybikes.com/system-data)
# - Washington, DC (Capital Bikeshare): [Link](https://www.capitalbikeshare.com/system-data)
# 
# If you visit these pages, you will notice that each city has a different way of delivering its data. Chicago updates with new data twice a year, Washington DC is quarterly, and New York City is monthly. **However, you do not need to download the data yourself.** The data has already been collected for you in the `/data/` folder of the project files. While the original data for 2016 is spread among multiple files for each city, the files in the `/data/` folder collect all of the trip data for the year into one file per city. Some data wrangling of inconsistencies in timestamp format within each city has already been performed for you. In addition, a random 2% sample of the original data is taken to make the exploration more manageable. 
# 
# **Question 2**: However, there is still a lot of data for us to investigate, so it's a good idea to start off by looking at one entry from each of the cities we're going to analyze. Run the first code cell below to load some packages and functions that you'll be using in your analysis. Then, complete the second code cell to print out the first trip recorded from each of the cities (the second line of each data file).

# In[ ]:


## import all necessary packages and functions.
import csv # read and write csv files
from datetime import datetime # operations to parse dates
from pprint import pprint # use to print data structures like dictionaries in
                          # a nicer way than the base print function.


# In[ ]:


def print_first_point(filename):
    """
    This function prints and returns the first data point (second row) from
    a csv file that includes a header row.
    """
    # print city name for reference
    city = filename.split('-')[0].split('/')[-1]
    print('\nCity: {}'.format(city))
    
    with open(filename, 'r') as f_in:
        ## TODO: Use the csv library to set up a DictReader object. ##
        ## see https://docs.python.org/3/library/csv.html           ##
        trip_reader = csv.DictReader(f_in)
        
        ## TODO: Use a function on the DictReader object to read the     ##
        ## first trip from the data file and store it in a variable.     ##
        ## see https://docs.python.org/3/library/csv.html#reader-objects ##
        first_trip = next(trip_reader)
        
        ## TODO: Use the pprint library to print the first trip. ##
        ## see https://docs.python.org/3/library/pprint.html     ##
        pprint(first_trip)
        
    # output city name and first trip for later testing
    return (city, first_trip)


# If everything has been filled out correctly, you should see below the printout of each city name (which has been parsed from the data file name) that the first trip has been parsed in the form of a dictionary. When you set up a `DictReader` object, the first row of the data file is normally interpreted as column names. Every other row in the data file will use those column names as keys, as a dictionary is generated for each row.
# 
# This will be useful since we can refer to quantities by an easily-understandable label instead of just a numeric index. For example, if we have a trip stored in the variable `row`, then we would rather get the trip duration from `row['duration']` instead of `row[0]`.
# 
# <a id='condensing'></a>
# ### Condensing the Trip Data
# 
# It should also be observable from the above printout that each city provides different information. Even where the information is the same, the column names and formats are sometimes different. To make things as simple as possible when we get to the actual exploration, we should trim and clean the data. Cleaning the data makes sure that the data formats across the cities are consistent, while trimming focuses only on the parts of the data we are most interested in to make the exploration easier to work with.
# 
# You will generate new data files with five values of interest for each trip: trip duration, starting month, starting hour, day of the week, and user type. Each of these may require additional wrangling depending on the city:
# 
# - **Duration**: This has been given to us in seconds (New York, Chicago) or milliseconds (Washington). A more natural unit of analysis will be if all the trip durations are given in terms of minutes.
# - **Month**, **Hour**, **Day of Week**: Ridership volume is likely to change based on the season, time of day, and whether it is a weekday or weekend. Use the start time of the trip to obtain these values. The New York City data includes the seconds in their timestamps, while Washington and Chicago do not. The [`datetime`](https://docs.python.org/3/library/datetime.html) package will be very useful here to make the needed conversions.
# - **User Type**: It is possible that users who are subscribed to a bike-share system will have different patterns of use compared to users who only have temporary passes. Washington divides its users into two types: 'Registered' for users with annual, monthly, and other longer-term subscriptions, and 'Casual', for users with 24-hour, 3-day, and other short-term passes. The New York and Chicago data uses 'Subscriber' and 'Customer' for these groups, respectively. For consistency, you will convert the Washington labels to match the other two.
# 
# 
# **Question 3a**: Complete the helper functions in the code cells below to address each of the cleaning tasks described above.

# In[ ]:


def duration_in_mins(datum, city):
    """
    Takes as input a dictionary containing info about a single trip (datum) and
    its origin city (city) and returns the trip duration in units of minutes.
    
    Remember that Washington is in terms of milliseconds while Chicago and NYC
    are in terms of seconds. 
    
    HINT: The csv module reads in all of the data as strings, including numeric
    values. You will need a function to convert the strings into an appropriate
    numeric type when making your transformations.
    see https://docs.python.org/3/library/functions.html
    """
    duration = 0
    if city == "Washington":
        duration = float(datum['Duration (ms)']) / 60000
    else:
        duration = float(datum['tripduration']) / 60
    
    return duration


# In[ ]:


def time_of_trip(datum, city):
    """
    Takes as input a dictionary containing info about a single trip (datum) and
    its origin city (city) and returns the month, hour, and day of the week in
    which the trip was made.
    
    Remember that NYC includes seconds, while Washington and Chicago do not.
    
    HINT: You should use the datetime module to parse the original date
    strings into a format that is useful for extracting the desired information.
    see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    """
    month = ""
    hour = ""
    day_of_week = ""
    
    if city == "NYC":
        nyc_datetime = datetime.strptime(datum['starttime'], "%m/%d/%Y %H:%M:%S")
        
        month = int(nyc_datetime.strftime("%-m"))
        hour = int(nyc_datetime.strftime("%-H"))
        day_of_week = nyc_datetime.strftime("%A")
        
    elif city == "Chicago":
        chicago_datetime = datetime.strptime(datum['starttime'], "%m/%d/%Y %H:%M")
        
        month = int(chicago_datetime.strftime("%-m"))
        hour = int(chicago_datetime.strftime("%-H"))
        day_of_week = chicago_datetime.strftime("%A")
        
    elif city == "Washington":
        washington_datetime = datetime.strptime(datum['Start date'], "%m/%d/%Y %H:%M")
        
        month = int(washington_datetime.strftime("%-m"))
        hour = int(washington_datetime.strftime("%-H"))
        day_of_week = washington_datetime.strftime("%A")
    
    return (month, hour, day_of_week)


# In[ ]:


def type_of_user(datum, city):
    """
    Takes as input a dictionary containing info about a single trip (datum) and
    its origin city (city) and returns the type of system user that made the
    trip.
    
    Remember that Washington has different category names compared to Chicago
    and NYC. 
    """
    user_type = ""
    if city == "Washington":
        user_type = datum['Member Type']
        if user_type == 'Registered':
            user_type = 'Subscriber'
        else:
            user_type = 'Customer'
    else:
        user_type = datum['usertype']
        
    return user_type


# **Question 3b**: Now, use the helper functions you wrote above to create a condensed data file for each city consisting only of the data fields indicated above. In the `/examples/` folder, you will see an example datafile from the [Bay Area Bike Share](http://www.bayareabikeshare.com/open-data) before and after conversion. Make sure that your output is formatted to be consistent with the example file.
# 
# **Please note that the following function and its test below won't work on Kaggle's Kernels, it was used to write files in a format that assists in the analysis. You can skip these and continue on the Exploratory Data Analysis part. Enjoy**

# In[ ]:


def condense_data(in_file, out_file, city):
    """
    This function takes full data from the specified input file
    and writes the condensed data to a specified output file. The city
    argument determines how the input file will be parsed.
    
    HINT: See the cell below to see how the arguments are structured!
    """
    
    with open(out_file, 'w') as f_out, open(in_file, 'r') as f_in:
        # set up csv DictWriter object - writer requires column names for the
        # first row as the "fieldnames" argument
        out_colnames = ['duration', 'month', 'hour', 'day_of_week', 'user_type']        
        trip_writer = csv.DictWriter(f_out, fieldnames = out_colnames)
        trip_writer.writeheader()
        
        ## TODO: set up csv DictReader object ##
        trip_reader = csv.DictReader(f_in)
        # collect data from and process each row
        for row in trip_reader:
            # set up a dictionary to hold the values for the cleaned and trimmed
            # data point
            new_point = {}
            ## TODO: use the helper functions to get the cleaned data from  ##
            ## the original data dictionaries.                              ##
            ## Note that the keys for the new_point dictionary should match ##
            ## the column names set in the DictWriter object above.         ##
            new_point['duration'] = duration_in_mins(row, city)
            new_point['month'] = time_of_trip(row, city)[0]
            new_point['hour'] = time_of_trip(row, city)[1]
            new_point['day_of_week'] = time_of_trip(row, city)[2]
            new_point['user_type'] = type_of_user(row, city)

            ## TODO: write the processed information to the output file.     ##
            ## see https://docs.python.org/3/library/csv.html#writer-objects ##
            trip_writer.writerow(new_point)


# In[ ]:


# Run this cell to check your work
city_info = {'Washington': {'in_file': '../input/Washington-CapitalBikeshare-2016.csv',
                            'out_file': '../input/Washington-2016-Summary.csv'},
             'Chicago': {'in_file': '../input/Chicago-Divvy-2016.csv',
                         'out_file': '../input/Chicago-2016-Summary.csv'},
             'NYC': {'in_file': '../input/NYC-CitiBike-2016.csv',
                     'out_file': '../input/NYC-2016-Summary.csv'}}

for city, filenames in city_info.items():
    condense_data(filenames['in_file'], filenames['out_file'], city)
    print_first_point(filenames['out_file'])


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# Now that you have the data collected and wrangled, you're ready to start exploring the data. In this section you will write some code to compute descriptive statistics from the data. You will also be introduced to the `matplotlib` library to create some basic histograms of the data.
# 
# <a id='statistics'></a>
# ### Statistics
# 
# First, let's compute some basic counts. The first cell below contains a function that uses the csv module to iterate through a provided data file, returning the number of trips made by subscribers and customers. The second cell runs this function on the example Bay Area data in the `/examples/` folder. Modify the cells to answer the question below.
# 
# **Question 4a**: Which city has the highest number of trips? Which city has the highest proportion of trips made by subscribers? Which city has the highest proportion of trips made by short-term customers?
# 
# **Answer**: NYC has the highest number of trips, and it also has the highest proportion of trips made by subscribers. However, Chicago has the highest proportion of trips made by short-term customers.

# In[ ]:


def number_of_trips(filename):
    """
    This function reads in a file with trip data and reports the number of
    trips made by subscribers, customers, and total overall.
    """
    with open(filename, 'r') as f_in:
        # set up csv reader object
        reader = csv.DictReader(f_in)
        
        # initialize count variables
        n_subscribers = 0
        n_customers = 0
        
        # tally up ride types
        for row in reader:
            if row['user_type'] == 'Subscriber':
                n_subscribers += 1
            else:
                n_customers += 1
        
        # compute total number of rides
        n_total = n_subscribers + n_customers
        
        # return tallies as a tuple
        subs_prop = n_subscribers/n_total
        customers_prop = n_customers/n_total
        return('Number of trips = {}\nSubscribers proportion = {:.2f}\nCustomers proportion = {:.2f}'.format(n_total, subs_prop, customers_prop))


# In[ ]:


## Modify this and the previous cell to answer Question 4a. Remember to run ##
## the function on the cleaned data files you created from Question 3.      ##
washington_summary = '../input/Washington-2016-Summary.csv'
print("Washington Summary:\n{}\n".format(number_of_trips(washington_summary)))

chicago_summary = '../input/Chicago-2016-Summary.csv'
print("Chicago Summary:\n{}\n".format(number_of_trips(chicago_summary)))

nyc_summary = '../input/NYC-2016-Summary.csv'
print("NYC Summary:\n{}".format(number_of_trips(nyc_summary)))


# 
# 
# Now, you will write your own code to continue investigating properties of the data.
# 
# **Question 4b**: Bike-share systems are designed for riders to take short trips. Most of the time, users are allowed to take trips of 30 minutes or less with no additional charges, with overage charges made for trips of longer than that duration. What is the average trip length for each city? What proportion of rides made in each city are longer than 30 minutes?
# 
# **Answer**: Washington has an average trip length of 19 minutes, Chiacago has an average trip length of 17 minutes, and NYC has an average trip length of 16 minutes. 10.84% of rides are longer than 30 minutes in Washingtion. In Chicago, 8.33% of rides are longer than 30 minutes. Only 7.3% of NYC's rides are longer than 30 minutes.

# In[ ]:


## Use this and additional cells to answer Question 4b.                 ##
##                                                                      ##
## HINT: The csv module reads in all of the data as strings, including  ##
## numeric values. You will need a function to convert the strings      ##
## into an appropriate numeric type before you aggregate data.          ##
## TIP: For the Bay Area example, the average trip length is 14 minutes ##
## and 3.5% of trips are longer than 30 minutes.                        ##
def trips_duration(filename, free_duration = 30):
    """
    This function reads in a file with trip data and reports the duration of
    trips made on average and whether or not they are extra charged.
    """
    with open(filename, 'r') as f_in:
        # set up csv reader object
        reader = csv.DictReader(f_in)
        
        # initialize total trips length, number of trips, and counters for whether or not the trip is extra charged. 
        total_trips_len = 0
        total_trips_count = 0
        extra_charge = 0
        no_extra_charge = 0
        
        for row in reader:
            current_row = float(row['duration'])
            
            if current_row > free_duration:
                extra_charge+=1
            else:
                no_extra_charge+=1
                
            total_trips_len+=current_row
            total_trips_count+=1
            
        average_trip_len = total_trips_len/total_trips_count
        extra_charge_prop = extra_charge/total_trips_count
        return ("Average trip length = {:.0f} minutes\nPropotion of extra charged trips = {:.4f}"
                .format(average_trip_len, extra_charge_prop))


# In[ ]:


washington_summary = '../input/Washington-2016-Summary.csv'
print("Washington Summary:\n{}\n".format(trips_duration(washington_summary)))

chicago_summary = '../input/Chicago-2016-Summary.csv'
print("Chicago Summary:\n{}\n".format(trips_duration(chicago_summary)))

nyc_summary = '../input/NYC-2016-Summary.csv'
print("NYC Summary:\n{}".format(trips_duration(nyc_summary)))


# **Question 4c**: Dig deeper into the question of trip duration based on ridership. Choose one city. Within that city, which type of user takes longer rides on average: Subscribers or Customers?
# 
# **Answer**: In NYC, on average, customers tend to take longer rides (i.e. 32.8 minutes) compared to subscribers (i.e. 13.7 minutes)

# In[ ]:


## Use this and additional cells to answer Question 4c. If you have    ##
## not done so yet, consider revising some of your previous code to    ##
## make use of functions for reusability.                              ##
##                                                                     ##
## TIP: For the Bay Area example data, you should find the average     ##
## Subscriber trip duration to be 9.5 minutes and the average Customer ##
## trip duration to be 54.6 minutes. Do the other cities have this     ##
## level of difference?                                                ##
def trips_duration2(filename):
    """
    This function reads in a file with trip data and reports the duration of
    trips made on average by subscribers and customers.
    """
    with open(filename, 'r') as f_in:
        # set up csv reader object
        reader = csv.DictReader(f_in)
        
        # initialize subscriber trips length, customer trips length and their counters
        sub_trips_len = 0
        cust_trips_len = 0
        sub_trips_count = 0
        cust_trips_count = 0
        
        for row in reader:
            current_row = float(row['duration'])
            
            if row['user_type'] == 'Subscriber':
                sub_trips_len+=current_row
                sub_trips_count+=1
            else:
                cust_trips_len+=current_row
                cust_trips_count+=1
            
        average_sub_trip_len = sub_trips_len/sub_trips_count
        average_cust_trip_len = cust_trips_len/cust_trips_count
        return ("Average trip length for subscribers = {:.1f} minutes\nAverage trip length for customers = {:.1f} minutes."
                .format(average_sub_trip_len, average_cust_trip_len))


# In[ ]:


nyc_summary = '../input/NYC-2016-Summary.csv'
print("NYC Summary:\n{}".format(trips_duration2(nyc_summary)))


# <a id='visualizations'></a>
# ### Visualizations
# 
# The last set of values that you computed should have pulled up an interesting result. While the mean trip time for Subscribers is well under 30 minutes, the mean trip time for Customers is actually _above_ 30 minutes! It will be interesting for us to look at how the trip times are distributed. In order to do this, a new library will be introduced here, `matplotlib`. Run the cell below to load the library and to generate an example plot.

# In[ ]:


# load library
import calendar
import matplotlib.pyplot as plt
import numpy as np

# this is a 'magic word' that allows for plots to be displayed
# inline with the notebook. If you want to know more, see:
# http://ipython.readthedocs.io/en/stable/interactive/magics.html
get_ipython().run_line_magic('matplotlib', 'inline')

# example histogram, data taken from bay area sample
data = [ 7.65,  8.92,  7.42,  5.50, 16.17,  4.20,  8.98,  9.62, 11.48, 14.33,
        19.02, 21.53,  3.90,  7.97,  2.62,  2.67,  3.08, 14.40, 12.90,  7.83,
        25.12,  8.30,  4.93, 12.43, 10.60,  6.17, 10.88,  4.78, 15.15,  3.53,
         9.43, 13.32, 11.72,  9.85,  5.22, 15.10,  3.95,  3.17,  8.78,  1.88,
         4.55, 12.68, 12.38,  9.78,  7.63,  6.45, 17.38, 11.90, 11.52,  8.63,]
plt.hist(data)
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (m)')
plt.show()


# In the above cell, we collected fifty trip times in a list, and passed this list as the first argument to the `.hist()` function. This function performs the computations and creates plotting objects for generating a histogram, but the plot is actually not rendered until the `.show()` function is executed. The `.title()` and `.xlabel()` functions provide some labeling for plot context.
# 
# You will now use these functions to create a histogram of the trip times for the city you selected in question 4c. Don't separate the Subscribers and Customers for now: just collect all of the trip times and plot them.

# In[ ]:


def get_data(filename):
    """
    This function returns a list of all trips durations.
    """
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        
        dataset = []
        for row in reader:
            dataset.append(round(float(row['duration']), 2))
            
        return dataset


# In[ ]:


## Use this and additional cells to collect all of the trip times as a list ##
nyc_data = get_data(nyc_summary)

## and then use pyplot functions to generate a histogram of trip times.     ##
plt.hist(nyc_data, range=(0,60))
plt.title('Distribution of NYC Trip Durations')
plt.xlabel('Duration (m)')
plt.show()


# If you followed the use of the `.hist()` and `.show()` functions exactly like in the example, you're probably looking at a plot that's completely unexpected. The plot consists of one extremely tall bar on the left, maybe a very short second bar, and a whole lot of empty space in the center and right. Take a look at the duration values on the x-axis. This suggests that there are some highly infrequent outliers in the data. Instead of reprocessing the data, you will use additional parameters with the `.hist()` function to limit the range of data that is plotted. Documentation for the function can be found [[here]](https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.hist.html#matplotlib.pyplot.hist).
# 
# **Question 5**: Use the parameters of the `.hist()` function to plot the distribution of trip times for the Subscribers in your selected city. Do the same thing for only the Customers. Add limits to the plots so that only trips of duration less than 75 minutes are plotted. As a bonus, set the plots up so that bars are in five-minute wide intervals. For each group, where is the peak of each distribution? How would you describe the shape of each distribution?
# 
# **Answer**: For subscribers, the peak is at 10 minutes duration. For customers, the peak is at ~20 minutes (15-20 minutes) duration. The shape of subscriber trip durations distribution in NYC seems significantly right-skewed. However, the shape of customer trip durations distribution in NYC looks more normally distributed than the previous one with a little amount of outliers.

# In[ ]:


## Use this and additional cells to answer Question 5. ##
def get_subscriber_data(filename):
    """
    This function returns a list of all trips durations by subscribers.
    """
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        
        subscriber_dataset = []
        for row in reader:
            if row['user_type'] == 'Subscriber':
                subscriber_dataset.append(round(float(row['duration']), 2))
            
        return subscriber_dataset

def get_customer_data(filename):
    """
    This function returns a list of all trips durations by customers.
    """
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        
        customer_dataset = []
        for row in reader:
            if row['user_type'] == 'Customer':
                customer_dataset.append(round(float(row['duration']), 2))
                
        return customer_dataset


# In[ ]:


nyc_subscriber_data = get_subscriber_data(nyc_summary)
plt.hist(nyc_subscriber_data, range=(0,75), width=5)
plt.title('Distribution of NYC\'s Subscriber Trip Durations')
plt.xlabel('Duration (m)')
plt.ylabel('Number of Trips')
plt.show()

nyc_customer_data = get_customer_data(nyc_summary)
plt.hist(nyc_customer_data, range=(0,75), width=5)
plt.title('Distribution of NYC\'s Customer Trip Durations')
plt.xlabel('Duration (m)')
plt.ylabel('Number of Trips')
plt.show()


# <a id='eda_continued'></a>
# ## Performing Your Own Analysis
# 
# So far, you've performed an initial exploration into the data available. You have compared the relative volume of trips made between three U.S. cities and the ratio of trips made by Subscribers and Customers. For one of these cities, you have investigated differences between Subscribers and Customers in terms of how long a typical trip lasts. Now it is your turn to continue the exploration in a direction that you choose. Here are a few suggestions for questions to explore:
# 
# - How does ridership differ by month or season? Which month / season has the highest ridership? Does the ratio of Subscriber trips to Customer trips change depending on the month or season?
# - Is the pattern of ridership different on the weekends versus weekdays? On what days are Subscribers most likely to use the system? What about Customers? Does the average duration of rides change depending on the day of the week?
# - During what time of day is the system used the most? Is there a difference in usage patterns for Subscribers and Customers?
# 
# If any of the questions you posed in your answer to question 1 align with the bullet points above, this is a good opportunity to investigate one of them. As part of your investigation, you will need to create a visualization. If you want to create something other than a histogram, then you might want to consult the [Pyplot documentation](https://matplotlib.org/devdocs/api/pyplot_summary.html). In particular, if you are plotting values across a categorical variable (e.g. city, user type), a bar chart will be useful. The [documentation page for `.bar()`](https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.bar.html#matplotlib.pyplot.bar) includes links at the bottom of the page with examples for you to build off of for your own use.
# 
# **Question 6**: Continue the investigation by exploring another question that could be answered by the data available. Document the question you want to explore below. Your investigation should involve at least two variables and should compare at least two groups. You should also use at least one visualization as part of your explorations.
# 
# **Answer**: How does ridership differ every month for each user type?

# In[ ]:


## Use this and additional cells to continue to explore the dataset. ##
## Once you have performed your exploration, document your findings  ##
## in the Markdown cell above.##
def get_subscriber_monthly_data(filename):
    """
    This function returns a dictionary of number of ridership per month of subscribers.
    """
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        trips_per_month = {}
        for row in reader:
            if row['user_type'] == 'Subscriber': 
                month_name = calendar.month_name[int(row['month'])]
                if trips_per_month.get(month_name) == None:
                    trips_per_month[month_name] = 1
                else:
                    trips_per_month[month_name] += 1
        return trips_per_month
    
    
def get_customer_monthly_data(filename):
    """
    This function returns a dictionary of number of ridership per month of customers.
    """
    with open(filename, 'r') as file_in:
        reader = csv.DictReader(file_in)
        trips_per_month = {}
        for row in reader:
            if row['user_type'] == 'Customer': 
                month_name = calendar.month_name[int(row['month'])]
                if trips_per_month.get(month_name) == None:
                    trips_per_month[month_name] = 1
                else:
                    trips_per_month[month_name] += 1
        return trips_per_month
        


# In[ ]:


nyc_sub_monthly_data = get_subscriber_monthly_data(nyc_summary)
nyc_cust_monthly_data = get_customer_monthly_data(nyc_summary)

bar_width = 0.3
x = np.arange(len(nyc_sub_monthly_data))
subscriber_bars = nyc_sub_monthly_data.values()
customer_bars = nyc_cust_monthly_data.values()

plt.figure(figsize=(14,7))
plt.bar(x, subscriber_bars, width=bar_width, label='Subscribers', color='#0D47A1')
plt.bar(x + bar_width, customer_bars, width=bar_width, label='Customers', color='#4DB6AC')

plt.xticks(x + bar_width/2, nyc_sub_monthly_data.keys())
plt.title('Monthly Ridership Per User Type in NYC')
plt.xlabel('Month')
plt.ylabel('Ridership')
plt.legend()
plt.show()


# We can observe that the highest subscriber ridership in NYC occur between August and October, and the highest customer ridership occur between July and September.
# 
# Both subscribers and customers do not use or rent bikes in Janurary and February compared to other months.

# ## References
# 
# The Python Graph Gallery website have inspired me to improve my visualizations to answer Q6. The link can be found [here](https://python-graph-gallery.com/11-grouped-barplot/)

# <a id='conclusions'></a>
# ## Conclusions
# 
# Congratulations on completing the project! This is only a sampling of the data analysis process: from generating questions, wrangling the data, and to exploring the data. Normally, at this point in the data analysis process, you might want to draw conclusions about the data by performing a statistical test or fitting the data to a model for making predictions. There are also a lot of potential analyses that could be performed on the data which are not possible with only the data provided. For example, detailed location data has not been investigated. Where are the most commonly used docks? What are the most common routes? As another example, weather has potential to have a large impact on daily ridership. How much is ridership impacted when there is rain or snow? Are subscribers or customers affected more by changes in weather?
# 
# **Question 7**: Putting the bike share data aside, think of a topic or field of interest where you would like to be able to apply the techniques of data science. What would you like to be able to learn from your chosen subject?
# 
# **Answer**: I would like to apply the techniques and skills of Data Science in the business world to help organizations utilize the vast amount of data in a valuable manner. In addition, applying them in the health industry to help assist doctors and professionals in early diagnosis of diseases.
