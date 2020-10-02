#!/usr/bin/env python
# coding: utf-8

# ## Covid-19 in India
# 
# ![COVID19INDIA.jpg](attachment:COVID19INDIA.jpg)
# 
# The [Covid-19](https://en.wikipedia.org/wiki/Coronavirus_disease_2019) disease has shaken the world. It has spread far and wide across the globe leaving no one safe. While most countries have come to a standstill, a lot of healthcase groups, medical professionals, researchers and various other communities are fighting hard to overcome these hard times.
# 
# WHO declared Covid-19 a pandemic and India is fighting back by invoking the Epidemic Diseases Act in March, 2020: https://www.businesstoday.in/latest/trends/coronavirus-outbreak-india-italy-iran-china-total-confirmed-cases/story/397950.html
# 
# I strongly believe and trust the healthcase experts of the country and the world to fight the virus in the best possible way. There are plenty of analysis, dashboards, insights, models, forecasts that have been shared by the community and we are very grateful to all the organzations that collect and share data publicly to facilitate these reports.
# 

# ## Analysis in April
# Each day of April, I will explore data on Covid-19 in India going deep into one particular aspect of it.
# 
# This notebook serves as a daily update of attempting to showcase data regarding many of the questions posed to India. It is, by no means, confirming or denying any hypothesis, suggestion or idea. I highly recommend to share questions, suggestions or feedback in the comments. I will also try to include as many of the suggestions in the coming days.
# 
# * [April 1: Population](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-1:-Population)   
# Can population be used to estimate mask and testing kit requirements in each state?
# * [April 2: Density](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-2:-Density)   
# Can we get city-level data to explore if densely populated cities are seeing more cases?
# * [April 3: Urbanization](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-3:-Urbanization)   
# Can we use geographical urbanization to control spread of cases?
# * [April 4: Monotonicity](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-4:-Monotonicity)   
# Can we improve the quality of data?
# * [April 5: Gender](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-5:-Gender)   
# Why doesn't the government share gender of cases?
# * [April 6: Funds](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-6:-Funds)   
# Can we contribute towards the fight from home?
# * [April 7: Trajectory](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-7:-Trajectory)   
# Has [Kerala's 2018 Nipah virus](https://en.wikipedia.org/wiki/2018_Nipah_virus_outbreak_in_Kerala) experience helped them flatten their curve?
# * [April 8: Testing](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-8:-Testing)   
# Can the four quadrants of lab availability help in determining the condition of testing?
# 

# In[ ]:


## importing packages
import numpy as np
import pandas as pd

from bokeh.layouts import column, row
from bokeh.models import Panel, Tabs, LinearAxis, Range1d, BoxAnnotation, LabelSet
from bokeh.models.tools import HoverTool
from bokeh.palettes import Category20, Spectral4
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.transform import dodge

from math import pi

output_notebook()


# In[ ]:


## defining constants
PATH_COVID = "/kaggle/input/covid19-in-india/covid_19_india.csv"
PATH_CENSUS = "/kaggle/input/covid19-in-india/population_india_census2011.csv"
PATH_TESTS = "/kaggle/input/covid19-in-india/ICMRTestingDetails.csv"
PATH_LABS = "/kaggle/input/covid19-in-india/ICMRTestingLabs.csv"

def read_covid_data():
    """
    Reads the main covid-19 data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_COVID)
    df.rename(columns = {"State/UnionTerritory": "state",
                         "Confirmed": "cases",
                         "Deaths": "deaths",
                         "Cured": "recoveries"},
              inplace = True)
    df["date"] = pd.to_datetime(df.Date, format = "%d/%m/%y").dt.date.astype(str)

    return df

def read_census_data():
    """
    Reads the 2011 Indian census data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_CENSUS)
    df.rename(columns = {"State / Union Territory": "state",
                         "Population": "population",
                         "Urban population": "urban_population",
                         "Gender Ratio": "gender_ratio"},
              inplace = True)

    df["area"] = df.Area.str.replace(",", "").str.split("km").str[0].astype(int)

    return df

def read_test_samples_data():
    """
    Reads the ICMR test samples data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_TESTS)
    df.drop(index = 0, inplace = True)
    df.rename(columns = {"TotalSamplesTested": "samples_tested"},
              inplace = True)
    df["date"] = pd.to_datetime(df.DateTime, format = "%d/%m/%y %H:%S").dt.date.astype(str)
    
    return df

def read_test_labs_data():
    """
    Reads the ICMR testing labs data and preprocesses it.
    """
    
    df = pd.read_csv(PATH_LABS, encoding = "ISO-8859-1")
    
    return df


# ## April-8: Testing
# ![test.jpeg](attachment:test.jpeg)
# 
# A lot of the Covid-19 metrics are in the form of a funnel. You need to have sufficient data at the top of the funnel for the outputs at the bottom to be reliable. So what is at the top of the funnel? Testing!
# 
# If India is not able to test everyone with symptoms, the cases are under-stated. There could be more cases but are unknown due to limited testing. So how is India managing this? Does India have sufficient labs and testing centres across the country to facilitate the testing of Covid-19 and minimize this unknown factor?
# 

# In[ ]:


df_covid = read_covid_data()
df_census = read_census_data()
df_testing = read_test_samples_data()
df_labs = read_test_labs_data()

df_country = df_covid.groupby("date")["cases"].sum().reset_index().merge(df_testing[["date", "samples_tested"]], on = "date")
df_country["lag_1_cases"] = df_country.cases.shift(1)
df_country["day_cases"] = df_country.cases - df_country.lag_1_cases
df_country["lag_1_samples_tested"] = df_country.samples_tested.shift(1)
df_country["day_samples_tested"] = df_country.samples_tested - df_country.lag_1_samples_tested

df_country = df_country[df_country.date >= "2020-03-18"]
df_country.dropna(subset = ["day_cases", "day_samples_tested"], inplace = True)
df_country["case_rate"] = df_country.day_cases / df_country.day_samples_tested

df_state = df_labs.groupby("state")["lab"].count().reset_index().rename(columns = {"lab": "labs"}).merge(df_census, on = "state")
df_state["people_per_lab"] = df_state.population / df_state.labs
df_state["area_per_lab"] = df_state.area / df_state.labs


# In[ ]:


source = ColumnDataSource(data = dict(
    date = df_country.date.values,
    day_cases = df_country.day_cases.values,
    day_samples_tested = df_country.day_samples_tested.values,
    case_rate = df_country.case_rate.values
))

tooltips_1 = [
    ("Date", "@date"),
    ("Samples Tested", "@day_samples_tested")
]

tooltips_2 = [
    ("Date", "@date"),
    ("Cases", "@day_cases")
]

tooltips_3 = [
    ("Date", "@date"),
    ("Case Rate", "@case_rate{0.00}")
]

v = figure(plot_width = 650, plot_height = 400, x_range = df_country.date.values, title = "Covid-19 cases and test from 19th March")
v.extra_y_ranges = {"Case Rate": Range1d(start = 0.0, end = 0.1)}

v1 = v.vbar(x = dodge("date", 0.25, range = v.x_range), top = "day_samples_tested", width = 0.2, source = source, color = "blue", legend_label = "Samples Tested")
v2 = v.vbar(x = dodge("date", -0.25, range = v.x_range), top = "day_cases", width = 0.2, source = source, color = "orange", legend_label = "Cases")
v3 = v.line("date", "case_rate", source = source, color = "red", y_range_name = "Case Rate", legend_label = "Case Rate")

v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1))
v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2))
v.add_tools(HoverTool(renderers = [v3], tooltips = tooltips_3))

v.xaxis.major_label_orientation = pi/4

v.xaxis.axis_label = "Date"
v.yaxis.axis_label = "Count"
v.add_layout(LinearAxis(y_range_name = "Case Rate", axis_label = "Case Rate"), "right")

v.legend.location = "top_left"

show(v)


# Its a good sign to see the number of samples being tested for Covid-19 is increasing at a good rate. But is it enough?   
# As per [ICMR](https://www.icmr.nic.in/) press releases, they claim that there are enough laboratories and testing kits to cater to anyone needing to be tested.
# 
# About 5% of samples being tested are turning out to be positive.
# 
# Let's look at how these laboratories are spread across the country. The list of [laboratories](https://covid.icmr.org.in/index.php/testing-labs-deatails) are ones stated by ICMR that conduct testing and collection of Covid-19 samples.
# 

# In[ ]:


h_mid = max(df_state.area_per_lab.values / 1000) / 2
v_mid = max(df_state.people_per_lab.values / 1000000) / 2

source = ColumnDataSource(data = dict(
    state = df_state.state.values,
    labs = df_state.labs.values,
    people_per_lab = df_state.people_per_lab.values / 1000000,
    area_per_lab = df_state.area_per_lab.values / 1000
))

source_labels = ColumnDataSource(data = dict(
    state = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].state.values,
    people_per_lab = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].people_per_lab.values / 1000000,
    area_per_lab = df_state[(df_state.people_per_lab >= v_mid * 1000000) | (df_state.area_per_lab >= h_mid * 1000)].area_per_lab.values / 1000
))

tooltips = [
    ("State", "@state"),
    ("Labs", "@labs"),
    ("People per Lab", "@people_per_lab{0.00} M"),
    ("Area per Lab", "@area_per_lab{0.00} K")
]

labels = LabelSet(x = "people_per_lab", y = "area_per_lab", text = "state", source = source_labels, level = "glyph", x_offset = -19, y_offset = -23, render_mode = "canvas")

v = figure(plot_width = 500, plot_height = 500, tooltips = tooltips, title = "People and Area per Lab by State")
v.circle("people_per_lab", "area_per_lab", source = source, size = 13, color = "blue", alpha = 0.41)

tl_box = BoxAnnotation(right = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "orange")
tr_box = BoxAnnotation(left = v_mid, bottom = h_mid, fill_alpha = 0.1, fill_color = "red")
bl_box = BoxAnnotation(right = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "green")
br_box = BoxAnnotation(left = v_mid, top = h_mid, fill_alpha = 0.1, fill_color = "orange")

v.add_layout(tl_box)
v.add_layout(tr_box)
v.add_layout(bl_box)
v.add_layout(br_box)

v.add_layout(labels)

v.xaxis.axis_label = "People per Lab (in Million)"
v.yaxis.axis_label = "Area per Lab (in Thousand sq km)"

show(v)


# Dividing the grid of people per lab and area per lab into four quadrants, it is ideal to push as many states into the bottom-left quadrant as possible.
# 
# * The **top-left** quadrant are states with high area per lab. It might take longer for people to reach a lab for testing Covid-19. Currently **Ladakh, Arunachal Pradesh and Chhattisgarh** lie here.
# * The **bottom-right** qudrant are states with high population per lab. A sudden outbreak could lead to long queues for testing Covid-19. Currently **Punjab, Jharkhand, Uttar Pradesh and Bihar** lie here.
# * The **top-right** quadrant are states with high area as well as population per lab. This is a danger zone and thankfully no state lies here though Chhattisgarh is close.
# * The **bottom-left** quadrant are states with low area as well as population per lab. The testing in these states are in good condition and an ideal scenario is to work towards pushing all the states in this quadrant. Rajasthan, Madhya Pradesh and Odisha are close to orange zones and its worth trying to add a lab in these states.
# 
# This can greatly help in understanding which states are in what condition to handle Covid-19. A further analysis can be done based on samples tested per state to quantify the quality of the labs. Please share if this dataset can be made available from any source.
# 

# ## April-7: Trajectory
# ![FTC.jpg](attachment:FTC.jpg)
# 
# You might have heard the phrase 'Flatten The Curve'. But what does it really mean? How can data identify if the curve is flattening or not?
# Let's look at the trend of all the states and the overall numbers of the country.
# 

# In[ ]:


df_covid = read_covid_data()

df = df_covid.copy()
df["log_cases"] = np.log10(df.cases)
df["lag_1_cases"] = df.groupby("state")["cases"].shift(1)
df["day_cases"] = df.cases - df.lag_1_cases
df["lag_1_day_cases"] = df.groupby("state")["day_cases"].shift(1)
df["lag_2_day_cases"] = df.groupby("state")["day_cases"].shift(2)
df["lag_3_day_cases"] = df.groupby("state")["day_cases"].shift(3)
df["lag_4_day_cases"] = df.groupby("state")["day_cases"].shift(4)
df["lag_5_day_cases"] = df.groupby("state")["day_cases"].shift(5)
df["lag_6_day_cases"] = df.groupby("state")["day_cases"].shift(6)
df["ma_7d_day_cases"] = df[["day_cases", "lag_1_day_cases", "lag_2_day_cases", "lag_3_day_cases",
                              "lag_4_day_cases", "lag_5_day_cases", "lag_6_day_cases"]].mean(axis = 1).values
df["log_ma_7d_day_cases"] = np.log10(df.ma_7d_day_cases)
df = df[df.state != "Unassigned"]
df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")
df = df[df.date >= pd.datetime(2020, 3, 21)]


# In[ ]:


v1 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 cases from 21st March")
v2 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 cases from 21st March")

tooltips = [
    ("State", "@state"),
    ("Date", "@date{%F}"),
    ("Cases", "@cases")
]
    
formatters = {
    "@date": "datetime"
}
    
for i in range(len(df.state.unique())):
    state = df.state.unique()[i]
    df_state = df[df.state == state]
    
    source = ColumnDataSource(data = dict(
        state = df_state.state.values,
        date = np.array(df_state.date.values, dtype = np.datetime64),
        cases = df_state.cases.values,
        log_cases = df_state.log_cases.values
    ))
    
    v1.line("date", "cases", source = source, color = Category20[20][i % 20])
    v2.line("date", "log_cases", source = source, color = Category20[20][i % 20])

v1.add_tools(HoverTool(tooltips = tooltips, formatters = formatters))
v2.add_tools(HoverTool(tooltips = tooltips, formatters = formatters))

v1.xaxis.axis_label = "Date"
v1.yaxis.axis_label = "Cases"

v2.xaxis.axis_label = "Date"
v2.yaxis.axis_label = "Cases (Log Scale)"

show(row(v1, v2))


# [John Burn-Murdoch](https://twitter.com/jburnmurdoch) from Financial Times has produced some great visualizations on Covid-19 of which the log-scale view of cases is very popular. That's the second graph above while the first one are raw values of cases.
# 
# Let's look at some particular states in detail:
# 
# **Kerala**: First state to get a case
# **Maharashtra, Tamil Nadu, Delhi**: Top-3 states with most cases
# 

# In[ ]:


state_list = ["Delhi", "Kerala", "Maharashtra", "Tamil Nadu"]

v1 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 cumulative cases since 21st March")
v2 = figure(plot_width = 333, plot_height = 333, x_axis_type = "datetime", title = "Covid-19 MA(7) day cases since 21st March")

tooltips_1 = [
    ("State", "@state"),
    ("Date", "@date{%F}"),
    ("Cases", "@cases")
]
    
tooltips_2 = [
    ("State", "@state"),
    ("Date", "@date{%F}"),
    ("Day Cases", "@ma_7d_day_cases")
]

formatters = {
    "@date": "datetime"
}

for i in range(len(state_list)):
    state = state_list[i]
    df_state = df[df.state == state]

    source = ColumnDataSource(data = dict(
        state = df_state.state.values,
        date = np.array(df_state.date.values, dtype = np.datetime64),
        cases = df_state.cases.values,
        log_cases = df_state.log_cases.values,
        ma_7d_day_cases = df_state.ma_7d_day_cases.values,
        log_ma_7d_day_cases = df_state.log_ma_7d_day_cases.values
    ))
    
    v1.line("date", "log_cases", source = source, color = Spectral4[i], legend_label = state)
    v2.line("date", "log_ma_7d_day_cases", source = source, color = Spectral4[i], legend_label = state)

v1.add_tools(HoverTool(tooltips = tooltips_1, formatters = formatters))
v2.add_tools(HoverTool(tooltips = tooltips_2, formatters = formatters))

v1.legend.location = "bottom_right"
v2.legend.location = "bottom_right"

v1.xaxis.axis_label = "Date"
v1.yaxis.axis_label = "Cases (Log Scale)"

v2.xaxis.axis_label = "Date"
v2.yaxis.axis_label = "MA(7) Day Cases (Log Scale)"

show(row(v1, v2))


# The first plot shows how the cumulative cases increase in log-scale. The second plot shows how the moving average of cases per day in last 7 days changes in log-scale.
# 
# Since Kerala has had cases since January, you can see that its curve has flattened out. The other three states are yet to see the flattening begin. Tamil Nadu looks the worst due to its steep rise in cases.
# 
# Note that Kerala fought the [Nipah virus in 2018](https://en.wikipedia.org/wiki/2018_Nipah_virus_outbreak_in_Kerala). Has that experience helped them against Covid-19?
# 
# Can we learn something from Kerala? Did the government take certain steps that helped the spread of Covid-19 or is this the gradual life we expect to see of the virus in India?
# 

# ## April-6: Funds
# ![fund.jpg](attachment:fund.jpg)
# 
# 
# We can all contribute to fighting against Covid-19 in some way or the other. The advancement of technology and easy of digital payments in India has exposed a large number of individual to contribute to funds with the click of a few buttons. Here is a list of funds available to which you can donate money to (in alphabetical order):
# 
# * Akshaya Patra: [Amazon Cares](https://www.akshayapatra.org/daan-ustaav-with-amazon-cares)
# * Building Dreams: [Razorpay](https://pages.razorpay.com/pl_EWNyAMQujbOKgR/view)
# * Elixir Ahmedabad: [Razorpay](https://pages.razorpay.com/pl_EW357Eyk0tOlaa/view)
# * Elixir Mumbai: [Razorpay](https://pages.razorpay.com/pl_EWwhkkXJ4tIsCG/view)
# * Elixir Vodadara: [Razorpay](https://pages.razorpay.com/pl_EWVhT1xvdhc8qD/view)
# * Give India: [Flipkart](https://flipkart.giveindia.org/)
# * Habitat for Humanity: [Amazon Cares](https://habitatindia.org/covid19appeal/)
# * Helpage India: [Official Website](https://www.helpageindia.org/covid-19/)
# * IAHV India: [Razorpay](https://pages.razorpay.com/pl_EXwrPCmXVbgCPM/view)
# * Jan Sahas: [Milaap](https://milaap.org/fundraisers/support-jan-sahas-social-development-society?community=10177) (Upto 23rd April, 2020) | [Razorpay](https://pages.razorpay.com/jansahasdonate)
# * Kriti: [Official Website](https://kriti.org.in/covid-relief.html)
# * KVN Foundation: [Razorpay](https://pages.razorpay.com/feedmybangalore)
# * Narayan Seva: [Razorpay](https://pages.razorpay.com/pl_EWqc9MjE5C5m9u/view)
# * NYDHEE: [Razorpay](https://pages.razorpay.com/nydheecovid19) (Upto 15th April, 2020)
# * OXFAM India: [Amazon Cares](https://donate.oxfamindia.org/coronavirus-amazoncares)
# * PharmEasy: [Razorpay](https://pages.razorpay.com/COVID-19-Mask-1)
# * PM Cares: [Official Website](https://www.pmindia.gov.in/en/news_updates/appeal-to-generously-donate-to-pms-citizen-assistance-and-relief-in-emergency-situations-fund-pm-cares-fund) | [Paytm](https://paytm.com/helpinghand/pm-cares-fund)
# * Razorpay: [Razorpay](https://pages.razorpay.com/razorpay-covid19-relief)
# * Responsenet: [Official Website](https://www.responsenet.org/donations-for-covid-19-migrant-workers-relief-fund/)
# * SAFA Society: [Razorpay](https://pages.razorpay.com/Covid19Relief) (Upto 22nd April, 2020)
# * Sangati: [Razorpay](https://pages.razorpay.com/pl_EW6aLym55b2cuT/view)
# * Samarthanam: [Official Website](https://samarthanam.org/covid19-rapid-response-for-relief-kit)
# * Seeds India: [Official Website](https://www.seedsindia.org/covid19/)
# * United Way Bengaluru: [Razorpay](https://pages.razorpay.com/pl_EXd8EirttNCYDF/view)
# * United Way Delhi: [Razorpay](https://pages.razorpay.com/pl_EVcjO765jZSsm9/view)
# * United Way Mumbai: [Amazon Cares](https://www.unitedwaymumbai.org/amazoncares)
# * World Vision: [Amazon Cares](https://www.worldvision.in/wvreliefresponse/index.aspx)
# * Feeding India: [Zomato](https://www.feedingindia.org/)
# 
# > ***No contribution is too little or too much***
# 
# Feel free to share more such funds that can be added to the list.
# 

# ## April-5: Gender
# ![gender-icons-wide.jpg](attachment:gender-icons-wide.jpg)
# 
# Does Covid-19 have an affinity to spread or infect individuals of a particular gender? Is there a difference in metrics across states with different gender ratios?
# 

# In[ ]:


df_covid = read_covid_data()
df_census = read_census_data()

df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "gender_ratio"]], on = "state")
df["death_rate"] = df.deaths / (df.deaths + df.recoveries)
df["rank_gender_ratio"] = df["gender_ratio"].rank(ascending = False)
df["rank_cases"] = df["cases"].rank(ascending = False)
df["rank_deaths"] = df["deaths"].rank(ascending = False)
df["rank_death_rate"] = df["death_rate"].rank(ascending = False)


# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    gender_ratio = df.gender_ratio.values,
    cases = df.cases.values,
    deaths = df.deaths.values,
    death_rate = df.death_rate.values
))

tooltips_1 = [
    ("state", "@state"),
    ("gender_ratio", "@gender_ratio"),
    ("cases", "@cases")
]

tooltips_2 = [
    ("state", "@state"),
    ("gender_ratio", "@gender_ratio"),
    ("deaths", "@deaths")
]

tooltips_3 = [
    ("state", "@state"),
    ("gender_ratio", "@gender_ratio"),
    ("death_rate", "@death_rate{0.000}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Gender Ratio vs Cases by State")
v1.circle("gender_ratio", "cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.xaxis.axis_label = "Gender Ratio"
v1.yaxis.axis_label = "Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Gender Ratio vs Deaths by State")
v2.circle("gender_ratio", "deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.xaxis.axis_label = "Gender Ratio"
v2.yaxis.axis_label = "Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Gender Ratio vs Death-Rate by State")
v3.circle("gender_ratio", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.xaxis.axis_label = "Gender Ratio"
v3.yaxis.axis_label = "Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its gender ratio and a Covid-19 metric (cases, deaths and death-rate for the three plots).   
# 
# The data points don't seem to exhibit any particular trend. Let's also look at the relative ranks of the gender ratio of states against the respective ranks in Covid-19 metrics.

# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    rank_gender_ratio = df.rank_gender_ratio.values,
    rank_cases = df.rank_cases.values,
    rank_deaths = df.rank_deaths.values,
    rank_death_rate = df.rank_death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Rank of Gender Ratio", "@rank_gender_ratio{0}"),
    ("Rank of Cases", "@rank_cases{0}")
]

tooltips_2 = [
    ("State", "@state"),
    ("Rank of Gender Ratio", "@rank_gender_ratio{0}"),
    ("Rank of Cases", "@rank_deaths{0}")
]

tooltips_3 = [
    ("State", "@state"),
    ("Rank of Gender Ratio", "@rank_gender_ratio{0}"),
    ("Rank of Death Rate", "@rank_death_rate{0}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Gender Ratio vs Cases by State")
v1.circle("rank_gender_ratio", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.x_range.flipped = True
v1.y_range.flipped = True
v1.xaxis.axis_label = "Rank of Gender Ratio"
v1.yaxis.axis_label = "Rank of Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Gender Ratio vs Deaths by State")
v2.circle("rank_gender_ratio", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.x_range.flipped = True
v2.y_range.flipped = True
v2.xaxis.axis_label = "Rank of Gender Ratio"
v2.yaxis.axis_label = "Rank of Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Gender Ratio vs Death-Rate by State")
v3.circle("rank_gender_ratio", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.x_range.flipped = True
v3.y_range.flipped = True
v3.xaxis.axis_label = "Rank of Gender Ratio"
v3.yaxis.axis_label = "Rank of Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its rank of gender ratio and rank of a Covid-19 metric (cases, deaths and death-rate for the three plots).
# 
# The Covid-19 metrics are pretty evenly spread across states irrespective of their relative gender ratio ranks. It would be very interesting to see and analyze the gender data of the individuals who have been infected with the virus. This is an open question and any dataset that could help with this analysis would be highly appreciated.
# 

# ## April-4: Monotonicity
# ![frequency.png](attachment:frequency.png)
# 
# Garbage-in Garbage-out. Before diving deeper into any dataset, it is crucial to understand it from a sanity perspective. In this case, the primary data is being publicly shared by the Government of India: https://www.mohfw.gov.in/ and is being maintained voluntarily by some amazing Kagglers [here](https://www.kaggle.com/sudalairajkumar/covid19-in-india).
# 
# While we must trust that the numbers shared by the government are their best efforts of providing correct data, can we at least perform some basic checks to see if the data is valid as per expectations. Is the data being updated regularly or daily? Are there some lags in publishing the data since even the government would be scrambling to compile this dataset from multiple sources across the country? Is the data clean enough to analyze trends and create features for deeper study?
# 

# In[ ]:


df_covid = read_covid_data()

df = df_covid.copy()
df = df[df.state != "Unassigned"]
df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")


# In[ ]:


tab_list = []

for state in sorted(df.state.unique()):
    df_state = df[df.state == state]
    
    source = ColumnDataSource(data = dict(
        date = np.array(df_state.date.values, dtype = np.datetime64),
        cases = df_state.cases.values,
        deaths = df_state.deaths.values,
        recoveries = df_state.recoveries.values
    ))
    
    tooltips_1 = [
        ("Date", "@date{%F}"),
        ("Cases", "@cases")
    ]
    
    tooltips_2 = [
        ("Deaths", "@deaths")
    ]
    
    tooltips_3 = [
        ("Recoveries", "@recoveries")
    ]
    
    formatters = {
        "@date": "datetime"
    }
    
    v = figure(plot_width = 650, plot_height = 400, x_axis_type = "datetime", title = "Covid-19 metrics over time")
    v1 = v.line("date", "cases", source = source, color = "blue", legend_label = "Cases")
    v2 = v.line("date", "deaths", source = source, color = "red", legend_label = "Deaths")
    v3 = v.line("date", "recoveries", source = source, color = "green", legend_label = "Recoveries")
    v.legend.location = "top_left"
    v.add_tools(HoverTool(renderers = [v3], tooltips = tooltips_3, formatters = formatters, mode = "vline"))
    v.add_tools(HoverTool(renderers = [v2], tooltips = tooltips_2, formatters = formatters, mode = "vline"))
    v.add_tools(HoverTool(renderers = [v1], tooltips = tooltips_1, formatters = formatters, mode = "vline"))
    v.xaxis.axis_label = "Date"
    v.yaxis.axis_label = "Count"
    tab = Panel(child = v, title = state)
    tab_list.append(tab)

tabs = Tabs(tabs = tab_list)
show(tabs)


# Each tab displays a cumulative plot of the state's total number of cases, deaths and recoveries.
# 
# Since the data is aggregated in a cumulative fashion, we should expect monotonicity for each state. The only inconsistency is with Rajasthan data.   
# I've opened a [thread](https://www.kaggle.com/sudalairajkumar/covid19-in-india/discussion/141379) to resolve this data issue.
# 
# Otherwise it looks good to go!
# 

# ## April-3: Urbanization
# ![mumbai.jpg](attachment:mumbai.jpg)
# 
# While [population](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-1:-Population) and [density](https://www.kaggle.com/rohanrao/india-s-fight-against-covid-19-april-2020#April-2:-Density) looks at an entire area as a whole, urbanization enables dissecting an area based on the characteristic of development.
# 
# So how do more urban areas compare against rural ones? Does Covid-19 have any effect based on urbanization? Is it likely to spread more in urban areas due to presence of airports, railways and higher social connectivity? Or is it likely to affect the rural areas due to limited healthcare facilities and medical supplies?
# 

# In[ ]:


df_covid = read_covid_data()
df_census = read_census_data()

df_census["urbanization"] = df_census.urban_population / df_census.population

df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "urbanization"]], on = "state")
df["death_rate"] = df.deaths / (df.deaths + df.recoveries)
df["rank_urbanization"] = df["urbanization"].rank(ascending = False)
df["rank_cases"] = df["cases"].rank(ascending = False)
df["rank_deaths"] = df["deaths"].rank(ascending = False)
df["rank_death_rate"] = df["death_rate"].rank(ascending = False)


# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    urbanization = df.urbanization.values,
    cases = df.cases.values,
    deaths = df.deaths.values,
    death_rate = df.death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Urbanization", "@urbanization{0.00}"),
    ("Cases", "@cases")
]

tooltips_2 = [
    ("State", "@state"),
    ("Urbanization", "@urbanization{0.00}"),
    ("Deaths", "@deaths")
]

tooltips_3 = [
    ("State", "@state"),
    ("Urbanization", "@urbanization{0.00}"),
    ("Death Rate", "@death_rate{0.000}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Urbanization vs Cases by State")
v1.circle("urbanization", "cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.xaxis.axis_label = "Urbanization"
v1.yaxis.axis_label = "Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Urbanization vs Deaths by State")
v2.circle("urbanization", "deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.xaxis.axis_label = "Urbanization"
v2.yaxis.axis_label = "Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Urbanization vs Death-Rate by State")
v3.circle("urbanization", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.xaxis.axis_label = "Urbanization"
v3.yaxis.axis_label = "Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its urbanization and a Covid-19 metric (cases, deaths and death-rate for the three plots).   
# 
# There seems to be a little spike in the mid-range of urbanization. Let's look at the relative ranks of the urbanizaton of states against the respective ranks in Covid-19 metrics and check if the spikes still exist.
# 

# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    rank_urbanization = df.rank_urbanization.values,
    rank_cases = df.rank_cases.values,
    rank_deaths = df.rank_deaths.values,
    rank_death_rate = df.rank_death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Rank of Urbanization", "@rank_urbanization{0}"),
    ("Rank of Cases", "@rank_cases{0}")
]

tooltips_2 = [
    ("State", "@state"),
    ("Rank of Urbanization", "@rank_urbanization{0}"),
    ("Rank of Deaths", "@rank_deaths{0}")
]

tooltips_3 = [
    ("State", "@state"),
    ("Rank of Urbanization", "@rank_urbanization{0}"),
    ("Rank of Death Rate", "@rank_death_rate{0}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Urbanization vs Cases by State")
v1.circle("rank_urbanization", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.x_range.flipped = True
v1.y_range.flipped = True
v1.xaxis.axis_label = "Rank of Urbanization"
v1.yaxis.axis_label = "Rank of Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Urbanization vs Deaths by State")
v2.circle("rank_urbanization", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.x_range.flipped = True
v2.y_range.flipped = True
v2.xaxis.axis_label = "Rank of Urbanization"
v2.yaxis.axis_label = "Rank of Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Urbanization vs Death-Rate by State")
v3.circle("rank_urbanization", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.x_range.flipped = True
v3.y_range.flipped = True
v3.xaxis.axis_label = "Rank of Urbanization"
v3.yaxis.axis_label = "Rank of Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its rank of urbanization and rank of a Covid-19 metric (cases, deaths and death-rate for the three plots).
# 
# The plot has values all over the place and we don't see the spikes anymore so at least as of now it seems like urbanization doesn't impact Covid-19 much.   
# How else can we confirm this? What other approaches and ways could we slice the data in?
# 

# ## April-2: Density
# ![density.jpg](attachment:density.jpg)
# 
# [Physical Distancing (or commonly called Social Distancing)](https://en.wikipedia.org/wiki/Social_distancing) has been the immediate strategy taken by many countries to curb the rapid spread of Covid-19. Since it involves minimizes contact with humans, how feasible is it in highly dense areas?
# 
# India is among the [Top-20 densely populated countries](https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population_density) of the world. Hence most of India is dense by default. But how has it affected the spread of Covid-19? Is it better and safer in less dense areas?
# 
# Identifying how density affects Covid-19 can help authorities prioritize physical distancing better.
# 

# In[ ]:


df_covid = read_covid_data()
df_census = read_census_data()

df_census["density"] = df_census.population / df_census.area

df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "density"]], on = "state")
df["death_rate"] = df.deaths / (df.deaths + df.recoveries)
df["rank_density"] = df["density"].rank(ascending = False)
df["rank_cases"] = df["cases"].rank(ascending = False)
df["rank_deaths"] = df["deaths"].rank(ascending = False)
df["rank_death_rate"] = df["death_rate"].rank(ascending = False)


# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    density = df.density.values,
    cases = df.cases.values,
    deaths = df.deaths.values,
    death_rate = df.death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Density", "@density{0}"),
    ("Cases", "@cases")
]

tooltips_2 = [
    ("State", "@state"),
    ("Density", "@density{0}"),
    ("Deaths", "@deaths")
]

tooltips_3 = [
    ("State", "@state"),
    ("Density", "@density{0}"),
    ("Death Rate", "@death_rate{0.000}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Density vs Cases by State")
v1.circle("density", "cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.xaxis.axis_label = "Density"
v1.yaxis.axis_label = "Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Density vs Deaths by State")
v2.circle("density", "deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.xaxis.axis_label = "Density"
v2.yaxis.axis_label = "Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Density vs Death-Rate by State")
v3.circle("density", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.xaxis.axis_label = "Density"
v3.yaxis.axis_label = "Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its density and a Covid-19 metric (cases, deaths and death-rate for the three plots).   
# 
# Since many of the density number are in a smaller range, we can look at the relative ranks of the density of states against the respective ranks in Covid-19 metrics.
# 

# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    rank_density = df.rank_density.values,
    rank_cases = df.rank_cases.values,
    rank_deaths = df.rank_deaths.values,
    rank_death_rate = df.rank_death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Rank of Density", "@rank_density{0}"),
    ("Rank of Cases", "@rank_cases{0}")
]

tooltips_2 = [
    ("State", "@state"),
    ("Rank of Density", "@rank_density{0}"),
    ("Rank of Deaths", "@rank_deaths{0}")
]

tooltips_3 = [
    ("State", "@state"),
    ("Rank of Density", "@rank_density{0}"),
    ("Rank of Death Rate", "@rank_death_rate{0}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Density vs Cases by State")
v1.circle("rank_density", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.x_range.flipped = True
v1.y_range.flipped = True
v1.xaxis.axis_label = "Rank of Density"
v1.yaxis.axis_label = "Rank of Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Density vs Deaths by State")
v2.circle("rank_density", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.x_range.flipped = True
v2.y_range.flipped = True
v2.xaxis.axis_label = "Rank of Density"
v2.yaxis.axis_label = "Rank of Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Density vs Death-Rate by State")
v3.circle("rank_density", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.x_range.flipped = True
v3.y_range.flipped = True
v3.xaxis.axis_label = "Rank of Density"
v3.yaxis.axis_label = "Rank of Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its rank of density and rank of a Covid-19 metric (cases, deaths and death-rate for the three plots).
# 
# It's hard to conclude anything for density looking at these. These look fairly distributed, right?   
# At least at a state macro-level it doesn't seem to be particularly causing any effect on the spread of Covid-19. Not sure if that is a positive or negative.
# 
# It might be worth exploring the same at a more granular geography like city since some of the recent outbreaks in India have come from populated and dense cities like Delhi, Mumbai. Consider this as an open call for the search and availability of such a dataset.
# 

# ## April-1: Population
# ![population.jpg](attachment:population.jpg)
# 
# Since Covid-19 is an airborne disease and can easily spread from human to human in close contact, does this put populated areas at higher risk? Should we expect to see higher cases and fatalities in populated areas? These are intuitive assumptions.
# 
# Identifying how population affects Covid-19 can help authorities plan processes better in fighting back.
# 

# In[ ]:


df_covid = read_covid_data()
df_census = read_census_data()

df = df_covid.groupby("state")["cases", "deaths", "recoveries"].max().reset_index().merge(df_census[["state", "population"]], on = "state")
df["death_rate"] = df.deaths / (df.deaths + df.recoveries)
df["rank_population"] = df.population.rank(ascending = False)
df["rank_cases"] = df.cases.rank(ascending = False)
df["rank_deaths"] = df.deaths.rank(ascending = False)
df["rank_death_rate"] = df.death_rate.rank(ascending = False)


# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    population = df.population.values / 1000000,
    cases = df.cases.values,
    deaths = df.deaths.values,
    death_rate = df.death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Population", "@population{0.00} M"),
    ("Cases", "@cases")
]

tooltips_2 = [
    ("State", "@state"),
    ("Population", "@population{0.00} M"),
    ("Deaths", "@deaths")
]

tooltips_3 = [
    ("State", "@state"),
    ("Population", "@population{0.00} M"),
    ("Death Rate", "@death_rate{0.000}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Population vs Cases by State")
v1.circle("population", "cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.xaxis.axis_label = "Population"
v1.yaxis.axis_label = "Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Population vs Deaths by State")
v2.circle("population", "deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.xaxis.axis_label = "Population"
v2.yaxis.axis_label = "Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Population vs Death-Rate by State")
v3.circle("population", "death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.xaxis.axis_label = "Population"
v3.yaxis.axis_label = "Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its population and a Covid-19 metric (cases, deaths and death-rate for the three plots).   
# 
# Nothing very clear to take away from this. Instead we can look at the relative ranks of the population of states against the respective ranks in Covid-19 metrics.

# In[ ]:


source = ColumnDataSource(data = dict(
    state = df.state.values,
    rank_population = df.rank_population.values,
    rank_cases = df.rank_cases.values,
    rank_deaths = df.rank_deaths.values,
    rank_death_rate = df.rank_death_rate.values
))

tooltips_1 = [
    ("State", "@state"),
    ("Rank of Population", "@rank_population{0}"),
    ("Rank of Cases", "@rank_cases{0}")
]

tooltips_2 = [
    ("State", "@state"),
    ("Rank of Population", "@rank_population{0}"),
    ("Rank of Deaths", "@rank_deaths{0}")
]

tooltips_3 = [
    ("State", "@state"),
    ("Rank of Population", "@rank_population{0}"),
    ("Rank of Death Rate", "@rank_death_rate{0}")
]

v1 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_1, title = "Ranks of Population vs Cases by State")
v1.circle("rank_population", "rank_cases", source = source, size = 13, color = "blue", alpha = 0.41)
v1.x_range.flipped = True
v1.y_range.flipped = True
v1.xaxis.axis_label = "Rank of Density"
v1.yaxis.axis_label = "Rank of Cases"

v2 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_2, title = "Ranks of Population vs Deaths by State")
v2.circle("rank_population", "rank_deaths", source = source, size = 13, color = "red", alpha = 0.41)
v2.x_range.flipped = True
v2.y_range.flipped = True
v2.xaxis.axis_label = "Rank of Density"
v2.yaxis.axis_label = "Rank of Deaths"

v3 = figure(plot_width = 200, plot_height = 200, tooltips = tooltips_3, title = "Ranks of Population vs Death-Rate by State")
v3.circle("rank_population", "rank_death_rate", source = source, size = 13, color = "orange", alpha = 0.41)
v3.x_range.flipped = True
v3.y_range.flipped = True
v3.xaxis.axis_label = "Rank of Density"
v3.yaxis.axis_label = "Rank of Death Rate"

show(row(v1, v2, v3))


# Every circle represent a state's position based on its rank of population and rank of a Covid-19 metric (cases, deaths and death-rate for the three plots).
# 
# These first rank scatterplot shows that higher populated states have higher number of Covid-19 cases, and also to some extent the same with deaths and death-rate. Since the deaths are low in number (I pray it continues to be so), only time will tell how this impacts in the future and whether we will continue seeing such a pattern.
# 
# I'm not sure how useful population as a factor is going to be in the curbing of the disease. It is probably more useful in certain activities around the fight against the disease. Like using it to forecast cases and requirements of testing kits so that inventory is in control.
# 

# ## WIP
# * Age
# * Neighbouring states
# * Neighbouring countries
# * Temperature
# * Growth Rate
# * Janata Curfew
# * Lockdown
# * Forecasting
# * Supply chain
# * AQI
# 
# Please share more ideas on areas of analysis to cover.
# 

# ## RIP Covid-19
# > ***Lets contribute in any way possible.   
# > Lets come together as a nation.   
# > Lets beat Covid-19 together.   
# > Lets help the world.***
