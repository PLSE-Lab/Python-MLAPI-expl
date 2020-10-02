#!/usr/bin/env python
# coding: utf-8

# # How is the World dealing with COVID-19
# #### Understanding the implementation of strategies and their effects on the spread of COVID-19

# ### Important Links
# 
# - [About COVID-19](https://www.who.int/emergencies/diseases/novel-coronavirus-2019)
# - [Wikipedia Page of ongoing pandemic](https://en.wikipedia.org/wiki/2019%E2%80%9320_coronavirus_pandemic_by_country_and_territory)

# ### Preventive Measures against COVID-19

# ![COVID Prevention Infographic](https://newsroom.clevelandclinic.org/wp-content/uploads/sites/4/2020/03/20-CCC-1869334-COVID-19-ProtectYourself-Infographic-2500x1969-1-1024x807.jpg)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import ipywidgets as widgets
from ipywidgets import interact_manual, interact


# In[ ]:


# Utility Functions

def load_data(path, enc="utf-8"):
    """
    Load and display basic info about the data
    """
    df = pd.read_csv(path, encoding=enc)
    print("Features :", list(df.columns))
    print("Number of Rows :", df.shape[0])
    print("Number of Columns :", df.shape[1])
    
    return df


# In[ ]:


# data from HDX
govt_measures = load_data("/kaggle/input/uncover/HDE/acaps-covid-19-government-measures-dataset.csv")
tests_performed = load_data("/kaggle/input/uncover/HDE/total-covid-19-tests-performed-by-country.csv")
covid_indicators = load_data("/kaggle/input/uncover/HDE/inform-covid-indicators.csv")
school_closures = load_data("/kaggle/input/uncover/HDE/global-school-closures-covid-19.csv")

# other data
test_vs_confirmed = load_data("/kaggle/input/uncover/our_world_in_data/tests-conducted-vs-total-confirmed-cases-of-covid-19.csv")
worldwide_data = load_data("/kaggle/input/uncover/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv")
who_report = load_data("../input/uncover/WHO/who-situation-reports-covid-19.csv")
wdi_health_systems = load_data("../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv")
economic_freedom = pd.read_excel("../input/economic-freedom-index-2020/economic_freedom_index2020_data.xls", enc='latin-1')


# ## The Response of Countries with regards to "Tests" conducted

# #### Data Preparation Methodology
# 
# - **Dataset 1 :** "/kaggle/input/uncover/our_world_in_data/tests-conducted-vs-total-confirmed-cases-of-covid-19.csv"
# - **Dataset 2 :** "/kaggle/input/uncover/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv"
# - **Dataset 1 Cleaning Steps**
#     - Imputation of #tets and #cases with 0
#     - Filling up the write "Codes" for countries without a code
#     - Removing all instances that talk about "International" entities(These do not have any tests performed as per the data)
# - **Dataset 2 Cleaning Steps**
#     - Remove rows with missing values
#     - In doing so, we end up removing the countries of Anguilla, Eritrea, Falkland_Islands and Cases on an international conveyance in Japan
#     - With more complete data in the future of this competition, I shall make an attempt to include these as well
# - Merge both the datasets on features of "code" and "date" (named differently in each of the individual datasets)
# - Convert type of "date" to datetime64
# - Select important features for analysis and create a **data subset**

# First 5 rows of the combined dataframe :

# In[ ]:


# Data Preparation
def change_entity(x):
    if(x.split("-")[0].strip() in entities):
        return (x.split("-")[0].strip())
    return x

def fill_codes(x):
    if(x=="Australia"):
        return "AUS"
    elif(x=="China"):
        return "CHN"
    elif(x=="Canada"):
        return "CAN"
    elif(x=="United States"):
        return "USA"
    else:
        return "INT"

# test_vs_confirmed
test_vs_confirmed["total_covid_19_tests"] = test_vs_confirmed["total_covid_19_tests"].fillna(0)
test_vs_confirmed["total_confirmed_cases_of_covid_19_cases"] = test_vs_confirmed["total_confirmed_cases_of_covid_19_cases"].fillna(0)
nomissing_code = test_vs_confirmed.dropna()
missing_code = test_vs_confirmed[test_vs_confirmed["code"].isna()]
entities = list(nomissing_code.entity.unique())
missing_code.entity = missing_code.entity.apply(change_entity)
missing_code.code = missing_code.entity.apply(fill_codes)
missing_code = missing_code[missing_code["entity"]!="International"]
test_vs_confirmed = pd.concat([nomissing_code, missing_code]).sort_values(by="entity")
# worldwide
worldwide_data = worldwide_data.dropna() 
# merge both
data = worldwide_data.merge(test_vs_confirmed, left_on=["countryterritorycode", "daterep"], right_on=["code", "date"], how="inner")
# select features
data = data[["date", "cases", "deaths", "popdata2018", "entity", "code", "total_covid_19_tests", "total_confirmed_cases_of_covid_19_cases"]]
data["date"] = pd.to_datetime(data["date"])
data.head()


# In[ ]:


# scatter plot
data=data.groupby(['date', 'cases', 'deaths', 'popdata2018', 'entity', 'code'],
             as_index=False).agg({'total_covid_19_tests':'sum',
                                  'total_confirmed_cases_of_covid_19_cases':'sum'})
temp = data[data.total_confirmed_cases_of_covid_19_cases!=0]    
temp = temp[temp.total_covid_19_tests!=0]
temp = temp.replace("United States", "United States of America")
pop_density = covid_indicators[["population_density", "country"]]
pop_density = pop_density.replace("Korea Republic of", "South Korea")
temp = temp.merge(pop_density, left_on=["entity"], right_on=["country"])
temp = temp.drop(["country"],axis=1)
temp["tests_by_cases"] = (temp["total_covid_19_tests"] / temp["total_confirmed_cases_of_covid_19_cases"])
temp = temp.drop([5,6,8,31]) # remove duplicate country values

def draw_scatter(temp, text):
    # visualization
    chart = alt.Chart(temp).mark_circle().encode(
        alt.X('total_covid_19_tests', scale=alt.Scale(zero=False)),
        alt.Y('total_confirmed_cases_of_covid_19_cases', scale=alt.Scale(zero=False, padding=1)),
        color=alt.Color('population_density:Q', scale=alt.Scale(scheme="viridis")),
        size=alt.Size('tests_by_cases', scale=alt.Scale(domain=[-1, 100])),
        #size='tests_by_cases',
        tooltip=['tests_by_cases:Q',
             'entity:N',
             'total_covid_19_tests:Q',
             'total_confirmed_cases_of_covid_19_cases:Q',
             'date:T']
    ).configure_mark(
        size=50
    ).properties(
        title="Tests Conducted vs Confirmed Cases - "+text,
        width=600,
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    ).configure_title(
        fontSize=30,
        font='Agency FB',
        anchor='start',
        color='black'
    )
    
    return chart


# The below graphs depicts a **scatter plot** that draws a relation between **total_covid_19_tests** and **total_confirmed_cases_of_covid_19_cases**.  Each point depicts a country and the size of the point is **directly proportional** to the **tests/cases** ratio. Each point is coloured based on the **population density**.
# 
# > **Tests/Cases Ratio :** (Total Covid 19 Tests) / (Total Confirmed Cases of Covid 19)
# 
# This ratio helps quantify the response of countries based on the number of tests conducted. For example, if this ratio is 30.0, it means that for every 1 confirmed case, 30 other tests were conducted. So, **higher this ratio, higher the response the country has taken in testing measures**(based on the assumptions that the provided data engulfs all possible tests undertaken in the country).  
# 
# - More tests conducted => Higher **tests/cases** ratio
# - Lesser confirmed cases => Higher **tests/cases** ratio
# - Higher **tests/cases** ratio => **Quicker response** by the country
# 
# **NOTE : With the help of day-to-day tests performed per country, there can be a better foothold to place this idea on.**
# 
# And if we assume that all these tests are taken **only once per individual**, it might sound like an intuitive idea that 
# 
# > In countries with high population density, the conduction of more tests would be a favourable situation to identify more positive cases and take the necessary measures to prevent community spread
# 
# ***The countries have been split across 3 separate graphs for better viewability*** =>

# In[ ]:


draw_scatter(temp[(temp.total_covid_19_tests > 0) & (temp.total_covid_19_tests <= 15000)], "Less than 15000 tests")


# **The above visualization is interactive**
# 
# - Countries like Malta, India and the Palestine have **(tests/cases)** ratio > 50

# In[ ]:


draw_scatter(temp[(temp.total_covid_19_tests > 15000) & (temp.total_covid_19_tests <= 50000)], "Between 15000 and 50000 tests")


# **The above visualization is interactive**
# 
# - Spain, which has been on the news for the past few weeks for the increasing number of cases has a pretty low (tests/cases) ratio of 2.68 OR *for every one person confirmed to be a positive COVID 19 patient, only about 3 more people were tested*
# - Belarus on the other hand has a ratio of close to 600

# In[ ]:


draw_scatter(temp[(temp.total_covid_19_tests > 50000)], "More than 50000 tests")


# **The above visualization is interactive**
# 
# - In China where the virus first originated, this ratio is as low as 4. Number of tests conducted here has been the largest among all nations as per the data. The low ratio can probably be attributed to the country being the first hotspot (that means the number of confirmed cases would be alarmingly high and the speed of response would be slower as this was where the virus first hit)
# - South Korea has conducted the second most tests among all nations in this data with a ratio of 36.6
# - In UAE, this ratio is the highest at almost 1500

# In[ ]:


draw_scatter(temp, "All Countries")


# ### Visualizing "Confirmed Cases" over the period, country-wise
# 
# This section encapsulates the code that will provide the trajectory of confirmed cases over the given time period in the data, provide the number of tests performed in a given country and the date when that record of number of tests was published.  
# 
# If you wish to identify these details for a particular country, you could **fork the kernel and run the following line of code** :
# 
# ```
# tests_and_cases(country_name)
# # replace country_name with the name of the country
# ```
# 
# 

# In[ ]:


def tests_and_cases(country):
    """
    country : Choose the country
    """
    alt.data_transformers.enable('default', max_rows=None)

    """
    input_dropdown = alt.binding_select(options=['Italy', 'China'])
    selection = alt.selection_single(fields=['entity'], bind=input_dropdown, name='Country of ')
    color = alt.condition(selection,
                    alt.Color('entity:N', legend=None),
                    alt.value('lightgray'))
    """

    dc = data[data.entity == country]

    # Tests Performed
    tests = list(dc[dc.total_covid_19_tests!=0].total_covid_19_tests.sort_values(ascending=False))[0]
    stamp = list(dc[dc.total_covid_19_tests!=0].date.sort_values(ascending=False))[0]
    month = (stamp.strftime("%B")+" "+stamp.strftime("%d"))

    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['date'], empty='none')

    graph = alt.Chart().mark_line().encode(
        x='date:T',
        y='cases:Q',
        #color=color,
        tooltip=['date:T', 'cases:Q']
    ).transform_timeunit(
        month='monthdate(date)'
    ).properties(
        title=("Confirmed Cases in "+country+" (Number of Tests: "+str(int(tests))+")"+" (Recorded: "+month+")"),
        width=600,
        height=300
    )

    selectors = alt.Chart().mark_point().encode(
        x='date:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    points = graph.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    text = graph.mark_text(align='left', dx=-15, dy=-10).encode(
        text=alt.condition(nearest, 'cases:Q', alt.value(' '))
    )

    rules = alt.Chart().mark_rule(color='gray').encode(
        x='date:T',
    ).transform_filter(
        nearest
    )

    # Combine all layers
    chart = alt.layer(graph, selectors, points, rules, text,
                       data=dc)

    chart.configure_title(
        fontSize=30,
        font='Agency FB',
        anchor='start',
        color='gray'
    )
    
    return chart


# **NOTE : All graphs in this section are interactive**  
# A few examples =>

# In[ ]:


tests_and_cases("China")


# - As of March 25, China has been successful in flattening the curve
# - The number of tests were recorded at 3,20,000 as on February 24
# - By this time, the number of cases had already begun to fall as shown in the graph

# In[ ]:


tests_and_cases("Italy")


# - As recorded on March 20, Italy has conducted 2,06,886 tests

# To know how the world is dealing with COVID-19, we first need to know ***what factors are to be dealt with if COVID-19 has to be beaten***. The next section is a study on the **Indicators affecting COVID-19**.
# 
# ---

# ## Understanding COVID-19 Indicators

# In this section, I have made an attempt at integrating **multiple data sources** and creating an interactive visualization to compare the effect of COVID-19 across multiple countries of the world with respect to several indicators that help differentiate the individualities of each country.

# #### Data Preparation Methodology
# 
# - **Dataset 1 :** "/kaggle/input/uncover/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv "
# - **Dataset 2 :** "/kaggle/input/uncover/HDE/inform-covid-indicators.csv"
# - **Dataset 3 :** "/kaggle/input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv"
# - **Dataset 4 :** "/kaggle/input/economic-freedom-index-2020/economic_freedom_index2020_data.xls"
# - Dataset 4 is a **private dataset** that I had created on 5th April, 2020 for the purpose of analysis. I shall make it public after adding relevant documentation to it and making it more usable for the Kaggle community
# - **Dataset 1 Cleaning Steps**
#     - Aggregate all cases and deaths, country-wise
#     - Clean values (remove underscores from names of countries)
#     - Rename certain country names
# - **Dataset 3 Cleaning Steps**
#     - Rename certain country names
#     - Select necessary features
# - **Dataset 4 Cleaning Steps**
#     - Rename certain country names
#     - Select necessary features
# - Merge all 4 datasets in separate steps (check code)
# - Remove unnecesary or redundant features
# 
# First 5 Rows of the cleaned dataframe :

# In[ ]:


# prepare data

def remove_underscores(x):
    return (x.replace("_", " "))

def bin_deaths(x):
    
    if(x<=1000):
        return ("Less than 1000(included)")
    elif(x>1000 and x<=2000):
        return ("1000 to 2000(included)")
    elif(x>2000 and x<=4000):
        return ("2000 to 4000(included)")
    elif(x>4000 and x<=8000):
        return ("4000 to 8000(included)")
    else:
        return ("More than 8000")

src = worldwide_data.groupby(["countriesandterritories", "countryterritorycode", "popdata2018"],
                       as_index=False).agg({"cases":"sum", "deaths":"sum"})
src.columns = ["country", "code", "popdata2018", "cases", "deaths"]
src.country = src.country.apply(remove_underscores)
covid_indicators = covid_indicators.replace('Korea Republic of', 'South Korea')
src = covid_indicators.merge(src, left_on=["country", "iso3"], right_on=["country", "code"], how="inner")
src = src.drop(["iso3"], axis=1)
wdi_health_systems = wdi_health_systems.replace("United States", "United States of America")
wdi_health_systems = wdi_health_systems.replace('Korea, Rep.', "South Korea")
wdi_health_systems = wdi_health_systems[['World_Bank_Name','Health_exp_pct_GDP_2016',
 'Health_exp_public_pct_2016',
'Health_exp_out_of_pocket_pct_2016',
 'per_capita_exp_PPP_2016',
 'External_health_exp_pct_2016',
 'Physicians_per_1000_2009-18',
 'Nurse_midwife_per_1000_2009-18',
 'Specialist_surgical_per_1000_2008-18'
]]
src = src.merge(wdi_health_systems, left_on=["country"], right_on=["World_Bank_Name"], how="inner")
src = src.drop(["inform_health_conditions", "current_health_expenditure_per_capita"], axis=1)
src = src.replace("No data", 0)
src = src.replace("x", 0)
src.population_living_in_urban_areas = src.population_living_in_urban_areas.astype('float')
src.proportion_of_population_with_basic_handwashing_facilities_on_premises = src.proportion_of_population_with_basic_handwashing_facilities_on_premises.astype('float')
src.mortality_rate_under_5 = src.mortality_rate_under_5.astype('float')
src.inform_access_to_healthcare = src.inform_access_to_healthcare.astype('float')
src.physicians_density = src.physicians_density.astype('float')
src["deaths_binned"] = src["deaths"].apply(bin_deaths)

selected_columns = ['World Rank',
       '2020 Score', 'Judical Effectiveness',
       'Government Integrity', 'Tax Burden', 'Fiscal Health', 'Financial Freedom',
       'Tax Burden % of GDP', 'Country',
       'Population (Millions)', 'GDP (Billions, PPP)',
       '5 Year GDP Growth Rate (%)', 'GDP per Capita (PPP)',
       'Unemployment (%)', 'Inflation (%)',
       'Public Debt (% of GDP)']
economic_freedom = economic_freedom[selected_columns]
economic_freedom = economic_freedom.replace('Korea, South', "South Korea")
economic_freedom = economic_freedom.replace('United States', "United States of America")
src = src.merge(economic_freedom, left_on="country", right_on="Country", how="inner")
src = src.drop(["Country", "popdata2018"], axis=1)
src.head()


# There are 44 indicators in this final dataframe.

# ### Correlation Heatmap
# 
# **NOTE :** Hover over each cell to view the correlation value

# In[ ]:


# correlation heatmap

cor_data = (src
              .corr().stack()
              .reset_index()
              .rename(columns={0: 'correlation', 'level_0': 'variable1', 'level_1': 'variable2'}))
cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)

base = alt.Chart(cor_data).encode(
    x='variable2:O',
    y='variable1:O',
    tooltip=['correlation_label:Q']
).properties(
    title="Correlation Heatmap (Hover over cell for correlation value)",
    width=700,
    height=700
)

cor_plot = base.mark_rect().encode(
    color=alt.Color('correlation:Q', scale=alt.Scale(scheme="inferno"))
)

cor_plot


# There is a strong positive correlation of 0.72 between **Cases** and **GDP(PPP)**!

# ### Interactive Visualization with the Indicators
# 
# The below visualization is interactive. It constructs a **Scatter Plot** that can be read as :
# - The X-axis can be any of the 44 indicators
# - The Y-axis can be either of **cases** or **deaths**
# - Each point depicts a country
# - The size of each point depicts the GDP of the country
# - The hue/color of each point depicts the number of deaths in the country
# 
# Usage :
# - Set the X_axis and Y_axis
# - Click on the **Run Interact** button to display visualization
# 

# In[ ]:


@interact_manual
def plot_scatter_chart(X_axis=list(src.select_dtypes('number').columns),
                       Y_axis=['cases', 'deaths']):
    
    xaxis=X_axis
    yaxis=Y_axis
    chart = alt.Chart(src).mark_circle().encode(
        alt.X(xaxis, scale=alt.Scale(zero=False)),
        alt.Y(yaxis, scale=alt.Scale(zero=False, padding=1)),
        color='deaths_binned',
        size=alt.Size('GDP (Billions, PPP)', scale=alt.Scale(domain=[-1, 10000])),
        tooltip=['country', 'GDP (Billions, PPP)', 'cases', 'deaths', xaxis]
    ).properties(
        title="Comparing "+"'"+yaxis+"'"+" with "+"'"+xaxis+"'",
        width=700,
        height=300
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    ).configure_title(
        fontSize=30,
        font='Agency FB',
        anchor='start',
        color='black'
    )
    
    return chart


# The above visualization can help unearth a lot of important finds with respect to the impact COVID-19 can have on countries.

# **EDIT :** After committing the kernel, I was unable to view the visualization in the Notebook's view mode. Therefore, if this problem arises for you too, **you can fork the kernel and watch the graph run in the edit mode**.  
# 
# As examples to what you can find, I have placed beneath a few forms of the visualization with different indicators on the x axis.

# In[ ]:


xaxis="Fiscal Health"
yaxis="cases"
chart = alt.Chart(src).mark_circle().encode(
    alt.X(xaxis, scale=alt.Scale(zero=False)),
    alt.Y(yaxis, scale=alt.Scale(zero=False, padding=1)),
    color='deaths_binned',
    size=alt.Size('GDP (Billions, PPP)', scale=alt.Scale(domain=[-1, 10000])),
    tooltip=['country', 'GDP (Billions, PPP)', 'cases', 'deaths', xaxis]
).properties(
    title="Comparing "+"'"+yaxis+"'"+" with "+"'"+xaxis+"'",
    width=700,
    height=300
).configure_axis(
    grid=False
).configure_view(
    strokeWidth=0
).configure_title(
    fontSize=30,
    font='Agency FB',
    anchor='start',
    color='black'
)

chart


# In[ ]:


xaxis="physicians_density"
yaxis="cases"
chart = alt.Chart(src).mark_circle().encode(
    alt.X(xaxis, scale=alt.Scale(zero=False)),
    alt.Y(yaxis, scale=alt.Scale(zero=False, padding=1)),
    color='deaths_binned',
    size=alt.Size('GDP (Billions, PPP)', scale=alt.Scale(domain=[-1, 10000])),
    tooltip=['country', 'GDP (Billions, PPP)', 'cases', 'deaths', xaxis]
).properties(
    title="Comparing "+"'"+yaxis+"'"+" with "+"'"+xaxis+"'",
    width=700,
    height=300
).configure_axis(
    grid=False
).configure_view(
    strokeWidth=0
).configure_title(
    fontSize=30,
    font='Agency FB',
    anchor='start',
    color='black'
)

chart


# In[ ]:


xaxis="inform_epidemic_vulnerability"
yaxis="cases"
chart = alt.Chart(src).mark_circle().encode(
    alt.X(xaxis, scale=alt.Scale(zero=False)),
    alt.Y(yaxis, scale=alt.Scale(zero=False, padding=1)),
    color='deaths_binned',
    size=alt.Size('GDP (Billions, PPP)', scale=alt.Scale(domain=[-1, 10000])),
    tooltip=['country', 'GDP (Billions, PPP)', 'cases', 'deaths', xaxis]
).properties(
    title="Comparing "+"'"+yaxis+"'"+" with "+"'"+xaxis+"'",
    width=700,
    height=300
).configure_axis(
    grid=False
).configure_view(
    strokeWidth=0
).configure_title(
    fontSize=30,
    font='Agency FB',
    anchor='start',
    color='black'
)

chart


# ---

# ## How Responsive have Governments across the World been?
# 
# In this section, I analyse the responsiveness of governments towards the spread of COVID-19.  The section is currently under progress.  
# 
# The Intention of this section will be to identify
# - Government Measures and Policies to tackle COVID-19
# - Government Responsiveness to the Pandemic
# - Effectiveness of strategies and measures

# In[ ]:


# prepare data

rem_cols = ['admin_level_name', 'pcode', 'alternative_source']
govt_measures = govt_measures.drop(rem_cols, axis=1)
govt_measures = govt_measures.replace('Korea Republic of', "South Korea")
govt_measures = govt_measures.replace('Viet Nam', "Vietnam")
govt_measures = govt_measures.fillna("Missing")
govt_measures = govt_measures.replace("Movement Restriction","Movement Restrictions")
govt_measures = govt_measures.replace("Movement restrictions","Movement Restrictions")
govt_measures = govt_measures.replace("Social and economic measures","Social and Economic Measures")
govt_measures = govt_measures.replace("Social distancing","Social Distancing")
govt_measures = govt_measures.replace("Public health measures","Public Health Measures")
govt_measures.head()


# In[ ]:


cats = []
count = []
for cat in list(govt_measures.category.unique()):
    cats.append(cat)
    count.append(govt_measures[govt_measures.category == cat].shape[0])
src = pd.DataFrame({"Category":cats, "Count":count})

alt.Chart(src).transform_joinaggregate(
    TotalCount='sum(Count)',
).transform_calculate(
    Percentage="datum.Count / datum.TotalCount"
).mark_bar(
    color="red",
    opacity=0.75
).configure_axis(
    grid=False
).encode(
    alt.X('Percentage:Q', axis=alt.Axis(format='.0%')),
    y='Category:N',
    tooltip=["Percentage:Q", "Count:Q"]
).properties(
    title="Distribution of Measures on the Basis of Categories",
    width=500,
    height=200
)


# In[ ]:


src = govt_measures.groupby(['category', 'targeted_pop_group'], as_index=False).agg({'measure':'count'})
alt.Chart(src).mark_bar().encode(
    x='sum(measure)',
    y='targeted_pop_group',
    color='category',
    tooltip=['targeted_pop_group', 'measure']
).properties(
    title="Distribution of Measures based on whether the measures were targeted at specific population groups",
    width=600,
    height=100
)


# In[ ]:


src = govt_measures.groupby(['country', 'region'], as_index=False).agg({'measure':'count'}).sort_values(by='measure', ascending=False).head(30)
bars = alt.Chart(src).mark_bar().encode(
    x='sum(measure)',
    y=alt.Y('country', sort='-x'),
)

text = bars.mark_text(
    align='left',
    baseline='middle',
    dx=3  # Nudges text to right so it doesn't appear on top of the bar
).encode(
    text='measure:Q'
)

(bars + text).properties(
    title = "Top 30 Countries in terms of the Number of Government Measures Undertaken in Response to COVID-19"
)


# In[ ]:


cats = []
count = []
for cat in list(govt_measures.measure.unique()):
    cats.append(cat)
    count.append(govt_measures[govt_measures.measure == cat].shape[0])
src = pd.DataFrame({"Measures":cats, "Count":count}).sort_values(by="Count", ascending=False).head(20)

alt.Chart(src).transform_joinaggregate(
    TotalCount='sum(Count)',
).transform_calculate(
    Percentage="datum.Count / datum.TotalCount"
).mark_bar(
    color="red",
    opacity=0.75
).configure_axis(
    grid=False
).encode(
    alt.X('Percentage:Q', axis=alt.Axis(format='.0%')),
    y=alt.Y('Measures:N', sort='-x'),
    tooltip=["Percentage:Q", "Count:Q"]
).properties(
    title="Most Common Measures Implemented across the World",
    width=700,
    height=300
)


# ### SECTION TO BE CONTINUED ...

# ### Acknowledgements
# - [Altair Official Documentation](https://altair-viz.github.io/index.html)
# - [Interactive Altair - Blog Post](https://matthewkudija.com/blog/2018/06/22/altair-interactive/)
# - [Visualizaing Correlation Structure of Weather Data](https://towardsdatascience.com/altair-plot-deconstruction-visualizing-the-correlation-structure-of-weather-data-38fb5668c5b1)
# - [Economic Freedom Index 2020 Rankings](https://www.heritage.org/index/?version=598)
# - [Interactive Control for Jupyter Notebooks](https://towardsdatascience.com/interactive-controls-for-jupyter-notebooks-f5c94829aee6)
