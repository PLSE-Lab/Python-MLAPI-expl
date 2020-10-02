#!/usr/bin/env python
# coding: utf-8

# ### COVID-19(+) Interactions Within Air Travel
# Modeling potential interactions between healthy individuals and those carrying COVID-19, denoted hereafter as (+), has been identified as a key methodology in the effort to predict, combat, and respond to COVID-19. In order to contribute to this effort within the domain of airline travel, this dataset allows users to see all flights during the time period from 01MAR-14APR where airline passengers may have come in contact with a COVID-19(+) TSA Screening Agent during their presumed [incubation period](https://www.cdc.gov/coronavirus/2019-ncov/hcp/faq.html), 7 days, before that agent went in quarantine.
# 
# ### Acknowledgements
# - **Federal Data Partner:** [Transportation Security Administration](https://www.tsa.gov/coronavirus) (TSA).
# - **Industry Data Partner:** [Airline Data Inc.](https://www.airlinedata.com/)
# 
# 
# ### Inspiration
# 
# The [CORD-19 Research Challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) has been a great inspiration for this effort. Its focus on natural language processing has prompted the need for additional efforts in other statistical machine learning methods, such as those used in the [UNCOVER COVID-19 Challenge](https://www.kaggle.com/roche-data-science-coalition/uncover). With COVID-19 research as a global focal point, I hope that this dataset provides researchers with another set of features to help build models towards finding answers. 
# 
# ### Methodology
# Airline Data Inc. provided airline schedule information for the time period of 01MAR-14APR. This is one of the data products available as a part of their [Data Hub](https://www.airlinedata.com/the-hub/). The airline schedule includes information on future and historical airline flights updated in real-time as it is filed by the airlines. This data provides access to origins and destinations, flight times, aircraft types, seats, customized route mapping, and much more. For this work, we focused on getting flight information to include terminals and carriers in order to determine potential contact of passengers and, at the time, unknowingly COVID-19(+) TSA agents. Airline Data Inc. additionally provided the T100 data from March and April of last year. The T100 provides information on particular routes (ORD-&gt;JFK) for U.S. domestic and international air service reported by carriers. This dataset includes passenger counts, available seats, load factors, equipment types, cargo, and other operating statistics. These datasets were combined to estimate the number of passengers flying various routes thought the time period in question. Undoubtedly these numbers are much lower than those of the previous year, but we make the assumption that airline travel declined in a relatively equal proportions across the US, making the load factors for last year comparatively accurate. Since the T100 data is only released on a monthly basis, these figures will not be able to be updated until the coming months. 
# 
# The Transportation Security Administration posted publicly on their website a [list of all Screening and Baggage Officers](https://www.tsa.gov/coronavirus) who tested positive for COVID-19. This list included the airport they worked in, their last day of work, and their work location with shift information. This data was taken and used to down-select the data from Airline Data Inc. to only include those flights that met the following criteria:
# - Origin airport with COVID-19(+) TSA Officer
# - Flight took off (the flight schedule data will show all *potential* flights even those that do not take off)
# - TSA Officer on shift at time of departure
# - TSA Officer working in terminal from which the flight departed

# # Exploring the Data
# This notebook seeks to provide some exploratory data analysis (EDA) on the set with a focus on charting, mapping, and visulizing the movement of persons who may have come in contact with COVID-19(+) TSA Screening Agents. 
# 
# ## Inital Summaries and Charts
# 
# ### Break Down of Funnel from All Flights to Those with Highest Potential Contact
# The bottom of this funnel encompasses the data shared in this dataset.

# In[ ]:


import pandas as pd
import plotly.express as px
summary_df = pd.read_csv("../input/us-flights-with-coivid19-tsa-screening-officer/Summary_Stats_TSA_Positive.csv",index_col=0).T
summary_df["number_seats"] = summary_df["number_seats"] / 100
summary_df.columns = ["Number of Flights Departing", "Total Seats in Hundreds"]
formated_summary = summary_df.stack().reset_index()
formated_summary.columns = ["stage", "feature", "number"]
fig = px.funnel(formated_summary, x='number', y='stage', color='feature')
fig.show()


# ### Pandas Profile
# [Pandas Profile](https://github.com/pandas-profiling/pandas-profiling) on minimal mode offers a quick and expansive EDA on the variables present in the data.

# In[ ]:



import pandas_profiling 
flights_df = pd.read_csv("../input/us-flights-with-coivid19-tsa-screening-officer/Flights_with_TSA_Contact.csv")
flights_df.profile_report(title='Profiling Report', html={'style':{'full_width':True}}, progress_bar=False, minimal=True)


# ## Summary Stats by Groups
# This seciton uses [IpyAgGrids](https://www.kaggle.com/dannellyz/ipyaggrids-interactive-and-editable-dataframes)

# In[ ]:


get_ipython().system('pip install -q ipyaggrid')

from ipyaggrid import Grid

def simple_grid(df):

    column_defs = [{'headername':c,'field': c} for c in df.columns]

    grid_options = {
        'columnDefs' : column_defs,
        'enableSorting': True,
        'enableFilter': True,
        'enableColResize': True,
        'enableRangeSelection': True,
        'rowSelection': 'multiple',
    }

    g = Grid(grid_data=df,
             grid_options=grid_options,
             quick_filter=True,
             show_toggle_edit=True,
             sync_on_edit=True,
             export_csv=True,
             export_excel=True,
             theme='ag-theme-balham',
             show_toggle_delete=True,
             columns_fit='auto',
             index=False)
    return g


# ### Grouped by Origin Airport

# In[ ]:


passenger_features = [("flight_departed","sum"), ("number_seats","sum"), ("load_factor", "mean"), ("weighted_seats", "sum")]
outbound_columns = ["Origin Airport", "Total Departed Flights", "Total Potential Passengers", "Average Load Factor", "Total Scaled Passengers", "Total Destination Airports"]
outbound_flights = flights_df.groupby("origin_airport").agg(["sum", "nunique", "mean"])[passenger_features]
outbound_flights.columns = outbound_flights.columns.get_level_values(0)
outbound_flights = outbound_flights.merge(flights_df.groupby("origin_airport").nunique()["dest_airport"], left_index=True, right_index=True).reset_index()
outbound_flights.columns = outbound_columns
outbound_flights["Total Potential Passengers"] = outbound_flights["Total Potential Passengers"].astype(int)
simple_grid(outbound_flights)


# ### Grouped by Destination Airport

# In[ ]:


passenger_features = [("flight_departed","sum"), ("number_seats","sum"), ("load_factor", "mean"), ("weighted_seats", "sum")]
inbound_columns = ["Destination Airport", "Total Arrived Flights", "Total Potential Passengers", "Average Load Factor", "Total Scaled Passengers", "Total Origin Airports"]
inbound_flights = flights_df.groupby("dest_airport").agg(["sum", "nunique", "mean"])[passenger_features]
inbound_flights.columns = inbound_flights.columns.get_level_values(0)
inbound_flights = inbound_flights.merge(flights_df.groupby("dest_airport").nunique()["origin_airport"], left_index=True, right_index=True).reset_index()
inbound_flights.columns = inbound_columns
inbound_flights["Total Potential Passengers"] = inbound_flights["Total Potential Passengers"].astype(int)
simple_grid(inbound_flights)


# ### Grouped by Route (Origin -> Destination)

# In[ ]:


passenger_features = [("flight_departed","sum"), ("number_seats","sum"), ("load_factor", "mean"), ("weighted_seats", "sum")]
route_columns = ["Origin Airport", "Destination Airport", "Total Arrived Flights", "Total Potential Passengers", "Average Load Factor", "Total Scaled Passengers"]
routes_flights = flights_df.groupby(["origin_airport", "dest_airport"]).agg(["sum", "nunique", "mean"])[passenger_features]
routes_flights.columns = routes_flights.columns.get_level_values(0)
routes_flights = routes_flights.reset_index()
routes_flights.columns = route_columns
routes_flights["Total Potential Passengers"] = routes_flights["Total Potential Passengers"].astype(int)
simple_grid(routes_flights)


# ### Routes with Scaled Passengers Organized by Destination

# In[ ]:


import plotly.express as px
fig = px.bar(routes_flights, x="Total Scaled Passengers", y="Destination Airport", color='Origin Airport', orientation='h',
             hover_data=["Total Scaled Passengers", "Average Load Factor"],
             height=1600,
             title='Routes')
fig.show()


# # Next Steps
# 1. Correlate these figures with COVID-19 diagnosis numbers
# 2. Combine with other features to explore interacitons
# 3. Provide Narrative Summary on these results
# 
# ## Please send along comments for other items that would be useful to your analysis!
