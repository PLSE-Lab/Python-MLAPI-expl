#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hello, this is an exploratory analysis of the Kiva datasets (in particular the _loans_ and _mpi region locations_ datasets). The notebook provides a high level overview of the loans data and starts to answer questions like: Who took out loans? What did they take them out  for? In what part of the world do people take out these loans? How many of the loans applied for are actually funded?
# 
# I hope it can help you become a little more familiar with the data!
# 
# ## Contents:
# 
# ### 1) Import Packages and Read in Data
# 
# ### 2) High Level Loan Data Exploration
# 
#         Based on this data, Kiva rejects less than 1% of all loans
# 
# ### 3) Analysis by Sector
# 
#         Loans made for Manufacturing have the highest funding rate, loans made for Entertainment have the lowest
# 
# ### 4) Analysis by Gender
# 
#         Women apply for (and receive) far more loans than men through Kiva (they're also funded at a higher rate)
# 
# ### 5) Analysis by Geography
# 
#         80% of Kiva loans go to East Asia & the Pacific, Sub-Saharan Africa, or Latin America & the Caribbean

# # 1) Import Packages and Read in Data

# In[1]:


import math
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from datetime import datetime
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tools
py.init_notebook_mode(connected=True)


# In[2]:


print("Provided datasets:\n")
loans = pd.read_csv("../input/kiva_loans.csv")
print("loans:\n", loans.shape)

mpi_region_locations = pd.read_csv("../input/kiva_mpi_region_locations.csv")
print("mpi_region_locations:\n", mpi_region_locations.shape)


# "loans" is the main file and contains information about each individual loan. We'll investigate it first, there will be a lot to learn about the data from this one dataset.

# In[3]:


loans.head()


# In[4]:


print("Columns in the dataset:\n", sorted(loans.columns.tolist()))


# # 2) High Level Loan Data Exploration
# In this section, we'll explore high level information about the loans made through Kiva:
# * How many loans were made to each partner?
# * How many people are associated with each partner and loan?
# * How long are the terms of the loans?
# * How many of the loans are actually funded?
# 
# ## How many loans were made to each partner?
# On Kiva, loans are made to partners, who often apply for and recieve more than one loan. In this dataset, partners can be identified by their unique **partner_id**, while loans can be identified by their unique **id**.

# In[5]:


loans_per_partner = loans.partner_id.value_counts()
loans_per_partner_no_outlier = loans_per_partner[loans_per_partner<100000]


# In[6]:


fig = tools.make_subplots(rows=2, cols=2,
                          subplot_titles=["Loans per partner",  "Loans per partner (upper outlier removed)", 
                                          "", ""],
                          shared_xaxes=True,)

# Left side
trace1 = go.Histogram(x=loans_per_partner.tolist(), 
                      marker=dict(color="#FF851B"), showlegend=False)
trace2 = go.Box(x=loans_per_partner.tolist(), boxpoints='all', orientation='h', 
                marker=dict(color="#FF851B"), showlegend=False)
# Right side
trace3 = go.Histogram(x=loans_per_partner_no_outlier.tolist(), 
                      marker=dict(color="#3D9970"), showlegend=False)
trace4 = go.Box(x=loans_per_partner_no_outlier.tolist(), boxpoints='all', orientation='h', 
                marker=dict(color="#3D9970"), showlegend=False)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 2, 2)

fig["layout"]["yaxis1"].update(dict(domain=[0.4, 1]))
fig["layout"]["yaxis2"].update(dict(domain=[0.4, 1]))
fig["layout"]["yaxis3"].update(dict(domain=[0, .4]), showticklabels=False)
fig["layout"]["yaxis4"].update(dict(domain=[0, .4]), showticklabels=False)

py.iplot(fig, filename="BasicStats")


# In[7]:


num_partners = len(loans.partner_id.unique())
num_loans = len(loans)
print("{} partners applied for loans".format(num_partners))
print("{:.2f} mean loans per partner".format(np.mean(loans_per_partner)))
print("{:.2f} median loans per partner".format(np.median(loans_per_partner)))


# There is one distant outlier partner with 108,000 loans, we removed that outlier in the chart on the right so the distribution becomes more clear. Almost all parnters receive between 1 and 1,000 loans, the median loans per partner is about 200.

# ## How many borrowers are associated with each loan?
# In this analysis, we're extrapolating from the **borrower_genders** column. As we can see by exploring the dataset manually, each entry in this column contains a list of genders associated with each loan.
# 
# We're making the assumption that "female, female" means that two people are associated with the partner/organization who borrowed money with the loan, and that both of those people are female. Similarly, we're assuming that "male, female" means that one male and one female are associated with a specific loan. We're assuming that "unknown" means that borrower didn't specify a gender, and we don't count "unknown" genders in our group size counts.

# In[8]:


def process_borrower_count(s):
    """
    Counts the number of genders listed in the `borrower_genders` column.
    If a borrower gender is unspecified, don't record the count.
    """
    borrower_list = [y.replace(" ", "") for y in s.split(",")]
    if len(borrower_list) == 1 and borrower_list[0]=="unknown":
        return None
    else:
        return len(borrower_list)

loans["borrower_count"] = loans["borrower_genders"].fillna("unknown").map(process_borrower_count)


# In[9]:


unique_borrower_counts = loans.pivot_table(index="borrower_count", values="partner_id", aggfunc=lambda x: len(x.unique())).reset_index()

trace = go.Bar(x=unique_borrower_counts["borrower_count"].tolist(),
               y=unique_borrower_counts["partner_id"].tolist())
layout = dict(
    title="Size of Borrower Groups",
    xaxis=dict(title="Number of People in Partner Group"),
    yaxis=dict(title="Number of Groups")
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)


# This chart counts each **partner_id** in the dataset only once, even if that partner_id is associated with multiple loans. It looks like the most common partner group is one (so most partners consist of one individual), while the largest partner groups are made up of fifty people (there are four partners with 50 people in the group).
# 
# Below, we'll take a look at a nonunique count of these **partner_id**s. From this, we'll be able to see how many loans groups of each size receive.

# In[10]:


fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=["Number of Loans to Groups of Each Size", "(with 550k single borrowers removed)"],
                          shared_xaxes=True,)

trace1 = go.Histogram(x=loans.borrower_count.dropna().tolist(), name="")
trace2 = go.Histogram(x=loans[loans["borrower_count"]>1].borrower_count.dropna().tolist(), name="")

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)

fig["layout"].update(showlegend=False,
                     width=1000, height=700,
                    xaxis1=dict(title="Number of People in Partner Groups"),
                    yaxis1=dict(title="Number of Loans"),
                    yaxis2=dict(title="Number of Loans"))

py.iplot(fig)


# As visible in the top chart, loans are overwhelmingly made to individuals rather than groups. 
# 
# In the bottom chart we removed the 500,000 loans made to individuals. With this view, we can see that when loans are made to groups, the groups are usually made up of five or fewer people. There were 31 loans made to groups made up of fifty people (the largest group size).

# ## How long are the terms of the loans?
# It might also be interesting to learn how long a typical loan is made for. We can figure this out using the **term_in_months** column in the loans dataset.

# In[11]:


loan_terms_months = pd.DataFrame(loans.term_in_months.value_counts().sort_index()).reset_index()
loan_terms_months.columns = ["months", "count"]

loan_terms_years = loan_terms_months.copy()
loan_terms_years["months"] = np.ceil(loan_terms_years/12)
loan_terms_years.columns = ["years", "count"]
loan_terms_years = loan_terms_years.sort_values(by="years", ascending=True)
loan_terms_years = loan_terms_years.pivot_table(index="years", values="count", aggfunc=np.sum).reset_index()
loan_terms_years["years"] = loan_terms_years["years"].apply(lambda x: "{} - ".format(str(int(x-1))) + str(int(x)))


# In[12]:


fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=["Loan terms (months)",  "Loans terms (years)"],
                          shared_xaxes=False,)

traceMonths = go.Bar(x=loan_terms_months["months"].tolist(), 
                     y=loan_terms_months["count"].tolist(),
                    showlegend=False,
                    text=loan_terms_months["months"].tolist(),
                    name="")
traceYears = go.Bar(x=loan_terms_years["years"].tolist(),
                    y=loan_terms_years["count"].tolist(),
                    showlegend=False,
                    text=loan_terms_years["years"].tolist(),
                   name="")

fig.append_trace(traceMonths, 1, 1)
fig.append_trace(traceYears, 2, 1)

fig["layout"]["yaxis1"].update(dict(domain=[0.55, 1]))
fig["layout"]["yaxis2"].update(dict(domain=[0, 0.45]))
fig["layout"].update(dict(width=1000,
                         height=700),
                    xaxis1=dict(tickvals=list(range(0,168,12))),
                    xaxis2=dict(tickangle=-55))

py.iplot(fig, filename="LoanTerms")


# Almost all of the loans are made for terms 0-2 years in length, the most common terms being 8 and 14 months.

# ## How many of the loans are actually funded?
# There are two columns which describe the amount of the loans, the **loan_amount** column and the **funded_amount** column. From these two columns, we can extrapolate the funding status of a loan into a few categories:
# * Full funding
#     * The loan_amount and the funded_amount are equal to each other
# * Not funded
#     * The funded_amount is equal to zero
# * Partial funding
#     * The funded_amount is greater than zero but less than the loan_amount
# * Overfunded
#     * The funded_amount is greater than the loan amount

# In[13]:


def process_funded(row):
    """
    Classify loans into one of the categories described above.
    """
    requested = row["loan_amount"]
    funded = row["funded_amount"]
    diff = requested - funded
    if diff == 0:
        val = "Full funding"
    elif diff > 0 and funded != 0:
        val = "Partial funding"
    elif funded == 0:
        val = "Not funded"
    elif funded > requested:
        val = "Overfunded"
    else:
        val = "error"
    return val

loans["funded"] = loans.apply(lambda x: process_funded(x), axis=1)


# In[14]:


funded_counts = loans["funded"].value_counts()
funded_total = np.sum(funded_counts)
funded_pcts = round(funded_counts/funded_total, 4)


# In[15]:


names = funded_pcts.index.tolist()
vals = funded_pcts.values.tolist()
colors = ["green", "orange", "red", "gray"]
overall_data = list()
for i in range(len(names)):
    trace = go.Bar(
        y=["Overall"],
        x=[vals[i]],
        text=funded_counts.values.tolist()[i],
        name=names[i],
        orientation='h',
        marker=dict(color=colors[i])
    )
    overall_data.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Funding Status of All Loans",
    width=1000, height=300,
)

fig = go.Figure(data=overall_data, layout=layout)
py.iplot(fig, filename='stacked-bar')


# About 93% of loans are funded and 6.5% are partially funded. Only about 0.5% of loans are not funded at all, there are 3383 of these loans.

# # 3) Analysis by Sector
# In this section, we'll take a look at how the **sector** column of the loans dataset, focusing on things like:
# * Which sectors are the loans made to?
# * How many loans are funded in each sector?
# 
# ## Which sectors are the loans made to?

# In[16]:


sectors = loans.sector.value_counts()

names = sectors.index.tolist()
vals = sectors.tolist()

sector_data = list()
for i in range(len(names)):
    trace = go.Bar(
        y=["Sectors"],
        x=[vals[i]],
#         text=vals[i],
        name=names[i],
        orientation='h',
    )
    sector_data.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Number of Loans per Sector",
    xaxis=dict(title="Number of Loans"),
    height=465, width=1000
)
fig = go.Figure(data=sector_data, layout=layout)
py.iplot(fig)


# The three most common Sectors _(Agriculture, Food, and Retail)_ make up over 60% of all loans.
# 
# The seven least common Sectors _(Wholesale, Entertainment, Manufacturing, Construction, Health, Arts, and Transportation)_ make up about 10% of all loans.

# ## How many loans are funded in each Sector?

# In[17]:


sector_counts = loans.pivot_table(index="sector", columns="funded", values="id", aggfunc=lambda x: len(x)).fillna(0)

# Create new df which contains "pct of total" instead of "count":
sector_totals = np.sum(sector_counts, axis=1)
sector_pcts = sector_counts.copy()
for col in sector_counts.columns:
    sector_pcts[col] = round(sector_counts[col]/sector_totals, 4)

# Reorganize each dataframe:
sector_pcts = sector_pcts[["Full funding", "Partial funding", "Not funded", "Overfunded"]].sort_values(by="Full funding")
sector_counts = sector_counts[["Full funding", "Partial funding", "Not funded", "Overfunded"]].loc[sector_pcts.index]


# In[18]:


sector_names = sector_pcts.index.tolist()
column_names = sector_pcts.columns.tolist()
colors = ["green", "orange", "red", "gray"]
sector_data = list()
for i in range(len(column_names)):
    trace = go.Bar(
        y=sector_names,
        x=sector_pcts[column_names[i]].tolist(),
        text=sector_counts[column_names[i]].tolist(),
        name=column_names[i],
        showlegend=False,
        orientation='h',
        marker=dict(color=colors[i])
    )
    sector_data.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Percent funded (by Sector)"
)

# fig = go.Figure(data=sector_data, layout=layout)
# py.iplot(fig, filename='stacked-bar')


# In[19]:


fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=["Percent funded (by Sector)", ""],
                          shared_xaxes=True,)

for x in overall_data:
    fig.append_trace(x, 1, 1)
for x in sector_data:
    fig.append_trace(x, 2, 1)

fig["layout"].update(barmode='stack')
fig["layout"]["yaxis2"].update(dict(domain=[0, 0.7]))
fig["layout"]["yaxis1"].update(dict(domain=[0.8, 1]))
fig['layout'].update(height=800, width=1000)


py.iplot(fig, filename='customizing-subplot-axes')


# It looks like the Manufacturing sector is the most likely to be funded, while the Entertainment sector is the least likely. Still, almost 90% of loans to the Entertainment sector are funded.
# 
# The most common three sectors _(Agriculture, Food, and Retail, which make up over 60% of all loans)_ are in the middle of the pack in terms of loan approval rate, though still boast between 91% and 94% fully funded loans.

# # 4) Analysis by Gender
# In this section, we'll utilize the **borrower_genders** column to analyze loans made to specified genders. Recall that the borrower_genders column is a list of genders containing either "female", "male", "unknown", or a mix of male and female. For this section, we'll group the genders into four buckets: Female, Male, Mixed, and Unknown.
# 
# * How many loans are made to each gender bucket?
# * How many loans are approved for partners of each gender bucket?

# ## How many loans are made to each gender bucket?

# In[20]:


# Genders
def process_genders(s):
    genders_list = list(set([y.replace(" ", "") for y in s.split(",")]))
    if len(genders_list) == 1:
        return genders_list[0]
    else:
        return "mixed"

loans["borrower_genders_processed"] = loans.borrower_genders.fillna("unknown").map(process_genders)


# In[21]:


loan_partner_genders = loans.pivot_table(index="borrower_genders_processed", values="partner_id", aggfunc=lambda x: len(x.unique())).reset_index()
loan_genders = loans["borrower_genders_processed"].value_counts()


# In[22]:


fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=["Gender",""],
                          shared_xaxes=True,)

traceUnique = go.Bar(x=loan_partner_genders["borrower_genders_processed"],
                     y=loan_partner_genders["partner_id"],
                     marker=dict(color=['#FFCDD2', "#A2D5F2", "#59606D"]),
                     showlegend=False,
                    )
traceLoans = go.Bar(x=loan_genders.index,
                    y=loan_genders.values,
                    marker=dict(color=['#FFCDD2', "#A2D5F2", "#59606D"]),
                    showlegend=False,
                    )

fig.append_trace(traceUnique, 1, 1)
fig.append_trace(traceLoans, 2, 1)

fig["layout"].update(dict(width=1000,
                         height=700),
                    yaxis1=dict(title="Number of Partners",
                                domain=[0.55, 1]),
                    yaxis2=dict(title="Number of Loans",
                                domain=[0, 0.45]))

py.iplot(fig, filename="LoanGenders")


# Above, the top chart displays a count of the number of partner groups in each gender bucket, while the bottom chart displays the number of loans to groups in each bucket.
# 
# We can see that the count of of Female and Male parter groups, without taking into account the number of loans issued to each group, is about even. However, in the bottom chart it becomes clear that groups made up of only Females receive (or maybe apply for) far more loans than groups made up of only Males.

# ## How many loans are approved for partners in each gender bucket?

# In[23]:


gender_counts = loans.pivot_table(index="borrower_genders_processed", columns="funded", values="id", aggfunc=lambda x: len(x)).fillna(0)

# Create new df which contains "pct of total" instead of "count":
gender_totals = np.sum(gender_counts, axis=1)
gender_pcts = gender_counts.copy()
for col in gender_counts.columns:
    gender_pcts[col] = round(gender_counts[col]/gender_totals, 4)

# Reorganize each dataframe:
gender_pcts = gender_pcts[["Full funding", "Partial funding", "Not funded", "Overfunded"]].sort_values(by="Full funding")
gender_counts = gender_counts[["Full funding", "Partial funding", "Not funded", "Overfunded"]].loc[gender_pcts.index]


# In[24]:


gender_names = gender_pcts.index.tolist()
column_names = gender_pcts.columns.tolist()
colors = ["green", "orange", "red", "gray"]
gender_data = list()
for i in range(len(column_names)):
    trace = go.Bar(
        y=gender_names,
        x=gender_pcts[column_names[i]].tolist(),
        text=gender_counts[column_names[i]].tolist(),
        name=column_names[i],
        showlegend=False,
        orientation='h',
        marker=dict(color=colors[i])
    )
    gender_data.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Percent funded (by Gender)"
)


# In[25]:


fig = tools.make_subplots(rows=2, cols=1,
                          subplot_titles=["Percent funded (by Gender)", ""],
                          shared_xaxes=True,)

for x in overall_data:
    fig.append_trace(x, 1, 1)
for x in gender_data:
    fig.append_trace(x, 2, 1)

fig["layout"].update(barmode='stack')
fig["layout"]["yaxis2"].update(dict(domain=[0, 0.60]))
fig["layout"]["yaxis1"].update(dict(domain=[0.7, 1]))
fig['layout'].update(height=500, width=1000)


py.iplot(fig, filename='customizing-subplot-axes')


# It looks like Female only and Mixed gender partner groups are the most likely to be funded, at ~95% and 92%, respectively. Meanwhile, Male only groups and groups which didn't specify gender are approved 84% and 83% of the time, respectively.

# # 5) Country/Regional analysis
# Now we'll take a look at the **mpi_region_locations** dataframe alongside the **loans** dataframe, ultimately merging the two to make the loans dataframe more valuable as the source of truth. First we want the ISO codes from the MPL region locations dataframe, they'll help us plot data about each country on a map.
# 
# We'll begin to answer some of these questions:
# 
# * Which countries are loans made in?
# * Which broader world regions are loans made in?
# * What are the most popular sectors for loans in each of these world regions?

# ## Data Prep
# We'll need to clean up the **mpi_region_locations** dataframe. First of all, there are a bunch of rows which are completely empty. In addition, there are some countries present in the **loans** dataframe which are not present in the mpi_region_locations dataframe -- we'll need to manually fill in the values for these countries.

# In[26]:


# Evaluate which countries are present in the loans DF
country_df = pd.DataFrame(data=loans["country"].value_counts()).reset_index()
country_df.columns = ["country", "count"]
# country_df.head()


# In[27]:


# Evaluate which countries are available in the mpi_region_locations dataframe.

# There are a lot of rows which are filled with null values, so we'll drop any of those rows
mpi_region_locations = mpi_region_locations.drop(mpi_region_locations[mpi_region_locations.ISO.isnull()].index)


# In[28]:


# Find the gap between countries present in the `loans` dataframe but missing from the MPL region locations dataframe.
missing = list()
for c in country_df.country.unique().tolist():
    if c not in mpi_region_locations.country.unique().tolist():
        missing.append(c)


# In[29]:


# Add missing countries the the mpi_region_locations dataframe.
# This was done manually

missing = sorted(missing)
country_codes = ["BOL","CHL","COG","CRI","CIV","GEO","GUM","ISR","KSV","LBN","MDA","MMR","PSE","PAN","PRY","PRI",
                 "VCT","WSM","SLB","TZA","COD","TUR","USA","VNM","VGB"]
world_regions = ["Latin America and Caribbean","Latin America and Caribbean","Sub-Saharan Africa",
                 "Latin America and Caribbean","Sub-Saharan Africa","Europe and Central Asia",
                 "East Asia and the Pacific","Arab States","Europe and Central Asia","Arab States",
                 "Europe and Central Asia","South Asia","Arab States","Latin America and Caribbean",
                 "Latin America and Caribbean","Latin America and Caribbean","Latin America and Caribbean",
                 "East Asia and the Pacific","East Asia and the Pacific","Sub-Saharan Africa","Sub-Saharan Africa",
                 "Europe and Central Asia","North America","East Asia and the Pacific","Latin America and Caribbean"]

# Create a list of missing countries and their properties (fill_in_countries) to append to the existing mpl dataframe
fill_in_countries=[]
for i in range(len(country_codes)):
    countries = dict(
                    LocationName="N/A",
                    ISO=country_codes[i],
                    country=missing[i],
                    region="N/A",
                    world_region=world_regions[i],
                    MPI=0.0,
                    geo=(1000,1000),
                    lat=0,
                    lon=0,
                    )
    fill_in_countries.append(countries)


# In[30]:


# Join the two dataframes so that all of the data is available in the `loans` dataframe.
mpi_region_locations = pd.concat([mpi_region_locations, pd.DataFrame(fill_in_countries)])

# Create a pivot table to keep only unique combinations of ISO, country, and world_region in the mpl dataframe
mpl_countries = mpi_region_locations.pivot_table(index=["ISO","country","world_region"]).reset_index()[["ISO","country","world_region"]]

# Merge the loans and new (complete) mpl_countries dataframes onto the loans dataframe.
# New columns in the loans dataframe will be: ISO, world_region.
loans = pd.merge(loans, mpl_countries, on="country", copy=False)


# In[31]:


loans.head()


# In[32]:


print("Columns in the dataset (Updated):\n", sorted(loans.columns.tolist()))


# ## Which countries are loans made in?

# In[33]:


country_counts = loans.country.value_counts().sort_values(ascending=False).head(25)[::-1]

country_freq = loans.pivot_table(index=["country","ISO"], values="id", aggfunc=lambda x: len(x.unique())).reset_index()

tracebar = go.Bar(
    y=country_counts.index,
    x=country_counts.values,
    orientation = 'h',
    marker={
        "color":country_counts.values,
        "colorscale":"Viridis",
        "reversescale":True
    },
)
tracemap = dict(type="choropleth",
             locations=country_freq.ISO.tolist(),
             z=country_freq["id"].tolist(),
             text=country_freq.country.tolist(),
             colorscale="Viridis",
             reversescale=True,
             showscale=False
            )

data = [tracemap, tracebar]
layout = {
    "title": "Loans by Country",
    "height": 1000,
    "width": 1000,
      "geo": {
      "domain": {
          "x": [0, 1], 
          "y": [0.52, 1]
      }
    }
    ,
    "yaxis1": {
        "domain": [0, 0.5]
    },
    "xaxis1": {
        "domain": [0.1, 1]
    }
}
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='mapbar')


# The country with the most loans issued is the Philippines, followed by Kenya, El Salvador, Cambodia, and Pakistan. This isn't illustrated incredibly well on the map because the range of number of loans is so large (only a few loans at minimum, and 160,000 at most), but the map does a good job of showing where in the world Kiva has been able to access with loans. 
# 
# Most of the world is covered, including southeastern Asia, southern and eastern Africa, and South America. Meanwhile, northern Africa, all of Europe, and Russia are (among others) absent.

# ## Which broader world regions are loans made in?

# In[34]:


world_regions = pd.DataFrame(loans.world_region.value_counts()).reset_index()
world_regions.columns = ["world_region", "count"]


# In[35]:


trace = go.Bar(x=world_regions["world_region"].tolist(),
              y=world_regions['count'].tolist())

names = world_regions["world_region"].tolist()
vals = world_regions["count"].tolist()

region_data = list()
for i in range(len(names)):
    trace = go.Bar(
        y=["World Regions"],
        x=[vals[i]],
        name=names[i],
        orientation='h',
    )
    region_data.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Number of Loans by World Region",
    xaxis=dict(title="Number of Loans"),
    height=360, width=1000
)

fig = go.Figure(data=region_data, layout=layout)
py.iplot(fig)


# 80% of loans are made to the top three regions _(East Asia & the Pacific, Sub-Saharan Africa, and Latin America & the Caribbean)_. There were about 200,000 loans made to each of these regions.

# ## What are the most popular sectors for loans in each of these world regions?

# In[36]:


world_region_sector_cts = loans.pivot_table(index="world_region", columns="sector", values="id", aggfunc=lambda x: len(x.unique()))

# Create pct dataframe
world_region_totals = np.sum(world_region_sector_cts, axis=1)
world_region_sector_pcts = world_region_sector_cts.copy()
for col in world_region_sector_cts.columns:
    world_region_sector_pcts[col] = round(world_region_sector_cts[col]/world_region_totals, 4)

world_region_sector_pcts = world_region_sector_pcts[["Food", "Agriculture", "Retail", "Services", "Education", 
                                                     "Clothing", "Housing", "Arts", "Transportation", "Health",
                                                     "Entertainment", "Personal Use", "Construction", "Manufacturing",
                                                    "Wholesale"]].sort_values(by="Food", ascending=True)
world_region_sector_cts = world_region_sector_cts[["Food", "Agriculture", "Retail", "Services", "Education", 
                                                     "Clothing", "Housing", "Arts", "Transportation", "Health",
                                                     "Entertainment", "Personal Use", "Construction", "Manufacturing",
                                                    "Wholesale"]].loc[world_region_sector_pcts.index]


# In[37]:


sector_col_names = world_region_sector_pcts.columns
world_region_names = world_region_sector_pcts.index.tolist()


trace_list = list()
for i in range(len(sector_col_names)):
    trace = go.Bar(
        y=world_region_names,
        x=world_region_sector_pcts[sector_col_names[i]].tolist(),
        text=world_region_sector_cts[sector_col_names[i]].tolist(),
        name=sector_col_names[i],
#         showlegend=False,
        orientation='h',
    )
    trace_list.append(trace)

layout = go.Layout(
    barmode='stack',
    title="Sectors by World Region",
    yaxis=dict(tickangle=-55),
)
fig = go.Figure(data=trace_list, layout=layout)
py.iplot(fig)


# It looks like the sectors the loans will be used to fund vary quite a bit depending on the region of the loan.
# 
# Food, Agriculture, and Retail make up ~60-70% of loans in the three regions which receive 80% loans _(East Asia & the Pacific, Sub-Saharan Africa, and Latin America & the Caribbean.)_ All of these regions are displayed at the top of this chart.
# 
# The Arab States raise far more loans for the purpose of Education than any other region (Europe & Central Asia is secont with just under 10% Education).
# 
# The Arab States and East Asia & the Pacific are the only regions where Personal Use is a large loan sector (about 10% in each region).
# 
# _________________________________
# 
# ## Thanks for reading, please comment with feedback!
# 
