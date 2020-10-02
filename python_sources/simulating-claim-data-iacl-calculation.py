#!/usr/bin/env python
# coding: utf-8

# >**SIMULATING CLAIMS DATA & INFLATION ADJUSTED CHAIN LADDER (IACL) Calculations** 
# 
# **SUMMARY**
# 
# The objective of this Kernel is to accentuate the application of Python for inflation adjusted chain ladder (IACL) calculations. Specifically, I have focued on just the claim triangles for 'Claim Amounts'. Do comment if you'd like to see 'Claim Numbers' or "Average cost per claim" as well. This kernel ends until we arrive at a reserve calculation. I have outlined the coding by chapters below.
# 
# I have also only used the Pandas module here as well. Note that the IACL branches from the basic chain ladder. The only difference is the past and future inflation accountability calculations. 
# 
# **Some pointers to note**
# 
# -I have included both the IACL (denoted with "Inflated" Headers) and basic chain ladder (denoted with "Non Inflated" Headers) codings required side by side for comparative purposes. 
# 
# -This is mainly self-taught, so do help is you feel any improvements I could use!
# 
# -Chapters 1 & 2 ==> improvising random probabilistic claims data
# 
# -Chapters 3, 5, 7, 9 ==> Plotted Charts and Table previews of our data
# 
# -Chapters 4, 6, 8 & 10 ==> Data manipulations and calculations

# **Chapters**
# 
# * 1	Introduction
# 
#         1.1	Import main modules
#         1.2	Establish random data range (Policy Counts, Years, Dates)
#         1.3	Other Assumptions
#     
# * 2	Create Ranomized Claims Data	
# 
#         2.1	Insured IDs
#         2.2	Insured Date
#         2.3	Claim Numbers
#         2.4	Claim Amounts   
#         2.5	Transaction Date
#     
# * 3	Table Preview Raw Data (Part A)
# 
#         3.1	Ranomized Claims Data
#     
# * 4	Quick DataCleaning
# 
#         4.1	Extract Years or Quarters (Depending on your choice of lag period)
#     
# 5	Calculations (Part A)
# 
#         5.1	Year Lags
#         5.2	Compile Past Claims Data (Incremental & Cumulative Amounts)
#         5.3	Establish Inflation Indexes
#         5.4	Uplift (Past Inflation) Incremental Amounts 
#         5.5	Derive corresponding uplifted Cumulative Amounts
#         5.6	Individual Loss Development Factors
#     
# * 6	Plot & Table Preview Triangles  (Part B)	
# 
#         6.0	Define General Plot Functions
#         6.1	Incremental Amount
#         6.2	Cumulative Amounts
#         6.3	Individual Loss Development Factors
#     
# * 7	Calculations (Part B)
# 
#         7.1	Establish Predicted_df
#         7.2	Coordinates (InsuredYear by LagYr) of predicted cells
#         7.3	Impute latest Cumulative Amounts available (as a base point for multiplying by LDF)
#         7.4	SimpleMeanLoss & Volume Weighted & Last 5 and 3 year & Selected LDF
#         7.5	Predicted Cumulative Amounts = Uplift previous Cumulative Amounts by LDF
#         7.6	Data-type adjustments (int & float)
#         7.7	Predicted Incremental Amount
#         7.8	Project (Future Inflation) Predicted Incremental Amount
#     
# * 8	Plot & Table Preview Predictions  (Part C)
# 
#         8.1	Incremental Amount
#         8.2	Cumulative Amounts
# 
# * 9	Plot Full Cumulative Triangle (Part D)
# 
#         9.0	Define General Plot Functions
#         9.1	Non Inflated Claims
#         9.2	Inflated Claims
# 
# * 10 Reserves
# 
#         10.1 Inflated Amounts
#         10.2 Non Inflated Amounts

# * **1. Introduction**

# **1.1** Import main modules
# 
# We will first import the main modules

# In[67]:


import pandas as pd
import numpy as np
import datetime
pd.options.display.max_columns = 100


# **1.2**	Establish random data range (Policy Counts, Years, Dates)
# 
# Now to set the breadth of data we are going to work with

# In[68]:


"""Define parameters of data"""
# Number of entries
PolicyCount = 15000
# 9-year period
YearEndCap = 2017
YearStartCap = 2007
# Dates
DateEndCap = datetime.date(YearEndCap, 12, 31)    # year, month, day
DateStartCap = datetime.date(YearStartCap, 12, 31)  # year, month, day


# **1.3** Other Assumptions

# -No admin costs, claims paid in bulk on transaction date rather than multiple dates for those with multiple claims, ignoring earned gross premium calculations etc

# * **2. Create Randomized Claims Data**
# 
# Now we will use randomisation with probabilistic distributions to simulate some claims data. First we establish the initial data-frame.

# In[69]:


"""Create Main DataFrame filled with NaN's"""
# Establish initial data-frame
columns_1 = ['Insured_ID', 'Insured_Date', 'Claims_Number', 'Claims_Amount', 'Transaction_Date',
           'Insured_Year', 'Insured_Quarter',
           'Transaction_Year', 'Transaction_Quarter']
ClaimsData = pd.DataFrame(columns=columns_1)


# **2.1** Insured IDs
# 
# This is just for labelling purposes to simulate some reality to this. Do note that I have hidden some outputs to neaten up this Kernel. *Unhide to view output

# In[70]:


# Insured_ID's
ClaimsData['Insured_ID'] = list(range(1, PolicyCount+1))
print(ClaimsData['Insured_ID'])


# **2.2** Insured Dates
# 
# This is the date where the insured signs for their policy. Here we will use a random selection of dates between our date range. *Unhide to view output

# In[71]:


# Insured_Date's
# Random distribution
import random
for row in range(0, PolicyCount):
    n_days = (DateEndCap-DateStartCap).days
    random_days = random.randint(0, n_days-1)
    Random_Insured_Date = DateStartCap + datetime.timedelta(days=1) + datetime.timedelta(days=random_days)
    ClaimsData.loc[row, 'Insured_Date'] = Random_Insured_Date
print(ClaimsData['Insured_Date'])


# **2.3** Claims Numbers
# 
# We will now simulate the claim numbers using randomised poisson distribution. In real world scenarios, this is the most common distribution for claim numbers. *Unhide to view output

# In[72]:


# Claims_Number's
# Poisson random distribution
# Poisson parameters
Lambda = 10
Size = 1
for row in range(0, PolicyCount):
    ClaimCount = np.random.poisson(1, 1)
    ClaimsData.loc[row, 'Claims_Number'] = ClaimCount

# Remove the square brackets (i.e.a list within a list) by passing into a list & back into df again
ClaimsData['Claims_Number'] = pd.DataFrame(ClaimsData['Claims_Number'].values.tolist())
print(ClaimsData['Claims_Number'])


# **2.4** Claim Amounts
# 
# Similarly, for claim amounts we will also use randomisation but with Log Normal distribution instead. Do note that here we use a nested loop referncing code. Where if the random claim number was 0 we will have 0 claim amount. For cases without 0 random claim numbers, we will respectively generate and sum 'n' number of random Log Normal distributed claim amount. E.g. Claim Number 1 will have 1 random Log Normal distributed amount, Claim Number 2 will have the sum of 2 random Log Normal distributed amount etc.
# 
# I have also included a spare repetitive Min Max function to simulate reinsurance if need. The code is a 'Do While' loop so as to not disrupt the distribution of claim amounts. But have excluded it in the case. 
# 
# *Unhide to view output

# In[ ]:


"""Special Case if need to simulate claims amount minimum & maximum limit. E.g. Reinsurance cases XOL"""
import random
def trunc_amt(mu, sigma, bottom, top):
    a = random.lognormal(mu,sigma)
    while (bottom <= a <= top) == False:
        a = random.lognormal(mu,sigma)
    return a


# In[73]:


# Claims_Amount's
# Gaussian random distribution
# Gaussian parameters
MeanClaimAmt = 10
StdDevClaimAmt = 4
for row in range(0, PolicyCount):
    if ClaimsData.loc[row, 'Claims_Number'] == 0:
        # Impute 0 so that ClaimAmount is 0
        ClaimsData.loc[row, 'Claims_Amount'] = 0
    else:
        ClaimNumber = ClaimsData.loc[row, 'Claims_Number']
        num = np.random.lognormal(MeanClaimAmt, StdDevClaimAmt, ClaimNumber).sum()
        ClaimsData.loc[row, 'Claims_Amount'] = num

# Remove the square brackets (i.e.a list within a list) by passing into a list & back into df again
ClaimsData['Claims_Amount'] = pd.DataFrame(ClaimsData['Claims_Amount'].values.tolist())
print(ClaimsData['Claims_Amount'])


# **2.5** Transaction Dates
# 
# The transaction dates are the dates that the insurer paid to the insured for a claim made. Just as in claim amounts, we will do a nested loop referencing. Where if the claim number was 0 we will input Transaction Date as the Insured Date, to achieve a 0 lag year. While for cases without 0 as the claim numbers, we will generate a random date between the 'Insured Date' and the YearEndCap of out data range. *Unhide to view output

# In[74]:


# Transaction_Date's
# Random distribution
import random
for row in range(0, PolicyCount):
    DateStart = ClaimsData.loc[row, 'Insured_Date']
    if ClaimsData.loc[row, 'Claims_Number'] == 0:
        # Impute InsuredDate so that Lag(i.e.DevelopmentPeriod) will be 0
        ClaimsData.loc[row, 'Transaction_Date'] = DateStart
    elif (DateEndCap-DateStart).days <=0:
        ClaimsData.loc[row, 'Transaction_Date'] = DateStart
    else:
        n_days = (DateEndCap-DateStart).days
        random_days = random.randint(1, n_days) # Min 1 day to avoid conflict of zero days and no claims
        Random_Transaction_Date = DateStart + datetime.timedelta(days=random_days)
        ClaimsData.loc[row, 'Transaction_Date'] = Random_Transaction_Date
print(ClaimsData['Transaction_Date'])


# * **3. Preview Raw Data (Part A)**

# **3.1** Randomized Claims Data
# 
# Now to preview our random data that we have improvised so far.

# In[76]:


display(ClaimsData.head(10))


# * **4. Preview Raw Data (Part A)**

# **4.1** Extract Years or Quarters (Depending on your choice of lag period)
# 
# Now to prepare our calculation for the different development periods. I have chosen years as the development period here. You may use months alternatively. *Unhide to view output
# 
# *Do note that I will use lag year synonymously with development year here.

# In[78]:


# Extract & Impute Date Components
# Jan-Mar=1, Apr-Jun=2, July-Sep=3, Oct-Dec=4
# Insured Year
ClaimsData['Insured_Year'] = ClaimsData['Insured_Date'].apply(lambda x: x.year)
ClaimsData['Transaction_Year'] = ClaimsData['Transaction_Date'].apply(lambda x: x.year)
# Insured Month
ClaimsData['Insured_Quarter'] = ClaimsData['Insured_Date'].apply(lambda x: x.month)
ClaimsData['Transaction_Quarter'] = ClaimsData['Transaction_Date'].apply(lambda x: x.month)
print(ClaimsData[['Insured_Date', 'Insured_Year', 'Transaction_Date', 'Transaction_Year']])


# * **5. Calculations (Part A)**

# **5.1**	Year Lags
# 
# This is pretty straightforward where we simply take the difference between the "Transaction year" and the "Insured year" columns. In simple terms, the number of years between the insured seured the policy and when a claim was received. *Unhide to view output

# In[79]:


# Year ONLY lag
ClaimsData['Year_Only_Lag'] = ClaimsData['Transaction_Year'] - ClaimsData['Insured_Year']
print(ClaimsData)


# **5.2**	Compile Past Claims Data (Incremental  & Cumulative Amounts)
# 
# Now to simply compile the claims data in a sorted format. Specifically, in a default ascending insured year and lag year order. This does help greatly in terms of indexing later on. *Unhide to view output
# 
# Code Explanation-
# 
# Incremental - We are using the "Claim Amounts" as the output column. We then set filtering rows ("Insured year") and columns ("Lag year"), as we want to see the aggregate for each combination of "Insured year" and "Lag year". The "sum" function acts as a the aggregate function. Subsequently, the "reset_index" function is used to force the data-frame to use the default numerical indexing for the leftmost column. We then assign this to a new data-frame "py_data"
# 
# Cumulative - Just as before, we group the data-frame by the "Insured year" and output the corresponding "Claim Amounts". However, now we instead use the new data-frame "py_data" and we do not require the added columns "Lag year". In addition, we also now use the "cumsum" as a the aggregate function to derive the cumulative claim amounts for each.

# In[80]:


# Compile Past Claims Data
# Incremental Claims Amount
py_data = ClaimsData['Claims_Amount'].groupby([ClaimsData['Insured_Year'], ClaimsData['Year_Only_Lag']]).sum().reset_index()
# Convert into data-frame
py_data = pd.DataFrame(py_data)
# Cumulative Claims Amount
py_data["cumsum"] = py_data["Claims_Amount"].groupby(py_data["Insured_Year"]).cumsum()
print(py_data)


# **Do note from this point onwards, we are dealing with Inflated Adjusted Chain Ladder (IACL) calculations. As mentioned before, I have separated the codings which are required for the IACL (with "Inflated" Headers) from those required for a basic chain ladder (with "Non Inflated" Headers)**

# **5.3**	Establish Inflation Indexes
# 
# For further real world simulation, we will now use the approximated past UK Inflation rates.

# In[81]:


# Establish Inflation Index
# Create data-frame of Cumulative inflation rates
columns_2 = ['Year', 'CumPastInflation']
Inflation_df = pd.DataFrame(columns=columns_2)
# Past Inflation Years
Inflation_df['Year'] = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
# Past Inflation Index
Inflation_df['CumPastInflation'] = [1.32, 1.27, 1.28, 1.22, 1.16, 1.12, 1.09, 1.07, 1.05, 1.04, 1.00]
display(Inflation_df)


# **5.4**	Uplift (Past Inflation) Incremental Amounts
# 
# Here we will account for past inflation for the incremental amounts, NOT the cumulative amounts.  *Unhide to view output
# 
# Code Explanation-
# 
# Here for each incremental claim amount, we continually iterate through the "inflation_df" to derive the corresponding inflation year and indexes and subsequently uplift the amount.
# 
# Just as before we will first set the inflated incremental amounts equal to the non-inflated for easy referencing.
# 
# The code executes 2 nested loops. The first loop is for each incremental claim amount in the "py_data" data-frame, while the second loop is for the each inflation year and inflation index in the "inflation_df" data-frame.
# 
# In the first loop, for each incremental claim amount in the "py_data" data-frame, we establish the "Insured Year", "Lag Year" and "Transaction year" (or "Insured Year" plus "Lag Year").
# 
# With this ongoing first loop, we then have a second loop to iterate through the "inflation_df" data-frame and establish the "inflation year". While iterating the "inflation_df", we set a conditional that upon reaching a equilibrium point where the respective claim amounts year of valuation (or "Transaction Year") is equal to the inflation year we will execute the proceeding uplift calculation below.  
# 
# We will now determine the corresponding "Transaction year" and year-end-cap inflation cumulative index. Finally, divide the latter by the former and multiply it to uplift the incremental claims amount. Note that we divide here as this is a cumulative index. 
# 
# Consequently, now we have nominal incremental claims amount with valuation as at the year-end-cap.
# 
# For cases where we do not reach the equilibrium point, we will simply do nothing. Hence, the value remains.

# In[82]:


# Uplift (Past Inflation) for Incremental Claims
py_data['Inflated_Claims_Amount'] = py_data['Claims_Amount']

for row in range(0, len(py_data['Insured_Year'])):
    InsuredYear = py_data.loc[row,'Insured_Year']
    LagYear = py_data.loc[row,'Year_Only_Lag']
    TransactionYear = InsuredYear + LagYear
    for year in range(0, len(Inflation_df['Year'])):
        CurrentYearInflation = Inflation_df.loc[year,'Year']
        if  CurrentYearInflation == InsuredYear:
            CurrentYearPerc = Inflation_df.loc[Inflation_df['Year'] == TransactionYear,'CumPastInflation']
            ToYearPerc = Inflation_df.loc[Inflation_df['Year']==YearEndCap,'CumPastInflation'].values[0]
            Uplift = ToYearPerc / CurrentYearPerc
            py_data['Inflated_Claims_Amount'][row] = py_data['Inflated_Claims_Amount'][row]*Uplift
        else:
             py_data['Inflated_Claims_Amount'][row] = py_data['Inflated_Claims_Amount'][row]

print(py_data)


# **5.5**	Derive corresponding uplifted Cumulative Amounts
# 
# Now we simply use the 'cumsum' function to derive the corresponding inflated cumulative amounts.

# In[84]:


# Get Uplift (Past Inflation) Cumulative Claims
py_data['Inflated_cumsum'] = py_data['Inflated_Claims_Amount'].groupby(py_data['Insured_Year']).cumsum()


# **5.6**	Individual Loss Development Factors
# 
# Here we will now calculate each development factor (the multiple that resulted in the subsequent years cumulative claim amount) for each insured year. Reason being the IACL underlying assumption is that historical claim trends will follow suit. *Unhide to view output
# 
# Code Explanation-
# 
# For each row in the "py_data" data-frame, we will retrieve the respective "Insured year", "Lag year", "Transaction year" and "Current cumulative claims amount".
# 
# Subsequently, impose a dual 'either or' condition where if the "Transaction year" exceeds the year-end-cap or does not have a proceeding cumulative amount in the next "Lag year" we will have a zero LDF. Reason being this falls into the predicted year range which exceeds our past data range.
# 
# Correspondingly, upon not meeting this condition we will derive the "Next cumulative claims amount". We do this by simply looking up the same "Insured year" but adding 1 to the current "Lag year" ("Lag year" plus one).
# 
# Finally, we divide the "Next cumulative claims amount" by the "Current cumulative claims amount" to derive the individual LDF and impute it.

# In[85]:


# Inflated
py_data['Inflated_LossDF'] = 1

for row in range(0, len(py_data['Insured_Year'])):
    InsuredYear = py_data.loc[row, 'Insured_Year']
    LagYr = py_data.loc[row, 'Year_Only_Lag']
    CurrentYear = py_data.loc[row, 'Insured_Year'] + py_data.loc[row, 'Year_Only_Lag']
    CurrCumAmt = py_data.loc[row, 'Inflated_cumsum']

    if CurrentYear > YearEndCap or len(py_data.loc[(py_data['Insured_Year'] == InsuredYear) & (
            py_data['Year_Only_Lag'] == (LagYr + 1)), 'Inflated_cumsum']) == 0:
        NextCumAmt = 0
    else:
        NextCumAmt = py_data.loc[(py_data['Insured_Year'] == InsuredYear) & (
                    py_data['Year_Only_Lag'] == (LagYr + 1)), 'Inflated_cumsum'].values[0]

    LDF = NextCumAmt / CurrCumAmt
    py_data.loc[row, 'Inflated_LossDF'] = LDF

print(py_data['Inflated_LossDF'])


# In[86]:


# Non Inflated
py_data['LossDF'] = 1

for row in range(0, len(py_data['Insured_Year'])):
    InsuredYear = py_data.loc[row, 'Insured_Year']
    LagYr = py_data.loc[row, 'Year_Only_Lag']
    CurrentYear = py_data.loc[row, 'Insured_Year'] + py_data.loc[row, 'Year_Only_Lag']
    CurrCumAmt = py_data.loc[row, 'cumsum']

    if CurrentYear > YearEndCap or len(py_data.loc[(py_data['Insured_Year'] == InsuredYear) & (
            py_data['Year_Only_Lag'] == (LagYr + 1)), 'cumsum']) == 0:
        NextCumAmt = 0
    else:
        # .values[0] code to output only values and not entire row
        NextCumAmt = py_data.loc[
            (py_data['Insured_Year'] == InsuredYear) & (py_data['Year_Only_Lag'] == (LagYr + 1)), 'cumsum'].values[0]

    LDF = NextCumAmt / CurrCumAmt
    py_data.loc[row, 'LossDF'] = LDF

print(py_data['LossDF'])


# * **6. Preview Triangles (Part B)**
# 
# Until this point we have only worked with past data. Now lets get a visual of what we have done so far.

# **6.0**	Define General Plot Functions

# In[87]:


"""Claims Data - Single Plot"""
def SinglePlotPartialClaims(DataFrameName, InsuredYearColumn, LagYearColumn, ValueColumn):
    import matplotlib.pyplot as plt
    """Create New df"""
    Filtered_NewColumnNames = ["Insured_Year","Year_Only_Lag","ClaimAmt"]
    Filtered_df = pd.DataFrame(DataFrameName[[InsuredYearColumn, LagYearColumn, ValueColumn]])
    Filtered_df.columns = Filtered_NewColumnNames
    """Unique Insured Years List"""
    InsuredYr_List = list(DataFrameName[InsuredYearColumn].unique())
    """Unique Lag Years List"""
    LagYr_List = list(DataFrameName[LagYearColumn].unique())
    """Color List"""
    ALL_Colors = ['r','b','g','y','k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']         
    Color_List = ALL_Colors[:len(InsuredYr_List)]
    """LineStyle List"""
    ALL_LineStyle = ['-', '--', '-.', ':','-','-','-','-','-','-','-','-','-']
    LineStyle_List = ALL_LineStyle[:len(InsuredYr_List)]
    """MarkerStyle List"""# First 4x empty 
    ALL_Markers = ['','','','','^','.','o','*', '+', '1', '2', '3', '4']
    Marker_List = ALL_Markers[:len(InsuredYr_List)]
    """Loop Plot"""
    for row_A in range(0,len(InsuredYr_List)):
        plt.figure(2, figsize=(10,5))
        Year_i = InsuredYr_List[row_A]
        SubFiltered_df = Filtered_df.loc[Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(SubFiltered_df['Year_Only_Lag'], SubFiltered_df['ClaimAmt'], 
                 label=str(Year_i), linestyle='-', color=Color_List[row_A])
    """Plot Attributes"""    
    plt.xlabel('Developement Year')
    plt.ylabel('Claims Value')
    plt.title('Single Plot Partial Claims Data')
    plt.legend()
    plt.show()


# In[88]:


"""Claims Data - Sub Plot"""
def SubPlotPartialClaims(DataFrameName, InsuredYearColumn, LagYearColumn, ValueColumn):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    """Create New df"""
    Filtered_NewColumnNames = ["Insured_Year","Year_Only_Lag","ClaimAmt"]
    Filtered_df = pd.DataFrame(DataFrameName[[InsuredYearColumn, LagYearColumn, ValueColumn]])
    Filtered_df.columns = Filtered_NewColumnNames
    """Unique Insured Years List"""
    InsuredYr_List = list(DataFrameName[InsuredYearColumn].unique())
    """Unique Lag Years List"""
    LagYr_List = list(DataFrameName[LagYearColumn].unique())
    """Color List"""
    ALL_Colors = ['r','b','g','y','k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']         
    Color_List = ALL_Colors[:len(InsuredYr_List)]
    """LineStyle List"""
    ALL_LineStyle = ['-', '--', '-.', ':','-','-','-','-','-','-','-','-','-']
    LineStyle_List = ALL_LineStyle[:len(InsuredYr_List)]
    """MarkerStyle List"""# First 4x empty 
    ALL_Markers = ['','','','','^','.','o','*', '+', '1', '2', '3', '4']
    Marker_List = ALL_Markers[:len(InsuredYr_List)]
    """Plot Attributes"""
    fig = plt.figure(2, figsize=(10,14))
    plt.xticks([]) # remove initial blank plot default ticks
    plt.yticks([]) # remove initial blank plot default ticks
    plt.title('Sub Plot Partial Claims Data')
    rcParams['axes.titlepad'] = 70 # position title
    plt.box(on=None) # Remove boundary line
    """Loop Plot"""
    i=0
    for row_A in range(0,len(InsuredYr_List)):
        ax = fig.add_subplot(5, 2, 1+i)
        Year_i = InsuredYr_List[row_A]
        SubFiltered_df = Filtered_df.loc[Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(SubFiltered_df['Year_Only_Lag'], SubFiltered_df['ClaimAmt'], 
                 label=str(Year_i), marker='o', linestyle='-', color=Color_List[row_A])
        plt.xticks(np.arange(0, (YearEndCap-YearStartCap), step=1))
        plt.legend()
        i += 1
        """Plot Attributes"""
        plt.xlabel('Developement Year')
        plt.ylabel('Claims Value')
    
    fig.tight_layout() # set size
    plt.show()


# In[89]:


"""Loss Development Ratios"""
def SinglePlotLDF(DataFrameName, Columns):
    import matplotlib.pyplot as  plt
    """Create New df"""
    Filtered_df = pd.DataFrame(DataFrameName[Columns])    
    """Lag Years"""
    LagYears_List = list(range(0, len(DataFrameName)))
    """Color List"""
    ALL_Colors = ['r','b','g','y','k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']         
    Color_List = ALL_Colors[:len(Columns)]
    """Loop Plot"""
    plt.figure(2, figsize=(10,5))
    for row_A in range(0,len(Columns)):
        Column_i = Columns[row_A]
        plt.plot(LagYears_List, Filtered_df[Column_i], label=str(Column_i), linestyle='-', color=Color_List[row_A])
        plt.legend()         
    """Plot Attributes"""    
    plt.xlabel('Developement Year')
    plt.ylabel('Ratio')
    plt.title('Loss Development Factors')
    plt.show()


# **6.1**	Incremental Amount
#         

# # Inflated Incremental

# In[90]:


# Incremental Claims Amount
# Inflated
py_triangle_inflated = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["Inflated_Claims_Amount"])
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
sns.distplot(py_data['Inflated_Claims_Amount'], kde=False, fit=stats.lognorm) # norm, pareto, loggamma, gompertz
plt.show()
display(py_triangle_inflated)


# # Non-Inflated Incremental

# In[91]:


# Incremental Claims Amount
# Non-Inflated
py_triangle = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["Claims_Amount"])
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
sns.distplot(py_data['Claims_Amount'], kde=False, fit=stats.lognorm) # norm, pareto, loggamma, gompertz
plt.show()
display(py_triangle)


# **6.2**	Cumulative Amounts

# # Inflated Cumulative

# In[92]:


# Cumulative Claims Amount
# Inflated
py_triangle_cum_inflated = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["Inflated_cumsum"])
SinglePlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='Inflated_cumsum')
SubPlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='Inflated_cumsum')
display(py_triangle_cum_inflated)


# # Non-Inflated Cumulative

# In[93]:


# Cumulative Claims Amount
# Non-Inflated
py_triangle_cum = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["cumsum"])
SinglePlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='cumsum')
SubPlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='cumsum')
display(py_triangle_cum)


# **6.3**	Individual Loss Development Factors

# # Inflated LDF

# In[95]:


# Individual Loss Development factors
# Inflated
py_InflatedLossDF_triangle = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["Inflated_LossDF"])
SinglePlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='Inflated_LossDF')
SubPlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='Inflated_LossDF')
display(py_InflatedLossDF_triangle)


# # Non-Inflated LDF

# In[96]:


# Individual Loss Development factors
# Non-Inflated
py_LossDF_triangle = pd.pivot_table(py_data, index=["Insured_Year"], columns=["Year_Only_Lag"], values=["LossDF"])
SinglePlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='LossDF')
SubPlotPartialClaims(DataFrameName=py_data, InsuredYearColumn='Insured_Year', LagYearColumn='Year_Only_Lag', ValueColumn='LossDF')
display(py_LossDF_triangle)


# * **7. Calculations (Part B)**

# **7.1**	Establish Predicted_df
# 
# We will first create a temporary dummy data-frame "temp_df" containing all the years and lag years that are within our data range for analysis only for referencing purposes. 
# 
# 

# In[97]:


# Create a Temp Df of Predicted Years & LagYears rates
columns_3 = ['InsuredYear', 'PredictedYear_Only_Lag',
             'Previous_cumsum', 'Predicted_cumsum', 'Predicted_Incremental',
             'Previous_Inflated_cumsum', 'Predicted_Inflated_cumsum', 'Predicted_Inflated_Incremental']
Temp_df = pd.DataFrame(columns=columns_3)
# +1 due to 31 Dec 2017 (also not a Bday) & +1 due to range exlusion of last value cap
InsuredYr = list(range(YearStartCap + 1, YearEndCap + 1, 1))  # [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
Temp_df['InsuredYear'] = InsuredYr
Lags = list(range(0, YearEndCap - YearStartCap, 1))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Temp_df['PredictedYear_Only_Lag'] = Lags

# Establish Predicted data-frame
Predicted_df = pd.DataFrame(columns=columns_3)


# **7.2**	Coordinates (InsuredYear by LagYr) of predicted cells
# 
# We will now move on to create the actual data-frame of the corresponding predicted insured years and lag years named as "Predicted_df". In other words, it is simply the combination of both the "Insured_Years" and "Lag Years" for all the NaNs that we have in the claims triangle seen before. *Unhide to view output
# 
# Code Explanation-
# 
# The code in short executes 2 nested loops to compare each "Transaction Year" (or "Insured_Years" plus "Lag Year") against the year-end-cap. Subsequently, impute the corresponding "Insured_Years" and "Predicted Lag Years". This will then act as the coordinates for the NaNs which we will use later as references for imputing our predictions.
# 
# The first loop iterates through each "InsuredYear" column in the "Temp_df" data-frame to establish the "InsuredYear".
# 
# From here, for each "InsuredYear" we will now start a second loop to iterate through the "Lag Years" and derive the "P_yr" (or also "Transaction Year").
# 
# Now we will set a condition, where if the "P_yr" exceeds the year-end-cap we will impute that "InsuredYear" and "Lag Year" combination. Reason being because it falls beyond our past data range hence it is a predicted year. Intuitively, just the opposite of what was done in calculating the individual LDFs.
# 
# Not meeting this condition, we will do nothing. Do also note the i=0 and i += 1 is for indexing purposes. 0 as python data-frames defaults via a 0 index at outset and +1 to move to the next row after imputing.

# In[98]:


# Coordinates of predicted Insured Years & Lag Years
x = 1 # Do nothing
i = 0 # For loop impute indexing
for row in range(0, len(Temp_df['InsuredYear'])):
    BaseYr = Temp_df.loc[row, 'InsuredYear']
    for lag in range(0, len(Temp_df['PredictedYear_Only_Lag'])):
        LagYr = Temp_df.loc[lag, 'PredictedYear_Only_Lag']
        P_yr = BaseYr + Temp_df.loc[lag, 'PredictedYear_Only_Lag']
        if P_yr > YearEndCap:

            Predicted_df.loc[i, 'InsuredYear'] = BaseYr
            Predicted_df.loc[i, 'PredictedYear_Only_Lag'] = LagYr
            i += 1
        else:
            x = x

print(Predicted_df[['InsuredYear', 'PredictedYear_Only_Lag']])


# **7.3**	Impute latest Cumulative Amounts available (as a base point for multiplying by LDF)
# 
# The reason for this is because we need a base point for multiplying by our LDFs and predicting future cumulative claim amounts later. Rather than referencing separatly, we will simply do a look-up and impute accordingly. It neatens the process. *Unhide to view output
# 
# Code Explanation-
# 
# In short, the code loops through the "Insured year" and "Lag year" columns in "Predicted_df" that we derived earlier and uses these two references as look-up references against the "py_data" (containing past data) to find and impute the corresponding latest cumulative amounts available for that respective insured year.
# 
# The loop and first 3 code lines establishes the "Insured year", "Lag year" and "PredYr" ("Insured year" + "Lag year") for each respective loop iteration while in the "Predicted_df" DataFrame.
# 
# We then set 3 levels of 'If' conditionals to determine the latest cumulative sum.
# 
# In chronological order-
# 
# First condition; if the "Insured year" is equivalent to the year-end-cap, there is only one previous cumulative sum (which is the value at the lowest bottom point of a claim triangle). Hence, we will only output that.
# 
# Second condition; if the predicted year "PredYr" exceeds the year-end-cap or if the look-up reference (via the same "Insured year" and "Lag year" minus one) for the previous cumulative sum renders none, we will keep the same insured year but replace the "Lag year" minus one formulae. Instead take the "Maximum Lag year" (the lag year of the latest cumulative claim amount available in that insured year) for that insured year as reference for the cumulative amount look-up.
# 
# To put it contextually, if you refer to the above PART-5 Raw Preliminary view of Claims Triangle it is the "Year_Only_Lag" numbers just before a NaN. For "Insured Year"-2017 it would be "Year_Only_Lag"-0, for "Insured Year"-2016 it would be "Year_Only_Lag"-1.... etc
# 
# Third condition; the residual of which does not fulfil the above two conditions uses this. Where we will simply execute the cumulative sum look-up references via the same "Insured year" and "Lag year" minus one as a reference.
# 
# Finally, we will impute the latest cumulative sum in the column "Previous_Inflated_cumsum" of that respective row iteration in the loop within the "Predicted_df" data-frame.
# 

# In[99]:


# Impute latest cumulative amounts available
# Inflated
for row in range(0, len(Predicted_df)):
    Base = Predicted_df.loc[row, 'InsuredYear']
    Lag = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    PredYr = Base + Lag

    if Base == YearEndCap:
        PrevInflatedCumSum = py_data.loc[(py_data['Insured_Year'] == Base), 'Inflated_cumsum'].values[0]

    else:
        if PredYr > YearEndCap or len(py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == Lag - 1), 'Inflated_cumsum']) == 0:
            MaxLag = py_data.loc[(py_data['Insured_Year'] == Base), 'Year_Only_Lag'].max()
            PrevInflatedCumSum = py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == MaxLag), 'Inflated_cumsum'].values[0]

        else:
            PrevInflatedCumSum = py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == Lag - 1), 'Inflated_cumsum'].values[0]

    Predicted_df.loc[row, 'Previous_Inflated_cumsum'] = PrevInflatedCumSum

print(Predicted_df['Previous_Inflated_cumsum'])


# In[100]:


# Impute latest cumulative amounts available
# Non-Inflated
for row in range(0, len(Predicted_df)):
    Base = Predicted_df.loc[row, 'InsuredYear']
    Lag = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    PredYr = Base + Lag

    if Base == YearEndCap:
        PrevCumSum = py_data.loc[(py_data['Insured_Year'] == Base), 'cumsum'].values[0]

    else:
        if PredYr > YearEndCap or len(
                py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == Lag - 1), 'cumsum']) == 0:
            MaxLag = py_data.loc[(py_data['Insured_Year'] == Base), 'Year_Only_Lag'].max()
            PrevCumSum = py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == MaxLag), 'cumsum'].values[0]

        else:
            PrevCumSum = py_data.loc[(py_data['Insured_Year'] == Base) & (py_data['Year_Only_Lag'] == Lag - 1), 'cumsum'].values[0]

    Predicted_df.loc[row, 'Previous_cumsum'] = PrevCumSum
    
print(Predicted_df['Previous_cumsum'])


# **7.4**	SimpleMeanLoss & Volume Weighted & Last 5/3 years & Selected LDF
# 
# Now with the individual LDFs calculated earlier in *Chapter 5.6* we will now derive the average-lag year-to-lag-year LDFs. 

# We first establish the initial data-frame columns and corresponding year lags for each average LDFs to reference against when calculating from the individual LDFs.

# In[101]:


# Establish averaged-year-to-year LDF
columns_4 = ['Year_Only_Lag',
             'SimpleMeanLossDF', 'VolWtdLossDF',
             'CumToUlt_SimpleMeanLossDF', 'CumToUlt_VolWtdLossDF',
             'SimpleMeanLossDF_5year', 'VolWtdLossDF_5year',
             'SimpleMeanLossDF_3year', 'VolWtdLossDF_3year',
             'SelectLossDF'
             'Inflated_SimpleMeanLossDF', 'Inflated_VolWtdLossDF',
             'Inflated_CumToUlt_SimpleMeanLossDF', 'Inflated_CumToUlt_VolWtdLossDF',
             'Inflated_SimpleMeanLossDF_5year', 'Inflated_VolWtdLossDF_5year',
             'Inflated_SimpleMeanLossDF_3year', 'Inflated_VolWtdLossDF_3year',
             'Inflated_SelectLossDF']
LossDF_df = pd.DataFrame(columns=columns_4)
Lags = list(range(0, YearEndCap-YearStartCap, 1))
LossDF_df['Year_Only_Lag'] = Lags
display(LossDF_df)


# Code Explanation-
# 
# The code loops though each "Lag Years" ("Year_Only_Lag" column) in the "LossDF_df" data-frame to reference the required lag year. Subsequently, for each "Lag Year" executes the simple mean and volume weight LDF calculations below.
# 
# Simple Mean - Looks up all the LDFs having that required "Lag Year" reference and takes the average. However, excluding the last LDF as that would be the LDF for moving into a year outside our data range. In other words, a predicted year.
# 
# Volume Weighted - Looks up all the cumulative sums having that required lag year and also the subsequent next lag year (required lag year plus 1 year) and takes the sum. However, for that required year it excludes the last cumulative sum to ensure equitable quantity of summing components. Just as in the simple mean calculation.
# 
# Finally, impute and +1 to move to the next row for the proceeding loop to impute accordingly.

# # Inflated All Year Average LDFs

# In[102]:


# Inflated
i=0
for lag in range(0, len(Temp_df['PredictedYear_Only_Lag'])):
    lagyr = Temp_df.loc[lag, 'PredictedYear_Only_Lag']
    # Simple Mean
    # due to 0 input so exlude last value
    SimpleMeanLossDF = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_LossDF'][:-1].mean()
    LossDF_df.loc[i, 'Inflated_SimpleMeanLossDF'] = SimpleMeanLossDF
    # Volume Weighted
    Deno = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'Inflated_cumsum'].sum()
    Neum = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_cumsum'][:-1].sum()
    VolWtdLossDF = Deno / Neum
    LossDF_df.loc[i, 'Inflated_VolWtdLossDF'] = VolWtdLossDF
    i += 1

# [::-1] to flip or invert the row order
LossDF_df['Inflated_CumToUlt_SimpleMeanLossDF']=LossDF_df['Inflated_SimpleMeanLossDF'][::-1].cumprod()
LossDF_df['Inflated_CumToUlt_VolWtdLossDF']=LossDF_df['Inflated_VolWtdLossDF'][::-1].cumprod()

SinglePlotLDF(DataFrameName=LossDF_df, Columns=['Inflated_SimpleMeanLossDF', 'Inflated_VolWtdLossDF'])
SinglePlotLDF(DataFrameName=LossDF_df, Columns=['Inflated_CumToUlt_SimpleMeanLossDF', 'Inflated_CumToUlt_VolWtdLossDF'])
display(LossDF_df[['Inflated_SimpleMeanLossDF', 'Inflated_VolWtdLossDF', 'Inflated_CumToUlt_SimpleMeanLossDF', 'Inflated_CumToUlt_VolWtdLossDF']])


# # Non-Inflated All Year Average LDFs

# In[103]:


# Non-Inflated
i=0
for lag in range(0, len(Temp_df['PredictedYear_Only_Lag'])):
    lagyr = Temp_df.loc[lag, 'PredictedYear_Only_Lag']
    # Simple Mean
    # due to 0 input so exlude last value
    SimpleMeanLossDF = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'LossDF'][:-1].mean()
    LossDF_df.loc[i, 'SimpleMeanLossDF'] = SimpleMeanLossDF
    # Volume Weighted
    Deno = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'cumsum'].sum()
    Neum = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'cumsum'][:-1].sum()
    VolWtdLossDF = Deno / Neum
    LossDF_df.loc[i, 'VolWtdLossDF'] = VolWtdLossDF
    i += 1

# [::-1] to flip or invert the row order
LossDF_df['CumToUlt_SimpleMeanLossDF']=LossDF_df['SimpleMeanLossDF'][::-1].cumprod()
LossDF_df['CumToUlt_VolWtdLossDF']=LossDF_df['VolWtdLossDF'][::-1].cumprod()

SinglePlotLDF(DataFrameName=LossDF_df, Columns=['SimpleMeanLossDF', 'VolWtdLossDF'])
SinglePlotLDF(DataFrameName=LossDF_df, Columns=['CumToUlt_SimpleMeanLossDF', 'CumToUlt_VolWtdLossDF'])
display(LossDF_df[['SimpleMeanLossDF', 'VolWtdLossDF', 'CumToUlt_SimpleMeanLossDF', 'CumToUlt_VolWtdLossDF']])


# Last 5 & 3 year averages
# 
# The 5/3 Year Averages are just as we did earlier. The only difference is that now rather than the '-1' to exclude the final entry, we replace that with the number of years we want. In this case, I declared them as Year_A for 5 year and Year_B for 3 year averages.

# # Inflated 5 & 3 year Average LDFs

# In[104]:


# Inflated
i=0
for lag in range(0, len(Temp_df['PredictedYear_Only_Lag'])):
    lagyr = Temp_df.loc[lag, 'PredictedYear_Only_Lag']
    # Simple Mean
    Year_A = 5   # 5 Year
    SimpleMeanLossDF_Ayear = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_LossDF'][:Year_A].mean()
    LossDF_df.loc[i, 'Inflated_SimpleMeanLossDF_5year'] = SimpleMeanLossDF_Ayear
    Year_B = 3   # 3 Year
    SimpleMeanLossDF_Byear = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_LossDF'][:Year_B].mean()
    LossDF_df.loc[i, 'Inflated_SimpleMeanLossDF_3year'] = SimpleMeanLossDF_Byear
    # Volume Weighted
    Deno_A = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'Inflated_cumsum'][:Year_A].sum()
    Neum_A = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_cumsum'][:Year_A].sum()
    VolWtdLossDF_A = Deno_A / Neum_A
    LossDF_df.loc[i, 'Inflated_VolWtdLossDF_5year'] = VolWtdLossDF_A
    Deno_B = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'Inflated_cumsum'][:Year_B].sum()
    Neum_B = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_cumsum'][:Year_B].sum()
    VolWtdLossDF_B = Deno_B / Neum_B
    LossDF_df.loc[i, 'Inflated_VolWtdLossDF_3year'] = VolWtdLossDF_B
    i += 1

SinglePlotLDF(DataFrameName=LossDF_df, Columns=['Inflated_SimpleMeanLossDF_5year', 'Inflated_VolWtdLossDF_5year'])
SinglePlotLDF(DataFrameName=LossDF_df, Columns=['Inflated_SimpleMeanLossDF_3year', 'Inflated_VolWtdLossDF_3year'])    
display(LossDF_df[['Inflated_SimpleMeanLossDF_5year', 'Inflated_VolWtdLossDF_5year', 'Inflated_SimpleMeanLossDF_3year', 'Inflated_VolWtdLossDF_3year']])


# # Non-Inflated 5 & 3 year Average LDFs

# In[105]:


# Non Inflated
i=0
for lag in range(0, len(Temp_df['PredictedYear_Only_Lag'])):
    lagyr = Temp_df.loc[lag, 'PredictedYear_Only_Lag']
    # Simple Mean
    Year_A = 5   # 5 Year
    SimpleMeanLossDF_Ayear = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_LossDF'][:Year_A].mean()
    LossDF_df.loc[i, 'SimpleMeanLossDF_5year'] = SimpleMeanLossDF_Ayear
    Year_B = 3   # 3 Year
    SimpleMeanLossDF_Byear = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_LossDF'][:Year_B].mean()
    LossDF_df.loc[i, 'SimpleMeanLossDF_3year'] = SimpleMeanLossDF_Byear
    # Volume Weighted
    Deno_A = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'Inflated_cumsum'][:Year_A].sum()
    Neum_A = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_cumsum'][:Year_A].sum()
    VolWtdLossDF_A = Deno_A / Neum_A
    LossDF_df.loc[i, 'VolWtdLossDF_5year'] = VolWtdLossDF_A
    Deno_B = py_data.loc[py_data['Year_Only_Lag'] == (lagyr + 1), 'Inflated_cumsum'][:Year_B].sum()
    Neum_B = py_data.loc[py_data['Year_Only_Lag'] == lagyr, 'Inflated_cumsum'][:Year_B].sum()
    VolWtdLossDF_B = Deno_B / Neum_B
    LossDF_df.loc[i, 'VolWtdLossDF_3year'] = VolWtdLossDF_B
    i += 1

SinglePlotLDF(DataFrameName=LossDF_df, Columns=['SimpleMeanLossDF_5year', 'VolWtdLossDF_5year'])
SinglePlotLDF(DataFrameName=LossDF_df, Columns=['SimpleMeanLossDF_3year', 'VolWtdLossDF_3year'])
display(LossDF_df[['SimpleMeanLossDF_5year', 'VolWtdLossDF_5year', 'SimpleMeanLossDF_3year', 'VolWtdLossDF_3year']])


# Selected LDF
# 
# In real world scenarios, actuaries will analyze the consistency in LDFs calculated above. Some reasons include abnormally large claims may distort the LDF trend, legislative reasons to exclude a specific number of years in the LDF averages etc..
# 
# In this case we will simply use the fully all year averaged LDFs since it is the smoothest amongst the choices.

# In[106]:


LossDF_df['Inflated_SelectLossDF'] = LossDF_df['Inflated_VolWtdLossDF']
LossDF_df['SelectLossDF'] = LossDF_df['VolWtdLossDF']


# **7.5**	Predicted Cumulative Amounts = Uplift previous Cumulative Amounts by LDF
# 
# Now with a selected LDF we will proceed to use historical claim trends to predict future claim trends! Pretty much self explanatory here. We simply apply the respective LDF to each past cumulative claim to derive the future cumulative claim. On the assumption trends will follow suit. *Unhide to view output
# 
# Code Explanation-
# 
# As mentioned before we are predicting using the latest cumulative amount as the baseline. Thus, we will set them equal first for easy reference.
# 
# In short, the code iterates through the "Predicted_df" to determine the year range to apply the LDF for prediction, and correspondingly references the LDF aligning to that year range from the "LossDF_df" to predict the amount.
# 
# The code uses 2 nested loops. The first loop in the "Predicted_df" is used to establish the combination of the "Insured year" and "Lag year" that the predicted year belongs to. It is also used to derive the "Maximum Lag year". Exactly, what we did in PART-7 Impute latest cumulative amounts.
# 
# The second loop in the "LossDF_df" data-frame is used to iterate over the various "Lag years" to reference the averaged-by-lag-year LDFs we calculated above to predict the respective cumulative amounts. Whilst under the second loop, we set 2 conditionals -
# 
# First condition: If this ongoing second loop iteration reaches the last "Lag year" we will do nothing. The reason being is that the final lag year is the ultimate "Lag year", hence no LDF is available.
# 
# Second condition: If this second loop iteration's "Lag year" reaches a equilibrium with the maximum "Lag year" for that "Insured year" from that ongoing first loop iteration we will only then proceed with the predicting calculation. In other words, when the "Lag year" of the "Predicted_df" and the "LossDF_df" are equal.
# 
# The calculation simply takes the product of all the averaged-by-lag-year LDFs falling within inclusively of the maximum lag year and the predicted lag year minus one range (both of which established from that ongoing first loop iteration) and multiplies that into the latest cumulative sum to attain the predicted cumulative sum.
# 
# Not fulfilling either condition we will do nothing.
# 

# In[107]:


# Predict Cumulative Claim Amounts
# Inflated
# Set Equal for easy reference
Predicted_df['Predicted_Inflated_cumsum'] = Predicted_df['Previous_Inflated_cumsum']
lagyearlimit = (YearEndCap - YearStartCap) - 1
x = 1  # Do nothing
for row in range(0, len(Predicted_df)):
    PredLagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    BaseInsuredYr = Predicted_df.loc[row, 'InsuredYear']
    MaxLagYr = py_data.loc[(py_data['Insured_Year'] == BaseInsuredYr), 'Year_Only_Lag'].max()
    for r in range(0, len(LossDF_df)):
        if (LossDF_df.loc[r, 'Year_Only_Lag'] == lagyearlimit):
            x = x  # To avoid NaN
        elif (LossDF_df.loc[r, 'Year_Only_Lag'] == MaxLagYr):
            # LDF multiplication
            LDF = LossDF_df.loc[(LossDF_df['Year_Only_Lag'] >= MaxLagYr) & (LossDF_df['Year_Only_Lag'] <= (PredLagYr - 1)), 'Inflated_SelectLossDF'].prod()
            Predicted_df.loc[row, 'Predicted_Inflated_cumsum'] = Predicted_df.loc[row, 'Predicted_Inflated_cumsum'] * LDF
        else:
            x = x  # Do nothing
            
print(Predicted_df['Predicted_Inflated_cumsum'])


# In[108]:


# Predict Cumulative Claim Amounts
# Non-Inflated
# Set Equal for easy reference
Predicted_df['Predicted_cumsum'] = Predicted_df['Previous_cumsum']
lagyearlimit = (YearEndCap - YearStartCap) - 1
x = 1  # Do nothing
for row in range(0, len(Predicted_df)):
    PredLagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    BaseInsuredYr = Predicted_df.loc[row, 'InsuredYear']
    MaxLagYr = py_data.loc[(py_data['Insured_Year'] == BaseInsuredYr), 'Year_Only_Lag'].max()
    for r in range(0, len(LossDF_df)):
        if (LossDF_df.loc[r, 'Year_Only_Lag'] == lagyearlimit):
            x = x  # To avoid NaN
        elif (LossDF_df.loc[r, 'Year_Only_Lag'] == MaxLagYr):
            # LDF multiplication
            LDF = LossDF_df.loc[(LossDF_df['Year_Only_Lag'] >= MaxLagYr) & (LossDF_df['Year_Only_Lag'] <= (PredLagYr - 1)), 'SelectLossDF'].prod()
            Predicted_df.loc[row, 'Predicted_cumsum'] = Predicted_df.loc[row, 'Predicted_cumsum'] * LDF
        else:
            x = x  # Do nothing

print(Predicted_df['Predicted_cumsum'])


# **7.6**	Data-type adjustments (int & float)
# 
# This is just a intermediate step to ensure consistent computational data-types as so many data manipulations were made before.

# In[109]:


# Data-type adjustments
# Years
Predicted_df[['InsuredYear','PredictedYear_Only_Lag']]=Predicted_df[['InsuredYear','PredictedYear_Only_Lag']].astype(int)
# Amounts
Predicted_df[['Predicted_cumsum','Previous_cumsum']]=Predicted_df[['Predicted_cumsum','Previous_cumsum']].astype(float)
Predicted_df[['Predicted_Inflated_cumsum','Previous_Inflated_cumsum']]=Predicted_df[['Predicted_Inflated_cumsum','Previous_Inflated_cumsum']].astype(float)


# **7.7**	Predicted Incremental Amount
# 
# **Do note that the Insured year column now starts from 2009 not 2008 as in the past claims data "py_data" data-frame**
# 
# With the predicted cumulative amount derive earlier, we will now derive the incremental amount. Reason being we need to use the incremental amount to project future inflation (just as we did for past inflation).
# 
# Code Explanation-
# 
# The code simply loops through both data-frames "Predicted_df" (predicted data) and "py_data" (past data) and looks up the respective current and previous cumulative amount based on "Insured Year" and "Lag Year" references. Subsequently, calculates the difference which is then the incremental amount.
# 
# Just as before we will loop the predicted data-frame and first establish the "Insured year", "Lag year" and "Current predicted cumulative amount" (belonging to the current loop iteration). After which using the "Insured year" and "Lag year" minus one combination as references to look-up the previous cumulative amount.
# 
# The code then sets 2 conditionals -
# 
# First condition: If we are not able to look up the respective previous cumulative values in the predicted data-frame "Predicted_df", we instead search in the past cumulative values data-frame "py_data". This specifically required for those amounts falling on the 'steps' of a claim triangle.
# 
# Second condition: Here we are simply searching the predicted data-frame "Predicted_df".

# In[110]:


# Predict Incremental Amount
# Inflated
for row in range(0, len(Predicted_df)):
    InsurYr = Predicted_df.loc[row, 'InsuredYear']
    LagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    CurrCum = Predicted_df.loc[row, 'Predicted_Inflated_cumsum']
    # For which we can't look up in Predicted_df
    if len(Predicted_df.loc[(Predicted_df['InsuredYear'] == InsurYr) & (Predicted_df['PredictedYear_Only_Lag'] == LagYr - 1), 'Predicted_Inflated_cumsum']) == 0:
        PrevCum = py_data.loc[(py_data['Insured_Year'] == InsurYr) & (py_data['Year_Only_Lag'] == LagYr - 1), 'Inflated_cumsum'].values[0]
    # For which we can look up in Predicted_df
    else:
        PrevCum = Predicted_df.loc[(Predicted_df['InsuredYear'] == InsurYr) & (Predicted_df['PredictedYear_Only_Lag'] == LagYr - 1), 'Predicted_Inflated_cumsum'].values[0]

    Predicted_df.loc[row, 'Predicted_Inflated_Incremental'] = (CurrCum - PrevCum)

Predicted_df[['Predicted_Inflated_Incremental']] = Predicted_df[['Predicted_Inflated_Incremental']].astype(float)
PredictedInflatedIncrementalTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"],columns=["PredictedYear_Only_Lag"],values=["Predicted_Inflated_Incremental"])

# print(PredictedInflatedIncrementalTriangle)
display(PredictedInflatedIncrementalTriangle)


# In[111]:


# Predict Incremental Amount
# Non-Inflated
for row in range(0, len(Predicted_df)):
    InsurYr = Predicted_df.loc[row, 'InsuredYear']
    LagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    CurrCum = Predicted_df.loc[row, 'Predicted_cumsum']

    if len(Predicted_df.loc[(Predicted_df['InsuredYear'] == InsurYr) & (Predicted_df['PredictedYear_Only_Lag'] == LagYr - 1), 'Predicted_cumsum']) == 0:
        PrevCum = py_data.loc[(py_data['Insured_Year'] == InsurYr) & (py_data['Year_Only_Lag'] == LagYr - 1), 'cumsum'].values[0]
    else:
        PrevCum = Predicted_df.loc[(Predicted_df['InsuredYear'] == InsurYr) & (Predicted_df['PredictedYear_Only_Lag'] == LagYr - 1), 'Predicted_cumsum'].values[0]

    Predicted_df.loc[row, 'Predicted_Incremental'] = CurrCum - PrevCum

Predicted_df[['Predicted_Incremental']] = Predicted_df[['Predicted_Incremental']].astype(float)
PredictedIncrementalTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"], columns=["PredictedYear_Only_Lag"],values=["Predicted_Incremental"])

# print(PredictedIncrementalTriangle)
display(PredictedIncrementalTriangle)


# **7.8**	Project (Future Inflation) Predicted Incremental Amount
# 
# Now to project for future inflation. *Unhide to view output
# 
# Code Explanation-
# 
# This is rather straightforward as well. First we determine the future inflation index ("FutureInflation") via look-up using the year-end-cap plus one as reference.
# 
# Now just as before, we simply first equate the future uplifted incremental claims amount derived above to the existing nominal valued as at year-end-cap for easy reference.
# 
# Likewise, we also first loop to establish the "Insured Year", "Lag Year" and "Current incremental amount". We then uplift by taking the "Current incremental amount" multiplied by the "FutureInflation" and the "Lag Year" being the index exponent.

# In[112]:


# Project (Future Inflation) Predicted Incremental Amount
# Inflated
FutureInflation = Inflation_df.loc[(Inflation_df['Year'] == (YearEndCap + 1)), 'CumPastInflation'].values[0]

Predicted_df['FutureUplifted_Predicted_Inflated_Incremental'] = Predicted_df['Predicted_Inflated_Incremental']
for row in range(0, len(Predicted_df)):
    InsurYr = Predicted_df.loc[row, 'InsuredYear']
    LagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    CurrIncremAmt = Predicted_df.loc[row, 'Predicted_Inflated_Incremental']
    Predicted_df.loc[row, 'FutureUplifted_Predicted_Inflated_Incremental'] = CurrIncremAmt * (FutureInflation ** LagYr)
    
print(Predicted_df['FutureUplifted_Predicted_Inflated_Incremental'])


# In[113]:


# Project (Future Inflation) Predicted Incremental Amount
# Non-Inflated
# Set equal for easy reference
Predicted_df['FutureUplifted_Predicted_Incremental'] = Predicted_df['Predicted_Incremental']
FutureInflation = Inflation_df.loc[(Inflation_df['Year'] == (YearEndCap + 1)), 'CumPastInflation'].values[0]

for row in range(0, len(Predicted_df)):
    InsurYr = Predicted_df.loc[row, 'InsuredYear']
    LagYr = Predicted_df.loc[row, 'PredictedYear_Only_Lag']
    CurrIncremAmt = Predicted_df.loc[row, 'Predicted_Incremental']

    Predicted_df.loc[row, 'FutureUplifted_Predicted_Incremental'] = CurrIncremAmt * (FutureInflation ** LagYr)
    
print(Predicted_df['FutureUplifted_Predicted_Incremental'])


# * **8. Preview Predictions (Part C)**
# 
# Now lets view what we have done so far!

# **8.1**	Incremental Amount  

# In[114]:


# Incremental
# Non-Inflated
PredictedTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"], columns=["PredictedYear_Only_Lag"], values=["FutureUplifted_Predicted_Incremental"])
# Inflated
PredictedInflatedTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"], columns=["PredictedYear_Only_Lag"], values=["FutureUplifted_Predicted_Inflated_Incremental"])


# # Non-Inflated Incremental

# In[115]:


SinglePlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='FutureUplifted_Predicted_Incremental')
SubPlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='FutureUplifted_Predicted_Incremental')
display(PredictedTriangle)


# # Inflated Incremental

# In[116]:


SinglePlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='FutureUplifted_Predicted_Inflated_Incremental')
SubPlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='FutureUplifted_Predicted_Inflated_Incremental')
display(PredictedInflatedTriangle)


# **8.2**	Cumulative Amounts

# In[117]:


# Cumulative
# Non-Inflated
PredictedCumTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"], columns=["PredictedYear_Only_Lag"], values=["Predicted_cumsum"])
# Inflated
PredictedInflatedCumTriangle = pd.pivot_table(Predicted_df, index=["InsuredYear"], columns=["PredictedYear_Only_Lag"], values=["Predicted_Inflated_cumsum"])


# # Non-Inflated Cumulative

# In[118]:


SinglePlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='Predicted_cumsum')
SubPlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='Predicted_cumsum')
display(PredictedCumTriangle)


# # Inflated Cumulative

# In[119]:


SinglePlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='Predicted_Inflated_cumsum')
SubPlotPartialClaims(DataFrameName=Predicted_df, InsuredYearColumn='InsuredYear', LagYearColumn='PredictedYear_Only_Lag', ValueColumn='Predicted_Inflated_cumsum')
display(PredictedInflatedCumTriangle)


# **9.**	**Full Triangle**

# **9.0** Define General Plot Functions

# In[120]:


def SinglePlotFullClaims(PastDataFrameName, PastInsuredYearColumn, PastLagYearColumn, PastValueColumn, 
                   FutureDataFrameName, FutureInsuredYearColumn, FutureLagYearColumn, FutureValueColumn):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    # https://stackoverflow.com/questions/16419670/increase-distance-between-title-and-plot-in-matplolib
    """Create New df"""
    Filtered_NewColumnNames = ["Insured_Year","Year_Only_Lag","ClaimAmt"]
    # Past
    Past_Filtered_df = pd.DataFrame(PastDataFrameName[[PastInsuredYearColumn, PastLagYearColumn, PastValueColumn]])
    Past_Filtered_df.columns = Filtered_NewColumnNames
    # Future
    Future_Filtered_df = pd.DataFrame(FutureDataFrameName[[FutureInsuredYearColumn, FutureLagYearColumn, FutureValueColumn]])
    Future_Filtered_df.columns = Filtered_NewColumnNames    
    """Unique Insured Years List"""
    # Past
    Past_InsuredYr_List = list(PastDataFrameName[PastInsuredYearColumn].unique())
    # Future
    Future_InsuredYr_List = list(FutureDataFrameName[FutureInsuredYearColumn].unique())
    """Unique Lag Years List"""
    # Past
    Past_LagYr_List = list(PastDataFrameName[PastLagYearColumn].unique())
    # Future
    Future_LagYr_List = list(FutureDataFrameName[FutureLagYearColumn].unique())
    """Color List"""
    ALL_Colors = ['r','b','g','y','k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']         
    Past_Color_List = ALL_Colors[:len(Past_InsuredYr_List)]
    Future_Color_List = ALL_Colors[:len(Future_InsuredYr_List)]
    """Plotting"""
    fig = plt.figure(2, figsize=(8,12))
    plt.title('Single Plot Full Claims Data')
    """Full Loop Plot"""
    Full_Filtered_df = pd.concat([Past_Filtered_df, Future_Filtered_df])
    for row_A in range(0,len(Past_InsuredYr_List)):
        Year_i = Past_InsuredYr_List[row_A]
        Full_SubFiltered_df = Full_Filtered_df.loc[Full_Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(Full_SubFiltered_df['Year_Only_Lag'], Full_SubFiltered_df['ClaimAmt'], 
                 label=('Predicted %d' % Year_i), linestyle='--', color=Past_Color_List[row_A])
        plt.legend()
        plt.xlabel('Developement Year')
        plt.ylabel('Claims Value')    
    """Past Loop Plot"""
    for row_A in range(0,len(Past_InsuredYr_List)):
        Year_i = Past_InsuredYr_List[row_A]
        Past_SubFiltered_df = Past_Filtered_df.loc[Past_Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(Past_SubFiltered_df['Year_Only_Lag'], Past_SubFiltered_df['ClaimAmt'], 
                 label=('Historical %d' % Year_i), linestyle='-', color=Past_Color_List[row_A], marker='o')
        plt.legend()
    #"""Future Loop Plot"""
    #for row_B in range(0,len(Future_InsuredYr_List)):
    #    Year_i = Future_InsuredYr_List[row_B]
    #    Future_SubFiltered_df = Future_Filtered_df.loc[Future_Filtered_df['Insured_Year'].isin([Year_i])]
    #    plt.plot(Future_SubFiltered_df['Year_Only_Lag'], Future_SubFiltered_df['ClaimAmt'], 
    #             label=str(Year_i), linestyle='--', color=Future_Color_List[row_B])    
    
    """Plot Attributes"""    
    plt.show()


# In[129]:


def SubPlotFullClaims(PastDataFrameName, PastInsuredYearColumn, PastLagYearColumn, PastValueColumn, 
                   FutureDataFrameName, FutureInsuredYearColumn, FutureLagYearColumn, FutureValueColumn):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    # https://stackoverflow.com/questions/16419670/increase-distance-between-title-and-plot-in-matplolib
    """Create New df"""
    Filtered_NewColumnNames = ["Insured_Year","Year_Only_Lag","ClaimAmt"]
    # Past
    Past_Filtered_df = pd.DataFrame(PastDataFrameName[[PastInsuredYearColumn, PastLagYearColumn, PastValueColumn]])
    Past_Filtered_df.columns = Filtered_NewColumnNames
    # Future
    Future_Filtered_df = pd.DataFrame(FutureDataFrameName[[FutureInsuredYearColumn, FutureLagYearColumn, FutureValueColumn]])
    Future_Filtered_df.columns = Filtered_NewColumnNames    
    """Unique Insured Years List"""
    # Past
    Past_InsuredYr_List = list(PastDataFrameName[PastInsuredYearColumn].unique())
    # Future
    Future_InsuredYr_List = list(FutureDataFrameName[FutureInsuredYearColumn].unique())
    """Unique Lag Years List"""
    # Past
    Past_LagYr_List = list(PastDataFrameName[PastLagYearColumn].unique())
    # Future
    Future_LagYr_List = list(FutureDataFrameName[FutureLagYearColumn].unique())
    """Color List"""
    ALL_Colors = ['r','b','g','y','k', 'c', 'm', 'saddlebrown', 'pink', 'lawngreen']         
    Past_Color_List = ALL_Colors[:len(Past_InsuredYr_List)]
    Future_Color_List = ALL_Colors[:len(Future_InsuredYr_List)]
    """Plotting"""
    fig = plt.figure(2, figsize=(12,16))
    plt.xticks([]) # remove initial blank plot default ticks
    plt.yticks([]) # remove initial blank plot default ticks
    plt.title('Sub Plot Full Claims Data')
    rcParams['axes.titlepad'] = 50 # position title
    plt.box(on=None) # Remove boundary line
    """Full Loop Plot"""
    Full_Filtered_df = pd.concat([Past_Filtered_df, Future_Filtered_df])
    i=0
    for row_A in range(0,len(Past_InsuredYr_List)):
        ax = fig.add_subplot(5, 2, 1+i)
        Year_i = Past_InsuredYr_List[row_A]
        Full_SubFiltered_df = Full_Filtered_df.loc[Full_Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(Full_SubFiltered_df['Year_Only_Lag'], Full_SubFiltered_df['ClaimAmt'], 
                 label=('Predicted %d' % Year_i), linestyle='--', color=Past_Color_List[row_A])
        plt.legend()
        i += 1
        plt.xticks(np.arange(0, (YearEndCap-YearStartCap), step=1))
        plt.xlabel('Developement Year')
        plt.ylabel('Claims Value') 
    """Past Loop Plot"""
    i=0
    for row_A in range(0,len(Past_InsuredYr_List)):
        ax = fig.add_subplot(5, 2, 1+i)
        Year_i = Past_InsuredYr_List[row_A]
        Past_SubFiltered_df = Past_Filtered_df.loc[Past_Filtered_df['Insured_Year'].isin([Year_i])]
        plt.plot(Past_SubFiltered_df['Year_Only_Lag'], Past_SubFiltered_df['ClaimAmt'], 
                 label=('Historical %d' % Year_i), linestyle='-', color=Past_Color_List[row_A], marker='o')
        plt.legend()
        i += 1
    #"""Future Loop Plot"""
    #for row_B in range(0,len(Future_InsuredYr_List)):
    #    Year_i = Future_InsuredYr_List[row_B]
    #    Future_SubFiltered_df = Future_Filtered_df.loc[Future_Filtered_df['Insured_Year'].isin([Year_i])]
    #    plt.plot(Future_SubFiltered_df['Year_Only_Lag'], Future_SubFiltered_df['ClaimAmt'], 
    #             label=str(Year_i), linestyle='--', color=Future_Color_List[row_B])    
    """Plot Attributes"""    
    fig.tight_layout()
    plt.show()


# # 9.1 Non-Inflated Claims

# In[122]:


SinglePlotFullClaims(PastDataFrameName=py_data, PastInsuredYearColumn="Insured_Year", PastLagYearColumn="Year_Only_Lag", PastValueColumn="cumsum", 
               FutureDataFrameName=Predicted_df, FutureInsuredYearColumn="InsuredYear", FutureLagYearColumn="PredictedYear_Only_Lag", FutureValueColumn="Predicted_cumsum")


# In[130]:


SubPlotFullClaims(PastDataFrameName=py_data, PastInsuredYearColumn="Insured_Year", PastLagYearColumn="Year_Only_Lag", PastValueColumn="cumsum", 
               FutureDataFrameName=Predicted_df, FutureInsuredYearColumn="InsuredYear", FutureLagYearColumn="PredictedYear_Only_Lag", FutureValueColumn="Predicted_cumsum")


# # 9.2 Inflated Claims

# In[124]:


SinglePlotFullClaims(PastDataFrameName=py_data, PastInsuredYearColumn="Insured_Year", PastLagYearColumn="Year_Only_Lag", PastValueColumn="Inflated_cumsum", 
               FutureDataFrameName=Predicted_df, FutureInsuredYearColumn="InsuredYear", FutureLagYearColumn="PredictedYear_Only_Lag", FutureValueColumn="Predicted_Inflated_cumsum")


# In[131]:


SubPlotFullClaims(PastDataFrameName=py_data, PastInsuredYearColumn="Insured_Year", PastLagYearColumn="Year_Only_Lag", PastValueColumn="Inflated_cumsum", 
               FutureDataFrameName=Predicted_df, FutureInsuredYearColumn="InsuredYear", FutureLagYearColumn="PredictedYear_Only_Lag", FutureValueColumn="Predicted_Inflated_cumsum")


# * **10. Reserves**
# 
# Last but simplest step of all. The amount insurers need to cover their predicted claim costs, assuming past trends continue.

# **10.1** Inflated Amounts

# In[125]:


InflatedReserves = Predicted_df['FutureUplifted_Predicted_Inflated_Incremental'].sum()
print(InflatedReserves)


# **10.2** Non Inflated Amounts

# In[126]:


NonInflatedReserves = Predicted_df['FutureUplifted_Predicted_Incremental'].sum()
print(NonInflatedReserves)


# In[135]:


PercDiff = 100*(InflatedReserves/NonInflatedReserves-1)
print('Percentage Difference {}'.format(PercDiff))


# **Conclusion**
# 
# Evidently, the inflated reserves far exceed that of the non inflated reserves. Despite inflation rates falling, the fact that we uplifted the amounts is a an easy attribution to the reserve results.
# 
# Do stay tuned as I plan to do some sensitivity analysis on this going forward!

# Thank you for reading till the end! Hope you now have a deeper understanding of Data Manipulation using Pandas & also IACL calculations.
# 
# Cheers!

# 
