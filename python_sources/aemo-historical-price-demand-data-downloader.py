"""
NOTE: COPY THIS SCRIPT TO A LOCAL WORKING DIRECTORY AND THEN RUN.
I CAN'T GET THE REQUESTS LIBRARY TO WORK ON KAGGLE AND I'M GIVING
UP. 

Annoyingly AEMO only allows you to download price and demand data
from their website by month rather than by year or specific time
period. This module tries to make downloading AEMO price and
demand data over a specific time period less laborious. 

Downloads monthly price and demand data from AEMO and combines the
monthly data into a single pandas dataframe. This is then written 
to an excel file in the current working directory (on unix pwd to 
check working directory). 

AEMO takes 1 month to finalise price and demand data, so the end
date for the data download should be 2 months prior to todays 
date if you want finalised data from AEMO. 

User-Agent needed so that AEMO thinks that data is being accessed
via a valid web browser.

NOTE: Consider threading this to speed up download time. 

Uses an AEMO URL request like the below: 

AEMO URL: 
-----------------------
https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_201902_QLD1.csv

"""

import io
from datetime import datetime
import requests
import pandas as pd
from dateutil.rrule import rrule, MONTHLY
import numpy as np


def month_iter(start_month, start_year, end_month, end_year):
    """
    Returns a generator that yields a tuple like (month, year). Generator
    has monthly frequency between start and end dates.
    """
    start = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    return ((d.month, d.year) for d in rrule(MONTHLY, dtstart=start, until=end))


def download_aemo_data(state, start_month, start_year, end_month, end_year):
    """
    Downloads price and demand data from AEMO from 1st day of the starting
    month in the starting year to the last day of the ending month in the
    ending year. Returns pandas DataFrame.
    """
    aemo_df = pd.DataFrame()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) \
        AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    for date in month_iter(start_month, start_year, end_month, end_year):
        month = str(date[0]).zfill(2)
        year = str(date[1])
        reference = year + month
        print(f'Adding Month/Year: {month}/{year}')
        url = f'https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{reference}_{state}1.csv'
        url_data = requests.get(url, headers=headers).content
        raw_data = pd.read_csv(io.StringIO(url_data.decode('utf-8')))
        aemo_df = aemo_df.append(raw_data, ignore_index=True)
    aemo_df['datetime'] = pd.to_datetime(aemo_df['SETTLEMENTDATE'])
    aemo_df = aemo_df.set_index('datetime')
    aemo_df.drop(['SETTLEMENTDATE', 'REGION', 'PERIODTYPE'],
                 axis=1, inplace=True)
    return aemo_df


def calculate_std_dev(aemo_df, start_time, end_time):
    """
    Calculates the quarterly and annual standard deviation for the specified time
    period as a percentage and then adds value to the first row of a new
    column in the dataframe. This historical volatility can be compared
    against the implied volatility in options on the ASX for the NEM.
    """
    aemo_df = aemo_df.between_time(start_time, end_time)
    aemo_df_qtr = pd.pivot_table(aemo_df, values='RRP', index=[
        aemo_df.index.year, aemo_df.index.quarter], aggfunc=np.mean)
    aemo_df_ann = pd.pivot_table(
        aemo_df, values='RRP', index=[aemo_df.index.year], aggfunc=np.mean)
    qtr_std_pct = np.std(aemo_df_qtr.to_numpy(), ddof=1) / \
        np.mean(aemo_df_qtr.to_numpy())
    ann_std_pct = np.std(aemo_df_ann.to_numpy(), ddof=1) / \
        np.mean(aemo_df_ann.to_numpy())
    aemo_df_qtr.loc[:, 'QUARTERLY_STDDEV'] = qtr_std_pct
    aemo_df_ann.loc[:, 'ANNUAL_STDDEV'] = ann_std_pct
    return aemo_df_qtr, aemo_df_ann


STATE = 'VIC'
START_MONTH = 1
START_YEAR = 2017
END_MONTH = 12
END_YEAR = 2018

AEMO_DATA_DF = download_aemo_data(
    STATE, START_MONTH, START_YEAR, END_MONTH, END_YEAR)
# Prints DataFrame to current working directory
AEMO_DATA_DF.to_csv('aemo_price_demand_data.csv')

# Want standard deviation of sunlight hours (6am->6pm) only
# in calculate_std_dev function.
AEMO_DATA_QTRLY, AEMO_DATA_ANN = calculate_std_dev(
    AEMO_DATA_DF, start_time='06:00', end_time='18:00')
print(AEMO_DATA_QTRLY)
print(AEMO_DATA_ANN)
