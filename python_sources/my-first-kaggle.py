# Questions:
#  1. Cars of which brand can be sold quicker in Germany.
#  2. Cars of which fuel type can be sold quicker in Germany.
#  3. Cars of which type can be sold quicker in Germany.
# Task for machine learning:
#  4. By car brand, year, power and KM predict selling price.
#    Think that car should be sold in one month.

# From the data description:
#   The fields lastSeen and dateCreated could be used to estimate
#     how long a car will be at least online before it is sold.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
full_table = pd.read_csv('../input/autos.csv', sep=','
    , header=0, encoding='cp1252')
print(full_table.size)
# full_table[full_table['seller'] == 'gewerblich'].count()
# => 3
# Here is only three business sellers, so we will drop them and will not use
#  this column at all.
full_table = full_table[full_table.seller == 'privat']

# also we don't need cars, which people are looking for. Will keep only
#  those, which are for sale
full_table = full_table[full_table.offerType == 'Angebot']

# I will analyse only cars which are repaired or not damaged
full_table = full_table.query("notRepairedDamage != 'ja'")

# Drop useless columns
full_table.drop(['dateCrawled', 'seller', 'offerType', 'abtest'
    , 'nrOfPictures', 'monthOfRegistration']
    , axis='columns', inplace=True)

# Drop data which seems to be not relevant or just weird
full_table = full_table[
    (full_table.yearOfRegistration >= 1996)
    & (full_table.yearOfRegistration < 2016)
    & (full_table.powerPS > 50)
    & (full_table.kilometer > 1000)
    & (full_table.postalCode > 10000)
    & (full_table.postalCode < 99999)
    ]

# Add column which will represent how fast the car was sold
full_table['soldWithin'] = pd.to_datetime(full_table.lastSeen) - pd.to_datetime(full_table.dateCreated)


# I don't believe that car could be sold faster than 3 hours.
#  I had experience selling it %)
# Also if car wasn't sold for more than 6 months - seems that seller
#  just forgot to remove it or we have some issue in data.
full_table = full_table[
    (full_table.soldWithin > pd.to_timedelta('3 hours'))
    & (full_table.soldWithin < pd.to_timedelta('180 days'))
    ]

full_table['soldWithinInSeconds'] = full_table['soldWithin'].astype(int)

# Answering first Question
cars_of_which_brand_can_be_sold_faster = full_table.groupby('brand').agg({
    'soldWithinInSeconds': np.mean
    }).sort_values('soldWithinInSeconds', ascending=True).head(5)

print("Cars of which brands can be sold faster:");
print(cars_of_which_brand_can_be_sold_faster);

cars_of_which_fuel_type_can_be_sold_faster = full_table.groupby('fuelType').agg({
    'soldWithinInSeconds': np.mean
    }).sort_values('soldWithinInSeconds', ascending=True).head(1)
    
print("Cars of what fuel type can be sold faster:");
print(cars_of_which_fuel_type_can_be_sold_faster);

cars_of_which_type_can_be_sold_faster = full_table.groupby('vehicleType').agg({
    'soldWithinInSeconds': np.mean
    }).sort_values('soldWithinInSeconds', ascending=True).head(2)
    
print("Cars of which type can be sold faster:");
print(cars_of_which_type_can_be_sold_faster);