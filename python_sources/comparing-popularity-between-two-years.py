import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

con = sqlite3.connect('../input/database.sqlite')

state_names = pd.read_sql_query("SELECT Name, State, Year, Gender, Count FROM StateNames", con)
national_names = state_names.groupby(['Year', 'Gender', 'Name'], as_index=False)[['Count']].sum()

state_totals = state_names.groupby(['State', 'Year', 'Gender'], as_index=False)[['Count']].sum().rename(columns={'Count': 'StateCount'})
national_totals = national_names.groupby(['Year', 'Gender'], as_index=False)[['Count']].sum().rename(columns={'Count': 'NationalCount'})

state_totals = pd.merge(state_totals, national_totals, on=['Year', 'Gender'])

state_totals['StateWeight'] = state_totals['StateCount'] / state_totals['NationalCount']

state_name_ratios = pd.merge(state_names, state_totals, on=['State', 'Year', 'Gender'])
state_name_ratios['StateRatio'] = state_name_ratios['Count'] / state_name_ratios['StateCount']

national_name_ratios = pd.merge(national_names, national_totals, on=['Year', 'Gender']).rename(columns={'Year': 'NationalYear'})
national_name_ratios['NationalRatio'] = national_name_ratios['Count'] / national_name_ratios['NationalCount']

plt.figure(figsize=(12,9))

def plotYear(year):
    relevant_year_ratios = national_name_ratios[national_name_ratios['NationalYear'] == year]
    
    state_national_ratios = pd.merge(state_name_ratios, relevant_year_ratios, how='outer', on=['Name', 'Gender'])
    state_national_ratios['Delta'] = abs(state_national_ratios['StateRatio'] - state_national_ratios['NationalRatio']) * state_national_ratios['StateWeight'] * 1000
    
    state_deltas = state_national_ratios.groupby(['State', 'Year'], as_index=False)[['Delta']].mean()
    
    plt.clf()

    ax = plt.gca()
    ax.set_xlim([state_names[['Year']].min()[0], state_names[['Year']].max()[0]])
    ax.set_ylim([0, 0.25])

    for key, grp in state_deltas.groupby(['State']):
        plt.plot(grp['Year'], grp['Delta'], label=key)

    plt.savefig('{0}.png'.format(year))
    print('Saved {0}.'.format(year))

years_to_plot = [1910, 1920, 1940, 1960, 1980, 2000, 2014]

for year in years_to_plot:
    plotYear(year)