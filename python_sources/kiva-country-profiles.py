#!/usr/bin/env python
# coding: utf-8

# In [my previous kernel](https://www.kaggle.com/goomba16/mapping-poverty-around-the-world) I looked at some general mappings of low and high MPI countries. In this kernel I focus on Kiva countries only. I generate country profiles based on Kiva loans data and external country level data.  Each profile contains the following information:
# 
# 1. Percentage of the loans that were funded
# 2. How long Kiva has been operating in the country - I look at the date of the earliest loan in the country and find the number of months from that date until now. Since the earliest date I could find for all of the countries was 49 months ago, I express the output as (number of months for the country)/49 months
# 3. Ratio of loans to borrowers - for countires for individual loans are popular this ratio is equal to or close to 1:1. For countries were group loans are popular the ratio will be more unbalanced.
# 4. Percentage of female borrowers out of total borrowers - as a general trend, there are more female borrowers on Kiva than male ones but this trend doesn't hold for all countries. Some countries have higher percentage of male borrowers. 
# 5. Total female borrowers per sector - some loan sectors, such as Agriculture, Food and Retail, are quite a bit more popular than others, but there are regional variations.
# 6. Total male borrowers per sector - some loan sectors, such as Agriculture, Food and Retail, are quite a bit more popular than others, but there are regional variations.
# 7. Repayment intervals - there are large variations in repayment intervals per country. In some countries bullet repayments are very popular, in others - irregular ones, etc.
# 8. Borrower group types - whether loan borrower(s) is/are a single female, single male, female pair, male pair, female and male pair, all female group or mixed group. A piece of profiling pinched from *doyouevendata* at [this kernel](https://www.kaggle.com/doyouevendata/kiva-exploration-by-a-kiva-lender-and-python-newb)  :)
# 9. Currencies loans taken in and total loans in each currency - loans are either taken in local country currency or US dollar if local currency is week, there are inflation problems etc. 
# 10. Number of loans per field parner - some field partners seem to be responsible for a large number of loans, others for just a few. There's also variation in the number of total field partners - assuming bigger countries need more field partners.
# 11. MPI Rural and MPI Urban for the country
# 12. Headcount Ratio Rural and Headcount Ratio Urban  for the country - the percentage of peple below poverty line in rural and urban areas respectively.
# 13. Individuals using the internet (% of total population)
# 14. Access to water source in urban and rural areas - percent of urban and rural populations that have access
# 15. Population growth in urban and rural areas
# 16. Global Peace Index score - the higher the score the less peaceful the country
# 17. World Happiness score - the higher the score, the happier the country :) 
# 18. Adult literacy - percent of adults that are literate
# 19. Ratio of female:male literacy - of those that are literate, what proportion is male and what - female.
# 20. Percentage of population employed in agriculture, industry and services - this information was present for very few countries
# 
# If you are curious to see more country profiles, you are welcome to fork the notebook and generate some more :)
# 
# EDIT: Some nice graphs added at the observations section. :)

# In[1]:


import numpy as np
import pandas as pd 
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


# In[15]:


kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')


# In[16]:


kiva_loans['total_borrower_count'] = kiva_loans['borrower_genders'].fillna('').str.split(',').apply(lambda x: len(x))
kiva_loans['female_borrower_count'] = kiva_loans['borrower_genders'].fillna('').apply(lambda x: str(x).replace(' ', '').split(',').count('female'))
kiva_loans['male_borrower_count'] = kiva_loans['borrower_genders'].fillna('').apply(lambda x: str(x).replace(' ', '').split(',').count('male'))
kiva_loans['funded_loan'] = np.where(kiva_loans['funded_amount'] < kiva_loans['loan_amount'], 0, 1)
kiva_loans['one_female_borrower'] = np.where(((kiva_loans['female_borrower_count'] == 1) & (kiva_loans['male_borrower_count'] == 0)), 1, 0)
kiva_loans['one_male_borrower'] = np.where(((kiva_loans['female_borrower_count'] == 0) & (kiva_loans['male_borrower_count'] == 1)), 1, 0)
kiva_loans['female_pair'] = np.where(((kiva_loans['female_borrower_count'] == 2) & (kiva_loans['male_borrower_count'] == 0)), 1, 0)
kiva_loans['male_pair'] = np.where(((kiva_loans['female_borrower_count'] == 0) & (kiva_loans['male_borrower_count'] == 2)), 1, 0)
kiva_loans['female_and_male_pair'] = np.where(((kiva_loans['female_borrower_count'] == 1) & (kiva_loans['male_borrower_count'] == 1)), 1, 0)
kiva_loans['all_female_group'] = np.where(((kiva_loans['female_borrower_count'] > 2) & (kiva_loans['male_borrower_count'] == 0)), 1, 0)
kiva_loans['mixed_group'] = np.where(((kiva_loans['female_borrower_count'] >= 1) & (kiva_loans['male_borrower_count'] >= 1) & (kiva_loans['total_borrower_count'] > 2)), 1, 0)


# In[17]:


def merge_dfs_on_column(dfs, column_name='country'):
    return reduce(lambda left,right: pd.merge(left,right,on=column_name, how='left'), dfs)

kiva_country_profile = pd.DataFrame({'country': kiva_loans.country.unique()})

total_loans = kiva_loans.groupby(['country']).count().reset_index()[['country', 'id']].rename(columns={'id': 'total_loans'})
total_funded_loans = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'funded_loan']].rename(columns={'funded_loan':'total_funded_loans'})
total_borrowers = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'total_borrower_count']]
total_female_borrowers = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'female_borrower_count']].rename(columns={'female_borrower_count':'total_female_borrowers'})
total_male_borrowers = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'male_borrower_count']].rename(columns={'male_borrower_count':'total_male_borrowers'})
total_one_female_borrowers = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'one_female_borrower']].rename(columns={'one_female_borrower':'total_one_female_borrowers_loans'})
total_one_male_borrowers = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'one_male_borrower']].rename(columns={'one_male_borrower':'total_one_male_borrowers_loans'})
total_female_pair_borrowers = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'female_pair']].rename(columns={'female_pair':'total_female_pair_borrowers_loans'})
total_male_pair_borrowers = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'male_pair']].rename(columns={'male_pair':'total_male_pair_borrowers_loans'})
total_male_female_pair_borrowers = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'female_and_male_pair']].rename(columns={'female_and_male_pair':'total_female_and_male_pair_borrowers_loans'})
total_all_female_group_borrowers = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'all_female_group']].rename(columns={'all_female_group':'total_all_female_group_borrowers_loans'})
total_mixed_group_borrowers = kiva_loans.groupby(['country']).sum().reset_index()[['country', 'mixed_group']].rename(columns={'mixed_group':'total_mixed_group_borrowers_loans'})
total_sector_borrowers = kiva_loans.groupby(['country', 'sector']).sum().reset_index()[['country', 'sector', 'female_borrower_count', 'male_borrower_count']]
currencies_used = kiva_loans.groupby(['country', 'currency']).sum().reset_index()[['country', 'currency']].groupby(['country']).apply(lambda x: list(x.currency)).reset_index().rename(columns={0:'currencies_used'})
loans_in_currency = kiva_loans.groupby(['country', 'currency']).count().reset_index()[['country', 'id']].groupby(['country']).apply(lambda x: list(x.id)).reset_index().rename(columns={0: 'total_loans_in_currency'})
loan_dates = kiva_loans.groupby(['country', 'date']).size().reset_index().groupby('country').first().reset_index().drop(columns=[0])
loan_dates['now'] = '2018-03-01'
loan_dates['months_since_earliest_loan'] = ((pd.to_datetime(loan_dates.now) - pd.to_datetime(loan_dates.date))/ np.timedelta64(1, 'M')).astype(int)
parnter_ids = kiva_loans.groupby(['country', 'partner_id']).count().reset_index().groupby('country').apply(lambda x: list(x.partner_id)).reset_index().rename(columns={0:'parner_ids'})
loans_per_parnter = kiva_loans.groupby(['country', 'partner_id']).count().reset_index()[['country', 'id']].groupby('country').apply(lambda x: list(x.id)).reset_index().rename(columns={0:'loans_per_partner'})
all_repayment_intervals = kiva_loans.groupby(['country', 'repayment_interval']).count().reset_index()[['country', 'repayment_interval', 'id']]
bullet_repayments = all_repayment_intervals[all_repayment_intervals.repayment_interval == 'bullet'].drop(columns=['repayment_interval']).rename(columns={'id': 'total_bullet_repayments'})
irregular_repayments = all_repayment_intervals[all_repayment_intervals.repayment_interval == 'irregular'].drop(columns=['repayment_interval']).rename(columns={'id': 'total_irregular_repayments'})
monthly_repayments = all_repayment_intervals[all_repayment_intervals.repayment_interval == 'monthly'].drop(columns=['repayment_interval']).rename(columns={'id': 'total_monthly_repayments'})
weekly_repayments = all_repayment_intervals[all_repayment_intervals.repayment_interval == 'weekly'].drop(columns=['repayment_interval']).rename(columns={'id': 'total_weekly_repayments'})

dfs = [kiva_country_profile, total_loans, total_funded_loans, total_borrowers, 
       total_female_borrowers, total_male_borrowers, total_one_female_borrowers,
      total_one_male_borrowers, total_female_pair_borrowers, total_male_pair_borrowers,
      total_male_female_pair_borrowers, total_all_female_group_borrowers,
      total_mixed_group_borrowers, currencies_used, loans_in_currency,
      loan_dates.drop(columns=['date', 'now']), parnter_ids, loans_per_parnter,
      bullet_repayments, irregular_repayments, monthly_repayments, weekly_repayments]
kiva_country_profile = merge_dfs_on_column(dfs)

for sector in kiva_loans.sector.unique():
    sector_female_borrowers = total_sector_borrowers[total_sector_borrowers['sector'] == sector][['country', 'female_borrower_count']].rename(columns={'female_borrower_count': 'total_female_borrowers_sector_{}'.format(sector)})
    sector_male_borrowers = total_sector_borrowers[total_sector_borrowers['sector'] == sector][['country', 'male_borrower_count']].rename(columns={'male_borrower_count': 'total_male_borrowers_sector_{}'.format(sector)})
    kiva_country_profile = merge_dfs_on_column([kiva_country_profile, sector_female_borrowers, sector_male_borrowers])

kiva_country_profile = kiva_country_profile.fillna(0)


# In[18]:


mpi_national = pd.read_csv('../input/mpi/MPI_national.csv')
gpi = pd.read_csv('../input/gpi2008-2016/gpi_2008-2016.csv')
whr = pd.read_csv('../input/world-happiness/2016.csv')
wdi = pd.read_csv('../input/world-development-indicators/WDIData.csv')


# In[19]:


country_mappings = {
    'Tanzania, United Republic of': 'Tanzania',
    'Viet Nam': 'Vietnam',
    'Palestine, State ofa': 'Palestine',
    'Bolivia, Plurinational State of': 'Bolivia',
    'Congo, Democratic Republic of the': 'The Democratic Republic of the Congo',
    'Congo, Republic of': 'Congo',
    'Myanmar': 'Myanmar (Burma)',
    'Moldova, Republic of': 'Moldova',
    "Cote d'Ivoire": "Cote D'Ivoire"
}
missing_countries = ['United States', 'Chile', 'Georgia', 'Kosovo', 'Costa Rica', 
                     'Turkey', 'Paraguay', 'Lebanon', 'Samoa', 'Israel', 'Panama',
                     'Virgin Islands', 'Saint Vincent and the Grenadines', 'Solomon Islands',
                     'Guam', 'Puerto Rico']

country_mappings_b = {
    'Democratic Republic of the Congo': 'The Democratic Republic of the Congo',
    'Republic of the Congo': 'Congo',
    'Laos': "Lao People's Democratic Republic",
    'Myanmar': 'Myanmar (Burma)',
    'Ivory Coast': "Cote D'Ivoire"
}
missing_countries_b = ['Samoa', 'Belize', 'Suriname', 'Vanuatu',
                       'Virgin Islands', 'Saint Vincent and the Grenadines', 'Solomon Islands',
                       'Guam', 'Puerto Rico']

country_mappings_c = {
    'Congo (Kinshasa)': 'The Democratic Republic of the Congo',
    'Congo (Brazzaville)': 'Congo',
    'Ivory Coast': "Cote D'Ivoire",
    'Laos': "Lao People's Democratic Republic",
    'Myanmar': 'Myanmar (Burma)',
    'Palestinian Territories': 'Palestine'
}
missing_countries_c = ['Guam', 'Lesotho', 'Mozambique', 'Saint Vincent and the Grenadines',
                       'Samoa', 'Solomon Islands', 'Suriname', 'Timor-Leste', 'Vanuatu', 
                       'Virgin Islands']

country_mappings_d = {
    'Congo, Dem. Rep.': 'The Democratic Republic of the Congo',
    'Congo, Rep.': 'Congo',
    "Cote d'Ivoire": "Cote D'Ivoire",
    'Egypt, Arab Rep.': 'Egypt',
    'Kyrgyz Republic': 'Kyrgyzstan',
    'Lao PDR': "Lao People's Democratic Republic",
    'Myanmar': 'Myanmar (Burma)',
    'West Bank and Gaza': 'Palestine',
    'Virgin Islands (U.S.)': 'Virgin Islands',
    'Yemen, Rep.': 'Yemen'
}
missing_countries_d = ['Saint Vincent and the Grenadines']


# In[20]:


def update_country_names(df, mappings, column_name='Country'):
    for key in mappings.keys():
        df.loc[df[column_name] == key, column_name] = mappings[key]
    return df

def extract_data_for_indicator(df, ind_name, years):
    return df[df['Indicator Name'] == ind_name][[*years, 'country']]

def update_countries_profile(countries_df, indicators, years):
    for i, indicator in enumerate(indicators):
        indicator_df = extract_data_for_indicator(wdi, indicator, [years[i]]).rename(index=str, columns={years[i]: '{} {}'.format(indicator, years[i])})
        countries_df = merge_dfs_on_column([countries_df, indicator_df])
    return countries_df

mpi_national = update_country_names(mpi_national, country_mappings).rename(index=str, columns={'Country': 'country'})
gpi = update_country_names(gpi, country_mappings_b, column_name='country').rename(index=str, columns={'score_2016': 'Global Peace Index 2016'}).drop(columns=['score_2008', 'score_2009', 'score_2010', 'score_2011', 'score_2012', 'score_2013', 'score_2014', 'score_2015'])
whr = update_country_names(whr, country_mappings_c).rename(index=str, columns={'Country': 'country', 'Happiness Score': 'Happiness Score 2016'})[['country', 'Region', 'Happiness Score 2016']]
wdi = update_country_names(wdi, country_mappings_d, column_name='Country Name').rename(index=str, columns={'Country Name': 'country'})

kiva_country_profile = merge_dfs_on_column([kiva_country_profile, mpi_national, gpi, whr])

indicators = ['Literacy rate, adult total (% of people ages 15 and above)',
             'Literacy rate, adult female (% of females ages 15 and above)',
             'Literacy rate, adult male (% of males ages 15 and above)',
             'Urban population growth (annual %)',
             'Rural population growth (annual %)',
             'Improved water source, urban (% of urban population with access)',
             'Improved water source, rural (% of rural population with access)',
             'Prevalence of undernourishment (% of population)',
             'Employment in agriculture (% of total employment)',
             'Employment in industry (% of total employment)',
             'Employment in services (% of total employment)',
             'Individuals using the Internet (% of population)']
years = ['2015' for _ in range(len(indicators))]

kiva_country_profile = update_countries_profile(kiva_country_profile, indicators, years)


# In[21]:


def generate_country_profile(country, allowed_countries=kiva_loans.country.unique(), 
                             df=kiva_country_profile, sectors=kiva_loans.sector.unique(),
                            ris=kiva_loans.repayment_interval.unique()):
    if country not in allowed_countries:
        print('This country is not in the list of Kiva countries.')
        return
    
    borrower_group_types = ['one_female', 'one_male','female_pair', 'male_pair','female_and_male_pair','all_female_group','mixed_group']
    total_borrowers = df[df['country'] == country]['total_borrower_count'].values
    total_loans = df[df['country'] == country]['total_loans'].values
    total_funded_loans = df[df['country'] == country]['total_funded_loans'].values
    months_since_first_loan = df[df['country'] == country]['months_since_earliest_loan'].values
    total_female_borrowers = df[df['country'] == country]['total_female_borrowers'].values
    female_borrowers_per_sector = df[df['country'] == country][['total_female_borrowers_sector_{}'.format(sector) for sector in sectors]].values.flatten()
    male_borrowers_per_sector = df[df['country'] == country][['total_male_borrowers_sector_{}'.format(sector) for sector in sectors]].values.flatten()
    repayment_intervals = df[df['country'] == country][['total_{}_repayments'.format(ri) for ri in ris]].values.flatten()
    borrower_groups = df[df['country'] == country][['total_{}_borrowers_loans'.format(group_type) for group_type in borrower_group_types]].values.flatten()
    currencies = df[df['country'] == country]['currencies_used'].values
    loans_in_currency = df[df['country'] == country]['total_loans_in_currency'].values
    parner_ids = df[df['country'] == country]['parner_ids'].values
    loans_per_partner = df[df['country'] == country]['loans_per_partner'].values
    mpis = df[df['country'] == country][['MPI Urban', 'MPI Rural']].values.flatten()
    hr = df[df['country'] == country][['Headcount Ratio Urban', 'Headcount Ratio Rural']].values.flatten()
    gpi = df[df['country'] == country][['Global Peace Index 2016']].values.flatten()
    hs = df[df['country'] == country][['Happiness Score 2016']].values.flatten()
    internet = df[df['country'] == country][['Individuals using the Internet (% of population) 2015']].values.flatten()
    literacy = df[df['country'] == country][['Literacy rate, adult total (% of people ages 15 and above) 2015']].values.flatten()
    literacy_f = df[df['country'] == country][['Literacy rate, adult female (% of females ages 15 and above) 2015']].values.flatten()
    literacy_m = df[df['country'] == country][['Literacy rate, adult male (% of males ages 15 and above) 2015']].values.flatten()
    undern = df[df['country'] == country][['Prevalence of undernourishment (% of population) 2015']].values.flatten()
    water_s = df[df['country'] == country][['Improved water source, urban (% of urban population with access) 2015',
                                            'Improved water source, rural (% of rural population with access) 2015']].values.flatten()
    employment = df[df['country'] == country][['Employment in agriculture (% of total employment) 2015',
                                              'Employment in industry (% of total employment) 2015',
                                              'Employment in services (% of total employment) 2015']].values.flatten()
    population_growth = df[df['country'] == country][['Urban population growth (annual %) 2015',
                                                    'Rural population growth (annual %) 2015']].values.flatten()
    
    
    def make_ticklabels_invisible(fig):
        for i, ax in enumerate(fig.axes):
            ax.tick_params(labelbottom=False, labelleft=False)

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(20, max(27+len(currencies[0]), 22+len(parner_ids[0]))))
    gs = GridSpec(max(27+len(currencies[0]), 22+len(parner_ids[0])), 20)
    
    ax1 = plt.subplot(gs[1, :-10])
    ax2 = plt.subplot(gs[1, 10:])
    ax3 = plt.subplot(gs[3, :-10])
    ax4 = plt.subplot(gs[3, 10:])
    ax5 = plt.subplot(gs[5:10, :-11])
    ax6 = plt.subplot(gs[5:10, 11:])
    ax7 = plt.subplot(gs[11:14, :-11])
    ax8 = plt.subplot(gs[11:14, 11:])
    ax9 = plt.subplot(gs[15:(14+len(currencies[0])), :-11])
    ax10 = plt.subplot(gs[15:(14+len(parner_ids[0])), 11:])
    ax11 = plt.subplot(gs[(15+len(currencies[0])):(16+len(currencies[0])), :-11])
    ax12 = plt.subplot(gs[(17+len(currencies[0])):(18+len(currencies[0])), :-11])
    ax13 = plt.subplot(gs[(15+len(parner_ids[0])), 11:])
    ax14 = plt.subplot(gs[(17+len(parner_ids[0])), 11:])
    ax15 = plt.subplot(gs[(19+len(currencies[0])), :-11])
    ax16 = plt.subplot(gs[(21+len(currencies[0])), :-11])
    ax17 = plt.subplot(gs[(23+len(currencies[0])), :-11])
    ax18 = plt.subplot(gs[(19+len(parner_ids[0])):(20+len(parner_ids[0])), 11:])
    ax19 = plt.subplot(gs[(21+len(parner_ids[0])):(22+len(parner_ids[0])), 11:])
    ax20 = plt.subplot(gs[(25+len(currencies[0])):(27+len(currencies[0])), :-11])
    
    make_ticklabels_invisible(fig)
    gs.tight_layout(fig, rect=[0, 0.03, 1, 1])
    fig.suptitle(country, fontsize=22)
    
    sns.set_color_codes("muted")
    sns.barplot(x=total_loans, y=[country], color="grey", ax=ax1)
    sns.barplot(x=total_funded_loans, y=[country], color="g", ax=ax1)
    ax1.set_title('Funded loans of total loans (%): {:.3}'.format(((total_funded_loans/total_loans)[0])*100), fontsize=18)
    
    sns.barplot(x=[48], y=[country], color="grey", ax=ax2)
    sns.barplot(x=months_since_first_loan, y=[country], color="g", ax=ax2)
    ax2.set_title('Months since first loan (out of 49 months since Jan 2014): {}'.format(months_since_first_loan[0]), fontsize=18)
    
    sns.barplot(x=[2], y=[country], color="yellow", ax=ax3, label='Total borrowers')
    sns.barplot(x=[(total_loans/total_borrowers)[0]], y=[country], color="orange", ax=ax3, label='Total loans')
    ax3.legend(ncol=2, loc="lower right", frameon=True)
    ax3.set_title('Ratio of loans to borrowers: {:.3}'.format((total_loans/total_borrowers)[0]), fontsize=18)
    
    sns.set_color_codes("pastel")
    sns.barplot(x=total_borrowers, y=[country], color="grey", ax=ax4)
    sns.barplot(x=total_female_borrowers, y=[country], color="r", ax=ax4)
    ax4.set_title('Female borrowers of total borrowers (%): {:.3}'.format(((total_female_borrowers/total_borrowers)[0])*100), fontsize=18)
    
    pal = sns.color_palette("Reds", len(female_borrowers_per_sector))
    rank = female_borrowers_per_sector.argsort().argsort()
    sns.barplot(x=female_borrowers_per_sector, y=sectors, palette=np.array(pal)[rank], ax=ax5)
    for i, p in enumerate(ax5.patches):
        ax5.annotate(int(female_borrowers_per_sector[i]), xy=(p.get_x() + p.get_width() + 0.05, p.get_y() + 0.7*p.get_height()))
    ax5.set_title('Total female borrowers per sector', fontsize=18)
    ax5.tick_params(labelleft=True)
    
    pal = sns.color_palette("Blues", len(male_borrowers_per_sector))
    rank = male_borrowers_per_sector.argsort().argsort()
    sns.barplot(x=male_borrowers_per_sector, y=sectors, palette=np.array(pal)[rank], ax=ax6)
    for i, p in enumerate(ax6.patches):
        ax6.annotate(int(male_borrowers_per_sector[i]), xy=(p.get_x() + p.get_width() + 0.05, p.get_y() + 0.7*p.get_height()))
    ax6.set_title('Total male borrowers per sector', fontsize=18)
    ax6.tick_params(labelleft=True)
    
    sns.set_color_codes("muted")
    rank = repayment_intervals.argsort().argsort()
    sns.barplot(x=repayment_intervals, y=ris, ax=ax7)
    for i, p in enumerate(ax7.patches):
        ax7.annotate(int(repayment_intervals[i]), xy=(p.get_x() + p.get_width() + 0.05, p.get_y() + 0.7*p.get_height()))
    ax7.set_title('Repayment intervals', fontsize=18)
    ax7.tick_params(labelleft=True)
    
    sns.barplot(x=borrower_groups, y=borrower_group_types, ax=ax8)
    for i, p in enumerate(ax8.patches):
        ax8.annotate(int(borrower_groups[i]), xy=(p.get_x() + p.get_width() + 0.05, p.get_y() + 0.7*p.get_height()))
    ax8.set_title('Borrower group types', fontsize=18)
    ax8.tick_params(labelleft=True)
    
    sns.barplot(x=loans_in_currency[0], y=currencies[0], ax=ax9)
    for i, p in enumerate(ax9.patches):
        ax9.annotate(int(loans_in_currency[0][i]), xy=(p.get_x() + p.get_width() + 0.05, p.get_y() + 0.7*p.get_height()))
    ax9.set_title('Number of loans taken in currency', fontsize=18)
    ax9.tick_params(labelleft=True)
    
    sns.barplot(x=loans_per_partner[0], y=['Partner id {}'.format(str(pid)) for pid in parner_ids[0]], ax=ax10, orient='h')
    for i, p in enumerate(ax10.patches):
        ax10.annotate(int(loans_per_partner[0][i]), xy=(p.get_x() + p.get_width() + 0.05, p.get_y() + 0.7*p.get_height()))
    ax10.set_title('Loans per field partner', fontsize=18)
    ax10.tick_params(labelleft=True)
    
    if country not in missing_countries:
        sns.set_color_codes("muted")
        sns.barplot(x=[1, 1], y=['MPI Urban', 'MPI Rural'], color="grey", ax=ax11)
        sns.barplot(x=mpis, y=['MPI Urban', 'MPI Rural'], color="r", ax=ax11)
        for i, p in enumerate([1, 2]):
            ax11.annotate('{:.3}'.format(mpis[i]), xy=(1.005, p - 0.9))
        ax11.set_title('MPIs', fontsize=18)
        ax11.tick_params(labelleft=True, labelbottom=True, labelsize=18)

        sns.barplot(x=[100, 100], y=['Headcount Ratio Urban', 'Headcount Ratio Rural'], color="grey", ax=ax12)
        sns.barplot(x=hr, y=['HR Urban', 'HR Rural'], color="r", ax=ax12)
        for i, p in enumerate([1, 2]):
            ax12.annotate('{:.3}'.format(hr[i]), xy=(100.005, p - 0.9))
        ax12.set_title('Headcount Ratios (% of population listed as poor)', fontsize=18)
        ax12.tick_params(labelleft=True, labelsize=18)
        
    if country not in missing_countries_b:
        sns.set_color_codes("muted")
        sns.barplot(x=[4], y=[country], color="grey", ax=ax13)
        sns.barplot(x=gpi, y=[country], color="r", ax=ax13)
        ax13.set_title('Global Peace Index 2016 score: {:.4}'.format(gpi[0]), fontsize=18)
        
    if country not in missing_countries_c:
        sns.set_color_codes("muted")
        sns.barplot(x=[7.6], y=[country], color="grey", ax=ax14)
        sns.barplot(x=hs, y=[country], color="g", ax=ax14)
        ax14.set_title('Happiness Score 2016: {:.3}'.format(hs[0]), fontsize=18)
        
    if country not in missing_countries_d:
        sns.set_color_codes("muted")
        sns.barplot(x=[100], y=[country], color="grey", ax=ax15)
        sns.barplot(x=internet, y=[country], color="g", ax=ax15)
        ax15.set_title('Individuals using the Internet (% of population) in 2015: {:.3}%'.format(internet[0]), fontsize=18)
        
        if not np.isnan(water_s).any():
            sns.barplot(x=[100, 100], y=['Water Access Urban', 'Water Access Rural'], color="grey", ax=ax16)
            sns.barplot(x=water_s, y=['Water Access Urban', 'Water Access Rural'], color="g", ax=ax16)
            for i, p in enumerate([1, 2]):
                ax16.annotate(water_s[i], xy=(100.005, p - 0.9))
            ax16.set_title('Improved water source (% of population with access) 2015', fontsize=18)
            ax16.tick_params(labelleft=True, labelsize=18)
            
        if not np.isnan(population_growth).any():
            sns.barplot(x=[10, 10], y=['Population Growth Urban', 'Population Growth Rural'], color="grey", ax=ax17)
            sns.barplot(x=population_growth, y=['Population Growth Urban', 'Population Growth Rural'], color="r", ax=ax17)
            for i, p in enumerate([1, 2]):
                ax17.annotate('{:.3}'.format(population_growth[i]), xy=(10.005, p - 0.9))
            ax17.set_title('Population growth (annual %) in 2015', fontsize=18)
            ax17.tick_params(labelleft=True, labelsize=18)
            
        if not np.isnan(literacy).any():
            sns.barplot(x=[100], y=[country], color="grey", ax=ax18)
            sns.barplot(x=literacy, y=[country], color="g", ax=ax18)
            ax18.set_title('Literacy rate, adult total (% of people ages 15 and above) in 2015: {:.3}%'.format(literacy[0]), fontsize=18)
            
            sns.set_color_codes("pastel")
            sns.barplot(x=[2], y=[country], color="b", ax=ax19, label='Male literacy')
            sns.barplot(x=[literacy_f[0]/literacy_m[0]], y=[country], color="r", ax=ax19, label='Female literacy')
            ax19.legend(ncol=2, loc="lower right", frameon=True)
            ax19.set_title('Ratio of female literacy to male literacy (of people ages 15 and above) for 2015: {:.3}'.format(literacy_f[0]/literacy_m[0]), fontsize=18)
            
        if not np.isnan(employment).any():
            sns.set_color_codes("muted")
            sns.barplot(x=employment, y=['Agriculture', 'Industry', 'Services'], ax=ax20)
            for i, p in enumerate(ax20.patches):
                ax20.annotate('{:.3}'.format(employment[i]), xy=(p.get_x() + p.get_width() + 0.05, p.get_y() + 0.7*p.get_height()))
            ax20.set_title('Employment sectors (% of total employment) in 2015', fontsize=18)
            ax20.tick_params(labelleft=True, labelsize=18)


# In[22]:


generate_country_profile('Philippines')


# In[ ]:


generate_country_profile('Nigeria')


# In[23]:


generate_country_profile('Bolivia')


# In[ ]:


generate_country_profile('India')


# ### Some observations

# Previously I just listed them, but I thought it would be nice to add a few graphs too :)

# * For high MPI values, there is usually a big gap between rural and urban MPI. The lower the MPI, the smaller the gap becomes - we can see from the plot below that when MPI Urban is between 0 and 0.5, MPI Rural is of similar value and close to the drawn line. However, for small changes in MPI Urban, MPI Rural changes are much greater and points are scattered much further from the line. 

# In[ ]:


mpis = kiva_country_profile[['MPI Urban', 'MPI Rural']].dropna()

sns.set_style("darkgrid")
plt.scatter(mpis['MPI Rural'], mpis['MPI Urban'])
plt.plot([0, 0.5], [0, 0.5])
plt.title('MPI Rural vs MPI Urban')
plt.ylabel('MPI Urban')
plt.xlabel('MPI Rural')
plt.show()


# * The same trend present for urban water access and rural water access - when urban water access is close to 100%, so is the rural one (the points are scattered close to the line at 95 to 100 Urban and Rural water source values). However, for a small decrease in urban water access, rural water access decrease is greater and points are scattered further from the line. 

# In[ ]:


water = kiva_country_profile[['Improved water source, urban (% of urban population with access) 2015', 
                              'Improved water source, rural (% of rural population with access) 2015']].dropna()

sns.set_style("darkgrid")
plt.scatter(water['Improved water source, rural (% of rural population with access) 2015'], 
            water['Improved water source, urban (% of urban population with access) 2015'])
plt.plot([0, 100], [0, 100])
plt.title('Water source Rural vs Water source Urban')
plt.ylabel('Water source Urban')
plt.ylim([0, 105])
plt.xlabel('Water source Rural')
plt.show()


# * Low literacy is also associated with unbalanced female:male literacy ratio, with higher proportion of those literate being male, whereas in lower MPI countries this ratio is close to 1:1 - the graph below shows the trend beautifully. The outlier on the graph is Lesotho, with female to male literacy ratio at 1.26.

# In[ ]:


literacy = kiva_country_profile[['Literacy rate, adult total (% of people ages 15 and above) 2015', 
                              'Literacy rate, adult female (% of females ages 15 and above) 2015',
                                'Literacy rate, adult male (% of males ages 15 and above) 2015']].dropna()
literacy['ratio'] = literacy['Literacy rate, adult female (% of females ages 15 and above) 2015']/literacy['Literacy rate, adult male (% of males ages 15 and above) 2015']

sns.set_style("darkgrid")
plt.scatter(literacy['ratio'], 
            literacy['Literacy rate, adult total (% of people ages 15 and above) 2015'])
plt.title('Total literacy vs Ratio of female to male literacy')
plt.ylabel('Total literacy')
plt.xlabel('Ratio of female to male literacy')
plt.show()


# * Very small proportion of population with access to Internet connection in high MPI countries - we can see the trend summarized in the graph below: the high MPI scores correlate with small proportion of population using the Internet. The outliers are South Sudan (Urban MPI Score of 0.459 and 17.93% of population using the Internet) and Nigeria (Urban MPI Score of 0.132 and 47.44% of the population using the Internet).

# In[ ]:


internet = kiva_country_profile[['Individuals using the Internet (% of population) 2015', 'MPI Urban']].dropna()

sns.set_style("darkgrid")
plt.scatter(internet['MPI Urban'], 
            internet['Individuals using the Internet (% of population) 2015'])
plt.title('MPI Urban vs Individuals using the Internet (% of population)')
plt.ylabel('Individuals using the Internet (% of population)')
plt.xlabel('MPI Urban')
plt.show()


# * High MPI countries often have small loans to borrowers ratios, i.e. there are often more group borrowers than individual borrowers - we can see below that for low MPIs, we have quite a few low ratio values. However, not enough to claim a correlation, especially due to number of outliers at 1:1 loans to borrowers ratios (i.e. countries with high Urban MPI where group loans are not popular at all). 

# In[ ]:


gr = kiva_country_profile[['total_borrower_count', 'total_loans', 'MPI Urban']].dropna()
gr['ratio'] = gr['total_loans']/gr['total_borrower_count']

sns.set_style("darkgrid")
plt.scatter(gr['MPI Urban'], 
            gr['ratio'])
plt.title('MPI Urban vs Ratio of Kiva total loans to total borrowers')
plt.ylabel('Ratio of Kiva total loans to total borrowers')
plt.xlabel('MPI Urban')
plt.show()


# ### Next Steps

# I explored numerous of external metrics plotted in the country profiles (MPIs, Global Peace Index, World Happiness score, a number of World Development Indicators) as potential features for modelling total borrower predictions for different sectors. However, Lasso was quite ruthless on the matter - I found that only a couple of World Development Indicators could be used as features for modelling Education and Housing sectors borrower numbers, not other sectors. The broader geographic region of the country was identified as a relevant feature for quite a few more sectors though. I haven't yet looked into sub-country level or finding ways to improve on [Kiva's current Poverty Targeting system](https://www.kaggle.com/annalie/kivampi), but that seems like a thing to do next :)
