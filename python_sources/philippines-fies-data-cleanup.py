#!/usr/bin/env python
# coding: utf-8

# In[1]:


#numeric
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium

plt.style.use('bmh')
get_ipython().run_line_magic('matplotlib', 'inline')

#system
import os
print(os.listdir("../input"));


# In[2]:


loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
loans.head()


# In[3]:


phil_inc = pd.read_csv('../input/family-income-and-expenditure/Family Income and Expenditure.csv')
phil_inc.head()


# In[4]:


phil_inc.columns


# In[5]:


phil_loans = loans[loans.country == 'Philippines']
phil_loans.head()


# In[6]:


phil_inc.Region.unique().tolist()


# In[7]:


loan_regions = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
loan_regions.head()


# In[8]:


loan_regions[loan_regions.country == 'Philippines'].region.unique().tolist()


# In[9]:


geonames_phil = pd.read_csv('../input/administrative-regions-in-the-philippines/ph_regions.csv')
geonames_phil.head()


# In[10]:


geonames_phil.info()


# In[11]:


geonames_phil.region.unique().tolist()


# In[12]:


region_mapping_phil_inc = {'CAR' : 'car',                           'Caraga' : 'caraga',                           'VI - Western Visayas' : 'western visayas',                           'V - Bicol Region' : 'bicol',                           ' ARMM' : 'armm',                           'III - Central Luzon' : 'central luzon',                           'II - Cagayan Valley' : 'cagayan valley',                           'IVA - CALABARZON' : 'calabarzon',                           'VII - Central Visayas' : 'central visayas',                           'X - Northern Mindanao' : 'northern mindanao',                           'XI - Davao Region' : 'davao',                           'VIII - Eastern Visayas' : 'eastern visayas',                           'I - Ilocos Region' : 'ilocos',                           'NCR' : 'ncr',                           'IVB - MIMAROPA' : 'mimaropa',                           'XII - SOCCSKSARGEN' : 'soccsksargen',                           'IX - Zasmboanga Peninsula' : 'zamboanga'}


# In[13]:


region_mapping_phil_loans = {'National Capital Region' : 'ncr',                             'Cordillera Admin Region' : 'car',                             'Ilocos Region' : 'ilocos',                             'Cagayan Valley' : 'cagayan valley',                             'Central Luzon' : 'central luzon',                             'Calabarzon' : 'calabarzon',                             'Mimaropa' : 'mimaropa',                             'Bicol Region' : 'bicol',                             'Western Visayas' : 'western visayas',                             'Central Visayas' : 'central visayas',                             'Eastern Visayas' : 'eastern visayas',                             'Zamboanga Peninsula' : 'zamboanga',                             'Northern Mindanao' : 'northern mindanao',                             'Davao Peninsula' : 'davao',                             'Soccsksargen' : 'soccsksargen',                             'CARAGA' : 'caraga',                             'Armm' : 'armm'}


# In[14]:


from difflib import get_close_matches

def match_region(loc_string, match_entity = 'province', split = True):
    if split == True:
        region = loc_string.split(',')[-1]
    else:
        region = loc_string
    
    matches = get_close_matches(region, geonames_phil[match_entity].unique().tolist())
    
    if not matches:
        return 'no_match'
    else:
        return geonames_phil.region[geonames_phil[match_entity] == matches[0]].iloc[0]


# In[ ]:


phil_loans.region.fillna('', inplace = True)
phil_loans.rename(columns = {'region' : 'location'}, inplace = True)
phil_loans['region'] = [match_region(loc_string) for loc_string in phil_loans.location]


# In[ ]:


phil_loans[['location', 'region']].head(20)


# In[ ]:


len(phil_loans[phil_loans.region == 'no_match'])


# In[ ]:


import re
city_drop = re.compile(r'(.*)(city)', re.I)
phil_loans.location[phil_loans.region == 'no_match'] = [re.match(city_drop, l).group(1).lower() if re.match(city_drop, l) else l for l in phil_loans.location[phil_loans.region == 'no_match']]


# In[ ]:


phil_loans[['location', 'region']][phil_loans.region == 'no_match'].head()


# In[ ]:


phil_loans['region'][phil_loans.region == 'no_match'] = np.vectorize(match_region)(phil_loans['location'][phil_loans.region == 'no_match'], 'city', False)


# In[ ]:


len(phil_loans[phil_loans.region == 'no_match'])


# In[ ]:


print(len(phil_loans[phil_loans.location == 'Sogod Cebu']))


# In[ ]:


phil_loans.region[phil_loans.location == 'Sogod Cebu'] = geonames_phil.region[geonames_phil.city == 'cebu'].iloc[0]


# In[ ]:


len(phil_loans[phil_loans.region == 'no_match'])


# In[ ]:


phil_inc.Region = phil_inc.Region.map(region_mapping_phil_inc)


# In[ ]:


phil_inc.head()


# In[ ]:


phil_inc[['Region', 'Total Food Expenditure', 'Total Household Income']].groupby(by = 'Region').mean().reset_index()


# In[ ]:


phil_inc['Main Source of Income'].unique()


# In[ ]:


phil_inc['Main Source of Income'] = phil_inc['Main Source of Income'].map({'Wage/Salaries' : 'main_inc_wage',      'Other sources of Income' : 'main_inc_other',      'Enterpreneurial Activities' : 'main_inc_entrepreneur'})
phil_inc.head()


# In[ ]:


phil_inc_extract = phil_inc.join(pd.get_dummies(phil_inc['Main Source of Income']))
phil_inc_extract.drop(['Main Source of Income', 'Bread and Cereals Expenditure',                       'Total Rice Expenditure', 'Meat Expenditure',                       'Total Fish and  marine products Expenditure',                       'Fruit Expenditure', 'Vegetables Expenditure'],                      axis = 1,                      inplace = True)


# In[ ]:


phil_inc_extract.head()


# In[ ]:


phil_inc_extract['non_essential_expenses'] = phil_inc_extract['Restaurant and hotels Expenditure'] +                                             phil_inc_extract['Alcoholic Beverages Expenditure'] +                                             phil_inc_extract['Tobacco Expenditure']

phil_inc_extract.drop(['Restaurant and hotels Expenditure',                       'Alcoholic Beverages Expenditure',                       'Tobacco Expenditure'],                      axis = 1,                      inplace = True)

phil_inc_extract.head()


# In[ ]:


phil_inc_extract['Household Head Sex'].unique()


# In[ ]:


phil_inc_extract['Household Head Sex'] = phil_inc_extract['Household Head Sex'].map({'Female' : 1,      'Male' : 0})

phil_inc_extract.rename(columns = {'Household Head Sex' : 'house_head_sex_f'}, inplace = True)


# In[ ]:


phil_inc_extract['Household Head Marital Status'].unique()


# In[ ]:


single_civil_statuses = ['Single', 'Widowed', 'Divorced/Separated', 'Annulled', 'Unknown']

phil_inc_extract['Household Head Marital Status'] = ['house_head_single'                                                     if s in single_civil_statuses                                                     else 'house_head_partner'                                                     for s                                                     in phil_inc_extract['Household Head Marital Status']]

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Household Head Marital Status']))

phil_inc_extract.drop(['Household Head Marital Status'],                      axis = 1,                      inplace = True)


# In[ ]:


illiterate = ['No Grade Completed', 'Preschool']
primary_ed = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Elementary Graduate']
secondary_ed = ['First Year High School', 'Second Year High School', 'Third Year High School', 'High School Graduate']

def get_education_level(ed_string):
    if ed_string in illiterate:
        return 'house_head_illiterate'
    elif ed_string in primary_ed:
        return 'house_head_primary_ed'
    elif ed_string in secondary_ed:
        return 'house_head_secondary_ed'
    else:
        return 'house_head_tertiary_ed'

phil_inc_extract['Household Head Highest Grade Completed'] = [get_education_level(e)                                                              for e                                                              in phil_inc_extract['Household Head Highest Grade Completed']]

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Household Head Highest Grade Completed']))

phil_inc_extract.drop(['Household Head Highest Grade Completed'],                      axis = 1,                      inplace = True)


# In[ ]:


phil_inc_extract['Household Head Job or Business Indicator'].unique().tolist()


# In[ ]:


phil_inc_extract.rename(columns = {'Household Head Job or Business Indicator' : 'house_head_empl'}, inplace = True)
phil_inc_extract.house_head_empl = phil_inc_extract.house_head_empl.map({'With Job/Business' : 1,                                                                         'No Job/Business' : 0})


# In[ ]:


phil_inc_extract['Household Head Occupation'].fillna('', inplace = True)


# In[ ]:


unque_occupations = phil_inc_extract['Household Head Occupation'].unique().tolist()
print(unque_occupations[:5])


# In[ ]:


import re
farmer_flag = re.compile(r'farm', re.I)

for w in unque_occupations:
    if re.findall(farmer_flag, w):
        print(w)


# In[ ]:


phil_inc_extract['house_head_farmer'] = [1 if re.findall(farmer_flag, w) else 0 for w in phil_inc_extract['Household Head Occupation']]
phil_inc_extract.head()


# In[ ]:


phil_inc_extract.drop(['Household Head Occupation', 'Household Head Class of Worker'], axis = 1, inplace = True)


# In[ ]:


phil_inc_extract['Type of Household'].unique()


# In[ ]:


phil_inc_extract['Type of Household'] = phil_inc_extract['Type of Household'].map({'Extended Family' : 'house_ext_family',
      'Single Family' : 'house_singl_family',
      'Two or More Nonrelated Persons/Members' : 'house_mult_family'})

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Type of Household']))

phil_inc_extract.drop(['Type of Household'], axis = 1, inplace = True)


# In[ ]:


phil_inc_extract.rename(columns = {'Total Number of Family members' : 'num_family_members',
                                   'Members with age less than 5 year old' : 'num_children_younger_5',
                                   'Members with age 5 - 17 years old' : 'num_children_older_5',
                                   'Total number of family members employed' : 'num_family_members_employed'},
                        inplace = True)


# In[ ]:


phil_inc_extract['Type of Building/House'] = phil_inc_extract['Type of Building/House'].map({'Single house' : 'house',
      'Duplex' : 'duplex',
      'Commercial/industrial/agricultural building' : 'living_at_workplace',
      'Multi-unit residential' : 'residential_block',
      'Institutional living quarter' : 'institutional_housing',
      'Other building unit (e.g. cave, boat)' : 'other_housing'})

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Type of Building/House']))
phil_inc_extract.drop(['Type of Building/House'], axis = 1, inplace = True)


# In[ ]:


phil_inc_extract['Type of Roof'] = phil_inc_extract['Type of Roof'].map({'Strong material(galvanized,iron,al,tile,concrete,brick,stone,asbestos)' : 'roof_material_strong',
      'Light material (cogon,nipa,anahaw)' : 'roof_material_light',
      'Mixed but predominantly strong materials' : 'roof_material_mostly_strong',
      'Mixed but predominantly light materials' : 'roof_material_mostly_light',
      'Salvaged/makeshift materials' : 'roof_material_makeshift',
      'Mixed but predominantly salvaged materials' : 'roof_material_mostly_makeshift',
      'Not Applicable' : 'no_roof'})

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Type of Roof']))
phil_inc_extract.drop(['Type of Roof'], axis = 1, inplace = True)


# In[ ]:


phil_inc_extract['Type of Walls'] = phil_inc_extract['Type of Walls'].map({'Strong' : 'wall_material_strong',
      'Light' : 'wall_material_light',
      'Quite Strong' : 'wall_material_quite_strong',
      'Very Light' : 'wall_material_quite_light',
      'Salvaged' : 'wall_material_salvaged',
      'NOt applicable' : 'no_walls'})

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Type of Walls']))
phil_inc_extract.drop(['Type of Walls'], axis = 1, inplace = True)


# In[ ]:


phil_inc_extract.rename(columns = {'House Floor Area' : 'house_area',
                                   'House Age' : 'house_age',
                                   'Number of bedrooms' : 'num_bedrooms'},
                        inplace = True)


# In[ ]:


phil_inc_extract['Toilet Facilities'] = phil_inc_extract['Toilet Facilities'].map({'Water-sealed, sewer septic tank, used exclusively by household' : 'ws_septic_toiled',
      'Water-sealed, sewer septic tank, shared with other household' : 'ws_septic_toiled',
      'Closed pit' : 'septic_toiled',
      'Water-sealed, other depository, used exclusively by household' : 'ws_other_toilet',
      'Open pit' : 'septic_toiled',
      'Water-sealed, other depository, shared with other household' : 'ws_other_toilet',
      'None' : 'no_toilet',
      'Others' : 'other_toilet'})

phil_inc_extract = phil_inc_extract.join(pd.get_dummies(phil_inc_extract['Toilet Facilities']))
phil_inc_extract.drop(['Toilet Facilities', 'Tenure Status'], axis = 1, inplace = True)


# In[ ]:


running_water = ['Own use, faucet, community water system', 'Shared, faucet, community water system']

phil_inc_extract['running_water'] = [1 if i in running_water else 0 for i in phil_inc_extract['Main Source of Water Supply']]

phil_inc_extract.drop(['Main Source of Water Supply'], axis = 1, inplace = True)


# In[ ]:


phil_inc_extract['num_electronics'] = phil_inc_extract['Number of Television'] +phil_inc_extract['Number of CD/VCD/DVD'] +phil_inc_extract['Number of Component/Stereo set'] +phil_inc_extract['Number of Personal Computer']

phil_inc_extract.drop(['Number of Television',                       'Number of CD/VCD/DVD',                       'Number of Component/Stereo set',                       'Number of Personal Computer'],                      axis = 1,                      inplace = True)


# In[ ]:


phil_inc_extract['num_comm_devices'] = phil_inc_extract['Number of Landline/wireless telephones'] +phil_inc_extract['Number of Cellular phone']

phil_inc_extract.drop(['Number of Landline/wireless telephones',                       'Number of Cellular phone'],                      axis = 1,                      inplace = True)


# In[ ]:


phil_inc_extract['num_vehicles'] = phil_inc_extract['Number of Car, Jeep, Van'] +phil_inc_extract['Number of Motorized Banca'] +phil_inc_extract['Number of Motorcycle/Tricycle']

phil_inc_extract.drop(['Number of Car, Jeep, Van',                       'Number of Motorized Banca',                       'Number of Motorcycle/Tricycle'],                      axis = 1,                      inplace = True)


# In[ ]:


phil_inc_extract.rename(columns = {'Total Household Income' : 'household_income',
                                   'Region' : 'region',
                                   'Total Food Expenditure' : 'food_expenses',
                                   'Agricultural Household indicator' : 'agricultural_household',
                                   'Clothing, Footwear and Other Wear Expenditure' : 'clothing_expenses',
                                   'Housing and water Expenditure' : 'house_and_water_expenses',
                                   'Imputed House Rental Value' : 'house_rental_value',
                                   'Medical Care Expenditure' : 'medical_expenses',
                                   'Transportation Expenditure' : 'transport_expenses',
                                   'Communication Expenditure' : 'comm_expenses',
                                   'Education Expenditure' : 'education_expenses',
                                   'Miscellaneous Goods and Services Expenditure' : 'misc_expenses',
                                   'Special Occasions Expenditure' : 'special_occasion_expenses',
                                   'Crop Farming and Gardening expenses' : 'farming_gardening_expenses',
                                   'Total Income from Entrepreneurial Acitivites' : 'income_from_entrepreneur_activities',
                                   'Household Head Age' : 'house_head_age',
                                   'Electricity' : 'electricity',
                                   'Number of Refrigerator/Freezer' : 'num_refrigerator',
                                   'Number of Washing Machine' : 'num_wash_machine',
                                   'Number of Airconditioner' : 'num_ac',
                                   'Number of Stove with Oven/Gas Range' : 'num_stove'},
                        inplace = True)


# In[ ]:


phil_inc_extract.columns


# In[ ]:


phil_inc_extract.to_csv('philippines_census_data_cleaned.csv')


# In[ ]:


phil_inc_extract_grouped = phil_inc_extract.groupby(by = ['region', 'house_head_sex_f']).mean().reset_index()
phil_inc_extract_grouped.head()


# In[ ]:


phil_loans[['id', 'region']].groupby(by = 'region').count().reset_index()


# In[ ]:


phil_loans_extract = phil_loans[(phil_loans.borrower_genders.notna()) & (phil_loans.region != 'no_match')]


# In[ ]:


phil_loans_extract.borrower_genders.unique()


# In[ ]:


phil_loans_extract['borrower_genders'] = phil_loans_extract['borrower_genders'].map({'female' : 1,      'male' : 0})

phil_loans_extract.rename(columns = {'borrower_genders' : 'house_head_sex_f'}, inplace = True)


# In[ ]:


phil_loans_extract.house_head_sex_f.unique()


# In[ ]:


phil_loans_extract.to_csv('kiva_loans_ph_transofrmed.csv')


# In[ ]:


merged_loans = pd.merge(left = phil_loans_extract, right = phil_inc_extract_grouped, how = 'left', on = ['region', 'house_head_sex_f'])
merged_loans.head()


# In[ ]:


loan_columns_to_drop = ['country_code', 'country', 'posted_time', 'disbursed_time', 'funded_time', 'date', 'location', 'use', 'tags', 'activity', 'funded_amount', 'lender_count']
merged_loans.drop(loan_columns_to_drop, axis = 1, inplace = True)


# In[ ]:


merged_loans.head()


# In[ ]:


partner_id_mapping = {}

for p_id in merged_loans.partner_id.unique():
    partner_id_mapping[p_id] = 'partner_' + str(int(p_id))

merged_loans.partner_id = merged_loans.partner_id.map(partner_id_mapping)


# In[ ]:


for c in ['sector', 'currency', 'partner_id', 'repayment_interval', 'region']:
    merged_loans = merged_loans.join(pd.get_dummies(merged_loans[c]))
    merged_loans.drop([c], axis = 1, inplace = True)


# In[ ]:


merged_loans.head()


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(merged_loans.drop(['id', 'loan_amount', 'term_in_months'], axis = 1),                                                    merged_loans['loan_amount'],                                                    test_size = 0.3,                                                    random_state = 42)

X_train.head()


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor

skl_regr = AdaBoostRegressor(n_estimators = 150, random_state = 42)
skl_regr.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error

score_ada = mean_squared_error(y_test, skl_regr.predict(X_test))
print(score_ada)


# In[ ]:




