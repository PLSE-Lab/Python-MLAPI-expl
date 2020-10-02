#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# # Exploratory Data Analysis

# In[ ]:


# do this to make Pandas show all the columns of a DataFrame, otherwise it just shows a summary
pd.set_option('display.max_columns', None) 


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

train_id = df_train['Id']
test_id = df_test['Id']

train_idhogar = df_train['idhogar']
test_idhogar = df_train['idhogar']

df_train.drop(columns=['Id'], inplace=True)
df_test.drop(columns=['Id'], inplace=True)

print("Shape of train data: ", df_train.shape)
print("Shape of test data: ", df_test.shape)

ntrain = df_train.shape[0]
ntest = df_test.shape[0]

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)


# In[ ]:


print("A glimpse at the columns of training data:")
df_train.head()


# These are the core data fields as described in the [data description](https://www.kaggle.com/c/costa-rican-household-poverty-prediction/data):
# 
# * Id - a unique identifier for each row.
# * Target - the target is an ordinal variable indicating groups of income levels. 
#     1 = extreme poverty 
#     2 = moderate poverty 
#     3 = vulnerable households 
#     4 = non vulnerable households
# * idhogar - this is a unique identifier for each household. This can be used to create household-wide features, etc. All rows in a given household will have a matching value for this identifier.
# * parentesco1 - indicates if this person is the head of the household.
# 

# In[ ]:


print("The feature that we need to predict: ", set(df_train.columns) - set(df_test.columns))


# Let's see a description of `Target`: 

# In[ ]:


df_train['Target'].describe()


# We need to make predictions on a household level whereas we have been given the data at an individual level. 
# 
# That is why during all our analysis we will focus only on those columns which have `parentesco1 == 1`. These are columns for the heads of households and each household has only one head. 

# In[ ]:


def barplot_with_anotate(feature_list, y_values, plotting_space=plt, annotate_vals=None):
    x_pos = np.arange(len(feature_list))
    plotting_space.bar(x_pos, y_values);
    plotting_space.xticks(x_pos, feature_list, rotation=270);
    if annotate_vals == None:
        annotate_vals = y_values
    for i in range(len(feature_list)):
        plotting_space.text(x=x_pos[i]-0.3, y=y_values[i]+1.0, s=annotate_vals[i]);


# In[ ]:


df_train_heads = df_train.loc[df_train['parentesco1'] == 1]
poverty_label_sizes = list(df_train_heads.groupby('Target').size())

barplot_with_anotate(['extreme', 'moderate', 'vulnerable', 'non-vulnerable'], poverty_label_sizes,
                     annotate_vals = [str(round((count/df_train_heads.shape[0])*100, 2))+'%' 
                                      for count in poverty_label_sizes]);
plt.rcParams["figure.figsize"] = [6, 6];
plt.xlabel('Poverty Label');
plt.ylabel('No. of people');


# So, we can see that **_a majority (>65%) of the households fall within the `Non-vulnerable` category_**. This means that we are dealing with an imbalanced classification problem.
# 
# Now, let's try to understand what it means to live under such conditions.

# ## Home Life for various poverty groups:

# In[ ]:


def plot_dwelling_property(property_df):
    _, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(16, 16))

    target_idx = 0
    for row in range(2):
        for col in range(2):
            percentage_list = [round((count/poverty_label_sizes[target_idx])*100, 2)
                                 for count in list(property_df.iloc[target_idx, :])]
            x_pos = list(range(len(property_df.columns)))
            
            axarr[row, col].bar(x_pos, 
                                percentage_list, 
                                color='y')
            
            axarr[row, col].set_title('For individuals in Poverty group=' + str(target_idx+1))
            
            xtick_labels = list(property_df.columns)
            xtick_labels.insert(0, '') # insert a blank coz `set_xticklabels()` skips the 1st element ##why??
            axarr[row, col].set_xticklabels(xtick_labels, rotation=300)
            
            axarr[row, col].set_ylim(bottom=0, top=100)
            #axarr[row, col].set_xlim(left=0, right=len(property_df.columns))
            
            for i in range(len(property_df.columns)):
                axarr[row, col].annotate(xy=(x_pos[i]-0.3, percentage_list[i]+1.0), s=percentage_list[i]);
            
            axarr[0, 0].set_ylabel("Percentage of the total in this poverty group");
            axarr[1, 0].set_ylabel("Percentage of the total in this poverty group");
            axarr[1, 0].set_xlabel("Types");
            axarr[1, 1].set_xlabel("Types");

            axarr[row, col].autoscale(enable=True, axis='x')
            target_idx+=1


# ### Outside wall material of the house:

# In[ ]:


outside_wall_material_df = df_train_heads.groupby('Target').sum()[['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 
                                  'paredzinc', 'paredfibras', 'paredother']]
outside_wall_material_df


# ```
# paredblolad, =1 if predominant material on the outside wall is block or brick
# paredzocalo, "=1 if predominant material on the outside wall is socket (wood,  zinc or absbesto"
# paredpreb, =1 if predominant material on the outside wall is prefabricated or cement
# pareddes, =1 if predominant material on the outside wall is waste material
# paredmad, =1 if predominant material on the outside wall is wood
# paredzinc, =1 if predominant material on the outside wall is zink
# paredfibras, =1 if predominant material on the outside wall is natural fibers
# paredother, =1 if predominant material on the outside wall is other
# ```

# In[ ]:


plot_dwelling_property(outside_wall_material_df)


# * We see that a majority (69.24%) of the households under poverty group 4 (non-vulnerable) have brick wall on the outside. 
# * As we go from there to group 1 (extreme), the percentage of houses having brick wall decreases. Cement wall and wood walls become increasingly more common.
# * The top 3 most common types of wall material across all the groups are (in descending order of popularity) - 
#   `brick` > `prefabricated or cement` > `wood`

# ### Floor material of the house:

# In[ ]:


floor_material_df = df_train_heads.groupby('Target').sum()[['pisomoscer', 'pisocemento', 'pisoother',
                                                      'pisonatur', 'pisonotiene', 'pisomadera']]
floor_material_df


# ```
# pisomoscer, "=1 if predominant material on the floor is mosaic,  ceramic,  terrazo"
# pisocemento, =1 if predominant material on the floor is cement
# pisoother, =1 if predominant material on the floor is other
# pisonatur, =1 if predominant material on the floor is  natural material
# pisonotiene, =1 if no floor at the household
# pisomadera, =1 if predominant material on the floor is wood
# ```

# In[ ]:


plot_dwelling_property(floor_material_df)


# * We see that a majority of households belonging to poverty group 3 (vulnerable) and group 4 (non-vulnerable) have `mossaic, ceramic, tazzo` floors (62.82% and 79.38% respectively)
# * This floor type becomes less common as we move across from group 4 to group 1 and other types (especially the cemented floors) become more common.
# * The top 3 most common types of floors across all the groups are (in descending order of popularity) -   
#   `mossaic, ceramic, tazzo` > `cemented` > `wooden`

# ### Toilet:

# In[ ]:


toilet_df = df_train_heads.groupby('Target').sum()[['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5',
                                              'sanitario6']]
toilet_df


# ```
# sanitario1, =1 no toilet in the dwelling
# sanitario2, =1 toilet connected to sewer or cesspool
# sanitario3, =1 toilet connected to  septic tank
# sanitario5, =1 toilet connected to black hole or letrine
# sanitario6, =1 toilet connected to other system
# ```

# In[ ]:


plot_dwelling_property(toilet_df)


# * A large majority of the households have a toilet connected to a septic tank (73% - 81%).
# * A toilet connected to sewer or cess pool becomes more common as we move to group 4. It is probably a better, more expensive type of installation. 

# ### Rubbish disposal:

# In[ ]:


rubbish_disposal_df = df_train_heads.groupby('Target').sum()[['elimbasu1', 'elimbasu2', 'elimbasu3',
                                                        'elimbasu4', 'elimbasu5', 'elimbasu6']]
rubbish_disposal_df


# ```
# elimbasu1, =1 if rubbish disposal mainly by tanker truck
# elimbasu2, =1 if rubbish disposal mainly by botan hollow or buried
# elimbasu3, =1 if rubbish disposal mainly by burning
# elimbasu4, =1 if rubbish disposal mainly by throwing in an unoccupied space
# elimbasu5, "=1 if rubbish disposal mainly by throwing in river,  creek or sea"
# elimbasu6, =1 if rubbish disposal mainly other
# ```

# In[ ]:


plot_dwelling_property(rubbish_disposal_df)


# * A large majority of the households in all the poverty groups dispose their rubbish using tanker trucks.
# * Rubbish disposal by burning is the 2nd most popular way and its popularity increases as we move from group 4 (non-vulnerable) to group 1 (extreme). This may be as a result of lack of environmental awareness or a lack of resources in the less fortunate hopes.

# ### Roof material of the house:

# In[ ]:


roof_material_df = df_train_heads.groupby('Target').sum()[['techozinc', 'techoentrepiso', 'techocane', 'techootro']]
roof_material_df


# ```
# techozinc, =1 if predominant material on the roof is metal foil or zink
# techoentrepiso, "=1 if predominant material on the roof is fiber cement,  mezzanine "
# techocane, =1 if predominant material on the roof is natural fibers
# techootro, =1 if predominant material on the roof is other
# ```

# In[ ]:


plot_dwelling_property(roof_material_df)


# * This distrubution is pretty much the same throughout all the groups. A huge majority (> 95%) of the individuals live in homes with metal foil or zinc roof. 
# 
# **_We may conclude that these features are not representative of the poverty levels._**

# ### Water provision:

# In[ ]:


water_provision_df = df_train_heads.groupby('Target').sum()[['abastaguadentro', 'abastaguafuera', 'abastaguano']]
water_provision_df


# ```
# abastaguadentro, =1 if water provision inside the dwelling
# abastaguafuera, =1 if water provision outside the dwelling
# abastaguano, =1 if no water provision
# ```

# In[ ]:


plot_dwelling_property(water_provision_df)


# * Again, this distribution is also pretty much the same across all the groups. Almost all (~95%) of the people in all the groups enjoy water provision inside their dwellings.
# 
# **_We may conclude that these features are not representative of the poverty levels._**

# ### Electricity:

# In[ ]:


electricity_df = df_train_heads.groupby('Target').sum()[['public', 'planpri', 'noelec', 'coopele']]
electricity_df


# ```
# public, "=1 electricity from CNFL,  ICE,  ESPH/JASEC"
# planpri, =1 electricity from private plant
# noelec, =1 no electricity in the dwelling
# coopele, =1 electricity from cooperative
# ```

# In[ ]:


plot_dwelling_property(electricity_df)


# * Again, this distribution is also pretty much the same across all the poverty groups. ~88% of the people in all the groups get electricity from `CNFL,  ICE,  ESPH/JASEC` and ~11% get it from cooperative.
# 
# **_We may conclude that these features are not representative of the poverty levels._**

# ### Main source of energy in cooking:

# In[ ]:


cooking_energy_df = df_train_heads.groupby('Target').sum()[['energcocinar1', 'energcocinar2', 'energcocinar3',
                                                      'energcocinar4']]
cooking_energy_df


# ```
# energcocinar1, =1 no main source of energy used for cooking (no kitchen)
# energcocinar2, =1 main source of energy used for cooking electricity
# energcocinar3, =1 main source of energy used for cooking gas
# energcocinar4, =1 main source of energy used for cooking wood charcoal
# ```

# In[ ]:


plot_dwelling_property(cooking_energy_df)


# * Here, we see that gas and electricity are the major sources of energy in the kitchens for all the people.
# * For the poverty group 4 (non-vulnerable), electricity is slightly more popular than gas whereas in the other groups, gas is the more popular choice.
# * As we move from group 1 (extreme) to group 4 (non-vulnerable), the popularity of electricity increases and that of gas decreases. 

# ### Household size:

# In[ ]:


avg_household_size_df = df_train_heads.groupby('Target').mean()['hhsize']
avg_household_size_df


# In[ ]:


df_train.groupby('Target').mean().head()


# ### Urban or rural:

# Let's try to understand the demographics of the urban and the rural population

# In[ ]:


urban_rural_df = df_train_heads.groupby('Target').sum()[['area1', 'area2']]
urban_rural_df['UrbanPercentage'] = urban_rural_df['area1'] * round((100/sum(urban_rural_df['area1'])), 6)
urban_rural_df['RuralPercentage'] = urban_rural_df['area2'] * round((100/sum(urban_rural_df['area2'])), 6)
urban_rural_df


# ```
# area1, =1 zona urbana
# area2, =1 zona rural
# ```

# The following pattern is seen in both urban and rural areas:
# 
# * __~ 58-68%__ of the houses are in Target 4 (__non-vulnerable__)
# * __~ 10-15%__ of the houses are in Target 2 (__moderate__)
# * __~ 13-18%__ of the houses are in Target 3 (__vulnerable__)
# * __~ 6-10%__ of the houses are in Target 1 (__extreme__)

# ### Region:

# In[ ]:


region_df = df_train_heads.groupby('Target').sum()[['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']]
region_df


# ```
# lugar1, =1 region Central
# lugar2, =1 region Chorotega
# lugar3, =1 region PacÃ­fico central
# lugar4, =1 region Brunca
# lugar5, =1 region Huetar AtlÃ¡ntica
# lugar6, =1 region Huetar Norte
# ```

# In[ ]:


plot_dwelling_property(region_df)


# In[ ]:


region_df.T


# * `Central` region has the maximum population
# * A major portion (65%) of the poverty group 4 (non-vulnerable) people live in `Central` region

# ### Monthly rent - 

# In[ ]:


round(((all_data.shape[0] - sum(all_data['v2a1'].value_counts())) / all_data.shape[0] ) * 100, 2)


# A large percentage of `v2a1` (the monthly rent column) is empty. We will analyse this after we impute the missing values in the next section.

# ## Education -

# ### escolari:

# In[ ]:


sns.boxplot(x='Target', y='escolari', data=all_data.loc[:ntrain]);


# ### Conclusion:-
# 
# These features do not convey any useful information about the `Target` variable:
# * 'sanitario1', 'sanitario6'
# * 'elimbasu4', 'elimbasu5', 'elimbasu6'
# * 'techozinc', 'techoentrepiso', 'techocane', 'techootro'
# * 'abastaguadentro', 'abastaguafuera', 'abastaguano'
# * 'public', 'planpri', 'noelec', 'coopele'
# 
# Removing them increased my F1-score by 1.5%.

# In[ ]:


all_data.drop(columns=['sanitario1', 'sanitario6',
                       'elimbasu4', 'elimbasu5', 'elimbasu6',
                       'techozinc', 'techoentrepiso', 'techocane', 'techootro',
                       'abastaguadentro', 'abastaguafuera', 'abastaguano',
                       'public', 'planpri', 'noelec', 'coopele'], inplace=True)


# Okay, so education affects the poverty label to some extent. Or, maybe poverty label affects one's ability to get education.

# ## Numerical or Categorical?

# In[ ]:


num_features = all_data._get_numeric_data().columns
num_features_length = len(num_features)

categ_features = pd.Index(list(set(all_data.columns) - set(num_features)))
categ_features_length = len(categ_features)

print("Number of numerical features: ", num_features_length)
print("Number of categorical features: ", categ_features_length)

labels = ['numeric', 'categorical']
colors = ['y', 'r']
plt.figure(figsize=(8, 8))
plt.pie([num_features_length, categ_features_length], 
        labels=labels, 
        autopct='%1.1f%%', 
        shadow=True, 
        colors=colors);


# Let's have a look at the categorical features:

# In[ ]:


all_data[categ_features].head()


# Its no surprise that `idhogar` is categorical but according to the [data description provided with the challenge](https://www.kaggle.com/c/costa-rican-household-poverty-prediction/data), the other 3 features should take numerical values. Instead they contain lots of 'yes' and 'no' values as well. 

# In[ ]:


_, axarr = plt.subplots(nrows=1, ncols=3, sharey='row', figsize=(12, 6))

for idx, feature in enumerate(['dependency', 'edjefe', 'edjefa']):
    sns.countplot(x=feature, data=all_data[all_data[feature].isin(['yes', 'no'])], ax=axarr[idx])


# A look at [this discussion](https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359554) showed that there is a glitch with `dependency`, `edjefe` and `edjefa`. In all of these cases,, 'yes' implies 1 and 'no' implies 0. So, let's fix that..

# In[ ]:


yes_no_map = {'no': 0, 'yes': 1}
    
all_data['dependency'] = all_data['dependency'].replace(yes_no_map).astype(np.float32)
all_data['edjefe'] = all_data['edjefe'].replace(yes_no_map).astype(np.float32)
all_data['edjefa'] = all_data['edjefa'].replace(yes_no_map).astype(np.float32)


# Now, all the features are numeric.

# ### Numerical features that are binary:

# In[ ]:


num_binary_features = []

for feature in all_data.columns:
    if sorted(df_train[feature].unique()) in [[0, 1], [0], [1]]:
        num_binary_features.append(feature)
        
print("Total number of binary-numerical features: ", len(num_binary_features))
print("Binary-numerical features: ")
num_binary_features


# ### Non-binary features:

# In[ ]:


num_non_binary_features = [feature for feature in all_data.columns if feature not in num_binary_features]

print("Total number of non-binary-numerical features: ", len(num_non_binary_features))
print("Non-binary numerical features: ")

num_non_binary_features_dict = {feature: len(all_data[feature].unique()) for feature in num_non_binary_features}

num_non_binary_features_sorted = sorted(num_non_binary_features_dict, 
                                        key=lambda feature: num_non_binary_features_dict[feature], 
                                        reverse=True)

num_non_binary_features_len_sorted = [num_non_binary_features_dict[feature] for feature in num_non_binary_features_sorted]

plt.figure(figsize=(16, 16))
barplot_with_anotate(num_non_binary_features_sorted, num_non_binary_features_len_sorted);
plt.ylabel("No. of unique values");
plt.xlabel("Non-binary numerical features");


# Out of these 39 features, the following are continuous in nature:
# * v2al
# * meaneduc
# * SQBmeaned
# * dependency
# * SQBdependency
# 
# 
# All the other features are discrete in nature.

# ## Summary

# ### Binary features:

# In[ ]:


all_data[num_binary_features].describe()


# ### Non-binary continuous features:

# In[ ]:


num_conti_features = pd.Index(['v2a1', 'meaneduc', 'dependency', 'SQBmeaned', 'SQBdependency'])
all_data[num_conti_features].describe()


# ### Non-binary discrete features:

# In[ ]:


num_discrete_features = pd.Index([feature for feature in num_non_binary_features if feature not in num_conti_features])
all_data[num_discrete_features].describe()


# # Preprocessing

# ## 1. Missing values imputation:

# In[ ]:


def missing_features(data, column_set):
    incomplete_features = {feature: data.shape[0]-sum(data[feature].value_counts())
                                   for feature in column_set
                                   if not sum(data[feature].value_counts()) == data.shape[0]}
    incomplete_features_sorted = sorted(incomplete_features, key=lambda feature: incomplete_features[feature], reverse=True)
    incompleteness = [round((incomplete_features[feature]/data.shape[0])*100, 2) for feature in incomplete_features_sorted]
    plt.figure(figsize=(12, 6))
    barplot_with_anotate(incomplete_features_sorted, incompleteness)
    plt.ylabel("Percentage (%) of values that are missing")
    #plt.rcParams["figure.figsize"] = [12, 6]
    
    for feature, percentage in zip(incomplete_features_sorted, incompleteness):
        print("Feature:", feature)
        print("No. of NaNs:", incomplete_features[feature], "(", percentage, ")")


# In[ ]:


missing_features(all_data, all_data.columns)


# * [This discussion](https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#360609) shows how missing values of `v2a1` and `v18q1` should be handled.
# 
# * `rez_esc` (Years behind in school): NaN implies that the person does not remember. Considering that along with the large percentage of NaN values, we are better off dropping that column.
# 
# 
# * `meaneduc` and `SQBmeaned`: With the average of the columns.

# #### `v2a1` :-

# In[ ]:


# entries which have both v2a1 as NaN and tipovivi3 as 0
all_data[['v2a1', 'tipovivi3']][all_data['tipovivi3'] == 0][all_data['v2a1'].isnull()].shape


# We see that all those entries where `v2a1` is Nan also have `tipovivi3` as 0, which implies that all those houses are not rented. 
# 
# Hence, we should fill the missing values of `v2a1` with 0.

# In[ ]:


# handling v2a1
all_data.loc[:, 'v2a1'].fillna(0, inplace=True)


# #### `v18q1` :-

# In[ ]:


# entries which have v18q as 0 and v18q1 as NaN
all_data[['v18q1', 'v18q']][all_data['v18q'] == 0][all_data['v18q1'].isnull()].shape


# We see that `v18q1` is `NaN` only for those entries which have `v18q` == 0. Thus, `v18q1` is missing only when the house does not have a tablet. 
# 
# Hence, we should fill the missing values of `v18q1` with 0.

# In[ ]:


# handling v18q1
all_data.loc[:, 'v18q1'].fillna(0, inplace=True)


# #### `meaneduc` and `SQBmeaned` :-

# In[ ]:


# handling meaneduc and SQBmeaned
all_data.loc[:, 'meaneduc'].fillna(all_data['meaneduc'].mean(), inplace=True)
all_data.loc[:, 'SQBmeaned'].fillna(all_data['SQBmeaned'].mean(), inplace=True)


# #### `rez_esc` :-

# Drop it.

# In[ ]:


all_data.drop(columns=['rez_esc'], inplace=True)


# ## 2. Convert dummy to ordinal:
# 
# These features have order in their meaning:
# ```
# epared1, =1 if walls are bad
# epared2, =1 if walls are regular
# epared3, =1 if walls are good
# etecho1, =1 if roof are bad
# etecho2, =1 if roof are regular
# etecho3, =1 if roof are good
# eviv1, =1 if floor are bad
# eviv2, =1 if floor are regular
# eviv3, =1 if floor are good
# instlevel1, =1 no level of education
# instlevel2, =1 incomplete primary
# instlevel3, =1 complete primary
# instlevel4, =1 incomplete academic secondary level
# instlevel5, =1 complete academic secondary level
# instlevel6, =1 incomplete technical secondary level
# instlevel7, =1 complete technical secondary level
# instlevel8, =1 undergraduate and higher education
# instlevel9, =1 postgraduate higher education
# ```
# We should use them as ordinal features.

# In[ ]:


all_data['WallQual'] = all_data['epared1'] + 2*all_data['epared2'] + 3*all_data['epared3']

all_data['RoofQual'] = all_data['etecho1'] + 2*all_data['etecho2'] + 3*all_data['etecho3']

all_data['FloorQual'] = all_data['eviv1'] + 2*all_data['eviv2'] + 3*all_data['eviv3']

all_data['EducationLevel'] = all_data['instlevel1'] + 2*all_data['instlevel2'] + 3*all_data['instlevel3'] +     4*all_data['instlevel4'] + 5*all_data['instlevel5'] + 6*all_data['instlevel6'] + 7*all_data['instlevel7'] +     8*all_data['instlevel8'] + 9*all_data['instlevel9']


# In[ ]:


all_data.drop(columns=['epared1', 'epared2', 'epared3',
                       'etecho1', 'etecho2', 'etecho3',
                       'eviv1', 'eviv2', 'eviv3',
                       'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
                       'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9'], inplace=True)


# ## 3. Remove redundant features:
# 
# I have used [this kernel](https://www.kaggle.com/kuriyaman1002/reduce-features-140-84-keeping-f1-score) to identify some of these features -
# 
# * The following can be generated from linear combination of r4h* and r4m*:
#     ```
#     r4t1, persons younger than 12 years of age
#     r4t2, persons 12 years of age and older
#     r4t3, Total persons in the household
#     ```
# 
# * The following mean the same as `hogar_total`:
#     ```
#     tamhog, size of the household
#     tamviv, number of persons living in the household
#     hhsize, household size
#     r4t3, Total persons in the household
#     ```
# 
# 
# * `v18q` can be generated by v18q1
# * `mobilephone` can be generated by qmobilephone

# In[ ]:


redundant_features = ['r4t1', 'r4t2', 'r4t3', 'tamhog', 'tamviv', 'hhsize', 'r4t3', 'v18q', 'mobilephone']
all_data.drop(columns=redundant_features, inplace=True)


# ## 4. Create new household-wide features:

# 1. Hand-engineered features:
# 
#     * Monthly rent per room - `v2a1/rooms`
#     * Monthly rent per adult - `v2a1/hogar_adul`
#     * No. of adults per room - `hogar_adul/rooms`
#     * No. of adults per bedroom - `hogar_adul/bedrooms`
#     
# 2. Average of individual-level features per household
# 
# 3. Minimum of individual-level features per household
# 
# 4. Maximum of individual-level features per household
# 
# 5. Sum of individual-level features per household
# 
# 6. Standard deviation of individual-level features per household
# 
# 
# I have taken help from [this excellent analysis](https://www.kaggle.com/willkoehrsen/start-here-a-complete-walkthrough) done by Will Koehrsen.

# In[ ]:


all_data['RentPerRoom'] = all_data['v2a1'] / all_data['rooms']

all_data['AdultsPerRoom'] = all_data['hogar_adul'] / all_data['rooms']

all_data['AdultsPerBedroom'] = all_data['hogar_adul'] / all_data['bedrooms']


# In[ ]:


# individual level boolean features
ind_bool = ['dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'EducationLevel']

# individual level ordered features
ind_ordered = ['escolari', 'age']


# In[ ]:


f = lambda x: x.std(ddof=0)
f.__name__ = 'std_0'
ind_agg = all_data.groupby('idhogar')[ind_ordered + ind_bool].agg(['mean', 'max', 'min', 'sum', f])

new_cols = []
for col in ind_agg.columns.levels[0]:
    for stat in ind_agg.columns.levels[1]:
        new_cols.append(f'{col}-{stat}')

ind_agg.columns = new_cols
ind_agg.head()


# In[ ]:


print("Original number of features:", all_data.shape[1])

all_data = all_data.merge(ind_agg, on = 'idhogar', how = 'left')

print("Number of features after merging transformed individual level features", all_data.shape[1])

all_data.drop(columns=ind_bool+ind_ordered, inplace=True)

print("Number of features after dropping the individual level features", all_data.shape[1])


# # Modelling

# In[ ]:


from sklearn.metrics import f1_score, make_scorer
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


# In[ ]:


# drop the idhogar column
all_data.drop(columns=['idhogar'], inplace=True)


# In[ ]:


df_train = all_data[:ntrain][:]
df_test = all_data[ntrain:][:]
df_test = df_test.drop('Target', axis=1)


# In[ ]:


print(df_train.shape)
print(df_test.shape)


# In[ ]:


X_train= df_train.drop('Target', axis= 1)
Y_train= df_train['Target']

X_test= df_test


# In[ ]:


validation_scores = {}


# In[ ]:


scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')


# I am going to try 2 gradient boosting machines - LightGBM and XGBoost.

# ### LightGBM:

# In[ ]:


skf = StratifiedKFold(n_splits=5)


# In[ ]:


lightgbm = lgb.LGBMClassifier(class_weight='balanced', boosting_type='dart',
                         drop_rate=0.9, min_data_in_leaf=100, 
                         max_bin=255,
                         n_estimators=500,
                         bagging_fraction=0.01,
                         min_sum_hessian_in_leaf=1,
                         importance_type='gain',
                         learning_rate=0.1, 
                         max_depth=-1, 
                         num_leaves=31)

#validation_scores['LightGBM'] = cross_val_score(lightgbm, X_train, Y_train, cv=3, scoring=scorer).mean()
#print(validation_scores['LightGBM'])


# In[ ]:


predicts_lgb = []
for train_index, test_index in skf.split(X_train, Y_train):
    X_t, X_v = X_train.iloc[train_index], X_train.iloc[test_index]
    y_t, y_v = Y_train.iloc[train_index], Y_train.iloc[test_index]
    
    lightgbm.fit(X_t, y_t, eval_set=[(X_v, y_v)], early_stopping_rounds=50)
    predicts_lgb.append(lightgbm.predict(X_test))


# In[ ]:


lightgbm_pred = np.array(predicts_lgb).mean(axis=0).round().astype(int)

submission_lgb = pd.DataFrame({'Id': test_id,
                           'Target': lightgbm_pred})
submission_lgb.to_csv('submissionLGB.csv', index=False)


# ### XGBoost:

# In[ ]:


xgboost = xgb.XGBClassifier()

#validation_scores['XGBoost'] = cross_val_score(xgboost, X_train, Y_train, cv=3, scoring=scorer).mean()
#print(validation_scores['XGBoost']);


# In[ ]:


predicts_xgb = []
for train_index, test_index in skf.split(X_train, Y_train):
    X_t, X_v = X_train.iloc[train_index], X_train.iloc[test_index]
    y_t, y_v = Y_train.iloc[train_index], Y_train.iloc[test_index]
    
    xgboost.fit(X_t, y_t, eval_set=[(X_v, y_v)], early_stopping_rounds=50)
    predicts_xgb.append(xgboost.predict(X_test))


# In[ ]:


xgboost_pred = np.array(predicts_xgb).mean(axis=0).round().astype(int)

submission_xgb = pd.DataFrame({'Id': test_id,
                           'Target': xgboost_pred})
submission_xgb.to_csv('submissionXGB.csv', index=False)


# ## Comparing the various scores:

# In[ ]:


'''models_with_scores = pd.DataFrame({
    'Model': list(validation_scores.keys()),
    'Validation Score': list(validation_scores.values())})

models_with_scores.sort_values(by='Validation Score', ascending=False)'''


# ## Submission Models

# ### LightGBM:

# In[ ]:


submission_model_lgb_old = lgb.LGBMClassifier(class_weight='balanced', boosting_type='dart',
                         drop_rate=0.9, min_data_in_leaf=100, 
                         max_bin=255,
                         n_estimators=500,
                         bagging_fraction=0.01,
                         min_sum_hessian_in_leaf=1,
                         importance_type='gain',
                         learning_rate=0.1, 
                         max_depth=-1, 
                         num_leaves=31)


# In[ ]:


submission_model_lgb_old.fit(X_train, Y_train);


# In[ ]:


final_pred_lgb_old = submission_model_lgb_old.predict(X_test)
final_pred_lgb_old = final_pred_lgb_old.astype(int)


# In[ ]:


submission_lgb_old = pd.DataFrame({'Id': test_id,
                           'Target': final_pred_lgb_old})
submission_lgb_old.to_csv('submissionLGBold.csv', index=False)


# ### XGBoost:

# In[ ]:


'''submission_model_xgboost = lgb.LGBMClassifier()
submission_model_xgboost.fit(X_train, Y_train);
final_pred_xgb = submission_model_xgboost.predict(X_test)
final_pred_xgb = final_pred_xgb.astype(int)'''


# In[ ]:


'''submission_xgb = pd.DataFrame({'Id': test_id,
                           'Target': final_pred_xgb})
submission_xgb.to_csv('submissionXGB.csv', index=False)'''


# ### LightGBM + XGBoost stacked:

# In[ ]:


'''final_pred_stacked = ((final_pred_lgb + final_pred_xgb) / 2).astype(int)
submission_stacked = pd.DataFrame({'Id': test_id,
                           'Target': final_pred_stacked})
submission_stacked.to_csv('submissionStacked.csv', index=False)'''

