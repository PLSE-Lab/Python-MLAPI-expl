#!/usr/bin/env python
# coding: utf-8

# # The environmental impact of food quality. Part 1: data exploration and cleaning#
# 
# This project focussed on the relationship between food quality and its environmental impact. Statistical analysis is carried out on the data sourced from the openfoodfact database. This notebook illustrates the first part of the project.

# Import the data directly from openfoodfacts website using Pandas library

# In[ ]:



import pandas as pd 

all_food_data = pd.read_csv( "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv", sep="\t", encoding="utf-8") # raw dataframe

all_food_data.info() # datasets infos and memory usage


# # Preliminary exploration and variables selection

# The file is fairly big, 177 columns with 1047591 rows each

# In[ ]:



all_food_data.shape # this returns the database size


# Detailed information about each column can be checked online at :
# https://static.openfoodfacts.org/data/data-fields.txt
# 
# But:
# Which columns are relevant for our study? and how much is stored in these columns? To answer these questions we run a for loop that returns for each the column name, its null values count and % of empty cells.

# In[ ]:


i = 0 # initialize column count
colnum = []
colname = []
nullpc = []
for col in all_food_data:
    i +=1 # update the counter
    nulsum = sum(pd.isnull(all_food_data[col])) # sum of null value for the column (empty cells)
    numrows = len(all_food_data)
    nuls_pourcent = (sum(pd.isnull(all_food_data[col]))/numrows)*100  # % of null value for the column
    r_nuls_pourcent = round(nuls_pourcent, 3) # return only the first 3 digits after the comma of the percentage float value
    #create columnstats database
    colnum.append(i) # first column: column number
    colname.append(col) # column name
    nullpc.append(r_nuls_pourcent) # % null values
    #print('Column ',i, ' name: ',col, '*   Null values (NaN) in this column: ', nulsum, ' % null: ', r_nuls_pourcent) # print the information for each row


# In[ ]:


df_nul = pd.DataFrame({'num':colnum, 'name':colname, '%null':nullpc})
df_nul = df_nul.sort_values(by='%null', ascending=False)


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(23,8))

plt.bar(df_nul['name'], df_nul['%null'])
plt.title('NaN % in the dataframe', color="red", fontsize = 14)
plt.ylabel('% empty cell in the column', color="red", fontsize = 14)
plt.xticks(rotation='vertical')
plt.rcParams['figure.constrained_layout.use'] = True
plt.savefig("null.png", format="PNG", dpi = 100)


# carbon footprint is a necessary variable but has a high % of empty cells. Variables with a higher percentage can be dropped

# In[ ]:


df_nul[df_nul['name'].str.contains(r'carbon(?!$)')] # check for % null in the carbon footprint column


# In[ ]:


#indexcarbon = df_nul.name[df_nul.name == 'carbon-footprint_100g'].index.tolist()
indexcarbon = df_nul.set_index('name').index.get_loc( 'carbon-footprint_100g')
print ('row index containing carbon footprint % null is : ',indexcarbon)
nulthresold = df_nul.iloc[indexcarbon, 2]
print('carbon footprint % null is : ',nulthresold)


# In[ ]:



df_del = df_nul[df_nul['%null'] > nulthresold] 
df_keep = df_nul[df_nul['%null'] <= nulthresold] # data to keep
print(df_del.shape) # check number of columns that get left out 
df_keep.head()


# In[ ]:


enoughfilled_list = df_keep['name'].tolist()
enoughfilled_list


# In[ ]:


new_food_data = all_food_data[enoughfilled_list] # create a new dataframe that contains only listed columns
new_food_data.shape


# There may be still row duplicates. Remove and check if there are any left.

# In[ ]:


no_duplicates__food_data = new_food_data.drop_duplicates(keep=False) #remove duplicates 
duplicateRowsDF = no_duplicates__food_data[no_duplicates__food_data.duplicated(keep=False)] # create a variable that identify any duplicate row in the dataframe
print("All Duplicate Rows based on all columns are :")
print(duplicateRowsDF)
print('no more duplicates: yes!')


# In[ ]:


no_duplicates__food_data.shape


# In[ ]:


print(list(no_duplicates__food_data)) # get headers


# In[ ]:


nonan_df = no_duplicates__food_data.dropna(subset=['carbon-footprint_100g']) #drop most empty rows from the thresold category and check contents
nonan_df.head()


# column containing in the name '_tag' or '_en' have redundant information. Should be deleted

# In[ ]:


df_filteren = nonan_df.filter(regex='_en')
en = list(df_filteren)
df_filtertag = nonan_df.filter(regex='_tag')
tag = list(df_filtertag)
filterlist= en + tag
print(filterlist)


# In[ ]:


no_duplicates__food_data.drop(filterlist, axis = 1, inplace = True)
no_duplicates__food_data.shape


# In[ ]:


# are French and British score the same? (suspiciously they have same %null)
print(nonan_df['nutrition-score-fr_100g'].equals(nonan_df['nutrition-score-uk_100g'])) # function that checks whether two columns contain same values


# In[ ]:





# After a first cleaning step, we can now create subsets of the dataframe by choosing the columns relevant the statistical analysis. 
# 

# In[ ]:


# first step : lists of columns we are interested for each section of the statistical study. 


eco_raw = ['product_name','nutrition-score-fr_100g', 'nutrition_grade_fr','nova_group', 'pnns_groups_1', 'pnns_groups_2', 'main_category', 'categories', 'carbon-footprint_100g', 'carbon-footprint-from-meat-or-fish_100g', 'ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n']
df_eco_raw = no_duplicates__food_data[eco_raw] # this is the corresponding dataframe with raw data



# # selection of relevant columns to describe the categories of food

# In[ ]:


print(df_eco_raw.pnns_groups_2.value_counts(dropna=True)) # sensible subdivision, even if a large part is unknown
print(df_eco_raw.pnns_groups_2.shape)


# some categories in the PNNS_2 classification are double:one with lower case (fruits), and one with capital (Fruits)

# In[ ]:


df_eco_raw['pnns_groups_2'] = df_eco_raw['pnns_groups_2'].str.lower()


# In[ ]:


df_eco_raw


# In[ ]:


print(df_eco_raw.pnns_groups_1.value_counts(dropna=True)) # very broad. Grouping fish, meat, eggs together would bias environmental impact 


# In[ ]:


# categories section has too many entries to be representative, stats would be confusing
df_cat_count = pd.value_counts(df_eco_raw['categories'].values, sort=True) # transform the counts in a database

df_cat_count = df_cat_count.reset_index()
df_cat_count.columns = ['category', 'count']
df_cat_count = df_cat_count[df_cat_count['count'] > 500] # showing only those above 500 it can be seen that some are very similar 
print (df_cat_count)


# In[ ]:


print(df_eco_raw.main_category.value_counts(dropna=True))


# In[ ]:


# many categories in the main_category section. Some containing only one product are not descriptive
df_cat_count = pd.value_counts(df_eco_raw['main_category'].values, sort=True) # transform the counts in a database

df_cat_count = df_cat_count.reset_index()
df_cat_count.columns = ['main_category', 'count']
df_cat_count = df_cat_count[df_cat_count['count'] > 100] # filter low counts
print (df_cat_count)


# main_category contains too many values. Some with very few entries and even among those that have more than 100 entries some don't correspond actually to a food category but rather to a product (escalopes, bavette d'aloyau...)

# To describe the food category the most informative column is pnns_groups_2. Even if a lot are labelled as unknown. 

# # selection of variables that describe the nutritional quality

#  And since the french food grade is inverse proportional to the score

# In[ ]:


grades = ['nutrition_grade_fr','nutrition-score-fr_100g']
df_food_grades = no_duplicates__food_data[grades] 
df_food_grades = df_food_grades.groupby(['nutrition_grade_fr']).mean()
df_food_grades = df_food_grades.reset_index(level=0, inplace=False) # nutrition grade as first column
df_food_grades


# based on the above table the two variables seem proportional. In the plot below it is clear that they are inverse proportional

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.scatter(df_food_grades['nutrition_grade_fr'],  df_food_grades['nutrition-score-fr_100g'])

plt.xlabel('nutrition_grade_fr', color="red", fontsize = 14)
plt.ylabel('average nutrition_score_fr', color="red", fontsize = 14)
from google.colab import files
plt.savefig(" Nutrition grade.png", format="PNG")


# To fit data of nutritional quality then rather than the average score, we can create a variable, with a value 'nutrition_av_grade_fr' directly proportional to the grade

# In[ ]:


df_food_grades['nutrition_grade_fr_n'] = 22 - df_food_grades['nutrition-score-fr_100g']
df_food_grades = df_food_grades.round(0)
df_food_grades.head()


# In[ ]:


plt.figure(figsize=(5,5))
plt.scatter(df_food_grades['nutrition_grade_fr'],  df_food_grades['nutrition_grade_fr_n'])

plt.xlabel('nutrition_grade_fr', color="red", fontsize = 14)
plt.ylabel('nutrition_av_grade_fr', color="red", fontsize = 14)
from google.colab import files
plt.savefig(" Nutrition grade 2.png", format="PNG")


# Due to the way the score is calculated, the above numerical representation works better than a simple inverse of the nutritional score

# In[ ]:


no_duplicates__food_data['Nutritional quality 100g'] = 1/no_duplicates__food_data['nutrition-score-fr_100g'] # inverse of nutritional score
plt.figure(figsize=(10,5))
plt.scatter(no_duplicates__food_data['Nutritional quality 100g'], no_duplicates__food_data['nutrition-score-fr_100g'])
plt.xlabel('quality', color="red", fontsize = 14)
plt.ylabel('Nutritional score', color="red", fontsize = 14)


# We examine now the correlation of quality with nova grade, which indicates how much the food is processed

# In[ ]:


# convert food grades into numbers based on the category average food score
df_eco_raw ['nutrition_grade_fr_n'] = df_eco_raw ['nutrition_grade_fr']
df_eco_raw  = df_eco_raw.replace({'nutrition_grade_fr_n': {'a': 25, 'b': 21, 'c': 16, 'd': 8, 'e':1}})


# In[ ]:


nova = ['nova_group','nutrition_grade_fr_n'] # how much is the food processed? nove class describes it
df_food_nova = df_eco_raw[nova] 
df_food_nova = df_food_nova.groupby(['nova_group']).mean()
df_food_nova = df_food_nova.reset_index(level=0, inplace=False) # gets nova class to the first column
df_food_nova


# In[ ]:


import numpy as np

from sklearn import linear_model
X0 = np.matrix([np.ones(df_food_nova.shape[0]),df_food_nova['nutrition_grade_fr_n'] ]).T
y0 = np.matrix([df_food_nova['nova_group']]).T

regr = linear_model.LinearRegression() 
regr.fit(X0, y0)
y_pred = regr.predict(X0)
accuracy0 = regr.score(X0, y0)
print(accuracy0)


# In[ ]:


# without the outlier
df_food_nova = df_food_nova.drop([1])
df_food_nova


# In[ ]:


X0 = np.matrix([np.ones(df_food_nova.shape[0]),df_food_nova['nutrition_grade_fr_n'] ]).T
y0 = np.matrix([df_food_nova['nova_group']]).T

regr = linear_model.LinearRegression() 
regr.fit(X0, y0)
y_pred = regr.predict(X0)
accuracy0 = regr.score(X0, y0)
print(accuracy0)


# In[ ]:


plt.figure(figsize=(10,5))
plt.scatter(df_food_nova['nutrition_grade_fr_n'], df_food_nova['nova_group'])
plt.plot(df_food_nova['nutrition_grade_fr_n'],y_pred, color = 'tomato' )
plt.xlabel('Nutritional grade', color="red", fontsize = 14)
plt.ylabel('Food nova group - how much is it processed', color="red", fontsize = 14)
from google.colab import files
plt.savefig("processed score.png", format="PNG")


# In[ ]:


eco_all = ['product_name','nutrition_grade_fr','nutrition_grade_fr_n', 'pnns_groups_2' , 'carbon-footprint_100g', 'carbon-footprint-from-meat-or-fish_100g', 'ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n']
df_eco_all = df_eco_raw[eco_all]
df_eco_all = df_eco_all.sort_values(by=['carbon-footprint_100g'], ascending=False) 
df_eco_all[:-10] # the first 10 rows of the dataframe after elimination of redundant columns
print(df_eco_all.shape)


# # selection of variables that describe the environmental impact

# In[ ]:


# lots of empty cells in the column 'carbon-footprint' and 'carbon-footprint-from-meat-or-fish_100g'. 
# Dropping for both limits the database to 10 rows only
df_eco_all_noNan = df_eco_all.dropna(subset=['carbon-footprint-from-meat-or-fish_100g', 'carbon-footprint_100g'])
df_eco_all_noNan = df_eco_all_noNan.sort_values(by=['carbon-footprint-from-meat-or-fish_100g'], ascending=False)
# After eliminating them we can see the relationship of this and the rest of the database
df_eco_all_noNan 


# In[ ]:


# It can be observed from the above table that carbon-footprint-from-meat-or-fish_100g 	is not really informative
# specific only to those 10 products, so also this category can be eliminated for statistical treatment


df_eco = df_eco_all.drop(columns=['carbon-footprint-from-meat-or-fish_100g'])
df_eco.head()


# In[ ]:


palmoil = ['ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n']
df_palm = df_eco_all[palmoil]
df_palm = df_palm.dropna()
df_palm = df_palm.sort_values(by=['ingredients_from_palm_oil_n'], ascending=False)
df_palm.head()


# While we don't consider the carbon footprint specific from meat and fish, both variables related to palm oil are kept since information for each is distinct

# #Stats and outliers final check

# In[ ]:


numeric = ['ingredients_from_palm_oil_n', 'ingredients_that_may_be_from_palm_oil_n', 'carbon-footprint_100g']
df_n = df_eco[numeric]
df_n.head()
stats = df_n.describe()
stats.to_csv('stat.csv')
stats


# the minimum value of carbon footprint is negative... rather unrealistic isn't it?

# In[ ]:





# In[ ]:



df_carbon = df_eco[df_eco['carbon-footprint_100g'] > 10] # filter unrealistic values
df_n = df_carbon[numeric]
df_n.describe() # have a look if now stats look reasonable


# Finally we can export clean data relevant to our study into a new csv file, easy to import since is 50MB vs. 1.4GB for the whole data)

# In[ ]:


df_scaled =((df_n-df_n.min())/(df_n.max()-df_n.min()))*10
df_scaled = df_scaled.rename(columns={'ingredients_from_palm_oil_n': "palm_oil", 'ingredients_that_may_be_from_palm_oil_n': "palm oil?", 'carbon-footprint_100g': "CO2"})
df_scaled


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
box_plot_scaled = sns.boxplot( data= df_scaled)
fig = box_plot_scaled.get_figure()
plt.ylabel("Scaled values")
fig.savefig("box.png", dpi= 100)


# In[ ]:


df_eco.to_csv('openfoodfacts_Eco.csv')


# In[ ]:





# In[ ]:





# 
# 
# 
# 
# 
# 
