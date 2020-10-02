#!/usr/bin/env python
# coding: utf-8

# **How to Query the NYC Open Data (BigQuery Dataset)**

# In[ ]:


import bq_helper
from bq_helper import BigQueryHelper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="new_york")


# # Start with 2015 Census

# In[ ]:


nyc.table_schema("tree_census_2015")


# ### From forked kernel. Sample questions about the tree census

# Can you find the Virginia Pines in New York City?
# 

# In[ ]:


query3 = """SELECT
  Latitude,
  Longitude
FROM
  `bigquery-public-data.new_york.tree_census_2015`
WHERE
  spc_common="Virginia pine";
        """
response3 = nyc.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(15)


# Where can you find the most of your favorite type of tree?

# In[ ]:


query3 = """SELECT
  UPPER(spc_common) AS common_name,
  ROUND(latitude, 1) AS lat,
  ROUND(longitude, 1) AS long,
  COUNT(*) AS tree_count
FROM
  `bigquery-public-data.new_york.tree_census_2015`
GROUP BY
  spc_common,
  spc_latin,
  lat,
  long
ORDER BY
  tree_count DESC;
        """
response3 = nyc.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(15)


# What are the most popular types of tree in New York City?

# In[ ]:


query4 = """SELECT
  UPPER(spc_common) as common_name,
  UPPER(spc_latin) as latin_name,
  COUNT(*) AS tree_count
FROM
  `bigquery-public-data.new_york.tree_census_2015`
GROUP BY
  spc_common,
  spc_latin
ORDER BY
  tree_count DESC
LIMIT
  10;
        """
response4 = nyc.query_to_pandas_safe(query4, max_gb_scanned=10)
response4.head(15)
# Empty strings '' instead of NULL -- not caught by pd.isna()


# In[ ]:


# Basic functions

def inspect(query, nrows=15, sample=False):
    """Display response from given query but don't save. 
    query: str, raw SQL query
    nrows: int, number of rows to display, default 15
    sample: bool, use df.sample instead of df.head, default False """
    response = nyc.query_to_pandas_safe(query, max_gb_scanned=10)
    if sample:
        return response.sample(nrows)
    return response.head(nrows) 

def retrieve(query, nrows=10):
    """Save response from given query and print a preview. 
    query: str, raw SQL query
    nrows: int, number of rows to display"""
    response = nyc.query_to_pandas_safe(query, max_gb_scanned=10)
    print(response.head(nrows))
    return response

# Save table name as variable for ease 
f = '`bigquery-public-data.new_york.tree_census_2015`'
f2 = '`bigquery-public-data.new_york.tree_census_2005`'


# ### Misc questions (move to end)

# In[ ]:


# What do rows with missing species names look like?
query = f'''SELECT * 
FROM {f}
WHERE spc_common = ''
LIMIT 5
  '''
inspect(query)


# In[ ]:


query = '''SELECT tree_id, user_type, spc_common
FROM
  `bigquery-public-data.new_york.tree_census_2015`
WHERE spc_common = ''

  '''
res = retrieve(query)
res['user_type'].value_counts()
# There can be missing species names from any type of census worker


# ## How well are trees doing within each neighborhood?
# Neighborhood Tabulation Areas (NTA)

# In[ ]:


# Count ntas 
query = f"""SELECT COUNT(DISTINCT nta) AS num_ntas, count(distinct nta_name) as num_nta_names
FROM {f} 
WHERE nta != '' """ 
inspect(query)


# In[ ]:


query = f"""SELECT distinct nta_name as nta_names
FROM {f} 
ORDER BY nta_names
""" 
nta_names = retrieve(query)


# In[ ]:


print([x for x in nta_names['nta_names']])
# Don't actually recognize all of these neighborhoods, but this should be more meaningful than grouping by lat/lon
    # Could merge with area to get tree density?
    # 


# In[ ]:





# In[ ]:


# Count total trees within neighborhood
query = f""" SELECT 
nta_name,
COUNT(tree_id) as tree_count,
AVG(latitude) as lat,
AVG(longitude) AS lon
FROM {f}
GROUP BY nta_name
ORDER BY tree_count desc

"""
inspect(query)
# Most up to date at https://tree-map.nycgovparks.org/


# In[ ]:


# Overall health of trees in NYC
query = f""" SELECT 
CASE WHEN health = '' THEN 'Dead' 
 ELSE health END AS health,
COUNT(tree_id)  as num_trees,
COUNT(tree_id) *100 /(SELECT COUNT(*) FROM {f}) as healthy_pct
FROM {f}
GROUP BY ROLLUP(health)
ORDER BY num_trees
"""
inspect(query)
# status = alive, stump, dead 
# health = Good, Fair, Poor, left blank if dead or stump 
# 77% trees in good health overeall


# In[ ]:


# Calc percent good trees within neighborhood
query = f""" SELECT g.nta_name, g.tree_count, t.total_trees, 
g.tree_count / t.total_trees as healthy_pct,
lon,
lat
FROM
(SELECT nta_name,
    COUNT(tree_id) as tree_count
FROM {f}
 WHERE health = 'Good'
 GROUP BY nta_name
 ORDER BY tree_count desc) g

JOIN --
    (SELECT nta_name,
        COUNT(tree_id) as total_trees,
        AVG(latitude) as lat,
        AVG(longitude) AS lon
    FROM {f} 
     GROUP BY nta_name) t
ON g.nta_name = t.nta_name
 ORDER BY healthy_pct DESC
"""
healthy_trees = retrieve(query, nrows=5)


# In[ ]:


# Get sense of health distribution 
sns.distplot(healthy_trees['healthy_pct'])


# In[ ]:


# Assign color - non uniform bins since everything is above 50%
healthy_trees['bin'] = pd.cut(x=healthy_trees['healthy_pct'], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).astype(str)


# In[ ]:


healthy_trees['bin'].value_counts()


# In[ ]:


print(healthy_trees.shape)
healthy_trees.columns


# In[ ]:


# healthy_trees.to_csv('healthy_trees.csv')


# In[ ]:


# plotly alternative 
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)

# sizeref = 2. * max(array of size values) / (desired maximum marker size ** 2)
healthy_trees['text'] = healthy_trees['nta_name'] + '<br>Tree Count ' + healthy_trees['tree_count'].astype(str)
bins = ["(0.0, 0.5]", "(0.5, 0.6]", "(0.6, 0.7]", "(0.7, 0.8]", "(0.8, 0.9]", "(0.9, 1.0]"]
colors = ["#b2182b","#ef8a62","#fddbc7","#d1e5f0","#67a9cf", "#2166ac"]
# http://colorbrewer2.org/#type=diverging&scheme=RdBu&n=6
health_grps = []
scale = 50

for i, bin in enumerate(bins):
    healthy_trees_bin = healthy_trees[healthy_trees['bin'] == bin]
    print(len(healthy_trees_bin))
    grp = go.Scattergeo(
        locationmode = 'USA-states',
        lon = healthy_trees_bin['lon'],
        lat = healthy_trees_bin['lat'],
        text = healthy_trees_bin['text'],
        marker = go.scattergeo.Marker(
            size = healthy_trees['tree_count']/scale,
            color = colors[i],
            line = go.scattergeo.marker.Line(
                width=0.5, color='rgb(40,40,40)'
            ),
            sizemode = 'area'
        ),
        name = bin )
    health_grps.append(grp)

layout = go.Layout(
        title = go.layout.Title(
            text = '2015 NYC neighborhood tree health status<br>(Click legend to toggle traces)'
        ),
        showlegend = True,
        geo = go.layout.Geo(
            scope = 'usa',
            projection = go.layout.geo.Projection(
                type='albers usa'
            ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        )
    )

fig = go.Figure(data=health_grps, layout=layout)
plotly.offline.iplot(fig, filename='d3-bubble-map-tree_counts')
# not working well yet


# ### Get actual distribution of health wthin each neighborhood, too.

# In[ ]:


query = f""" SELECT 
    nta_name,
    CASE WHEN health = '' THEN 'Dead' 
     ELSE health END AS health,
    COUNT(tree_id) as tree_count
    FROM {f}
    GROUP BY nta_name, health
    ORDER BY nta_name, health
"""
tree_health_by_location = retrieve(query)


# In[ ]:


# add location details for tableau, dropping redundant good trees column
neighborhood_tree_health = tree_health_by_location.merge(healthy_trees.drop(columns=['total_trees']), on='nta_name')


# In[ ]:


# Export for tooltip view of the health breakdown in Tableau
neighborhood_tree_health.to_csv('neighborhood_tree_health.csv')


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1553360799175' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;NY&#47;NYCTreeHealthbyNeighborhood&#47;NYCTreeHealthbyNeighborhood2015&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='NYCTreeHealthbyNeighborhood&#47;NYCTreeHealthbyNeighborhood2015' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;NY&#47;NYCTreeHealthbyNeighborhood&#47;NYCTreeHealthbyNeighborhood2015&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1553360799175');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# ### Neighborhood observations
# 1. Rockaways have some of the neighborhoods with the worst tree health.
# 2. Queens and Staten Island have good overall tree health.
# 3. Stuyvesant Town has the highest percentage of healthy trees but has the fewest trees. 
# 
# What does your neighborhood's tree health looks like?
# 

# In[ ]:





# In[ ]:


# Alternative: How about color by most popular tree species and tool tip tree species breakdown
    # basically same type of queries, color column = max(counts of species grps) instead of bins


# # Let's compare 2015 with 2005 census
# Unfortunately, tree_id's are not consistent across surveys, so we can't track individual trees over the years. However, we can compare how things have changed as a neighborhood.

# In[ ]:


query = f""" SELECT *
FROM {f2}
LIMIT 5
"""
inspect(query)
# note: coordinates are a 2-tuple in location_1 coolumn
# status has different coding
# nta/nta_name have some empty entries


# In[ ]:


# Start by repeating queries to get 2005 data
# Overall health of trees in NYC
query = f""" SELECT 
status,
COUNT(objectid)  as num_trees, -- new name for id
COUNT(objectid) *100 /(SELECT COUNT(*) FROM {f2}) as healthy_pct
FROM {f2}
GROUP BY ROLLUP(status)
ORDER BY num_trees
"""
inspect(query)
# status is too different from health to compare directly - excellent actually means something different - can't just make excellent -> 2015 good, and 2005 good -> fair
# by this metric 90% trees in "good" health 
# notably, poor is higher in 2005 


# In[ ]:


# Count ntas 
query = f"""SELECT COUNT(DISTINCT nta) AS num_ntas, count(distinct nta_name) as num_nta_names
FROM {f2} 
 """ 
inspect(query)
# more than 2015. What are they?


# In[ ]:


query = f"""SELECT distinct nta_name
FROM {f2} 
 """ 
nta_names_2005 = retrieve(query)


# In[ ]:


print([x for x in nta_names_2005['nta_name'] if x not in nta_names['nta_names'].tolist()])
# 6 ntas that didn't have a match in 2015


# In[ ]:


print([x for x in nta_names['nta_names'] if x not in nta_names_2005['nta_name'].tolist()])


# In[ ]:


query = f"""SELECT status,
count(objectid) as excluded_trees
FROM {f2} 
WHERE nta_name IN ('', 'park-cemetery-etc-Brooklyn', 'park-cemetery-etc-Bronx', 'park-cemetery-etc-Manhattan', 'Airport', 'park-cemetery-etc-Queens')
GROUP BY ROLLUP(status)
 """ 
inspect(query)
# "None" is the total row
# 25k so not insignificant - maybe search closest lat/lon neighborhood
# but still relatively small, will exclude for now by using inner join


# In[ ]:


# Calc change in total trees per neighborhood between 2005 and 2015
query = f""" SELECT f.nta_name, 
f2.total_trees as cnt_2005,
f.tree_count as cnt_2015,
f.tree_count - f2.total_trees as change,
(f.tree_count - f2.total_trees) / f2.total_trees as pct_change
FROM
(SELECT nta_name,
    COUNT(tree_id) as tree_count
FROM {f} -- 2015
 GROUP BY nta_name
 ORDER BY nta_name) f

JOIN --
    (SELECT nta_name,
        COUNT(objectid) as total_trees
    FROM {f2} --2005
     GROUP BY nta_name) f2
ON f.nta_name = f2.nta_name
 ORDER BY change, pct_change
"""
tree_diff = retrieve(query, nrows=5)


# In[ ]:


# Calc change in total alive trees per neighborhood between 2005 and 2015
query = f""" SELECT f.nta_name, f2.boroname,
f2.total_trees as cnt_2005,
f.tree_count as cnt_2015,
f.tree_count - f2.total_trees as change,
(f.tree_count - f2.total_trees) / f2.total_trees as pct_change
FROM
(SELECT nta_name,
    COUNT(tree_id) as tree_count
FROM {f} -- 2015
 WHERE health IN ('Good', 'Fair', 'Poor') --alive 
 GROUP BY nta_name
 ORDER BY nta_name) f

JOIN --
    (SELECT boroname,
    nta_name,
        COUNT(objectid) as total_trees
    FROM {f2} --2005
     WHERE status in ('Good', 'Excellent', 'Poor')
     GROUP BY boroname, nta_name) f2
ON f.nta_name = f2.nta_name
 ORDER BY change, pct_change
"""
tree_diff = retrieve(query, nrows=5)


# In[ ]:


tree_diff.head()


# In[ ]:


tree_diff['boroname'].unique()


# In[ ]:


# staten island was encoded numerically 
tree_diff['boroname'] = tree_diff['boroname'].str.replace('5', 'Staten Island')


# Look at neighborhoods aggregated at the borough level.

# In[ ]:


tree_diff.groupby('boroname')['change', 'pct_change'].describe()


# In[ ]:


sns.boxplot(x='change', y='boroname', data=tree_diff)
plt.title('Raw change in tree count by borough, between 2005 and 2015')


# In[ ]:


# Check out the outlier
tree_diff[tree_diff['pct_change'] > 500]
# Sunnyside only had 1 tree in 2005?


# In[ ]:


sns.boxplot(x='pct_change', y='boroname', data=tree_diff[tree_diff['pct_change'] < 500]) # remove outlier 
plt.title('Percent change in tree count by borough, between 2005 and 2015')
# Bronx had some of the bigger increases 
# Notably, Queens and Staten Island had small changes but had a lot of healthy trees already


# In[ ]:


sns.boxplot(x='cnt_2005', y='boroname', data=tree_diff)
plt.title('Tree count distribution by borough, 2005')


# In[ ]:


sns.boxplot(x='cnt_2015', y='boroname', data=tree_diff) 
plt.title('Tree count distribution by borough, 2015')


# In[ ]:


plt.figure(figsize=(10, 40))
sns.barplot(y='nta_name', x='change', data=tree_diff.sort_values('change'))
plt.title('Raw change in tree count by neighborhood, between 2005 and 2015')
# Not the most informative plot but kinda nice to see -> setting hue to borough didn't work out well - unsure why


# In[ ]:


# top 5 biggest increases and decreases
top5s = pd.concat([tree_diff.head(), tree_diff.tail()]).sort_values(by='cnt_2005', ascending=False)
top5m = top5s.melt(id_vars=['nta_name', 'boroname'], value_vars=['cnt_2005', 'cnt_2015'], var_name='Year', value_name='tree_count')


# In[ ]:


ordered_names = top5s['nta_name'].tolist()


# In[ ]:


# Visualize
plt.figure(figsize=(4, 8))
sns.lineplot(x='Year', y='tree_count', hue='nta_name', data=top5m)
plt.legend(bbox_to_anchor=(1.05, 1))
plt.title('Raw changes in tree counts for neighborhoods with the biggest changes')
# East New York probably the most notable for doubling its tree count from <4k to 9k


# In[ ]:


## Inspect tree species


# In[ ]:


# There are some trees threatened by beetles: advised to no longer plant. 
query = f"""
SELECT * 
FROM `bigquery-public-data.new_york.tree_species`
WHERE comments LIKE "Asian Long Horn%"
"""
inspect(query)


# In[ ]:


# 2015 state of threatened species
query = f"""
SELECT r.spc_latin as tree,r.spc_common as common_name, COUNT(*) as tree_count
FROM {f} r
JOIN `bigquery-public-data.new_york.tree_species` s
ON r.spc_latin = s.species_scientific_name
WHERE s.comments LIKE "Asian Long Horn%"
GROUP BY tree, common_name
ORDER BY tree_count DESC
"""
species2015 = retrieve(query)


# In[ ]:


query = f"""
SELECT  r.spc_latin as tree,r.spc_common as common_name,
COUNT(*) as tree_count
FROM {f2} r
JOIN `bigquery-public-data.new_york.tree_species` s
ON lower(r.spc_latin) = lower(s.species_scientific_name)
WHERE s.comments LIKE "Asian Long Horn%"
GROUP BY tree, common_name
ORDER BY tree_count DESC
"""
species2005_incomplete = retrieve(query)
# missing London Planetree?
# common name is different, can't join that easily to 2015
# Neither year has the leprachaun green ash


# In[ ]:


query = f"""
SELECT  r.spc_latin as tree,r.spc_common as common_name,
COUNT(*) as tree_count
FROM {f2} r
WHERE lower(r.spc_common) LIKE '%plane%'
GROUP BY tree, common_name
ORDER BY tree_count DESC
"""
planetree = retrieve(query)


# In[ ]:


species2005 = pd.concat([species2005_incomplete, planetree], axis=0)


# In[ ]:


# convert lower case for joining
species2005['tree'] = species2005['tree'].str.lower()


# In[ ]:


# Get rid of the random x from fraxinus
species2015['tree'] = species2015['tree'].str.replace(' x ', ' ').str.lower()


# In[ ]:


species_cnts = pd.merge(species2005, species2015, how='right', on='tree', suffixes=('_2005', '_2015'))
species_cnts['tree_count_2005'] = species_cnts['tree_count_2005'].fillna(0).astype('int') # fill missing species with count of 0


# In[ ]:


species_cnts['diff'] = species_cnts['tree_count_2015'] - species_cnts['tree_count_2005']
species_cnts.sort_values('diff')


# In[ ]:


sns.barplot(y='common_name_2015', x='diff', data=species_cnts.sort_values('diff'))
# NYC is still planting threatened trees, particularly elms

# To do:
    # See where these trees are being planted?


# 

# In[ ]:


# Check
query = f"""
SELECT  r.spc_latin as tree,r.spc_common as common_name,
COUNT(*) as tree_count
FROM {f2} r
WHERE lower(r.spc_common) LIKE '%maple%'
GROUP BY tree, common_name
ORDER BY tree_count DESC
"""

inspect(query)


# # 2015 Tree Health Modeling
# Can we predict a tree's status based on observations made during the tree census?  
# Data dictionary available at [link](https://data.cityofnewyork.us/api/views/pi5s-9p35/files/2e1e0292-20b4-4678-bea5-6936180074b3?download=true&filename=StreetTreeCensus2015TreesDataDictionary20161102.pdf)

# In[ ]:


query = f"""
SELECT * 
FROM {f}
LIMIT 2"""
head = retrieve(query)


# In[ ]:


head.columns


# In[ ]:


# Create select statement using python loop
q = ''
for col in head.columns:
    q += f'COUNT (DISTINCT {col}) as {col}, ' 


# In[ ]:


q[:-2]


# In[ ]:


# WARNING takes some time to run - could load in the whole table and do .describe() as alternative
query = f"""
SELECT {q[:-2]}
FROM {f}

"""
inspect(query)


# In[ ]:


# Refresher on status distribution  - maybe do health instead- poor/good
query = f"""
SELECT status,
COUNT(*) as tree_count
FROM {f}
GROUP BY status
"""
inspect(query)
# very imbalanced target classes


# In[ ]:


# Get binary classification along with features
query = f"""
SELECT status,
tree_dbh as diam,
curb_loc,
root_stone,
root_grate,
root_other,
trunk_wire,
trnk_light,
trnk_other,
brch_light,
brch_shoe,
brch_other

FROM {f}
WHERE status IN ('Alive', 'Dead')

"""
doa_data = retrieve(query)


# In[ ]:


doa_data.shape


# In[ ]:


doa_data.columns[1:]


# In[ ]:


y = (doa_data['status'] == "Dead")  # Convert from string "Yes"/"No" to binary
doa_data['curb_loc'] = doa_data['curb_loc'].map({'OnCurb': 0, 'OffsetFromCurb':1})
for col in doa_data.columns[3:]:
    doa_data[col] = doa_data[col].map({'No': 0, 'Yes':1})


# ### Very rough modeling to look at feature importance

# In[ ]:


feature_names = doa_data.columns[1:].tolist()
X = doa_data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, stratify=y)
my_model = RandomForestClassifier(n_estimators=100, random_state=1).fit(train_X, train_y)


# In[ ]:


model2 = RandomForestClassifier(n_estimators=100, class_weight='balanced').fit(train_X, train_y)


# In[ ]:


print(classification_report(val_y, model2.predict(val_X)))


# In[ ]:


confusion_matrix(val_y, model2.predict(val_X))


# In[ ]:


# takes a long time
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model2, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
# doesn't do well with  imbalanced binary variables where most of the values are the same 


# In[ ]:


import shap
# For speed, we will calculate shap values on smaller subset of the validation data
small_val_X = val_X.iloc[:150]

explainer = shap.TreeExplainer(model2)
shap_values = explainer.shap_values(small_val_X)

shap.summary_plot(shap_values[1], small_val_X)


# In[ ]:


val_y.iloc[:150].value_counts()
# Imbalanced classes don't do well with simple model interpretation. Need to revisit.


# In[ ]:




