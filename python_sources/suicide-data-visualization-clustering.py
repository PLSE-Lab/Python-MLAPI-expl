#!/usr/bin/env python
# coding: utf-8

# # How suicide rate has evolved since 1985 worldwide in regards of some socio-economic and demographic indicators ?
# 
# The World Health Organisation (WHO) is a specialised agency for the United Nations that is concerned with international public health (Wikipedia). For a long period of time the WHO has collected data on the suicide worlwide since 1985.  
# According to the WHO close to 800 000 people die due to suicide every year, which is one person every 40 seconds. Suicide is a global phenomenon and occurs throughout the lifespan. Effective and evidence-based interventions can be implemented at population, sub-population and individual levels to prevent suicide and suicide attempts.  
# 
# I did previously an analysis of this data with R, in this notebook I'd like to use Python to explore the data. 
# Feel free to fork this notebook if you want to extend it, and if you enjoy please upvote ! Thanks

# In[ ]:


# loading the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Plot settings
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.dpi"] = 150


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd ../input/suicide-rates-overview-1985-to-2016\nls')


# In[ ]:





# In[ ]:


# Reading the data into memory
suicide = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")


# Let's view the shape of the dataset. 

# In[ ]:


suicide.shape


# The dataset contains 27820 observation and 12 columns.

# In[ ]:


suicide.columns


# In[ ]:


suicide.head()


# For the purpose of consistency on the column names, let's redefine the columns `suicides/100k pop , country-year, gdp_for_year ($) , gdp_per_capita ($) `

# In[ ]:


replacement_cols = {"suicides/100k pop" : "suicide_100k", "country-year": "country_year",
                   "HDI for year": "hdi_for_year", " gdp_for_year ($) " : "current_gdp",
                   "gdp_per_capita ($)" : "gdp_capita"}


# In[ ]:


suicide = suicide.rename(columns = replacement_cols)
suicide["current_gdp"] = suicide["current_gdp"].str.replace(",", "").astype("float64")


# In[ ]:


suicide.head()


# In[ ]:


suicide.isna().sum()


# Fortunately the dataset doesn't have missing values except the `hdi_for_year` column.

# # Data dictionary
# This short section aims to present what some of the columns stand for.  
# - suicide_100k : This column represents the number of death by suicide out of total of 100.000 deaths. You can think of it as percentage but with a base 100.000 instead of 100.
# - hdi_for_year : It stands for the the Human Development Index for the year. It's between 0 and 1. Higher values represent higher life quality (education, health, and income )
# - current_gdp : It stands for the the Gross Domestic Product (the amount of wealth produced in the country in one year).
# 
# ## Note on the structure of the data
# As we can see on the first five rows of the dataset, there seem to be redundant entries but there are not. There seems to have redundant entries because of the levels of the categorical columns (sex, age, generation). Each row represents different entries with one or more changes in the categorical columns.

# # Exploration
# Here we want to understand the dataset, the number of countries, the number of levels in different group categories, visualisations etc...

# In[ ]:


# Number of unique countries
len(pd.unique(suicide["country"]))


# There is a total of 101 unique countries in the dataset. But does this mean that every countries have the same number of occurences ? Let's check for the number of occurences

# In[ ]:


Counter(suicide["country"]).most_common()[:10]


# In[ ]:


Counter(suicide["country"]).most_common()[-10:]


# The dataset does not contain the same number of occurence for each country. This is mainly because the data is not available for every year for each country.  
# Let's explore the number of data points available for each years.

# In[ ]:


len(pd.unique(suicide["year"]))


# The dataset covers a period of 32 years. What is the range of the dataset ?

# In[ ]:


min(suicide["year"])


# In[ ]:


max(suicide["year"])


# The data covers the period of 1985 to 2016

# In[ ]:


suicide["year"].value_counts().plot.bar()
plt.title("Number of occurences of each year", size = 30)
plt.ylabel("# Counts");


# We don't have data for all the countries everytime. Some years have very few observation, that's the case of 2016.

# ## Categorical variables

# In[ ]:


suicide["sex"].value_counts().plot.bar()
plt.title("Gender repartition", size = 30)
plt.ylabel("# Counts");


# In[ ]:


suicide["age"].value_counts().plot.bar()
plt.title("Age group repartition", size = 30)
plt.ylabel("# Counts");


# The two previous columns are well balanced

# In[ ]:


suicide["generation"].value_counts().plot.bar()
plt.title("generation repartition", size = 30)
plt.ylabel("# Counts");


# ## Correlation among numeric variables
# We need to group the data in order to aggregate the values that are spread into multiple rows. 

# In[ ]:


def correlation_plot(data, year):
    df = suicide[suicide["year"] == year].drop(columns = "year").copy()
    sns.heatmap(
    df.groupby("country").agg("sum").corr(), cmap = "viridis", annot = True)
    plt.title(f"Correlation among numeric variables in {year}")
    plt.show();


# In[ ]:


plt.figure(figsize= (5, 3))
correlation_plot(suicide, 1985)


# In[ ]:


plt.figure(figsize= (5, 3))
correlation_plot(suicide, 2015)


# # Evolution of the variables

# In[ ]:


suicide["year"] = pd.to_datetime(suicide["year"], format = "%Y")


# In[ ]:


df = suicide.groupby(["year", "sex"]).agg("mean").reset_index()
sns.lineplot(x = "year", y = "suicide_100k", hue = "sex", data = df)
plt.xlim("1985", "2015")
plt.title("Evolution of the mean of suicide 100k (1985 - 2015)", size = 30);


# In[ ]:


df = suicide.groupby(["year", "sex", "age"]).agg("mean").reset_index()

sns.relplot(x = "year", y = "suicide_100k", 
            hue = "sex", col = "age", col_wrap = 3, data = df, 
            facet_kws=dict(sharey=False), kind = "line")

plt.xlim("1985", "2015")
plt.subplots_adjust(top = 0.9)
plt.suptitle("Evolution of suicide per sex and age category (1985 - 2015)", size = 30);


# There is a significant gap between the suicide rate of man and woman. 

# # Top / Bottom countries in variables

# Let's define a function that plots horizontal barplots to represent the countries with higher or smallest value for variables.

# In[ ]:


def barplot(year:int, nb:int, column = "suicide_100k"):
    """
    This function plots the top / bottom n countries of the specified variable.
    """
    df = suicide.groupby(["year", "country"]).agg("sum").reset_index()
    suicide_rate_year = df[df["year"] == f"{year}-01-01"].set_index("country")[column]
    if nb > 0:
        suicide_rate_year.nlargest(nb).sort_values().plot(kind = "barh")
        plt.title(f"Top {nb} country with highest {column} in {year}", size = 20)
        plt.show()
    elif nb < 0:
        suicide_rate_year.nsmallest(abs(nb)).sort_values().plot(kind = "barh")
        plt.title(f" Top {abs(nb)} countries with smallest {column} in {year}", size = 20)
        plt.show()


# Let's now use the function

# ## Countries with highest suicide rate over years

# In[ ]:


barplot(year= 1985, nb = 12, column= "suicide_100k")


# In[ ]:


barplot(year= 1985, nb = -12, column= "suicide_100k")


# In[ ]:


barplot(2015, 10, "suicide_100k")


# In[ ]:


barplot(2015, -10, "suicide_100k")


# # Link between the evolution of suicide and the evolution of other variables  
# We want to see whether there has been links amongs the numeric variables

# In[ ]:


def plot_scatter_links(x:str, y:str, years:list, 
                       agg_fun = "sum", n_col = None):
    
    df = suicide[suicide["year"].isin(years)]
    
    df = df.groupby(["year", "country"]).agg(agg_fun).reset_index()
    df["year"] = df["year"].dt.year.astype("object")
    sns.lmplot(x, y, scatter = True, fit_reg= True, 
               data = df, col = "year", col_wrap = n_col, sharex = False, sharey = False)


# ## Suicide vs GDP per capita

# In[ ]:


plot_scatter_links(x = "gdp_capita", y = "suicide_100k", 
                   years = ["1985","1995", "2005", "2015"], n_col= 2, agg_fun= "mean")
plt.subplots_adjust(top=.9)
plt.suptitle("Sense of causation between Suicide and GDP/capita", size = 30);


# # Cluster Analysis

# In[ ]:


suicide.head()


# Before we aggregate the data, it's important we track every single occurence so we can have back the original values. For that purpose I'm adding a new column called `occurence` which gives tha value 1 to every row so that whenever we group and sum the numeric data we can come back to the original data by dividing it.

# In[ ]:


suicide["occurence"] = 1
suicide["part_generation"] = 0


# In[ ]:


suicide_wide = suicide.groupby(["year", "country"]).agg("sum").reset_index()

suicide_wide["year"] = suicide_wide["year"].dt.year


# In[ ]:


suicide_wide.head(10)


# In[ ]:


suicide_wide.shape


# # Make the dataset wider 
# Here I want to have a column for each of the generation. For that purpose I'll use the pivot function.

# In[ ]:


interm = suicide.pivot(columns = "generation", values = "population")
suicide_new = pd.concat([suicide, interm], axis = 1).fillna(0)
suicide_new = suicide_new.groupby(["year", "country"]).agg("sum").reset_index()


# In[ ]:


suicide_new.shape


# In[ ]:


suicide_new["current_gdp"] = suicide_new["current_gdp"] / suicide_new["occurence"]
suicide_new["gdp_capita"] = suicide_new["gdp_capita"] / suicide_new["occurence"]
suicide_new["hdi_for_year"] = suicide_new["hdi_for_year"] / suicide_new["occurence"]
del suicide_new["occurence"]
del suicide_new["population"]


# In[ ]:


suicide_new.head()


# Now we have a column for each of the generation. 
# # Scaling the dataset
# It's now important to scale the dataset so each column will have the same weight. This is important because of the distance metric the KNN algorithm uses.

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


X = StandardScaler()


# In[ ]:


X = X.fit_transform(suicide_new.drop(["year", "country"], axis = 1))


# In[ ]:


selected_columns = ['suicides_no', 'suicide_100k', 'hdi_for_year',
       'current_gdp', 'gdp_capita', 'part_generation', 'Boomers',
       'G.I. Generation', 'Generation X', 'Generation Z', 'Millenials',
       'Silent']

suicide_new[selected_columns] = X


# In[ ]:


suicide_new.head()


# # Kmeans clustering
# ## for year 1985

# In[ ]:


from sklearn.cluster import KMeans
data = suicide_new[suicide_new["year"] == "1985"].drop(["year", "suicides_no"], 
                                                       axis = 1).set_index("country")


# In[ ]:


model = KMeans(init= "k-means++", n_clusters= 4, n_init = 20)
model = model.fit(data)


# In[ ]:


pd.value_counts(model.labels_, sort = False)


# In[ ]:


k = 4
clusters = []
for j in range(0,k):
    clusters.append([])
for i in range(0,data.shape[0]):
    clusters[model.labels_[i]].append(data.index[i])
#
# Print out clusters
#
for j in range(0,k):
    print(30*"-", "\n")
    print( j, clusters[j])


# ## For year 2015

# In[ ]:


data = suicide_new[suicide_new["year"] == "2015"].drop(["year", "suicides_no"], 
                                                       axis = 1).set_index("country")
model = KMeans(init= "k-means++", n_clusters= 4, n_init = 20)
model = model.fit(data)


# In[ ]:


k = 4
clusters = []
for j in range(0,k):
    clusters.append([])
for i in range(0,data.shape[0]):
    clusters[model.labels_[i]].append(data.index[i])
#
# Print out clusters
#
for j in range(0,k):
    print(30*"-", "\n")
    print( j, clusters[j])


# # Hierarchical clustering algorithm

# According to the KMeans Algorithms this is how similar are the countries in 1985. Let's build something much more deeper

# In[ ]:


from scipy.cluster import hierarchy as sch


# In[ ]:


z = sch.linkage(data, method = "ward")


# In[ ]:


info = sch.dendrogram(z, orientation = "top", labels= data.index)
plt.title("Countries similarities in 2015", size = 30);

