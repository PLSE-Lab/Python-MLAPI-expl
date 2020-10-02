#!/usr/bin/env python
# coding: utf-8

# # Use your understanding of unsupervised learning and clustering to find patterns in a survey conducted about masculinity.

# # Investigate the Data
# 
# Welcome to the cumulative project on clustering algorithms! In this project, we will be investigating the way people think about masculinity by applying the KMeans algorithm to data from  <a href="https://fivethirtyeight.com/" target = "_blank">FiveThirtyEight</a>. FiveThirtyEight is a popular website known for their use of statistical analysis in many of their stories.
# 
# To begin, take a look at `masculinity-survey.pdf`. FiveThirtyEight and WNYC studios used this survey to get their male readers' thoughts on masculinity. After looking through some of the questions asked, take a look at FiveThirtyEight's article <a href="https://fivethirtyeight.com/features/what-do-men-think-it-means-to-be-a-man/" target = "_blank">What Do Men Think It Means To Be A Man?</a> to see their major takeaways. We're going to try to find more insights using machine learning.
# 
# In the code block below, we've loaded `masculinity.csv` into a DataFrame named `survey`. This file contains the raw responses to the masculinity survey. Let's start getting a sense of how this data is structured. Try to answer these questions using your Pandas knowledge:
# * What are the names of the columns? How do those columns relate to the questions in the PDF?
# * How many rows are there?
# * How is a question with multiple parts, like question 7, represented in the DataFrame?
# * How many people said they often ask a friend for professional advice? This is the first sub-question in question 7.
# 
# To answer that last question, use the `value_counts()` function. For example, `df["col_a"].value_counts()` gives you a nice summary of the values found in `"col_a"` of the DataFrame `df`.
# 
# You may also want to print `survey.head()` to get a sense of all of the columns.
# 

# In[ ]:


import pandas as pd

survey = pd.read_csv("../input/masculinity/masculinity.csv")
print('total rows:',survey.shape[0])
print('total columns:',survey.shape[1])
print('breakdown of column types:')
display(survey.dtypes.value_counts())
pd.set_option('display.max_columns', None)
display(survey.head(3))


# # Mapping the Data
# 
# In order for us to start thinking about using the KMeans algorithm with this data, we need to first figure out how to turn these responses into numerical data. Let's once again consider question 7. We can't cluster the data using the phrases `"Often"` or `"Rarely"`, but we can turn those phrases into numbers. For example, we could map the data in the following way: 
# * `"Often"` -> `4`
# * `"Sometimes"` ->  `3`
# * `"Rarely"` -> `2` 
# * `"Never, but open to it"` -> `1`
# * `"Never, and not open to it"` -> `0`.
# 
# Note that it's important that these responses are somewhat linear. `"Often"` is at one end of the spectrum with `"Never, and not open to it"` at the other. The other values fall in sequence between the two. You could perform a similar mapping for the `"educ4"` responses (question 29), but there isn't an obvious linear progression in the `"racethn4"` responses (question 28).
# 
# In order to do this transformation, use the `map()` function. `map()` takes a dictionary as a parameter. For example, the following line of code would turn all the `"A"`s into `1`s and all the `"B"`s into `2`s in the column `"col_one"`.
# 
# ```py
# df["col_one"] = df["col_one"].map({"A": 1, "B": 2})
# ```
# 
# We've given you a list of the columns that should be mapped. Loop through the values of the list and map each column using the mapping described above.
# 
# Be careful of your spelling! Punctuation and whitespace is important. Take a look at the `value_counts()` of one of these columns to see if the mapping worked.
# 

# In[ ]:


summary = survey.nunique().sort_values(ascending=False).reset_index()
summary = summary[summary['index'].str.contains("q")]
print('total questions:',len(summary))
print('questions with more than 2 kind of answers:',len(summary[summary[0]>2]))
print('questions with only 2 kind of answers:',len(summary[summary[0]==2]))
display(summary[summary[0]>2],summary[summary[0]==2])


# In[ ]:


print(survey['q0007_0005'].unique())


# In[ ]:


cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",
       "q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", "q0007_0009",
       "q0007_0010", "q0007_0011"]
count = 0
for i in cols_to_map:
    survey[i] = survey[i].map({'Often':4 ,'Sometimes':3,'Rarely':2,'Never, but open to it':1,'Never, and not open to it':0})
    count+=1
print('total',count,'columns mapped')


# In[ ]:


display(survey["q0007_0001"].unique())
display(survey["q0007_0002"].unique())
display(survey["q0007_0001"].head(5),survey["q0007_0002"].head(5))


# In[ ]:


questions=[['Ask a friend for professional advice'],
          ['Ask a friend for personal advice'],
['Express physical affection to male friends, like hugging, rubbing shoulders'],
['Cry'],
['Get in a physical fight with another person'],
['Have sexual relations with women, including anything from kissing to sex'],
['Have sexual relations with men, including anything from kissing to sex'],
['Watch sports of any kind'],
['Work out'],
['See a therapist'],
['Feel lonely or isolated']]


# # Plotting the Data
# 
# We now have 11 different features that we could use in our KMeans algorithm. Before we jump into clustering, let's graph some of these features on a 2D graph. Call `plt.scatter` using `survey["q0007_0001"]` and `survey["q0007_0002"]` as parameters. Include `alpha = 0.1`. We want to include `alpha` because many of the data points will be on top of each other. Adding `alpha` will make the points appear more solid if there are many stacked on top of each other.
# 
# Include axis labels on your graph. The x-axis corresponds with the first column you gave the `scatter()` function. So in this case, it corresponds to the question about asking a friend for professional advice.
# 
# Does it make sense that there are few points in the top left and bottom right corners of the graph? Why? Try graphing other dimensions against each other. Are there any combinations that give you surprising results?
# 

# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
plt.style.use('bmh')
plt.figure(figsize=[15,12])
plt.subplot(2,2,1)
x=survey['q0007_0001']
y=survey['q0007_0002']
both_finite = np.isfinite(x) & np.isfinite(y)
plt.xlabel('professional advice')
plt.ylabel('personal advice')
m, b = np.polyfit(x[both_finite], y[both_finite], 1) #get best fit, 1 means linear.
plt.plot(x,m*x+b)
plt.scatter(x,y,alpha=0.06)

plt.subplot(2,2,2)
x=survey['q0007_0001']
y=survey['q0007_0003']
both_finite = np.isfinite(x) & np.isfinite(y)
plt.xlabel('professional advice')
plt.ylabel('Express physical affection to male friends, like hugging, rubbing shoulders')
m, b = np.polyfit(x[both_finite], y[both_finite], 1) #get best fit, 1 means linear.
plt.plot(x,m*x+b)
plt.scatter(x,y,alpha=0.06)

plt.subplot(2,2,3)
x=survey['q0007_0008']
y=survey['q0007_0006']
both_finite = np.isfinite(x) & np.isfinite(y)
plt.xlabel(questions[7])
plt.ylabel(questions[5])
m, b = np.polyfit(x[both_finite], y[both_finite], 1) #get best fit, 1 means linear.
plt.plot(x,m*x+b)
plt.scatter(x,y,alpha=0.06)

plt.subplot(2,2,4)
x=survey['q0007_0008']
y=survey['q0007_0006']
both_finite = np.isfinite(x) & np.isfinite(y)
plt.xlabel(questions[8])
plt.ylabel(questions[5])
m, b = np.polyfit(x[both_finite], y[both_finite], 1) #get best fit, 1 means linear.
plt.plot(x,m*x+b)
plt.scatter(x,y,alpha=0.06)


# # Build the KMeans Model
# 
# It's now time to start clustering! There are so many interesting questions we could ask about this data. Let's start by seeing if clusters form based on traditionally masculine concepts. 
# 
# Take a look at the first four sub-questions in question 7. Those four activities aren't necessarily seen as traditionally masculine. On the other hand, sub-questions 5, 8, and 9 are often seen as very masculine activities. What would happen if we found 2 clusters based on those 7 questions? Would we find clusters that represent traditionally feminine and traditionally masculine people? Let's find out.
# 
# We need to first drop all of the rows that contain a `NaN` value in any of the columns we're interested in. Create a new variable named `rows_to_cluster` and set it equal to the result of calling `dropna` on `survey`. `dropna` should have a parameter `subset` equal to a list of the 7 columns we want. If you don't include `subset`, the function will drop all rows that have an `NaN` in *any* column. This would drop almost all the rows in the dataframe!
# 
# Create a `KMeans` object named `classifier` where `n_clusters = 2`. Call `classifier`'s `.fit()` method. The parameter of `.fit()` should be the 7 columns we're interested in. For example, the following line of code will fit the model based on the columns `"col_one"` and `"col_two"` of the Dataframe `df`. 
# 
# ```py
# classifier.fit(df[["col_one", "col_two"]])
# ```
# 
# Make sure to only include the columns that you want to train off of. Make sure to use `rows_to_cluster` rather than `survey` to avoid including those `NaN`s!
# 
# 
# 
# After fitting your model, print out the model's `cluster_centers_`.
# 

# In[ ]:


rows_to_cluster = survey.dropna(subset = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"])
print('initial rows:',survey[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]].shape[0])
print('drop nan rows...')
print('total rows:',rows_to_cluster.shape[0])
print('total columns:',rows_to_cluster.shape[1])

from sklearn.cluster import KMeans
model = KMeans(2,random_state=42)
model.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]])
print('The cluster centroids:\n',model.cluster_centers_)


# In[ ]:


centroids = model.cluster_centers_
centroids_list = []
for i in range(len(centroids)):
    centroids_list.append(list(centroids[i]))

x=range(len(["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]))
for i in range(len(centroids_list)):
    y = centroids_list[i]
    plt.plot(x,y,label='cluster '+str(i))
plt.legend()


# * One cluster has a higher centroid for all questions. So it simply means we have 1 cluster that tend to do 'more' things and another cluster that does 'fewer' things.

# # Separate the Cluster Members
# 
# When we look at the two clusters, the first four numbers represent the traditionally feminine activities and the last three represent the traditionally masculine activities. If the data points separated into a feminine cluser and a masculine cluseter, we would expect to see one cluster to have high values for the first four numbers and the other cluster to have high values for the last three numbers.
# 
# Instead, the first cluster has a higher value in every feature. Since a higher number means the person was more likely to "often" do something, the clusters seem to represent "people who do things" and "people who don't do things".
# 
# We might be able to find out more information about these clusters by looking at the specific members of each cluster. Print `classifier.labels_`. This list shows which cluster every row in the DataFrame corresponds to.
# 
# For example,  if `classifier.labels_` was `[1, 0 ,1]`, then the first row in the DataFrame would be in cluster one, the second row would be in cluster 0, and the third row would be in cluster one. A row represents one persons answers to every question in the survey.
# 
# Create two new empty lists named `cluster_zero_indices` and `cluster_one_indices`. Loop through `classifier.labels_` and whenever a label is `0` add that index to `cluster_zero_indices`. Do the same whenever a label is a `1`.
# 
# Print `cluster_zero_indices`

# In[ ]:


print(model.labels_)
print('total labels',len(model.labels_))
print('predicted labels\n',pd.Series(model.labels_).value_counts())
cluster_zero_indices = []
cluster_one_indices = []
for i in range(len(model.labels_)):
    if model.labels_[i] == 0: cluster_zero_indices.append(i)
    else: cluster_one_indices.append(i)
print(cluster_zero_indices)


# # Investigate the Cluster Members
# 
# Now that we have the indices for each cluster, let's look at some stats about these two clusters. You can get the rows of the DataFrame that correspond to cluster zero by doing the following:
# 
# ```py
# cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
# ```
# 
# Do the same for `cluster_one_df`.
# 
# Finally, let's look at some information about these two clusters. Print the `value_counts()` of the `educ4` column of each cluster. What do you notice? Try looking at different columns. For example, are the people in cluster zero significantly older than those in cluster one? You can look at the `age3` column to see.
# 
# If you divide the result of `value_counts()` by the size of the cluster, you get the percentage of people in each category rather than the total number. This will make it easier to compare the two clusters.

# In[ ]:


# survey['educ4'] = survey['educ4'].map({'Post graduate degree':3,'College or more':2,'Some college':1,'High school or less':0})


# In[ ]:


cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]


# In[ ]:


from scipy.stats import chi2_contingency
cols_compare = ['race2','racethn4', 'educ3', 'educ4', 'age3', 'kids', 'orientation']
outcome = []
significant_factors = []

print('\nAnalyze significant factors to the clusters:')
for i in cols_compare:
    x =list(zip(list(cluster_zero_df[i].value_counts()),list(cluster_one_df[i].value_counts())))
    chi2, pval, dof, expect = chi2_contingency(x)
    outcome.append([i,pval])
    if pval <= 0.05: significant_factors.append([i,pval]) 
    print([i,pval])
    
print('\nThe factors that contribute to the different clusters could be:')
for i in range(len(significant_factors)):
    print(significant_factors[i])


# In[ ]:


print('cluster 1 size:',len(cluster_zero_df))
print('cluster 2 size:',len(cluster_one_df))
print('\nDistribution of education:')
print(cluster_zero_df['educ4'].value_counts()/len(cluster_zero_df))
print(cluster_one_df['educ4'].value_counts()/len(cluster_one_df))


# **Cluster classification has a significiant relationship to educational background & sexual orientation!**

# # What if we suspect more clusters?

# In[ ]:


#check optimal clusters
min_ = 1
max_ = 10
inertia_ = []
cluster_centers_ = []
for i in range(min_,max_):
    model = KMeans(i,random_state=42)
    model.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]])
#     model.cluster_centers_
    inertia_.append(model.inertia_)
    
plt.figure(figsize=[13,5])
ax1 = plt.subplot(1,2,1)
plt.title('inertia of clusters used')
plt.plot(list(range(min_,max_)),inertia_)
plt.xlabel('n cluster')
plt.ylabel('inertia')


# In[ ]:


model = KMeans(3,random_state=42)
model.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]])
centroids = model.cluster_centers_
centroids_list = []
for i in range(len(centroids)):
    centroids_list.append(list(centroids[i]))

ax = plt.subplot(1,1,1)
x=range(len(["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]))
name = ['A','B','C']
for i in range(len(centroids_list)):
    y = centroids_list[i]
    plt.plot(x,y,label='cluster'+str(name[i]))
ax.set_xticklabels(["","q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"], rotation =90)
plt.legend()


# * K means With 3 clusters produced the 2 clusters (cluster A, C) similiar to clusters previously seen when Kmeans with 2 clusters. 
# * A third cluster (cluster B) is similar to the cluster C for some questions, but differ a lot in q3, q8, q9.
#     * (B lower) q3: Express physical affection to male friends, like hugging, rubbing shoulders
#     * (B higher) q8: Watch sports of any kind
#     * (B higher) q9: Work out

# In[ ]:


print(model.labels_)
print('total labels',len(model.labels_))
print('predicted labels\n',pd.Series(model.labels_).value_counts())
cluster_a_indices = []
cluster_b_indices = []
cluster_c_indices = []
for i in range(len(model.labels_)):
    if model.labels_[i] == 0: cluster_a_indices.append(i)
    elif model.labels_[i] == 1: cluster_b_indices.append(i)
    else: cluster_c_indices.append(i)

cluster_a_df = rows_to_cluster.iloc[cluster_a_indices]
cluster_b_df = rows_to_cluster.iloc[cluster_b_indices]
cluster_c_df = rows_to_cluster.iloc[cluster_c_indices]


# In[ ]:


cols_compare = ['race2','racethn4', 'educ3', 'educ4', 'age3', 'kids', 'orientation']
outcome = []
significant_factors = []

print('\nAnalyze significant factors to the clusters:')
for i in cols_compare:
    x =list(zip(list(cluster_a_df[i].value_counts()),list(cluster_b_df[i].value_counts()),list(cluster_c_df[i].value_counts())))
    print(x)
    chi2, pval, dof, expect = chi2_contingency(x)
    outcome.append([i,pval])
    if pval <= 0.05: significant_factors.append([i,pval]) 
    print([i,pval])
    
print('\nThe factors that contribute to the different clusters could be:')
for i in range(len(significant_factors)):
    print(significant_factors[i])


# In[ ]:


column = 'orientation'
print('cluster A size:',len(cluster_a_df))
print('cluster B size:',len(cluster_b_df))
print('cluster C size:',len(cluster_c_df))
print('\nDistribution of education:')
print('a\n',cluster_a_df[column].value_counts()/len(cluster_a_df))
print('b\n',cluster_b_df[column].value_counts()/len(cluster_b_df))
print('c\n',cluster_c_df[column].value_counts()/len(cluster_c_df))


# * Notice cluster C has high percentage of Gay/Bisexual.

# # Explore on Your Own
# 
# Great work! You've found out that by answering those 7 questions people don't fall into a "masculine" category or a "feminine" category. Instead, they seem to be divided by their level of education!
# 
# Now it's time for you to explore this data on your own. In this project, we've really focused on question 7 and its sub-questions. Take a look at some of the other questions in the survey and try to ask yourself some interesting questions. Here's a list of questions you could dive into:
# 
# * Which demographic features have stronger correlations with ideas of masculinity (sexual orientation, age, race, marital status, parenthood?)
# * Are certain beliefs or actions linked to more self-described masculine or feminine individuals?
# * How do insecurities change as people grow older?
# 
# 
# Special thanks to the team at FiveThirtyEight and specifically Dhrumil Mehta for giving us access to the data!
# 

# In[ ]:


#we do a survey hypothesis testing on education
from scipy.stats import chi2_contingency
X = [ [30, 10],
         [35, 15],
         [28, 12] ]
chi2, pval, dof, expect = chi2_contingency(X)


# In[ ]:




