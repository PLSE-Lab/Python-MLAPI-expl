#!/usr/bin/env python
# coding: utf-8

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
survey = pd.read_csv("../input/masculinity.csv")
survey.head(2)


# In[ ]:


survey.isna().any()


# In[ ]:


print(survey.columns)
print(len(survey))
print("multiple parts -> multiple columns with each entry indicating personal choice for each part")
print(len(survey[survey.q0007_0001=='Often']))
print(survey.q0007_0001.value_counts())


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

# **Questions in Part 7 of the survey.**
# 
# How often would you say you do each of the following? [RANDOMIZE, matrix]
# 
# 
# Often|| Sometimes|| Rarely|| Never, but open to it || Never, and not open to it || No Answer
# 
# 1.Ask a friend for professional advice
# 
# 2.Ask a friend for personal advice
# 
# 3.Express physical affection to male friends, like hugging, rubbing shoulders
# 
# 4.Cry
# 
# 5.Get in a physical fight with another person
# 
# 6.Have sexual relations with women, including anything from kissing to sex
# 
# 7.Have sexual relations with men, including anything from kissing to sex
# 
# 8.Watch sports of any kind
# 
# 9.Work out
# 
# 10.See a therapist
# 
# 11.Feel lonely or isolated
# 

# In[ ]:


cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",
       "q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", "q0007_0009",
       "q0007_0010", "q0007_0011"]


# In[ ]:


# These questions are all related to frequency. We can map to numerical values.
choices = [[] for i in range(len(cols_to_map))]
mappings = [{} for i in range(len(cols_to_map))]
for i in range(len(cols_to_map)):
    #choices[i]= list(survey[cols_to_map[i]].value_counts().index.unique()) # order is not correct
    choices[i] = ['Sometimes',
                  'Rarely',
                  'Often',
                  'Never, but open to it',
                  'Never, and not open to it',
                  'No answer']
    mappings[i]= dict(zip(choices[i], list(range(len(choices[i])-1,-1,-1))))
    survey[cols_to_map[i]] = survey[cols_to_map[i]].map(mappings[i])
survey.head(20)


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
fig = plt.figure(figsize=(12,7))
ax = plt.subplot(1,1,1)
plt.scatter(survey["q0007_0001"],survey["q0007_0002"],alpha=0.1)
ax.set_xticks(range(len(choices[0])-1,-1,-1))
ax.set_yticks(range(len(choices[1])-1,-1,-1))
ax.set_xticklabels(choices[0][::-1])
ax.set_yticklabels(choices[1][::-1])
plt.xlabel('How often do you ask a friend for professional advice?')
plt.ylabel('How often do you ask a friend for personal advice?')


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


from sklearn.cluster import KMeans
print(len(survey))
rows_to_cluster = survey.dropna(subset=cols_to_map[0:5] + cols_to_map[7:9])
data_to_cluster = rows_to_cluster[cols_to_map[0:5] + cols_to_map[7:9]]
data_to_cluster = data_to_cluster[(data_to_cluster!=0).all(axis=1)]
data_to_cluster['not_masc'] = (data_to_cluster[cols_to_map[0]] + data_to_cluster[cols_to_map[1]] +                               data_to_cluster[cols_to_map[2]] + data_to_cluster[cols_to_map[3]])/4
data_to_cluster['masc'] = (data_to_cluster[cols_to_map[4]] + data_to_cluster[cols_to_map[7]] +                               data_to_cluster[cols_to_map[8]])/4
#0 => no answer. drop 0.
print(len(data_to_cluster))
classifier = KMeans(n_clusters=2)
classifier.fit(data_to_cluster[cols_to_map[0:5] + cols_to_map[7:9]])
print(classifier.cluster_centers_)


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


print(classifier.labels_)
cluster_zero_indices = [i for i in range(len(classifier.labels_)) if classifier.labels_[i]==0]
cluster_one_indices = [i for i in range(len(classifier.labels_)) if classifier.labels_[i]==1]
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


cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]
print(cluster_zero_df.educ4.value_counts()/len(cluster_zero_df))
print(cluster_one_df.educ4.value_counts()/len(cluster_one_df))
print(cluster_zero_df.age3.value_counts()/len(cluster_zero_df))
print(cluster_one_df.age3.value_counts()/len(cluster_one_df))


# In[ ]:


import numpy as np

figure = plt.figure(figsize=(12,7))
ax = plt.subplot(1,1,1)
plt.scatter(data_to_cluster['not_masc'], data_to_cluster['masc'], c=classifier.labels_, alpha=0.2, s=50, cmap='cool') #cool: 0-1 => green to red
plt.xlabel('Average of frequency of \"non-masculine\" activities')
plt.ylabel('Average of frequency of \"masculine\" activities')

#centers = ...
#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

#The figure suggest that the first 4 questions in Part 7 is more significant in masculinity clustering than questions 589.

#If you have time you can even plot this clustering figure for every 2-question combination. E.g. question 3 and 4


# In[ ]:


#Check question 3 and 4

figure = plt.figure(figsize=(12,7))
ax = plt.subplot(1,1,1)
plt.scatter(data_to_cluster[cols_to_map[2]], data_to_cluster[cols_to_map[3]], c=classifier.labels_, alpha=0.2, s=50, cmap='cool') #cool: 0-1 => green to red
plt.xlabel('Question 3')
plt.ylabel('Question 4')
#Somewhat diagonal separation

#Question 3 performs best in classifying masculinity.


# **JZDSML: It seems that our clustering algorithm thinks express physical affection to male friends (like hugging, rubbing shoulders) makes you less masculine.
# Do you agree with it?**

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
