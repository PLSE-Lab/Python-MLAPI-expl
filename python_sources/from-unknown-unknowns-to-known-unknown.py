#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# So this is the first time I am using jupyter notebook, the first time I am participating in a kaggle contest (and even first time using kaggle) and the first time I am going to visualize data using any python libraries. In short I am complete newbie to the data science and machine learning world. Hence, quite nervous and not sure what to do. Apart from that, I am just left with around 5 hours till the deadline of this contest. Although I am not sure about where to start but I really want to give it a try. I think there will be some other fellows like me who are clueless about what to do :) A typical scenario in telling a story through data visualization involves finding answers to certain well defined questions. However, in this case, we don't even know the questions. So let's just start examining the questions asked in the survey one by one. Let's see what interesting information is available, may be we can find some good questions on the way, which we can then answer. The approach that I am going to use is to consider the simplest possible visualization for visualizing the results of each of the questions that were asked in the survey. At this step, I am dealing with unknown unknown. Based on the results of the visualizations, we will get some questions for further exploration: so we move from unknown unknown to known unknown which we can then further explore. I'll list these questions and will get back to answering them if time allows me to do so :)
# <br>
# <h1> Step 1: Load Data</h1>
# The first step is to read the dataset from multiple_choice_responses.csv
# 

# In[ ]:


df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', header=None)
print(df.head())


# We may further need to see textual responses categorized as "Other" for some questions, so we also need to load data from other_text_responses.csv

# In[ ]:


df_other = pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv' , header=None)
print(df_other.head())


# Removing the first two rows from the datasets/dataframes which contain question number and question text.

# In[ ]:


df = df.iloc[2:]
print(df.head())
df_other = df_other.iloc[2:]
print(df_other.head())


# <h4>A few import statements</h4>

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# <h1> Step 2: Age Distribution</h1>
# Let's see what is the age (in years) of the most of the respondants!
# 

# In[ ]:


sns.set(style="darkgrid")

ax = sns.countplot(y=df[1], data=df,
                   facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
ax.set_title("Q1: What is your age (# years)?")
plt.show()


# As evident from the age distribution, most of the respondants belong to the age bracket of 25-29 years, the second most common group is 22-24 years old people, then 30-34 and 18-21 and so on. As data science is an emerging field, most of the respondants are young people who have recently graudated or professionals in their early career stage.
# <h3>Points for further exploration:</h3>
# P1. We can use a box plot to find the mean, max and min age.

# <h1> Step 3: Gender Distribution </h1>
# The next question that was asked from the respondants is about their gender. Let's see how they are distributed on the basis of gender. 

# In[ ]:


ax = sns.countplot(y=df[2], data=df,
                   facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
ax.set_title("Q2: What is your gender? - Selected Choice")

plt.show()


# Most of the respondants were male. It clearly shows that women need to be encouraged to enter the data science and machine learning domain for ensuring gender equality and remove gender biases.
# Those who selected "prefer to self-describe" have given relatively funny answers.

# In[ ]:


q2_other_texts =df_other[20].dropna()
q2_others_list = q2_other_texts.tolist()
q2_others_list.pop(0) #remove the text corresponding to first row which states 'What is your gender? - Prefer to self-describe - Text'
q2_others_string = ' '.join([str(elem) for elem in q2_others_list]) #list to string conversion as word cloud api requires data in string

# Create the wordcloud object
wordcloud = WordCloud(width=960, height=960, margin=0).generate(q2_others_string)
plt.figure( figsize=(20,10) )

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# 
# <h3>Points for Further Exploration</h3>
# P2. Those who responded either prefer to self-describe, or prefer not to say, their responses can be analyzed using some text visualization techniques such as a word cloud.<br>
# P3. As evident by the above barchart, the responses are dominated by men. Why females are lacking behind needs to be explored further. Is it true for all age groups and geographical locations or their is some location/age group in which females are outperforming?

# <h1> Step 4: Geographical Distribution </h1>
# Next we consider the countries from which the respondants are. Plotting a simple bar chart (as we did for gender and age) wasn't a successful choice as we ended up having too many countries.

# In[ ]:


ax = sns.countplot(x=df[4], data=df,
                   facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='10'

)
ax.set_title("Q3: In which country do you currently reside?")

plt.show()


# As shown above, the number of countries is large enough to not fit on the axis of a bar plot so the results are not very intuitive. So just let's see the results in a tabular form to see which country is leading the world of data science and machine learning.

# In[ ]:


country_count =df[4].value_counts()
country_count.drop(country_count.tail(1).index,inplace=True) # remove the count for 'In which country do you currently reside?'
print(country_count)
print("Total No. of Countries: ", len(country_count))


# So India is the leading the world in machine learning and data science (based on the survey results, which might not be true in actual)
# <h3>Points for further exploration</h3>
# P4. As a reader/viewer, the above list is not an extra-ordinarly intuitive one for me. So a better approach can be to aggregate the above results at the continent level. In the original dataset, we just have country names which need to be converted to country codes. These country codes can then be passed to some third-party api for getting the latitude, longitude values and name of the continent from which the country belongs. The continents information can then be used to plot a bar chart with fewer values (from 58 to 7)
# <br>
# P5. Country names can be converted to latitude, longitude values using reverse geocoding techniques. Lat,long pairse can then be used to plot a choropleth map.

# <h1>Step 5: Educational Background</h1>
# The next question in the survey was related to the highest level of formal education that the respondants have attained or plan to attain within the next 2 years. The results show that most of the respondants have a Master's degree or atleast a Bachelor's degree. The third most common category of respondants have an even higher level of education i.e. Doctoral degree. 
# <h3>Points for further exploration</h3>
# P6. Enough data is not available for further analysis in this regard, otherwise we could have analyzed the undergraduate, post-graduate and doctoral degree programs to identify which degree programs are commonly taken by data science and machine learning enthusiasts. This can give insights for further segmentation of learners in terms of self-learning versus university-taught courses. If educational background related data is gathered in more detail, other interesting patterns can be identified such as the most common machine learning/data science programs offered at masters or doctoral level which aspiring data scientists can plan to enrol.<br>
# P7. A relatively small proportion of respondants don't have a formal education or have some college/univeristy study without earning a bachelor's degree. This can be further explored to identify the reasons why they were attracted towards machine learning/data science. Was it their interest in programming/problem solving/mathematical thinking or the attractive salary packages offered to data scientist and machine learning engineers.

# In[ ]:


ax = sns.countplot(y=df[5], data=df,
                   facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
ax.set_title("Q4:  What is the highest level of formal education that you have attained or plan to attain within the next 2 years?")

plt.show()


# <h1>Step 6: Current Role in the Data Science/ Machine Learning Domain</h1>
# The respondants were asked about the title most similar to their current role. The answers were as follows. 

# In[ ]:


ax = sns.countplot(y=df[6], data=df,
                   facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
ax.set_title("Q5: Select the title most similar to your current role (or most recent title if retired): - Selected Choice")

plt.show()


# As clearly evident from the above chart, "Other" is an important category. So we explore the responses categorized as "Other" further using a word cloud which shows a very diverse range of roles.

# In[ ]:


q5_other_texts =df_other[26].dropna()
q5_others_list = q5_other_texts.tolist()
q5_others_list.pop(0) #remove the text corresponding to first row which states 'Select the title most similar to your current role (or most recent title if retired): - Other - Text'
q5_others_string = ' '.join([str(elem) for elem in q5_others_list]) #list to string conversion as word cloud api requires data in string

# Create the wordcloud object
wordcloud = WordCloud(width=960, height=960, margin=0).generate(q5_others_string)
plt.figure( figsize=(20,10) )

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# <h1>Step 7: Company Size</h1>
# The next question was 'What is the size of the company where you are employed?' The responses were as follows.

# In[ ]:


ax = sns.countplot(y=df[8], data=df,
                   facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
ax.set_title("Q6: What is the size of the company where you are employed?")

plt.show()


# The results give an indication that machine learning and data science are most commonly used in startup environments or SMEs (0-49 employees). The second most common type of organizations which are using ML/DS are the ones who are really well-established (>10,000 employees) and want data driven analytics or automation to improve their business process or incorporate customer feedback and so on. An interesting observation here is that none of the categories has a significantly small value which indicates that the potential of using ML/DS for improving business, marketing and finance and providng creative and innovative solutions is there for organizations of all sizes.

# <h1>Step 8: Number of individuals responsible for Data Science workloads</h1>
# The next question in the survey was 'Approximately how many individuals are responsible for data science workloads at your place of business?'
# 

# In[ ]:


ax = sns.countplot(y=df[9], data=df,
                   facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
ax.set_title("Q7: Approximately how many individuals are responsible for data science workloads at your place of business?")

plt.show()


# <h4>Points for further exploration:</h4>
# P8. The number of individuals responsible for data science workloads may depend on the overall size of organization. So the results of the above two questions when combined can give some insights which may not be visible when considered individually.

# In[ ]:


plt.figure(figsize=(10,5))

ax = sns.countplot(x=df[8], hue=df[9], data=df,   palette='Set1')

ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'

)
ax.set_title("Combining results for organizational size and individuals responsible for data science workload")

plt.show()


# The combined results above clearly indicate that large scale organizations have dedicated data science/machine learning teams/departments as they have started taking data-driven innovation seriously. Small to medium scale organizations (<1000 employees) have generally less than 10 employees working in the ML/DS domain. It clearly indicates that DS/ML skills can help you enter a well-established organization in a respectable role which was extremely comptetive otherwise for entering in other deparments.

# <h1>Step 9: Machine Learning methods employed in the company or not</h1>
# The next question in the survey was 'Does your current employer incorporate machine learning methods into their business?'

# In[ ]:


ax = sns.countplot(y=df[10], data=df,
                   facecolor=(0, 0, 0, 0),
                    linewidth=5,
                    edgecolor=sns.color_palette("dark", 3))
ax.set_title("Q8: Does your current employer incorporate machine learning methods into their business?")

plt.show()


# Again, we can have a better understanding of the data if we combine the above results with the comapny size.

# In[ ]:


plt.figure(figsize=(10,5))

ax = sns.countplot(y=df[10], hue=df[8], data=df,   palette='Set1')

ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'

)
ax.set_title("Combining results for organizational size and use of ML methods in the organization")

plt.show()


# The results are quite interesting. Lets consider each category of responses one by one, <br>
# "We have well-established ML methods...."  most of the respondants belong to organizations having >10,000 employees which somehow validates our previous assumption that large organizations are taking ML/DS seriously and are among the early adopters.<br>
# "No (we don't use ML methods)" was selected by people mostly belonging to companies have 0-49 employees, indicating that they are mostly interested in learning ML/DS at an individual level. The same is true for "We are exploring ML methods (and may one day put a model into production)" where startups are planning to offer data-driven solutions.<br>
# "We recently started using ML methods ...." is almost equally true for large as well as small scale organizations. However, medium scale organizations are lagging behind in this case.<br>
# "We use ML methods for generating insights....." again the startup environments are experimenting with ML methods more than other categories of employers.
# 
# 

# <h1>Step 10: Approximate Yearly Compensation($USD)</h1>
# 

# In[ ]:


ax = sns.countplot(x=df[20], data=df,
                   facecolor=(0, 0, 0, 0),
                    linewidth=5,
                   palette='Set1',
                    edgecolor=sns.color_palette("dark", 3))
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='10'

)
ax.set_title("Q10: What is your current yearly compensation (approximate $USD)?")

plt.show()


# The most common category of results indicates the yearly compensation between $0-999 which indicates that most of the respondants are in their early career stage. Similar to previous examples, salary range can also be better understood when considered along with organizations' size.

# In[ ]:


plt.figure(figsize=(10,5))

ax = sns.countplot(x=df[20], hue=df[8], data=df,   palette='Set1')

ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='10'

)
ax.set_title("Combining results for organizational size and yearly compensation")

plt.show()


# The combined results clearly indicate that the most common category of yearly compensation applies to startups (0-49 employees).However, higher values of yearly compensation vary from organization to organization.
