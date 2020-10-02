#!/usr/bin/env python
# coding: utf-8

# # Udemy Courses Research

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


udemy = pd.read_csv('../input/udemy-courses/udemy_courses.csv')


# # Data preparation.

# Let's check the data and prepare it first.

# In[ ]:


udemy.head(5)


# In[ ]:


udemy.info()


# In[ ]:


udemy.isna().sum()


# In[ ]:


udemy.duplicated().sum()


# There are 6 duplicates, so we need to remove them.

# In[ ]:


udemy = udemy.drop_duplicates().reset_index(drop=True)


# In[ ]:


udemy.duplicated().sum()


# Now there isn't any duplicates.

# Let's look for some artefacts in numeric data of the DataFrame.
# 
# We can have the zero values in price column, subscribers amount or reviews. But if there any courses without lectures it probably can have some effect on the research.

# In[ ]:


udemy[udemy['num_lectures']==0]['course_id'].count()


# There is one. Of course, it's not significant, but we'll delete this row.

# In[ ]:


udemy.loc[udemy['num_lectures']==0]


# In[ ]:


udemy.drop([890], inplace = True)


# In[ ]:


udemy.reset_index(drop=True, inplace=True)


# Check result.

# In[ ]:


udemy[udemy['num_lectures']==0]['course_id'].count()


# In[ ]:


udemy.info()


# # What are the best free courses by subject?

# I would say the best course is the one that has many subscribers and reviews and great ratings at the same time. But we don't have rating information here. We know that rating based on reviews, but reviews can be also bad. That's why we cannot rely on the number of reviews if we're looking for the best courses. The course can have many bad reviews as well. So with data we have, we only can consider here the number of subscribers for this task. 
# 
# Firstly, we need to find all subject names from this DataFrame.

# In[ ]:


udemy['subject'].unique()


# Choose all free courses by every subject and assign them to variables.

# In[ ]:


udemy_bf = udemy.loc[
    (udemy['subject'] == 'Business Finance') & (udemy['is_paid'] == False)]

udemy_gd = udemy.loc[
    (udemy['subject'] == 'Graphic Design') & (udemy['is_paid'] == False)]

udemy_mi = udemy.loc[
    (udemy['subject'] == 'Musical Instruments') & (udemy['is_paid'] == False)]

udemy_wd = udemy.loc[
    (udemy['subject'] == 'Web Development') & (udemy['is_paid'] == False)]


# Create the function for searching 5 the best free courses by every subject. I will print 5 courses by every subject just because it's interesting for me to look over top-5. Then I'm going to consolidate the first ones.

# In[ ]:


def find_best_courses(df):
    return df.sort_values('num_subscribers', ascending=False)[
       ['course_id', 'course_title', 'level', 'num_subscribers', 'num_reviews', 'subject']].head()


# In[ ]:


# <sort the free Business Finance courses in descending order by 'num_subscribers' column and show the first 5 of them>
find_best_courses(udemy_bf)


# It's surprising for me that there is an intermediate level course on the 4th place of the top. I thought that the most popular courses will be only for beginners or all levels.

# In[ ]:


# <sort the free Graphic Design courses in descending order by 'num_subscribers' column and show the first 5 of them>
find_best_courses(udemy_gd)


# In[ ]:


# <sort the free Musical Instruments courses in descending order by 'num_subscribers' column and show the first 5 of them>
find_best_courses(udemy_mi)


# The intermediate level course again in the top-5. 

# In[ ]:


# <sort the free Web Development courses in descending order by 'num_subscribers' column and show the first 5 of them>
find_best_courses(udemy_wd)


# Now we can consolidate the received data.

# In[ ]:


data = [
    [udemy.loc[492][0], udemy.loc[492][1], udemy.loc[492][5], 
     udemy.loc[492][8], udemy.loc[492][9], udemy.loc[492][11]],
    [udemy.loc[2820][0], udemy.loc[2820][1], udemy.loc[2820][5], 
     udemy.loc[2820][8], udemy.loc[2820][9], udemy.loc[2820][11]],
    [udemy.loc[1456][0], udemy.loc[1456][1], udemy.loc[1456][5], 
     udemy.loc[1456][8], udemy.loc[1456][9], udemy.loc[1456][11]],
    [udemy.loc[1890][0], udemy.loc[1890][1], udemy.loc[1890][5], 
      udemy.loc[1890][8], udemy.loc[1890][9], udemy.loc[1890][11]]
]

columns = ['course_id', 'course_title', 'num_subscribers', 'level', 'content_duration', 'subject']


# The table below presents 4 courses that turned out to be the best free ones in their subject on Udemy platform.

# In[ ]:


best_courses_per_subject = pd.DataFrame(data = data, columns = columns)
best_courses_per_subject


# Among the best free courses by every subject all of them are for beginners and all levels. 

# # What are the most popular courses?

# Unlike the previous task, we can consider the number of reviews here. I think so because the number of reviews shows us the people's interest. It doesn't matter here if reviews are good or bad. If people write them it means that they are interested in it. And if there are many reviews it means that people are very interested (even if the course turned out to be bad for them). 
# 
# Therefore, the logic here is to find some kind of public interest index and sort the data by it. I'm going to sum numbers of subscribers and reviews and this number will be our index.

# In[ ]:


udemy['public_interest'] = udemy['num_subscribers'] + udemy['num_reviews']
udemy.head()


# In[ ]:


udemy.sort_values('public_interest', ascending=False).head()


# As we can see the most popular course is the Web development course "Learn HTML5 Programming From Scratch". Also, top-5 of the most popular courses are the web development. Which was pretty expectedly because the IT in general and especially Web Technology are the most promising and interesting areas now. All top-5 courses are relevant for beginners.

# # What are the most engaging courses?

# I assume that this task needs to be solved by dividing the subjects. I guess that different subjects may have different "attractiveness indicators" and therefore the results.
# 
# I'm going to split up the data by subjects first. Then I will find mean course prices, mean content duration and mean public interest index for each subject separately.

# In[ ]:


udemy_bf = udemy.loc[(udemy['subject'] == 'Business Finance')]
udemy_gd = udemy.loc[(udemy['subject'] == 'Graphic Design')]
udemy_mi = udemy.loc[(udemy['subject'] == 'Musical Instruments')]
udemy_wd = udemy.loc[(udemy['subject'] == 'Web Development')]


# Here I'll find mean prices for each subject.

# In[ ]:


# <Business Finance mean price>
bf_mean_price = udemy_bf['price'].mean()

# <Graphic Design mean price>
gd_mean_price = udemy_gd['price'].mean()

# <Musical Instruments mean price>
mi_mean_price = udemy_mi['price'].mean()

# <Web Development mean price>
wd_mean_price = udemy_wd['price'].mean()

print('Business Finance mean price: {:.3f}'.format(bf_mean_price))
print('Graphic Design mean price: {:.3f}'.format(gd_mean_price))
print('Musical Instruments mean price: {:.3f}'.format(mi_mean_price))
print('Web Development mean price: {:.3f}'.format(wd_mean_price))


# So, I assume that the most financially attractive courses are the ones which have not more than mean subject price.
# 
# Now I'll find mean content durations for each subject.

# In[ ]:


# <Business Finance mean content duration>
bf_mean_con_dur = udemy_bf['content_duration'].mean()

# <Graphic Design mean content duration>
gd_mean_con_dur = udemy_gd['content_duration'].mean()

# <Musical Instruments mean content duration>
mi_mean_con_dur = udemy_mi['content_duration'].mean()

# <Web Development mean content duration>
wd_mean_con_dur = udemy_wd['content_duration'].mean()

print('Business Finance mean content duration: {:.3f}'.format(bf_mean_con_dur))
print('Graphic Design mean content duration: {:.3f}'.format(gd_mean_con_dur))
print('Musical Instruments mean content duration: {:.3f}'.format(mi_mean_con_dur))
print('Web Development mean content duration: {:.3f}'.format(wd_mean_con_dur))


# And I'll find the mean public interest index which I've calculated in the previous section. 

# In[ ]:


# <Business Finance mean index of public interest>
bf_mean_ipi = udemy_bf['public_interest'].mean() 

# <Graphic Design mean index of public interest>
gd_mean_ipi = udemy_gd['public_interest'].mean() 

# <Musical Instruments mean index of public interest>
mi_mean_ipi = udemy_mi['public_interest'].mean()

# <Web Development mean index of public interest>
wd_mean_ipi = udemy_wd['public_interest'].mean() 

print('Business Finance mean public interest index: {:.3f}'.format(bf_mean_ipi))
print('Graphic Design mean public interest index: {:.3f}'.format(gd_mean_ipi))
print('Musical Instruments mean public interest index: {:.3f}'.format(mi_mean_ipi))
print('Web Development mean public interest index: {:.3f}'.format(wd_mean_ipi))


# Eventually, I suggest that the most engaging courses in each subject are the ones which: 
# - are cost not more than mean subject price;
# - have at least the mean content duration or more;
# - have the public interest index not less than mean one.
# 
# Lets find suitable courses.

# The most engaging Business Finance courses.

# In[ ]:


bf_eng = udemy.loc[
    (udemy['subject'] == 'Business Finance') &
    (udemy['price'] <= bf_mean_price) & 
    (udemy['content_duration'] >= bf_mean_con_dur) &
    (udemy['public_interest'] >= bf_mean_ipi)
]

print('Number of courses: ', bf_eng['course_id'].count())


# In[ ]:


#The most engaging Business Finance courses rows
bf_eng


# The most engaging Graphic Design courses.

# In[ ]:


gd_eng = udemy.loc[
    (udemy['subject'] == 'Graphic Design') &
    (udemy['price'] <= gd_mean_price) & 
    (udemy['content_duration'] >= gd_mean_con_dur) &
    (udemy['public_interest'] >= gd_mean_ipi)
]

print('Number of courses: ', gd_eng['course_id'].count())


# In[ ]:


#The most engaging Graphic Design courses rows
gd_eng


# The most engaging Musical Instruments courses.

# In[ ]:


mi_eng = udemy.loc[
    (udemy['subject'] == 'Musical Instruments') &
    (udemy['price'] <= mi_mean_price) & 
    (udemy['content_duration'] >= mi_mean_con_dur) &
    (udemy['public_interest'] >= mi_mean_ipi)
]

print('Number of courses: ', mi_eng['course_id'].count())


# In[ ]:


#The most engaging Musical Instruments courses rows
mi_eng


# The most engaging Web Development courses.

# In[ ]:


wd_eng = udemy.loc[
    (udemy['subject'] == 'Web Development') &
    (udemy['price'] <= wd_mean_price) & 
    (udemy['content_duration'] >= wd_mean_con_dur) &
    (udemy['public_interest'] >= wd_mean_ipi)
]

print('Number of courses: ', wd_eng['course_id'].count())


# In[ ]:


#The most engaging Web Development courses rows
wd_eng


# According to the proposed selection conditions the most engaging courses were found for each subject:
# - 32 courses among the Business Finance subject;
# - 11 coures among the Graphic Design subject;
# - 9 courses among the Musical Instruments subject;
# - 34 courses among the Web Development subject.
# 
# Besides, if you look at the values calculated above, you can also see some more interesting things.

# In[ ]:


data = [
    [wd_mean_price, wd_mean_con_dur, wd_mean_ipi, wd_eng['course_id'].count()],
    [gd_mean_price, gd_mean_con_dur, gd_mean_ipi, gd_eng['course_id'].count()],
    [bf_mean_price, bf_mean_con_dur, bf_mean_ipi, bf_eng['course_id'].count()],
    [mi_mean_price, mi_mean_con_dur, mi_mean_ipi, mi_eng['course_id'].count()]
]

index = ['Web Development', 'Graphic Design', 'Business Finance', 'Musical Instruments']
columns = ['Mean price', 'Content duration', 'Public interest', 'Engaging courses']

total_df = pd.DataFrame(data = data, index = index, columns = columns)
total_df


# The Web Development courses have the biggest public interest index, furthemore it's almost 1,5 times bigger than all other subjects' indexes together have. And this despite the fact that the mean price of Web Development courses is also more than others.
# 
# Also comparing Business Finance and Graphic Design, we can see that despite the Business Finance courses' lower public interest and higher price that subject has more amount of engaging courses than Graphic Design subject.

# # Which courses offer the best cost benefit?

# If we talk about the best cost benefit, so we should search among the paid courses. Let's exclude free courses from DataFrame and remaining ones assign to a new variable. 

# In[ ]:


paid_udemy = udemy.loc[
    udemy['is_paid'] == True
].copy()
paid_udemy.sample(5)


# The most beneficial courses have the lowest price for unit of content. So, let's find this price. 

# In[ ]:


# <Find the content duration in minutes>
paid_udemy['contdur_min'] = paid_udemy['content_duration'] * 60


# In[ ]:


# <Find the price for 1 minute of content>
paid_udemy['price/min'] = paid_udemy['price'] / paid_udemy['contdur_min']


# And now we can sort received values.

# In[ ]:


paid_udemy.sort_values('price/min').head()


# Among the paid courses these 5 offer the best cost benefit:

# In[ ]:


paid_udemy[
    ['course_id', 'course_title', 'price', 'content_duration', 'price/min','subject']
].sort_values(by = 'price/min').head() 


# These results are based on numbers that don't take into account the ratings of courses. Rating is important when we look for courses that offer the best cost benefit, just as when we looked for the best free courses. But we have to work with what we have :)
# 
# And we can also assume that there are some beneficial courses among free ones either. At least we can probably consider so the ones that have the biggest index of public interest.

# In[ ]:


free_udemy = udemy.loc[
    udemy['is_paid'] == False
]

free_udemy[['course_id', 'course_title', 'price','num_subscribers', 'num_reviews', 'public_interest', 'content_duration', 'subject']].sort_values('public_interest', ascending = False).head(3)


# I think these 3 Web Development courses also can be considered as beneficial. They have the most values of public interest among all the free courses which certainly tells us about user trust in them and their efficiency.

# # Some other interesting things.

# Let's find out how many courses per subjects there are on Udemy.

# In[ ]:


subjects = 'Business Finance', 'Web Development', 'Graphic Design', 'Musical Instruments'
amounts = [
    udemy[udemy['subject']=='Business Finance']['course_id'].count(),
    udemy[udemy['subject']=='Web Development']['course_id'].count(),
    udemy[udemy['subject']=='Graphic Design']['course_id'].count(),
    udemy[udemy['subject']=='Musical Instruments']['course_id'].count()
]


# In[ ]:


pie_chart, axes = plt.subplots()
axes.pie(amounts, labels = subjects, autopct='%.2f%%', shadow=True, radius=1.8, startangle = 45)

centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)


# It was pretty expectable because the business and money will exist forever (at least 'til the world ends). And also looking at how and in what directions our world is developing now, people realize that web technologies will be demanded for many decades in the future. 
# 
# Let's find out how many courses per levels there are on Udemy.

# In[ ]:


udemy['level'].unique()


# In[ ]:


levels = 'All Levels', 'Beginner Level','Intermediate Level', 'Expert Level'
values = [
    udemy[udemy['level']=='All Levels']['course_id'].count(),
    udemy[udemy['level']=='Beginner Level']['course_id'].count(),
    udemy[udemy['level']=='Intermediate Level']['course_id'].count(),
    udemy[udemy['level']=='Expert Level']['course_id'].count()
]


# In[ ]:


bar_chart, axes = plt.subplots(figsize=(8,7))
axes.bar(levels, values)
axes.set_title('The number of courses of each subject by level')


# It turned out that overwhelming amount of courses are for beginners and all levels. I would say that "all levels" could be also attributed to "beginner level" becuase if course fits for everybody it means that course is presented in simple language. So, we get that intermediate and expert courses are not even  close so popular. I would suggest that this is due to the fact that experts (who actually are experienced people) spend much more time to train their skills in real practice than taking theoretical courses on the Internet. Online courses are more useful for beginners, who need to start from somewhere. That's why courses for everybody are such popular.

# What was the annual increase of courses in general and for each subject separately?

# In[ ]:


# <Convert the column 'published_timestamp' with string values to timestamp and 
# add new column with years only>
udemy['published_timestamp'] = pd.to_datetime(udemy['published_timestamp'])
udemy['published_year'] = udemy['published_timestamp'].dt.year


# In[ ]:


udemy['published_year'].unique()


# In[ ]:


y2011 = udemy[udemy['published_year'] == 2011]['course_id'].count()
y2012 = udemy[udemy['published_year'] == 2012]['course_id'].count()
y2013 = udemy[udemy['published_year'] == 2013]['course_id'].count()
y2014 = udemy[udemy['published_year'] == 2014]['course_id'].count()
y2015 = udemy[udemy['published_year'] == 2015]['course_id'].count()
y2016 = udemy[udemy['published_year'] == 2016]['course_id'].count()
y2017 = udemy[udemy['published_year'] == 2017]['course_id'].count()


# In[ ]:


# <Data for a future graph>
years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
courses_amount = [y2011, y2012, y2013, y2014, y2015, y2016, y2017]


# In[ ]:


# <Create a function to count annual amount of published courses per subjects>
def count_courses_per_years(df, subject, year):
    amount = df[(df['subject'] == subject) & (df['published_year'] == year)]['course_id'].count()
    return amount


# In[ ]:


# <Amount of Web Development courses were published in each year>
wd_2011 = count_courses_per_years(udemy, 'Web Development', 2011)
wd_2012 = count_courses_per_years(udemy, 'Web Development', 2012)
wd_2013 = count_courses_per_years(udemy, 'Web Development', 2013)
wd_2014 = count_courses_per_years(udemy, 'Web Development', 2014)
wd_2015 = count_courses_per_years(udemy, 'Web Development', 2015)
wd_2016 = count_courses_per_years(udemy, 'Web Development', 2016)
wd_2017 = count_courses_per_years(udemy, 'Web Development', 2017)


# In[ ]:


# <Amount of Business Finance courses were published in each year>
bf_2011 = count_courses_per_years(udemy, 'Business Finance', 2011)
bf_2012 = count_courses_per_years(udemy, 'Business Finance', 2012)
bf_2013 = count_courses_per_years(udemy, 'Business Finance', 2013)
bf_2014 = count_courses_per_years(udemy, 'Business Finance', 2014)
bf_2015 = count_courses_per_years(udemy, 'Business Finance', 2015)
bf_2016 = count_courses_per_years(udemy, 'Business Finance', 2016)
bf_2017 = count_courses_per_years(udemy, 'Business Finance', 2017)


# In[ ]:


# <Amount of Graphic Design courses were published in each year>
gd_2011 = count_courses_per_years(udemy, 'Graphic Design', 2011)
gd_2012 = count_courses_per_years(udemy, 'Graphic Design', 2012)
gd_2013 = count_courses_per_years(udemy, 'Graphic Design', 2013)
gd_2014 = count_courses_per_years(udemy, 'Graphic Design', 2014)
gd_2015 = count_courses_per_years(udemy, 'Graphic Design', 2015)
gd_2016 = count_courses_per_years(udemy, 'Graphic Design', 2016)
gd_2017 = count_courses_per_years(udemy, 'Graphic Design', 2017)


# In[ ]:


# <Amount of Musical Instruments courses were published in each year>
mi_2011 = count_courses_per_years(udemy, 'Musical Instruments', 2011)
mi_2012 = count_courses_per_years(udemy, 'Musical Instruments', 2012)
mi_2013 = count_courses_per_years(udemy, 'Musical Instruments', 2013)
mi_2014 = count_courses_per_years(udemy, 'Musical Instruments', 2014)
mi_2015 = count_courses_per_years(udemy, 'Musical Instruments', 2015)
mi_2016 = count_courses_per_years(udemy, 'Musical Instruments', 2016)
mi_2017 = count_courses_per_years(udemy, 'Musical Instruments', 2017)


# In[ ]:


graph_data = [
    [wd_2011, wd_2012, wd_2013, wd_2014, wd_2015, wd_2016, wd_2017],
    [bf_2011, bf_2012, bf_2013, bf_2014, bf_2015, bf_2016, bf_2017],
    [mi_2011, mi_2012, mi_2013, mi_2014, mi_2015, mi_2016, mi_2017],
    [gd_2011, gd_2012, gd_2013, gd_2014, gd_2015, gd_2016, gd_2017],
]


# In[ ]:


annual_increase = plt.figure()
axes1 = annual_increase.add_axes([0, 0, 1.6, 0.9])
axes2 = annual_increase.add_axes([0, 1.1, 1.6, 0.5])

axes2.plot(years, courses_amount, lw = 3, marker = 'o')
axes1.plot(years, graph_data[0], label = 'Web Development', lw = 3, marker = 'o')
axes1.plot(years, graph_data[1], label = 'Business Finance', lw = 3, marker = 'o')
axes1.plot(years, graph_data[2], label = 'Musical Instruments', lw = 3, marker = 'o')
axes1.plot(years, graph_data[3], label = 'Graphic Design', lw = 3, marker = 'o')
axes1.legend()
axes1.set_title("Annual increase in subject courses")
axes2.set_title('General annual increase in courses')


# In[ ]:


udemy['published_timestamp'].sort_values(ascending = False).head(1)


# In[ ]:


udemy['published_timestamp'].sort_values().head(1)


# In the data we have that the first course was published in the middle of 2011 and the last one was published in the middle of the 2017 year, so the statistic for these two years isn't clear to analyze it. We only can see that between 2011 and 2012 the amount of courses increase slowly and evenly, there is no big difference between subjects. The general annual graph also confirms that fact. This may be due to the fact that in 2011 online learning platforms weren't as popular.
# 
# We can see that in 2012 Business Finance courses start to gain popularity and grows steadily for the next 3 years, and then in 2015 growth slows down. And also it's seen that the amount of Web Development courses stats to increase at the same time with Business Finance, but not so fast at first. Just in 2014 Web Development increases dramatically and going up very fast. Apparently, that growth is going on in 2017 too, but we can't affirm it for sure due to lack of data.
# 
# And more creative directions Music and Design also started to increase faster since 2012-2013. But overall their popularity is twice as slow as the first two.

# If you are here reading this - thank you very much! It was my first real analysis of real data with python. Which was a very interesting experience for me. I'm sure that some things I've done in the research could be written in a much shorter way. But I'm just learning :)
# 
# So, any comments and advices are expectable. Thanks :)
