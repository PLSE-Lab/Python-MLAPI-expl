#!/usr/bin/env python
# coding: utf-8

# # DATA ANALYSIS OF AMAZON'S ECHO REVIEWS 
# ### (Data set obtained from Kaggle.com)  
# #### @MoisesBarbera  //   https://www.github.com/MoisesBarbera   //   https://www.kaggle.com/moisesbarbera  // https://www.linkedin.com/in/moises-barbera-ramos-8a3848164/
# #### A coverage of:
# #### ---- Extraction of general information pre-analysis; 
# #### ---- Data cleaning;
# #### ---- Data imaging/visualization; 
# #### ---- Creation of new data sets;
# #### ---- Feature engineering

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


Kaggle=1
if Kaggle==0:
    df_alexa1 =pd.read_csv("amazon_alexa.tsv",sep="\t")
else:
    df_alexa1 = pd.read_csv("../input/amazon_alexa.tsv",sep="\t")


# In[ ]:


df_alexa1.head()


# In[ ]:


df_alexa1.tail()


# In[ ]:


df_alexa1.info() #Obtaining general information about the dataset we are using and checking there is no missing information with non-null


# In[ ]:


df_alexa1.describe() #displaying other relevant information from the dataset, the number of elemnts on it the values and percentages of the feedback, to know 1 as positive feedback and 0 as negative feedbak. Highlighting the mean on rating since that's the overall opinion of costumers about the product.


# In[ ]:


df_alexa1.keys() #Obtaining the headers of the specific information provided from the dataset


# In[ ]:


df_alexa1['verified_reviews'].tail() #Obtaining the reviews from real customers to later on analyse what are their most common opinions on the product


# ## Cleaning relevant data to guarantee proper analysis

# #### On the data provided by "Kaggle.com" for the study of the reviews on the Amazon Echo devices we have spotted a set of data that provides information about the Amazon Fire TV Stick which is not an Alexa eenabled device and hence, every piece of information on the over 3000 rows of data regarding this specific object have to be removed.

# In[ ]:


df_alexa = df_alexa1[~df_alexa1.variation.str.contains('Fire')]


# In[ ]:


df_alexa.describe()


# #### As seen, now the data has been reduced in over 300 elements, from 3159 to 2800, ensuring that the visualization of products will be conducted only for those Amazon Echo with alexa enabled. And impotant to reindex afterwards to produce an accurate analysis.

# In[ ]:


df_alexa = df_alexa.reset_index(drop=True) #This way we reindex the whole dataframe from 0 to 2799 as for all 2800 elements on the dataset.


# ## Visualisation of data

# #### Analysing when have the reviews been written during the week we can obtain information about when is it more common for users to order the product. Considering shipment from Amazon takes as low as 1-2 days and users usually write the review after after 12 hours of use since Amazon keeps track of users satisfaction by asking for reviews by email after 12-24 hours from the product deliery.

# In[ ]:


df_alexa['date'] = pd.to_datetime(df_alexa['date'], errors='coerce') #Enable access to datetime programming features


# In[ ]:


weekday_ratings = df_alexa['date'].dt.weekday_name.value_counts()
weekday_ratings = weekday_ratings.sort_index()
sns.barplot(weekday_ratings.index, weekday_ratings.values, order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
plt.xticks(rotation='vertical')
plt.ylabel('Count')

plt.show()


# #### Observing the graph and with the assumption of the majority of reviews been written after 12 hours from delivery we can deduce that most products have been delivered on Monday which implies that the majority of Alexa Echo products were bought during the weekend. This information suggests that there is a marketing action during that period and uses the free time users might have during their weekends as an advantage.

# In[ ]:


positive = df_alexa[df_alexa['feedback'] == 1] # From "df_alexa" visualise the positive reviews
positive


# In[ ]:


positive['feedback'].count() #providing the number of positive reviews


# In[ ]:


negative = df_alexa[df_alexa['feedback'] == 0]  # From "df_alexa" visualise the negative reviews
negative


# In[ ]:


negative['feedback'].count() #providing the number of negative reviews


# In[ ]:


sns.countplot(df_alexa['feedback'], label = 'count') #visualizing how many of those reviews were positive (1) and how many negative (0). In accordance with our expectations.


# In[ ]:


sns.countplot(df_alexa['rating'], label = 'count') #visualizing the satisfaction of the clients though the 5 stars rating


# #### Creating a small and easy to do table to find the exact number of counts for each rating.

# In[ ]:


five_star = df_alexa[df_alexa['rating'] == 5]  
four_star = df_alexa[df_alexa['rating'] == 4]
three_star = df_alexa[df_alexa['rating'] == 3]
two_star = df_alexa[df_alexa['rating'] == 2]
one_star = df_alexa[df_alexa['rating'] == 1]

df_ratings = pd.DataFrame({'rating' : ['5', '4', '3', '2', '1'],
                           'count' : [five_star['rating'].count(), four_star['rating'].count(),
                                      three_star['rating'].count(), two_star['rating'].count(),
                                      one_star['rating'].count()]})
df_ratings


# #### Knowing the number of counts for each rating from the table above we can then obtain their percentage.

# In[ ]:


print('5 Star Rating = {0} %'.format((2004/2800)*100))
print('4 Star Rating = {0} %'.format((421/2800)*100))
print('3 Star Rating = {0} %'.format((146/2800)*100))
print('2 Star Rating = {0} %'.format((81/2800)*100))
print('1 Star Rating = {0} %'.format((148/2800)*100))


# #### Since there are several configurations for this product, we can annalyse them to find the most popular ones as well as other features

# In[ ]:


plt.figure(figsize = (30,10))

sns.barplot(x = 'variation', y = 'rating', data = df_alexa, palette = 'deep')


# 

# In[ ]:


plt.figure(figsize = (30,10))

df_alexa.variation.value_counts().plot(kind='bar', fontsize=18)


# 1. #### Knowing the most commonly boght models of this product we can predict the future successful colour options and then produce a range of models and quantities in accordance to that. It would make little sense to produce as many "Walnut Finish" configurations (right hand side of the graph) as "Black Dot" configurations (left hand side of the graph) since the first one barely sold any compared to the Black Dot model which sold over 50 times more units.

# ### Word Cloud

# In[ ]:


df_alexa['verified_reviews'].iloc[:5]  #All reviews from real customers on the Amazon echo devices (excluding the already deleted Fire TV stick information) / Only the first 5 elements shown to gain space on desktop (displaying all of them would take longer for the program to upload and would add no useful information) also, improving the comparison with other parts of the code.  


# In[ ]:


for i in df_alexa['verified_reviews'].iloc[:5]: #A more elegant way to visualize the reviews  / Only the first 5 elements shown to gain space on desktop (displaying all of them would take longer for the program to upload and would add no useful information) also, improving the comparison with other parts of the code.  
    print(i, '\n')


# In[ ]:


words = df_alexa['verified_reviews'].tolist() #dataframe selected transformed into different strings
words[:5]  # Only the first 5 elements shown to gain space on desktop (displaying all of them would take longer for the program to upload and would add no useful information) also, improving the comparison with other parts of the code.  


# In[ ]:


words_as_one_string = ' '.join(words) #dataframe selected transformed into one single string
words_as_one_string[:500] # Only a certain number of character shown to gain space on desktop (displaying all of them would take longer for the program to upload and would add no useful information) also,  improving the comparison with other parts of the code.  


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


plt.figure(figsize = (15,15))
plt.imshow(WordCloud().generate(words_as_one_string), interpolation = 'bilinear') #interpolation = 'bilinear' increases the quality of the image
plt.title("Averall words used on all reviews", fontsize=40)


# #### Now that we know the most commons used by all users of this product, we now want to study those reviews with a bad feedback so the most common wword will be the features to improve and create a better product in the future.

# In[ ]:


df_bad_alexa = df_alexa[df_alexa['feedback'] == 0] #Creating a dataset with the bad reviews on it only.

bad_words = df_bad_alexa['verified_reviews'].tolist() #dataframe selected transformed into different strings
bad_words_as_one_string = ' '.join(bad_words) #dataframe selected transformed into one single string
plt.figure(figsize = (15,15))
plt.imshow(WordCloud().generate(bad_words_as_one_string), interpolation = 'bilinear')

plt.title("Averall words used on negative reviews", fontsize=40)


# #### After neglecting the common words by which the device is refered to as Amazon, Alexa, dot, echo and product; we can see how there is a big enphasis on the words  "speaker", "music" and "sound" suggesting, since these are the negative reviews, there should be an improvement on the speakers of this product to generate a better sound experience for the user.

# ## Feature Engineering

# #### To provide better decisiontaking about the product we want to convert the comments from the audience into words to produce quantitative conclusions.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
alexa_CountVectorizer = vectorizer.fit_transform(df_alexa['verified_reviews'])


# In[ ]:


alexa_CountVectorizer.shape #The reviews are now structured in a structured array.


# In[ ]:


print(vectorizer.get_feature_names()) #every single word that has been mentioned on the reviews.


# In[ ]:


print(alexa_CountVectorizer.toarray())


# In[ ]:


word_count_array = alexa_CountVectorizer.toarray()


# In[ ]:


word_count_array[0,:] #Obtain the first review (first row and all columns)


# In[ ]:


plt.plot(word_count_array[0,:]) #All points are at 0 but three peaks, corresponding to the three words used on this 1st review 
df_alexa['verified_reviews'][0] #displaying the first review on screen to check the 3 words used by this user. 


# In[ ]:


plt.plot(word_count_array[3,:]) #Some points are now at 0 but some peaks appear now, corresponding to the total number of words used on this 4th review 
df_alexa['verified_reviews'][3] #displaying the 4th review on screen to check the words used by this user. 


# In[ ]:


plt.plot(word_count_array[13,:]) #All points are at 0 but 1 peaks, corresponding to the three words used on this 3rd review since it is the same word. 
df_alexa['verified_reviews'][13] #displaying the 3rd review on screen to check the 3 words used by this user, as  they are the same word, only one peak is shown. 


# #### Each peak on some position on the x axis corresponds to a word. Always been that word the same for the same peak.

# In[ ]:


df_alexa['length'] = df_alexa['verified_reviews'].apply(len)
df_alexa.head() #This process gives a value to the total number of charachters used in the description.


# #### We can use this information of the number of characters each user has used to review the product to also study their level of satisfaction. For instance, happy costumers tend to writte less words than those who are unhappy who usually write longer complaints. 

# In[ ]:


df_alexa['length'].hist(bins = 100) 
plt.xlabel('Number of characters used')
plt.ylabel('Users')


# #### The curve shows that a higher number of users have written a shorter number of words to describe the product. Selecting those comments with a maximum number of characters will provide the company more information about what they are doing well and what they are doing bad in order to improve their product while getting an inshight on what they are doing well to keep exploring further on.

# In[ ]:


min_char = df_alexa['length'].min()
df_alexa[df_alexa['length'] == min_char] ['verified_reviews'].iloc[0]


# In[ ]:


max_char = df_alexa['length'].max()
df_alexa[df_alexa['length'] == max_char] ['verified_reviews'].iloc[0]


# ## Conclusions
# 
# #### This analysis has shown relevant data spoing how most of the purchase of this product have been produced on weekends, information that can be used to improve the marketing action on this product as well as the need of enough delivery personel available during that period to deliver the increased amount of requested products on time.  
# #### Also, analysing the good and bad reviews we spot the love for this product, with over 70% of five star rating, but there is a need to improve the sound system of the device as it has been the most common word used on negative reviews.
# #### The study of the most commonly ordered configurations prepares the company to predict the most succeful color options for the future and act according to this information. Also, knowing the most ordered colours and the less ordered ones can earn money to the company by predicting how many units of each configuration should be produced. In this case, the Black dot option has been a huge success, so this model should be produced in larger quantities compared to the Walnut Finish wich, despite having good reviews, is still the least requested configuration.

# #### @MoisesBarbera  //   https://www.github.com/MoisesBarbera   //   https://www.kaggle.com/moisesbarbera // https://www.linkedin.com/in/moises-barbera-ramos-8a3848164/

# In[ ]:




