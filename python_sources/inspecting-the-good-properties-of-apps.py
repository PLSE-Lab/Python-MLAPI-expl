#!/usr/bin/env python
# coding: utf-8

# **Overview**
# 
# Could you predict which applications are being downloaded by people ?And how many of them appeals to teenagers,how many of them appeals to matures or everyone ?What are installation numbers and ratings ?If we have a good analysis of applications that are downloaded by people everyday then we can produce much better applications.Currently,application stores are already classifying and suggesting the new applications according to user's interest and before we download some applications we can see the previous experiences of people by investigating and observing the ratings values.With all of these properties that we can keep track of peoples interests can be predicted by analiysts and can be designed in a more effective way by developers.
# 
# Thus,we need to build a machine learning model to find the properties of best appplications.Or, we can say that we'll be able to know which properties should our application has while we creating a new one.
# 
# 
# **About the Dataset**
# 
# We'll be working with two csv file in this dataset.The first dataset includes application name,category,reviews,sizes,install,types,android version etc. it's like more technical details...And the second data set includes sentiment,sentiment polarity,sentiment subjectivity etc.Emotions are closely related to sentiments. The strength of a sentiment or opinion is typically linked to the intensity of certain emotions, e.g., joy and anger.Polarity, also known as orientation is he emotion expressed in the sentence. It can be positive, neagtive or neutral. Subjectivity is when text is an explanatory article which must be analysed in context.Literally...Let's implement them into applications.
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let us first import the needed csv files from the dataset bwith the belowing code.

# In[ ]:



data2=pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')
data=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')


# Here is an example image of the data from the Google play store applications.Whenever you want to download an application you visit this part to get feedbacks from other peoples and see ratings.

# In[ ]:


from IPython.display import Image
import os
get_ipython().system('ls ../input/')

Image("../input/images1/rating-system.jpg")


# The data famework holds the App,Category,Rating,Reviews,Size,Installs and so on...We can check these categories and types we can use data.info() command. 

# In[ ]:


data.info()


# To see which types are exist in the Type column we use the command data.Type.unique().It returns the unique column indexes.

# In[ ]:


print( data.Type.unique() )

print( data.Category.unique() )

data.describe()


# Here in the last line we use data.describe(),it returns the result of just the numerical data.And below you can see the name data2.We use it to hold the content of the second csv file which inludes the sentiment polarity and subjectivity values.

# In[ ]:


print( data2.info() )

print( data2.Sentiment.unique())

data2.describe()


# The term "**correlation**" refers to a mutual relationship or association between quantities. In almost any business, it is useful to express one quantity in terms of its relationship with others. For example, sales might increase when the marketing department spends more on TV advertisements, or a customer's average purchase amount on an e-commerce website might depend on a number of factors related to that customer.In the first dataset we just have one numerical data colum so let's investigate it in the second data set.

# In[ ]:


data.corr()


# **Correlation** can have a value:
# 
# **1** is a perfect positive correlation,
# **0** is no correlation (the values don't seem linked at all),
# **-1** is a perfect negative correlation.
# The value shows how good the correlation is (not how steep the line is), and if it is positive or negative.So when we observe the values of the dataframe2 we see there are two perfect positive correlation between the same groups( it must be ).And between Sentiment polarity and Sentiment Subjectivity the correlation value is weak,approximately 0.27 .

# In[ ]:


data2.corr()


# In[ ]:



f,ax=plt.subplots( figsize=(5,5))
sns.heatmap( data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

plt.show()


# In[ ]:


f,ax=plt.subplots( figsize=(5,5))
sns.heatmap( data2.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)
plt.show()


# To see the content of the files we use head() function with parameter 10.It gives us the first 10 rows of the file from the starting of the file.

# In[ ]:


data.head(10)


# In[ ]:


data2.head(10)


# Also we can see the names of the columns with the command data.columns  .

# In[ ]:


print( data.columns+'\n' )


# In[ ]:


data2.columns


# In the following graph we see the rating values of the applications with the plot line.We see they have at most 5.0 as a value.

# In[ ]:


plt.figure( figsize=( 10,10))
data.Rating.plot( kind='line' ,color='green' ,label='Rating' ,linewidth=1,alpha=0.5,grid=True,linestyle=':')
plt.legend(loc='upper right',labelspacing=0.5)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line plot of Ratings')


# An in the following graph again we've used line plot, but this time we have two column type of float.They are called Sentiment Polarity and Sentiment Subjectivity.We observe that Sentiment Subjectivity values are intense at the above of the zero and we observe that the Sentiment Polarity values are under the zero.

# In[ ]:


plt.figure( figsize=( 10,10))
data2.Sentiment_Polarity.plot( kind='line' ,color='pink' ,label='Sentiment_Polarity' ,linewidth=1,grid=True,linestyle=':')
data2.Sentiment_Subjectivity.plot( kind='line' ,color='purple' ,label='Sentiment_Subjectivity' ,alpha=0.5,linewidth=1,grid=True,linestyle=':')
plt.legend(loc='upper right',labelspacing=0.5)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line plot of Ratings')


# Here we see Application ratings and some of their names with the horizontal bar graph.

# In[ ]:



#plt.figure( figsize=( 20,20)) also we can use this before plotting the barh
fig, ax = plt.subplots(figsize=(20, 10)) 

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
              ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(10)

plt.barh( data.App[:15],data.Rating[:15],align='center',orientation='horizontal',color='pink',edgecolor='purple',linewidth=2)

plt.title("Application-Rating")
plt.xlabel("Ratings")
plt.ylabel("Applications")
plt.legend()
plt.show()


# In the following piece of code first we classify the Content Rating types.And then we plotted the scatter graph with the dimensions Rating and Reviews.Reviews were initially in the type of object(str) but we converted it float with the to_numeric() method.We can say that Mature 17+ class gives better reviews but less values at the ratings than others( with green spots at the graph ).And the best ratings and best reviews are coming from the teenagers.

# In[ ]:


data_updated=data.rename( index=str ,columns={"Content Rating":"Content_Rating"})

print( data_updated.Content_Rating.unique())
plt.figure( figsize=( 10,10))


content_rating_Everyone=data_updated[ data_updated.Content_Rating	=='Everyone' ]
content_rating_Teen=data_updated[ data_updated.Content_Rating=='Teen' ]
content_rating_Every10=data_updated[ data_updated.Content_Rating=='Everyone 10+' ]
content_rating_Mature17=data_updated[ data_updated.Content_Rating=='Mature 17+' ]
content_rating_Adults=data_updated[ data_updated.Content_Rating=='Adults only 18+' ]
content_rating_Unrated=data_updated[ data_updated.Content_Rating=='Unrated' ]

plt.scatter( content_rating_Everyone.Rating, pd.to_numeric(content_rating_Everyone.Reviews),color='red',label='EveryOne')
plt.scatter( content_rating_Teen.Rating, pd.to_numeric(content_rating_Teen.Reviews),color='yellow',label='Teen')
plt.scatter( content_rating_Every10.Rating, pd.to_numeric(content_rating_Every10.Reviews),color='pink',label='Everyone 10+')
plt.scatter( content_rating_Mature17.Rating, pd.to_numeric(content_rating_Mature17.Reviews),color='green',label='Matue 17+')
plt.scatter( content_rating_Adults.Rating, pd.to_numeric(content_rating_Adults.Reviews),color='blue',label='Adults only 1+')
plt.scatter( content_rating_Unrated.Rating, pd.to_numeric(content_rating_Unrated.Reviews),color='grey',label='Unrated')
plt.legend()
plt.show()


# We have the types Free,Paid,nan,0 as a type of the applications.We'have found the number of each types of applications and plotted a basic bar plot.

# In[ ]:


plt.figure( figsize=( 10,10) )


x=data['Type']

f=0
p=0
n=0
z=0

types=['Free', 'Paid', 'nan', '0']


for each in x:
    if( each=='Free'):
       f+=1
    if( each=='Paid'):
       p+=1
    if( each=='\0' ):
       n+=1
    if( each=='0' ):
       z+=1
numbers=np.array([f,p,n,z])    

plt.bar( types,numbers,color='pink',edgecolor='purple' )
plt.title("bar plot")
plt.xlabel("Types")
plt.ylabel("Numbers")
plt.show()


# As we can see also in the dataset we have three types of Sentiments.In the grapgh below we can see the polarity and subjectivity of the positive,neutral and nan Sentiment's classification.In the positive Sentiments we observe the less sentiment polarity and much sentiment subjectivity.And in the neutral sentiments the polarity is 0.0 and they are taking the different values at the subjectivity. 

# In[ ]:


plt.figure( figsize=( 10,10) )

positive=data2[ data2.Sentiment=='Positive' ]
neutral=data2[ data2.Sentiment=='Neutral' ]
nan=data2[ data2.Sentiment=='NaN' ]

plt.scatter( positive.Sentiment_Polarity,positive.Sentiment_Subjectivity,color='green',label='positive',alpha=0.5,linewidths=0.01,norm=0.5 )
plt.scatter( neutral.Sentiment_Polarity,neutral.Sentiment_Subjectivity,color='grey' ,label='neutral',alpha=0.2,linewidths=0.001,norm=0.5)
plt.scatter( nan.Sentiment_Polarity,nan.Sentiment_Subjectivity,color='black' ,label='Nan')

plt.legend()
plt.xlabel('Sentiment Polarity')
plt.ylabel('Sentiment Subjectivity')

plt.title("Classification of Positive-Neutral-Nan Sentiments")
plt.show()



# In[ ]:


data_new=data[np.logical_and(data['Rating']==5.0 , data['Category']=='MEDICAL')]
data_new2=data_new[ np.logical_and( data_new['Type']=='Paid',data_new['Reviews']=='2')]
data_new2

       


# In[ ]:


data2[ (data2['Sentiment']=='Neutral')  & (data2['Sentiment_Subjectivity']>=1 ) & ( data2['Translated_Review']=='I downloaded only, well I know')]

