#!/usr/bin/env python
# coding: utf-8

# My third notebook! I have to admit that I find the plot function from Pandas very convenient for basic plots. Seaborn was pretty interesting with more visualizations, and had options to add in colors (although with some extra work, I'm pretty sure it'd be possible in matplotlib)
# 
# This time I'm trying out Plotly mostly because it allows for creating more interactive graphs. This one sure has too many options to configure, so I'm still finding my way around the libraries and options. I also found that Plotly requires me to define properties as dictionaries or manually as properties (something I didn't think of or even pass by in the other libraries)
# 
# This is also my first notebook involving reviews or languages. I first thought I'd use the NLTK library but realized that I'd have to train the models myself. Since I'm just getting into NLP, I decided to use TextBlob which is built on top of NLP.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import re
# trying my hands on plotly
from plotly import tools
import plotly.graph_objs as go
# TextBlob for quick text analysis
from textblob import TextBlob
# wordcloud for, well like the name suggests
from wordcloud import WordCloud, STOPWORDS
# matplotlib to visualize the word clouds
import matplotlib.pyplot as plt
# To generate word clouds using a mask object. I might not use this
from PIL import Image
# Wordcloud usually generates in various colors. this library might help to convert to a gray color
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Reading the Data**
# 
# The usual steps, read the data and then remove special characters from the column headers to make it easier to access

# In[ ]:


# Read the data
employee_reviews=pd.read_csv("../input/employee_reviews.csv", index_col=[0])
# Clean up the column headers
employee_reviews.columns = employee_reviews.columns.str.replace("-","_")

replace_none_as_nan = lambda t: float('nan') if t=="none" else t
employee_reviews.overall_ratings = employee_reviews.overall_ratings.map(replace_none_as_nan)
employee_reviews.work_balance_stars = employee_reviews.work_balance_stars.map(replace_none_as_nan)
employee_reviews.culture_values_stars = employee_reviews.culture_values_stars.map(replace_none_as_nan)
employee_reviews.carrer_opportunities_stars = employee_reviews.carrer_opportunities_stars.map(replace_none_as_nan)
employee_reviews.comp_benefit_stars = employee_reviews.comp_benefit_stars.map(replace_none_as_nan)
employee_reviews.senior_mangemnet_stars = employee_reviews.senior_mangemnet_stars.map(replace_none_as_nan)

# A quick view of the data to get the lay of the land, also because Excel has spoiled me that way
employee_reviews.head(5)


# **The Data**
# 
# The columns in this are pretty much expected. We have the company, location, reviews, and also a rating

# In[ ]:


# Get the list of reviews per company
employee_companies = employee_reviews.company.value_counts()
# Prepare a Bar graph of the number of records
data = [go.Bar(
    x=employee_companies.index.str.title().tolist(),
    y=employee_companies.values
)]
layout = go.Layout(title="# Reviews by Organization",yaxis=dict(title="# Reviews"), xaxis=dict(title="Organizations"))
figure = go.Figure(data=data, layout=layout)
iplot(figure)


# **Reviews per Company**
# 
# Learning from my past experiences, I learnt that having more data means that the inferences would be pretty accurate.
# 
# In this case, there's way too many Amazon, Microsoft, and Apple Reviews, and comparitively less data for Google, Facebook, and Netflix, I'm not sure how I'd be able to compare company reviews. Netflix does have around 800 reviews, but it' 800 netflix reviews against 26000 amazon reviews.
# 
# I know normalization would help for varying data, but what should be done when the data is already normalized (take the ratings for instance), but the number of samples are off by a very large margin (800 for netflix against the others)
# 
# I tried looking up how to handle visualizations for data of unequal sizes. I came up empty handed; Maybe there's a term for this that I don't know. Maybe I could randomly select reviews for all companies based on the company with the least count? or maybe I'm overthinking this by thinking of machine learning vs data analysis?

# **Polarity and Subjectivity**
# 
# TextBlob provides a polarity number between -1 to +1, with -1 indicating very negative and +1 indicate very positive. It also provides a subjectivity score between 0 and 1 with 0 being least subjective and 1 being very subjective.
# 
# So I generate the polarity and subjectivity to the summary. Not very helpful when it comes to the Pros or Cons because the statements speak for themselves

# In[ ]:


# Generate polarity and subjectivity number for just the summary column
employee_reviews["summary_polarity"] = employee_reviews.summary.apply(lambda t: TextBlob(str(t)).sentiment.polarity)
employee_reviews["summary_subjectivity"] = employee_reviews.summary.apply(lambda t: TextBlob(str(t)).sentiment.subjectivity)


# In[ ]:


# A short function to generate histograms for the polarity and subjectivity for each company. I could repeat the statements over and over, but that would make the graph generation appear too large
def generate_histogram(for_column,company_name,opacity=0.5):
    '''
    returns a plotly Histogram object with the parameters specified
    
    for_column: Specify the columns with which the histogram must be generated. In this case, it would be either "summary_polarity" or "summary_subjectivity"
    
    company_name: Specify the company name in lower case. In this case, it would be one of the following: amazon, apple, facebook, google, microsoft, or netflix
    
    opacity: the opacity of each hisogram visualization. By default it will be 0.5 or 50% opaque
    '''
    return go.Histogram(
        x = employee_reviews[employee_reviews.company==company_name][for_column],
        opacity=opacity,
        xbins=dict(start=-1.0,end=1.1,size=0.2),
        name=company_name.title()
    )


# In[ ]:


# Generate a Histogram if Polarity for each company.

amazon_polarity=generate_histogram("summary_polarity","amazon")
apple_polarity=generate_histogram("summary_polarity","apple")
google_polarity=generate_histogram("summary_polarity","google")
facebook_polarity=generate_histogram("summary_polarity","facebook")
microsoft_polarity=generate_histogram("summary_polarity","microsoft")
netflix_polarity=generate_histogram("summary_polarity","netflix")
data=[
    amazon_polarity,
    apple_polarity,
    google_polarity,
    facebook_polarity,
    microsoft_polarity,
    netflix_polarity
]
layout = go.Layout(barmode="overlay", title="Summary Polarity by Company",xaxis=dict(title="Polarity"),yaxis=dict(title="# of Reviews"))
figure = go.Figure(data=data,layout=layout)
iplot(figure)


# **Polarity of Summary per company**
# 
# Never thought that a title would rhyme, but I digress. 
# 
# The summaries for each company is mostly neutral to slightly positive. There are a few negative reviews, but the positives appear to outweigh the negatives.
# 
# Going back to the sample size problem that I've encountered, there's way too many reviews for certain companies. This has suppressed the smaller numbers. For instance, there's 382 summaries for Amazon that are very negative and 1181 reviews that are very positive.
# 
# **Note to Self:** If I took the company with the least reviews, and then took the same amount of random samples from the other companies, It'd be more uniform but I'm guessing it wouldn't make much of a difference (since it's a distribution). But if I were generating a bar graph of the ratings and didn't consider the # of reviews per company, then the comparitively large number of reviews for Amazon would indicate that it is probably the best place to work as opposed to the other companies

# In[ ]:


# Generate Subjectivity Histogram for each company
amazon_subjectivity=generate_histogram("summary_subjectivity","amazon")
apple_subjectivity=generate_histogram("summary_subjectivity","apple")
google_subjectivity=generate_histogram("summary_subjectivity","google")
facebook_subjectivity=generate_histogram("summary_subjectivity","facebook")
microsoft_subjectivity=generate_histogram("summary_subjectivity","microsoft")
netflix_subjectivity=generate_histogram("summary_subjectivity","netflix")
data=[
    amazon_subjectivity,
    apple_subjectivity,
    google_subjectivity,
    facebook_subjectivity,
    microsoft_subjectivity,
    netflix_subjectivity
]
layout = go.Layout(barmode="overlay", title="Summary Subjectivity by Company",xaxis=dict(title="Subjectivity"),yaxis=dict(title="# of Reviews"))
figure = go.Figure(data=data,layout=layout)
iplot(figure)


# **Subjectivity of Summary per company**
# 
# Not much here other than inferring that the summaries are mostly "objective" and a few of them could be either objective or subjective (subjectively neutral maybe?)
# 
# Similar to Polarity, identifying the company with the fewest reviews and taking an equal amount of random samples from all of them might make the visualization a bit more uniform, but with this being a histogram, it wouldn't skew the results by a very large margin

# In[ ]:


# Slightly larger function to generate wordclouds for the summary, pros, and cons.

# configure stop words aka words that dont need to be considered for word clouds
stopwords=set(STOPWORDS)
stopwords.add("let")
stopwords.add("to")
stopwords.add("from")
stopwords.add("a")
stopwords.add("an")
stopwords.add("the")
stopwords.add("of")

# The function itself
def generate_wordcloud(reviews, generate_by_frequency=False, addl_stopwords=[],):
    '''
    return a word cloud object that can be used in a matplotlib.pyplot's imshow function
    
    reviews: specify the entire employee_object for which the word cloud needs to be generated. Filtering must be done manually
    
    generate_by_frequency: default False. Configure whether the word cloud must be generated using a text string (WordCloud.generate_by_text) or if the word cloud must be generated using frequencies (WordCloud.generate_by_frequencies). See wordcloud documentation for further reference
    
    addl_stopwords: array of additonal words that must be added to the list of stop words
    '''
    
#   Add the additional stopwords to the stopword list
    for t in addl_stopwords:
        stopwords.add(str(t))

#   Combine all the reviews in the review series so that it becomes one really large text that can be passed to the wordcloud's generate function
    def format_reviews(review):
        processed_reviews = " ".join(str(t) for t in review)
        processed_reviews = processed_reviews.replace("\,+"," ").replace("\.+"," ").replace("\*+"," ").replace("\n+", " ")
        return processed_reviews
    
#   Function to generate the word cloud words in gray color instead of various colors
    def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
        return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)
    
#   Initialize the WordCloud object
    wc = WordCloud(background_color="black", max_words=100, stopwords=stopwords, max_font_size=40, random_state=42,width=600,height=200,margin=1)
    if generate_by_frequency:
        w_counts = TextBlob(format_reviews(reviews)).word_counts
        for t in stopwords:
            if t in w_counts:
                del w_counts[t]
        wc.generate_from_frequencies(w_counts)
    else:
        wc.generate_from_text(format_reviews(reviews))
    wc.recolor(color_func=grey_color_func, random_state=3)
    return wc


# **Onto the word clouds**
# 
# The first thing that comes to mind when I think of text data is word clouds. In time, my first thoughts might change to NLP or Text generation or Bag of Words or Tokenization or something on those lines.
# 
# WordCloud generates a word cloud using a text or frequencies. generating from text automatically removes the default stopwords, while generating from frequencies requires us to manually remove the text. At first I tried generating form text (since it was the easy way out) and found that some words were repeated. Resoluted to generating from frequencies (had to manually remove stopwords but was worth it)

# In[ ]:


fig, ax = plt.subplots(nrows=6,ncols=3,figsize=(36,36))

# For each company name, generate a wordcloud of the company's summary, pros, and cons (in that order) and display the visualizations
for i in np.arange(employee_reviews.company.nunique()):
    company_name=employee_reviews.company.unique()[i]
    summary=generate_wordcloud(employee_reviews[employee_reviews.company==company_name].summary, addl_stopwords=[company_name], generate_by_frequency=True)
    pros=generate_wordcloud(employee_reviews[employee_reviews.company==company_name].pros, addl_stopwords=[company_name], generate_by_frequency=True)
    cons=generate_wordcloud(employee_reviews[employee_reviews.company==company_name].cons, addl_stopwords=[company_name], generate_by_frequency=True)
    ax[i,0].set_title("{0} Summary".format(company_name.title()),fontsize=36)
    ax[i,1].set_title("{0} Pros".format(company_name.title()),fontsize=36)
    ax[i,2].set_title("{0} Cons".format(company_name.title()),fontsize=36)
    ax[i,0].imshow(summary, interpolation='bilinear')
    ax[i,1].imshow(pros, interpolation='bilinear')
    ax[i,2].imshow(cons, interpolation='bilinear')
    ax[i,0].axis("off")
    ax[i,1].axis("off")
    ax[i,2].axis("off")


# **Word clouds for Summary, Pros, and Cons**
# 
# I realize after running the visualization that this is overkill. 18 visualizations in total and so many ways this can be arranged! 
# * Run 1 visualization for everything
# * Run everything in 1 visualization
# * Run visualizations of pros, cons, and summary and group by company 
# * Run visualizations by company and group by summary, pros, and cons
# 
# There might be other ways of running this, but I'll go with running 1 visualization for everything
# 
# Anyway, all company summaries have common words like *good, great, place, work*. So I guess it's great working for all of them. There are some words which are more frequent in some companies than the others. Like *Software* for Google and Microsoft for instance, or *Engineer* for Microsoft and Facebook, and so on. 
# 
# Pros list words like *benefits, environment, culture*. Sure working for one of the big companies does stand out among peers, on paper and(or) in conversation. This is backedup with the pros. I guess people are happier with the perks of the job (*perks* also mentioned in the pros).
# 
# When it comes to cons, I'd like to think that people don't find it as easy as they expected; aka the word *hard*, which is found in all visualizations but not the most common. *Managers* and *Management*, on the other hand, are most commonly mentioned in the cons for each company. Interestingly, the word *work* is found in the Pros and Cons for each company. Maybe working there is like cutting with a double edged blade? (Probably not, but who am I to comment on companies I haven't worked for)
# 
# Looking at the notes, I'd like to make my own guess that the work becomes too difficult, or that there's differences of opinion/ideas between employee and management.

# **The Ratings**
# 
# The ratings are mostly complete numbers, if not complete, they're definite values like a 3 or a 3.5 or something. There are some rows with no ratings which have not been in this (although I think a visualization can be generated discarding the no ratings)
# 
# The data contains an Overall rating, along with ratings for Work Balance, Culture Values, Career Opportunities, Compensation Benefits, and Senior Management; all of which were listed under summaries, pros, and (or) cons for each company

# In[ ]:


# A short function to generate bar graphs for ratings. That being said, this is also the largest code block in this notebook! (Yikes!)
def generate_bargraphs(for_rating):
    temp = pd.concat([
        employee_reviews[(employee_reviews["company"]==t)][for_rating].value_counts(normalize=True).rename(t.title()) for t in employee_reviews.company.unique().tolist()
    ],axis=1,sort=True);
    return [
        go.Bar(
            x=temp.T[t].index.tolist(),
            y=temp.T[t].values.tolist(),
            hovertemplate="%{y:.2%}",
            name=t,
#             orientation="h",
        )
        for t in temp.index];

rating_colorpalette=["#aff895","#89d471","#64b04e","#3f8e2b","#106d00"]

# Generate graphs for each rating
overall_rating_per_company = generate_bargraphs("overall_ratings")
work_balance_rating_per_company = generate_bargraphs('work_balance_stars')
culture_values_rating_per_company = generate_bargraphs('culture_values_stars')
carrer_opportunities_rating_per_company = generate_bargraphs('carrer_opportunities_stars')
comp_benefit_rating_per_company = generate_bargraphs('comp_benefit_stars')
senior_mangemnet_rating_per_company = generate_bargraphs('senior_mangemnet_stars')

# Generate Subplots
figure = tools.make_subplots(rows=2,cols=5,shared_yaxes=True,specs=[[{'colspan':5},{},{},{},{}],[{},{},{},{},{}]],
subplot_titles=("Overall Rating per Company","","","","","Work<br>Balance", "Culture<br>Values","Career<br>Opportunities","Compensation<br>Benefits","Senior<br>Management"))

# Stack the graphs so it forms a 100% stacked bar graph
figure.layout.barmode="stack"

# A decent height for the large plot
figure.layout.height=800

# Hide the y axis since the numbers are visible on hover
figure.layout.yaxis.visible=False
figure.layout.yaxis2.visible=False

# Set the palette for the ratings. 
# Figuring out how to add in the palette for a stacked graph took me around 350 failed tries. 
# Was trying so much with the Bar.marker.colorscales because I was using colorscales to set Pandas plot for the earlier visualizations
# Turns out that it was easier using the figure.layout.colorway and I still figure out how the figure.layout.colorscale fits into all this. In time.
figure.layout.colorway=rating_colorpalette

# Hide the legend because this subplot ends up showing a legend graph for each subplot
figure.layout.showlegend=False

# These subplots can only handle a single graph per subplot area. If the data is an array of plots, then iteration over each element is necessary
for t in overall_rating_per_company:
    figure.add_trace(t,1,1)
for t in work_balance_rating_per_company:
    figure.add_trace(t,2,1)
for t in culture_values_rating_per_company:
    figure.add_trace(t,2,2)
for t in carrer_opportunities_rating_per_company:
    figure.add_trace(t,2,3)
for t in comp_benefit_rating_per_company:
    figure.add_trace(t,2,4)
for t in senior_mangemnet_rating_per_company:
    figure.add_trace(t,2,5)

# Reaching the plot finally
iplot(figure);


# **Overall Rating**: Facebook appears to be the clear winner, followed by Google, Apple, Netflix, Amazon, and Microsoft
# 
# **Work Balance**: Google appears to provide better work balance followed by Facebook, Microsoft, Apple, Netflix, and finally Amazon
# 
# **Culture Values**: Facebook comes out on top again followed by Google, Apple, Netflix, Amazon, Microsoft
# 
# **Career Opportunities**: Facebook appears to provide better career opportunities followed by Google, Amazon, Microsoft, Apple, Netflix
# 
# **Compensation Benefits**: Facebook pay/benefits is valued more over the other companies followed by Google, Netflix, Apple, Microsoft, Amazon
# 
# **Senior Management**: The senior management appears to be better with facebook followed by Google, Apple, Netflix, Amazon, and Microsoft
# 
# **Thoughts**
# 
# Facebook and Google rank among the top consistently
# Amazon is ranked as not so good when it comes to Work Balance, Culture values, Compensation benefits, and Senior management
# Microsoft doesn't get any better when it comes to Culture Values, Career Opportunities, Compensation benefits, and Senior management
# 
# Now these are just peer ratings and each person is entitled to their opinion. Whether or not this holds true is to be experienced by the employee themselves. Will this influence my opinion if I was applying to one of these organizations? No.
# 
# I'd like to think that it comes down to how compatible an employee's ideals/goals/ideas align with the management that they're working for. Or rather how flexible an employee can be with their ideals/goals/ideas when it is not aligned with those of the company.

# **Final Thoughts**
# 
# Plotly seems to be quite interesting when it comes to visualizations, I might just end up using this as much as the pandas plot. There's several aspects to the TextBlob library that I haven't explored yet and might do in a future notebook. I also tried documenting as much of the code as possible, just so that a future more experienced me that forgot the basics, can come back and reminisce or laugh at it.
# 
# Can the existing graphs be made better? Definitely! Take the ratings graph for instance, I tried to make the greens on each of the ratings as consistent as possible with the least amount of code. Turns out it was close enough but not just right. (I'm happy with it anyway)
# 
# Regardless, This was an interesting experience for me. Before this, I was having a "measure twice, cut once" mindset and would get stuck on finding out the best visualization (lost many hours that way). This time, I went head first into the visualization and built from there, and the results were much more better for me.
# 
# Throughout the notebook, I was trying to figure my way around getting the visualizations with unequal data sizes; I still haven't found the most accurate solution to it. Maybe there is a specific term to this, or maybe there isn't a clear cut way to approaching unequal data without cutting it down to size, I don't know.
# 
# **Closing note**
# To you, the reader, I thank you for reaching this far in what I consider to be my first analysis of reviews and ratings. Comments, criticism, or just general advice on the visualizations or the code? I'd love to hear from you.
