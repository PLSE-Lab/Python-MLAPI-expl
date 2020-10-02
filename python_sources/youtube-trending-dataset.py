#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# Youtube videos are not only produced by vbloger or their other name, "Youtubers". Media corporations including Disney, CNN, BBC, and Hulu also offer some of their material via YouTube as part of the YouTube partnership program. 
# 
# If your company, or yourself, a potential million-view youtuber, intend to employ this huge platform to publish your video, it is essential to enhance the content quality, and to increase its visibility. But why Youtube? Because it offers the possibility to monetize your videos, by adding ads during the video progression. With an in-depth analysis of thousands of videos, we could find several keys to increase views, likes, and the most important of all, your incomes.
# 
# The data used in this report can be found at:
# https://www.kaggle.com/datasnaek/youtube-new/ 
# 
# The website says that it was last updated on May, 2019; however the latest publish date in the data in 2018/06/14

# # Description
# 
# The dataset includes data gathered from **40949 videos** on YouTube that are contained within the trending category each day.
# 
# There are two kinds of data files, one includes comments (JSON) and one includes video statistics (CSV). They are linked by the unique video_id field.
# 
# #### The columns in the video file are:
# 
# * title
# * channel_title
# * video_id(Unique id of each video)
# * trending_date 
# * title
# * channel_title
# * category_id (Can be looked up using the included JSON file)
# * publish_time
# * tags (Separated by | character, [none] is displayed if there are no tags)
# * views
# * likes
# * dislikes
# * comment_count
# * thumbnail_link
# * comments_disabled
# * ratings_disabled
# * video_error_or_removed
# * description
# 
# # Data Preparation

# In[ ]:


import pandas as pd
import numpy as np
import json
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import math
from PIL import Image


# In[ ]:



original = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")
new = original.copy()
with open("/kaggle/input/youtube-new/US_category_id.json","r") as category:
    category = json.load(category)
### Extract the category information from the JSON file
vid_cat = []
cat_id = []
for i in category['items']:
    vid_cat.append(i['snippet']['title'])
    cat_id.append(int(i['id']))
    
### Mapping the category_id
new.category_id = original.category_id.map(dict(zip(cat_id,vid_cat)))
new.category_id.isnull().sum() # we have no nan values.

### Prepare date type columns
new['trending_date'] = pd.to_datetime(new['trending_date'], format='%y.%d.%m')
new['publish_time'] = pd.to_datetime(new['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

### Add column for publish time
new['publish_date'] = new['publish_time'].dt.date
new['publish_wd'] = new['publish_time'].dt.weekday
new['publish_hr'] = new['publish_time'].dt.hour
new['publish_time'] = new['publish_time'].dt.time

new.head()
#,'comments_disabled','ratings_disabled'
#For the purpose of this analysis, some columns are irrelevant, we should remove them
new = new.drop(['tags','video_error_or_removed','description'],axis = 1)
# Remove duplicates in the data
new = new.drop_duplicates(keep = 'first')


# In[ ]:


new.info() # We do not have any nan.


# # **Data Exploration**

# ## First, let's look at how many views is associated with each category. 
# This number is important, since it tells us about popularity of a video. How can we utilize this piece of information?
# If you are a Youtuber, it is quite straight forward: the more views, which usually implies the more popular your channel is, the more money you can make from ads. If you are in a marketing team, knowing what type of video people watch the most would help your advertising plan be more effective. Politicians, producers, media companies are other examples who can deploy this information. It is about how to spread out what you want to convey as quick as possible.  

# In[ ]:


df = new[['category_id','views']].groupby('category_id').aggregate(np.sum).reset_index().sort_values(by ='views', ascending = False)
df.views = df.views/10**6
plt.figure(figsize = (20,10))
view_box = sns.barplot(y = 'category_id', x = 'views', data = df, orient = 'h')
plt.title('Barplot of number of views in each category (Unit: milliions)')
plt.ylabel('Category')
plt.xlabel('Views')
#view_box.set_xticklabels(view_box.get_xticklabels(), rotation=45, horizontalalignment='right')


# ### What we discovered: 
# Music, entertainment, film, and comedy are what Americans watch the most, which may not be remarkably surprising, but having a look at real data is apparently much better than a mere guess.

# ## Relationship between number of Likes and Views
# The simplest way is to look at their correlation. View count is very important, how about number of likes? Being able to make someone willing to double-tap on that thumbs-up button may be more crucial than just getting as many views as possible. However, logically, these numbers should vary together. Although, the number of dislikes is in question. Would the dislike count vary together with number of views too? Or if a video is popular, it gets less dislikes?

# In[ ]:


print(new[['views','likes']].corr())
print(new[['views','dislikes']].corr())


# ### What we discovered: 
# 1. The correlation between the view count and like count is 0.85, very high, which confirms our thoughts. If your video can attract a lot of viewers (high view count), it is very likely has a good content (high like count).
# 2. The correlation between the view count and dislike count is 0.47, implying that the dislike count would vary together with the view count, too. In other words, popularity does not equate high content quality/positive viewer reaction. We will look more into this in the following part.

# ## Publish Date
# 
# What day of the week should I publish my video, you might ask? Does it really matter? Isn't it if my video has really good content, the number of views will eventually increase? It is not quite that simple. How important is it to choose a right day, at the right time to post your video? Time is the key point. If your video can get a strong burst of views in the first three days after it comes out, and if its content is excellent, it will get onto the trending list faster. Once your video appears on the trending list, more people will see and click on it, meaning more views, longer time remaining on the trending list and harder for other videos to beat yours.

# In[ ]:


data_bar = new['publish_wd'].map(dict(zip(range(7),
        ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']))).value_counts()
# Use textposition='auto' for direct text
fig = go.Figure(data=[go.Bar(
            x=data_bar.index.values, y=data_bar,
            textposition='auto',
        )])
fig.update_layout(title = "Number of Videos Published in Each Weekday",yaxis=dict(
            title='Videos'))
fig.show()


# ### Comments: 
# It is reasonable to assume that videos posted on the weekend, when people are off from work and can spend some time on Youtube, will get more views within the first 24 hours than those that are posted on a weekday. However, the histogram shows that most Youtubers do not follow this logic. Most of the videos were published on the weekdays, and surprisingly, not too many videos were publish on Saturday and Sunday. We will evaluate this with the following bubble charts. We will look at the videos that have got than one million view within a week (2018-06-07 or after)

# In[ ]:


lastdate = max(new.publish_date)-dt.timedelta(days = 7)
print(lastdate)


# In[ ]:


# Load data, define hover text and bubble size, only look at videos with 10M views or above.
data = new[['title','channel_title','category_id',
            'views','publish_wd','publish_hr','likes','dislikes','publish_date']].loc[new.views > 10**6].reset_index()
data.publish_wd = data.publish_wd.map(dict(zip(range(7),
        ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'])))
def bubble_plot(target, plot_title, target_title, data):
    hover_text = []
    bubble_size = []
    for index, row in data.iterrows():
        hover_text.append(('Title: {title}<br>'+
                      'Channel: {channel_title}<br>'+
                      'Category: {category_id}<br>'+
                      'Views: {views}<br>'+
                      'Likes: {likes} <br>'+
                       'Dislikes: {dislikes}<br>'
                      ).format(title=row['title'],
                                            channel_title=row['channel_title'],
                                            category_id=row['category_id'],
                                            views = row['views'],
                                            likes = row['likes'],
                                            dislikes = row['dislikes']))
        bubble_size.append(row[target]/row['views'])
    data['text'] = hover_text
    data['size'] = bubble_size
    fig = go.Figure()
    # Dictionary with dataframes for each weekday
    weekday = ['Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday']
    wd_data = {wd:data.query("publish_wd == '%s'" %wd)
                              for wd in weekday}
    # Create figure

    for key, values in wd_data.items():
        fig.add_trace(go.Scatter(
            x=values['views'], y=values[target]/values['views'],
            name=key, text=values['text'],
            marker_size=values['size'],
            ))
    # The following formula is recommended by https://plotly.com/python/bubble-charts/
    sizeref = 2.*max(data['size'])/(1000)


    # Tune marker appearance and layout
    fig.update_traces(mode='markers', marker=dict(sizemode='area',
                                              sizeref=sizeref, line_width=2))
    fig.update_layout(
        title=plot_title,
        xaxis=dict(
            title='Number of views in millions',
            gridcolor='white',
            type='log',
            gridwidth=2,
        ),
        yaxis=dict(
            title=target_title,
            gridcolor='white',
            gridwidth=2,
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        legend= {'itemsizing': 'constant'}
    )
    
    fig.show()
pd.options.mode.chained_assignment = None 
bubble_plot('likes', "Like/View Ratio vs. Number of Views", "Like/View Ratio", data.loc[data.publish_date >= lastdate])


# In[ ]:


data.loc[data.publish_date >= lastdate,'publish_wd'].value_counts()


# ### What we discovered:
# 1. ***Despite that we can assume both number of views and likes can tell us about how good a video is, the ratio between them may not.*** 
# We showed earlier the correlation between like count and view count is highly positive, meaning they grow together, but this graph reveals that the view count grows much faster than the like count. The bubbles should stay roughly horizonal if they in fact grow with the same speed. Hence, we should not use this ratio to evaluate the content quality of a video.
# 2. Double-click on each weekday to observe the impact of publish day on the number of views. Most if not all videos that have more than 8 million views were published on Friday or Monday, and those that were published on Tuesday, cound not reach 8 million views in the latest version of this dataset. If we lower our bar to one million views, it appears that most videos were published on weekends. ***Therefore, half of the Youtubers did the right thing to publish their products on the three "hot days", while other half did not have a very great choice. However, this impact is only obvious when the number of views passes 8 million.***

# In[ ]:


bubble_plot('dislikes', "Dislike/View Ratio vs. Number of Views", "Dislike/View Ratio",data)


# ### What we discovered:
# Unlike the Like/View ratio, which decreases as the number of view increases, the Dislike/View remains almost the same regardless of the change in views. If your video does not receive favorable reviews in the first couple of days, it may very likely remain so, even though your views increases eventually as time goes. ***If you work in a marketing team and are choosing a channel to carry out your plan with, closely observing a youtuber's newly published videos reviews after their first week is already enough to make your decision. If you are a youtuber, do not experiment new/unsure content, as "bad" videos will likely just stay "bad": take it down if you receive too many dislikes in the first three days.  ***

# # Predict the number of days to make your video trending
# 
# * ***Idea:***  As explained above, making your videos appear on the trending page is very important. If a video takes too long to become trending, there are some factors we should look into. 
# * ***Target:*** number of days to make a video trending.
# * ***Predictors:*** publish day (weekday), publish hour(0-24), views, likes, dislikes, and comments.
# * ***Notes:*** It should be noticed here that the number of views, likes, dislikes, and comments are not the values collected after a fixed amount of time since publish date (e.g. 3 days, a week,...). Therefore, they do not have the predicting power we wish. However, we would still proceed the predicting task with our dataset as an experiment.
# * ***Model:*** the models we will examine are Random Forest and Xgboost
# 

# In[ ]:


# Create a dataframe for modeling
learn_data = new.loc[(new.comments_disabled) &
                   (~new.ratings_disabled)].copy()
# Create a column for number of days a video takes to get on the trending list
learn_data['day_to_trend'] = abs(np.subtract(learn_data.trending_date.dt.date,learn_data.publish_date,dtype=np.float32).apply(lambda x: x.days))
rel_vars = ['views','likes','dislikes','comment_count','publish_wd', 'publish_hr', 'day_to_trend','title']
learn_data = learn_data[rel_vars]
learn_data.reset_index(inplace=True)
learn_data.head()


# ### Let's first take a look at the distribution of our data

# In[ ]:


from pandas.plotting import scatter_matrix
scatter_matrix(learn_data[['publish_wd', 'publish_hr', 'day_to_trend']])
plt.show()
plt.hist(learn_data['day_to_trend'])
plt.title("Histogram of Original Days to Trend ")
plt.show()

learn_data = learn_data.loc[learn_data.day_to_trend <= 14]
plt.hist(learn_data['day_to_trend'])
plt.title("Histogram of Days to Trend After Removing Values > 7")
plt.show()


# From the histograms of the numerical variables in our dataset, we can see that none of them follow Gaussian distribution. The number of views seem to follow exponential/gamma distribution, and the target variable seems to only cluster at two locations. **This suggest that we may need to discard the few observations**, and let's narrow down to videos that become trending within two weeks.
# 
# Secondly, the scatter plots indicate that **an OLS linear regression will not be a sufficient model**. This is the reason why we should try using more complex learning algorithms like random forest and Xgboost.

# ## Random Forest Algorithm
# 
# ### We will use Random Forest Classifier to predict if the trending day is less than a week

# In[ ]:


from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
learn_data.day_to_trend = learn_data.day_to_trend <=7


# In[ ]:


def rfr_model(X, y, my_param_grid = None ):
# Perform Grid-Search
    if my_param_grid is None:
        #the followings are hyperparameter to optimize: max depth of a tree and number of trees in the forest
        my_param_grid = {
                'max_depth': range(6,10),
                'n_estimators': range(155,170),
                }
    gsc = GridSearchCV(
        estimator=RandomForestClassifier(),    
        param_grid= my_param_grid,
        cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
    
    grid_result = gsc.fit(X, y)

    return grid_result.best_params_,grid_result.best_score_


# In[ ]:


X = learn_data[['views','likes','dislikes','publish_wd', 'publish_hr']]
y = learn_data['day_to_trend']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=4, test_size = .3)


# In[ ]:


print(rfr_model(X_train,y_train)) # ({'max_depth': 9, 'n_estimators': 160}, 0.889456740442656)


# In[ ]:


from sklearn.metrics import classification_report
my_forest = RandomForestClassifier(max_depth = 9,n_estimators = 160,oob_score = True,warm_start = True )
my_forest.fit(X_train,y_train)
print(my_forest.oob_score_)# 0.8696883852691218
print(my_forest.score( X_test, y_test))# 0.9276315789473685
print(my_forest.feature_importances_)
#print(pd.crosstab(pd.Series(y_train, name='Actual'), pd.Series(my_forest.predict(X_train), name='Predicted')))
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(my_forest.predict(X_test), name='Predicted')))
pred = my_forest.predict(X_test)
print(classification_report(y_test, pred))


# In[ ]:


import scikitplot as skplt
from sklearn.metrics import average_precision_score, plot_precision_recall_curve
prob = my_forest.predict_proba(X_test)
myplot = skplt.metrics.plot_roc(y_test, prob)
average_precision = average_precision_score(y_test, prob[:,1]) # prob[:,1] is the estimated probability of positive outcome
disp = plot_precision_recall_curve(my_forest, X_test,y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
score = metrics.f1_score(np.array(y_test),pred)
print('The f1 score for this model is {}'.format(score))


# ### Comments:
# With the Random Forest algorithm, we obtained the parameter estimates that can predict whether or not a video can be on trend within a week with the **90.05% accuracy for the training dataset and 88.5% for the testing dataset**. Since Random Forest algorithm uses a stochastic process to yield a model, what we obtain each time from fitting it to the data will be different. The feature importances indicate that whether or not the comment/rating section is available does not seem to affect the chance of getting on the trend within one week. It also reveals that the most three important factors are number of views, likes, and dislikes. We will try fitting the model again without these two variables.
# 
# Also, since we are interested in both true positive and true negative guesses, and since we have a some imbalance between the two classes (whether a video gets on the trend within one week), we first use ROC curve to check the performance on both of the two classes. The ROC-AUC of both classes is about 91%.
# 
# Let's say, we only want to focus on how good we predict the positive class (or when the positive case is rare in the data). The Precision-Recall Curve should be used instead. The PR-AUC is 96%, meaning that the model seems to predict very well for the positive class. Another way to look at this is using f1-score whose formula is:
# $$f1 = 2\frac{\textrm{Precision}\times \textrm{Recall}}{\textrm{Precision}+ \textrm{Recall}}$$
# 
# This score gives a balance between the precision and recall values. Using this score avoids misleading information from either precision or recall values in certain cases (e.g. data imbalance). Our model f1 is about .94

# ## XGboost Algorithm
# 
# ### We will use XGboost Classifier to predict if the trending day is less than a week

# In[ ]:


from xgboost import XGBClassifier
parameters = [{'n_estimators': range(100,150,1)},
              {'learning_rate': np.arange(0.01,1.0, 0.01)}]
gbm=XGBClassifier(max_features='sqrt', subsample=0.8, random_state=10)
grid_search = GridSearchCV(estimator = gbm, param_grid = parameters, scoring='accuracy', cv = 4, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
#grid_search.cv_results_
#grid_search.best_params_, grid_search.best_score_
grid_search.best_estimator_


# In[ ]:


gbm = XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.24, max_delta_step=0, max_depth=6,
              max_features='sqrt', min_child_weight=1, missing=None,
              monotone_constraints=None, n_estimators=100, n_jobs=0,
              num_parallel_tree=1, objective='binary:logistic', random_state=10,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.8,
              tree_method=None, validate_parameters=False, verbosity=None)
gbm.fit(X_train,y_train)
y_pred = gbm.predict(X_test)
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
print(classification_report(y_test, y_pred))


# In[ ]:


prob = gbm.predict_proba(X_test)
myplot = skplt.metrics.plot_roc(y_test, prob)
average_precision = average_precision_score(y_test, prob[:,1])
disp = plot_precision_recall_curve(gbm, X_test,y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


# ### Comments:
# It seems that **the XGboost using random forest performs very similarly to the random forest model**. However, it took a longer time to train the original random forest model. **Again, our models are not capable of giving legitimate predictions, because we assumed that the number of views, dislikes, and likes are the values collected after a certain timeframe, which is not true.** When the appropriate data become available, we can be more confident about our results. Also, features like thumbnail picture are more likely to be the good ones in predicting days to trend target. However, this would be a big project by itself and needs contributions from more people who have experience with computer vision/image processing.

# In[ ]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
title = learn_data.loc[learn_data.day_to_trend <= 7].title.copy()
title = " ".join(x for x in title)
stopwords = set(STOPWORDS)
stopwords.update(["Official", "Trailer"])

mask = np.array(Image.open("/kaggle/input/logoyou/logo.jpg"))
wordcloud_usa = WordCloud(stopwords=stopwords, background_color="white", mode="RGBA", max_words=500, mask=mask).generate(title)
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[10,14])
plt.imshow(wordcloud_usa.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud = WordCloud(max_words=100, background_color="white", stopwords = stopwords, colormap = 'Reds').generate(title)
plt.figure(figsize=[10,14])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

